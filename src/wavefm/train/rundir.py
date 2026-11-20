"""The files of a run: what every stage writes and what the next stage, the judge and the summary read.

A run is one seed of one pipeline. ``stages: [ssl, ridge, student]`` in its config gives one directory per stage:

    <output>/<group>/<name>/config.json          the expanded config of the run
    <output>/<group>/<name>/<stage>/
        result.json     the keys of RESULT_KEYS; written last, so its presence means the stage is complete
        history.json    [{"step", "train_loss", "val_loss", "lr", "grad_norm", "clip_frac"}, ..]: the full curves
                        (ssl has no validation set: val_loss is absent; ridge has no curve: it writes loo.json)
        weights.pt      state dict handed to the next stage (ssl, finetune); only the encoder is passed on
        verdict.json    written by the judge after reading the curves, see write_verdict
        teacher_labels.npz   (ridge) the teacher's prediction for every dev row, physical units
        student.npz          (student) weights, structure, input transform and output scaling of the student
    <output>/<group>/<name>/heldout.json         written once by scripts/evaluate.py, never by a stage (the
                                                 compressed model's: compress/heldout.json, by compress.py --evaluate)

A stage is ``run(cfg, out, parent) -> dict`` in ``wavefm.train.<stage>``: ``out`` is its directory, ``parent`` the
directory of the stage before it (None for the first). It returns the content of its result.json.

Identity. A stage is the same stage only if three things are the same: the config (config_fp: the keys this stage
depends on, so not the run names, the deadline or the sections of other stages), the data (data_hash of the data
set), and the parent (sha256 of the file it starts from, which carries every change upstream). A finished stage with
another identity is never reused, resumed or overwritten: check_identity says so.

result.json lists the files the stage promises (``artifacts``) with size and sha256. A stage that lost or changed
one is not accepted into a summary (size is checked there; the sha256 is checked by whoever starts from the file).

The two exchange files are plain npz so that the compression environment, which has neither torch nor wavefm, reads
them with numpy alone. Their readers refuse a file whose rows, names, units or shapes are not what the reader has.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from wavefm.utils.config import STAGES
from wavefm.utils.io import save_json, save_npz

RESULT_KEYS = {"stage", "smoke", "config_fp", "data_hash", "parent", "artifacts", "requested", "actual", "hints", "wall_s"}
VERDICTS = ("pass", "overfit", "undertrained", "unclear")
NO_CURVE = ("ridge",)            # closed form: no training curve, so no verdict is asked for
NOT_IDENTITY = {("run", "group"), ("run", "name"), ("run", "deadline")}


class RunError(RuntimeError):
    pass


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def config_fp(cfg: dict, stage: str | None = None) -> str:
    """Fingerprint of the config of one stage: without the run names and deadline, the ``stages`` list and the
    sections of the other stages (what they change reaches this stage through its parent). ``stage=None``: the
    whole run."""
    skip = set() if stage is None else {"stages", *STAGES} - {stage}
    c = {k: ({a: b for a, b in v.items() if (k, a) not in NOT_IDENTITY} if isinstance(v, dict) else v)
         for k, v in cfg.items() if k not in skip}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def parent_record(path) -> dict | None:
    """What a stage records about the file it starts from."""
    return None if path is None else {"file": str(path), "sha256": file_sha256(path)}


# ---- result.json ----------------------------------------------------------------------------------------------------

def write_result(out, stage, cfg, data_hash, parent, *, requested, actual, hints, wall_s, artifacts=(), **more) -> dict:
    """Call it last: its file marks the stage complete. ``requested`` and ``actual``: {"steps", "epochs", "samples"}
    asked for and done (ridge: {"rows"}). ``hints``: numbers that help the judge, never a gate. ``artifacts``: names
    of the files in ``out`` that the stage promises (weights, curves, exchange files); they must exist.
    ``more``: stage-specific values (losses, alphas, sizes)."""
    out = Path(out)
    clash = RESULT_KEYS & more.keys()
    if clash or stage != out.name:
        raise RunError(f"stage values {sorted(clash)} would overwrite reserved keys of result.json, or the stage "
                       f"{stage!r} is not written to a directory of its name ({out})")
    missing = [a for a in artifacts if not (out / a).is_file()]
    if missing:
        raise RunError(f"{out}: the stage promises {missing} but did not write them")
    files = {a: {"sha256": file_sha256(out / a), "bytes": (out / a).stat().st_size} for a in artifacts}
    r = {"stage": stage, "smoke": bool(cfg["run"]["smoke"]), "config_fp": config_fp(cfg, stage), "data_hash": data_hash,
         "parent": parent, "artifacts": files, "requested": requested, "actual": actual, "hints": hints,
         "wall_s": float(wall_s), **more}
    save_json(r, out / "result.json")
    return r


def _snapshot(out):
    """(result, sha256) of result.json from one read: what is judged is exactly what is hashed."""
    p = Path(out) / "result.json"
    if not p.exists():
        raise RunError(f"{out} has no result.json: the stage did not finish")
    raw = p.read_bytes()
    r = json.loads(raw)
    if RESULT_KEYS - r.keys():
        raise RunError(f"{p} lacks {sorted(RESULT_KEYS - r.keys())}: written by another version, run the stage again")
    if r["stage"] != Path(out).name:
        raise RunError(f"{p} is the result of stage {r['stage']!r}, not of {Path(out).name!r}: it was copied or moved")
    return r, hashlib.sha256(raw).hexdigest()


def read_result(out) -> dict:
    return _snapshot(out)[0]


def check_identity(out, cfg, data_hash, parent) -> dict | None:
    """The result of a finished stage if it is this stage, None if the stage has not finished.
    A finished stage of another config, data set or parent raises: use a new run name, nothing is overwritten."""
    if not (Path(out) / "result.json").exists():
        return None
    r = read_result(out)
    for what, got, want in (("config", r["config_fp"], config_fp(cfg, Path(out).name)), ("data set", r["data_hash"], data_hash),
                            ("parent weights", (r["parent"] or {}).get("sha256"), (parent or {}).get("sha256"))):
        if got != want:
            raise RunError(f"{out} finished with another {what} ({got} != {want}): it is not this run. "
                           f"Use a new run.name; a finished stage is never overwritten or reused across identities")
    return r


# ---- verdict.json ---------------------------------------------------------------------------------------------------

def write_verdict(out, verdict: str, reason: str, judge: str) -> dict:
    """The judgement of the curves of one stage (analysis/JUDGING.md). ``judge`` names who looked: the reviewer."""
    if verdict not in VERDICTS or not reason.strip() or not judge.strip():
        raise RunError(f"a verdict is one of {VERDICTS} with a reason and a judge")
    v = {"verdict": verdict, "reason": reason, "judge": judge, "result_sha256": _snapshot(out)[1]}
    save_json(v, Path(out) / "verdict.json")
    return v


def accepted(out) -> bool:
    """True if this stage may enter a summary: finished, not a smoke run, every promised file present with its
    recorded size, and judged "pass" on these very results."""
    out = Path(out)
    if not (out / "result.json").exists():
        return False
    r, sha = _snapshot(out)
    if r["smoke"] or any(not (out / a).is_file() or (out / a).stat().st_size != f["bytes"] for a, f in r["artifacts"].items()):
        return False
    if r["stage"] in NO_CURVE:
        return True
    if not (out / "verdict.json").exists():
        return False
    v = json.loads((out / "verdict.json").read_text())
    return v.get("verdict") == "pass" and v.get("result_sha256") == sha


# ---- teacher labels -------------------------------------------------------------------------------------------------

def write_teacher_labels(path, dev, pred, teacher: dict):
    """The teacher's prediction for every row of ``dev`` (a data.store.Split), in the units of the data set.
    ``teacher``: {"config_fp", "data_hash", "parent"} of the ridge stage (identity of the teacher)."""
    pred = np.asarray(pred, np.float64)
    if pred.shape != dev.labels.shape or not np.isfinite(pred).all():
        raise RunError(f"teacher predictions {pred.shape} must be finite and {dev.labels.shape}: one row per dev row")
    if dev.meta["split"] != "dev" or teacher["data_hash"] != dev.meta["data_hash"]:
        raise RunError(f"teacher labels are written for the dev split the teacher was fitted on, not for {dev.path}")
    meta = {"labels": dev.meta["labels"], "data_hash": dev.meta["data_hash"], "content": dev.content, "split": "dev",
            "teacher": {k: teacher[k] for k in ("config_fp", "parent")}}
    save_npz(path, rows=dev.rows, pred=pred, meta=json.dumps(meta))


def read_teacher_labels(path, dev) -> np.ndarray:
    """(N, k) float64 predictions aligned with the positions of ``dev``; refused unless made for exactly this split."""
    with np.load(path) as z:
        rows, pred, meta = z["rows"], z["pred"], json.loads(str(z["meta"]))
    if (meta["data_hash"], meta["content"], meta["split"], "dev") != (dev.meta["data_hash"], dev.content, "dev", dev.meta["split"]):
        raise RunError(f"{path} was made on another data set or another build of it (data hash "
                       f"{meta['data_hash'][:12]}), not on {dev.path}")
    if meta["labels"] != dev.meta["labels"]:
        raise RunError(f"{path}: label names or units {meta['labels']} differ from the data set's {dev.meta['labels']}")
    if rows.shape != dev.rows.shape or (rows != dev.rows).any() or pred.shape != dev.labels.shape:
        raise RunError(f"{path}: rows or shape do not match {dev.path}: predictions would be given to the wrong events")
    if pred.dtype != np.float64 or not np.isfinite(pred).all():
        raise RunError(f"{path}: predictions must be finite float64")
    return pred


# ---- student weights ------------------------------------------------------------------------------------------------

STUDENT_KEYS = {"sizes", "activation", "input", "output", "labels", "data_hash"}
INPUT_KEYS = {"length", "pad_to", "every", "lo", "hi"}


def _check_student(weights, biases, spec, where):
    """Everything a student file must satisfy, for the writer and the reader alike."""
    if not isinstance(spec, dict) or STUDENT_KEYS - spec.keys() or INPUT_KEYS - set(spec["input"]) or {"scale", "offset"} - set(spec["output"]):
        raise RunError(f"{where}: the student spec needs {sorted(STUDENT_KEYS)}, input {sorted(INPUT_KEYS)}, output scale and offset")
    sizes, i, o = spec["sizes"], spec["input"], spec["output"]
    if spec["activation"] != "relu":
        raise RunError(f"{where}: activation {spec['activation']!r}; only relu is defined")
    if [w.shape for w in weights] != [(b, a) for a, b in zip(sizes, sizes[1:])] or [b.shape for b in biases] != [(n,) for n in sizes[1:]]:
        raise RunError(f"{where}: weights {[w.shape for w in weights]} and biases {[b.shape for b in biases]} do not "
                       f"match sizes {sizes} (weights are (out, in), the PyTorch layout)")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in [*weights, *biases]):
        raise RunError(f"{where}: weights and biases must be finite float32")
    ints = all(isinstance(i[k], int) and i[k] > 0 for k in ("length", "pad_to", "every"))
    if not ints or i["pad_to"] < i["length"] or -(-i["pad_to"] // i["every"]) > sizes[0] or not (np.isfinite([i["lo"], i["hi"]]).all() and i["hi"] > i["lo"]):
        raise RunError(f"{where}: input needs positive integers length <= pad_to, every; at most sizes[0] kept samples; finite lo < hi")
    if not (len(o["scale"]) == len(o["offset"]) == len(spec["labels"]) == sizes[-1]) or not np.isfinite([*o["scale"], *o["offset"]]).all():
        raise RunError(f"{where}: output scale, offset and labels need one finite entry per output")


def write_student(path, weights, biases, spec: dict):
    """A small MLP as numpy arrays. Layer i is ``y = x @ weights[i].T + biases[i]`` (weights[i] is (out, in), the
    PyTorch layout; Keras wants the transpose), ReLU between layers, none after the last.

    spec["sizes"]       [n_in, h1, .., n_out]
    spec["activation"]  "relu"
    spec["input"]       {"length": L of the waveform, "pad_to": L padded by repeating the last sample,
                         "every": keep sample 0, every, 2 * every, ..; then pad the kept samples to n_in by repeating
                         the last one; "lo", "hi": x -> (x - lo) / (hi - lo)}
    spec["output"]      {"scale": [..], "offset": [..]}: physical = network output * scale + offset
    spec["labels"]      [{"name", "unit"}, ..] of the outputs, as in the data set
    spec["data_hash"]   the data set the student was made for: whoever feeds it waveforms compares it with the
                        split's (same hash: same length, sampling, units and labels)
    """
    weights, biases = [np.asarray(w, np.float32) for w in weights], [np.asarray(b, np.float32) for b in biases]
    _check_student(weights, biases, spec, path)
    arrays = {f"w{i}": w for i, w in enumerate(weights)}
    arrays.update({f"b{i}": b for i, b in enumerate(biases)})
    save_npz(path, spec=json.dumps(spec), **arrays)


def read_student(path, data_hash: str):
    """(weights, biases, spec) of a student made for the data set ``data_hash``, checked as in write_student."""
    with np.load(path) as z:
        spec = json.loads(str(z["spec"]))
        n = len(spec.get("sizes", [])) - 1
        if any(f"{k}{i}" not in z for i in range(n) for k in "wb"):
            raise RunError(f"{path}: weights or biases of some layer are missing")
        weights, biases = [z[f"w{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)]
    _check_student(weights, biases, spec, path)
    if spec["data_hash"] != data_hash:
        raise RunError(f"{path} is a student for another data set (data hash {spec['data_hash'][:12]}, not {data_hash[:12]})")
    return weights, biases, spec


def student_input(x, spec: dict) -> np.ndarray:
    """The input transform of the student on raw waveforms (N, L): pad, decimate, pad, scale. float32."""
    i = spec["input"]
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != i["length"]:
        raise RunError(f"student made for waveforms of {i['length']} samples, got {x.shape}")
    x = np.pad(x, ((0, 0), (0, i["pad_to"] - i["length"])), mode="edge")[:, ::i["every"]]
    x = np.pad(x, ((0, 0), (0, spec["sizes"][0] - x.shape[1])), mode="edge")
    return ((x - np.float32(i["lo"])) / np.float32(i["hi"] - i["lo"])).astype(np.float32)


def student_predict(weights, biases, spec, x) -> np.ndarray:
    """Reference forward pass in numpy, raw waveforms in, physical units out (float64)."""
    h = student_input(x, spec)
    for k, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w.T + b
        if k < len(weights) - 1:
            h = np.maximum(h, 0)
    return h.astype(np.float64) * np.asarray(spec["output"]["scale"]) + np.asarray(spec["output"]["offset"])
