"""The student stage: the small MLP (models/student.py) trained on teacher labels or on true labels, and its export.
``run(cfg, out, parent)``, ``predict(out, split)``.

Two modes, ``student.targets``:
  teacher  distillation (the paper's): the targets are the teacher's predictions for dev rows, read from
           ``<parent>/teacher_labels.npz`` of the ridge stage (rundir.read_teacher_labels; its sha256 is this stage's
           parent identity and must equal the one the ridge stage recorded). No true label is a training target.
  truth    from scratch: the targets are the true labels of the label budget's training rows. There is no parent
           (``stages: [student]``); a parent is refused, because nothing of it would be used.

Rows (positions in dev; budget = data.splits.labelled(n, label_frac, val_frac, seed, min_labelled, min_val)):
  truth    train = the budget's train rows, val = the budget's validation rows. No other label is read.
  teacher  perm = numpy.random.default_rng(seed).permutation(n), the permutation of splits.labelled;
           val = perm[:round(val_frac * n)] (the teacher-label curve: val_frac of ALL dev rows),
           val_truth = the budget's validation rows (the true-label curve), train = every other dev row.
           Because the budget's validation rows are perm[:max(round(val_frac * n_labelled), min_val)], they are a
           prefix of val whenever that count is at most round(val_frac * n) (always unless min_val dominates); either
           way they are never training rows. True labels are read for val_truth only, all inside the budget; the
           teacher-label rows ``val`` are dev rows that may lie outside it (only their teacher labels are read).
Validation, at every evaluation of the shared loop: ``val_loss`` = the training loss on ``val`` (teacher targets in
teacher mode, true targets in truth mode) is the primary value (keep: best_val, schedule: plateau); in teacher mode
``val_truth_loss`` = the same loss on the true labels of ``val_truth`` is recorded only.

Input (model level): models.student.kept (pad to ``pad_to``, every ``every``-th sample from 0, pad to ``n_in``), then
(x - lo) / (hi - lo) with ``input_scaling``:
  dev_minmax   lo, hi = minimum and maximum of all dev waveforms (dev meta.json; no fitting, no labels)
  kept_minmax  lo, hi = minimum and maximum of the kept samples of the training rows (the DCH paper's rule)
Targets: physical = output * scale + offset, with ``target_scaling``:
  standard     offset, scale = mean and spread (float64, ddof 0) of the training targets (teacher labels of the
               training rows in teacher mode)
  divisor      offset 0, scale = ``divisors`` (one per output; the DCH paper: 53)
Loss: mean over rows of sum_k w_k (output_k - target_k)^2 in the scaled space, w = ``loss_weights`` (null: all 1).
Initialisation: runtime.seed_all(cfg["seed"]) and then PyTorch's default nn.Linear initialisation of every layer;
the LAST bias is then set to 0 (``output_bias: zero``, the DRO paper) or to the mean scaled training target
(``mean``, the DCH paper). Nothing else is changed.
Training: one AdamW group (``lr``, ``weight_decay``) through train.trainer with the dynamics keys of ``student``;
the data live on the device as kept, scaled samples (float32), targets likewise.

Files in ``out``: weights.pt and history.json (trainer), rows.npz (``train``, ``val``; teacher mode also
``val_truth``: int64 dev positions; the compression stage trains and validates on ``train`` / ``val``),
student.npz (rundir.write_student: weights, sizes, input transform, output scaling, labels, data hash),
result.json. ``predict`` uses student.npz alone through rundir.read_student / student_predict (numpy), so the
exported file is exactly what is evaluated.
"""
from pathlib import Path

import numpy as np
import torch

import wavefm.data as wdata
from wavefm.data import splits
from wavefm.models.student import kept, linears, mlp, n_kept, scale
from wavefm.train import guards, pipeline, rundir, trainer
from wavefm.utils import runtime
from wavefm.utils.io import save_npz

TARGETS = ("teacher", "truth")
INPUT_SCALING = ("dev_minmax", "kept_minmax")
TARGET_SCALING = ("standard", "divisor")
OUTPUT_BIAS = ("zero", "mean")
CHUNK = 8192            # waveforms read and transformed at a time


class StudentError(ValueError):
    pass


def check_config(s: dict, length: int, n_out: int):
    """Refuse a student section that cannot be what it says, before any work."""
    for key, known in (("targets", TARGETS), ("input_scaling", INPUT_SCALING), ("target_scaling", TARGET_SCALING),
                       ("output_bias", OUTPUT_BIAS)):
        if s[key] not in known:
            raise StudentError(f"student.{key} {s[key]!r}: known {known}")
    if not s["hidden"] or any(not isinstance(h, int) or h < 1 for h in s["hidden"]):
        raise StudentError(f"student.hidden {s['hidden']}: a list of positive layer widths")
    if s["pad_to"] < length or s["every"] < 1 or n_kept(s["pad_to"], s["every"]) > s["n_in"]:
        raise StudentError(f"student.pad_to {s['pad_to']}, every {s['every']}, n_in {s['n_in']} for waveforms of "
                           f"{length} samples: need pad_to >= {length}, every >= 1 and ceil(pad_to / every) <= n_in")
    if s["target_scaling"] == "divisor":
        d = s["divisors"]
        if not isinstance(d, list) or len(d) != n_out or not all(np.isfinite(v) and v > 0 for v in d):
            raise StudentError(f"student.divisors {d}: one positive number per output ({n_out}) with target_scaling "
                               f"divisor")
    elif s["divisors"] is not None:
        raise StudentError("student.divisors is set but target_scaling is standard: set it to null, or use divisor")
    w = s["loss_weights"]
    if w is not None and (not isinstance(w, list) or len(w) != n_out or not all(np.isfinite(v) and v > 0 for v in w)):
        raise StudentError(f"student.loss_weights {w}: null or one positive weight per output ({n_out})")


def rows(cfg: dict, n: int) -> dict:
    """Positions in dev (sorted int64): {"train", "val"} and, in teacher mode, "val_truth". See the module docstring."""
    d = cfg["data"]
    b_train, b_val = splits.labelled(n, d["label_frac"], d["val_frac"], cfg["seed"], d["min_labelled"], d["min_val"])
    if cfg["student"]["targets"] == "truth":
        return {"train": b_train, "val": b_val}
    n_val = int(round(d["val_frac"] * n))
    val = np.sort(np.random.default_rng(cfg["seed"]).permutation(n)[:n_val]).astype(np.int64)
    train = np.setdiff1d(np.arange(n, dtype=np.int64), np.union1d(val, b_val))
    if n_val < 1 or not len(train):
        raise StudentError(f"data.val_frac {d['val_frac']} of {n} dev rows leaves no validation or no training rows "
                           f"for the student")
    return {"train": train, "val": val, "val_truth": b_val}


def kept_samples(split, idx, s: dict, device) -> torch.Tensor:
    """(len(idx), n_in) float32 kept samples (not yet scaled) of the rows ``idx`` of ``split``, on ``device``."""
    parts = [kept(torch.from_numpy(split.take(idx[a:a + CHUNK])), split.length, s["pad_to"], s["every"], s["n_in"])
             for a in range(0, len(idx), CHUNK)]
    return torch.cat(parts).to(device)


def input_range(s: dict, dev, k_train: torch.Tensor) -> tuple:
    if s["input_scaling"] == "dev_minmax":
        st = dev.meta["waveform"]["stats"]
        lo, hi = float(st["min"]), float(st["max"])
    else:
        lo, hi = float(k_train.min()), float(k_train.max())
    if not (np.isfinite([lo, hi]).all() and hi > lo):
        raise StudentError(f"input range {lo} .. {hi} ({s['input_scaling']}): the waveforms are constant or not finite")
    return lo, hi


def target_scaling(s: dict, y_train: np.ndarray) -> tuple:
    """(offset, scale), float64 arrays (k,): physical = output * scale + offset."""
    if s["target_scaling"] == "divisor":
        return np.zeros(y_train.shape[1]), np.asarray(s["divisors"], np.float64)
    offset, spread = y_train.mean(0), y_train.std(0)
    if not (spread > 0).all():
        raise StudentError(f"a training target is constant (spread {spread}): standard scaling is undefined")
    return offset, spread


def run(cfg: dict, out, parent) -> dict:
    runtime.setup_numerics()                       # strict fp32 before anything else runs
    out, s = Path(out), cfg["student"]
    dev = wdata.dev(cfg)
    data_hash, n_out = dev.meta["data_hash"], len(dev.names)
    check_config(s, dev.length, n_out)
    teacher = s["targets"] == "teacher"
    if teacher and parent is None:
        raise StudentError("student.targets teacher needs the ridge stage before it (stages: [.., ridge, student]); "
                           "for a student from scratch set student.targets truth")
    if not teacher and parent is not None:
        raise StudentError(f"student.targets truth trains from scratch and uses nothing of the stage before "
                           f"({parent}): run it as stages: [student]")
    labels_file = pipeline.handover(parent) if teacher else None
    prec = rundir.parent_record(labels_file)
    if teacher and rundir.read_result(parent)["artifacts"].get("teacher_labels.npz", {}).get("sha256") != prec["sha256"]:
        raise StudentError(f"{labels_file} is not the file the ridge stage {parent} wrote (sha256 differs from its "
                           f"result.json): run the ridge stage again")
    done = rundir.check_identity(out, cfg, data_hash, prec)
    if done is not None:
        return done
    out.mkdir(parents=True, exist_ok=True)
    with guards.RunLock(out) as lock:
        device = runtime.device()
        r = rows(cfg, len(dev))
        save_npz(out / "rows.npz", **r)
        if teacher:
            t = rundir.read_teacher_labels(labels_file, dev)
            y = {"train": t[r["train"]], "val": t[r["val"]],
                 "val_truth": np.asarray(dev.labels[r["val_truth"]], np.float64)}   # the only true labels read
        else:
            y = {k: np.asarray(dev.labels[r[k]], np.float64) for k in ("train", "val")}
        offset, tscale = target_scaling(s, y["train"])
        k = {name: kept_samples(dev, idx, s, device) for name, idx in r.items()}
        lo, hi = input_range(s, dev, k["train"])
        x = {name: scale(v, lo, hi) for name, v in k.items()}
        z = {name: torch.from_numpy(((v - offset) / tscale).astype(np.float32)).to(device) for name, v in y.items()}
        del k

        runtime.seed_all(cfg["seed"])
        sizes = [s["n_in"], *s["hidden"], n_out]
        model = mlp(sizes).to(device)
        with torch.no_grad():
            last = linears(model)[-1].bias
            last.copy_(z["train"].double().mean(0).float() if s["output_bias"] == "mean" else torch.zeros_like(last))
        w = torch.tensor(s["loss_weights"] or [1.0] * n_out, dtype=torch.float32, device=device)
        fwd = torch.compile(model) if s["compile"] and device.type == "cuda" else model

        def sq(o, t):
            return ((o - t) ** 2 * w).sum(1).mean()

        def loss(b, gen):
            return sq(fwd(x["train"][b]), z["train"][b])

        def mean_sq(name):       # sq over the rows ``name``, student.eval_batch rows per forward pass
            e, n = s["eval_batch"], len(x[name])
            return sum(sq(model(x[name][a:a + e]), z[name][a:a + e]).item() * len(x[name][a:a + e])
                       for a in range(0, n, e)) / n

        def validate():
            v = {"val_loss": mean_sq("val")}
            if teacher:
                v["val_truth_loss"] = mean_sq("val_truth")
            return v

        groups = [{"params": list(model.parameters()), "lr": s["lr"], "weight_decay": s["weight_decay"],
                   "name": "student"}]
        summary = trainer.train(out, cfg, "student", model, groups, loss, len(r["train"]), data_hash=data_hash,
                                parent=prec, validate=validate, lock=lock)
        if summary.get("preempted"):
            return summary
        spec = {"sizes": sizes, "activation": "relu",
                "input": {"length": dev.length, "pad_to": s["pad_to"], "every": s["every"], "lo": lo, "hi": hi},
                "output": {"scale": tscale.tolist(), "offset": offset.tolist()},
                "labels": dev.meta["labels"], "data_hash": data_hash}
        layers = linears(model)
        rundir.write_student(out / "student.npz", [m.weight.detach().cpu().numpy() for m in layers],
                             [m.bias.detach().cpu().numpy() for m in layers], spec)
        return trainer.finish(out, cfg, "student", summary, artifacts=("rows.npz", "student.npz"), mode=s["targets"],
                              n_params=sum(p.numel() for p in model.parameters()), n_val=len(r["val"]),
                              n_val_truth=len(r.get("val_truth", ())),
                              input_lo_hi=[lo, hi], target_offset=offset.tolist(), target_scale=tscale.tolist())


def predict(out, split) -> np.ndarray:
    """(N, k) float64 predictions of the finished student ``out`` for every row of ``split``, physical units, from
    student.npz alone (numpy; the file the compression stage reads)."""
    out = Path(out)
    res, f = rundir.read_result(out), out / "student.npz"
    guards.check_artifact(out, res, "student.npz")
    weights, biases, spec = rundir.read_student(f, split.meta["data_hash"])
    idx = np.arange(len(split))
    return np.concatenate([rundir.student_predict(weights, biases, spec, split.take(idx[a:a + CHUNK]))
                           for a in range(0, len(idx), CHUNK)])
