"""The compress stage: the distilled student in QKeras, pruned and trained with quantisation, then scored once.

    python compression/compress.py <run>              writes <run>/compress/ (model.h5, history.json, result.json)
    python compression/compress.py --evaluate <run>   scores model.h5 on heldout once: <run>/compress/heldout.json

This environment has TensorFlow / QKeras and neither torch nor wavefm. It reads plain files: the run's config.json
(keys ``compress``, ``data``, ``seed``, ``run.smoke``, ``stages``), the student (<run>/student/student.npz and the
train / validation positions <run>/student/rows.npz), the teacher labels (<run>/ridge/teacher_labels.npz) and the dev
split through the repo's ``data`` package (numpy + yaml only). The readers below re-implement, in numpy, the few lines
of wavefm.train.rundir they need (config_fp, read_result, read_student, student_input, the teacher-label checks):
that file is the specification, and the tests pin the shared values.

Upstream files are authenticated before anything is reused or trained (``upstream``): the ridge and student stages
must be finished (result.json with the reserved keys, naming its directory); student.npz and rows.npz must have the
size and sha256 the student recorded, teacher_labels.npz the sha256 the ridge stage recorded and the student's parent
records. A student trained on true labels (``student.targets: truth``) is refused: compression distils on teacher
labels. One process at a time works on the stage (``<run>/compress/.claim``, created exclusively, removed on exit).
There is no resume: an interrupted stage starts again from the beginning; history.json is rewritten at every
evaluation, so the partial curve of an interrupted run stays readable.

``--evaluate`` follows scripts/evaluate.py: a non-smoke run is scored only if every stage up to compress is judged
"pass" on its present result (ridge excepted); ``<run>/compress/.heldout.claim`` is created exclusively before heldout
is opened and removed when the evaluation ends, also when it fails; heldout.json is published with an exclusive link,
never overwritten, in the record of scripts/evaluate.py: {metrics, stage, data_hash, n, result: {config_fp, parent,
smoke, sha256 of compress/result.json}}.

Conventions
  * Layer i of the student is ``x @ w_i.T + b_i`` with w_i of shape (out, in) (PyTorch); the Keras kernel is w_i.T.
    The input transform (pad, decimate, pad, min-max) is done in numpy outside the model, exactly as student_input;
    the model maps scaled inputs to the network's output space, physical = output * scale + offset.
  * ``compress.kernel`` (one quantizer per layer), ``bias`` and ``activation`` (one for all, hidden layers only) are
    QKeras quantizer strings; all three null is the float model: Dense + ReLU. The output layer is linear.
  * Phase A, pruning: ``prune_low_magnitude`` with ConstantSparsity(s_i, begin_step 0, frequency 1) on every layer with
    ``sparsity[i] > 0``, ``prune_steps`` steps on the float weights. The magnitude mask is taken at the first step;
    pruned weights get no gradient, so the mask of the first step is the mask of the run (no regrowth, and a
    validation pass cannot change the trajectory). ``prune_steps: 0`` still prunes: the mask, nothing trained.
  * Phase B, quantisation-aware training: ``steps`` steps, pruned weights get no gradient and stay exactly 0.
    ``steps: 0`` (with ``prune_steps: 0``) is the baseline "quantise the given weights" and may be the best.
  * Both phases: Adam at the constant rate ``lr``, ``batch_size`` rows per step, epochs are permutations of the
    student's train rows joined end to end (numpy generator of ``seed``); loss = mean squared error to the teacher's
    predictions scaled into the network's output space; validation loss (the same, on the student's val rows) at step
    0, every max(steps // 30, 1) steps and at the last step. The last weights are kept. history.json holds both
    phases: {"phase", "step", "train_loss", "val_loss", "lr"}; ``step`` counts both phases (QAT continues the count
    of pruning), train_loss is the mean batch loss since the last evaluation, null at a phase's first row. A
    non-finite training or validation loss, or non-finite final weights, stop the stage before anything is published.
  * result.json: the reserved keys of rundir.RESULT_KEYS; ``requested`` / ``actual`` count the two phases together,
    ``phases`` gives each one. Stage values: per-layer parameters, pruned fraction, sparsity (exactly-zero weights of
    the kernel the model computes with, i.e. after its quantizer), non-zero parameters, quantizers, validation losses.
  * model.h5 holds the LATENT float weights (QKeras quantises them at every call); what the model computes with is
    quantizer(latent), and load_qmodel restores the same function.
  * Strict fp32: ``setup_numerics`` (TF32 off, deterministic ops) runs before any model is built or loaded, in
    training, in --evaluate and in every reload (``load_model``). Every file write is atomic.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root: its `data` package (numpy + yaml only)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QActivation, QDense
from qkeras.utils import load_qmodel

from data import store

DATA_ROOT = store.ROOT     # where data/<set>/configs live; tests point it at a temporary data root
NOT_IDENTITY = {("run", "group"), ("run", "name"), ("run", "deadline")}
STAGES = ("finetune", "ssl", "ridge", "student", "compress")     # wavefm.utils.config.STAGES
RESULT_KEYS = {"stage", "smoke", "config_fp", "data_hash", "parent", "artifacts", "requested", "actual", "hints",
               "wall_s"}                                               # rundir.RESULT_KEYS
NO_CURVE = ("ridge",)                                                  # rundir.NO_CURVE: no verdict asked for
STUDENT_KEYS = {"sizes", "activation", "input", "output", "labels", "data_hash"}
INPUT_KEYS = {"length", "pad_to", "every", "lo", "hi"}
CHUNK = 4096               # waveforms per input transform / prediction batch
EVAL_POINTS = 30           # the curve has 30 validation points per phase (+ its first row); not a config key


class CompressError(RuntimeError):
    pass


# ---- files of the run (rundir.py is the specification) --------------------------------------------------------------

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def config_fp(cfg: dict, stage: str) -> str:
    """rundir.config_fp of a stage: without the run names and deadline, the ``stages`` list and the other stages."""
    skip = {"stages", *STAGES} - {stage}
    c = {k: ({a: b for a, b in v.items() if (k, a) not in NOT_IDENTITY} if isinstance(v, dict) else v)
         for k, v in cfg.items() if k not in skip}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def read_result(out) -> dict:
    """rundir.read_result: the result.json of a finished stage, with the reserved keys, naming its own directory."""
    p = Path(out) / "result.json"
    if not p.exists():
        raise CompressError(f"{out} has no result.json: the stage did not finish, run it first")
    r = json.loads(p.read_text())
    if RESULT_KEYS - r.keys() or r["stage"] != Path(out).name:
        raise CompressError(f"{p} lacks {sorted(RESULT_KEYS - r.keys())} or is the result of stage {r.get('stage')!r}, "
                            f"not of {Path(out).name!r}: written by another version, copied or moved; run it again")
    return r


def check_artifact(out, r, name) -> str:
    """The sha256 of a file the stage promised, refused unless its size and sha256 are those result.json records."""
    f, want = Path(out) / name, r["artifacts"].get(name)
    if want is None or not f.is_file() or f.stat().st_size != want["bytes"] or file_sha256(f) != want["sha256"]:
        raise CompressError(f"{f} is not the file the stage {out} wrote (missing, or its size or sha256 differs from "
                            f"its result.json): run that stage again")
    return want["sha256"]


def upstream(run_dir) -> dict:
    """Authenticate the finished ridge and student stages and their files (module docstring); returns the parent
    record {"file", "sha256"} of student.npz."""
    s_dir, r_dir = run_dir / "student", run_dir / "ridge"
    s = read_result(s_dir)
    if s.get("mode") != "teacher" or not s["parent"]:
        raise CompressError(f"{s_dir} was trained on true labels (student.targets {s.get('mode')!r}): compression "
                            f"distils on teacher labels, so it needs stages [.., ridge, student (targets teacher)]")
    sha = check_artifact(s_dir, s, "student.npz")
    check_artifact(s_dir, s, "rows.npz")
    if check_artifact(r_dir, read_result(r_dir), "teacher_labels.npz") != s["parent"]["sha256"]:
        raise CompressError(f"{r_dir}/teacher_labels.npz is not the file the student {s_dir} was distilled from: "
                            f"run the student again after the ridge stage")
    return {"file": str(s_dir / "student.npz"), "sha256": sha}


def check_accepted(run_dir, cfg):
    """rundir.accepted for --evaluate: a non-smoke run is scored only when every stage up to compress has the verdict
    "pass" on its present result.json (ridge has no curve). Smoke runs never enter a summary: exempt."""
    if cfg["run"]["smoke"]:
        return
    for s in cfg["stages"][:cfg["stages"].index("compress") + 1]:
        v = run_dir / s / "verdict.json"
        v = json.loads(v.read_text()) if v.exists() else {}
        if s not in NO_CURVE and (v.get("verdict") != "pass"
                                  or v.get("result_sha256") != file_sha256(run_dir / s / "result.json")):
            raise CompressError(f"{run_dir / s} is not judged 'pass' on its present result: heldout is opened only "
                                f"after the curves are judged (scripts/judge.py)")


@contextlib.contextmanager
def claim(path, what):
    """``path`` created exclusively for the time of the block, removed at its end, also on an error."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664))
    except FileExistsError:
        raise CompressError(f"{what} is being done by another process ({path} exists); if none is running, delete "
                            f"{path}") from None
    try:
        yield
    finally:
        os.unlink(path)


def read_student(path, data_hash: str):
    """rundir.read_student: (weights (out, in), biases, spec) of a student made for the data set ``data_hash``."""
    with np.load(path) as z:
        spec = json.loads(str(z["spec"]))
        n = len(spec.get("sizes", [])) - 1
        if any(f"{k}{i}" not in z for i in range(n) for k in "wb"):
            raise CompressError(f"{path}: weights or biases of some layer are missing")
        weights, biases = [z[f"w{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)]
    if STUDENT_KEYS - spec.keys() or INPUT_KEYS - set(spec["input"]) or {"scale", "offset"} - set(spec["output"]):
        raise CompressError(f"{path}: the student spec needs {sorted(STUDENT_KEYS)}, input {sorted(INPUT_KEYS)}, "
                            f"output scale and offset")
    sizes, i, o = spec["sizes"], spec["input"], spec["output"]
    if spec["activation"] != "relu":
        raise CompressError(f"{path}: activation {spec['activation']!r}; only relu is defined")
    if [w.shape for w in weights] != [(b, a) for a, b in zip(sizes, sizes[1:])] or [b.shape for b in biases] != [(m,) for m in sizes[1:]]:
        raise CompressError(f"{path}: weights {[w.shape for w in weights]} and biases {[b.shape for b in biases]} do "
                            f"not match sizes {sizes} (weights are (out, in), the PyTorch layout)")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in [*weights, *biases]):
        raise CompressError(f"{path}: weights and biases must be finite float32")
    ints = all(isinstance(i[k], int) and i[k] > 0 for k in ("length", "pad_to", "every"))
    if (not ints or i["pad_to"] < i["length"] or -(-i["pad_to"] // i["every"]) > sizes[0]
            or not (np.isfinite([i["lo"], i["hi"]]).all() and i["hi"] > i["lo"])):
        raise CompressError(f"{path}: input needs positive integers length <= pad_to, every; at most sizes[0] kept "
                            f"samples; finite lo < hi")
    if not (len(o["scale"]) == len(o["offset"]) == len(spec["labels"]) == sizes[-1]) or not np.isfinite([*o["scale"], *o["offset"]]).all():
        raise CompressError(f"{path}: output scale, offset and labels need one finite entry per output")
    if spec["data_hash"] != data_hash:
        raise CompressError(f"{path} is a student for another data set (data hash {spec['data_hash'][:12]}, "
                            f"not {data_hash[:12]})")
    return weights, biases, spec


def student_input(x, spec: dict) -> np.ndarray:
    """rundir.student_input: raw waveforms (N, L) -> pad, decimate, pad, scale. float32."""
    i = spec["input"]
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != i["length"]:
        raise CompressError(f"student made for waveforms of {i['length']} samples, got {x.shape}")
    x = np.pad(x, ((0, 0), (0, i["pad_to"] - i["length"])), mode="edge")[:, ::i["every"]]
    x = np.pad(x, ((0, 0), (0, spec["sizes"][0] - x.shape[1])), mode="edge")
    return ((x - np.float32(i["lo"])) / np.float32(i["hi"] - i["lo"])).astype(np.float32)


def read_teacher_labels(path, dev) -> np.ndarray:
    """rundir.read_teacher_labels: (N, k) float64 aligned with the positions of ``dev``, or refused."""
    with np.load(path) as z:
        rows, pred, meta = z["rows"], z["pred"], json.loads(str(z["meta"]))
    if (meta["data_hash"], meta["content"], meta["split"], "dev") != (dev.meta["data_hash"], dev.content, "dev", dev.meta["split"]):
        raise CompressError(f"{path} was made on another data set or another build of it (data hash "
                            f"{meta['data_hash'][:12]}), not on {dev.path}")
    if meta["labels"] != dev.meta["labels"]:
        raise CompressError(f"{path}: label names or units {meta['labels']} differ from the data set's {dev.meta['labels']}")
    if rows.shape != dev.rows.shape or (rows != dev.rows).any() or pred.shape != dev.labels.shape:
        raise CompressError(f"{path}: rows or shape do not match {dev.path}: predictions would be given to the wrong events")
    if pred.dtype != np.float64 or not np.isfinite(pred).all():
        raise CompressError(f"{path}: predictions must be finite float64")
    return pred


def read_rows(path, n):
    """The student's train and validation positions in dev: disjoint, non-empty, inside [0, n)."""
    with np.load(path) as z:
        train, val = np.asarray(z["train"]), np.asarray(z["val"])
    for name, a in (("train", train), ("val", val)):
        if a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= n or len(np.unique(a)) != len(a):
            raise CompressError(f"{path}: {name} must be unique integer positions inside [0, {n}), non-empty")
    if np.intersect1d(train, val).size:
        raise CompressError(f"{path}: train and val share {np.intersect1d(train, val).size} rows: the student stage is broken")
    return train.astype(np.int64), val.astype(np.int64)


def split(cfg: dict, name: str) -> store.Split:
    """A split of the run's data set, refused if it was built from another config or other data-set code."""
    dcfg = store.load_config(cfg["data"]["set"], cfg["data"]["config"], DATA_ROOT)
    s = store.Split(Path(dcfg["out"]) / name)
    if s.meta["data_hash"] != store.data_hash(dcfg, DATA_ROOT):
        raise store.StaleDataError(f"{s.path} was built from another config or data-set code: rebuild it or restore them")
    return s


def inputs(sp: store.Split, positions, spec) -> np.ndarray:
    return np.concatenate([student_input(sp.take(positions[i:i + CHUNK]), spec) for i in range(0, len(positions), CHUNK)])


# ---- the model ------------------------------------------------------------------------------------------------------

def check_numbers(c: dict):
    """Before any work: integer step counts (a fraction or NaN would never end the batch loop), a finite rate."""
    ints = all(isinstance(c[k], int) and not isinstance(c[k], bool) and c[k] >= 0
               for k in ("steps", "prune_steps", "batch_size"))
    lr_ok = isinstance(c["lr"], (int, float)) and not isinstance(c["lr"], bool) and math.isfinite(c["lr"]) and c["lr"] > 0
    if not ints or c["batch_size"] < 1 or not lr_ok:
        raise CompressError(f"compress: steps and prune_steps integers >= 0, batch_size an integer >= 1, lr > 0 and "
                            f"finite; got steps {c['steps']}, prune_steps {c['prune_steps']}, batch_size "
                            f"{c['batch_size']}, lr {c['lr']}")


def check_settings(c: dict, n_layers: int):
    for key in ("kernel", "sparsity"):
        if c[key] is not None and (not isinstance(c[key], list) or len(c[key]) != n_layers):
            raise CompressError(f"compress.{key}: one entry per layer ({n_layers}) or null, got {c[key]}")
    if c["sparsity"] is not None and not all(0 <= s < 1 for s in c["sparsity"]):
        raise CompressError(f"compress.sparsity {c['sparsity']}: each fraction in [0, 1)")
    if c["prune_steps"] > 0 and not any(c["sparsity"] or []):
        raise CompressError("compress.prune_steps > 0 with nothing to prune: set compress.sparsity or prune_steps: 0")


def setup_numerics():
    """Strict fp32, as in training: TF32 off (TensorFlow's default is on), deterministic ops. Call it before any
    model is built or loaded."""
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()


def load_model(path):
    """model.h5 reloaded under strict fp32: the entry of --evaluate and of every reload check."""
    setup_numerics()
    return load_qmodel(str(path), compile=False)


def check_finite(model, val: dict):
    """Validation losses, latent weights and the weights the model computes with must be finite before publishing."""
    ws = [a for d in dense_layers(model) for a in (d.kernel.numpy(), d.bias.numpy(), computed(d), computed(d, "bias"))]
    if not np.isfinite(list(val.values())).all() or not all(np.isfinite(a).all() for a in ws):
        raise CompressError(f"non-finite validation losses {val} or weights: nothing is published; lower compress.lr")


def build(sizes, kernel=None, bias=None, activation=None, sparsity=None):
    """Dense (or QDense with quantizers) layers ``dense_i``, ReLU (or QActivation) ``relu_i`` between them; layers with
    sparsity > 0 wrapped by prune_low_magnitude."""
    layers, n = [tf.keras.Input((sizes[0],))], len(sizes) - 1
    for i, width in enumerate(sizes[1:]):
        if kernel or bias:
            d = QDense(width, kernel_quantizer=kernel[i] if kernel else None, bias_quantizer=bias, name=f"dense_{i}")
        else:
            d = tf.keras.layers.Dense(width, name=f"dense_{i}")
        if sparsity and sparsity[i] > 0:
            d = tfmot.sparsity.keras.prune_low_magnitude(
                d, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(sparsity[i], begin_step=0, frequency=1))
        layers.append(d)
        if i < n - 1:
            layers.append(QActivation(activation, name=f"relu_{i}") if activation else tf.keras.layers.ReLU(name=f"relu_{i}"))
    return tf.keras.Sequential(layers)


def dense_layers(model):
    """The Dense / QDense layers in order, unwrapped from their pruning wrappers."""
    out = [getattr(l, "layer", l) for l in model.layers]
    return [l for l in out if isinstance(l, tf.keras.layers.Dense)]


def set_weights(model, kernels, biases):
    for d, k, b in zip(dense_layers(model), kernels, biases):
        d.kernel.assign(k); d.bias.assign(b)


def get_weights(model):
    return [d.kernel.numpy() for d in dense_layers(model)], [d.bias.numpy() for d in dense_layers(model)]


def computed(d, which="kernel"):
    """The weights a layer computes with: its quantizer applied to the latent weights (none: the latent ones)."""
    q, w = getattr(d, f"{which}_quantizer_internal", None), getattr(d, which)
    return (q(w) if q else w).numpy()


def check_pruned(model, masks):
    """Pruned weights are exactly 0 in the latent kernel and in the kernel the model computes with."""
    for d, m in zip(dense_layers(model), masks):
        if m is not None and ((d.kernel.numpy()[m == 0] != 0).any() or (computed(d)[m == 0] != 0).any()):
            raise CompressError(f"{d.name}: pruned weights are not 0 in the model's kernel. A kernel quantizer that "
                                f"maps 0 to a non-zero value (e.g. quantized_po2 with few bits) undoes the pruning: "
                                f"use more bits or another quantizer")


def check_sparsity(kernel, target, name):
    """tfmot keeps round(n * (1 - s)) weights: the pruned fraction is s up to that rounding."""
    got = float(np.mean(kernel == 0))
    if abs(got - target) > 0.5 / kernel.size + 1e-12:
        raise CompressError(f"{name}: {got:.4f} of the kernel pruned, asked {target}: the pruning did not take effect")
    return got


def save_model(model, path):
    """model.h5 (latent weights, quantizers in the config), written under a temporary name and renamed."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    model.save(tmp, save_format="h5")
    os.replace(tmp, path)


def predict(model, x):
    return np.concatenate([model(x[i:i + CHUNK], training=False).numpy() for i in range(0, len(x), CHUNK)])


def mse(model, x, y) -> float:
    return float(np.mean((predict(model, x).astype(np.float64) - y) ** 2))


# ---- training -------------------------------------------------------------------------------------------------------

def batches(n, size, rng):
    """Positions of successive batches: permutations of range(n) joined end to end, every batch of ``size`` rows."""
    buf = np.empty(0, np.int64)
    while True:
        while len(buf) < size:
            buf = np.concatenate([buf, rng.permutation(n)])
        yield buf[:size]
        buf = buf[size:]


def train(model, steps, grad_masks, data, c, rng, history, phase, out, prune_cb=None):
    """``steps`` Adam steps at the constant rate; ``grad_masks``: per trainable variable, a mask (variable or array)
    that multiplies its gradient, or None. Appends the curve to ``history``, its steps continuing the last row's, and
    writes it to ``out``/history.json at every evaluation."""
    xt, yt, xv, yv = data
    opt = tf.keras.optimizers.Adam(c["lr"])
    variables = model.trainable_variables

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True) - y))
        grads = [g if m is None else g * tf.cast(m, g.dtype) for g, m in zip(tape.gradient(loss, variables), grad_masks)]
        opt.apply_gradients(zip(grads, variables))
        return loss

    every, done, losses, start = max(steps // EVAL_POINTS, 1), 0, [], history[-1]["step"] if history else 0

    def evaluate(train_loss):
        v = mse(model, xv, yv)
        if not np.isfinite(v):
            raise CompressError(f"{phase}: validation loss {v} at step {done}: lower compress.lr")
        history.append({"phase": phase, "step": start + done, "train_loss": train_loss, "val_loss": v, "lr": c["lr"]})
        save_json(history, out / "history.json")

    evaluate(None)
    for idx in batches(len(xt), c["batch_size"], rng):
        if done == steps:
            break
        if prune_cb:
            prune_cb.on_train_batch_begin(batch=-1)
        losses.append(float(step(xt[idx], yt[idx])))
        done += 1
        if not np.isfinite(losses[-1]):
            raise CompressError(f"{phase}: loss {losses[-1]} at step {done}: lower compress.lr")
        if done % every == 0 or done == steps:
            evaluate(float(np.mean(losses)))
            losses = []
    return {"steps": steps, "samples": steps * c["batch_size"], "epochs": steps * c["batch_size"] / len(xt)}


def prune(kernels, biases, sizes, sparsity, data, c, rng, history, out):
    """Phase A on the float model. Returns the pruned kernels, biases, masks (None for unpruned layers), counts."""
    model = build(sizes, sparsity=sparsity)
    set_weights(model, kernels, biases)
    wrappers = [l for l in model.layers if hasattr(l, "pruning_vars")]    # the PruneLowMagnitude wrappers
    mask_of = {w.ref(): m for l in wrappers for w, m, _ in l.pruning_vars}
    cb = tfmot.sparsity.keras.UpdatePruningStep()
    cb.set_model(model)
    cb.on_train_begin()
    if c["prune_steps"]:
        grad_masks = [mask_of.get(v.ref()) for v in model.trainable_variables]
        done = train(model, c["prune_steps"], grad_masks, data, c, rng, history, "prune", out, cb)
    else:                      # the mask of the first step without the step: one forward pass in training mode
        cb.on_train_batch_begin(batch=-1)
        model(data[0][:1], training=True)
        done = {"steps": 0, "samples": 0, "epochs": 0.0}
    cb.on_epoch_end(batch=-1)  # weights = weights * mask after the last update
    kernels, biases = get_weights(tfmot.sparsity.keras.strip_pruning(model))
    masks = [(k != 0).astype(np.float32) if s > 0 else None for k, s in zip(kernels, sparsity)]
    for i, (k, s) in enumerate(zip(kernels, sparsity)):
        check_sparsity(k, s, f"dense_{i}")
    return kernels, biases, masks, done


def summary(model, masks, c) -> dict:
    layers = []
    for d, m in zip(dense_layers(model), masks):
        k, b = computed(d), computed(d, "bias")
        layers.append({"name": d.name, "shape": list(k.shape), "params": int(k.size + b.size),
                       "pruned": 0.0 if m is None else float(np.mean(m == 0)), "sparsity": float(np.mean(k == 0)),
                       "nonzero": int(np.count_nonzero(k) + np.count_nonzero(b))})
    n_kernel = sum(np.prod(l["shape"]) for l in layers)
    return {"layers": layers, "params": sum(l["params"] for l in layers),
            "nonzero": sum(l["nonzero"] for l in layers),
            "sparsity": float(sum(l["sparsity"] * np.prod(l["shape"]) for l in layers) / n_kernel),
            "quantizers": {k: c[k] for k in ("kernel", "bias", "activation")}}


# ---- the stage ------------------------------------------------------------------------------------------------------

def run(run_dir) -> dict:
    t0, run_dir = time.time(), Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    out, c = run_dir / "compress", cfg["compress"]
    check_numbers(c)
    out.mkdir(parents=True, exist_ok=True)
    with claim(out / ".claim", f"the compress stage of {run_dir}"):
        parent = upstream(run_dir)
        dev = split(cfg, "dev")
        identity = {"config_fp": config_fp(cfg, "compress"), "data_hash": dev.meta["data_hash"],
                    "parent": parent["sha256"]}
        if (out / "result.json").exists():
            r = read_result(out)
            have = {"config_fp": r["config_fp"], "data_hash": r["data_hash"],
                    "parent": (r["parent"] or {}).get("sha256")}
            if have != identity:
                raise CompressError(f"{out} finished with another config, data set or student ({have} != {identity}): "
                                    f"use a new run.name; a finished stage is never overwritten")
            print(f"[compress] {out} already finished", flush=True)
            return r
        return _compress(run_dir, cfg, dev, parent, identity, t0)


def _compress(run_dir, cfg, dev, parent, identity, t0) -> dict:
    out, c = run_dir / "compress", cfg["compress"]
    kernels, biases, spec = read_student(parent["file"], dev.meta["data_hash"])
    kernels = [w.T.copy() for w in kernels]                   # (out, in) -> Keras (in, out)
    sizes, n_layers = spec["sizes"], len(spec["sizes"]) - 1
    check_settings(c, n_layers)
    if spec["labels"] != dev.meta["labels"]:
        raise CompressError(f"student outputs {spec['labels']} are not the labels of the data set {dev.meta['labels']}")
    teacher = read_teacher_labels(run_dir / "ridge" / "teacher_labels.npz", dev)
    train_rows, val_rows = read_rows(run_dir / "student" / "rows.npz", dev.n)
    scale, offset = np.asarray(spec["output"]["scale"], np.float64), np.asarray(spec["output"]["offset"], np.float64)
    target = ((teacher - offset) / scale).astype(np.float32)
    data = (inputs(dev, train_rows, spec), target[train_rows], inputs(dev, val_rows, spec), target[val_rows])

    tf.keras.utils.set_random_seed(cfg["seed"])
    setup_numerics()
    rng, history = np.random.default_rng(cfg["seed"]), []
    sparsity = c["sparsity"] or [0.0] * n_layers
    float_model = build(sizes)
    set_weights(float_model, kernels, biases)
    val = {"float": mse(float_model, data[2], data[3])}
    phases = {"prune": {"steps": 0, "samples": 0, "epochs": 0.0}}
    masks = [None] * n_layers
    if any(sparsity):
        kernels, biases, masks, phases["prune"] = prune(kernels, biases, sizes, sparsity, data, c, rng, history, out)
        float_model = build(sizes)
        set_weights(float_model, kernels, biases)
        val["pruned"] = mse(float_model, data[2], data[3])

    model = build(sizes, c["kernel"], c["bias"], c["activation"])
    set_weights(model, kernels, biases)
    check_pruned(model, masks)
    grad_masks = [m for mask in masks for m in (mask, None)]   # kernel, bias of each layer; biases are never pruned
    phases["qat"] = train(model, c["steps"], grad_masks, data, c, rng, history, "qat", out)
    check_pruned(model, masks)
    val["quantised"] = next(h["val_loss"] for h in history if h["phase"] == "qat")
    val["final"] = history[-1]["val_loss"]
    check_finite(model, val)

    save_model(model, out / "model.h5")
    artifacts = {a: {"sha256": file_sha256(out / a), "bytes": (out / a).stat().st_size} for a in ("model.h5", "history.json")}
    total = {k: sum(p[k] for p in phases.values()) for k in ("steps", "samples", "epochs")}
    n_req = c["prune_steps"] + c["steps"]
    r = {"stage": "compress", "smoke": bool(cfg["run"]["smoke"]), "config_fp": identity["config_fp"],
         "data_hash": identity["data_hash"], "parent": parent, "artifacts": artifacts,
         "requested": {"steps": n_req, "epochs": 0, "samples": n_req * c["batch_size"]}, "actual": total,
         "hints": {"val_loss_first": history[0]["val_loss"], "val_loss_last": val["final"]},
         "wall_s": time.time() - t0, "phases": phases, "val_loss": val, "rows": {"train": len(train_rows), "val": len(val_rows)},
         **summary(model, masks, c)}
    save_json(r, out / "result.json")
    print(f"[compress] {out}: val loss {val}, sparsity {r['sparsity']:.3f}, {r['nonzero']} of {r['params']} non-zero",
          flush=True)
    return r


def _refuse_if_done(path):
    if path.exists():
        raise CompressError(f"{path} exists: heldout is scored once, never again.\n{path.read_text()}")


def evaluate_heldout(run_dir) -> dict:
    """The final score of the compressed model: every heldout row predicted once, physical units, data set's evaluate."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    out = run_dir / "compress"
    done = out / "heldout.json"
    _refuse_if_done(done)
    r, parent = read_result(out), upstream(run_dir)
    if r["config_fp"] != config_fp(cfg, "compress") or r["parent"]["sha256"] != parent["sha256"]:
        raise CompressError(f"{out}: config.json or the student changed since the stage ran: it is not this model")
    check_artifact(out, r, "model.h5")
    check_accepted(run_dir, cfg)
    with claim(out / ".heldout.claim", f"the evaluation of {out}"):
        _refuse_if_done(done)                # finished by another evaluator just before the claim
        held = split(cfg, "heldout")
        if held.meta["data_hash"] != r["data_hash"]:
            raise CompressError(f"{held.path} is another data set than the one the model was compressed on")
        _, _, spec = read_student(parent["file"], r["data_hash"])
        model = load_model(out / "model.h5")
        x = inputs(held, np.arange(held.n), spec)
        pred = (predict(model, x).astype(np.float64) * np.asarray(spec["output"]["scale"])
                + np.asarray(spec["output"]["offset"]))
        h = {"metrics": store.evaluate(cfg["data"]["set"])(pred, held.labels), "stage": "compress",
             "data_hash": r["data_hash"], "n": held.n,
             "result": {"config_fp": r["config_fp"], "parent": r["parent"], "smoke": r["smoke"],
                        "sha256": file_sha256(out / "result.json")}}
        tmp = done.with_name(f".{done.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(h, indent=1, default=float))
        try:
            os.link(tmp, done)               # atomic, and fails if another call wrote it meanwhile
        finally:
            tmp.unlink()
    print(f"[compress] heldout {h['metrics']}", flush=True)
    return h


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("run", type=Path, help="run directory <output>/<group>/<name>")
    a.add_argument("--evaluate", action="store_true", help="score the compressed model on heldout (once)")
    args = a.parse_args()
    evaluate_heldout(args.run) if args.evaluate else run(args.run)
