"""The compress stage on a tiny fake run (student 8-8-4-3, 200 dev and 100 heldout waveforms of 30 samples).

Acceptance of package W5: (1) the float conversion equals the numpy reference; (2) per-layer pruned fractions equal
the config and pruned weights stay exactly 0 through QAT; (3) one QAT step changes the unpruned weights and every
weight the saved model computes with lies on its quantizer's grid; (4) zero steps give a complete result; (5) save ->
reload gives identical outputs; (6) heldout is scored by the data set's evaluate. Then one test per guard. The fake
run carries genuine result.json manifests of ridge and student (``sign``); a test that edits an upstream file on
purpose to reach a reader's check signs it again.

Run on a GPU node from the repo root: `uv run --frozen --project compression python -m pytest -q compression/tests`.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from qkeras.quantizers import quantized_bits, quantized_po2
from qkeras.utils import load_qmodel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # compression/: the stage module
import compress  # noqa: E402
from data import store  # noqa: E402

SIZES, LENGTH, N_DEV, N_HELD = [8, 8, 4, 3], 30, 200, 100
LABELS = [{"name": "C", "unit": "photons"}, {"name": "S", "unit": "photons"}, {"name": "t0", "unit": "ns"}]
RESULT_KEYS = {"stage", "smoke", "config_fp", "data_hash", "parent", "requested", "actual", "hints", "wall_s",
               "artifacts"}                                          # wavefm.train.rundir.RESULT_KEYS
DEFAULTS = {"kernel": None, "bias": None, "activation": None, "sparsity": None, "prune_steps": 0, "steps": 0,
            "batch_size": 32, "lr": 1.0e-3}
QUANT = {"kernel": ["quantized_po2(10)", "quantized_bits(10,2,alpha=1)", "quantized_po2(10)"],
         "bias": "quantized_bits(10,2,alpha=1)", "activation": "quantized_relu(10,2)"}
SPARSITY = [0.5, 0.25, 0.0]       # 32 of 64, 8 of 32, 0 of 12 kernel weights


@pytest.fixture(autouse=True)
def gpu():
    assert tf.config.list_physical_devices("GPU"), "TensorFlow sees no GPU: run on a GPU node"


def make_run(tmp_path, monkeypatch, set_name="w5fake", name="run", **compress_cfg):
    """A data root with set ``set_name`` (dev + heldout), and a run directory holding config.json, the student, its
    rows and the teacher labels. Returns the run directory."""
    root = tmp_path / "data_root"
    monkeypatch.setattr(compress, "DATA_ROOT", root)
    if not (root / set_name).exists():
        (root / set_name / "configs").mkdir(parents=True)
        (root / set_name / "build.py").write_text("# fake\n")
        (root / set_name / "configs" / "tiny.yaml").write_text(f"out: {tmp_path / 'data_out'}\nseed: 1\n")
        dcfg = store.load_config(set_name, "tiny", root)
        rng = np.random.default_rng(1)
        for split, n, rows in (("dev", N_DEV, np.arange(N_DEV) * 2), ("heldout", N_HELD, np.arange(N_HELD) * 2 + 1)):
            W = rng.normal(0, 1, (n, LENGTH)).astype(np.float32)
            Y = np.stack([rng.uniform(100, 1500, n), rng.uniform(100, 900, n), rng.uniform(-2, 2, n)], 1)
            store.write_split(Path(dcfg["out"]) / split, dcfg, split, Y.astype(np.float32), rows, LABELS, dt_ns=0.32,
                              t_first_ns=0.0, unit="mV", waveform=W, root=root)
    dev = compress.split({"data": {"set": set_name, "config": "tiny"}}, "dev")

    run = tmp_path / name
    (run / "student").mkdir(parents=True)
    (run / "ridge").mkdir()
    cfg = {"run": {"group": "g", "name": name, "smoke": True, "deadline": None}, "seed": 0,
           "stages": ["ssl", "ridge", "student", "compress"], "data": {"set": set_name, "config": "tiny"},
           "compress": {**DEFAULTS, **compress_cfg}}
    (run / "config.json").write_text(json.dumps(cfg))
    rng = np.random.default_rng(2)
    weights = [rng.normal(0, 0.5, (b, a)).astype(np.float32) for a, b in zip(SIZES, SIZES[1:])]
    biases = [rng.normal(0, 0.1, b).astype(np.float32) for b in SIZES[1:]]
    wst = dev.meta["waveform"]["stats"]
    spec = {"sizes": SIZES, "activation": "relu", "labels": LABELS, "data_hash": dev.meta["data_hash"],
            "input": {"length": LENGTH, "pad_to": 32, "every": 4, "lo": wst["min"], "hi": wst["max"]},
            "output": {"scale": [400.0, 250.0, 1.4], "offset": [750.0, 450.0, 0.0]}}
    np.savez(run / "student" / "student.npz", spec=json.dumps(spec), **{f"w{i}": w for i, w in enumerate(weights)},
             **{f"b{i}": b for i, b in enumerate(biases)})
    perm = rng.permutation(N_DEV)
    np.savez(run / "student" / "rows.npz", train=np.sort(perm[40:]), val=np.sort(perm[:40]))
    meta = {"labels": LABELS, "data_hash": dev.meta["data_hash"], "content": dev.content, "split": "dev", "teacher": {}}
    teacher = dev.labels.astype(np.float64) + rng.normal(0, 1, dev.labels.shape)
    np.savez(run / "ridge" / "teacher_labels.npz", rows=dev.rows, pred=teacher, meta=json.dumps(meta))
    sign(run)
    return run


def manifest(d, files, **more):
    """result.json of the stage directory ``d`` as rundir.write_result writes it, for the present ``files``."""
    art = {f: {"sha256": compress.file_sha256(d / f), "bytes": (d / f).stat().st_size} for f in files}
    r = {"stage": d.name, "smoke": True, "config_fp": "f", "data_hash": "h", "parent": None, "artifacts": art,
         "requested": {}, "actual": {}, "hints": {}, "wall_s": 0.0, **more}
    (d / "result.json").write_text(json.dumps(r))


def sign(run, mode="teacher", ridge=True):
    """Finished ridge and student stages for the present files: the student distilled from these teacher labels."""
    labels = run / "ridge" / "teacher_labels.npz"
    if ridge:
        manifest(run / "ridge", ["teacher_labels.npz"])
    parent = {"file": str(labels), "sha256": compress.file_sha256(labels)} if mode == "teacher" else None
    manifest(run / "student", ["rows.npz", "student.npz"], parent=parent, mode=mode)


def reference(run, waveforms):
    """The student in numpy, network output space (before the output scaling), written out from rundir.py."""
    with np.load(run / "student" / "student.npz") as z:
        spec = json.loads(str(z["spec"]))
        ws, bs = [z[f"w{i}"] for i in range(3)], [z[f"b{i}"] for i in range(3)]
    x = np.pad(waveforms, ((0, 0), (0, 2)), mode="edge")[:, ::4]                    # 32 samples -> 8 kept
    h = (x - np.float32(spec["input"]["lo"])) / np.float32(spec["input"]["hi"] - spec["input"]["lo"])
    for k, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w.T + b
        h = np.maximum(h, 0) if k < 2 else h
    return h


def dev_waveforms(run):
    cfg = json.loads((run / "config.json").read_text())
    dev = compress.split(cfg, "dev")
    return dev.take(np.arange(dev.n))


def on_grid(q, v) -> bool:
    """Every value of ``v`` is representable by the quantizer: po2 sign x 2^k with k in its exponent range, fixed
    point (alpha=1) an integer multiple of 2^(integer - bits + keep_negative) inside the signed range."""
    v = np.asarray(v, np.float64)
    if isinstance(q, quantized_po2):
        e = np.log2(np.abs(v[v != 0]))
        return bool(np.all(e == np.round(e)) and (e.size == 0 or (e.min() >= q._min_exp and e.max() <= q._max_exp)))
    assert isinstance(q, quantized_bits) and q.alpha == 1
    m = 2.0 ** (q.bits - q.keep_negative)
    k = v * m / 2.0 ** q.integer
    return bool(np.all(k == np.round(k)) and k.min() >= q.keep_negative * (-m + q.symmetric) and k.max() <= m - 1)


# ---- acceptance -----------------------------------------------------------------------------------------------------

def test_float_conversion_equals_numpy_reference(tmp_path, monkeypatch):
    """(1) all quantizers null, no steps: the saved Keras model is the student (|diff| < 1e-5, output space)."""
    run = make_run(tmp_path, monkeypatch)
    r = compress.run(run)
    model = load_qmodel(str(run / "compress" / "model.h5"), compile=False)
    assert [type(l).__name__ for l in model.layers] == ["Dense", "ReLU", "Dense", "ReLU", "Dense"]
    W = dev_waveforms(run)
    with np.load(run / "student" / "student.npz") as z:
        spec = json.loads(str(z["spec"]))
    keras_out = compress.predict(model, compress.student_input(W, spec))
    diff = np.abs(keras_out - reference(run, W)).max()
    print("float conversion max |diff|", diff)
    assert diff < 1e-5
    assert r["val_loss"]["float"] == r["val_loss"]["final"] and r["sparsity"] == 0.0


def test_pruning_masks_survive_qat_and_weights_on_grid(tmp_path, monkeypatch):
    """(2) per-layer pruned fraction = config, pruned weights exactly 0 after QAT; (3) one QAT step moves the
    unpruned weights (all but those of units the random student never activates), and the weights the saved model
    computes with lie on the quantizer grids."""
    pruned_only = make_run(tmp_path, monkeypatch, "w5fake", "pruned", sparsity=SPARSITY, prune_steps=5, **QUANT)
    one_step = make_run(tmp_path, monkeypatch, "w5fake", "qat", sparsity=SPARSITY, prune_steps=5, steps=1, **QUANT)
    ra, rb = compress.run(pruned_only), compress.run(one_step)
    a = load_qmodel(str(pruned_only / "compress" / "model.h5"), compile=False)
    b = load_qmodel(str(one_step / "compress" / "model.h5"), compile=False)
    da, db = compress.dense_layers(a), compress.dense_layers(b)
    assert [l["pruned"] for l in rb["layers"]] == SPARSITY
    for la, lb, s, layer in zip(da, db, SPARSITY, rb["layers"]):
        ka, kb = la.kernel.numpy(), lb.kernel.numpy()           # latent weights: the same run up to the QAT step
        assert np.mean(ka == 0) == s and np.array_equal(ka == 0, kb == 0), "pruned weights moved during QAT"
        assert (compress.computed(lb)[kb == 0] == 0).all(), "pruned weights are not 0 in what the model computes"
        moved = np.abs(kb - ka)[(ka != 0) & (kb != ka)]   # Adam's first step: |g| / (|g| + eps) x lr, 0 for dead units
        assert moved.size > 0.5 * np.count_nonzero(ka) and moved.max() <= 1e-3 * 1.0001, "QAT step did not train"
        assert on_grid(lb.kernel_quantizer_internal, compress.computed(lb)), f"{lb.name} kernel off its grid"
        assert on_grid(lb.bias_quantizer_internal, compress.computed(lb, "bias")), f"{lb.name} bias off its grid"
        assert layer["sparsity"] >= s
    assert rb["actual"]["steps"] == 6 and rb["phases"]["prune"]["steps"] == 5 and rb["phases"]["qat"]["steps"] == 1
    assert [h["phase"] for h in json.loads((one_step / "compress" / "history.json").read_text())] == ["prune"] * 6 + ["qat"] * 2


def test_zero_step_baseline_gives_complete_result(tmp_path, monkeypatch):
    """(4) prune_steps 0 and steps 0: the given weights pruned by magnitude and quantised, nothing trained."""
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, **QUANT)
    r = compress.run(run)
    assert RESULT_KEYS <= r.keys() and r["stage"] == "compress"
    assert r["actual"] == {"steps": 0, "samples": 0, "epochs": 0.0} and r["requested"]["steps"] == 0
    assert set(r["artifacts"]) == {"model.h5", "history.json"}
    assert [l["pruned"] for l in r["layers"]] == SPARSITY and r["params"] == 8 * 8 + 8 + 8 * 4 + 4 + 4 * 3 + 3
    assert r["nonzero"] <= r["params"] - 32 - 8 and np.isfinite(r["val_loss"]["final"])
    hist = json.loads((run / "compress" / "history.json").read_text())
    assert hist == [{"phase": "qat", "step": 0, "train_loss": None, "val_loss": r["val_loss"]["final"], "lr": 1e-3}]
    assert r == json.loads((run / "compress" / "result.json").read_text())


def test_save_reload_identical_outputs(tmp_path, monkeypatch):
    """(5) a quantised, pruned, trained model and its reloaded copy compute the same outputs, bit for bit."""
    model = compress.build(SIZES, sparsity=None, **QUANT)
    rng = np.random.default_rng(0)
    compress.set_weights(model, [rng.normal(0, .5, (a, b)).astype(np.float32) for a, b in zip(SIZES, SIZES[1:])],
                         [rng.normal(0, .1, b).astype(np.float32) for b in SIZES[1:]])
    compress.save_model(model, tmp_path / "m.h5")
    loaded = load_qmodel(str(tmp_path / "m.h5"), compile=False)
    x = rng.uniform(0, 1, (500, 8)).astype(np.float32)
    assert np.array_equal(compress.predict(model, x), compress.predict(loaded, x))
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, prune_steps=3, steps=3, **QUANT)  # and the stage's file
    r = compress.run(run)
    cfg = json.loads((run / "config.json").read_text())
    dev = compress.split(cfg, "dev")
    spec = compress.read_student(run / "student" / "student.npz", dev.meta["data_hash"])[2]
    with np.load(run / "student" / "rows.npz") as z:
        val = z["val"]
    target = (compress.read_teacher_labels(run / "ridge" / "teacher_labels.npz", dev)[val] - spec["output"]["offset"]) \
        / spec["output"]["scale"]
    loaded = load_qmodel(str(run / "compress" / "model.h5"), compile=False)
    assert compress.mse(loaded, compress.inputs(dev, val, spec), target.astype(np.float32)) == pytest.approx(
        r["val_loss"]["final"], rel=1e-6)


def test_heldout_scored_by_the_data_set(tmp_path, monkeypatch):
    """(6) every heldout row predicted once in physical units and scored by data.<set>.evaluate; never twice."""
    seen = {}

    def fake_evaluate(pred, truth):
        seen["pred"], seen["truth"] = pred, truth
        return {"mae": float(np.abs(pred - truth).mean())}

    monkeypatch.setitem(sys.modules, "data.w5fake.evaluate", types.SimpleNamespace(evaluate=fake_evaluate))
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, steps=2, **QUANT)
    compress.run(run)
    h = compress.evaluate_heldout(run)
    cfg = json.loads((run / "config.json").read_text())
    held = compress.split(cfg, "heldout")
    spec = compress.read_student(run / "student" / "student.npz", held.meta["data_hash"])[2]
    model = load_qmodel(str(run / "compress" / "model.h5"), compile=False)
    want = compress.predict(model, compress.student_input(held.take(np.arange(N_HELD)), spec)).astype(np.float64) \
        * spec["output"]["scale"] + spec["output"]["offset"]
    assert seen["pred"].dtype == np.float64 and np.array_equal(seen["pred"], want)
    assert np.array_equal(seen["truth"], held.labels) and h["n"] == N_HELD
    assert h["metrics"] == {"mae": float(np.abs(want - held.labels).mean())} and h["stage"] == "compress"
    assert h["result"]["sha256"] == compress.file_sha256(run / "compress" / "result.json")   # evaluate.py's record
    assert json.loads((run / "compress" / "heldout.json").read_text()) == h
    with pytest.raises(compress.CompressError, match="scored once"):
        compress.evaluate_heldout(run)


@pytest.mark.skipif(not (store.ROOT / "dro_gen" / "evaluate.py").exists(), reason="data/dro_gen/evaluate.py absent")
def test_heldout_scored_by_the_real_dro_evaluate(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, set_name="dro_gen", **QUANT)
    compress.run(run)
    h = compress.evaluate_heldout(run)
    print("dro_gen evaluate on the fake run:", h["metrics"])
    assert set(h["metrics"]) == {"err68_ratio_pct", "err68_t0_ns", "n", "n_invalid"} and h["metrics"]["n"] == N_HELD


# ---- guards ---------------------------------------------------------------------------------------------------------

def test_config_fp_matches_the_main_environment():
    cfg = {"run": {"group": "g", "name": "n", "smoke": True}, "seed": 0, "compress": {"steps": 0}}
    print("config_fp of the reference config:", compress.config_fp(cfg, "compress"))
    assert compress.config_fp(cfg, "compress") == REFERENCE_FP
    other = {**cfg, "run": {"group": "x", "name": "y", "smoke": True, "deadline": 5}, "stages": ["student", "compress"],
             "student": {"lr": 1.0}, "ridge": {}, "ssl": {}, "finetune": {}}
    assert compress.config_fp(other, "compress") == REFERENCE_FP
    assert compress.config_fp({**cfg, "seed": 1}, "compress") != REFERENCE_FP


def test_identity_reuse_and_refusal(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, **QUANT)
    r = compress.run(run)
    assert compress.run(run) == r                                               # same identity: returned, not redone
    cfg = json.loads((run / "config.json").read_text())
    cfg["compress"]["lr"] = 1e-2
    (run / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(compress.CompressError, match="another config"):
        compress.run(run)


def test_teacher_labels_of_other_rows_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    p = run / "ridge" / "teacher_labels.npz"
    with np.load(p) as z:
        d = dict(z)
    np.savez(p, **{**d, "rows": d["rows"] + 1})
    sign(run)
    with pytest.raises(compress.CompressError, match="wrong events"):
        compress.run(run)


def test_overlapping_rows_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    np.savez(run / "student" / "rows.npz", train=np.arange(0, 150), val=np.arange(140, 180))
    sign(run)
    with pytest.raises(compress.CompressError, match="share"):
        compress.run(run)


def test_stale_data_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    (tmp_path / "data_root" / "w5fake" / "build.py").write_text("# fake, changed\n")
    with pytest.raises(store.StaleDataError):
        compress.run(run)


def test_student_of_another_data_set_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    p = run / "student" / "student.npz"
    with np.load(p) as z:
        d = dict(z)
    spec = json.loads(str(d["spec"]))
    np.savez(p, **{**d, "spec": json.dumps({**spec, "data_hash": "0" * 64})})
    sign(run)
    with pytest.raises(compress.CompressError, match="another data set"):
        compress.run(run)
    np.savez(p, **{**d, "spec": json.dumps({**spec, "labels": LABELS[::-1]})})
    sign(run)
    with pytest.raises(compress.CompressError, match="labels"):
        compress.run(run)


@pytest.mark.parametrize("bad, match", [({"kernel": ["quantized_po2(10)"]}, "one entry per layer"),
                                        ({"sparsity": [0.5, 1.0, 0.0]}, r"in \[0, 1\)"),
                                        ({"prune_steps": 3}, "nothing to prune"),
                                        ({"lr": 0.0}, "lr > 0")])
def test_bad_settings_refused(tmp_path, monkeypatch, bad, match):
    run = make_run(tmp_path, monkeypatch, **bad)
    with pytest.raises(compress.CompressError, match=match):
        compress.run(run)


def test_quantizer_that_undoes_pruning_refused(tmp_path, monkeypatch):
    """quantized_po2(5) maps 0 to 2^-8: the pruned weights would be non-zero in what the model computes."""
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, kernel=["quantized_po2(5)"] * 3)
    with pytest.raises(compress.CompressError, match="undoes the pruning"):
        compress.run(run)


def test_missed_sparsity_refused():
    k = np.ones((8, 8), np.float32)
    k[:4] = 0
    assert compress.check_sparsity(k, 0.5, "x") == 0.5
    with pytest.raises(compress.CompressError, match="did not take effect"):
        compress.check_sparsity(k, 0.75, "x")


def test_heldout_refused_after_the_student_changed(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, **QUANT)
    compress.run(run)
    p = run / "student" / "student.npz"
    with np.load(p) as z:
        d = dict(z)
    np.savez(p, **{**d, "b0": d["b0"] + 1})
    with pytest.raises(compress.CompressError, match="not the file the stage"):     # edited behind the student
        compress.evaluate_heldout(run)
    sign(run)                                                                       # the student run again
    with pytest.raises(compress.CompressError, match="changed since"):
        compress.evaluate_heldout(run)
    assert not (run / "compress" / "heldout.json").exists()


REFERENCE_FP = "915e50d59de6e016"   # rundir.config_fp(cfg, "compress") of the config in the test



def _edit_npz(p, **change):
    with np.load(p) as z:
        d = dict(z)
    np.savez(p, **{**d, **change})
    return d


def _spec(run):
    with np.load(run / "student" / "student.npz") as z:
        return json.loads(str(z["spec"]))


@pytest.mark.parametrize("edit, match", [
    (lambda s, d: {"w2": None}, "missing"),
    (lambda s, d: {"spec": {k: v for k, v in s.items() if k != "output"}}, "spec needs"),
    (lambda s, d: {"spec": {**s, "activation": "tanh"}}, "only relu"),
    (lambda s, d: {"w1": d["w1"].T.copy()}, "do not match sizes"),
    (lambda s, d: {"b1": d["b1"].astype(np.float64)}, "finite float32"),
    (lambda s, d: {"spec": {**s, "input": {**s["input"], "pad_to": 3}}}, "length <= pad_to"),
    (lambda s, d: {"spec": {**s, "output": {**s["output"], "scale": [1.0]}}}, "one finite entry per output")])
def test_student_file_guards(tmp_path, monkeypatch, edit, match):
    run = make_run(tmp_path, monkeypatch)
    p = run / "student" / "student.npz"
    with np.load(p) as z:
        d = dict(z)
    change = edit(json.loads(str(d["spec"])), d)
    if change.get("w2", 0) is None:
        np.savez(p, **{k: v for k, v in d.items() if k != "w2"})
    else:
        np.savez(p, **{**d, **{k: json.dumps(v) if k == "spec" else v for k, v in change.items()}})
    sign(run)
    with pytest.raises(compress.CompressError, match=match):
        compress.run(run)


def test_teacher_label_rows_and_input_guards(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    p = run / "ridge" / "teacher_labels.npz"
    d = _edit_npz(p)
    meta = json.loads(str(d["meta"]))
    for change, match in (({"meta": json.dumps({**meta, "content": "0" * 64})}, "another build"),
                          ({"meta": json.dumps({**meta, "labels": LABELS[::-1]})}, "label names or units"),
                          ({"pred": d["pred"].astype(np.float32)}, "finite float64")):
        np.savez(p, **{**d, **change})
        sign(run)
        with pytest.raises(compress.CompressError, match=match):
            compress.run(run)
    np.savez(p, **d)
    np.savez(run / "student" / "rows.npz", train=np.arange(40, 200), val=np.array([0, 0, 1]))
    sign(run)
    with pytest.raises(compress.CompressError, match="unique integer positions"):
        compress.run(run)
    with pytest.raises(compress.CompressError, match="made for waveforms of 30 samples"):
        compress.student_input(np.zeros((2, 31), np.float32), _spec(run))


def test_non_finite_training_loss_stops_the_stage(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, steps=20, lr=1e38)
    with pytest.raises(compress.CompressError, match="lower compress.lr"):
        compress.run(run)
    assert not (run / "compress" / "result.json").exists() and not (run / "compress" / ".claim").exists()
    hist = json.loads((run / "compress" / "history.json").read_text())      # the partial curve stays readable
    assert hist[0]["step"] == 0 and np.isfinite(hist[0]["val_loss"])


def test_heldout_needs_the_finished_unchanged_model_on_the_same_data(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, **QUANT)
    with pytest.raises(compress.CompressError, match="no result.json"):
        compress.evaluate_heldout(run)
    compress.run(run)
    h5 = run / "compress" / "model.h5"
    good = h5.read_bytes()
    h5.write_bytes(good + b"\0")
    with pytest.raises(compress.CompressError, match="model.h5 is not the file the stage"):
        compress.evaluate_heldout(run)
    h5.write_bytes(good)
    r = json.loads((run / "compress" / "result.json").read_text())
    (run / "compress" / "result.json").write_text(json.dumps({**r, "data_hash": "0" * 64}))
    with pytest.raises(compress.CompressError, match="another data set than the one"):
        compress.evaluate_heldout(run)
    assert not (run / "compress" / "heldout.json").exists()


# ---- fixes of the final review (R2) ----------------------------------------------------------------------------------

def _labels(run, shift):
    p = run / "ridge" / "teacher_labels.npz"
    d = _edit_npz(p)
    np.savez(p, **{**d, "pred": d["pred"] + shift})                    # same rows, same shape: another teacher


def _rows(run):
    perm = np.random.default_rng(99).permutation(N_DEV)
    np.savez(run / "student" / "rows.npz", train=np.sort(perm[40:]), val=np.sort(perm[:40]))   # disjoint, other rows


def _student(run):
    _edit_npz(run / "student" / "student.npz", b0=_edit_npz(run / "student" / "student.npz")["b0"] + 1)


def _redone(run):                                      # the ridge stage run again, the student not
    _labels(run, 1.0)
    manifest(run / "ridge", ["teacher_labels.npz"])


MUTATIONS = {"teacher labels": (lambda run: _labels(run, 1.0), "teacher_labels.npz is not the file the stage"),
             "teacher labels, ridge redone": (_redone, "not the file the student"),
             "rows": (_rows, "rows.npz is not the file the stage"),
             "student": (_student, "student.npz is not the file the stage"),
             "unfinished student": (lambda run: (run / "student" / "result.json").unlink(), "did not finish")}


@pytest.mark.parametrize("finished", [False, True], ids=["fresh", "after a finished compress"])
@pytest.mark.parametrize("what", list(MUTATIONS))
def test_upstream_files_are_authenticated(tmp_path, monkeypatch, what, finished):
    """H2: every upstream file must be the one its finished stage recorded, also before a finished result is reused."""
    run = make_run(tmp_path, monkeypatch, **QUANT)
    if finished:
        compress.run(run)
    mutate, match = MUTATIONS[what]
    mutate(run)
    with pytest.raises(compress.CompressError, match=match):
        compress.run(run)


def test_a_truth_student_is_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    sign(run, mode="truth")
    with pytest.raises(compress.CompressError, match="distils on teacher labels"):
        compress.run(run)
    assert not (run / "compress" / "result.json").exists()
    sign(run)
    manifest(run / "ridge", ["teacher_labels.npz"], stage="student")              # a copied result.json
    with pytest.raises(compress.CompressError, match="not of 'ridge'"):
        compress.run(run)


@pytest.mark.parametrize("bad", [{"steps": 0.5}, {"steps": float("nan")}, {"prune_steps": -1}, {"steps": True},
                                 {"batch_size": 0}, {"batch_size": 2.0}, {"lr": float("inf")}, {"lr": float("nan")}])
def test_bad_numbers_refused_before_any_work(tmp_path, monkeypatch, bad):
    run = make_run(tmp_path, monkeypatch, **bad)
    with pytest.raises(compress.CompressError, match="integers >= 0"):
        compress.run(run)
    assert not (run / "compress").exists()


def test_one_process_per_stage(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    (run / "compress").mkdir()
    (run / "compress" / ".claim").write_text("")                    # another compress process is at work
    with pytest.raises(compress.CompressError, match="another process"):
        compress.run(run)
    assert sorted(p.name for p in (run / "compress").iterdir()) == [".claim"]
    (run / "compress" / ".claim").unlink()
    compress.run(run)
    assert not (run / "compress" / ".claim").exists()


def test_non_finite_validation_or_weights_publish_nothing(tmp_path, monkeypatch):
    """A zero-step model whose forward pass overflows (finite float32 weights, outputs beyond float32)."""
    run = make_run(tmp_path, monkeypatch)
    d = _edit_npz(run / "student" / "student.npz")
    _edit_npz(run / "student" / "student.npz", w0=d["w0"] * np.float32(1e20), w1=d["w1"] * np.float32(1e20))
    sign(run)
    with pytest.raises(compress.CompressError, match="validation loss"):
        compress.run(run)
    assert not (run / "compress" / "result.json").exists()
    model = compress.build(SIZES)
    k, b = compress.get_weights(model)
    k[1][0, 0] = np.nan
    compress.set_weights(model, k, b)
    with pytest.raises(compress.CompressError, match="non-finite"):
        compress.check_finite(model, {"final": 1.0})
    with pytest.raises(compress.CompressError, match="non-finite"):
        compress.check_finite(compress.build(SIZES), {"final": float("inf")})


def test_history_counts_both_phases(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, prune_steps=5, steps=3, **QUANT)
    compress.run(run)
    hist = json.loads((run / "compress" / "history.json").read_text())
    steps = [("prune", s) for s in range(6)] + [("qat", s) for s in range(5, 9)]     # QAT continues the count
    assert [(h["phase"], h["step"]) for h in hist] == steps


def test_training_never_opens_heldout(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, sparsity=SPARSITY, prune_steps=2, steps=2, **QUANT)
    opened, init, load = [], store.Split.__init__, np.load
    monkeypatch.setattr(store.Split, "__init__", lambda self, path: (opened.append(str(path)), init(self, path))[1])
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), load(f, *a, **k))[1])
    compress.run(run)
    assert any(p.endswith("dev") for p in opened) and not [p for p in opened if "heldout" in p], opened


def _evaluable(tmp_path, monkeypatch, **cfg_change):
    monkeypatch.setitem(sys.modules, "data.w5fake.evaluate", types.SimpleNamespace(evaluate=lambda p, t: {"mae": 1.0}))
    run = make_run(tmp_path, monkeypatch, **QUANT)
    cfg = json.loads((run / "config.json").read_text())
    (run / "config.json").write_text(json.dumps({**cfg, **cfg_change}))
    compress.run(run)
    opened, real = [], compress.split
    monkeypatch.setattr(compress, "split", lambda cfg, name: (opened.append(name), real(cfg, name))[1])
    return run, opened


def test_overlapping_evaluators(tmp_path, monkeypatch):
    """H3: a second evaluator that starts while the first predicts never opens heldout; a sequential one neither;
    a failed evaluation removes its claim; heldout.json is never overwritten."""
    run, opened = _evaluable(tmp_path, monkeypatch)
    real, second = compress.predict, []

    def failing(model, x):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(compress, "predict", failing)
    with pytest.raises(RuntimeError, match="prediction failed"):
        compress.evaluate_heldout(run)
    assert opened == ["heldout"] and not (run / "compress" / ".heldout.claim").exists()

    def overlapping(model, x):
        with pytest.raises(compress.CompressError, match="another process") as e:
            compress.evaluate_heldout(run)
        second.append(e)
        return real(model, x)

    monkeypatch.setattr(compress, "predict", overlapping)
    h = compress.evaluate_heldout(run)
    assert second and opened == ["heldout"] * 2 and not (run / "compress" / ".heldout.claim").exists()
    before = (run / "compress" / "heldout.json").read_text()
    with pytest.raises(compress.CompressError, match="scored once"):
        compress.evaluate_heldout(run)
    assert opened == ["heldout"] * 2 and (run / "compress" / "heldout.json").read_text() == before
    assert json.loads(before) == h and set(h) == {"metrics", "stage", "data_hash", "n", "result"}


def test_a_non_smoke_run_is_scored_only_after_its_curves_are_judged(tmp_path, monkeypatch):
    run, opened = _evaluable(tmp_path, monkeypatch, stages=["ridge", "student", "compress"],
                             run={"group": "g", "name": "run", "smoke": False, "deadline": None})
    for s in ("student", "compress"):
        with pytest.raises(compress.CompressError, match=f"{s} is not judged 'pass'"):
            compress.evaluate_heldout(run)
        assert opened == []
        sha = compress.file_sha256(run / s / "result.json")
        (run / s / "verdict.json").write_text(json.dumps({"verdict": "overfit", "result_sha256": sha}))
        with pytest.raises(compress.CompressError, match="not judged"):
            compress.evaluate_heldout(run)
        (run / s / "verdict.json").write_text(json.dumps({"verdict": "pass", "result_sha256": sha}))
    assert compress.evaluate_heldout(run)["result"]["smoke"] is False and opened == ["heldout"]


FRESH = """
import json, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[3])
import compress
import tensorflow as tf
compress.DATA_ROOT = Path(sys.argv[2])
sys.modules["data.w5fake.evaluate"] = types.SimpleNamespace(evaluate=lambda p, t: {"mae": 1.0})
seen, real = [], compress.predict
compress.predict = lambda m, x: (seen.append(tf.config.experimental.tensor_float_32_execution_enabled()), real(m, x))[1]
compress.evaluate_heldout(sys.argv[1])
print(json.dumps(seen))
"""


def test_evaluation_in_a_fresh_process_is_strict_fp32(tmp_path, monkeypatch):
    """H1: --evaluate in its own process predicts with TF32 off (TensorFlow's default is on). The child sees no GPU,
    so that it does not compete for the memory this process holds: the flag is process-wide either way."""
    run = make_run(tmp_path, monkeypatch, **QUANT)
    compress.run(run)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, "-c", FRESH, str(run), str(tmp_path / "data_root"),
                        str(Path(compress.__file__).parent)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout.strip().splitlines()[-1]) == [False]
