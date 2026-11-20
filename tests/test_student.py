"""The student (models/student.py, train/student.py): sizes, the input transform against rundir.student_input, both
training modes on the fake data sets with a faked ridge stage, the row rule, the export and its contract, the
guards, exact resume, and [ridge, student] through the pipeline and the final evaluation entry (CPU, seconds)."""
import importlib.util
import os
import pathlib
import signal

import numpy as np
import pytest
import torch

import wavefm.data as wdata
from data import store
from wavefm import paths
from wavefm.data import splits
from wavefm.models import student as ms
from wavefm.train import guards, pipeline, rundir, student, trainer
from wavefm.utils import config, runtime
from wavefm.utils.io import load_json

DRO, DCH = [640, 16, 24, 8, 3], [602, 16, 24, 8, 1]


# ---- ① sizes, ② input transform -------------------------------------------------------------------------------------

def test_parameter_counts_and_structure():                                                    # acceptance 1
    for sizes, n in ((DRO, 10891), (DCH, 10265)):
        m = ms.mlp(sizes)
        assert sum(p.numel() for p in m.parameters()) == n
        assert [type(x).__name__ for x in m] == ["Linear", "ReLU"] * 3 + ["Linear"]
        assert [(l.in_features, l.out_features) for l in ms.linears(m)] == list(zip(sizes, sizes[1:]))
        assert all(l.bias is not None for l in ms.linears(m))


def spec(sizes, length, pad_to, every, lo=0.0, hi=1.0):
    return {"sizes": sizes, "activation": "relu", "input": {"length": length, "pad_to": pad_to, "every": every,
                                                            "lo": lo, "hi": hi},
            "output": {"scale": [1.0] * sizes[-1], "offset": [0.0] * sizes[-1]},
            "labels": [{"name": f"y{i}", "unit": "u"} for i in range(sizes[-1])], "data_hash": "h"}


def test_ramp_dro_and_dch():                                                                   # acceptance 2
    ramp = torch.arange(6246, dtype=torch.float32)[None]
    k = ms.kept(ramp, 6246, 6246, 10, 640)[0].numpy()
    assert np.array_equal(k, np.r_[np.arange(0, 6241, 10), np.full(15, 6240)]) and len(k) == 640
    ramp = torch.arange(3000, dtype=torch.float32)[None]
    k = ms.kept(ramp, 3000, 3008, 5, 602)[0].numpy()
    padded = np.r_[np.arange(3000), np.full(8, 2999)]                                         # 3000 -> 3008 by the last
    assert np.array_equal(k, padded[0:3006:5]) and len(k) == 602 and k[-3:].tolist() == [2995, 2999, 2999]
    for sizes, L, pad_to, every in ((DRO, 6246, 6246, 10), (DCH, 3000, 3008, 5)):           # torch == numpy, bit for bit
        x = np.arange(L, dtype=np.float32)[None]
        s = spec(sizes, L, pad_to, every)
        assert np.array_equal(ms.student_input(torch.from_numpy(x), s).numpy(), rundir.student_input(x, s))
        x = np.random.default_rng(0).normal(3, 40, (5, L)).astype(np.float32)
        s = spec(sizes, L, pad_to, every, lo=-123.4567, hi=987.654321)
        assert np.array_equal(ms.student_input(torch.from_numpy(x), s).numpy(), rundir.student_input(x, s))
    with pytest.raises(ms.StudentInputError, match="6246 samples"):
        ms.kept(torch.zeros(2, 6000), 6246, 6246, 10, 640)
    with pytest.raises(ms.StudentInputError, match="n_in"):
        ms.kept(torch.zeros(2, 6246), 6246, 6246, 9, 640)


# ---- the stage on the fake data sets ---------------------------------------------------------------------------------

def student_cfg(fake, targets="teacher", **over):
    """Fake DRO: 256 samples -> every 10th -> 26 kept, padded to 32; fake DCH: 250 -> 256 -> every 5th -> 52."""
    dro = fake["task"] == "dro"
    c = config.merge(fake, {"stages": ["ridge", "student"] if targets == "teacher" else ["student"],
                            "data": {"label_frac": 0.25}, "run": {"smoke": True}})
    c["student"].update({"targets": targets, "pad_to": 256, "every": 10 if dro else 5, "n_in": 32 if dro else 52,
                         "steps": 20, "batch_size": 32, "eval_every": 10, "compile": False, **over})
    config.check(c)
    return c


def fake_ridge(cfg, directory, shift=0.0, labels_of=None):
    """A finished ridge stage directory: teacher labels = true labels * 0.9 + shift (or those of another dev split)."""
    dev = labels_of or wdata.dev(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    teacher = {"config_fp": "fakeridge", "data_hash": dev.meta["data_hash"], "parent": None}
    rundir.write_teacher_labels(directory / "teacher_labels.npz", dev, dev.labels * 0.9 + shift, teacher)
    rundir.write_result(directory, "ridge", cfg, dev.meta["data_hash"], None, requested={"rows": 1},
                        actual={"rows": 1}, hints={}, wall_s=0.0, artifacts=("teacher_labels.npz",))
    return directory


def run(cfg, base, name="run", parent=None):
    """Run the stage in <base>/<name>/student, with the ridge parent <base>/ridge in teacher mode."""
    if parent is None and cfg["student"]["targets"] == "teacher":
        parent = base / "ridge"
        if not parent.exists():
            fake_ridge(cfg, parent)
    out = base / name / "student"
    return student.run(cfg, out, parent), out


def spy_initial(monkeypatch):
    """Record the model's parameters as the stage hands it to the trainer."""
    before = {}
    real = trainer.train

    def spy(out, cfg, stage, model, groups, *a, **k):
        before.clear()
        before.update({n: p.detach().cpu().clone() for n, p in model.named_parameters()})
        return real(out, cfg, stage, model, groups, *a, **k)

    monkeypatch.setattr(trainer, "train", spy)
    return before


def weights(out):
    return torch.load(out / "weights.pt", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("fake", ["fake_dro", "fake_dch"])
@pytest.mark.parametrize("targets", ["teacher", "truth"])
def test_both_modes_train_and_export(fake, targets, request, tmp_path, monkeypatch):          # acceptance 3 and 5
    cfg = student_cfg(request.getfixturevalue(fake), targets)
    before = spy_initial(monkeypatch)
    r, out = run(cfg, tmp_path)
    dev = wdata.dev(cfg)
    k = len(dev.names)
    assert r["actual"]["steps"] == 20 and r["mode"] == targets and r["smoke"]
    assert r["n_params"] == sum(a * b + b for a, b in zip([cfg["student"]["n_in"], 16, 24, 8], [16, 24, 8, k]))
    assert set(r["artifacts"]) == {"weights.pt", "history.json", "rows.npz", "student.npz"}
    assert (r["parent"] is None) == (targets == "truth")
    w = weights(out)
    assert all(not torch.equal(w[n], p) for n, p in before.items()), "a tensor did not change in 20 steps"
    hist = load_json(out / "history.json")
    assert [h["step"] for h in hist] == [10, 20] and all(np.isfinite(h["val_loss"]) for h in hist)
    assert all(("val_truth_loss" in h) == (targets == "teacher") for h in hist)
    assert not (out / "ckpt.pt").exists() and not (out / "run.lock").exists()

    # acceptance 5: the export passes the contract, numpy == torch on the same waveforms, reloading gives the same
    wts, bs, sp = rundir.read_student(out / "student.npz", dev.meta["data_hash"])
    assert sp["sizes"] == [cfg["student"]["n_in"], 16, 24, 8, k] and sp["labels"] == dev.meta["labels"]
    pred = student.predict(out, dev)
    assert pred.shape == dev.labels.shape and pred.dtype == np.float64
    model = ms.mlp(sp["sizes"]); model.load_state_dict(w)
    with torch.no_grad():
        o = model(ms.student_input(torch.from_numpy(dev.take(np.arange(len(dev)))), sp)).double().numpy()
    ref = o * np.array(sp["output"]["scale"]) + np.array(sp["output"]["offset"])
    np.testing.assert_allclose(pred, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())
    assert np.array_equal(student.predict(out, dev), pred)
    for a, b in zip(wts + bs, [l.weight for l in ms.linears(model)] + [l.bias for l in ms.linears(model)]):
        assert np.array_equal(a, b.detach().numpy())
    # finished: reused, not redone
    assert student.run(cfg, out, r["parent"] and out.parent.parent / "ridge") == r


@pytest.mark.parametrize("targets", ["teacher", "truth"])
def test_rows_and_no_label_outside_the_budget(fake_dro, targets, tmp_path, monkeypatch):     # acceptance 4
    cfg = student_cfg(fake_dro, targets)
    d = cfg["data"]
    b_train, b_val = splits.labelled(400, d["label_frac"], d["val_frac"], cfg["seed"], d["min_labelled"], d["min_val"])
    _, a = run(cfg, tmp_path, "clean")
    rows = np.load(a / "rows.npz")
    if targets == "truth":
        assert sorted(rows.files) == ["train", "val"]
        assert np.array_equal(rows["train"], b_train) and np.array_equal(rows["val"], b_val)
    else:
        tr, val, vt = rows["train"], rows["val"], rows["val_truth"]
        perm = np.random.default_rng(cfg["seed"]).permutation(400)
        assert np.array_equal(val, np.sort(perm[:40])) and np.array_equal(vt, b_val)
        assert not np.intersect1d(tr, val).size and not np.intersect1d(tr, vt).size
        assert np.isin(vt, val).all() and len(tr) == 360                                      # budget val is a prefix
        assert np.array_equal(np.union1d(tr, val), np.arange(400))

    real = wdata.dev
    budget = np.union1d(b_train, b_val)

    def changed(c):                                                  # every true label outside the budget is NaN
        s = real(c)
        s.labels = s.labels.copy()
        s.labels[np.setdiff1d(np.arange(len(s)), budget)] = np.nan
        return s

    monkeypatch.setattr(wdata, "dev", changed)
    _, b = run(cfg, tmp_path, "changed")
    wa, wb = weights(a), weights(b)
    assert all(torch.equal(wa[n], wb[n]) for n in wa)
    assert load_json(a / "history.json") == load_json(b / "history.json")
    for f in ("rows.npz", "student.npz"):
        with np.load(a / f) as x, np.load(b / f) as y:
            assert x.files == y.files and all(np.array_equal(x[n], y[n]) for n in x.files)


def test_true_labels_only_reach_the_diagnostic_curve(fake_dro, tmp_path, monkeypatch):
    """Teacher mode: the true labels of the budget's validation rows changed, the teacher labels fixed. Only
    val_truth_loss moves; val_loss and the weights do not. Both curves recomputed from the exported student."""
    cfg = student_cfg(fake_dro)
    _, a = run(cfg, tmp_path, "clean")
    vt = np.load(a / "rows.npz")["val_truth"]
    real = wdata.dev

    def shifted(c):
        s = real(c)
        s.labels = s.labels.copy()
        s.labels[vt] += 5.0
        return s

    monkeypatch.setattr(wdata, "dev", shifted)
    _, b = run(cfg, tmp_path, "shifted")
    ha, hb = load_json(a / "history.json"), load_json(b / "history.json")
    assert [h["val_loss"] for h in ha] == [h["val_loss"] for h in hb]
    assert all(x["val_truth_loss"] != y["val_truth_loss"] for x, y in zip(ha, hb))
    wa, wb = weights(a), weights(b)
    assert all(torch.equal(wa[n], wb[n]) for n in wa)
    monkeypatch.setattr(wdata, "dev", real)
    dev = wdata.dev(cfg)
    rows = np.load(a / "rows.npz")
    sp = rundir.read_student(a / "student.npz", dev.meta["data_hash"])[2]
    pred = student.predict(a, dev)
    teacher = rundir.read_teacher_labels(tmp_path / "ridge" / "teacher_labels.npz", dev)
    for key, idx, y in (("val_loss", rows["val"], teacher), ("val_truth_loss", rows["val_truth"], dev.labels)):
        p, t = ((v[idx] - sp["output"]["offset"]) / sp["output"]["scale"] for v in (pred, y))
        assert ((p - t) ** 2).sum(1).mean() == pytest.approx(ha[-1][key], rel=1e-4), key


def test_validation_does_not_depend_on_eval_batch(fake_dro, tmp_path):
    _, a = run(student_cfg(fake_dro, eval_batch=7), tmp_path, "small")
    _, b = run(student_cfg(fake_dro, eval_batch=4096), tmp_path, "large")
    for x, y in zip(load_json(a / "history.json"), load_json(b / "history.json")):
        assert x["val_loss"] == pytest.approx(y["val_loss"], rel=1e-5)
        assert x["val_truth_loss"] == pytest.approx(y["val_truth_loss"], rel=1e-5)


@pytest.mark.parametrize("targets", ["teacher", "truth"])
def test_training_never_opens_heldout(fake_dro, targets, tmp_path, monkeypatch):
    """Every split opened, array loaded and file read by the stage is recorded; none is under heldout/."""
    cfg = student_cfg(fake_dro, targets)
    if targets == "teacher":
        fake_ridge(cfg, tmp_path / "ridge")
    opened, init, load, read = [], store.Split.__init__, np.load, pathlib.Path.read_text
    monkeypatch.setattr(store.Split, "__init__", lambda self, path: (opened.append(str(path)), init(self, path))[1])
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), load(f, *a, **k))[1])
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: (opened.append(str(self)), read(self, *a, **k))[1])
    run(cfg, tmp_path)
    assert any(p.endswith("dev") for p in opened) and not [p for p in opened if "heldout" in p], opened


def test_scaling_constants_and_output_bias(fake_dro, tmp_path, monkeypatch):
    before = spy_initial(monkeypatch)
    dev = wdata.dev(fake_dro)
    for name, over in (("a", {}), ("b", {"input_scaling": "kept_minmax", "target_scaling": "divisor",
                                         "divisors": [50.0, 20.0, 0.5], "output_bias": "mean",
                                         "loss_weights": [1.0, 1.0, 0.01]})):
        cfg = student_cfg(fake_dro, "teacher", **over)
        _, out = run(cfg, tmp_path, name)
        tr = np.load(out / "rows.npz")["train"]
        teacher = rundir.read_teacher_labels(tmp_path / "ridge" / "teacher_labels.npz", dev)[tr]
        sp = rundir.read_student(out / "student.npz", dev.meta["data_hash"])[2]
        res = rundir.read_result(out)
        bias = before["6.bias"].numpy()
        runtime.seed_all(cfg["seed"])                                # PyTorch default initialisation from the seed
        fresh = ms.mlp(sp["sizes"]).state_dict()
        assert all(torch.equal(before[n], fresh[n]) for n in fresh if n != "6.bias")
        if name == "a":
            st = dev.meta["waveform"]["stats"]
            assert (sp["input"]["lo"], sp["input"]["hi"]) == (st["min"], st["max"])
            assert np.array_equal(sp["output"]["offset"], teacher.mean(0))
            assert np.array_equal(sp["output"]["scale"], teacher.std(0))
            assert np.array_equal(bias, np.zeros(3, np.float32))
        else:
            x = np.pad(dev.take(tr)[:, ::10], ((0, 0), (0, 6)), mode="edge")              # 256 -> 26 kept -> 32
            assert (sp["input"]["lo"], sp["input"]["hi"]) == (float(x.min()), float(x.max()))
            assert sp["output"]["scale"] == [50.0, 20.0, 0.5] and sp["output"]["offset"] == [0.0] * 3
            z = (teacher / np.array([50.0, 20.0, 0.5])).astype(np.float32)
            np.testing.assert_allclose(bias, z.astype(np.float64).mean(0), rtol=1e-6)
        assert res["input_lo_hi"] == [sp["input"]["lo"], sp["input"]["hi"]]


def test_teacher_labels_of_another_data_set_or_changed_are_refused(fake_dro, fake_dch, tmp_path):
    cfg = student_cfg(fake_dro)
    other = fake_ridge(cfg, tmp_path / "other" / "ridge", labels_of=wdata.dev(fake_dch))
    with pytest.raises(rundir.RunError, match="another data set"):
        student.run(cfg, tmp_path / "x" / "student", other)
    r, out = run(cfg, tmp_path)
    assert r["parent"]["sha256"] == rundir.file_sha256(tmp_path / "ridge" / "teacher_labels.npz")
    fake_ridge(cfg, tmp_path / "ridge", shift=1.0)                   # ridge redone after the student finished
    with pytest.raises(rundir.RunError, match="another parent"):
        student.run(cfg, out, tmp_path / "ridge")
    dev = wdata.dev(cfg)                                             # the label file edited behind the ridge stage
    rundir.write_teacher_labels(tmp_path / "ridge" / "teacher_labels.npz", dev, dev.labels * 1.1,
                                {"config_fp": "f", "data_hash": dev.meta["data_hash"], "parent": None})
    with pytest.raises(student.StudentError, match="not the file the ridge stage"):
        student.run(cfg, tmp_path / "y" / "student", tmp_path / "ridge")


@pytest.mark.parametrize("over, match", [({"targets": "both"}, "targets"), ({"input_scaling": "z"}, "input_scaling"),
                                         ({"target_scaling": "divisor"}, "divisors"),
                                         ({"divisors": [1.0, 1.0, 1.0]}, "divisors is set"),
                                         ({"loss_weights": [1.0]}, "loss_weights"), ({"output_bias": "x"}, "output_bias"),
                                         ({"every": 7}, "n_in"), ({"pad_to": 200}, "pad_to"), ({"hidden": []}, "hidden")])
def test_bad_config_is_refused_before_any_work(fake_dro, tmp_path, over, match):
    with pytest.raises(student.StudentError, match=match):
        run(student_cfg(fake_dro, **over), tmp_path)
    assert not (tmp_path / "run" / "student").exists()


def test_mode_and_parent_must_agree(fake_dro, tmp_path):
    with pytest.raises(student.StudentError, match="ridge stage before"):
        student.run(student_cfg(fake_dro, "teacher"), tmp_path / "a" / "student", None)
    p = fake_ridge(student_cfg(fake_dro), tmp_path / "ridge")
    with pytest.raises(student.StudentError, match="from scratch"):
        student.run(student_cfg(fake_dro, "truth"), tmp_path / "b" / "student", p)


def test_exact_resume(fake_dro, tmp_path, monkeypatch):
    cfg = student_cfg(fake_dro, steps=30)
    _, ref = run(cfg, tmp_path, "ref")
    real, calls = trainer.train, {"n": 0}

    def killing(out, cfg, stage, model, groups, loss, *a, **k):
        def kl(b, gen):
            calls["n"] += 1
            if calls["n"] == 15:
                os.kill(os.getpid(), signal.SIGTERM)
            return loss(b, gen)
        return real(out, cfg, stage, model, groups, kl, *a, **k)

    monkeypatch.setattr(trainer, "train", killing)
    first, out = run(cfg, tmp_path, "int")
    assert first == {"preempted": True} and (out / "ckpt.pt").exists()
    monkeypatch.setattr(trainer, "train", real)
    run(cfg, tmp_path, "int")
    wa, wb = weights(ref), weights(out)
    assert all(torch.equal(wa[n], wb[n]) for n in wa)
    assert load_json(ref / "history.json") == load_json(out / "history.json")
    assert not (out / "ckpt.pt").exists()


# ---- [ridge, student] through the pipeline, then the final evaluation entry ------------------------------------------

class TinyEncoder(torch.nn.Module):
    """build_encoder's interface: forward(x, block=-1) -> (B, N, D) tokens of 32-sample patches."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.randn(32, 24, generator=torch.Generator().manual_seed(0)) / 4)

    def forward(self, x, block=-1):
        return torch.tanh(x.reshape(len(x), -1, 32) @ self.proj)


def script(name):
    spec_ = importlib.util.spec_from_file_location(f"script_{name}", paths.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def test_ridge_then_student_through_the_pipeline_and_evaluate(fake_dro, monkeypatch, tmp_path):
    from wavefm.models import backbones, build
    monkeypatch.setattr(build, "build_encoder", lambda model_cfg, init_from=None, seed=0: TinyEncoder(), raising=False)
    (tmp_path / "pretrained").write_bytes(b"stand-in")                    # what ridge, a first stage, records
    monkeypatch.setattr(backbones, "pretrained_file", lambda version: str(tmp_path / "pretrained"))
    cfg = student_cfg(fake_dro)
    done = pipeline.run(cfg)
    d = pipeline.run_dir(cfg)
    r = done["results"]["student"]
    assert r["parent"]["sha256"] == rundir.file_sha256(d / "ridge" / "teacher_labels.npz")
    assert pipeline.run(cfg)["results"]["student"] == r                                     # finished: reused
    rec = script("evaluate").evaluate(d)                                                  # heldout, physical units
    assert rec["stage"] == "student" and rec["n"] == 100 and rec["metrics"]
    with wdata.final_evaluation():
        held = wdata.heldout(cfg)
        assert student.predict(d / "student", held).shape == held.labels.shape
        with np.load(d / "student" / "student.npz") as z:
            z = dict(z)
        np.savez(d / "student" / "student.npz", **{**z, "b0": z["b0"] + 1})       # edited after the stage finished
        with pytest.raises(guards.GuardError, match="student.npz changed after the stage finished"):
            student.predict(d / "student", held)


def test_constant_targets_or_inputs_are_refused():
    s = {"target_scaling": "standard", "input_scaling": "kept_minmax"}
    with pytest.raises(student.StudentError, match="constant"):
        student.target_scaling(s, np.array([[1.0, 2.0], [1.0, 3.0]]))
    with pytest.raises(student.StudentError, match="constant"):
        student.input_range(s, None, torch.full((4, 8), 2.0))


def test_too_few_dev_rows_for_a_teacher_label_validation_set(fake_dro):
    """val_frac * n rounds to no validation row."""
    cfg = config.merge(fake_dro, {"student": {"targets": "teacher"}, "data": {"val_frac": 0.001}})
    with pytest.raises(student.StudentError, match="leaves no validation"):
        student.rows(cfg, 400)
