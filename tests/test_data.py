"""wavefm.data: the dev / heldout access and its guards, the labelled rows, the scalers, the waveform sets."""
import json
import pathlib

import numpy as np
import pytest
import torch

import wavefm.data
from data import store
from wavefm import tasks
from wavefm.data import scaling, splits
from wavefm.data.batches import WaveformSet, pad_to_patch


# ---- padding --------------------------------------------------------------------------------------------------------

def test_pad_to_patch_end_edge():
    x = torch.arange(2 * 6246, dtype=torch.float32).reshape(2, 6246)
    p = pad_to_patch(x)
    assert p.shape == (2, 6272)
    assert torch.equal(p[:, :6246], x) and torch.all(p[:, 6246:] == x[:, -1:])
    assert pad_to_patch(p).shape == (2, 6272)


# ---- labelled rows --------------------------------------------------------------------------------------------------

def test_labelled_rule():
    """Exactly the rule of the docstring, written out again."""
    for n, lf, vf, seed in ((1000, 0.1, 0.1, 0), (1000, 1.0, 0.1, 3), (80000, 0.01, 0.1, 7), (50, 1.0, 0.2, 1)):
        tr, va = splits.labelled(n, lf, vf, seed, 32, 8)
        perm = np.random.default_rng(seed).permutation(n)
        n_lab = n if lf == 1 else round(lf * n)
        n_val = max(round(vf * n_lab), 8)
        np.testing.assert_array_equal(va, np.sort(perm[:n_val]))
        np.testing.assert_array_equal(tr, np.sort(perm[n_val:n_lab]))
        assert not np.intersect1d(tr, va).size and len(tr) + len(va) == n_lab
        assert (np.diff(tr) > 0).all() and (np.diff(va) > 0).all() and max(tr.max(), va.max()) < n


def test_labelled_minimums_budget_and_seeds():
    tr, va = splits.labelled(1000, 0.04, 0.1, 0, 32, 8)                # 4 validation rows asked: min_val applies
    assert (len(tr), len(va)) == (32, 8)
    with pytest.raises(splits.BudgetError, match="min_labelled"):      # a budget is never silently enlarged
        splits.labelled(1000, 0.001, 0.1, 0, 32, 8)
    tr1, va1 = splits.labelled(1000, 1.0, 0.1, 0, 32, 8)
    assert len(tr1) + len(va1) == 1000                                 # label_frac 1: every row
    a, b = splits.labelled(1000, 0.2, 0.1, 5, 32, 8), splits.labelled(1000, 0.2, 0.1, 5, 32, 8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))             # deterministic
    c = splits.labelled(1000, 0.2, 0.1, 6, 32, 8)
    assert not np.array_equal(a[0], c[0]) and not np.array_equal(a[1], c[1])    # another seed, another draw
    small, big = splits.labelled(1000, 0.05, 0.1, 5, 32, 8), splits.labelled(1000, 0.2, 0.1, 5, 32, 8)
    assert set(np.concatenate(small)) <= set(np.concatenate(big))      # nested budgets


@pytest.mark.parametrize("args", [(100, 0.0, 0.1), (100, 1.5, 0.1), (100, -0.1, 0.1), (100, 0.5, 0.0), (100, 0.5, 1.2),
                                  (20, 0.5, 0.1), (40, 0.5, 0.9), (8, 1.0, 0.1), (100, 0.5, 1.0),
                                  (100, 0.31, 0.1)])
def test_labelled_guards(args):
    """Fractions outside (0, 1], a dev too small for the minimums, and an empty training set are refused."""
    with pytest.raises(splits.SplitError):
        splits.labelled(*args, seed=0, min_labelled=32, min_val=8)


# ---- scalers --------------------------------------------------------------------------------------------------------

def test_scalers_round_trip():
    meta = {"waveform": {"stats": {"min": -3.0, "max": 5.0, "mean": 0.1, "std": 1.0}}}
    g = scaling.input_scaler("global_minmax", meta)
    assert (g.lo, g.hi) == (-3.0, 5.0) and scaling.input_scaler("native", meta) is None
    x = np.array([[-3.0, 1.0, 5.0]], np.float32)
    np.testing.assert_allclose(g(x), [[0.0, 0.5, 1.0]])
    Y = np.random.default_rng(1).normal(3, 2, size=(200, 3))
    st = scaling.target_scaler("standard", Y)
    np.testing.assert_allclose(st.mean, Y.mean(0), rtol=1e-15); np.testing.assert_allclose(st.std, Y.std(0))
    dv = scaling.target_scaler("divisor", Y, [500, 220, 2.5])
    for s in (g, st, dv):
        r = scaling.from_dict(json.loads(json.dumps(s.to_dict())))
        assert r == s and type(r) is type(s)
    for s in (st, dv):
        np.testing.assert_allclose(s.inverse(s.forward(Y)), Y, rtol=1e-12)
    with pytest.raises(ValueError, match="divisors"):
        scaling.target_scaler("divisor", Y, [1.0])
    with pytest.raises(ValueError, match="constant"):
        scaling.target_scaler("standard", np.ones((10, 1)))
    with pytest.raises(ValueError, match="constant"):
        scaling.input_scaler("global_minmax", {"waveform": {"stats": {"min": 1.0, "max": 1.0}}})


# ---- waveform sets --------------------------------------------------------------------------------------------------

def test_waveform_set_reads_pads_and_masks(fake_dch):
    """Rows come from Split.take in chunks; waveforms are edge-padded to 256, per-sample targets padded with 0, and
    the mask marks the 250 real samples."""
    dev = wavefm.data.dev(fake_dch)
    idx = np.array([399, 0, 5, 199, 200, 201, 37])                     # across the two external files
    cls = dev.extra("classes")
    w = WaveformSet(dev, idx, "cpu", targets={"classes": cls[idx], "n": dev.labels[idx]}, chunk=3)
    W = dev.take(idx)
    assert w.x.shape == (7, 256) and torch.equal(w.x[:, :250], torch.from_numpy(W))
    assert torch.all(w.x[:, 250:] == w.x[:, 249:250])
    assert w.t["classes"].shape == (7, 256) and (w.t["classes"][:, 250:] == 0).all()
    assert torch.equal(w.t["classes"][:, :250], torch.from_numpy(cls[idx]))
    assert w.mask.sum() == 250 and w.mask[:250].all() and w.t["n"].shape == (7, 1)
    x, t = w.batch(torch.tensor([2, 0]))
    assert torch.equal(x, w.x[[2, 0]]) and torch.equal(t["mask"], w.mask)
    got = [(x.shape[0], n) for x, _, n in w.ordered_batches(3)]
    assert got == [(3, 3), (3, 3), (3, 1)]


# ---- dev, heldout and their guards ----------------------------------------------------------------------------------

def test_dev_checks_the_data_hash(fake_dro):
    dev = wavefm.data.dev(fake_dro)
    assert dev.meta["split"] == "dev" and len(dev) == 400 and dev.names == ["C", "S", "t0"]
    f = wavefm.data.ROOT / "dro_gen" / "configs" / "fake.yaml"
    f.write_text(f.read_text().replace("dt_ns: 0.32", "dt_ns: 0.33"))           # the definition changed
    with pytest.raises(store.StaleDataError, match="prepare_data.py dro_gen fake"):
        wavefm.data.dev(fake_dro)
    f.write_text(f.read_text().replace("dt_ns: 0.33", "dt_ns: 0.32"))
    (wavefm.data.ROOT / "dro_gen" / "build.py").write_text("# changed code\n")    # the code changed
    with pytest.raises(store.StaleDataError):
        wavefm.data.dev(fake_dro)


def _damage(f):
    a = np.load(f)
    a.flat[7] += 1
    np.save(f, a)


@pytest.mark.parametrize("name", ["waveform", "labels", "rows"])
def test_damaged_dev_is_refused(fake_dro, name):
    """② A dev file whose content does not match its digest is never used, through dev() or WaveformSet."""
    _damage(wavefm.data.dev(fake_dro).path / f"{name}.npy")
    with pytest.raises(store.DigestError):
        dev = wavefm.data.dev(fake_dro)
        WaveformSet(dev, np.arange(10), "cpu")
    with pytest.raises(store.DigestError):
        tasks.get("dro").prepare(fake_dro, "cpu")


def test_damaged_external_part_and_extra_are_refused(fake_dch):
    dev = wavefm.data.dev(fake_dch)
    f = dev.meta["waveform"]["parts"][1]["file"]
    with np.load(f) as z:
        a = z["wf_i"].copy()
    a[3, 3] += 1
    np.savez_compressed(f, wf_i=a)
    with pytest.raises(store.DigestError):
        WaveformSet(wavefm.data.dev(fake_dch), np.array([0, 350]), "cpu")
    _damage(dev.path / "classes.npy")
    with pytest.raises(store.DigestError):
        tasks.get("dch").prepare(fake_dch, "cpu")


def test_heldout_is_locked(fake_dro):
    """① heldout() outside the final evaluation raises; inside it opens heldout, and the lock closes again, also
    after an error."""
    with pytest.raises(wavefm.data.HeldoutError):
        wavefm.data.heldout(fake_dro)
    with wavefm.data.final_evaluation():
        h = wavefm.data.heldout(fake_dro)
    assert h.meta["split"] == "heldout" and len(h) == 100
    with pytest.raises(ZeroDivisionError):
        with wavefm.data.final_evaluation():
            1 / 0
    with pytest.raises(wavefm.data.HeldoutError):
        wavefm.data.heldout(fake_dro)


def _dch_forward(x):
    return {"logits": torch.stack([x, -x, 0.5 * x], -1), "count": x.mean(1)}


@pytest.mark.parametrize("fixture", ["fake_dro", "fake_dch"])
def test_training_path_never_opens_heldout(fixture, request, monkeypatch):
    """① Fault injection: every split opened, every array loaded and every file read on the training path is
    recorded; none is under heldout/ (arrays or meta.json)."""
    cfg = request.getfixturevalue(fixture)
    opened, init = [], store.Split.__init__
    load, read = np.load, pathlib.Path.read_text
    monkeypatch.setattr(store.Split, "__init__", lambda self, path: (opened.append(str(path)), init(self, path))[1])
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), load(f, *a, **k))[1])
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (opened.append(str(self)), read(self, *a, **k))[1])
    task = tasks.get(cfg["task"])
    train, val = task.prepare(cfg, "cpu")
    fwd = _dch_forward if cfg["task"] == "dch" else (lambda x: x[:, :3] * 0.01)
    x, t = train.batch(torch.arange(8))
    if cfg["task"] == "dro":
        x, t = task.augment(x, t, torch.Generator().manual_seed(0)) if task.shift else (x, t)
    assert torch.isfinite(task.loss(fwd(x), t))
    assert np.isfinite(task.val_loss(fwd, val, 16))
    task.score(task.predict(fwd, val, 16), val)
    assert any(p.endswith("dev") for p in opened) and any("dev" in p and p.endswith(".npy") for p in opened)
    assert not [p for p in opened if "heldout" in p], opened



def test_waveform_set_refuses_misaligned_or_non_finite_targets(fake_dro):
    dev = wavefm.data.dev(fake_dro)
    with pytest.raises(ValueError, match="3 rows for 4 positions"):
        WaveformSet(dev, np.arange(4), "cpu", targets={"y": np.zeros(3)})
    with pytest.raises(ValueError, match="non-finite"):
        WaveformSet(dev, np.arange(4), "cpu", targets={"y": np.array([0.0, np.nan, 1.0, 2.0])})


def test_unknown_target_scaling_and_task_are_refused():
    with pytest.raises(ValueError, match="unknown finetune.target_scaling"):
        scaling.target_scaler("minmax", np.ones((4, 1)))
    with pytest.raises(KeyError, match="unknown task"):
        tasks.get("bso")


def test_data_config_without_out_is_refused(tmp_path):
    (tmp_path / "s" / "configs").mkdir(parents=True)
    (tmp_path / "s" / "configs" / "c.yaml").write_text("seed: 1\n")
    with pytest.raises(store.FormatError, match="must say where it is written"):
        store.load_config("s", "c", tmp_path)


def test_write_split_refuses_rows_that_do_not_match_the_waveforms(fake_dro, tmp_path):
    dev = wavefm.data.dev(fake_dro)
    cfg = store.load_config("dro_gen", "fake", wavefm.data.ROOT)
    with pytest.raises(store.FormatError, match="5 waveforms for 4 rows"):
        store.write_split(tmp_path / "new" / "dev", cfg, "dev", dev.labels[:4], dev.rows[:4], dev.meta["labels"],
                          dt_ns=0.32, t_first_ns=0.0, unit="mV", waveform=dev.take(np.arange(5)),
                          root=wavefm.data.ROOT)


def test_part_count_shape_and_statistics_disagreeing_with_meta_are_refused(fake_dch):
    dev = wavefm.data.dev(fake_dch)
    meta_file = dev.path / "meta.json"
    meta = json.loads(meta_file.read_text())
    bad = json.loads(meta_file.read_text())
    bad["waveform"]["parts"][0]["n"] += 1
    meta_file.write_text(json.dumps(bad))
    with pytest.raises(store.FormatError, match="waveform parts hold"):
        store.Split(dev.path)
    bad = json.loads(json.dumps(meta))
    bad["label_stats"]["n_primary"]["mean"] += 1.0
    meta_file.write_text(json.dumps(bad))
    with pytest.raises(store.FormatError, match="statistics of n_primary differ"):
        store.Split(dev.path).verify()
    meta_file.write_text(json.dumps(meta))
    f = meta["waveform"]["parts"][1]["file"]
    with np.load(f) as z:
        a = z["wf_i"][:-1].copy()
    np.savez_compressed(f, wf_i=a)                              # one event short
    with pytest.raises(store.FormatError, match="waveform part"):
        store.Split(dev.path).take(np.array([399]))


def test_rows_not_increasing_are_refused_by_verify(fake_dro):
    """Digests can be made to match (meta.json rewritten with them): verify still refuses unordered rows."""
    dev = wavefm.data.dev(fake_dro)
    rows = dev.rows[::-1].copy()
    np.save(dev.path / "rows.npy", rows)
    meta = json.loads((dev.path / "meta.json").read_text())
    meta["sha256"]["rows"] = store.digest(rows)
    (dev.path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(store.FormatError, match="rows not increasing"):
        store.Split(dev.path).verify()
