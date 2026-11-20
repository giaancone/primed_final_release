"""data/dch_wf: labels and classes from the tag arrays, the metric, file reads, and the built data set on CFS."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from data import store
from data.dch_wf import build
from data.dch_wf.evaluate import ShapeError, evaluate

REAL = Path(store.load_config("dch_wf", "pion")["out"])
L, T, SIZES = 40, 6, {"dev": (7, 5, 4), "heldout": (3, 6)}


def fake_tags(rng, n):
    """Tags with padding (value 0), times outside [0, L), and a sample tagged both 1 and 2 (in both orders)."""
    tt = rng.integers(-3, L + 3, (n, T)).astype(np.int32)
    tv = rng.integers(0, 3, (n, T)).astype(np.int32)
    tt[:, -2:], tv[:, -2:] = 0, 0                                  # padded slots
    tt[0, :4], tv[0, :4] = [5, 5, 9, 9], [1, 2, 2, 1]              # samples 5 and 9 tagged both
    tt[1, :3], tv[1, :3] = [-1, L, L - 1], [1, 1, 1]               # two primaries outside the window, one inside
    tv[1, 3:] = 0
    return tt, tv


def fake_set(tmp_path, wf_dtype=np.float32):
    """Fake npz files in tmp_path/src and the config that builds them to tmp_path/out. Returns (cfg, data)."""
    rng = np.random.default_rng(0)
    cfg = copy.deepcopy(store.load_config("dch_wf", "pion"))
    cfg.update(source=str(tmp_path / "src"), out=str(tmp_path / "out"), files={})
    (tmp_path / "src").mkdir()
    data = {}
    for split, sizes in SIZES.items():
        cfg["files"][split], data[split] = [], []
        for i, n in enumerate(sizes):
            wf = rng.normal(0, 1, (n, L)).astype(wf_dtype)
            tt, tv = fake_tags(rng, n)
            name = f"{split}_{i}.npz"
            np.savez_compressed(tmp_path / "src" / name, wf_i=wf, mom=rng.uniform(0.2, 20, n).astype(np.float32),
                                tag_times=tt, tag_values=tv)
            cfg["files"][split].append(name)
            data[split].append((wf.astype(np.float32), tt, tv))
    return cfg, data


def hand_classes(tt, tv):
    out = np.zeros((len(tt), L), np.int8)
    for r in range(len(tt)):
        for t, v in zip(tt[r], tv[r]):
            if v in (1, 2) and 0 <= t < L:
                out[r, t] = max(out[r, t], v)
    return out


def test_labels_and_classes(tmp_path):
    """① n_primary is the count of tags == 1 inside [0, L) row by row; classes equal a hand-written loop."""
    cfg, data = fake_set(tmp_path)
    assert store.prepare(cfg) == "built"
    for split, files in data.items():
        s = store.Split(Path(cfg["out"]) / split).verify()
        wf = np.concatenate([f[0] for f in files])
        tt, tv = np.concatenate([f[1] for f in files]), np.concatenate([f[2] for f in files])
        want = [sum(1 for t, v in zip(tt[r], tv[r]) if v == 1 and 0 <= t < L) for r in range(len(tt))]
        np.testing.assert_array_equal(s.labels[:, 0], want)
        np.testing.assert_array_equal(s.extra("classes"), hand_classes(tt, tv))
        np.testing.assert_array_equal(s.take(np.arange(len(s))), wf)
        np.testing.assert_array_equal(s.rows, np.arange(sum(SIZES[split])))
        assert s.names == ["n_primary"] and s.meta["labels"][0]["unit"] == "clusters" and s.labels.dtype == np.float32
        assert (s.meta["dt_ns"], s.meta["t_first_ns"], s.meta["unit"], s.length) == (1 / 1.5, 0.0, "a.u.", L)
    cls = store.Split(Path(cfg["out"]) / "dev").extra("classes")
    assert cls[0, 5] == 2 and cls[0, 9] == 2                             # a 2 overwrites a 1, in either order
    assert store.Split(Path(cfg["out"]) / "dev").labels[1, 0] == 1                # tags at -1 and L are not counted


def test_float64_source_is_stored_as_float32(tmp_path):
    cfg, data = fake_set(tmp_path, wf_dtype=np.float64)
    store.prepare(cfg)
    s = store.Split(Path(cfg["out"]) / "heldout")
    np.testing.assert_array_equal(s.take(np.arange(len(s))), np.concatenate([f[0] for f in data["heldout"]]))


def _rewrite(cfg, i, **members):
    """Replace members of the i-th dev file of the fake set."""
    f = Path(cfg["source"]) / cfg["files"]["dev"][i]
    with np.load(f) as z:
        m = {k: z[k] for k in z.files}
    np.savez_compressed(f, **{**m, **members})


@pytest.mark.parametrize("fault", ["tag_shape", "length", "nan"])
def test_build_guards(tmp_path, fault):
    """Inconsistent tags, waveform lengths that differ between files and non-finite samples stop the build before
    anything is written."""
    cfg, data = fake_set(tmp_path)
    wf, tt, tv = data["dev"][1]
    if fault == "tag_shape":
        _rewrite(cfg, 1, tag_values=tv[:, :-1])
    elif fault == "length":
        _rewrite(cfg, 1, wf_i=wf[:, :-1])
    elif fault == "nan":
        _rewrite(cfg, 1, wf_i=np.where(np.arange(L) == 3, np.nan, wf).astype(np.float32))
    with pytest.raises(store.FormatError):
        store.prepare(cfg)
    assert not (Path(cfg["out"]) / "dev").exists() and not (Path(cfg["out"]) / "heldout").exists()


def test_evaluate():
    """③ MAE and RMSE equal hand-computed values; invalid predictions give inf, never NaN; shapes must agree."""
    truth = np.array([[10.0], [20.0], [30.0], [41.0]])
    pred = np.array([[12.0], [19.0], [30.0], [37.0]])                 # errors 2, 1, 0, 4
    r = evaluate(pred, truth)
    assert r == {"count_mae": 7 / 4, "count_rmse": math.sqrt(21 / 4), "n": 4}
    assert evaluate(pred[:, 0], truth[:, 0]) == r
    assert isinstance(r["count_mae"], float)
    for bad in (np.nan, np.inf, -np.inf):
        for k in (1, 4):
            p = pred.copy(); p[:k] = bad
            r = evaluate(p, truth)
            assert r["count_mae"] == math.inf and r["count_rmse"] == math.inf and r["n"] == 4
    for p, t in ((pred, truth[:, 0]), (pred[:3], truth), (np.ones((4, 2)), np.ones((4, 2))), (pred[:0], truth[:0])):
        with pytest.raises(ShapeError):
            evaluate(p, t)
    with pytest.raises(ValueError):
        evaluate(pred, np.where(truth == 10, np.nan, truth))


def test_take_reads_each_file_once(tmp_path, monkeypatch):
    """④ A shuffled batch crossing files is correct, and many take calls open each npz file exactly once."""
    cfg, data = fake_set(tmp_path)
    store.prepare(cfg)
    wf = np.concatenate([f[0] for f in data["dev"]])
    opened, load = [], np.load
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(Path(f)), load(f, *a, **k))[1])
    s = store.Split(Path(cfg["out"]) / "dev")
    rng = np.random.default_rng(1)
    idx = np.array([15, 0, 7, 6, 12, 11, 3, 3])                       # crosses all three files, repeats, any order
    np.testing.assert_array_equal(s.take(idx), wf[idx])
    for _ in range(50):
        idx = rng.integers(0, len(s), 5)
        np.testing.assert_array_equal(s.take(idx), wf[idx])
    npz = [p for p in opened if p.suffix == ".npz"]
    assert sorted(npz) == sorted(Path(cfg["source"]) / f for f in cfg["files"]["dev"])


def test_first_rows(tmp_path, monkeypatch):
    """first_rows: a split is its first N events in file order, stored in the split as waveform.npy; files after the
    one that completes N are not read; labels and classes are those of the same rows; pion itself has no limit."""
    assert "first_rows" not in store.load_config("dch_wf", "pion")
    cfg, data = fake_set(tmp_path)
    cfg["first_rows"] = {"dev": 9, "heldout": 3}                   # dev crosses files 0 and 1 (7 + 2), never reads 2
    opened, load = [], np.load
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(Path(f).name), load(f, *a, **k))[1])
    store.prepare(cfg)
    assert "dev_2.npz" not in opened and "heldout_1.npz" not in opened
    monkeypatch.setattr(np, "load", load)
    for split, n in cfg["first_rows"].items():
        s = store.Split(Path(cfg["out"]) / split).verify()
        wf = np.concatenate([f[0] for f in data[split]])[:n]
        tt, tv = (np.concatenate([f[k] for f in data[split]])[:n] for k in (1, 2))
        assert len(s) == n and [p["file"] for p in s.meta["waveform"]["parts"]] == ["waveform.npy"]
        np.testing.assert_array_equal(s.take(np.arange(n)), wf)
        np.testing.assert_array_equal(s.rows, np.arange(n))
        np.testing.assert_array_equal(s.labels[:, 0], (tv == 1).sum(1) - ((tv == 1) & ((tt < 0) | (tt >= L))).sum(1))
        np.testing.assert_array_equal(s.extra("classes"), hand_classes(tt, tv))


@pytest.mark.parametrize("limit", [17, 0, -1, 2.5, True])
def test_first_rows_guards(tmp_path, limit):
    """More rows than the files hold, or a limit that is not a positive integer, stop the build before anything is
    written."""
    cfg, _ = fake_set(tmp_path)
    cfg["first_rows"] = {"dev": limit}
    with pytest.raises(store.FormatError, match="first_rows"):
        store.prepare(cfg)
    assert not (Path(cfg["out"]) / "dev").exists()


def test_real_data():
    """② The built data set on CFS: 500 000 rows per split; for 1000 random rows of each, n_primary, the waveforms and
    the classes equal what is read directly from the npz files. Slow (decompresses every file twice)."""
    if not all((REAL / s / "meta.json").exists() for s in store.SPLITS):
        pytest.fail(f"{REAL} is not built: on a compute node run python -c \"from data import store; "
                    f"print(store.prepare(store.load_config('dch_wf', 'pion')))\"")
    cfg = store.load_config("dch_wf", "pion")
    assert Path(cfg["out"]) == REAL
    for split in store.SPLITS:
        s = store.Split(REAL / split)
        assert len(s) == 500_000 and s.length == 3000 and s.meta["data_hash"] == store.data_hash(cfg)
        idx = np.sort(np.random.default_rng(0).choice(len(s), 1000, replace=False))
        got_wf, got_cls = s.take(idx), s.extra("classes")[idx]
        got_y = s.labels[idx, 0]
        del s
        start = 0
        for name in cfg["files"][split]:
            with np.load(Path(cfg["source"]) / name) as z:
                tt, tv, n = z["tag_times"], z["tag_values"], len(z["mom"])
                assert ((tt[tv > 0] >= 0) & (tt[tv > 0] < 3000)).all()             # so no time condition is needed
                m = (idx >= start) & (idx < start + n)
                r = idx[m] - start
                np.testing.assert_array_equal(got_y[m], (tv[r] == 1).sum(1))
                cls = np.zeros((len(r), 3000), np.int8)
                for j, (t, v) in enumerate(zip(tt[r], tv[r])):
                    cls[j, t[v == 1]] = 1
                    cls[j, t[v == 2]] = 2
                np.testing.assert_array_equal(got_cls[m], cls)
                np.testing.assert_array_equal(got_wf[m], z["wf_i"][r])
            start += n
        assert start == 500_000
