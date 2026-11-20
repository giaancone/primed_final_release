"""The contract between a data set and every method (data/store.py, data/score.py), on both storage forms."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import fakes
from data import score, store

KINDS = ("npy", "npz")


@pytest.mark.parametrize("kind", KINDS)
def test_batch_by_position(tmp_path, kind):
    """Any order, repeats, and positions that cross the external files give the rows of the reference array."""
    path, W, Y, rows = fakes.make_split(tmp_path, kind)
    s = store.Split(path)
    idx = np.array([59, 0, 24, 25, 44, 45, 3, 3, 30])
    np.testing.assert_array_equal(s.take(idx), W[idx])
    np.testing.assert_array_equal(s.take(np.arange(fakes.N)), W)
    assert s.take(idx).dtype == np.float32 and s.take(np.array([], int)).shape == (0, fakes.LENGTH)
    for bad in (np.array([fakes.N]), np.array([1.9]), np.array([[1, 2]]), np.array([-1])):
        with pytest.raises(IndexError):
            s.take(bad)
    for a in (s.labels, s.rows, s.part(0), s.extra("classes")):   # what was checked once cannot be changed in place
        with pytest.raises(ValueError):
            a[0] = 0


@pytest.mark.parametrize("kind", KINDS)
def test_rows_labels_meta(tmp_path, kind):
    """Source rows are stable and increasing; label columns keep their order and units; meta.json has every key."""
    path, W, Y, rows = fakes.make_split(tmp_path, kind)
    s = store.Split(path).verify()
    np.testing.assert_array_equal(s.rows, rows)
    assert s.rows.dtype == np.int64 and (np.diff(s.rows) > 0).all() and s.rows.max() >= fakes.N   # not positions
    np.testing.assert_array_equal(s.labels, Y)
    assert s.labels.dtype == np.float32 and s.names == ["C", "S", "t0"]
    assert [x["unit"] for x in s.meta["labels"]] == ["photons", "photons", "ns"]
    assert s.meta["commit"] == "anonymous"
    assert set(s.meta) == store.META_KEYS and len(s) == fakes.N and s.length == fakes.LENGTH
    mask = os.umask(0); os.umask(mask)
    assert path.stat().st_mode & 0o777 == 0o777 & ~mask           # as any directory the user makes, not private
    assert (s.meta["split"], s.meta["dt_ns"], s.meta["t_first_ns"], s.meta["unit"]) == ("dev", 0.32, -400.0, "mV")
    st = s.meta["waveform"]["stats"]
    w = W.astype(np.float64)
    np.testing.assert_allclose([st["min"], st["max"], st["mean"], st["std"]], [w.min(), w.max(), w.mean(), w.std()], rtol=1e-12)
    np.testing.assert_allclose(s.meta["label_stats"]["t0"]["mean"], Y[:, 2].astype(np.float64).mean(), rtol=1e-12)
    np.testing.assert_array_equal(s.extra("classes"), (W > 1).astype(np.int8))
    with pytest.raises(KeyError):
        s.extra("peaks")
    assert len(s.content) == 64 and s.content == store.Split(path).content
    assert s.content != store.Split(fakes.make_split(tmp_path / "h", kind, "heldout")[0]).content
    reordered = {k: dict(reversed(list(v.items()))) if isinstance(v, dict) else v for k, v in s.meta.items()}
    (path / "meta.json").write_text(json.dumps(reordered, sort_keys=True))
    store.Split(path).verify()                                    # the order of keys in meta.json means nothing


def _damage_npy(f):
    a = np.load(f)
    a.flat[5] += 1
    np.save(f, a)


@pytest.mark.parametrize("kind", KINDS)
def test_changed_content_is_refused(tmp_path, kind):
    """A changed waveform, label, row or extra array is never returned."""
    path, W, Y, rows = fakes.make_split(tmp_path, kind)
    if kind == "npy":
        _damage_npy(path / "waveform.npy")
    else:
        np.savez_compressed(tmp_path / "ext_dev_1.npz", wf_i=W[25:45] + 1)
    s = store.Split(path)
    with pytest.raises(store.DigestError):
        s.take(np.array([30]))
    with pytest.raises(store.DigestError):
        store.Split(path).verify()
    for name in ("labels", "rows"):
        p2 = fakes.make_split(tmp_path / name, kind)[0]
        _damage_npy(p2 / f"{name}.npy")
        with pytest.raises(store.DigestError):
            store.Split(p2)
    p3 = fakes.make_split(tmp_path / "extra", kind)[0]
    _damage_npy(p3 / "classes.npy")
    with pytest.raises(store.DigestError):
        store.Split(p3).extra("classes")
    # the same bytes under another dtype or shape are other values: refused although the digest of the bytes agrees
    p4 = fakes.make_split(tmp_path / "dtype", kind)[0]
    np.save(p4 / "labels.npy", np.load(p4 / "labels.npy").view(np.int32))
    with pytest.raises(store.FormatError, match="int32"):
        store.Split(p4)
    p5 = fakes.make_split(tmp_path / "shape", kind)[0]
    np.save(p5 / "classes.npy", np.load(p5 / "classes.npy").reshape(fakes.LENGTH, fakes.N))
    with pytest.raises(store.FormatError):
        store.Split(p5).extra("classes")


def test_meta_and_shape_guards(tmp_path):
    path = fakes.make_split(tmp_path, "npy")[0]
    meta = json.loads((path / "meta.json").read_text())
    (path / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "data_hash"}))
    with pytest.raises(store.FormatError, match="data_hash"):
        store.Split(path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        store.Split(tmp_path / "nowhere")
    W, Y, rows = fakes.arrays()
    kw = dict(dt_ns=1, t_first_ns=0, unit="mV", root=fakes.make_root(tmp_path / "r"))
    with pytest.raises(store.StaleDataError):                      # never overwritten
        store.write_split(path, fakes.CFG, "dev", Y, rows, fakes.LABELS, waveform=W, **kw)
    for bad in (dict(labels=Y[:-1]), dict(rows=rows[::-1]), dict(waveform=W.astype(np.float64)), dict(split="test"),
                dict(rows=rows + 0.5), dict(rows=rows[:, None]), dict(extras={"../x": (W, "escapes the split")}),
                dict(labels=np.where(np.arange(fakes.N)[:, None] == 2, np.nan, Y))):
        a = {"labels": Y, "rows": rows, "waveform": W, "split": "dev", **bad}
        with pytest.raises(store.FormatError):
            store.write_split(tmp_path / "bad", fakes.CFG, a["split"], a["labels"], a["rows"], fakes.LABELS,
                              waveform=a["waveform"], extras=a.get("extras"), **kw)
        assert not (tmp_path / "bad").exists() and not list(tmp_path.glob(".bad*"))    # nothing half-written


def test_each_external_file_is_read_once(tmp_path, monkeypatch):
    path, W, Y, rows = fakes.make_split(tmp_path, "npz")
    opened, real = [], np.load
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), real(f, *a, **k))[1])
    s = store.Split(path)
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(fakes.N)[:16]
        np.testing.assert_array_equal(s.take(idx), W[idx])
    ext = [f for f in opened if f.endswith(".npz")]
    assert sorted(ext) == sorted(set(ext)) and len(ext) == len(fakes.SIZES)


def test_p68():
    """numpy's rule on finite errors; invalid errors count as +inf; never NaN."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 26, 1000, 20001):
        e = rng.exponential(1, n)
        np.testing.assert_allclose(score.p68(e), np.percentile(e, 68), rtol=1e-12)
    e = np.arange(100.0)                                   # position 0.68 * 99 = 67.32, between e[67] and e[68]
    assert score.p68(e) == pytest.approx(67.32)
    few = e.copy(); few[[3, 50]] = [np.nan, np.inf]        # 2 invalid: e[67] and e[68] of the sorted errors stay finite
    assert score.p68(few) == pytest.approx(np.percentile(np.delete(e, [3, 50]), 68 * 99 / 97))
    edge = e.copy(); edge[68:] = np.inf                    # e[67] finite, e[68] invalid: no finite bound
    assert score.p68(edge) == np.inf
    many = e.copy(); many[40:] = -np.inf
    assert score.p68(many) == np.inf and score.p68(np.full(7, np.nan)) == np.inf
    for n in (26, 51, 76, 101, 20001):                     # 68 (n - 1) / 100 is a whole number k: exactly e[k], the
        k = 68 * (n - 1) // 100                            # last finite value (0.68 * 75 is 51.00000000000001 in floats)
        exact = np.arange(float(n)); exact[k + 1:] = np.inf
        assert score.p68(exact) == k
    assert score.p68([0, 1, 2, np.inf]) == np.inf          # small n: position 2.04 already needs the invalid one
    assert score.p68([5.0]) == 5.0 and score.p68([3.0, 3.0, 3.0, 3.0]) == 3.0
    with pytest.raises(ValueError):
        score.p68([])


def test_stats_are_stable():
    """A small spread on a large offset is not lost to cancellation (float32 values of 1e8 and one of 1e8 + 8)."""
    a = np.full(60, 1e8, np.float32); a[-1] += 8
    st = store.Stats().add(a[:20]).add(a[20:]).result()
    np.testing.assert_allclose([st["mean"], st["std"]], [a.astype(np.float64).mean(), a.astype(np.float64).std()], rtol=1e-9)
    assert (st["min"], st["max"]) == (1e8, 1e8 + 8)


def test_config_and_data_hash(tmp_path):
    """The hash follows the expanded config and the code of the set, and ignores where the data is written."""
    root = fakes.make_root(tmp_path)
    c = root / "fake" / "configs"
    (c / "_base.yaml").write_text("out: /a\nnoise: {dark: 1.0, xtalk: 0.07}\nn: 10\n")
    (c / "one.yaml").write_text("defaults: _base.yaml\nnoise: {dark: 2.0}\n")
    cfg = store.load_config("fake", "one", root)
    assert cfg == {"out": "/a", "noise": {"dark": 2.0, "xtalk": 0.07}, "n": 10, "set": "fake", "config": "one"}
    with pytest.raises(KeyError, match="one"):
        store.load_config("fake", "_base", root)
    with pytest.raises(KeyError):
        store.load_config("fake", "two", root)
    h = store.data_hash(cfg, root)
    assert store.data_hash({**cfg, "out": "/b"}, root) == h
    assert store.data_hash({**cfg, "n": 11}, root) != h
    (root / "fake" / "README.md").write_text("words"); (root / "fake" / "reference").mkdir()
    (root / "fake" / "reference" / "one.json").write_text("{}")
    assert store.data_hash(cfg, root) == h                                   # not code
    (root / "fake" / "evaluate.py").write_text("x = 1\n")
    h2 = store.data_hash(cfg, root)
    (root / "score.py").write_text("y = 1\n")
    assert len({h, h2, store.data_hash(cfg, root)}) == 3                      # set code and shared code both count


def test_prepare(tmp_path):
    """Builds once; a second call verifies and touches nothing; another config or damaged data is refused."""
    root = fakes.make_root(tmp_path)
    cfg = {**fakes.CFG, "out": str(tmp_path / "npy")}
    calls = []

    def build(c):
        calls.append(1)
        for s in store.SPLITS:
            fakes.make_split(tmp_path, "npy", s, cfg=c, root=root)

    assert store.prepare(cfg, build, root) == "built"
    stamp = {p: p.stat().st_mtime_ns for p in (tmp_path / "npy").rglob("*") if p.is_file()}
    assert store.prepare(cfg, build, root) == "exists" and len(calls) == 1
    assert stamp == {p: p.stat().st_mtime_ns for p in (tmp_path / "npy").rglob("*") if p.is_file()}
    assert store.Split(tmp_path / "npy" / "heldout").meta["split"] == "heldout"
    with pytest.raises(store.StaleDataError, match="another config"):
        store.prepare({**cfg, "seed": 4}, build, root)
    _damage_npy(tmp_path / "npy" / "dev" / "waveform.npy")
    with pytest.raises(store.DigestError):
        store.prepare(cfg, build, root)
    import shutil
    shutil.rmtree(tmp_path / "npy" / "dev")
    with pytest.raises(store.StaleDataError, match="incomplete"):
        store.prepare(cfg, build, root)
    assert len(calls) == 1


def test_import_from_anywhere(tmp_path):
    """``data`` is an installed top-level package, not a path relative to the working directory."""
    r = subprocess.run([sys.executable, "-c", "import data.store, data.score; print(data.store.ROOT)"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(store.ROOT)
