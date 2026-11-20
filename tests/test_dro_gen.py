"""data/dro_gen: generator reproducibility, split-then-filter, the ROOT source, the score, and the six configs.

CPU only, small n."""
import copy

import numpy as np
import pytest
import uproot

from data import store
from data.dro_gen import build, evaluate, generate

CONFIGS = [f"{c}_{v}" for c in ("pwo", "bso", "bgo") for v in ("ideal", "realistic")]
# Digests of rows [0, 500) (chunk 0, default_rng([20261118, 0])) of the paper file PWO_real3_100k, measured on that
# file with data.store.digest.
PAPER_CHUNK0 = {"waveform": "03731e7ee733638b6beee833c458efc3102267376dca1d68338f960a07ad5579",
                "labels": "0660a7d0b99cf0fa5d256b66f106262e85255f4623676a62337a3a776bd62050"}


def config(name="pwo_realistic", out=None, n=200, chunk=None, **sections):
    """A data config with run.n = n, written to ``out``, without expect; ``sections`` replace whole sections."""
    cfg = store.load_config("dro_gen", name)
    cfg.pop("expect", None)
    cfg["run"] = {**cfg["run"], "n": n, **({"chunk": chunk} if chunk else {})}
    cfg["out"] = str(out)
    for k, v in sections.items():
        cfg[k] = v
    return cfg


# ---- 1. reproducibility ------------------------------------------------------------------------------------------

def test_workers_do_not_change_the_rows():
    """Real spawned worker processes: 3 workers give the arrays of 1 worker, bit for bit."""
    cfg = config(n=150, chunk=25)
    W1, Y1 = generate.generate(cfg, workers=1)
    W3, Y3 = generate.generate(cfg, workers=3)
    assert W1.shape == (150, 6246) and W1.dtype == np.float32 and Y1.dtype == np.float32
    np.testing.assert_array_equal(W1, W3)
    np.testing.assert_array_equal(Y1, Y3)


def test_first_chunk_is_the_paper_file():
    W, Y = generate.generate(config("pwo_realistic", n=500))
    assert store.digest(W) == PAPER_CHUNK0["waveform"] and store.digest(Y) == PAPER_CHUNK0["labels"]


def test_expect_mismatch_writes_nothing(tmp_path):
    cfg = config(out=tmp_path / "x", n=20, expect={"waveform_sha256": "0" * 64, "labels_sha256": "0" * 64})
    with pytest.raises(build.ReproductionError):
        build.build(cfg)
    assert not (tmp_path / "x").exists()


BAD_NOISE = {"dark_rate_hz": 0.0, "spe_sigma": -0.1, "crosstalk": 0.0, "electronic_spe": 0.0}


@pytest.mark.parametrize("change", [{"crystal": "lyso"}, {"noise": BAD_NOISE}, {"run": {"n": 0, "seed": 1, "chunk": 5}}])
def test_generator_config_guards(tmp_path, change):
    with pytest.raises(generate.GeneratorConfigError):
        generate.generate(config(**change))


# ---- 2. split by original row, then filter -----------------------------------------------------------------------

def test_split_rows_by_hand():
    C = [1, 0, 3, 4, 0, 6, 7, 8, 0, 1]
    S = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    sp = build.split_rows(np.array([C, S, [0.0] * 10], np.float32).T, 0.2)      # dev: rows 0..7, heldout: 8, 9
    np.testing.assert_array_equal(sp["dev"], [0, 3, 5, 6, 7])
    np.testing.assert_array_equal(sp["heldout"], [9])
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            build.split_rows(np.ones((10, 3)), bad)


def test_split_then_filter_on_a_built_set(tmp_path):
    """Tiny photon ranges: about 3 rows in 4 have C == 0 or S == 0. Kept and dropped rows equal a count by hand."""
    cfg = config(out=tmp_path / "set", n=200, chunk=50, source={"kind": "random", "c_max": 1, "s_max": 1})
    W, Y = generate.generate(cfg)
    want = {"dev": [], "heldout": []}
    for i in range(200):
        if Y[i, 0] > 0 and Y[i, 1] > 0:
            want["dev" if i < 160 else "heldout"].append(i)
    assert 0 < len(want["dev"]) < 160 and 0 < len(want["heldout"]) < 40      # rows are dropped in both splits
    build.build(cfg)
    dev, held = store.Split(tmp_path / "set" / "dev").verify(), store.Split(tmp_path / "set" / "heldout").verify()
    np.testing.assert_array_equal(dev.rows, want["dev"])
    np.testing.assert_array_equal(held.rows, want["heldout"])
    assert 160 - len(dev) == 160 - len(want["dev"]) and 40 - len(held) == 40 - len(want["heldout"])
    assert not np.intersect1d(dev.rows, held.rows).size and dev.rows.max() < held.rows.min()
    for s in (dev, held):
        np.testing.assert_array_equal(s.labels, Y[s.rows])
        np.testing.assert_array_equal(s.take(np.arange(len(s))), W[s.rows])


# ---- 3. the ROOT source --------------------------------------------------------------------------------------------

def write_root(path, c, s, t0=None):
    branches = {"nc": np.asarray(c, np.int32), "es": np.asarray(s, np.float32)}
    if t0 is not None:
        branches["t"] = np.asarray(t0, np.float64)
    with uproot.recreate(path) as f:
        f["events"] = branches
    return str(path)


def root_source(file, t0_branch=None, s_factor=1.0):
    return {"kind": "root", "file": file, "tree": "events", "c_branch": "nc", "c_factor": 1.0, "s_branch": "es",
            "s_factor": s_factor, "t0_branch": t0_branch}


def test_root_labels_are_the_input(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    c, s, t0 = rng.integers(1, 800, n), rng.integers(1, 500, n).astype(np.float32), rng.uniform(-2, 2, n)
    c[[3, 35]] = 0                                                       # dropped, one in each split
    f = write_root(tmp_path / "fake.root", c, s, t0)
    cfg = config(out=tmp_path / "set", n=n, chunk=20, source=root_source(f, "t"))
    build.build(cfg)
    rows = []
    for split in ("dev", "heldout"):
        sp = store.Split(tmp_path / "set" / split).verify()
        np.testing.assert_array_equal(sp.labels[:, 0], c[sp.rows])
        np.testing.assert_array_equal(sp.labels[:, 1], s[sp.rows])
        np.testing.assert_array_equal(sp.labels[:, 2], t0[sp.rows].astype(np.float32))
        rows += list(sp.rows)
    assert rows == [i for i in range(n) if i not in (3, 35)]
    # without a t0 branch t0 is drawn; scaled branches are rounded to whole photons
    given = build.source_rows(config(n=n, source=root_source(f, s_factor=0.25)))
    np.testing.assert_array_equal(given, np.stack([c, np.rint(s * 0.25)], 1))
    W, Y = generate.generate(config(n=n, chunk=20), given)
    np.testing.assert_array_equal(Y[:, :2], given)
    assert np.abs(Y[:, 2]).max() <= 2.5 and len(np.unique(Y[:, 2])) == n


def test_root_guards(tmp_path):
    f = write_root(tmp_path / "short.root", [1, 2, 3], [1, 2, 3])
    with pytest.raises(build.SourceError):
        build.source_rows(config(n=4, source=root_source(f)))
    g = write_root(tmp_path / "neg.root", [1, -2, 3], [1, 2, 3])
    with pytest.raises(build.SourceError):
        build.source_rows(config(n=3, source=root_source(g)))
    with pytest.raises(build.SourceError):
        build.source_rows(config(n=3, source={"kind": "h5"}))


# ---- 4. the score ------------------------------------------------------------------------------------------------

def test_evaluate_by_hand():
    truth = np.array([[100, 100, 0], [200, 100, 0], [300, 100, 0], [400, 100, 0], [500, 100, 0]], float)
    pred = np.array([[110, 100, 0], [240, 100, 1], [390, 100, -2], [560, 100, 3], [750, 100, 4]], float)
    # ratio errors 0.1 .. 0.5 and t0 errors 0 .. 4: p68 at position 0.68 * 4 = 2.72 -> 0.372 and 2.72
    r = evaluate.evaluate(pred, truth)
    assert r["err68_ratio_pct"] == pytest.approx(37.2, rel=1e-12) and r["err68_t0_ns"] == pytest.approx(2.72, rel=1e-12)
    assert (r["n"], r["n_invalid"]) == (5, 0)


@pytest.mark.parametrize("n_bad, finite", [(1, True), (4, False), (10, False)])
def test_evaluate_invalid_predictions(n_bad, finite):
    """Few, many (> 32 %) and all invalid: finite, inf, inf; never NaN. Each kind of invalid value is used."""
    truth = np.column_stack([np.arange(1, 11) * 10.0, np.full(10, 50.0), np.zeros(10)])
    pred = truth * 1.05
    bad = [[1, 0, np.nan], [1, -3, np.inf], [np.nan, 5, -np.inf], [np.inf, 5, np.nan]]
    for i in range(n_bad):
        pred[i] = bad[i % 4]
    r = evaluate.evaluate(pred, truth)
    assert not np.isnan([r["err68_ratio_pct"], r["err68_t0_ns"]]).any() and r["n_invalid"] == n_bad
    assert np.isfinite(r["err68_ratio_pct"]) == finite and np.isfinite(r["err68_t0_ns"]) == finite
    if finite:
        assert r["err68_ratio_pct"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("truth", [[[1, 0, 0], [2, 2, 0]], [[-1, 2, 0], [2, 2, 0]], [[1, 2, np.nan], [2, 2, 0]],
                                   [[1, 2], [2, 2]]])
def test_evaluate_truth_guards(truth):
    truth = np.array(truth, float)
    with pytest.raises(evaluate.EvaluationError):
        evaluate.evaluate(np.ones_like(truth), truth)
    with pytest.raises(evaluate.EvaluationError):
        evaluate.evaluate(np.ones((3, 3)), np.ones((2, 3)))


# ---- 5. the six configs --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONFIGS)
def test_config_builds_and_keeps_the_contract(tmp_path, name):
    cfg = config(name, out=tmp_path / name, n=200)
    assert store.prepare(cfg) == "built"
    for split in store.SPLITS:
        s = store.Split(tmp_path / name / split).verify()
        assert set(s.meta) == store.META_KEYS
        assert (s.meta["set"], s.meta["config"], s.meta["split"]) == ("dro_gen", name, split)
        assert s.names == ["C", "S", "t0"] and [x["unit"] for x in s.meta["labels"]] == ["photons", "photons", "ns"]
        assert (s.length, s.meta["dt_ns"], s.meta["t_first_ns"], s.meta["unit"]) == (6246, 0.32, -400.0, "mV")
        assert s.labels.dtype == np.float32 and s.take(np.arange(3)).dtype == np.float32
        assert (np.diff(s.rows) > 0).all() and (s.labels[:, :2] > 0).all()
        assert (s.rows < 160).all() if split == "dev" else (s.rows >= 160).all()
        assert s.meta["data_hash"] == store.data_hash(cfg)
    assert store.prepare(cfg) == "exists"
    changed = copy.deepcopy(cfg)
    changed["cherenkov"]["decay_ns"] = 0.2
    with pytest.raises(store.StaleDataError):
        store.prepare(changed)


def test_ideal_differs_from_realistic_only_in_the_sensor_effects():
    for c in ("pwo", "bso", "bgo"):
        ideal, real = store.load_config("dro_gen", f"{c}_ideal"), store.load_config("dro_gen", f"{c}_realistic")
        diff = {k for k in ideal.keys() | real.keys() if ideal.get(k) != real.get(k)}
        assert diff == {"noise", "out", "config", "expect"} if c == "pwo" else diff == {"noise", "out", "config"}
        assert set(ideal["noise"].values()) == {0.0}


def test_more_generator_and_source_guards(tmp_path):
    """crystal constants, the sampling window, the random source."""
    c = config()
    c["crystals"][c["crystal"]] = {**c["crystals"][c["crystal"]], "t1_ns": -1.0}
    with pytest.raises(generate.GeneratorConfigError, match="t1 and a1 must be positive"):
        generate.check(c)
    c = config()
    c["window"] = {**c["window"], "t_last_ns": c["window"]["t_first_ns"]}
    with pytest.raises(generate.GeneratorConfigError, match="t_last_ns after t_first_ns"):
        generate.check(c)
    with pytest.raises(build.SourceError, match="non-negative"):
        build.source_rows(config(source={"kind": "random", "c_max": -1.0, "s_max": 10.0}))
