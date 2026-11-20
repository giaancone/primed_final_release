"""Fixtures: tiny fake data sets of both tasks, built on disk through the real format, and a run config for each.

    fake_dro   task dro, data set dro_gen/fake: 400 dev + 100 heldout rows, waveforms of 256 samples (stored in the
               split, waveform.npy), labels C, S (photons) and t0 (ns) that are learnable from the trace
    fake_dch   task dch, data set dch_wf/fake: 400 dev + 100 heldout rows, waveforms of 250 samples (not a multiple
               of 32) in external npz files, label n_primary (clusters) and the extra ``classes`` (per sample)

Each returns the full config of a run (configs/teacher.yaml with ``task``, ``data.set``, ``data.config`` and
``run`` set) and, for the test, points
    wavefm.data.ROOT  at <tmp_path>/data_root (holding <set>/configs/fake.yaml and a stand-in build.py), and
    wavefm.paths.OUTPUT at <tmp_path>/out.
So ``wavefm.data.dev(cfg)`` opens the fake dev split, its data hash checked, and ``data.store.evaluate(set)`` is the
real metric of the set. The data directory is ``<tmp_path>/<set>`` (``wavefm.data.dev(cfg).path.parent``).
Both fixtures can be used in the same test. The arrays are deterministic (seed 1).
"""
import numpy as np
import pytest

import wavefm.data
from data import store
from wavefm import paths
from wavefm.utils import config

N_DEV, N_HELDOUT = 400, 100


def _dro_build(cfg):
    n, L, dt = N_DEV + N_HELDOUT, cfg["length"], cfg["dt_ns"]
    rng = np.random.default_rng(cfg["seed"])
    C, S, t0 = rng.uniform(1, 1000, n), rng.uniform(1, 600, n), rng.uniform(-2.5, 2.5, n)
    t = np.arange(L)[None] - 60 - t0[:, None] / dt                      # samples after the onset
    on = t >= 0
    W = (C[:, None] * np.exp(-np.clip(t, 0, None) / 3.0) + 0.2 * S[:, None] * np.exp(-np.clip(t, 0, None) / 40.0)) * on
    W = (W + rng.normal(0, 1, (n, L))).astype(np.float32)
    Y = np.stack([C, S, t0], 1)
    labels = [{"name": "C", "unit": "photons"}, {"name": "S", "unit": "photons"}, {"name": "t0", "unit": "ns"}]
    for split, r in (("dev", np.arange(N_DEV)), ("heldout", np.arange(N_DEV, n))):
        store.write_split(f"{cfg['out']}/{split}", cfg, split, Y[r], r, labels, dt_ns=dt, t_first_ns=0.0, unit="mV",
                          waveform=W[r], root=wavefm.data.ROOT)


def _dch_build(cfg):
    n, L = N_DEV + N_HELDOUT, cfg["length"]
    rng = np.random.default_rng(cfg["seed"])
    u = rng.uniform(size=(n, L))
    cls = np.where(u < 0.04, 1, np.where(u < 0.08, 2, 0)).astype(np.int8)
    W = (3.0 * (cls == 1) + 1.5 * (cls == 2) + rng.normal(0, 0.3, (n, L))).astype(np.float32)
    Y = (cls == 1).sum(1, keepdims=True).astype(np.float32)
    for split, r in (("dev", np.arange(N_DEV)), ("heldout", np.arange(N_DEV, n))):
        parts, st = [], store.Stats()
        for i, part in enumerate(np.array_split(r, 2)):
            f = f"{cfg['out']}_{split}_{i}.npz"
            np.savez_compressed(f, wf_i=W[part])
            parts.append(store.external_part(f, "wf_i", W[part])); st.add(W[part])
        store.write_split(f"{cfg['out']}/{split}", cfg, split, Y[r], r, [{"name": "n_primary", "unit": "clusters"}],
                          dt_ns=0.5, t_first_ns=0.0, unit="a.u.", waveform={"parts": parts, "length": L,
                                                                             "stats": st.result()},
                          extras={"classes": (cls[r], "class of every sample: 0 noise, 1 primary, 2 secondary")},
                          root=wavefm.data.ROOT)


def _fake(tmp_path, monkeypatch, task, set_name, data_cfg, build):
    root = tmp_path / "data_root"
    (root / set_name / "configs").mkdir(parents=True, exist_ok=True)
    (root / set_name / "build.py").write_text("# stand-in: tests/conftest.py builds this set\n")
    lines = [f"out: {tmp_path / set_name}"] + [f"{k}: {v}" for k, v in data_cfg.items()]
    (root / set_name / "configs" / "fake.yaml").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(wavefm.data, "ROOT", root)
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "out")
    store.prepare(store.load_config(set_name, "fake", root), build, root)
    cfg = config.load(paths.ROOT / "configs" / "teacher.yaml")
    return config.merge(cfg, {"task": task, "data": {"set": set_name, "config": "fake"},
                              "run": {"group": "test", "name": task}})


@pytest.fixture
def fake_dro(tmp_path, monkeypatch):
    return _fake(tmp_path, monkeypatch, "dro", "dro_gen", {"seed": 1, "length": 256, "dt_ns": 0.32}, _dro_build)


@pytest.fixture
def fake_dch(tmp_path, monkeypatch):
    return _fake(tmp_path, monkeypatch, "dch", "dch_wf", {"seed": 1, "length": 250}, _dch_build)
