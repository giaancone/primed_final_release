"""Tiny fake data sets in the two storage forms of data/store.py, for tests that need a split but not a generator.

``npy``: the waveform is stored in the split (as dro_gen does). ``npz``: it stays in external npz files (as dch_wf).
Both hold the same 60 events of 3 labels, with some source rows filtered out, so a test can run on either.
"""
import numpy as np

from data import store

N, LENGTH, SIZES = 60, 32, (25, 20, 15)
LABELS = [{"name": "C", "unit": "photons"}, {"name": "S", "unit": "photons"}, {"name": "t0", "unit": "ns"}]
CFG = {"set": "fake", "config": "tiny", "out": None, "seed": 3}


def arrays(seed=3):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N, LENGTH)).astype(np.float32)
    Y = np.stack([rng.integers(1, 1500, N), rng.integers(1, 900, N), rng.uniform(-2.5, 2.5, N)], 1).astype(np.float32)
    rows = np.sort(rng.permutation(100)[:N]).astype(np.int64)     # 40 of 100 source rows were filtered out
    return W, Y, rows


def make_root(tmp_path):
    """An empty data root with the code of a set called ``fake`` (one file), for the data hash."""
    root = tmp_path / "data_root"
    (root / "fake" / "configs").mkdir(parents=True)
    (root / "fake" / "build.py").write_text("# fake\n")
    return root


def make_split(tmp_path, store_kind, split="dev", cfg=CFG, root=None, extras=True):
    """Write the fake split to tmp_path/<store_kind>/<split> and return (path, W, Y, rows). The heldout split holds
    other events (another seed, source rows 100 and up)."""
    W, Y, rows = arrays(3 if split == "dev" else 4)
    rows = rows + (0 if split == "dev" else 100)
    root = root or make_root(tmp_path)
    path = tmp_path / store_kind / split
    path.parent.mkdir(parents=True, exist_ok=True)
    wf = W
    if store_kind == "npz":
        parts, st, a = [], store.Stats(), 0
        for i, n in enumerate(SIZES):
            f = tmp_path / f"ext_{split}_{i}.npz"
            np.savez_compressed(f, wf_i=W[a:a + n], other=np.arange(3))
            parts.append(store.external_part(f, "wf_i", W[a:a + n]))
            st.add(W[a:a + n])
            a += n
        wf = {"parts": parts, "length": LENGTH, "stats": st.result()}
    ex = {"classes": ((W > 1).astype(np.int8), "1 where the sample is above 1")} if extras else None
    store.write_split(path, cfg, split, Y, rows, LABELS, dt_ns=0.32, t_first_ns=-400.0, unit="mV", waveform=wf,
                      extras=ex, root=root)
    return path, W, Y, rows
