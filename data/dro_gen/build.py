"""Build a dro_gen data set: generate every row of the source, check it, split, filter, write (data/store.py).

Conventions, and why
  * All run.n rows are generated in source order first. If the config holds ``expect`` (digests of the UNFILTERED
    waveform and label arrays, data/store.py digest), they are compared before anything is written: a data set that
    claims to be the paper's is never written from a changed generator.
  * The split is by original row number, set by ``split.heldout_frac``: rows < round((1 - heldout_frac) * n) are
    ``dev``, the rest ``heldout``. Only then are rows with C == 0 or S == 0 dropped inside each split (the ratio C/S
    is undefined for them). Filtering never changes the random stream or moves a row to the other split.
  * ``rows`` are the original row numbers; labels are C, S in photons and t0 in ns.
  * Sources (``source.kind``): ``random`` (uniform C, S and t0, the paper's) and ``root`` (C and S per entry of a
    ROOT tree, times a factor, rounded to whole photons; t0 from an optional branch). The ROOT path is a technical
    demonstration, see README.md.

Parallel generation spawns workers: call build(cfg, workers > 1) only under ``if __name__ == "__main__":``.
"""
from pathlib import Path

import numpy as np
import uproot

from data import store
from data.dro_gen import generate

LABELS = [{"name": "C", "unit": "photons"}, {"name": "S", "unit": "photons"}, {"name": "t0", "unit": "ns"}]
SOURCES = ("random", "root")


class SourceError(ValueError):
    pass


class ReproductionError(RuntimeError):
    pass


def read_root(src: dict, n: int) -> np.ndarray:
    """The first n entries of the ROOT tree as an (n, 2|3) float64 array of C, S [, t0]: C and S are the branches
    times their factors, rounded to whole photons; t0 (ns) is read as is when ``t0_branch`` is set."""
    with uproot.open(src["file"]) as f:
        tree = f[src["tree"]]
        if tree.num_entries < n:
            raise SourceError(f"{src['file']}: tree {src['tree']} has {tree.num_entries} entries, run.n is {n}: "
                              f"lower run.n or use a larger file")
        col = lambda b: np.asarray(tree[b].array(library="np", entry_stop=n), np.float64)
        out = [np.rint(col(src["c_branch"]) * src["c_factor"]), np.rint(col(src["s_branch"]) * src["s_factor"])]
        if src.get("t0_branch"):
            out.append(col(src["t0_branch"]))
    out = np.stack(out, 1)
    if not np.isfinite(out).all() or (out[:, :2] < 0).any():
        raise SourceError(f"{src['file']}: C and S must be finite and non-negative after scaling, t0 finite; check "
                          f"the branches and factors in source")
    return out


def source_rows(cfg: dict):
    """None for the random source (drawn inside the generator), else the given C, S [, t0] of every row."""
    src = cfg["source"]
    if src.get("kind") not in SOURCES:
        raise SourceError(f"source.kind {src.get('kind')!r} is not one of {SOURCES}")
    if src["kind"] == "random":
        if src["c_max"] < 0 or src["s_max"] < 0:
            raise SourceError("source.c_max and source.s_max must be non-negative")
        return None
    return read_root(src, cfg["run"]["n"])


def split_rows(labels, heldout_frac: float) -> dict:
    """Original row numbers of dev and heldout: split by row number first, then drop C == 0 or S == 0 in each."""
    n = len(labels)
    if not 0 < heldout_frac < 1:
        raise ValueError(f"split.heldout_frac must be in (0, 1), got {heldout_frac}")
    n_dev = round((1 - heldout_frac) * n)
    valid = (labels[:, 0] > 0) & (labels[:, 1] > 0)
    rows = np.arange(n)
    return {"dev": rows[:n_dev][valid[:n_dev]], "heldout": rows[n_dev:][valid[n_dev:]]}


def build(cfg: dict, workers: int = 1) -> Path:
    """Write <out>/dev and <out>/heldout of the data config ``cfg`` (expanded, see data.store.load_config)."""
    W, Y = generate.generate(cfg, source_rows(cfg), workers)
    want = cfg.get("expect")
    if want:
        got = {"waveform_sha256": store.digest(W), "labels_sha256": store.digest(Y)}
        if got != {k: want[k] for k in got}:
            raise ReproductionError(f"{cfg['config']}: the generated arrays differ from expect: {got} vs {want}. The "
                                    f"generator, its parameters or numpy changed: find the change, do not edit expect")
    out = Path(cfg["out"])
    for split, rows in split_rows(Y, cfg["split"]["heldout_frac"]).items():
        store.write_split(out / split, cfg, split, Y[rows], rows, LABELS, dt_ns=cfg["window"]["dt_ns"],
                          t_first_ns=cfg["window"]["t_first_ns"], unit=cfg["sipm"]["unit"], waveform=W[rows])
    return out
