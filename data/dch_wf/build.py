"""Build the dch_wf splits from the source npz files. The waveforms are not copied: they stay in the npz files.

Each npz file holds ``wf_i (n, L)`` and, per event, up to T tags ``tag_times (n, T)`` (sample index of a peak) and
``tag_values (n, T)`` (1 primary, 2 secondary ionisation, 0 an unused slot). Per split, in file order, every event:
  * label ``n_primary`` = the number of tags with value 1 and time in [0, L). The time condition is the definition of
    the old repository (cache._write_dch_npz) and of the paper's code (dch_ssl.read_dch); on the pion files it removes
    nothing (every used tag lies in [1, L - 2]).
  * extra ``classes (n, L) int8``: 0 noise, 1 primary, 2 secondary at the tagged samples, same time condition; where
    one sample is tagged both, 2 wins (old cache.dch_peak_classes). The auxiliary target of the 2-layer method.
  * the waveform part ``data.store.external_part(file, member, float32 array)``.
Nothing is filtered: rows are 0..N-1 in file order. A member of an npz cannot be memory-mapped (reading it
decompresses all of it), so each file is opened once and only one file's waveforms are in memory at a time.

Optional data-config key ``first_rows: {dev: N, heldout: M}`` (absent: every row): a split is only its first N events
in file order (rows 0..N-1), for small data sets such as the smoke set ``pion_smoke``. Its waveforms are then a part
of a member, not a whole member, so they cannot stay external: they are written into the split as ``waveform.npy``
(the form dro_gen uses). Files after the one that completes the N rows are not read; fewer than N rows in the listed
files is an error.
"""
from pathlib import Path

import numpy as np

from data import store

LABELS = [{"name": "n_primary", "unit": "clusters"}]
CLASSES = "per-sample class: 0 noise, 1 primary, 2 secondary ionisation peak (2 where a sample is tagged both)"


def n_primary(tag_times, tag_values, length):
    """(n,) float32: tags with value 1 and time in [0, length)."""
    return ((tag_values == 1) & (tag_times >= 0) & (tag_times < length)).sum(1).astype(np.float32)


def peak_classes(tag_times, tag_values, length):
    """(n, length) int8: 0 noise, 1 primary, 2 secondary at the tagged samples (a later 2 overwrites a 1)."""
    out = np.zeros((len(tag_times), length), dtype=np.int8)
    for cls in (1, 2):
        rows, cols = np.nonzero((tag_values == cls) & (tag_times >= 0) & (tag_times < length))
        out[rows, tag_times[rows, cols]] = cls
    return out


def read_file(path, members):
    """(waveform float32 (n, L), tag_times, tag_values) of one npz file, each member decompressed once."""
    with np.load(path) as z:
        wf = np.ascontiguousarray(z[members["waveform"]], np.float32)
        tt, tv = z[members["tag_times"]], z[members["tag_values"]]
    if wf.ndim != 2 or tt.ndim != 2 or tt.shape != tv.shape or len(tt) != len(wf):
        raise store.FormatError(f"{path}: waveform {wf.shape} and tags {tt.shape}/{tv.shape} do not describe the "
                                f"same events; check the member names in the data config")
    if not np.isfinite(wf).all():
        raise store.FormatError(f"{path}: non-finite waveform samples; the source file is damaged, fetch it again")
    return wf, tt, tv


def row_limit(cfg, split):
    """N of ``first_rows`` for this split, or None (every row)."""
    limit = (cfg.get("first_rows") or {}).get(split)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise store.FormatError(f"first_rows.{split} = {limit!r}: a positive number of rows, or leave the key out")
    return limit


def build_split(cfg, split):
    files, limit = [Path(cfg["source"]) / f for f in cfg["files"][split]], row_limit(cfg, split)
    parts, labels, classes, stats, length, n = [], [], [], store.Stats(), None, 0
    for f in files:
        if limit is not None and n == limit:
            break
        wf, tt, tv = read_file(f, cfg["members"])
        if limit is not None:
            wf, tt, tv = wf[:limit - n].copy(), tt[:limit - n], tv[:limit - n]     # a copy: the file's array is freed
        length, n = length or wf.shape[1], n + len(wf)
        if wf.shape[1] != length:
            raise store.FormatError(f"{f}: {wf.shape[1]} samples per waveform, the files before it have {length}")
        parts.append(wf if limit is not None else store.external_part(f, cfg["members"]["waveform"], wf))
        stats.add(wf)
        labels.append(n_primary(tt, tv, length))
        classes.append(peak_classes(tt, tv, length))
        del wf, tt, tv
    if limit is not None and n < limit:
        raise store.FormatError(f"first_rows.{split} = {limit}, but the files of {split} hold only {n} events")
    labels, classes = np.concatenate(labels)[:, None], np.concatenate(classes)
    waveform = (np.concatenate(parts) if limit is not None
                else {"parts": parts, "length": length, "stats": stats.result()})
    store.write_split(Path(cfg["out"]) / split, cfg, split, labels, np.arange(len(labels)), LABELS,
                      dt_ns=cfg["dt_ns"], t_first_ns=cfg["t_first_ns"], unit=cfg["unit"], waveform=waveform,
                      extras={"classes": (classes, CLASSES)})


def build(cfg):
    for split in store.SPLITS:
        build_split(cfg, split)
