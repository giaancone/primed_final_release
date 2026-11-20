"""The on-disk form of a data set, the same for every data set and read by every method.

A data config ``data/<set>/configs/<name>.yaml`` is one concrete problem. Building it writes two splits:

    <out>/dev/       a method may use it in any way
    <out>/heldout/   predicted once and scored by ``data.<set>.evaluate``; a method never opens it

A split directory holds
    labels.npy    (N, k) float32; the columns and their units are listed in meta["labels"]
    rows.npy      (N,) int64, the row of every event in the unfiltered source: the stable name of an event
    meta.json     the keys of META_KEYS
    the waveform  (N, L) float32, in parts listed in order in meta["waveform"]["parts"]: ``waveform.npy`` in the
                  directory, or members of external npz files that are too large to copy
    <extra>.npy   further per-event arrays named in meta["extras"] (DCH: the class of every sample)

Conventions
  * An event is addressed by its position 0..N-1 in the split; ``rows`` maps positions to source rows.
  * A digest is the sha256 of the C-order bytes of an array. Every array is checked the first time it is read
    (bytes, dtype and shape), and a part is read from disk once and is read-only afterwards. A mismatch raises
    DigestError or FormatError: wrong data is never returned. ``Split.content`` is one digest of all of them.
  * The data hash names the problem: the expanded config without ``out``, and the content of every code file of the
    data set (``data/*.py`` and ``data/<set>/`` without configs, reference and README). Two results are comparable
    only if their data hashes are equal. The commit field is anonymised in this distribution.
  * A split appears under its name only when complete (written to a temporary directory, then renamed).
  * ``prepare`` builds and verifies both splits, heldout included: it is run by a person (scripts/prepare_data.py).
    A method opens ``dev`` through ``wavefm.data`` and nothing else.
  * This package needs numpy and yaml only, and nothing of ``wavefm``: the problem does not depend on a solution.
"""
import copy
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SPLITS = ("dev", "heldout")
META_KEYS = {"set", "config", "split", "data_hash", "commit", "n", "length", "dt_ns", "t_first_ns", "unit",
             "labels", "label_stats", "waveform", "extras", "sha256"}
CHUNK = 4096    # rows per read when an array is digested or summarised
NOT_CODE = {"configs", "reference", "README.md", "__pycache__", ".ipynb_checkpoints"}


class DigestError(RuntimeError):
    pass


class FormatError(RuntimeError):
    pass


class StaleDataError(RuntimeError):
    pass


# ---- config and data hash -------------------------------------------------------------------------------------------

def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("defaults", None)
    return _merge(_load_yaml(path.parent / parent), cfg) if parent else cfg


def load_config(set_name: str, name: str, root=ROOT) -> dict:
    """The expanded data config ``<root>/<set>/configs/<name>.yaml`` (``defaults: other.yaml`` is loaded first).
    Files starting with ``_`` hold shared defaults and are not data sets."""
    path = Path(root) / set_name / "configs" / f"{name}.yaml"
    if name.startswith("_") or not path.exists():
        known = sorted(p.stem for p in path.parent.glob("[!_]*.yaml"))
        raise KeyError(f"no data config {set_name}/{name}; known configs of {set_name}: {known}")
    cfg = _load_yaml(path)
    if "out" not in cfg:
        raise FormatError(f"{path}: a data config must say where it is written (out: <directory>)")
    return {**cfg, "set": set_name, "config": name}


def code_files(set_name: str, root=ROOT) -> list:
    """The files whose content defines the data set: shared modules and everything of the set but NOT_CODE."""
    root = Path(root)
    own = [p for p in (root / set_name).rglob("*") if p.is_file() and not NOT_CODE & set(p.relative_to(root).parts)]
    return sorted(set(root.glob("*.py")) | set(own))


def data_hash(cfg: dict, root=ROOT) -> str:
    h = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != "out"}, sort_keys=True).encode())
    for p in code_files(cfg["set"], root):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


# ---- digests and statistics -----------------------------------------------------------------------------------------

def digest(a) -> str:
    """sha256 of the C-order bytes of ``a``, read in chunks of rows (a memory map is never loaded whole)."""
    h = hashlib.sha256()
    if a.ndim == 0 or len(a) == 0:
        h.update(np.ascontiguousarray(a).tobytes())
    for i in range(0, len(a) if a.ndim else 0, CHUNK):
        h.update(np.ascontiguousarray(a[i:i + CHUNK]).tobytes())
    return h.hexdigest()


class Stats:
    """Minimum, maximum, mean and standard deviation of all values added, accumulated in float64."""

    def __init__(self):
        self.n, self.ref, self.s, self.ss, self.lo, self.hi = 0, None, 0.0, 0.0, np.inf, -np.inf

    def add(self, a):
        for i in range(0, len(a), CHUNK):
            x = np.asarray(a[i:i + CHUNK], np.float64)
            if self.ref is None:
                self.ref = float(x.mean())      # sums are taken about a value near the mean: no cancellation
            d = x - self.ref
            self.n += x.size
            self.s += d.sum()
            self.ss += (d * d).sum()
            self.lo, self.hi = min(self.lo, x.min()), max(self.hi, x.max())
        return self

    def result(self) -> dict:
        m = self.s / self.n
        return {"min": float(self.lo), "max": float(self.hi), "mean": float(self.ref + m),
                "std": float(np.sqrt(max(self.ss / self.n - m * m, 0.0)))}


# ---- writing --------------------------------------------------------------------------------------------------------

def external_part(file, key, array) -> dict:
    """The meta entry of a waveform part that stays in an external npz file. ``array`` is that member as float32."""
    return {"file": str(file), "key": key, "n": len(array), "sha256": digest(array)}


def write_split(path, cfg: dict, split: str, labels, rows, label_spec, *, dt_ns, t_first_ns, unit, waveform,
                extras=None, root=ROOT) -> Path:
    """Write one split of the data set of ``cfg`` to the directory ``path``.

    ``label_spec``: [{"name": .., "unit": ..}, ..] in column order.
    ``waveform``: an (N, L) array, stored here as waveform.npy; or {"parts": [external_part(..), ..],
    "length": L, "stats": Stats().result()} for waveforms that stay in external files.
    ``extras``: name -> (array of N rows, one line saying what it is)."""
    path, extras = Path(path), extras or {}
    if split not in SPLITS:
        raise FormatError(f"split {split!r} is not one of {SPLITS}")
    if path.exists():
        raise StaleDataError(f"{path} exists: remove it to rebuild (it is never overwritten)")
    if np.ndim(rows) != 1 or np.asarray(rows).dtype.kind not in "iu":
        raise FormatError("rows must be a 1-d integer array of source row numbers")
    labels, rows = np.ascontiguousarray(labels, np.float32), np.ascontiguousarray(rows, np.int64)
    if labels.ndim != 2 or labels.shape != (len(rows), len(label_spec)) or not np.isfinite(labels).all():
        raise FormatError(f"labels {labels.shape} must be finite, (N, {len(label_spec)}) and have one row per event")
    if len(rows) == 0 or (np.diff(rows) <= 0).any() or rows[0] < 0:
        raise FormatError("rows must be non-empty, non-negative and strictly increasing")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp", dir=path.parent))     # one per call: writers never share it
    try:
        if isinstance(waveform, dict):
            wf = {"parts": waveform["parts"], "stats": waveform["stats"]}
            n, length = sum(p["n"] for p in wf["parts"]), waveform["length"]
        else:
            if waveform.dtype != np.float32 or waveform.ndim != 2:
                raise FormatError(f"waveform must be (N, L) float32, got {waveform.shape} {waveform.dtype}")
            np.save(tmp / "waveform.npy", waveform)
            wf = {"parts": [{"file": "waveform.npy", "key": None, "n": len(waveform), "sha256": digest(waveform)}],
                  "stats": Stats().add(waveform).result()}
            n, length = waveform.shape
        if n != len(rows):
            raise FormatError(f"{n} waveforms for {len(rows)} rows")
        sha = {"labels": digest(labels), "rows": digest(rows)}
        np.save(tmp / "labels.npy", labels)
        np.save(tmp / "rows.npy", rows)
        info = {}
        for name, (arr, about) in extras.items():
            arr = np.ascontiguousarray(arr)
            if len(arr) != n or not name.isidentifier() or name in ("labels", "rows", "waveform", "meta"):
                raise FormatError(f"extra {name!r}: {len(arr)} rows for {n} events, or not a plain, unreserved name")
            np.save(tmp / f"{name}.npy", arr)
            sha[name], info[name] = digest(arr), {"dtype": str(arr.dtype), "shape": list(arr.shape), "about": about}
        meta = {"set": cfg["set"], "config": cfg["config"], "split": split, "data_hash": data_hash(cfg, root),
                "commit": "anonymous", "n": n, "length": int(length), "dt_ns": float(dt_ns),
                "t_first_ns": float(t_first_ns), "unit": unit, "labels": list(label_spec),
                "label_stats": {s["name"]: Stats().add(labels[:, j]).result() for j, s in enumerate(label_spec)},
                "waveform": wf, "extras": info, "sha256": sha}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
        mask = os.umask(0)
        os.umask(mask)                  # a temporary directory is private: give the split the usual access rights
        os.chmod(tmp, (tmp.stat().st_mode & 0o7000) | (0o777 & ~mask))
        os.rename(tmp, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return path


# ---- reading --------------------------------------------------------------------------------------------------------

class Split:
    """One split on disk. ``labels`` and ``rows`` are loaded and checked at once, waveform parts on first use."""

    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(f"{self.path} is not a built split: run scripts/prepare_data.py for its config")
        self.meta = json.loads((self.path / "meta.json").read_text())
        if META_KEYS - self.meta.keys():
            raise FormatError(f"{self.path}/meta.json lacks {sorted(META_KEYS - self.meta.keys())}")
        self.n, self.length = self.meta["n"], self.meta["length"]
        self.names = [s["name"] for s in self.meta["labels"]]
        self.labels = self._array("labels", "float32", (self.n, len(self.names)))
        self.rows = self._array("rows", "int64", (self.n,))
        parts = self.meta["waveform"]["parts"]
        self._starts = np.cumsum([0] + [p["n"] for p in parts])
        self._parts = [None] * len(parts)
        if self._starts[-1] != self.n:
            raise FormatError(f"{self.path}: the waveform parts hold {self._starts[-1]} events, meta.json says {self.n}")
        ids = {"sha256": self.meta["sha256"], "parts": [p["sha256"] for p in parts]}
        self.content = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()   # names the very arrays

    def __len__(self):
        return self.n

    def _checked(self, a, want, what):
        if digest(a) != want:
            raise DigestError(f"{what}: content differs from the digest in {self.path}/meta.json. The file was "
                              f"changed or damaged: rebuild the data set, do not edit meta.json")
        return a

    def _array(self, name, dtype, shape):
        """An array of the split, refused unless its dtype, shape and bytes are the recorded ones (the digest is of
        bytes: the same bytes read as another dtype would be other values)."""
        a = np.load(self.path / f"{name}.npy")
        if str(a.dtype) != dtype or a.shape != tuple(shape):
            raise FormatError(f"{self.path}/{name}.npy is {a.dtype} {a.shape}, meta.json says {dtype} {tuple(shape)}")
        a = self._checked(a, self.meta["sha256"][name], f"{self.path}/{name}.npy")
        a.flags.writeable = False
        return a

    def extra(self, name):
        if name not in self.meta["extras"]:
            raise KeyError(f"{self.path} has no extra array {name!r}, only {sorted(self.meta['extras'])}")
        return self._array(name, self.meta["extras"][name]["dtype"], self.meta["extras"][name]["shape"])

    def part(self, i):
        """Waveform part i as (n_i, L) float32: a memory map of waveform.npy, or an npz member held in memory."""
        if self._parts[i] is None:
            p = self.meta["waveform"]["parts"][i]
            if p["key"] is None:
                a = np.load(self.path / p["file"], mmap_mode="r")
            else:
                with np.load(p["file"]) as z:
                    a = np.ascontiguousarray(z[p["key"]], np.float32)
            if a.shape != (p["n"], self.length) or a.dtype != np.float32:
                raise FormatError(f"{p['file']}: waveform part {a.shape} {a.dtype}, expected ({p['n']}, {self.length}) float32")
            self._parts[i] = self._checked(a, p["sha256"], p["file"])
            if p["key"] is not None:
                self._parts[i].flags.writeable = False
        return self._parts[i]

    def take(self, idx):
        """The waveforms at positions ``idx`` (any order, repeats allowed), a new (len(idx), L) float32 array."""
        idx = np.asarray(idx)
        if idx.ndim != 1 or (len(idx) and (idx.dtype.kind not in "iu" or idx.min() < 0 or idx.max() >= self.n)):
            raise IndexError(f"positions must be a 1-d integer array inside [0, {self.n})")
        idx = idx.astype(np.int64)
        out = np.empty((len(idx), self.length), np.float32)
        which = np.searchsorted(self._starts, idx, side="right") - 1
        for i in np.unique(which):
            m = which == i
            order = np.argsort(idx[m], kind="stable")        # ascending reads from a memory map
            out[np.flatnonzero(m)[order]] = self.part(i)[idx[m][order] - self._starts[i]]
        return out

    def verify(self):
        """Read everything once and check it against meta.json: digests, shapes, and the recorded statistics."""
        st = Stats()
        for i in range(len(self._parts)):
            st.add(self.part(i))
        for name in self.meta["extras"]:
            self.extra(name)
        want = {"waveform": self.meta["waveform"]["stats"], **self.meta["label_stats"]}
        got = {"waveform": st.result(), **{n: Stats().add(self.labels[:, j]).result() for j, n in enumerate(self.names)}}
        for name in want:
            if not np.allclose([got[name][k] for k in sorted(got[name])], [want[name][k] for k in sorted(got[name])], rtol=1e-9, atol=0):
                raise FormatError(f"{self.path}: statistics of {name} differ from meta.json")
        if (np.diff(self.rows) <= 0).any() or not np.isfinite(self.labels).all():
            raise FormatError(f"{self.path}: rows not increasing or labels not finite")
        return self


def prepare(cfg: dict, build=None, root=ROOT) -> str:
    """Build the splits of ``cfg`` unless they exist. Returns "built" or "exists".

    Existing splits are verified and left untouched; if they were built from another config or other code
    (data hash), StaleDataError says so: a data set is never silently replaced."""
    out, want = Path(cfg["out"]), data_hash(cfg, root)
    have = [(out / s / "meta.json").exists() for s in SPLITS]
    if all(have):
        for s in SPLITS:
            got = Split(out / s).verify().meta["data_hash"]
            if got != want:
                raise StaleDataError(f"{out}/{s} was built from another config or other data-set code (data hash "
                                     f"{got[:12]}, now {want[:12]}). Results on it are not comparable with the present "
                                     f"definition: move it away and rebuild, or restore the config and code")
        return "exists"
    if any(have) or any((out / s).exists() for s in SPLITS):
        raise StaleDataError(f"{out} holds an incomplete data set: remove its dev/ and heldout/ and rebuild")
    (build or importlib.import_module(f"data.{cfg['set']}.build").build)(cfg)
    return "built"


def evaluate(set_name: str):
    """The scoring function of a data set: evaluate(pred, truth) -> dict, both in physical units."""
    return importlib.import_module(f"data.{set_name}.evaluate").evaluate
