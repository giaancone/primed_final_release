"""Atomic file writes: a crash never leaves a half-written file under the final name."""
import json
import os
from pathlib import Path

import numpy as np
import torch


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_torch(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def remove_tmp_files(directory):
    """Delete half-written temporary files of a killed process (call only while holding the run lock)."""
    for f in Path(directory).glob(".*.tmp*"):
        f.unlink(missing_ok=True)
