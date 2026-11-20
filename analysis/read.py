"""Reading the run directories of train/rundir.py into plain dicts, and flattening them into a table.

    runs(root)          one dict per <root>/<group>/<name>/ that has a config.json
    table(runs, stage)  one flat row per run that has that stage: what a person compares side by side

A run dict:
    {"group", "name", "path", "seed", "set", "config", "label_frac",      identity, from config.json
     "cfg":     the expanded config,
     "heldout": <run>/heldout.json or None (written once by scripts/evaluate.py; heldout(run, stage) gives the
                record that scores one stage),
     "stages":  {stage: {"group", "name", "stage", "path", "result", "history", "loo", "verdict"}}}

A stage appears as soon as its directory exists. ``result`` is None until the stage has finished (result.json is
written last), so an unfinished or crashed stage shows up with its partial curves. ``history`` is a dict of float64
arrays, one per key of history.json, NaN where a row lacks the key (validation is evaluated at fewer steps than
training is logged; ssl has no validation, so it has no "val_loss" at all); None if there is no history (ridge).
``loo`` is the leave-one-out table of the ridge stage, ``verdict`` verdict.json; None when absent.

Nothing here decides anything: no run is dropped for its verdict, smoke flag or data, and nothing is averaged. The
gate (rundir.accepted) and the summary are applied by the caller.
"""
from pathlib import Path

import numpy as np

from analysis.curves import hints
from wavefm.train import rundir
from wavefm.utils.io import load_json


def history_arrays(rows) -> dict:
    """history.json rows -> {key: float64 array}, NaN where a row lacks the key; a key with text values (the
    ``phase`` of the compress stage) gives an object array of those texts."""
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    cols = {k: [r.get(k) for r in rows] for k in keys}
    return {k: np.array(v, object) if any(isinstance(x, str) for x in v)
            else np.array([np.nan if x is None else x for x in v], np.float64) for k, v in cols.items()}


def _maybe(path):
    return load_json(path) if path.exists() else None


def _stage(run_dir: Path, group, name, stage) -> dict:
    d = run_dir / stage
    h = _maybe(d / "history.json")
    return {"group": group, "name": name, "stage": stage, "path": d,
            "result": rundir.read_result(d) if (d / "result.json").exists() else None,
            "history": history_arrays(h) if h else None, "loo": _maybe(d / "loo.json"),
            "verdict": _maybe(d / "verdict.json")}


def heldout(run, stage):
    """The heldout record that scores ``stage`` of ``run``, None if there is none: <run>/compress/heldout.json for
    compress (scored in its own environment), else <run>/heldout.json if it evaluated this stage."""
    if stage == "compress":
        return _maybe(Path(run["path"]) / "compress" / "heldout.json")
    return run["heldout"] if run["heldout"] and run["heldout"]["stage"] == stage else None


def runs(root) -> list:
    """Every run under ``root`` (the output directory), sorted by group and name."""
    out = []
    for cfg_file in sorted(Path(root).glob("*/*/config.json")):
        d = cfg_file.parent
        cfg, group, name = load_json(cfg_file), d.parent.name, d.name
        out.append({"group": group, "name": name, "path": d, "seed": cfg["seed"], "set": cfg["data"]["set"],
                    "config": cfg["data"]["config"], "label_frac": cfg["data"]["label_frac"], "cfg": cfg,
                    "heldout": _maybe(d / "heldout.json"),
                    "stages": {s: _stage(d, group, name, s) for s in cfg["stages"] if (d / s).is_dir()}})
    return out


def _last(a):
    a = a[np.isfinite(a)]
    return float(a[-1]) if len(a) else None


def table(runs, stage) -> list:
    """One row per run that has ``stage``: identity, requested and actual length, final losses, hints (recomputed from
    the curves by analysis.curves.hints), verdict and the held-out metrics of this stage (``heldout``). Missing
    values are None."""
    rows = []
    for r in runs:
        s = r["stages"].get(stage)
        if s is None:
            continue
        res, h = s["result"] or {}, s["history"]
        row = {"group": r["group"], "name": r["name"], "seed": r["seed"], "set": r["set"], "config": r["config"],
               "label_frac": r["label_frac"], "stage": stage, "finished": s["result"] is not None,
               "smoke": res.get("smoke")}
        row.update({f"{k}_{x}": v for k in ("requested", "actual") for x, v in (res.get(k) or {}).items()})
        row["final_train_loss"] = _last(h["train_loss"]) if h and "train_loss" in h else None
        row["final_val_loss"] = _last(h["val_loss"]) if h and "val_loss" in h else None
        row.update({f"hint_{k}": v for k, v in (hints(h) if h else {}).items()})
        row["verdict"] = s["verdict"]["verdict"] if s["verdict"] else None
        row.update({f"heldout_{k}": v for k, v in ((heldout(r, stage) or {}).get("metrics") or {}).items()})
        rows.append(row)
    return rows
