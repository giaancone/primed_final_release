"""The data-efficiency figure: a held-out metric against the number of labelled rows, one line per method.

    efficiency(runs, metric, path) -> ({group: {"x", "mean", "lo", "hi", "n_seeds"}}, figure)

``runs`` are run dicts of analysis.read.runs that the caller has ALREADY SELECTED (the verdict gate and the check
that a group is one recipe with a complete set of seeds belong to the summary, not here). A method is a run group.
A point is the mean over the seeds of a group at one labelled-row count; the bar spans the lowest and the highest
seed (well defined from two seeds on). The figure is saved, and the points and the (closed) figure are returned, so
both can be checked.

Everything named on the figure comes from the input files: the data set and config from config.json, the metric
from the keys of heldout.json, the number of dev rows from the dev meta.json, and the reference from
``data/<set>/reference/<config>.json`` ({"method", "metrics": {name: value}}). The reference is a horizontal line
only if that file has this metric; an empty reference draws nothing.

The labelled-row count follows the rule of wavefm.data.splits.labelled: round(label_frac * n), all n rows at
label_frac 1 (a minimum never enlarges the budget).

Runs of different data sets, configs or data hashes are refused: a comparison is only made on the same data.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.curves import _figure, _save
from data import store


class EfficiencyError(ValueError):
    pass


def n_labelled(cfg, n_dev) -> int:
    d = cfg["data"]
    return n_dev if d["label_frac"] >= 1 else int(round(d["label_frac"] * n_dev))


def _dev_meta(set_name, config, root):
    p = Path(store.load_config(set_name, config, root)["out"]) / "dev" / "meta.json"
    if not p.exists():
        raise EfficiencyError(f"{p} is missing: build the data set (scripts/prepare_data.py) to count its dev rows")
    return json.loads(p.read_text())


def reference(set_name, config, root=store.ROOT) -> dict:
    """The reference of a data set config, {} if it has none."""
    p = Path(root) / set_name / "reference" / f"{config}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def efficiency(runs, metric, path, root=store.ROOT) -> dict:
    """Plot ``metric`` of heldout.json against the labelled-row count; ``root`` is the data package (data/)."""
    if not runs:
        raise EfficiencyError("no runs given")
    data = {(r["set"], r["config"]) for r in runs}
    if len(data) != 1:
        raise EfficiencyError(f"runs on different data {sorted(data)}: compare only on the same data set and config")
    (set_name, config), = data
    meta = _dev_meta(set_name, config, root)
    pts = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        ho = r["heldout"]
        if ho is None or metric not in ho["metrics"]:
            raise EfficiencyError(f"{r['path']}: no {metric!r} in heldout.json "
                                  f"(has {sorted((ho or {}).get('metrics', {}))}); run scripts/evaluate.py")
        if ho["data_hash"] != meta["data_hash"]:
            raise EfficiencyError(f"{r['path']}: evaluated on data hash {ho['data_hash'][:12]}, the data set is now "
                                  f"{meta['data_hash'][:12]}: not the same problem, evaluate again")
        x = n_labelled(r["cfg"], meta["n"])
        if r["seed"] in pts[r["group"]][x]:
            raise EfficiencyError(f"{r['path']}: seed {r['seed']} appears twice in group {r['group']} at {x} rows")
        pts[r["group"]][x][r["seed"]] = float(ho["metrics"][metric])
    out = {}
    for g, by_x in sorted(pts.items()):
        xs = sorted(by_x)
        v = [np.array(list(by_x[x].values())) for x in xs]
        out[g] = {"x": xs, "mean": [float(a.mean()) for a in v], "lo": [float(a.min()) for a in v],
                  "hi": [float(a.max()) for a in v], "n_seeds": [len(a) for a in v]}
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, (g, p) in enumerate(out.items()):
        m = np.array(p["mean"])
        ax.errorbar(p["x"], m, yerr=[m - p["lo"], np.array(p["hi"]) - m], color=f"C{k % 10}", marker="o", capsize=3,
                    label=f"{g} (seeds: {', '.join(map(str, sorted(set(p['n_seeds']))))})")
    ref = reference(set_name, config, root)
    if metric in ref.get("metrics", {}):
        ax.axhline(ref["metrics"][metric], color="0.3", ls="--", label=f"reference: {ref.get('method', '?')}")
    ax.set_xscale("log")
    ax.set_xlabel(f"labelled rows of dev (of {meta['n']})"); ax.set_ylabel(metric)
    ax.set_title(f"{set_name} / {config}, held out", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); _save(fig, path); plt.close(fig)
    return out, fig
