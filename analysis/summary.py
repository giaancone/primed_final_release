"""The summary of a stage: only accepted runs, averaged only over seeds of one recipe.

    select(runs, stage)          the runs whose pipeline is accepted up to and including ``stage``: every stage of
                                 ``cfg["stages"]`` up to it passes rundir.accepted (finished, not smoke, promised
                                 files present, verdict "pass" on the very result; ridge needs no verdict) and was
                                 run with the present config.json (its recorded config_fp equals
                                 rundir.config_fp(cfg, stage): a config edited after training is refused, never
                                 grouped by its edited content). The hard rule: nothing else enters a summary.
    rejected(runs, stage)        [(run, why)] of the runs select leaves out, for the record
    groups(runs, stage)          the sets of seeds that may be averaged
    require_seeds(groups, seeds) every set has exactly these seeds, or SummaryError naming what is missing
    table(sets, stage)           mean and spread of the held-out metrics over the seeds of each set

Sets of seeds. Two runs are seeds of one recipe only if everything but the seed is the same: the data (the data hash
of the stage's result), the label budget (data.label_frac) and the recipe, which is the fingerprint of the whole
config without ``seed``, ``run.group``, ``run.name`` and ``run.deadline`` (rundir.config_fp drops the run keys). It
is computed from the configs, never taken from a group name: two recipes that differ in one learning rate are two
sets even inside one group, and one recipe in two groups is one set (its group names are listed). A seed that
appears twice in a set is an error: one of the two is a copy or a re-run under another name.

Held-out metrics come from the record of the stage (analysis.read.heldout: ``<run>/heldout.json`` of
scripts/evaluate.py, ``<run>/compress/heldout.json`` of compression/compress.py --evaluate), their names from its
keys; a set whose runs
have other metric names, were evaluated on another stage or on other results than the accepted ones is refused.
The spread is the sample standard deviation (ddof 1; 0 for one seed) and the lowest and highest seed.
"""
import hashlib
from pathlib import Path

import numpy as np

from analysis import read
from wavefm.train import rundir


class SummaryError(ValueError):
    pass


def _why_not(run, stage):
    stages = run["cfg"]["stages"]
    if stage not in stages:
        return f"{stage} is not a stage of this run ({stages})"
    for s in stages[:stages.index(stage) + 1]:
        try:
            if not rundir.accepted(Path(run["path"]) / s):
                return f"stage {s} is not accepted (unfinished, smoke, a promised file missing, or no 'pass' verdict " \
                       f"on its present result)"
            fp = rundir.read_result(Path(run["path"]) / s)["config_fp"]
        except rundir.RunError as e:
            return f"stage {s}: {e}"
        if fp != rundir.config_fp(run["cfg"], s):
            return f"stage {s} was run with another config ({fp}) than {run['path']}/config.json " \
                   f"({rundir.config_fp(run['cfg'], s)}): the config was edited after training"
    return None


def select(runs, stage) -> list:
    return [r for r in runs if _why_not(r, stage) is None]


def rejected(runs, stage) -> list:
    return [(r, w) for r in runs if (w := _why_not(r, stage)) is not None]


def recipe(cfg) -> str:
    """Fingerprint of everything in the config but the seed and the run names and deadline."""
    return rundir.config_fp({k: v for k, v in cfg.items() if k != "seed"})


def groups(runs, stage) -> list:
    """Sets of seeds of the SELECTED ``runs``: [{"data_hash", "label_frac", "recipe", "set", "config", "groups",
    "runs" (sorted by seed), "seeds"}], sorted by group names and budget. A repeated seed raises."""
    sets = {}
    for r in runs:
        key = (rundir.read_result(Path(r["path"]) / stage)["data_hash"], r["label_frac"], recipe(r["cfg"]))
        sets.setdefault(key, []).append(r)
    out = []
    for (data_hash, frac, rec), rs in sets.items():
        seeds = sorted(r["seed"] for r in rs)
        twice = sorted({s for s in seeds if seeds.count(s) > 1})
        if twice:
            names = [str(r["path"]) for r in rs if r["seed"] in twice]
            raise SummaryError(f"seed {twice} appears more than once in one recipe ({rec}, label_frac {frac}): "
                               f"{names}. One of them is a copy or a re-run under another name: remove it")
        out.append({"data_hash": data_hash, "label_frac": frac, "recipe": rec, "set": rs[0]["set"],
                    "config": rs[0]["config"], "groups": sorted({r["group"] for r in rs}),
                    "runs": sorted(rs, key=lambda r: r["seed"]), "seeds": seeds})
    return sorted(out, key=lambda g: (g["groups"], g["label_frac"], g["recipe"]))


def require_seeds(sets, seeds) -> list:
    """Keep exactly ``seeds`` in every set; a set that lacks one raises (seeds beyond them are left out)."""
    seeds, out, missing = sorted(seeds), [], []
    for g in sets:
        lack = [s for s in seeds if s not in g["seeds"]]
        if lack:
            missing.append(f"{'/'.join(g['groups'])} label_frac {g['label_frac']} recipe {g['recipe']}: seeds {lack}")
        rs = [r for r in g["runs"] if r["seed"] in seeds]
        out.append({**g, "runs": rs, "seeds": [r["seed"] for r in rs]})
    if missing:
        raise SummaryError("sets of seeds lack accepted runs (not run, not finished or not judged 'pass'):\n  "
                           + "\n  ".join(missing))
    return out


def _heldout(run, stage):
    ho = read.heldout(run, stage)
    if ho is None:
        raise SummaryError(f"{run['path']} has no heldout record of stage {stage}: run scripts/evaluate.py "
                           f"{run['path']} (compress: compression/compress.py --evaluate)")
    sha = hashlib.sha256((Path(run["path"]) / stage / "result.json").read_bytes()).hexdigest()
    if ho["stage"] != stage or ho["result"]["sha256"] != sha:
        raise SummaryError(f"{run['path']}/heldout.json predicts with stage {ho['stage']} of another result, not with "
                           f"the accepted {stage}: it does not belong to this summary")
    return ho["metrics"]


def table(sets, stage) -> list:
    """One row per set: its identity, seeds, and for every held-out metric its mean, std, min and max."""
    rows = []
    for g in sets:
        ms = [_heldout(r, stage) for r in g["runs"]]
        names = sorted(ms[0])
        if any(sorted(m) != names for m in ms):
            raise SummaryError(f"the runs of {g['groups']} label_frac {g['label_frac']} were scored with different "
                               f"metrics: {[sorted(m) for m in ms]}")
        row = {k: g[k] for k in ("groups", "set", "config", "label_frac", "recipe", "data_hash", "seeds")}
        row["n_seeds"] = len(ms)
        for n in names:
            v = np.array([m[n] for m in ms], np.float64)
            row.update({f"{n}_mean": float(v.mean()), f"{n}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        f"{n}_min": float(v.min()), f"{n}_max": float(v.max())})
        rows.append(row)
    return rows
