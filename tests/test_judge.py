"""The verdict gate and the summary on hand-written run directories: what enters a summary (analysis.summary.select),
which runs are seeds of one recipe (groups, require_seeds), the averaged table, and what the judge is shown
(analysis.judge, scripts/judge.py)."""
import hashlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import pytest

from analysis import judge, read, summary
from wavefm import paths
from wavefm.train import rundir
from wavefm.utils import config
from wavefm.utils.io import save_json

HIST = [{"step": s, "train_loss": t, "val_loss": v, "lr": 1e-3} for s, t, v in
        zip([0, 1, 2, 3], [4.0, 2.0, 1.0, 1.0], [5.0, 3.0, 2.0, 2.5])]


def make_run(root, group, name, *, seed=0, frac=0.1, stages=("finetune",), lr=1e-3, smoke=False, data_hash="h1",
             verdicts=None, heldout=None):
    """One run directory: config.json, and per stage history.json (not for ridge) and result.json; ``verdicts``
    {stage: verdict}; ``heldout`` {metric: value} written as heldout.json of the last stage."""
    cfg = config.load(paths.ROOT / "configs" / "teacher.yaml")
    cfg = config.merge(cfg, {"seed": seed, "stages": list(stages),
                             "run": {"group": group, "name": name, "smoke": smoke},
                             "data": {"label_frac": frac}, "finetune": {"lr_head": lr}})
    d = root / group / name
    save_json(cfg, d / "config.json")
    for s in stages:
        files = []
        if s != "ridge":
            save_json(HIST, d / s / "history.json")
            files = ["history.json"]
        (d / s).mkdir(parents=True, exist_ok=True)
        rundir.write_result(d / s, s, cfg, data_hash, None, requested={"steps": 3}, actual={"steps": 3}, hints={},
                            wall_s=1.0, artifacts=files)
        if (verdicts or {}).get(s):
            rundir.write_verdict(d / s, verdicts[s], "by hand", "test")
    if heldout is not None:
        sha = hashlib.sha256((d / stages[-1] / "result.json").read_bytes()).hexdigest()
        save_json({"metrics": heldout, "stage": stages[-1], "data_hash": data_hash, "n": 10,
                   "result": {"sha256": sha}}, d / "heldout.json")
    return d


def names(runs):
    return sorted(f"{r['group']}/{r['name']}" for r in runs)


# ---- ① the gate -----------------------------------------------------------------------------------------------------

def test_only_runs_judged_pass_on_their_present_result_are_selected(tmp_path):
    make_run(tmp_path, "g", "ok", verdicts={"finetune": "pass"})
    make_run(tmp_path, "g", "unjudged")
    make_run(tmp_path, "g", "overfit", verdicts={"finetune": "overfit"})
    make_run(tmp_path, "g", "smoke", smoke=True, verdicts={"finetune": "pass"})
    old = make_run(tmp_path, "g", "rerun", verdicts={"finetune": "pass"})
    cfg = config.load(old / "config.json")
    rundir.write_result(old / "finetune", "finetune", cfg, "h1", None, requested={"steps": 4}, actual={"steps": 4},
                        hints={}, wall_s=2.0, artifacts=["history.json"])     # run again: the verdict is of the old one
    lost = make_run(tmp_path, "g", "lost_file", verdicts={"finetune": "pass"})
    (lost / "finetune" / "history.json").unlink()
    runs = read.runs(tmp_path)
    assert names(summary.select(runs, "finetune")) == ["g/ok"]
    assert names(r for r, _ in summary.rejected(runs, "finetune")) == \
        ["g/lost_file", "g/overfit", "g/rerun", "g/smoke", "g/unjudged"]
    assert names(summary.select(runs, "ssl")) == []                          # not a stage of these runs


def test_a_later_stage_needs_every_earlier_stage_accepted(tmp_path):
    st = ("ssl", "ridge", "finetune")
    make_run(tmp_path, "g", "all", stages=st, verdicts={"ssl": "pass", "finetune": "pass"})
    make_run(tmp_path, "g", "ssl_unjudged", stages=st, verdicts={"finetune": "pass"})
    make_run(tmp_path, "g", "ssl_under", stages=st, verdicts={"ssl": "undertrained", "finetune": "pass"})
    runs = read.runs(tmp_path)
    assert names(summary.select(runs, "ssl")) == ["g/all"]
    assert names(summary.select(runs, "ridge")) == ["g/all"]                 # ridge itself needs no verdict
    assert names(summary.select(runs, "finetune")) == ["g/all"]


def test_pending_lists_what_needs_a_verdict_and_prepare_draws_it(tmp_path):
    st = ("ssl", "ridge", "finetune")
    make_run(tmp_path, "g", "a", stages=st, verdicts={"ssl": "pass"})
    make_run(tmp_path, "g", "b", stages=st, verdicts={"ssl": "overfit", "finetune": "unclear"})
    make_run(tmp_path, "g", "smoke", stages=st, smoke=True)
    old = make_run(tmp_path, "g", "rerun", stages=("finetune",), verdicts={"finetune": "pass"})
    rundir.write_result(old / "finetune", "finetune", config.load(old / "config.json"), "h1", None,
                        requested={}, actual={}, hints={}, wall_s=0)
    got = [(s["name"], s["stage"]) for s in judge.pending(tmp_path)]
    assert got == [("a", "finetune"), ("rerun", "finetune")]                 # no ridge, no smoke, judged ones out
    rows = judge.prepare_pending(tmp_path)
    assert [(r["name"], r["stage"]) for r in rows] == got
    r = rows[0]
    assert (r["requested_steps"], r["final_train_loss"], r["final_val_loss"]) == (3, 1.0, 2.5)
    assert (r["hint_on"], r["hint_final_over_min"], r["hint_min_step"]) == ("val_loss", 0.25, 2)
    assert (tmp_path / "g" / "a" / "finetune" / "curves.png").stat().st_size > 0
    assert "hint_final_over_min" in judge.show(rows)


def test_judge_command_line(tmp_path):
    make_run(tmp_path, "g", "a")
    script = str(paths.ROOT / "scripts" / "judge.py")
    out = subprocess.run([sys.executable, script, "--pending", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0 and "finetune" in out.stdout, out.stderr
    stage = tmp_path / "g" / "a" / "finetune"
    out = subprocess.run([sys.executable, script, str(stage), "pass", "flat at the end", "--judge", "test"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert rundir.accepted(stage) and judge.pending(tmp_path) == []
    out = subprocess.run([sys.executable, script, str(stage), "good"], capture_output=True, text=True)
    assert out.returncode != 0                                               # not a verdict, no reason, no judge


# ---- ② sets of seeds ------------------------------------------------------------------------------------------------

def test_only_one_data_budget_and_recipe_form_a_set_of_seeds(tmp_path):
    p = {"verdicts": {"finetune": "pass"}}
    for s in (0, 1, 2):
        make_run(tmp_path, "g", f"lr1_s{s}", seed=s, **p)
    for s in (0, 1):
        make_run(tmp_path, "g", f"lr3_s{s}", seed=s, lr=3e-3, **p)           # same group, another learning rate
        make_run(tmp_path, "g", f"frac_s{s}", seed=s, frac=0.5, **p)
        make_run(tmp_path, "g", f"data_s{s}", seed=s, data_hash="h2", **p)
    make_run(tmp_path, "other", "lr1_s3", seed=3, **p)                       # the same recipe under another group
    sets = summary.groups(summary.select(read.runs(tmp_path), "finetune"), "finetune")
    got = sorted((g["label_frac"], g["data_hash"], tuple(g["seeds"]), tuple(g["groups"]),
                  tuple(r["name"] for r in g["runs"])) for g in sets)
    assert got == sorted([(0.1, "h1", (0, 1, 2, 3), ("g", "other"), ("lr1_s0", "lr1_s1", "lr1_s2", "lr1_s3")),
                          (0.1, "h1", (0, 1), ("g",), ("lr3_s0", "lr3_s1")),
                          (0.5, "h1", (0, 1), ("g",), ("frac_s0", "frac_s1")),
                          (0.1, "h2", (0, 1), ("g",), ("data_s0", "data_s1"))])
    with pytest.raises(summary.SummaryError, match=r"seeds \[2\]") as e:
        summary.require_seeds(sets, [0, 1, 2])
    assert "lr1" not in str(e.value) and str(e.value).count("seeds [2]") == 3
    kept = summary.require_seeds(sets, [0, 1])
    assert sorted(len(g["runs"]) for g in kept) == [2, 2, 2, 2]               # seeds beyond the asked ones left out


def test_a_config_edited_after_training_is_rejected_not_regrouped(tmp_path):
    """Two seeds trained with different learning rates; one config.json is then edited to the other's rate. The
    edited run is rejected (its result's config_fp is not that of its config.json) and never averaged with the other."""
    p = {"verdicts": {"finetune": "pass"}}
    make_run(tmp_path, "g", "s0", seed=0, **p)
    d = make_run(tmp_path, "g", "s1", seed=1, lr=3e-3, **p)
    cfg = config.load(d / "config.json")
    cfg["finetune"]["lr_head"] = 1e-3
    save_json(cfg, d / "config.json")
    runs = read.runs(tmp_path)
    assert names(summary.select(runs, "finetune")) == ["g/s0"]
    (_, why), = summary.rejected(runs, "finetune")
    assert "edited after training" in why
    assert [g["seeds"] for g in summary.groups(summary.select(runs, "finetune"), "finetune")] == [[0]]


def test_a_repeated_seed_is_refused(tmp_path):
    make_run(tmp_path, "g", "s0", seed=0, verdicts={"finetune": "pass"})
    make_run(tmp_path, "g", "s0_copy", seed=0, verdicts={"finetune": "pass"})
    with pytest.raises(summary.SummaryError, match="seed \\[0\\] appears more than once"):
        summary.groups(read.runs(tmp_path), "finetune")


def test_summary_table_by_hand(tmp_path):
    p = {"verdicts": {"finetune": "pass"}}
    for s, v in ((0, 1.0), (1, 2.0), (2, 6.0)):
        make_run(tmp_path, "g", f"s{s}", seed=s, heldout={"some_metric": v, "other": 10 * v}, **p)
    sets = summary.require_seeds(summary.groups(summary.select(read.runs(tmp_path), "finetune"), "finetune"), [0, 1, 2])
    row, = summary.table(sets, "finetune")
    assert row["n_seeds"] == 3 and row["seeds"] == [0, 1, 2] and row["groups"] == ["g"]
    assert (row["some_metric_mean"], row["some_metric_min"], row["some_metric_max"]) == (3.0, 1.0, 6.0)
    assert row["some_metric_std"] == pytest.approx(7 ** 0.5) and row["other_mean"] == 30.0
    # heldout.json of another result (the stage was run again after the evaluation) does not belong to the summary
    d = tmp_path / "g" / "s0"
    rundir.write_result(d / "finetune", "finetune", config.load(d / "config.json"), "h1", None, requested={},
                        actual={}, hints={}, wall_s=0, artifacts=["history.json"])
    rundir.write_verdict(d / "finetune", "pass", "again", "test")
    sets = summary.groups(summary.select(read.runs(tmp_path), "finetune"), "finetune")
    with pytest.raises(summary.SummaryError, match="does not belong"):
        summary.table(sets, "finetune")


def test_a_set_without_heldout_or_with_other_metrics_is_refused(tmp_path):
    make_run(tmp_path, "g", "s0", seed=0, verdicts={"finetune": "pass"}, heldout={"a": 1.0})
    make_run(tmp_path, "g", "s1", seed=1, verdicts={"finetune": "pass"}, heldout={"b": 1.0})
    sets = summary.groups(read.runs(tmp_path), "finetune")
    with pytest.raises(summary.SummaryError, match="different metrics"):
        summary.table(sets, "finetune")
    make_run(tmp_path, "h", "s0", seed=0, verdicts={"finetune": "pass"})
    sets = summary.groups([r for r in read.runs(tmp_path) if r["group"] == "h"], "finetune")
    with pytest.raises(summary.SummaryError, match="scripts/evaluate.py"):
        summary.table(sets, "finetune")


def test_each_stage_is_scored_by_its_own_heldout_record(tmp_path):
    """compress is scored into <run>/compress/heldout.json, the other stages into <run>/heldout.json: a row or a
    summary of one stage never shows another stage's score."""
    d = make_run(tmp_path, "g", "s0", stages=("ridge", "finetune", "compress"),
                 verdicts={"finetune": "pass", "compress": "pass"})
    for stage, m, f in (("finetune", 1.0, d / "heldout.json"), ("compress", 2.0, d / "compress" / "heldout.json")):
        sha = hashlib.sha256((d / stage / "result.json").read_bytes()).hexdigest()
        save_json({"metrics": {"m": m}, "stage": stage, "data_hash": "h1", "n": 10, "result": {"sha256": sha}}, f)
    runs = read.runs(tmp_path)
    rows = [r for s in ("ridge", "finetune", "compress") for r in read.table(runs, s)]
    assert [r.get("heldout_m") for r in rows] == [None, 1.0, 2.0]
    for stage, want in (("finetune", 1.0), ("compress", 2.0)):
        row, = summary.table(summary.groups(summary.select(runs, stage), stage), stage)
        assert row["m_mean"] == want
    with pytest.raises(summary.SummaryError, match="no heldout record of stage ridge"):
        summary.table(summary.groups(summary.select(runs, "ridge"), "ridge"), "ridge")
