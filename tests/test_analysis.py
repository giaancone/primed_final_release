"""analysis/: the reader, the hints, the curve figures and the data-efficiency figure on hand-written run directories,
checked against hand-computed numbers; and that nothing about a data set is hard-coded in analysis/."""
import json
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import fakes
from analysis import curves, efficiency, overview, read
from wavefm import paths
from wavefm.train import rundir
from wavefm.utils import config
from wavefm.utils.io import save_json

STEPS = list(range(0, 101, 10))
V_MID = [10, 8, 6, 4, 2, 1, 2, 3, 4, 5, 6]          # minimum in the middle
V_DOWN = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]        # still falling at the end


def hist_rows(val=None, train=V_DOWN, lr=1e-3):
    return [{"step": s, "train_loss": float(t), "lr": lr, "grad_norm": 1.0, "clip_frac": 0.0,
             **({"val_loss": float(v)} if val is not None else {})} for s, t, v in zip(STEPS, train, val or train)]


def test_hints_by_hand():
    h = curves.hints(hist_rows(val=V_MID))           # min 1 at step 50, final 6; value at step 90 is 5
    assert h == {"on": "val_loss", "final": 6.0, "min": 1.0, "min_step": 50, "final_over_min": 5.0,
                 "last10_drop": pytest.approx(-0.2)}
    h = curves.hints(hist_rows(val=V_DOWN))          # min = final 1 at step 100; value at step 90 is 2
    assert (h["min_step"], h["final_over_min"], h["last10_drop"]) == (100, 0.0, 0.5)
    h = curves.hints(hist_rows(val=None, train=V_MID))   # no validation (ssl): the training loss
    assert (h["on"], h["min_step"], h["final_over_min"]) == ("train_loss", 50, 5.0)
    rows = hist_rows(val=V_MID)
    for r in rows[1::2]:                              # validation only at every other logged step
        del r["val_loss"]
    assert curves.hints(rows)["min_step"] == 40 and curves.hints(rows)["final"] == 6.0
    with pytest.raises(curves.CurveError, match="no finite"):
        curves.hints([{"step": 0, "train_loss": float("nan")}])


# ---- fake run directories -------------------------------------------------------------------------------------------

def data_root(tmp_path, reference=None):
    """A data package with one set ``fake``, config ``tiny`` (60 dev rows) and its reference file."""
    root = fakes.make_root(tmp_path)
    path, *_ = fakes.make_split(tmp_path, "npy", root=root)
    (root / "fake" / "configs" / "tiny.yaml").write_text(f"out: {path.parent}\nseed: 3\n")
    (root / "fake" / "reference").mkdir()
    (root / "fake" / "reference" / "tiny.json").write_text(json.dumps(reference or {}))
    return root, json.loads((path / "meta.json").read_text())["data_hash"]


# group, seed, label_frac, stages, heldout score: 60 dev rows -> 0.1: 6 labelled rows, 0.5: 30, 1.0: 60
RUNS = [("cnn", 0, 0.1, ["finetune"], 5.0), ("cnn", 1, 0.1, ["finetune"], 7.0), ("cnn", 2, 0.1, ["finetune"], 9.0),
        ("cnn", 0, 1.0, ["finetune"], 2.0), ("cnn", 1, 1.0, ["finetune"], 4.0),
        ("fm", 0, 0.5, ["ssl", "ridge"], 3.0), ("fm", 1, 0.5, ["ssl", "ridge"], 3.5)]


def make_runs(tmp_path, data_hash):
    base = config.load(paths.ROOT / "configs" / "teacher.yaml")
    out = tmp_path / "out"
    for group, seed, frac, stages, score in RUNS:
        name = f"f{frac}_s{seed}"
        cfg = config.merge(base, {"seed": seed, "stages": stages, "run": {"group": group, "name": name},
                                  "data": {"set": "fake", "config": "tiny", "label_frac": frac}})
        d = out / group / name
        save_json(cfg, d / "config.json")
        for st in stages:
            if st == "ridge":
                save_json({"alpha": [1e-3, 1.0], "loo": [0.5, 0.4]}, d / st / "loo.json")
                rundir.write_result(d / st, st, cfg, data_hash, None, requested={"rows": 6}, actual={"rows": 6},
                                    hints={}, wall_s=1.0)
                continue
            save_json(hist_rows(val=None if st == "ssl" else V_MID, train=[x + seed for x in V_DOWN]), d / st / "history.json")
            rundir.write_result(d / st, st, cfg, data_hash, None, requested={"steps": 100, "epochs": 2.0},
                                actual={"steps": 100, "epochs": 2.0}, hints={}, wall_s=1.0)
            if st == "finetune" and seed != 2:
                rundir.write_verdict(d / st, "pass" if seed == 0 else "overfit", "minimum at step 50", "test")
        save_json({"metrics": {"score_a": score, "n": 10}, "stage": stages[-1], "data_hash": data_hash, "n": 10},
                  d / "heldout.json")
    (out / "cnn" / "unfinished" / "finetune").mkdir(parents=True)             # crashed: config and a partial history
    save_json(config.merge(base, {"run": {"group": "cnn", "name": "unfinished"}, "data": {"set": "fake", "config": "tiny"}}),
              out / "cnn" / "unfinished" / "config.json")
    save_json(hist_rows(val=V_MID)[:3], out / "cnn" / "unfinished" / "finetune" / "history.json")
    return out


def test_read_and_table(tmp_path):
    root, dh = data_root(tmp_path)
    runs = read.runs(make_runs(tmp_path, dh))
    assert [(r["group"], r["name"]) for r in runs][:2] == [("cnn", "f0.1_s0"), ("cnn", "f0.1_s1")] and len(runs) == 8
    fm = next(r for r in runs if r["group"] == "fm")
    assert "val_loss" not in fm["stages"]["ssl"]["history"] and fm["stages"]["ridge"]["history"] is None
    assert fm["stages"]["ridge"]["loo"]["loo"] == [0.5, 0.4] and fm["heldout"]["metrics"]["score_a"] == 3.0
    rows = read.table(runs, "finetune")
    assert len(rows) == 6                                                   # the 5 finished and the unfinished one
    r = next(x for x in rows if x["name"] == "f1.0_s1")
    assert r == {"group": "cnn", "name": "f1.0_s1", "seed": 1, "set": "fake", "config": "tiny", "label_frac": 1.0,
                 "stage": "finetune", "finished": True, "smoke": False, "requested_steps": 100, "requested_epochs": 2.0,
                 "actual_steps": 100, "actual_epochs": 2.0, "final_train_loss": 2.0, "final_val_loss": 6.0,
                 "hint_on": "val_loss", "hint_final": 6.0, "hint_min": 1.0, "hint_min_step": 50,
                 "hint_final_over_min": 5.0, "hint_last10_drop": pytest.approx(-0.2), "verdict": "overfit",
                 "heldout_score_a": 4.0, "heldout_n": 10}
    assert [x["verdict"] for x in rows] == ["pass", "overfit", None, "pass", "overfit", None]   # nothing filtered
    u = rows[-1]
    assert (u["name"], u["finished"], u["final_val_loss"], u["hint_min_step"]) == ("unfinished", False, 6.0, 20)
    assert "heldout_score_a" not in u and "actual_steps" not in u
    ssl = read.table(runs, "ssl")
    assert len(ssl) == 2 and ssl[1]["final_val_loss"] is None and ssl[1]["hint_on"] == "train_loss"
    assert ssl[1]["final_train_loss"] == 2.0 and ssl[1]["hint_last10_drop"] == pytest.approx(1 / 3)
    ridge = read.table(runs, "ridge")
    assert ridge[0]["actual_rows"] == 6 and ridge[0]["final_train_loss"] is None and "hint_on" not in ridge[0]


def test_history_with_text_column():
    """The compress stage's history.json has a text column ``phase``: read.runs must not fail on it (found by the smoke
    test of the whole chain)."""
    h = read.history_arrays([{"phase": "prune", "step": 0, "train_loss": None, "val_loss": 2.0},
                             {"phase": "qat", "step": 5, "train_loss": 1.0, "val_loss": 1.5}])
    assert list(h["phase"]) == ["prune", "qat"] and np.isnan(h["train_loss"][0]) and h["val_loss"][1] == 1.5


COMPRESS = [{"phase": ph, "step": s, "train_loss": t, "val_loss": v, "lr": 1e-3} for ph, s, t, v in
            [("prune", 0, None, 3.0), ("prune", 5, 2.0, 1.0), ("prune", 10, 1.5, 2.0),
             ("qat", 10, None, 3.0), ("qat", 15, 2.5, 2.5), ("qat", 20, 2.2, 2.0)]]


def test_compress_phases_are_judged_and_drawn_separately(monkeypatch):
    """Hints on the last phase (QAT fell throughout: no deterioration against the pruning minimum 1.0); the figure
    marks where each phase starts and draws the student's val_truth_loss."""
    assert curves.hints(COMPRESS) == {"on": "val_loss", "final": 2.0, "min": 2.0, "min_step": 20,
                                      "final_over_min": 0.0, "last10_drop": pytest.approx(0.2)}
    figs = []
    monkeypatch.setattr(curves, "_save", lambda fig, path: figs.append(fig))
    student = [dict(r, val_truth_loss=r["val_loss"] + 1) for r in hist_rows(val=V_MID)]
    for h, stage in ((COMPRESS, "compress"), (student, "student")):
        curves.plot({"history": read.history_arrays(h), "result": None, "path": "p", "stage": stage, "group": "g",
                     "name": "n"}, "unused.png")
    ax_c, ax_s = (f.axes[0] for f in figs)
    marks = [ln.get_xdata()[0] for ln in ax_c.get_lines() if ln.get_linestyle() == "--"]
    assert marks == [0, 10] and [t.get_text() for t in ax_c.texts[:2]] == [" prune", " qat"]
    truth, = [ln for ln in ax_s.get_lines() if ln.get_label() == "val_truth_loss"]
    assert list(truth.get_ydata()) == [v + 1.0 for v in V_MID]


def test_curve_figures(tmp_path):
    root, dh = data_root(tmp_path)
    runs = read.runs(make_runs(tmp_path, dh))
    for r in runs:
        for s in r["stages"].values():
            if s["history"] is not None:
                curves.plot(s, tmp_path / "fig" / f"{r['group']}_{r['name']}_{s['stage']}.png")
    assert len(list((tmp_path / "fig").glob("*.png"))) == 8 and not list((tmp_path / "fig").glob(".*"))
    with pytest.raises(curves.CurveError, match="loo"):
        curves.plot(runs[-1]["stages"]["ridge"], tmp_path / "ridge.png")
    curves.plot_many([s for r in runs for s in r["stages"].values()], tmp_path / "many.png")
    assert (tmp_path / "many.png").stat().st_size > 0


def selected(tmp_path, reference=None):
    root, dh = data_root(tmp_path, reference)
    runs = [r for r in read.runs(make_runs(tmp_path, dh)) if r["name"] != "unfinished"]
    return root, runs


def ref_lines(fig):
    return [ln for ln in fig.axes[0].get_lines() if len(set(ln.get_ydata())) == 1 and len(ln.get_ydata()) == 2
            and tuple(ln.get_xdata()) == (0, 1)]


def test_efficiency_points_by_hand(tmp_path):
    root, runs = selected(tmp_path)
    pts, fig = efficiency.efficiency(runs, "score_a", tmp_path / "eff.png", root=root)
    assert pts == {"cnn": {"x": [6, 60], "mean": [7.0, 3.0], "lo": [5.0, 2.0], "hi": [9.0, 4.0], "n_seeds": [3, 2]},
                   "fm": {"x": [30], "mean": [3.25], "lo": [3.0], "hi": [3.5], "n_seeds": [2]}}
    assert (tmp_path / "eff.png").exists() and ref_lines(fig) == []         # empty reference: no line
    assert fig.axes[0].get_ylabel() == "score_a" and "fake / tiny" in fig.axes[0].get_title()


def test_efficiency_reference_from_file(tmp_path):
    root, runs = selected(tmp_path, {"method": "some teacher", "metrics": {"score_a": 2.5, "other": 1.0}})
    _, fig = efficiency.efficiency(runs, "score_a", tmp_path / "eff.png", root=root)
    lines = ref_lines(fig)
    assert len(lines) == 1 and set(lines[0].get_ydata()) == {2.5} and "some teacher" in lines[0].get_label()
    _, fig = efficiency.efficiency(runs, "n", tmp_path / "eff_n.png", root=root)
    assert ref_lines(fig) == []                                             # reference without this metric


def test_efficiency_refuses(tmp_path):
    root, runs = selected(tmp_path)
    with pytest.raises(efficiency.EfficiencyError, match="no 'missing'"):
        efficiency.efficiency(runs, "missing", tmp_path / "e.png", root=root)
    other = dict(runs[0], set="other")
    with pytest.raises(efficiency.EfficiencyError, match="same data"):
        efficiency.efficiency(runs + [other], "score_a", tmp_path / "e.png", root=root)
    with pytest.raises(efficiency.EfficiencyError, match="twice"):
        efficiency.efficiency(runs + [runs[0]], "score_a", tmp_path / "e.png", root=root)
    stale = dict(runs[0], heldout=dict(runs[0]["heldout"], data_hash="0" * 64))
    with pytest.raises(efficiency.EfficiencyError, match="not the same problem"):
        efficiency.efficiency([stale], "score_a", tmp_path / "e.png", root=root)
    with pytest.raises(efficiency.EfficiencyError, match="no runs"):
        efficiency.efficiency([], "score_a", tmp_path / "e.png", root=root)
    assert not (tmp_path / "e.png").exists()


def test_overview_reads_dev_only(tmp_path):
    root, _ = data_root(tmp_path)                                           # no heldout split exists at all
    pos = overview.overview("fake", "tiny", tmp_path / "ov.png", n_examples=3, root=root)
    assert list(pos) == [0, 30, 59] and (tmp_path / "ov.png").stat().st_size > 0


def test_nothing_about_a_data_set_is_hard_coded():
    """No data set name, config name, metric name or PRIMED number in analysis/: they come from the input files."""
    src = "\n".join(p.read_text() for p in sorted((paths.ROOT / "analysis").glob("*.py")))
    data = paths.ROOT / "data"
    names = {p.name for p in data.iterdir() if (p / "configs").is_dir()}
    names |= {p.stem for p in data.glob("*/configs/[!_]*.yaml")}
    names |= {m for p in data.glob("*/evaluate.py") for m in re.findall(r'"(\w+)":', p.read_text())} - {"n"}
    names |= {"PRIMED", "13.49", "10.13", "4.43", "2.806", "4.21", "photons", "err68", "DRO", "DCH"}
    assert names and not [n for n in sorted(names) if re.search(rf"\b{re.escape(n)}\b", src)]


def test_efficiency_refuses_an_unbuilt_data_set(tmp_path):
    """the dev rows are counted from the built data set's meta.json."""
    root = fakes.make_root(tmp_path)
    (root / "fake" / "configs" / "tiny.yaml").write_text(f"out: {tmp_path / 'never_built'}\nseed: 3\n")
    with pytest.raises(efficiency.EfficiencyError, match="is missing: build the data set"):
        efficiency._dev_meta("fake", "tiny", root)
