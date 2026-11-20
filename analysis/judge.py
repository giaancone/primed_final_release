"""What a judge needs to give a verdict on the curves of a stage, and which stages still need one.

    pending(root)        finished stages under ``root`` (the output directory) that still need a verdict: not a smoke
                         run, not a stage without a curve (ridge: rundir.NO_CURVE), and no verdict.json made on the
                         present result.json (a verdict of an earlier result of the same directory no longer counts)
    prepare(run, stage)  draws the curves next to the stage (``<stage>/curves.png``) and returns its table row
    prepare_pending(root)   prepare for every pending stage
    show(rows)           the rows as fixed-width text

A row's ``verdict`` is whatever verdict.json says, also when it was made on an earlier result (then the stage is
pending again).

The verdict itself is written by rundir.write_verdict (scripts/judge.py), by a reviewer who looked at the
figure and applied analysis/JUDGING.md. Nothing here decides a verdict: the hints in the row are the numbers of
analysis.curves.hints, starting points for the eye, never a threshold.

``stage`` is a stage dict of analysis.read.runs; a row is one row of analysis.read.table plus the figure's path.
"""
import hashlib
from pathlib import Path

from analysis import curves, read
from wavefm.train import rundir

COLUMNS = ("group", "name", "stage", "seed", "label_frac", "requested_steps", "actual_steps", "final_train_loss",
           "final_val_loss", "hint_on", "hint_final_over_min", "hint_min_step", "hint_last10_drop", "verdict", "curves")


def needs_verdict(stage) -> bool:
    """True if this stage finished, is judged by its curves, is not a smoke run and has no verdict on this result."""
    res = stage["result"]
    if res is None or res["smoke"] or res["stage"] in rundir.NO_CURVE:
        return False
    v = stage["verdict"]
    sha = hashlib.sha256((Path(stage["path"]) / "result.json").read_bytes()).hexdigest()
    return v is None or v.get("result_sha256") != sha


def pending(root) -> list:
    """Stage dicts of analysis.read.runs, in group, name and pipeline order."""
    return [s for r in read.runs(root) for s in r["stages"].values() if needs_verdict(s)]


def prepare(run, stage_name) -> dict:
    """The figure of one stage of ``run`` (a run dict of analysis.read.runs) and its table row. A stage without a
    curve gets no figure (``curves`` None) and says so in its row."""
    stage = run["stages"][stage_name]
    row, = read.table([run], stage_name)
    row["curves"] = None
    if stage["history"] is not None:
        row["curves"] = str(Path(stage["path"]) / "curves.png")
        curves.plot(stage, row["curves"])
    return row


def prepare_pending(root) -> list:
    """Figures and rows of every stage that needs a verdict."""
    by_path = {Path(r["path"]): r for r in read.runs(root)}
    return [prepare(by_path[Path(s["path"]).parent], s["stage"]) for s in pending(root)]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "-" if v is None else str(v)


def show(rows, columns=COLUMNS) -> str:
    """Rows as an aligned text table (missing values as -)."""
    cells = [list(columns)] + [[_fmt(r.get(c)) for c in columns] for r in rows]
    width = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, width)).rstrip() for row in cells)

