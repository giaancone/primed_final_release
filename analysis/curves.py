"""The curves of a stage: a few indicative numbers, and the figure the judge looks at.

    hints(history)          numbers that point the judge at what to look for. They are HINTS, never a gate: a run is
                            not passed or failed on them (plan: "指标只是提示，不是硬门槛"); the judge decides from the
                            whole curve and writes verdict.json (rundir.write_verdict).
    plot(stage, path)       training loss, validation loss (and the student's val_truth_loss) and learning rate
                            against step for one stage; the phases of a compress history are marked
    plot_many(stages, path) the losses of several stages on one figure (seeds, label fractions, scan points)

``stage`` is a stage dict of analysis.read.runs; ``history`` its dict of arrays (or the raw rows of history.json).
Figures are PNG, written to a temporary name and renamed, so a half-written figure never has the final name.
"""
import os
from pathlib import Path

import numpy as np


class CurveError(ValueError):
    pass


def _arrays(history) -> dict:
    if isinstance(history, list):
        from analysis.read import history_arrays
        return history_arrays(history)
    return history


def hints(history) -> dict:
    """Indicative numbers of one curve, never a gate.

    The curve is the validation loss if the stage has one, else the training loss (ssl). Points where it is not
    finite (steps without an evaluation) are left out. A history with a ``phase`` column (compress: pruning, then
    quantisation-aware training, one step count across both) gives the hints of its LAST phase: the phases are
    separate trainings, and a minimum of one is not the other's.
        on              "val_loss" or "train_loss": the curve the numbers are about
        final, min      its last and its lowest value; min_step the step of the lowest value
        final_over_min  (final - min) / min: how far the end is above the best point (overfitting shows here)
        last10_drop     (v - final) / v, v = the value at the last point at or before 90 % of the way from the first
                        to the last step: how much it still fell over the last 10 % of the steps (under-training)
    """
    h = _arrays(history)
    if "phase" in h:
        last = h["phase"] == h["phase"][-1]
        h = {k: v[last] for k, v in h.items()}
    on = "val_loss" if "val_loss" in h and np.isfinite(h["val_loss"]).any() else "train_loss"
    ok = np.isfinite(h[on])
    step, y = h["step"][ok], h[on][ok]
    if len(y) == 0:
        raise CurveError(f"history has no finite {on}: nothing to give hints on")
    i = int(np.argmin(y))
    j = int(np.flatnonzero(step <= step[-1] - 0.1 * (step[-1] - step[0]))[-1]) if len(y) > 1 else 0
    return {"on": on, "final": float(y[-1]), "min": float(y[i]), "min_step": int(step[i]),
            "final_over_min": float((y[-1] - y[i]) / abs(y[i])), "last10_drop": float((y[j] - y[-1]) / abs(y[j]))}


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    fig.savefig(tmp, format="png", dpi=120)
    os.replace(tmp, path)


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _losses(ax, h, color=None, label=""):
    """Draw train (thin, faint) and validation (solid) loss; log-y when every value drawn is positive."""
    lines = []
    for key, style in (("train_loss", dict(lw=1, alpha=0.5)), ("val_loss", dict(lw=2, marker="o", ms=3)),
                       ("val_truth_loss", dict(lw=1.5, ls="--", marker="x", ms=3))):
        if key in h and np.isfinite(h[key]).any():
            ok = np.isfinite(h[key])
            lines += ax.plot(h["step"][ok], h[key][ok], color=color, label=f"{label}{key}", **style)
    if lines and all((ln.get_ydata() > 0).all() for ln in lines):
        ax.set_yscale("log")
    return lines


def _length(res) -> str:
    if not res:
        return "unfinished: no result.json"
    fmt = lambda d: ", ".join(f"{k} {v:g}" for k, v in d.items())
    return f"requested: {fmt(res['requested'])}\nactual: {fmt(res['actual'])}"


def plot(stage, path):
    """One stage: losses on the left axis, learning rate on the right, hints and length written into the figure."""
    h = stage["history"]
    if h is None:
        raise CurveError(f"{stage['path']}: stage {stage['stage']} has no training curve (ridge: read its loo.json)")
    h, plt = _arrays(h), _figure()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lines = _losses(ax, h)
    if "lr" in h and np.isfinite(h["lr"]).any():
        ax2 = ax.twinx()
        lines += ax2.plot(h["step"], h["lr"], color="0.5", lw=1, ls=":", label="lr")
        ax2.set_ylabel("learning rate")
    if "phase" in h:                                  # compress: where each phase starts
        for i in [0, *np.flatnonzero(h["phase"][1:] != h["phase"][:-1]) + 1]:
            ax.axvline(h["step"][i], color="0.6", ls="--", lw=1)
            ax.text(h["step"][i], 1.0, f" {h['phase'][i]}", transform=ax.get_xaxis_transform(), fontsize=8, va="top")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", frameon=False)
    hi = hints(h)
    text = (f"{_length(stage['result'])}\nhints on {hi['on']}{' of the last phase' if 'phase' in h else ''} "
            f"(not a gate): final / min - 1 = "
            f"{hi['final_over_min']:.3g}, min at step {hi['min_step']}, drop over last 10 % = {hi['last10_drop']:.3g}")
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_title(f"{stage['group']}/{stage['name']}  {stage['stage']}", fontsize=11)
    fig.tight_layout(); _save(fig, path); plt.close(fig)


def plot_many(stages, path, label=None):
    """The losses of several stages on one axis, one colour per stage. ``label(stage) -> str`` names each one
    (default group/name)."""
    plt = _figure()
    label = label or (lambda s: f"{s['group']}/{s['name']}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, s in enumerate(x for x in stages if x["history"] is not None):
        _losses(ax, _arrays(s["history"]), color=f"C{k % 10}", label=f"{label(s)} ")
    ax.set_xlabel("step"); ax.set_ylabel("loss (thin: train, solid: validation)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); _save(fig, path); plt.close(fig)
