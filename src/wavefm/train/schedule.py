"""Learning-rate schedules, chosen by ``<stage>.schedule``. Each is a multiplier of the base rates of every group.

``constant`` the default: the base rate for the whole run (after the warm-up, if ``warmup_frac`` > 0).
``cosine``   advanced: cosine decay from the base rate at the first update after the warm-up to ``final_lr_frac`` of
             it at the LAST update. The updates are numbered 0..steps-1; the D = steps - warm-up updates after the
             warm-up get 0.5 * (1 + cos(pi * j / (D - 1))) mapped onto [final_lr_frac, 1], j = 0..D-1. Example:
             4 steps, no warm-up, final_lr_frac 0: 1, 0.75, 0.25, 0. A single update after the warm-up (D = 1) gets the
             base rate: there is nothing to decay over.
``plateau``  advanced, needs a validation loss and a fixed evaluation interval: a constant rate that is multiplied by
             ``lr_factor`` whenever the validation loss has not improved for ``patience`` evaluations; once it has
             been lowered ``max_reductions`` times, the next plateau ends the run (early stopping). ``steps`` /
             ``epochs`` is then only an upper bound.
A linear warm-up over the first W = round(warmup_frac * steps) updates (update i at (i + 1) / W of the rate)
multiplies any of them; off by default. W is at most steps - 1, so the warm-up never takes every update: at least one
update runs at the full rate (a run of one step has no warm-up).

Every schedule has ``factor(step)`` (the multiplier at the update that follows ``step`` completed steps),
``observe(val_loss)`` after each evaluation (lower is better) and ``done``; ``state_dict`` / ``load_state_dict``
hold what changes during a run, for an exact resume.
"""
import math


class ScheduleError(ValueError):
    pass


class Schedule:
    """constant and cosine: the factor is a function of the step alone, nothing is observed."""

    def __init__(self, kind: str, steps: int, warmup: int, final_frac: float = 0.0):
        self.kind, self.steps, self.warmup, self.final = kind, steps, warmup, final_frac
        self.done = False

    def _warm(self, step):
        return min(1.0, (step + 1) / self.warmup) if self.warmup else 1.0

    def factor(self, step):
        if self.kind == "constant":
            return self._warm(step)
        decay = self.steps - self.warmup                       # updates after the warm-up, >= 1
        t = min(max(step - self.warmup, 0) / (decay - 1), 1.0) if decay > 1 else 0.0
        return self._warm(step) * (self.final + (1 - self.final) * 0.5 * (1 + math.cos(math.pi * t)))

    def observe(self, value):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, s):
        pass


class Plateau(Schedule):
    def __init__(self, warmup: int, patience: int, lr_factor: float, max_reductions: int):
        if patience < 1 or not 0 < lr_factor < 1 or max_reductions < 0:
            raise ScheduleError(f"plateau: patience {patience} >= 1, 0 < lr_factor {lr_factor} < 1, max_reductions "
                                f"{max_reductions} >= 0 are needed")
        super().__init__("plateau", 0, warmup)
        self.patience, self.lr_factor, self.max_red = patience, lr_factor, max_reductions
        self.scale, self.best, self.bad, self.n_red = 1.0, math.inf, 0, 0

    def factor(self, step):
        return self.scale * self._warm(step)

    def observe(self, value):
        if value < self.best:
            self.best, self.bad = value, 0
            return
        self.bad += 1
        if self.bad < self.patience:
            return
        self.bad = 0
        if self.n_red < self.max_red:
            self.scale *= self.lr_factor; self.n_red += 1
        else:
            self.done = True

    def state_dict(self):
        return {k: getattr(self, k) for k in ("scale", "best", "bad", "n_red", "done")}

    def load_state_dict(self, s):
        for k, v in s.items():
            setattr(self, k, v)


def make(d: dict, steps: int) -> Schedule:
    """The schedule of a stage from its dynamics section ``d`` and its number of steps."""
    kind, warmup = d["schedule"], min(int(round(d["warmup_frac"] * steps)), max(steps - 1, 0))
    if not 0 <= d["warmup_frac"] < 1:
        raise ScheduleError(f"warmup_frac {d['warmup_frac']} must be in [0, 1)")
    if kind in ("constant", "cosine"):
        return Schedule(kind, steps, warmup, d["final_lr_frac"])
    if kind == "plateau":
        return Plateau(warmup, d["patience"], d["lr_factor"], d["max_reductions"])
    raise ScheduleError(f"schedule {kind!r}: known constant | cosine | plateau")
