"""The one score of a dro_gen data set, whatever the method.

    err68_ratio_pct  100 * p68(|Rp - Rt| / Rt), R = C / S
    err68_t0_ns      p68(|t0p - t0t|)
    n                number of events
    n_invalid        events whose ratio error is +inf

p68 is data.score.p68. Every event counts: a prediction with S <= 0 or a non-finite C or S has a ratio error of
+inf, a non-finite t0 a t0 error of +inf. They are never dropped, so a method cannot gain by failing on hard events;
the score is finite, or inf when 32 % or more of the events are invalid, never NaN. Truth rows with C <= 0 or S <= 0
cannot occur (build.py drops them) and raise.
"""
import numpy as np

from data.score import p68


class EvaluationError(ValueError):
    pass


def evaluate(pred, truth) -> dict:
    """pred, truth: (N, 3) arrays of C, S in photons and t0 in ns, physical units, rows in the same order."""
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    if p.shape != t.shape or t.ndim != 2 or t.shape[1] != 3 or len(t) == 0:
        raise EvaluationError(f"pred {p.shape} and truth {t.shape} must both be (N, 3) with N > 0: C, S, t0")
    if not np.isfinite(t).all() or (t[:, :2] <= 0).any():
        raise EvaluationError("truth must be finite with C > 0 and S > 0: score against the labels of a built split")
    r_true = t[:, 0] / t[:, 1]
    ok = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1]) & (p[:, 1] > 0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        r_pred = p[:, 0] / np.where(ok, p[:, 1], 1.0)
        e_ratio = np.where(ok, np.abs(r_pred - r_true) / r_true, np.inf)
        e_ratio = np.where(np.isfinite(e_ratio), e_ratio, np.inf)          # overflow of a huge ratio: +inf too
        e_t0 = np.where(np.isfinite(p[:, 2]), np.abs(p[:, 2] - t[:, 2]), np.inf)
    return {"err68_ratio_pct": 100 * p68(e_ratio), "err68_t0_ns": p68(e_t0), "n": len(t),
            "n_invalid": int(np.isinf(e_ratio).sum())}
