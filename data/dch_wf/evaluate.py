"""The dch_wf metric: mean absolute and root-mean-square error of the predicted number of primary clusters.

    count_mae  = mean |pred - truth|            count_rmse = sqrt(mean (pred - truth)^2)      in clusters, float64
A prediction that is not finite is an error of +inf: it is counted, never dropped, so both metrics become inf (never
NaN) and a method cannot improve its score by failing on hard events.
"""
import numpy as np


class ShapeError(ValueError):
    pass


def evaluate(pred, truth) -> dict:
    """``pred`` and ``truth``: (N,) or (N, 1), the same shape, in clusters."""
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    if p.shape != t.shape or p.ndim not in (1, 2) or (p.ndim == 2 and p.shape[1] != 1) or p.size == 0:
        raise ShapeError(f"pred {p.shape} and truth {t.shape} must be the same non-empty (N,) or (N, 1): "
                         f"one predicted count per event")
    if not np.isfinite(t).all():
        raise ValueError("truth is not finite: it must be the labels of the split, not a prediction")
    err = np.where(np.isfinite(p), np.abs(p - t), np.inf).ravel()
    return {"count_mae": float(err.mean()), "count_rmse": float(np.sqrt((err * err).mean())), "n": int(err.size)}
