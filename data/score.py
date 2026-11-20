"""The one percentile every data set scores with.

err68 is the 68th percentile of absolute errors, with numpy's default rule: linear interpolation between the two
order statistics around position 0.68 (n - 1) of the sorted errors. An invalid prediction (not finite, or a ratio
with a non-positive denominator) is an error of +inf: it is counted, never dropped, so a method cannot improve its
score by failing on hard events. The result is finite when the order statistics it needs are finite, else inf (for
large n: when about 32 % or more of the errors are invalid). It is never NaN.
"""
import numpy as np


def p68(err) -> float:
    e = np.asarray(err, np.float64).ravel()
    if e.size == 0:
        raise ValueError("p68 of no events")
    e = np.sort(np.where(np.isfinite(e), e, np.inf))
    lo, rem = divmod(68 * (e.size - 1), 100)        # the position in integers: 0.68 * 75 is not exactly 51 in floats
    if rem == 0:
        return float(e[lo])
    if np.isinf(e[lo + 1]):          # inf - finite is fine, but inf * 0 and inf - inf are NaN: never interpolate
        return float("inf")
    return float(e[lo] + rem / 100 * (e[lo + 1] - e[lo]))
