"""The rows of dev whose labels a run may use, and their cut into training and validation.

The data set fixes dev and heldout; a run only chooses inside dev. The label budget is ``label_frac`` of dev and it
includes the validation rows, so a fraction counts every truth label the run used:

    perm = numpy.random.default_rng(seed).permutation(n)
    labelled   = perm[:round(label_frac * n)]                              (label_frac == 1: every row)
    validation = labelled[:max(round(val_frac * len(labelled)), min_val)]
    train      = the rest of labelled

The minimums are feasibility checks, never a silent enlargement of the budget: fewer than ``min_labelled`` labelled
rows is an error (BudgetError), and so is a validation set that leaves no training rows.

Everything follows the run seed: another seed is another draw of labelled rows and another cut. The budgets are
nested (a smaller fraction is a prefix of the same permutation), which makes data-efficiency curves smoother.
Positions are returned sorted, so reading them from disk is sequential.
"""
import numpy as np


class SplitError(ValueError):
    pass


class BudgetError(SplitError):
    pass


def labelled(n, label_frac, val_frac, seed, min_labelled, min_val):
    """(train, val): sorted, disjoint int64 positions in a dev split of n rows."""
    for name, f in (("label_frac", label_frac), ("val_frac", val_frac)):
        if not 0 < f <= 1:
            raise SplitError(f"data.{name} = {f} is outside (0, 1]")
    n_lab = n if label_frac == 1 else int(round(label_frac * n))
    if n_lab < min_labelled:
        raise BudgetError(f"label budget {label_frac} of {n} dev rows is {n_lab} rows, fewer than data.min_labelled = "
                          f"{min_labelled}: too small for this data set; raise data.label_frac or lower data.min_labelled")
    n_val = max(int(round(val_frac * n_lab)), min_val)
    if n_val >= n_lab or n_val < 1:
        raise BudgetError(f"{n_lab} labelled rows with {n_val} for validation leave no training rows: raise "
                          f"data.label_frac or lower data.val_frac / data.min_val")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:n_lab]), np.sort(perm[:n_val])
