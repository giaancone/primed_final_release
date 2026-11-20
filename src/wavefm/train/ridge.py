"""The ridge stage: closed-form float64 ridge on frozen encoder features, the teacher's head. ``run(cfg, out, parent)``.

Mathematics (``fit``, numpy float64, no torch):
  * features are standardised, ``z = (f - mean) / scale``, with mean and spread of the rows of ``ridge.standardise_on``
    (``labelled``, or all ``dev`` rows: features need no labels). A feature that is constant on those rows (minimum
    equal to maximum) gets ``mean`` = that value and ``scale`` = 1; one that is constant on the fitted rows carries
    no information and is left out of the decomposition: its coefficient is exactly 0, never a NaN.
  * per output, ``min_{b, w} sum_i (y_i - b - z_i w)^2 + alpha |w|^2``: the intercept b is not penalised.
    Alpha convention: alpha weighs |w|^2 of the STANDARDISED features against the SUM (not the mean) of squared
    residuals over the labelled rows; it is not relative to n. So the fit and the choice of alpha are invariant
    under any per-feature affine change of the features, and change with n.
  * no target scaling: with an unpenalised intercept and alpha chosen per output, the fit is equivariant under an
    affine map ``y -> c y + m`` of each target (predictions map the same way, the leave-one-out table is multiplied
    by c^2, the chosen alpha is the same). Predictions are in the units of the data set.
  * alpha is chosen per output by the closed-form leave-one-out squared error over the whole grid, from ONE
    eigendecomposition of the centred Gram matrix: Z^T Z (d x d) if n > d ("primal"), Z Z^T (n x n) otherwise
    ("dual"), so n < d is exact and finite. Leave-one-out residual e_i / (1 - h_ii), with h_ii = 1/n (the intercept)
    + sum_k U_ik^2 lam_k / (lam_k + alpha). The standardisation is held fixed when a row is left out. Eigenvalues
    below the numerical rank tolerance (as numpy.linalg.matrix_rank) are dropped, which is the minimum-norm solution
    at alpha = 0. There an interpolating fit (n <= rank + 1: no residual degree of freedom) gets an infinite
    leave-one-out error by rule, not from a division by a 1 - h_ii that round-off leaves slightly above 0.
  * the grid is 10 ** (ridge.log10_alpha_min + i / ridge.alphas_per_decade) up to ridge.log10_alpha_max. A chosen
    alpha at the first or last grid point is flagged ("edge"): the optimum may lie outside the grid.

Stage (``run``): features are the mean over all tokens of block ``ridge.block`` of the encoder of the parent stage
(started from the file the parent hands over, pipeline.handover), computed in strict fp32 (``run`` and ``predict`` set
it, utils.runtime.setup_numerics; guards.check_strict_fp32 refuses otherwise) and accumulated in float64; no feature
cache. ``model.scaling: global_minmax`` scales the waveforms with the minimum and maximum of dev (meta.json), saved in
ridge.npz and used for every later prediction, whatever split is predicted. Without a parent, the encoder's own start
is recorded (``encoder_from`` in result.json and ridge.npz): the sha256 of the pretrained file that was loaded
(backbones.pretrained_file; ``predict`` refuses if the file now resolved differs), or for ``blocks: random`` the seed.
The fit uses every labelled row (train and validation of the label budget together: leave-one-out needs no validation
split) and reads no other label.
Files in ``out``: ridge.npz (everything ``predict`` needs), loo.json (the leave-one-out table; this stage has no
training curve), teacher_labels.npz (prediction for every dev row, rundir.write_teacher_labels), result.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import wavefm.data as wdata
from wavefm.data import splits
from wavefm.data.batches import pad_to_patch
from wavefm.train import guards, pipeline, rundir
from wavefm.utils import runtime
from wavefm.utils.io import save_json, save_npz, load_json


class RidgeError(RuntimeError):
    pass


# ---- mathematics ----------------------------------------------------------------------------------------------------

def alpha_grid(r: dict) -> np.ndarray:
    """The alphas of the ``ridge`` config section, ascending."""
    span = (r["log10_alpha_max"] - r["log10_alpha_min"]) * r["alphas_per_decade"]
    if r["alphas_per_decade"] <= 0 or span < 0 or abs(span - round(span)) > 1e-9:
        raise RidgeError(f"ridge alpha grid: log10 {r['log10_alpha_min']} .. {r['log10_alpha_max']} must be a whole "
                         f"number of steps of 1/{r['alphas_per_decade']} decade")
    return 10.0 ** (r["log10_alpha_min"] + np.arange(int(round(span)) + 1) / r["alphas_per_decade"])


def standardisation(F) -> tuple:
    """(mean, scale) of the columns of F; a constant column gets its value and 1."""
    F = np.asarray(F, np.float64)
    lo, hi = F.min(0), F.max(0)
    const = lo == hi
    mean = np.where(const, lo, F.mean(0))
    scale = np.where(const, 1.0, F.std(0))
    return mean, scale


def _eigen(Zc):
    """(U, lam, V) of the centred Zc = U diag(sqrt(lam)) V^T from one symmetric eigendecomposition, on the smaller
    side; components below the rank tolerance are dropped."""
    n, d = Zc.shape
    solver = "primal" if n > d else "dual"
    if d == 0:
        return np.zeros((n, 0)), np.zeros(0), np.zeros((0, 0)), solver
    lam, E = np.linalg.eigh(Zc.T @ Zc if solver == "primal" else Zc @ Zc.T)
    keep = lam > (lam.max(initial=0.0) * max(n, d) * np.finfo(np.float64).eps)
    lam, E = lam[keep], E[:, keep]
    s = np.sqrt(lam)
    if solver == "primal":
        return (Zc @ E) / s, lam, E, solver
    return E, lam, (Zc.T @ E) / s, solver


def fit(F, Y, alphas, F_standardise=None) -> dict:
    """Ridge of every column of Y (n, k) on the features F (n, d), alpha chosen per output by leave-one-out.

    ``F_standardise``: the rows that give the feature mean and spread (default F itself).
    Returns, all float64: ``coef`` (d, k) and ``intercept`` (k,) such that the prediction is ``F @ coef + intercept``
    in the units of F and Y; ``mean``, ``scale`` (d,) of the standardisation (coef = w / scale, with w the coefficients
    of the standardised features); ``alphas`` the grid; ``loo`` (len(alphas), k) mean squared leave-one-out error;
    ``index`` and ``alpha`` (k,) the choice; ``edge`` per output None, "min" or "max"; ``rank`` and ``solver``."""
    F, Y, alphas = np.asarray(F, np.float64), np.asarray(Y, np.float64), np.asarray(alphas, np.float64)
    n = len(F)
    if F.ndim != 2 or Y.ndim != 2 or len(Y) != n or n < 3:
        raise RidgeError(f"ridge needs features (n, d) and targets (n, k) with n >= 3, got {F.shape} and {Y.shape}")
    if not (np.isfinite(F).all() and np.isfinite(Y).all()):
        raise RidgeError("non-finite features or targets in the fitted rows")
    if alphas.ndim != 1 or not len(alphas) or (alphas < 0).any() or (np.diff(alphas) <= 0).any():
        raise RidgeError(f"alphas must be non-negative and strictly ascending, got {alphas}")
    mean, scale = standardisation(F if F_standardise is None else F_standardise)
    Z = (F - mean) / scale
    active = Z.min(0) < Z.max(0)                                      # constant on the fitted rows: coefficient 0
    ybar = Y.mean(0)
    Zc, Yc = Z[:, active] - Z[:, active].mean(0), Y - ybar
    U, lam, V, solver = _eigen(Zc)

    # residual and 1 - h_ii written with alpha / (lam + alpha), so that neither cancels when the fit nearly
    # interpolates (n < d, small alpha): e = (I - U U^T) Yc + U diag(alpha / (lam + alpha)) U^T Yc,
    # 1 - h_ii = (1 - 1/n - sum_k U_ik^2) + sum_k U_ik^2 alpha / (lam_k + alpha)
    UtY, U2 = U.T @ Yc, U * U                                         # (r, k), (n, r)
    keep = alphas[None] / (lam[:, None] + alphas[None])               # (r, A); lam > 0, so no 0 / 0
    rest = Yc - U @ UtY                                               # (n, k) outside the span of the features
    A, k = len(alphas), Y.shape[1]
    e = rest[:, None, :] + (U @ (keep[:, :, None] * UtY[:, None, :]).reshape(len(lam), A * k)).reshape(n, A, k)
    one_minus_h = np.maximum(1.0 - 1.0 / n - U2.sum(1), 0.0)[:, None] + U2 @ keep          # (n, A)
    with np.errstate(divide="ignore", invalid="ignore"):              # alpha = 0 and an interpolating fit: inf
        loo = ((e / one_minus_h[:, :, None]) ** 2).mean(0)            # (A, k)
    loo = np.where(np.isnan(loo), np.inf, loo)
    if len(lam) >= n - 1:                                             # interpolating at alpha = 0: inf by rule
        loo[alphas == 0] = np.inf

    index = np.argmin(loo, axis=0)
    a = alphas[index]
    w = np.zeros((F.shape[1], k))                            # (d, k) on standardised features
    w[active] = V @ (np.sqrt(lam)[:, None] / (lam[:, None] + a[None]) * UtY)
    zbar = Z.mean(0)
    edge = ["min" if i == 0 else "max" if i == len(alphas) - 1 else None for i in index]
    return {"coef": w / scale[:, None], "intercept": ybar - (mean / scale + zbar) @ w, "mean": mean, "scale": scale,
            "alphas": alphas, "loo": loo, "index": index, "alpha": a, "edge": edge, "rank": int(len(lam)),
            "solver": solver}


# ---- features -------------------------------------------------------------------------------------------------------

def encoder(cfg: dict, init) -> torch.nn.Module:
    """The encoder started from the file ``init`` the parent stage handed over (or the one model.* builds, for a first
    stage); random parts from cfg["seed"]."""
    from wavefm.models import build
    return build.build_encoder(cfg["model"], init_from=init, seed=cfg["seed"])


def encoder_from(cfg: dict, init) -> dict | None:
    """How a first stage's encoder starts: the sha256 of the pretrained file, or the seed of the random control."""
    if init is not None:
        return None
    if cfg["model"]["blocks"] == "random":
        return {"seed": int(cfg["seed"])}
    from wavefm.models import backbones
    return {"pretrained_sha256": rundir.file_sha256(backbones.pretrained_file(str(cfg["model"]["version"])))}


def input_range(cfg: dict, dev) -> tuple | None:
    """(lo, hi) of model.scaling global_minmax: the minimum and maximum of the dev waveforms; None otherwise."""
    if cfg["model"]["scaling"] != "global_minmax":
        return None
    st = dev.meta["waveform"]["stats"]
    return float(st["min"]), float(st["max"])


def features(enc, split, cfg: dict, lo_hi) -> np.ndarray:
    """(N, D) float64: token mean of block ridge.block for every row of ``split``, in batches of ridge.batch_size.
    ``lo_hi``: (lo, hi) of global_minmax, those of dev (input_range), never of ``split``; None for no scaling."""
    device = next(enc.parameters()).device
    enc.eval()
    guards.check_strict_fp32(enc)
    lo, hi = lo_hi if lo_hi is not None else (None, None)
    bs, out = cfg["ridge"]["batch_size"], []
    with torch.no_grad():
        for a in range(0, len(split), bs):
            x = torch.from_numpy(split.take(np.arange(a, min(a + bs, len(split)))))
            if lo is not None:
                x = (x - lo) / (hi - lo)
            t = enc(pad_to_patch(x).to(device), block=cfg["ridge"]["block"])
            if t.dtype != torch.float32 or t.ndim != 3 or len(t) != len(x):
                raise RidgeError(f"encoder returned {tuple(t.shape)} {t.dtype}, expected ({len(x)}, N, D) float32")
            out.append(t.to(torch.float64).mean(1).cpu().numpy())
    F = np.concatenate(out)
    if not np.isfinite(F).all():
        raise RidgeError("non-finite encoder features")
    return F


# ---- the stage ------------------------------------------------------------------------------------------------------

def labelled_rows(cfg: dict, n: int) -> np.ndarray:
    """Every labelled position of dev (train and validation of the budget), sorted."""
    d = cfg["data"]
    train, val = splits.labelled(n, d["label_frac"], d["val_frac"], cfg["seed"], d["min_labelled"], d["min_val"])
    return np.union1d(train, val)


def run(cfg: dict, out, parent) -> dict:
    runtime.setup_numerics()                       # strict fp32 before anything else runs
    t0, out = time.time(), Path(out)
    dev = wdata.dev(cfg)
    init = pipeline.handover(parent)
    prec = rundir.parent_record(init)
    done = rundir.check_identity(out, cfg, dev.meta["data_hash"], prec)
    if done is not None:
        return done
    r = cfg["ridge"]
    if r["standardise_on"] not in ("labelled", "dev"):
        raise RidgeError(f"ridge.standardise_on is labelled or dev, got {r['standardise_on']!r}")
    lab = labelled_rows(cfg, len(dev))
    start = encoder_from(cfg, init)
    lo_hi = input_range(cfg, dev)
    enc = encoder(cfg, init).to("cuda" if torch.cuda.is_available() else "cpu")
    F = features(enc, dev, cfg, lo_hi)
    Y = np.asarray(dev.labels[lab], np.float64)                       # the only labels this stage reads
    m = fit(F[lab], Y, alpha_grid(r), F_standardise=F if r["standardise_on"] == "dev" else None)

    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "ridge.npz", coef=m["coef"], intercept=m["intercept"], mean=m["mean"], scale=m["scale"],
             alpha=m["alpha"], block=np.int64(r["block"]), input_lo_hi=np.asarray(lo_hi or [], np.float64),
             encoder_from=json.dumps(start))
    names = [s["name"] for s in dev.meta["labels"]]
    var = Y.var(0)
    loo_rel = [float(m["loo"][i, k] / var[k]) if var[k] > 0 else None for k, i in enumerate(m["index"])]
    save_json({"outputs": names, "alphas": m["alphas"].tolist(), "log10_alpha": np.log10(m["alphas"]).tolist(),
               "loo_mse": m["loo"].tolist(), "chosen_alpha": m["alpha"].tolist(), "chosen_index": m["index"].tolist(),
               "edge": m["edge"], "loo_over_target_var": loo_rel, "rows": len(lab), "rank": m["rank"],
               "solver": m["solver"], "standardise_on": r["standardise_on"]}, out / "loo.json")
    teacher = {"config_fp": rundir.config_fp(cfg, "ridge"), "data_hash": dev.meta["data_hash"], "parent": prec}
    rundir.write_teacher_labels(out / "teacher_labels.npz", dev, F @ m["coef"] + m["intercept"], teacher)
    rows = {"rows": int(len(lab))}
    return rundir.write_result(out, "ridge", cfg, dev.meta["data_hash"], prec, requested=rows, actual=rows,
                               hints={"edge": m["edge"], "loo_over_target_var": loo_rel},
                               wall_s=time.time() - t0, artifacts=("ridge.npz", "loo.json", "teacher_labels.npz"),
                               alpha=m["alpha"].tolist(), n_features=int(F.shape[1]), encoder_from=start)


def predict(out, split) -> np.ndarray:
    """(N, k) float64 predictions of the finished ridge stage ``out`` for every row of ``split``, physical units.
    The encoder is rebuilt from the run's config.json (one directory up) and the parent weights in result.json; the
    input scaling is the one saved in ridge.npz."""
    runtime.setup_numerics()
    out = Path(out)
    cfg, res = load_json(out.parent / "config.json"), rundir.read_result(out)
    parent = res["parent"]
    if parent is not None and rundir.file_sha256(parent["file"]) != parent["sha256"]:
        raise RidgeError(f"{parent['file']} changed after the ridge stage {out} was fitted: refit it, its head belongs "
                         f"to the old encoder")
    if rundir.config_fp(cfg, "ridge") != res["config_fp"]:
        raise RidgeError(f"{out.parent}/config.json is not the config the ridge stage {out} was fitted with")
    guards.check_artifact(out, res, "ridge.npz")
    with np.load(out / "ridge.npz") as z:
        coef, intercept, block, lo_hi = z["coef"], z["intercept"], int(z["block"]), tuple(z["input_lo_hi"].tolist())
        start = json.loads(str(z["encoder_from"]))
    if block != cfg["ridge"]["block"]:
        raise RidgeError(f"{out}/ridge.npz was fitted on block {block}, the config says {cfg['ridge']['block']}")
    init = None if parent is None else Path(parent["file"])
    now = encoder_from(cfg, init)
    if start != now:
        raise RidgeError(f"the ridge stage {out} was fitted on an encoder started from {start}, the pretrained file "
                         f"now resolved gives {now}: refit the stage")
    enc = encoder(cfg, init)
    F = features(enc.to("cuda" if torch.cuda.is_available() else "cpu"), split, cfg, lo_hi or None)
    if F.shape[1] != len(coef):
        raise RidgeError(f"encoder gives {F.shape[1]} features, {out}/ridge.npz has {len(coef)} coefficients")
    return F @ coef + intercept
