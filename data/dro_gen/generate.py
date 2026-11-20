"""Dual-readout crystal waveform generator: Cherenkov and scintillation photons of one SiPM channel, summed on the
digitisation grid and convolved with the measured single-photoelectron response.

Every parameter comes from the data config (``configs/_base.yaml`` and the file that inherits it): the crystal's
scintillation template, the Cherenkov template, the SiPM response, the photon source and the sensor effects.

One waveform (row) of chunk k, all draws from ``rng = numpy.random.default_rng([run.seed, k])``:
  1. numbers of Cherenkov and scintillation photons C, S and the per-event offset t0. Source ``random``: C, S
     uniform integers in 0..c_max, 0..s_max, then t0 uniform in +-t0.half_width_ns. Source ``root``: C and S are
     given (no draw); t0 is given, or drawn as for ``random``
  2. arrival times: Cherenkov ~ Exp(cherenkov.decay_ns); each scintillation photon from the first or the second
     exponential component of the crystal; all shifted by signal.onset_ns + t0
  3. dark counts: Poisson number over the window, uniform in time
  4. every photoelectron has amplitude Normal(1, spe_sigma); with probability ``crosstalk`` it fires one more cell
  5. photoelectrons are put on the sample grid with linear weights (sub-sample timing is kept), convolved with the
     response, and white noise is added

The labels are (C, S, t0): crosstalk cells and dark counts are not counted. The order of the draws and the chunk
size are part of the data set definition: chunks are independent, so the result does not depend on the number of
worker processes, and the ``random`` source reproduces the paper file PWO_real3_100k bit for bit.

Parallel mode uses ``spawn`` worker processes (safe next to threaded libraries). Spawned workers import the module of
the calling script: a script that calls ``generate(.., workers > 1)`` must do so under
``if __name__ == "__main__":``, or every worker starts the generation again.

Needs numpy only, nothing of ``wavefm``: the problem does not depend on a solution.
"""
import multiprocessing
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class GeneratorConfigError(ValueError):
    pass


def check(cfg: dict):
    """The generator parameters are present and physical; raises GeneratorConfigError saying which is not."""
    if cfg.get("crystal") not in cfg.get("crystals", {}):
        raise GeneratorConfigError(f"crystal {cfg.get('crystal')!r} has no entry in crystals "
                                   f"{sorted(cfg.get('crystals', {}))}: set crystal: in the data config")
    cr = cfg["crystals"][cfg["crystal"]]
    if cr["t1_ns"] <= 0 or cr["a1"] <= 0 or cr["t2_ns"] < 0 or cr["a2"] < 0:
        raise GeneratorConfigError(f"crystal {cfg['crystal']}: t1 and a1 must be positive, t2 and a2 non-negative")
    n = cfg["noise"]
    if min(n["dark_rate_hz"], n["spe_sigma"], n["electronic_spe"]) < 0 or not 0 <= n["crosstalk"] <= 1:
        raise GeneratorConfigError("noise settings must be non-negative and noise.crosstalk a probability")
    if cfg["run"]["n"] <= 0 or cfg["run"]["chunk"] <= 0 or cfg["t0"]["half_width_ns"] < 0:
        raise GeneratorConfigError("run.n and run.chunk must be positive, t0.half_width_ns non-negative")
    w = cfg["window"]
    if w["dt_ns"] <= 0 or w["t_last_ns"] <= w["t_first_ns"]:
        raise GeneratorConfigError("window: dt_ns must be positive and t_last_ns after t_first_ns")


def n_samples(cfg: dict) -> int:
    w = cfg["window"]
    return int((w["t_last_ns"] - w["t_first_ns"]) / w["dt_ns"])


def fast_fraction(cfg: dict) -> float:
    """Probability that a scintillation photon belongs to the first decay component."""
    c = cfg["crystals"][cfg["crystal"]]
    return c["a1"] * c["t1_ns"] / (c["a1"] * c["t1_ns"] + c["a2"] * c["t2_ns"])


def response(cfg: dict) -> np.ndarray:
    """The measured single-photoelectron response on the sample grid, peak-normalised to sipm.spe_amplitude.
    Outside the template its first / last value is held."""
    r, dt = cfg["sipm"]["response"], cfg["window"]["dt_ns"]
    x = r["t_start_ns"] + np.arange(r["n_samples"]) * dt
    t, a = np.loadtxt(HERE / r["file"], delimiter=",", comments="#", skiprows=3, unpack=True)
    a = a / a.max() * cfg["sipm"]["spe_amplitude"]
    keep = slice(None, -r["drop_last"]) if r["drop_last"] else slice(None)
    return np.interp(x, t[keep], a[keep])


def generate_chunk(cfg: dict, k: int, n: int, given=None):
    """Chunk k of n rows: (waveform (n, L) float32, labels (n, 3) float32 = C, S, t0).

    ``given``: None for the random source, else an (n, 2) or (n, 3) array of C, S [, t0] for these rows."""
    w, cr, ns_ = cfg["window"], cfg["crystals"][cfg["crystal"]], cfg["noise"]
    lo, hi, dt, L = w["t_first_ns"], w["t_last_ns"], w["dt_ns"], n_samples(cfg)
    spe, f1, onset = cfg["sipm"]["spe_amplitude"], fast_fraction(cfg), cfg["signal"]["onset_ns"]
    spr = response(cfg)
    nfft = 1 << int(np.ceil(np.log2(L + len(spr))))
    spr_f = np.fft.rfft(spr, nfft)
    half = cfg["t0"]["half_width_ns"]

    rng = np.random.default_rng([cfg["run"]["seed"], k])
    W, Y = np.empty((n, L), np.float32), np.empty((n, 3), np.float32)
    for i in range(n):
        # the order of the draws is part of the data set definition: do not reorder
        if given is None:
            nc, ns = int(rng.integers(0, cfg["source"]["c_max"] + 1)), int(rng.integers(0, cfg["source"]["s_max"] + 1))
        else:
            nc, ns = int(given[i, 0]), int(given[i, 1])
        t0 = rng.uniform(-half, half) if given is None or given.shape[1] == 2 else float(given[i, 2])
        n_fast = rng.binomial(ns, f1)
        nd = rng.poisson(ns_["dark_rate_hz"] * (hi - lo) * 1e-9)
        t_c = rng.exponential(cfg["cherenkov"]["decay_ns"], nc) + onset + t0
        t_fast = rng.exponential(cr["t1_ns"], n_fast) + onset + t0
        t_slow = (rng.exponential(cr["t2_ns"], ns - n_fast) if cr["t2_ns"] > 0 else np.zeros(0)) + onset + t0
        t = np.concatenate([t_c, t_fast, t_slow, rng.uniform(lo, hi, nd)])
        xtalk = rng.random(len(t)) < ns_["crosstalk"]
        amp = rng.normal(1.0, ns_["spe_sigma"], len(t)) + xtalk * rng.normal(1.0, ns_["spe_sigma"], len(t))
        q = (t - lo) / dt
        inside = (q >= 0) & (q < L - 1)
        i0 = np.floor(q[inside]).astype(np.int64)
        frac = q[inside] - i0
        hits = np.bincount(i0, amp[inside] * (1 - frac), L) + np.bincount(i0 + 1, amp[inside] * frac, L)
        W[i] = np.fft.irfft(np.fft.rfft(hits, nfft) * spr_f, nfft)[:L] + rng.normal(0, ns_["electronic_spe"] * spe, L)
        Y[i] = (nc, ns, t0)
    return W, Y


def _work(job):
    """One chunk in a worker process. Module level, so that spawned workers can import it."""
    cfg, k, n, given = job
    return (k, *generate_chunk(cfg, k, n, given))


def generate(cfg: dict, given=None, workers: int = 1):
    """All run.n rows of ``cfg`` in source order: (waveform (n, L) float32, labels (n, 3) float32).

    ``given``: None (random source) or the (n, 2|3) array of C, S [, t0] of every row. With workers > 1 the caller
    must run under ``if __name__ == "__main__":`` (see the module docstring)."""
    check(cfg)
    n, chunk = cfg["run"]["n"], cfg["run"]["chunk"]
    jobs = [(cfg, k, min(chunk, n - k * chunk), None if given is None else given[k * chunk:(k + 1) * chunk])
            for k in range((n + chunk - 1) // chunk)]
    W, Y = np.empty((n, n_samples(cfg)), np.float32), np.empty((n, 3), np.float32)

    def store(res):
        k, w, y = res
        W[k * chunk:k * chunk + len(w)], Y[k * chunk:k * chunk + len(w)] = w, y

    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for res in pool.imap_unordered(_work, jobs):
                store(res)
    else:
        for job in jobs:
            store(_work(job))
    return W, Y
