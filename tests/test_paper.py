"""The paper configs: every config file is valid; the encoder and the SSL terms reproduce numbers of the old paper code;
and the repo's own teacher smoke test (ssl + ridge on the small DRO set).

Regression values (``EXPECTED``). They were computed once, on 2026-11-20, on an A100 in strict fp32, from the inputs
below (``synth``, ``synth_mask``, ``synth_token``: seeded, no data set needed) with the pretrained TimesFM 2.5 weights,
the adapter as built (last convolution zero) and the SSL heads of ``ssl.build`` for seed 0, by BOTH the new code and
the old paper scripts (verbatim copies of work_dch1pct/rebuild_teacher.py ``Backbone(norm="gmm")`` + work_dro/
fm_adapters.py + the loss lines of work_dro/ssl_adapt2.py for DRO; of dch_norm_aux_v2/src/dch_ssl.py ``Encoder``
(legacy_double_revin) + its loss lines for DCH), with the same heads, mask token and mask given to both. That
one-off check, and its result on real rows and with a trained-looking adapter, is summarised in configs/paper/README.md:
old and new agree to < 1e-5 relative. The numbers written here are the new code's; the old code's agree with them to
the relative difference written next to EXPECTED. A change of the encoder, the scalings, the SSL views or
terms that moves any of them by more than 1e-5 (relative) breaks the reproduction of the paper method.
"""
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from data import store
from wavefm import paths
from wavefm.train import pipeline, rundir, ssl
from wavefm.utils import config, runtime
from wavefm.utils.io import load_json
import wavefm.data as wdata

CONFIG_DIRS = [paths.ROOT / "configs" / "paper", paths.ROOT / "configs" / "smoke"]
SMOKE_DATA = Path(store.load_config("dro_gen", "pwo_smoke")["out"])
# fixed scaling constants of the synthetic DRO batch: the old paper constants (pwo_consts.json, the SSL sidecar)
CONST = {"lo": -305.341796875, "hi": 9083.935546875, "mu": 332.269775390625, "sd": 1112.33349609375}
LENGTH = {"dro": 6272, "dch": 3008}
TOL = 1e-5

# new code, 2026-11-20 (A100, torch 2.9.1, timesfm 3.0.2); the old code on the same inputs (configs/paper/README.md):
# DRO identical in every digit; DCH within 3.1e-8 (mean), 3.2e-8 (abs_mean), 0 (e000), 1.3e-7 (e1_50_100),
# 6.3e-8 (absmax), 1.1e-7 (next), 0 (masked) relative
EXPECTED = {
    "dro": {"mean": 0.0031292111740249917, "abs_mean": 0.01566859352552946, "e000": -0.0031147003173828125,
            "e1_50_100": 0.009909715503454208, "absmax": 3.3842244148254395, "next": 0.10818897187709808,
            "masked": 0.15153934061527252},
    "dch": {"mean": 0.009780487354924134, "abs_mean": 0.14596907842927753, "e000": -0.21723413467407227,
            "e1_50_100": -0.117253378033638, "absmax": 122.05574798583984, "next": 1.127365231513977,
            "masked": 0.6603513360023499},
}


def synth(task: str) -> torch.Tensor:
    """(4, L) float32 raw waveforms: a baseline with white noise and one exponential pulse per trace, of four sizes."""
    g = torch.Generator().manual_seed(1234)
    L = LENGTH[task]
    t = torch.arange(L, dtype=torch.float64)
    x = []
    for b in range(4):
        t0, amp = 600.0 + 400.0 * b, (500.0, 1500.0, 3000.0, 6000.0)[b]
        pulse = torch.where(t >= t0, torch.exp(-(t - t0) / 30.0), torch.zeros_like(t))
        noise = torch.randn(L, generator=g, dtype=torch.float64)
        x.append(amp * pulse + 3.0 * noise if task == "dro" else amp / 1000.0 * pulse + 0.05 * noise + 0.2)
    return torch.stack(x).float()


def synth_mask(task: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(5)
    return torch.rand(4, LENGTH[task] // 32, generator=g) < 0.25


def synth_token() -> torch.Tensor:
    g = torch.Generator().manual_seed(6)
    return 0.1 * torch.randn(32, generator=g)


def regression_values(task: str) -> dict:
    """The new code's numbers on the synthetic inputs (see the module docstring)."""
    runtime.setup_numerics()
    cfg = config.load(paths.ROOT / "configs" / "paper" / f"{task}.yaml")
    model, _, _ = ssl.build(cfg)
    dev = torch.device("cuda")
    model.to(dev)
    enc = model.encoder
    c = {"scaling": cfg["model"]["scaling"], "space": cfg["ssl"]["target_space"], **CONST}
    x, m, token = synth(task).to(dev), synth_mask(task).to(dev), synth_token().to(dev)
    with torch.no_grad():
        t = enc(ssl.scale(x, c)).double()
        nxt, msk = ssl.terms(enc.forward_patches, model, *ssl.views(enc, x, m, token, c), m)
    return {"mean": float(t.mean()), "abs_mean": float(t.abs().mean()), "e000": float(t[0, 0, 0]),
            "e1_50_100": float(t[1, 50, 100]), "absmax": float(t.abs().max()), "next": float(nxt),
            "masked": float(msk)}


@pytest.mark.gpu
@pytest.mark.parametrize("task", ["dro", "dch"])
def test_reproduces_the_old_paper_code(task):
    """Tokens of the last block (global_minmax for DRO, trace_native for DCH) and both SSL terms (zscore / input target
    space) on fixed inputs equal the values the old paper scripts give on the same inputs."""
    assert torch.cuda.is_available(), "this test needs a GPU (the values were computed on one)"
    got, want = regression_values(task), EXPECTED[task]
    bad = {k: (got[k], want[k]) for k in want if not math.isclose(got[k], want[k], rel_tol=TOL, abs_tol=0)}
    assert not bad, f"{task}: values differ from the old paper code by more than {TOL} relative: {bad}"


def test_configs_are_valid():
    """Every paper and smoke config passes config.check and the stage-combination check of the pipeline, and names
    a data config that exists."""
    files = sorted(f for d in CONFIG_DIRS for f in d.glob("*.yaml"))
    assert len(files) >= 8
    for f in files:
        cfg = config.load(f)
        config.check(cfg)
        pipeline.check_stages(cfg)
        store.load_config(cfg["data"]["set"], cfg["data"]["config"])        # KeyError if it does not exist
        if f.parent.name == "smoke":
            assert cfg["run"]["smoke"]
            for s in ("ssl", "student"):
                assert cfg[s]["steps"] <= 20 and cfg[s]["epochs"] == 0 and not cfg[s]["compile"], (f, s)
            assert cfg["compress"]["prune_steps"] <= 20 and cfg["compress"]["steps"] <= 20


@pytest.mark.gpu
def test_teacher_smoke(tmp_path, monkeypatch):
    """configs/smoke/dro.yaml, stages ssl and ridge, end to end with the real weights on dro_gen/pwo_smoke."""
    assert torch.cuda.is_available(), "this test needs a GPU"
    if not (SMOKE_DATA / "dev" / "meta.json").exists():
        pytest.fail(f"{SMOKE_DATA} is not built: on a compute node run python scripts/prepare_data.py dro_gen pwo_smoke")
    monkeypatch.setattr(paths, "OUTPUT", tmp_path)
    cfg = config.apply_overrides(config.load(paths.ROOT / "configs" / "smoke" / "dro.yaml"),
                                 ["stages=[ssl,ridge]", "run.group=test", "run.name=smoke"])
    t = time.time()
    r = pipeline.run(cfg)
    assert not r["preempted"] and list(r["results"]) == ["ssl", "ridge"]
    run = Path(r["run"])
    for stage in ("ssl", "ridge"):
        assert (run / stage / "result.json").exists()
    assert all(math.isfinite(h["train_next"]) and math.isfinite(h["train_masked"])
               for h in load_json(run / "ssl" / "history.json"))
    pred = rundir.read_teacher_labels(run / "ridge" / "teacher_labels.npz", wdata.dev(cfg))
    assert pred.shape == (1993, 3) and np.isfinite(pred).all()
    assert time.time() - t < 120, f"the smoke test took {time.time() - t:.0f} s, more than 2 minutes"
