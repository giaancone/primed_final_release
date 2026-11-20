"""DRO time-shift augmentation: a shifted trace is the trace of the shifted pulse, labels follow, edges are padded,
and prepare() sets it up from finetune.augment_shift on the training rows only."""
import numpy as np
import pytest
import torch

from wavefm import tasks
from wavefm.data.scaling import Divisor, Standard
from wavefm.utils import config

DT, N, T_PULSE = 0.32, 400, 40.0


def traces(t0):
    """Two-component pulse (fast + slow with a long flat tail) sampled every DT ns, starting at T_PULSE + t0."""
    t = np.arange(N) * DT - T_PULSE - t0[:, None]
    on = t >= 0
    return np.where(on, np.exp(-np.clip(t, 0, None) / 2.0) + 0.3 * np.exp(-np.clip(t, 0, None) / 500.0), 0.0) + 0.01


def task_for(scaler, shift, lo=-2.5, hi=2.5):
    task = tasks.get("dro")
    task.y_scaler, task.shift = scaler, shift
    e = np.zeros((2, 3)); e[1, 2] = 1.0
    f = scaler.forward(e)
    task.t0 = (2, lo, hi, DT, float(f[1, 2] - f[0, 2]), N)
    return task


@pytest.mark.parametrize("scaler", [Divisor([500.0, 220.0, 2.5]), Standard([500.0, 300.0, 0.1], [290.0, 170.0, 1.4])])
def test_shift_is_the_shifted_pulse(scaler):
    rng = np.random.default_rng(0)
    t0 = rng.uniform(-2.5, 2.5, 256)
    raw = np.stack([rng.uniform(0, 1000, 256), rng.uniform(0, 600, 256), t0], 1)
    x = torch.tensor(traces(t0), dtype=torch.float32)
    tg = {"raw": torch.tensor(raw), "y": torch.tensor(scaler.forward(raw), dtype=torch.float32)}
    task = task_for(scaler, shift=7)
    xa, ta = task.augment(x, tg, torch.Generator().manual_seed(1))
    k = np.round((ta["raw"][:, 2] - tg["raw"][:, 2]).numpy() / DT)
    assert np.abs(k).max() <= 7 and len(np.unique(k)) > 5                       # bounded, and really random
    assert (ta["raw"][:, 2] >= -2.5 - 1e-9).all() and (ta["raw"][:, 2] <= 2.5 + 1e-9).all()   # inside the t0 range
    assert torch.equal(ta["raw"][:, :2], tg["raw"][:, :2])                       # C and S unchanged
    assert np.allclose(ta["y"].numpy(), scaler.forward(ta["raw"].numpy()), atol=1e-5)   # scaled labels follow
    expect = traces(ta["raw"][:, 2].numpy())                                     # the trace of the shifted pulse
    assert np.allclose(xa.numpy()[:, 8:-8], expect[:, 8:-8], atol=1e-5)
    assert torch.equal(x, torch.tensor(traces(t0), dtype=torch.float32))         # input batch not modified in place


def test_shift_padding_and_range_edge():
    """Edges take the first / last real sample (padding convention); an event at the edge of the t0 range is only
    shifted inwards."""
    task = task_for(Divisor([1.0, 1.0, 1.0]), shift=5)
    x = torch.arange(N, dtype=torch.float32)[None].repeat(64, 1)
    x = torch.cat([x, x[:, -1:].repeat(1, 16)], 1)                              # 16 padded samples
    raw = torch.zeros(64, 3, dtype=torch.float64); raw[:, 2] = 2.5               # t0 at the upper edge
    xa, ta = task.augment(x, {"raw": raw, "y": raw.float()}, torch.Generator().manual_seed(0))
    k = torch.round((ta["raw"][:, 2] - 2.5) / DT).long()
    assert (k <= 0).all() and (k < 0).any()
    for row, kk in zip(xa, k):
        i = torch.arange(N + 16) - kk
        assert torch.equal(row, i.clamp(0, N - 1).float())


def test_prepare_sets_up_the_shift_from_the_training_rows(fake_dro):
    cfg = config.merge(fake_dro, {"finetune": {"augment_shift": 4}})
    task = tasks.get("dro")
    train, val = task.prepare(cfg, "cpu")
    j, lo, hi, dt, slope, n_real = task.t0
    t0 = train.t["raw"][:, 2].numpy()
    assert (j, n_real, dt, task.shift) == (2, 256, 0.32, 4) and (lo, hi) == (t0.min(), t0.max())
    assert slope == pytest.approx(1 / task.y_scaler.std[2])
    x, t = train.batch(torch.arange(32))
    xa, ta = task.augment(x, t, torch.Generator().manual_seed(0))
    assert xa.shape == x.shape and not torch.equal(xa, x)


def test_shift_needs_a_t0_label(fake_dro, monkeypatch):
    from wavefm.tasks import common
    dev, tr, va = common.labelled_rows(fake_dro)
    dev.names = ["C", "S", "time"]
    monkeypatch.setattr(common, "labelled_rows", lambda cfg: (dev, tr, va))
    with pytest.raises(ValueError, match="t0"):
        tasks.get("dro").prepare(config.merge(fake_dro, {"finetune": {"augment_shift": 4}}), "cpu")
