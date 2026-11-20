"""DRO: regress every label of the data set (dro_gen: C, S in photons, t0 in ns) from the crystal waveform.

Loss: mean over rows and targets of ``finetune.loss_weights`` x squared error of the scaled targets (PRIMED:
weights [4, 1, 0.3] with divisors [500, 220, 2.5]). Augmentation (``finetune.augment_shift``): a random time shift of
each training waveform, with t0 moved along.
"""
import numpy as np
import torch

from wavefm.data import scaling
from wavefm.data.batches import WaveformSet
from wavefm.models.backbones import DIM
from wavefm.models.heads import MeanPoolHead
from wavefm.tasks import common


class DRO:
    name = "dro"

    def prepare(self, cfg, device):
        f = cfg["finetune"]
        dev, tr, va = common.labelled_rows(cfg)
        k = len(dev.names)
        w = f["loss_weights"] or [1.0] * k
        if len(w) != k:
            raise ValueError(f"finetune.loss_weights {w}: one weight per label {dev.names}")
        Y = dev.labels[tr].astype(np.float64)
        self.restore({"task": self.name, "set": dev.meta["set"], "labels": dev.meta["labels"], "length": dev.length,
                      "input": common.input_constants(cfg, dev),
                      "target": scaling.target_scaler(f["target_scaling"], Y, f["divisors"]).to_dict(),
                      "loss_weights": [float(v) for v in w]}, device)
        self.shift = int(f["augment_shift"])
        if self.shift:   # t0 range of the training rows, sampling interval, and the change of the scaled t0 per ns
            if "t0" not in dev.names:
                raise ValueError(f"finetune.augment_shift needs a label t0, the data set has {dev.names}")
            j = dev.names.index("t0")
            e = np.zeros((2, k)); e[1, j] = 1.0
            s = self.y_scaler.forward(e)
            self.t0 = (j, float(Y[:, j].min()), float(Y[:, j].max()), float(dev.meta["dt_ns"]),
                       float(s[1, j] - s[0, j]), dev.length)
        return self.waveforms(dev, tr, device), self.waveforms(dev, va, device)

    def restore(self, constants, device):
        """Set every constant of the task from a dict (as saved by a stage): nothing is computed from data."""
        self.constants, self.device = constants, device
        self.in_scaler = common.input_scaler(constants)
        self.y_scaler = scaling.from_dict(constants["target"])
        self.w = torch.tensor(constants["loss_weights"], dtype=torch.float32, device=device)
        self.weights = {"mse": 1.0}
        return self

    def waveforms(self, split, idx, device):
        common.check_labels(self.constants, split)
        Y = split.labels[idx].astype(np.float64)
        return WaveformSet(split, idx, device, self.in_scaler,
                           {"y": self.y_scaler.forward(Y).astype(np.float32), "raw": Y})

    def head(self, m):
        return MeanPoolHead(DIM, len(self.constants["labels"]), hidden=m["head_hidden"],
                            n_extra=2 if m["scaling"] == "native_stats" else 0)

    def augment(self, x, t, gen):
        """Shift each trace by k samples (|k| <= augment_shift, t0 + k dt inside the training t0 range): the edges are
        filled with the first (baseline) or last sample, as the padding; t0 moves by k dt, C and S are unchanged."""
        j, lo, hi, dt, slope, n_real = self.t0
        t0 = t["raw"][:, j]
        kmin = torch.clamp(torch.ceil((lo - t0) / dt - 1e-6), min=-self.shift, max=0)
        kmax = torch.clamp(torch.floor((hi - t0) / dt + 1e-6), min=0, max=self.shift)
        k = (kmin + torch.floor(torch.rand(len(x), generator=gen, device=x.device, dtype=t0.dtype) * (kmax - kmin + 1))).clamp(max=kmax)
        i = (torch.arange(x.shape[1], device=x.device)[None] - k.long()[:, None]).clamp(0, n_real - 1)
        raw, y = t["raw"].clone(), t["y"].clone()
        raw[:, j] += k * dt
        y[:, j] += (k * dt * slope).to(y.dtype)
        return x.gather(1, i), {**t, "raw": raw, "y": y}

    def terms(self, out, t):
        e = (out - t["y"]) ** 2 * self.w
        return {"mse": (e.sum(), e.numel())}

    def loss(self, out, t):
        return common.total(self.terms(out, t), self.weights)

    def scaled(self, out):
        return out

    def val_loss(self, forward, wset, batch_size):
        return common.val_loss(self, forward, wset, batch_size)

    def predict(self, forward, wset, batch_size):
        return common.predict(self, forward, wset, batch_size)

    def score(self, pred, wset):
        return common.score(self, pred, wset)

