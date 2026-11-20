"""DCH: count the primary ionisation clusters of a drift-chamber waveform (label ``n_primary``), with the class of
every sample (extra ``classes``: noise 0, primary 1, secondary 2) as an auxiliary dense target (PRIMED's scheme).

Loss: ``ce_weight`` x class-weighted cross entropy of the per-sample classes + ``count_weight`` x squared error of the
scaled count. With ``class_weighted`` the class weights are the inverse class frequencies of the training rows.

The weighted cross entropy is sum_i w(y_i) nll_i / sum_i w(y_i) over the real samples: the padded samples
(batches.WaveformSet.mask) are selected out before the cross entropy, so nothing of them (not even a NaN logit) enters
the sum or the normalisation, and a validation loss sums numerator and
denominator over all batches before dividing, so it does not depend on the evaluation batch size.
"""
import numpy as np
import torch

from wavefm.data import scaling
from wavefm.data.batches import WaveformSet
from wavefm.models.backbones import DIM
from wavefm.models.heads import PeakCountHead
from wavefm.tasks import common

COUNT, CLASSES, N_CLASSES = "n_primary", "classes", 3


class DCH:
    name = "dch"

    def prepare(self, cfg, device):
        f = cfg["finetune"]
        dev, tr, va = common.labelled_rows(cfg)
        if dev.names != [COUNT]:
            raise ValueError(f"task dch predicts the label {COUNT!r} only; {dev.path} has {dev.names}")
        freq = np.bincount(dev.extra(CLASSES)[tr].ravel(), minlength=N_CLASSES).astype(np.float64)
        cw = freq.sum() / (N_CLASSES * np.maximum(freq, 1)) if f["class_weighted"] else np.ones(N_CLASSES)
        self.restore({"task": self.name, "set": dev.meta["set"], "labels": dev.meta["labels"], "length": dev.length,
                      "input": common.input_constants(cfg, dev),
                      "target": scaling.target_scaler(f["target_scaling"], dev.labels[tr].astype(np.float64),
                                                      f["divisors"]).to_dict(),
                      "class_frequency": (freq / freq.sum()).tolist(), "class_weights": cw.tolist(),
                      "ce_weight": float(f["ce_weight"]), "count_weight": float(f["count_weight"])}, device)
        return self.waveforms(dev, tr, device), self.waveforms(dev, va, device)

    def restore(self, constants, device):
        """Set every constant of the task from a dict (as saved by a stage): nothing is computed from data."""
        self.constants, self.device = constants, device
        self.in_scaler = common.input_scaler(constants)
        self.y_scaler = scaling.from_dict(constants["target"])
        self.class_w = torch.tensor(constants["class_weights"], dtype=torch.float32, device=device)
        self.weights = {"ce": constants["ce_weight"], "count": constants["count_weight"]}
        return self

    def waveforms(self, split, idx, device):
        common.check_labels(self.constants, split)
        Y = split.labels[idx].astype(np.float64)
        return WaveformSet(split, idx, device, self.in_scaler,
                           {"count": self.y_scaler.forward(Y)[:, 0].astype(np.float32), "raw": Y,
                            "classes": split.extra(CLASSES)[idx]})

    def head(self, m):
        return PeakCountHead(DIM, hidden=m["head_hidden"])

    def terms(self, out, t):
        real = t["mask"].expand(t["classes"].shape)          # padded samples are dropped before anything is computed
        y = t["classes"][real].long()
        nll = torch.nn.functional.cross_entropy(out["logits"][real], y, reduction="none")
        w = self.class_w[y]
        e = (out["count"] - t["count"]) ** 2
        return {"ce": ((w * nll).sum(), w.sum()), "count": (e.sum(), e.numel())}

    def loss(self, out, t):
        return common.total(self.terms(out, t), self.weights)

    def scaled(self, out):
        return out["count"][:, None]

    def val_loss(self, forward, wset, batch_size):
        return common.val_loss(self, forward, wset, batch_size)

    def predict(self, forward, wset, batch_size):
        return common.predict(self, forward, wset, batch_size)

    def score(self, pred, wset):
        return common.score(self, pred, wset)
