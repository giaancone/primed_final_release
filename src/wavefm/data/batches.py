"""Device-resident waveform sets: the selected rows of a split are read once, scaled, padded and moved to the device;
a batch is an index slice. No DataLoader and no host-device copies inside the training loop.

Rows are read through ``Split.take`` in chunks of positions, so a split whose waveforms live in external npz files
(dch_wf) reads each file once and keeps it, and every array read is checked against its digest by the split.

Padding (model level): a waveform is padded at the end to a multiple of the patch length by repeating its last
sample. A per-sample target (one value per waveform sample, e.g. the DCH classes) is padded with 0, and ``mask``
(padded length,) is True on the real samples, so that a loss can leave the padded samples out.

``BatchSampler`` keeps its permutation and position, so a resumed run sees exactly the batches it would have seen.
"""
import numpy as np
import torch

PATCH = 32


def pad_to_patch(x: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """Pad (B, L) at the end to a multiple of ``patch`` by repeating the last sample."""
    r = (-x.shape[-1]) % patch
    return x if r == 0 else torch.cat([x, x[..., -1:].expand(*x.shape[:-1], r)], dim=-1)


class WaveformSet:
    def __init__(self, split, idx, device, input_scaler=None, targets: dict | None = None, chunk=8192):
        """split: a data.store.Split; idx: positions in it. targets: name -> array with one row per position of idx;
        an array of shape (len(idx), split.length) is a per-sample target."""
        self.idx = np.asarray(idx, np.int64)
        parts = []
        for a in range(0, len(self.idx), chunk):
            x = split.take(self.idx[a:a + chunk])
            if input_scaler is not None:
                x = input_scaler(x)
            parts.append(pad_to_patch(torch.from_numpy(np.ascontiguousarray(x, np.float32))).to(device))
        self.x = torch.cat(parts)
        self.length = split.length
        self.mask = torch.arange(self.x.shape[1], device=device) < split.length
        self.t = {}
        for name, arr in (targets or {}).items():
            arr = np.asarray(arr)
            if len(arr) != len(self.idx):
                raise ValueError(f"target {name}: {len(arr)} rows for {len(self.idx)} positions")
            t = torch.from_numpy(arr)
            if t.ndim == 2 and t.shape[1] == split.length:
                t = torch.nn.functional.pad(t, (0, self.x.shape[1] - t.shape[1]))
            self.t[name] = t.to(device)
        if not torch.isfinite(self.x).all() or any(not torch.isfinite(v).all() for v in self.t.values() if v.is_floating_point()):
            raise ValueError("non-finite waveform or label values in the selected rows")

    def __len__(self):
        return len(self.idx)

    def batch(self, b):
        """Waveforms and targets of the rows ``b`` (a tensor of set positions); targets include ``mask``."""
        return self.x[b], {**{k: v[b] for k, v in self.t.items()}, "mask": self.mask}

    def ordered_batches(self, batch_size):
        """(x, t, n) in row order: x of fixed shape (batch_size, padded length), the last batch padded by repeating
        its last row (a fixed shape avoids recompiling a compiled model); t and n: targets and number of real rows."""
        for a in range(0, len(self), batch_size):
            x, t = self.batch(slice(a, a + batch_size))
            n = len(x)
            if n < batch_size:
                x = torch.cat([x, x[-1:].expand(batch_size - n, -1)])
            yield x, t, n


class BatchSampler:
    """Random batches without replacement, a new permutation every epoch, resumable through state_dict()."""

    def __init__(self, n: int, batch_size: int, seed: int, device):
        self.n, self.bs, self.device = n, batch_size, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.perm, self.pos = None, 0

    def next(self) -> torch.Tensor:
        if self.perm is None or self.pos + self.bs > self.n:
            self.perm, self.pos = torch.randperm(self.n, device=self.device, generator=self.gen), 0
        b = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return b

    def state_dict(self):
        return {"gen": self.gen.get_state(), "perm": None if self.perm is None else self.perm.cpu(), "pos": self.pos}

    def load_state_dict(self, s):
        self.gen.set_state(s["gen"].cpu())
        self.perm = None if s["perm"] is None else s["perm"].to(self.device)
        self.pos = s["pos"]
