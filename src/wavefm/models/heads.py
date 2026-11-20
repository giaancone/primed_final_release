"""Regression heads on encoder tokens."""
import torch
import torch.nn as nn


class MeanPoolHead(nn.Module):
    """Mean over tokens, optionally joined by per-trace scale features, then an MLP (PRIMED's 1280-128-n head).

    ``n_extra = 2`` takes the trace mean and spread that TimesFM's own scaling removed, as (mean / spread, log spread),
    so the head can restore the amplitude.
    """

    def __init__(self, dim: int, n_out: int, hidden: int = 128, n_extra: int = 0):
        super().__init__()
        self.n_extra = n_extra
        self.mlp = nn.Sequential(nn.Linear(dim + n_extra, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, tokens, stats=None):
        z = tokens.mean(1)
        if self.n_extra:
            mu, sigma = stats[:, 0], stats[:, 1].clamp_min(1e-6)
            z = torch.cat([z, torch.stack([mu / sigma, sigma.log()], -1)], -1)
        return self.mlp(z)


class PeakCountHead(nn.Module):
    """DCH teacher head (PRIMED's ftpc): the mean over tokens -> MLP -> number of primary clusters, and every token ->
    a 3-class logit (noise, primary, secondary) for each of its 32 samples."""

    def __init__(self, dim: int, hidden: int = 128, patch: int = 32, n_classes: int = 3):
        super().__init__()
        self.patch, self.n_classes = patch, n_classes
        self.count = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.per_sample = nn.Linear(dim, patch * n_classes)

    def forward(self, tokens, stats=None):
        B, N, _ = tokens.shape
        return {"count": self.count(tokens.mean(1))[:, 0],
                "logits": self.per_sample(tokens).reshape(B, N * self.patch, self.n_classes)}
