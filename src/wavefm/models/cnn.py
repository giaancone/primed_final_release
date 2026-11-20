"""Full-rate streaming CNN from scratch (cnnL family), the reference baseline.

Strided first convolution (16 samples, stride 8), three local blocks with max-pooling, global average, linear head.
Widths (32, 64, 128) give the 36 k model.
"""
import torch.nn as nn


class StreamingCNN(nn.Module):
    def __init__(self, n_out: int = 3, widths=(32, 64, 128), k1: int = 16, s1: int = 8):
        super().__init__()
        c1, c2, c3 = widths
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, k1, stride=s1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c3, n_out))

    def forward(self, x):
        return self.net(x[:, None])
