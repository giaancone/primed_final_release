"""Convolutional adapter after the tokenizer (conv3): ``h + Conv1x1(GELU(Conv3(h)))`` over the patch axis.

Each token gets a correction computed from itself and its two neighbours, so a pulse edge that straddles a patch
boundary is seen before attention. The last convolution starts at zero: at construction the model is unchanged.

Known defect, kept on purpose because the paper ran it: the convolution is symmetric, so token i sees
patch i+1, which leaks the target of the SSL next-patch prediction into its input.
"""
import torch.nn as nn


class ConvAdapter(nn.Module):
    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, 1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, h):                       # h (B, N, D)
        return h + self.conv2(self.act(self.conv1(h.transpose(1, 2)))).transpose(1, 2)
