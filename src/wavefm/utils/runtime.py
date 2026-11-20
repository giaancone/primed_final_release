"""Seeds, device and numerics settings.

Networks are trained and evaluated in strict fp32: no TF32 for matmul or cudnn, float32 matmul precision "highest",
no autocast or bf16. Every stage
calls ``setup_numerics()`` first; ``wavefm.train.guards.check_strict_fp32`` refuses to train otherwise.
"""
import random

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def setup_numerics():
    """Strict fp32 and repeatable convolutions for everything that follows in this process."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True      # else two identical GPU runs differ in the last bits (measured
                                                   # on the adapter's convolutions) and a resume is not exact


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
