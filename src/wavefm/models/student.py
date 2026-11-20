"""The small student: an MLP on a decimated waveform, and its input transform in torch.

    mlp(sizes)             n_in -> h1 -> .. -> n_out: Linear layers with biases, ReLU between them, none after the last,
                           nothing else (no normalisation): exactly what rundir.write_student can hold and the
                           compression stage rebuilds. The paper's students: DRO 640-16-24-8-3 (10 891 parameters),
                           DCH 602-16-24-8-1 (10 265).
    kept(x, length, pad_to, every, n_in)
                           raw waveforms (B, length) -> (B, n_in): pad to ``pad_to`` by repeating the last sample, keep
                           samples 0, every, 2 every, .., pad the kept ones to ``n_in`` by repeating the last one.
    scale(k, lo, hi)       (k - lo) / (hi - lo) in float32, the constants rounded to float32 as numpy does.
    student_input(x, spec) scale(kept(x)): the same definition as rundir.student_input, which is the specification
                           (tests/test_student.py checks them against each other bit for bit).

The transform is split in two because a stage may need the kept samples before it knows lo and hi (the scaling
constant ``kept_minmax`` is the range of the kept samples of the training rows). Scaling is element-wise, so scaling
after decimation equals scaling before it.
"""
import torch
import torch.nn as nn


class StudentInputError(ValueError):
    pass


def mlp(sizes) -> nn.Sequential:
    """Linear(sizes[0], sizes[1]), ReLU, .., Linear(sizes[-2], sizes[-1]); PyTorch's default initialisation (the caller
    seeds it). The Linear layers are ``linears(model)``."""
    sizes = [int(s) for s in sizes]
    if len(sizes) < 2 or min(sizes) < 1:
        raise StudentInputError(f"student sizes {sizes}: need an input and an output width, all positive")
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def linears(model: nn.Sequential) -> list:
    return [m for m in model if isinstance(m, nn.Linear)]


def n_kept(pad_to: int, every: int) -> int:
    return -(-pad_to // every)


def kept(x: torch.Tensor, length: int, pad_to: int, every: int, n_in: int) -> torch.Tensor:
    """(B, length) -> (B, n_in): pad, decimate (phase 0), pad. Padding repeats the last sample."""
    if x.ndim != 2 or x.shape[1] != length:
        raise StudentInputError(f"student made for waveforms of {length} samples, got {tuple(x.shape)}")
    if pad_to < length or every < 1 or n_kept(pad_to, every) > n_in:
        raise StudentInputError(f"pad_to {pad_to} >= length {length}, every {every} >= 1 and at most n_in = {n_in} "
                                f"kept samples ({n_kept(pad_to, every)} here): fix the student section of the config")
    if pad_to > length:
        x = torch.cat([x, x[:, -1:].expand(-1, pad_to - length)], 1)
    x = x[:, ::every]
    if x.shape[1] < n_in:
        x = torch.cat([x, x[:, -1:].expand(-1, n_in - x.shape[1])], 1)
    return x.contiguous()


def scale(k: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """float32 (k - lo) / (hi - lo); hi - lo is taken in float64 and then rounded, as in rundir.student_input."""
    k = k.to(torch.float32)
    return (k - torch.tensor(lo, dtype=torch.float32, device=k.device)) / torch.tensor(hi - lo, dtype=torch.float32,
                                                                                        device=k.device)


def student_input(x: torch.Tensor, spec: dict) -> torch.Tensor:
    """The input transform of a student file's spec (rundir.write_student) on raw waveforms (B, L)."""
    i = spec["input"]
    return scale(kept(x.to(torch.float32), i["length"], i["pad_to"], i["every"], spec["sizes"][0]), i["lo"], i["hi"])
