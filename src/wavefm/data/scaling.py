"""Model-level input and target scaling, saved with a run and rebuilt from the saved constants.

``GlobalMinMax`` (``model.scaling: global_minmax``): one minimum and one maximum, those of all dev waveforms as
recorded by the data set (dev meta.json), so relative amplitudes between events are kept and every method of a data
set scales identically. Nothing is fitted. The other input scalings happen inside the encoder.

Targets: ``Standard`` (mean and standard deviation of the given rows, float64) or ``Divisor`` (fixed divisors).

``to_dict`` / ``from_dict``: the final evaluation predicts heldout with the constants saved at training time, never
with constants recomputed from data, so a change of dev after training cannot change a heldout prediction.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GlobalMinMax:
    lo: float
    hi: float

    @classmethod
    def from_meta(cls, meta: dict):
        st = meta["waveform"]["stats"]
        if not st["max"] > st["min"]:
            raise ValueError(f"dev waveforms are constant ({st['min']}): min-max scaling is undefined")
        return cls(float(st["min"]), float(st["max"]))

    def __call__(self, x):
        return (x - np.float32(self.lo)) / np.float32(self.hi - self.lo)

    def to_dict(self):
        return {"kind": "global_minmax", **asdict(self)}


@dataclass
class Standard:
    mean: list
    std: list

    @classmethod
    def fit(cls, Y):
        Y = np.asarray(Y, np.float64)
        if (Y.std(0) == 0).any():
            raise ValueError("a target is constant on the training rows: standard scaling is undefined")
        return cls(Y.mean(0).tolist(), Y.std(0).tolist())

    def forward(self, y):
        return (y - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse(self, y):
        return y * np.asarray(self.std) + np.asarray(self.mean)

    def to_dict(self):
        return {"kind": "standard", **asdict(self)}


@dataclass
class Divisor:
    div: list

    def forward(self, y):
        return y / np.asarray(self.div)

    def inverse(self, y):
        return y * np.asarray(self.div)

    def to_dict(self):
        return {"kind": "divisor", **asdict(self)}


KINDS = {"global_minmax": GlobalMinMax, "standard": Standard, "divisor": Divisor}


def from_dict(d: dict):
    return KINDS[d["kind"]](**{k: v for k, v in d.items() if k != "kind"})


def input_scaler(scaling: str, meta: dict):
    """The scaler applied when waveforms are loaded, or None when the encoder scales (model.scaling)."""
    return GlobalMinMax.from_meta(meta) if scaling == "global_minmax" else None


def target_scaler(kind: str, Y, divisors=None):
    """``Y``: the targets of the training rows, (n, k) physical units."""
    if kind == "standard":
        return Standard.fit(Y)
    if kind == "divisor":
        if not divisors or len(divisors) != np.shape(Y)[1]:
            raise ValueError(f"finetune.target_scaling=divisor needs {np.shape(Y)[1]} finetune.divisors, got {divisors}")
        return Divisor([float(v) for v in divisors])
    raise ValueError(f"unknown finetune.target_scaling {kind!r}: standard | divisor")
