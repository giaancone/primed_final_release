"""Pieces shared by the tasks: the labelled rows of dev, losses as (numerator, denominator) terms, the exact
validation loss, and prediction in physical units."""
import numpy as np
import torch

import wavefm.data
from data import store
from wavefm.data import scaling, splits


def labelled_rows(cfg):
    """(dev split, train positions, validation positions), drawn from ``cfg["seed"]`` and the ``data`` keys."""
    d = cfg["data"]
    dev = wavefm.data.dev(cfg)
    tr, va = splits.labelled(len(dev), d["label_frac"], d["val_frac"], cfg["seed"], d["min_labelled"], d["min_val"])
    return dev, tr, va


def input_constants(cfg, dev) -> dict:
    """The model-level input scaling: global_minmax with the dev minimum and maximum, else only its name (the encoder
    scales)."""
    sc = scaling.input_scaler(cfg["model"]["scaling"], dev.meta)
    return sc.to_dict() if sc else {"kind": cfg["model"]["scaling"]}


def input_scaler(constants):
    return scaling.from_dict(constants["input"]) if constants["input"]["kind"] == "global_minmax" else None


def first(out, n):
    """The first n rows of a model output (a tensor or a dict of tensors): the real rows of a padded batch."""
    return {k: v[:n] for k, v in out.items()} if isinstance(out, dict) else out[:n]


def total(terms: dict, weights: dict):
    """sum_k weights[k] * numerator_k / denominator_k."""
    return sum(weights[k] * num / den for k, (num, den) in terms.items())


@torch.no_grad()
def val_loss(task, forward, wset, batch_size) -> float:
    """The training objective on a whole set. Numerators and denominators are summed over the batches (float64)
    before they are divided, so the value does not depend on the batch size."""
    acc = {}
    for x, t, n in wset.ordered_batches(batch_size):
        for k, (num, den) in task.terms(first(forward(x), n), t).items():
            a = acc.get(k, (0.0, 0.0))
            acc[k] = (a[0] + num.double(), a[1] + (den.double() if torch.is_tensor(den) else den))
    return float(total(acc, task.weights))


@torch.no_grad()
def predict(task, forward, wset, batch_size) -> np.ndarray:
    """(N, k) float64 predictions in the units of the data set's labels, rows in the order of ``wset``."""
    out = torch.cat([task.scaled(first(forward(x), n)) for x, _, n in wset.ordered_batches(batch_size)])
    return task.y_scaler.inverse(out.double().cpu().numpy())


def score(task, pred, wset) -> dict:
    """The data set's own metric (data/<set>/evaluate.py) of ``pred`` on the rows of ``wset``."""
    return store.evaluate(task.constants["set"])(pred, wset.t["raw"].cpu().numpy())


def check_labels(constants: dict, split):
    """A task restored from saved constants predicts only a split with the same label columns and length."""
    if split.meta["labels"] != constants["labels"] or split.length != constants["length"]:
        raise ValueError(f"{split.path}: labels {split.meta['labels']} and length {split.length} differ from those the "
                         f"task was trained with ({constants['labels']}, {constants['length']})")
