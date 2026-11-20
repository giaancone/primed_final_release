"""Tasks: everything that differs between DRO and DCH (targets, head, loss, prediction). The training loop is shared.

A task object (``get(cfg["task"])``) has, and the trainer and the stages use, only this:

    train, val = task.prepare(cfg, device)
        WaveformSets of dev positions (``train.idx``, ``val.idx``) drawn by data.splits.labelled from cfg["seed"] and
        the ``data`` keys; the target scaler is fitted on the training rows, the input scaling taken from dev
        meta.json. Never a heldout row. Sets ``task.constants``.
    task.constants
        a JSON-able dict of every constant the task uses (labels, length, input and target scaling, loss weights,
        class weights). A stage saves it; ``task.restore(constants, device)`` sets the task from it without data.
    task.waveforms(split, idx, device) -> WaveformSet
        positions ``idx`` of any split, scaled with the task's constants (the final evaluation: heldout).
    task.head(cfg["model"]) -> torch.nn.Module                  the head on encoder tokens (B, N, 1280)
    task.loss(out, t) -> scalar tensor                          the training objective on one batch
    task.val_loss(forward, wset, batch_size) -> float           the same objective on a whole set, exact: it does not
                                                                depend on batch_size
    task.predict(forward, wset, batch_size) -> (N, k) float64   physical units, columns = the data set's labels
    task.score(pred, wset) -> dict                              the data set's own metric (data/<set>/evaluate.py),
                                                                for the record; selection uses the loss, never this
    task.augment(x, t, gen) -> (x, t)                           dro only, when finetune.augment_shift > 0

``forward(x) -> out`` is the model on a batch of scaled, padded waveforms (B, padded length); the caller puts the
model in eval mode. ``t`` is the target dict of ``wset.batch(b)``.
"""
from wavefm.tasks.dch import DCH
from wavefm.tasks.dro import DRO

TASKS = {"dro": DRO, "dch": DCH}


def get(name: str):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}, known {sorted(TASKS)}")
    return TASKS[name]()
