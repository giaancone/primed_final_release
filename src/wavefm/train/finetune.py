"""The supervised stage of the 2-layer method and of the CNN baseline: ``run(cfg, out, parent)``, ``predict``.

What it does: the task (wavefm.tasks) draws the labelled rows of dev inside the label budget and cuts them into
training and validation rows (data.splits.labelled, from cfg["seed"]); the model is ``models.build.build_model`` with
the task's head (the encoder starts from ``<parent>/weights.pt`` when there is a parent stage, else from the
pretrained file); the optimiser groups are ``build.param_groups(model, cfg["finetune"])``; the shared loop
(train.trainer) trains it with the task's loss and records the task's exact validation loss on the validation rows.
Selection, when a config asks for it, uses that loss, never a metric.

Refused before any work: ``model.kind: cnn`` with a task other than dro (the CNN has no per-sample head), and the
controls ``model.blocks: random | none`` with ``finetune.train_layers != 0`` (the controls are for ssl + ridge).

Files in ``out``: rows.npz (``train``, ``val``: the dev positions the stage used), constants.json (the task's
constants: input and target scaling, loss weights, labels), weights.pt (state dict of the kept weights, keys of
build.py: ``encoder.*`` and ``head.*``, or the CNN's), history.json, result.json.
``predict`` rebuilds the model from the run's config.json and weights.pt, restores the saved task constants and
refits nothing.
"""
from pathlib import Path

import numpy as np
import torch

import wavefm.data as wdata
from wavefm import tasks
from wavefm.models.build import build_model, param_groups, unused_parameters
from wavefm.train import guards, pipeline, rundir, trainer
from wavefm.utils import runtime
from wavefm.utils.io import save_json, save_npz, load_json


class FinetuneError(ValueError):
    pass


def check_config(cfg: dict):
    m, f = cfg["model"], cfg["finetune"]
    if m["kind"] == "cnn" and cfg["task"] != "dro":
        raise FinetuneError(f"model.kind cnn is implemented for task dro only, not {cfg['task']!r} (on purpose: it has "
                            f"no per-sample head); use model.kind timesfm for {cfg['task']}")
    if m["kind"] == "timesfm" and m["blocks"] != "pretrained" and f["train_layers"] != 0:
        raise FinetuneError(f"model.blocks {m['blocks']!r} is a control for ssl + ridge (frozen or absent blocks); "
                            f"with finetune.train_layers = {f['train_layers']} it would train them: set train_layers 0")


def _forward(model, f, device):
    return torch.compile(model) if f["compile"] and device.type == "cuda" else model


def run(cfg: dict, out, parent) -> dict:
    runtime.setup_numerics()                       # strict fp32 before anything else runs
    out = Path(out)
    check_config(cfg)
    f = cfg["finetune"]
    init = pipeline.handover(parent)
    prec = rundir.parent_record(init)
    data_hash = wdata.dev(cfg).meta["data_hash"]
    done = rundir.check_identity(out, cfg, data_hash, prec)
    if done is not None:
        return done
    out.mkdir(parents=True, exist_ok=True)
    with guards.RunLock(out) as lock:
        runtime.seed_all(cfg["seed"])
        device = runtime.device()
        task = tasks.get(cfg["task"])
        train, val = task.prepare(cfg, device)
        save_npz(out / "rows.npz", train=train.idx, val=val.idx)
        save_json(task.constants, out / "constants.json")
        n_out = len(task.constants["labels"])
        model = build_model(cfg, task.head(cfg["model"]), n_out, init_from=init).to(device)
        groups = param_groups(model, f)
        fwd = _forward(model, f, device)
        augment = f["augment_shift"] > 0

        def loss(b, gen):
            x, t = train.batch(b)
            if augment:
                x, t = task.augment(x, t, gen)
            return task.loss(fwd(x), t)

        def validate():
            return task.val_loss(fwd, val, f["eval_batch"])

        s = trainer.train(out, cfg, "finetune", model, groups, loss, len(train), data_hash=data_hash, parent=prec,
                          validate=validate, unused=unused_parameters(model), lock=lock)
        if s.get("preempted"):
            return s
        n_trained = sum(p.numel() for g in groups for p in g["params"])
        return trainer.finish(out, cfg, "finetune", s, artifacts=("rows.npz", "constants.json"), n_val=len(val),
                              n_trained=n_trained, n_params=sum(p.numel() for p in model.parameters()))


def load(out, device=None):
    """(model, task, cfg) of the finished finetune stage ``out``, rebuilt from config.json and weights.pt."""
    out = Path(out)
    cfg, res = load_json(out.parent / "config.json"), rundir.read_result(out)
    if rundir.config_fp(cfg, "finetune") != res["config_fp"]:
        raise FinetuneError(f"{out.parent}/config.json is not the config the finetune stage {out} was trained with")
    w = out / "weights.pt"
    for name in ("weights.pt", "constants.json"):
        guards.check_artifact(out, res, name)
    device = device or runtime.device()
    task = tasks.get(cfg["task"]).restore(load_json(out / "constants.json"), device)
    # the encoder is built from weights.pt itself (its encoder.* keys), so the pretrained file is not read
    model = build_model(cfg, task.head(cfg["model"]), len(task.constants["labels"]),
                        init_from=w if cfg["model"]["kind"] == "timesfm" else None)
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), task, cfg


def predict(out, split) -> np.ndarray:
    """(N, k) float64 predictions of the finished stage ``out`` for every row of ``split``, physical units."""
    runtime.setup_numerics()
    model, task, cfg = load(out)
    wset = task.waveforms(split, np.arange(len(split)), task.device)
    return task.predict(model, wset, cfg["finetune"]["eval_batch"])
