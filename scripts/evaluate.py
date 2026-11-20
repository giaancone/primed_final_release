"""The final evaluation of a run: predict heldout once, in physical units, and score it with the data set's metric.

Usage: python scripts/evaluate.py <run directory>

This is one of the two entries that open heldout (wavefm.data.final_evaluation); the other is
compression/compress.py --evaluate, which scores the compress stage in its own environment. It takes the last stage
of the run's ``stages`` that predicts, i.e. whose module has ``predict`` (ssl has none: it trains the encoder only),
and refuses unless that stage has finished: evaluating an earlier one would spend the run's one evaluation on it. A
non-smoke run is evaluated only after its curves are judged: every stage up to the evaluated one must be accepted
(rundir.accepted: finished, verdict "pass" on its present result; ridge needs none); smoke runs never enter a
summary and are exempt. It asks that stage for its predictions, ``wavefm.train.<stage>.predict(stage_dir, split) -> (N, k)``
in the units of the data set's labels, made with the constants the stage saved (nothing is fitted here or there), and
writes ``<run>/heldout.json``: {metrics, stage, data_hash, n, result}, where result identifies the stage's results
(config_fp, parent, smoke and the sha256 of its result.json). Numerics are set to strict fp32 first
(utils.runtime.setup_numerics).

Once per run. Before heldout is opened, the evaluation is claimed with ``<run>/.heldout.claim`` (created with
O_EXCL): a second evaluator that starts meanwhile refuses without opening heldout. The claim is removed when the
evaluation ends, also when it fails, so a failed attempt can be retried; heldout.json, once written, refuses every
later call, which prints the existing file.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import wavefm.data
from data import store
from wavefm.train import rundir
from wavefm.utils import runtime
from wavefm.utils.io import load_json


class EvaluationError(RuntimeError):
    pass


def _refuse_if_done(out):
    if out.exists():
        raise FileExistsError(f"{out} exists: heldout is predicted once per run and the file is never overwritten.\n"
                              f"{out.read_text()}")


def evaluate(run) -> dict:
    run = Path(run)
    out, claim = run / "heldout.json", run / ".heldout.claim"
    _refuse_if_done(out)
    cfg = load_json(run / "config.json")
    if not any((run / s / "result.json").exists() for s in cfg["stages"]):
        raise EvaluationError(f"{run}: no stage of {cfg['stages']} has finished (result.json)")
    # compress lives in the other environment and is scored by compression/compress.py --evaluate
    pred = [s for s in cfg["stages"]
            if s != "compress" and hasattr(importlib.import_module(f"wavefm.train.{s}"), "predict")]
    if not pred:
        raise EvaluationError(f"{run}: no stage of {cfg['stages']} predicts (ssl trains the encoder only): "
                              f"run a stage that predicts (ridge, finetune, student) after it")
    stage = pred[-1]
    if not (run / stage / "result.json").exists():
        raise EvaluationError(f"{run}: stage {stage} of {cfg['stages']} has not finished: finish it first; evaluating "
                              f"now would spend the run's one evaluation on an earlier stage")
    if not cfg["run"]["smoke"]:
        for s in cfg["stages"][:cfg["stages"].index(stage) + 1]:
            if not rundir.accepted(run / s):
                raise EvaluationError(f"{run / s} is not accepted (unfinished, a promised file missing, or no 'pass' "
                                      f"verdict on its present result): judge the curves (scripts/judge.py) first")
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664))
    except FileExistsError:
        raise EvaluationError(f"{run} is being evaluated by another process ({claim} exists); if none is running, "
                              f"delete {claim}") from None
    try:
        _refuse_if_done(out)                 # finished by another evaluator just before the claim
        return _predict_and_write(run, cfg, stage, out)
    finally:
        claim.unlink()


def _predict_and_write(run, cfg, stage, out) -> dict:
    result = rundir.read_result(run / stage)
    runtime.setup_numerics()
    with wavefm.data.final_evaluation():
        split = wavefm.data.heldout(cfg)
        if result["data_hash"] != split.meta["data_hash"]:
            raise EvaluationError(f"{run / stage} was trained on another definition of the data set (data hash "
                                  f"{result['data_hash'][:12]}, heldout {split.meta['data_hash'][:12]}): not comparable")
        pred = importlib.import_module(f"wavefm.train.{stage}").predict(run / stage, split)
    pred = np.asarray(pred, np.float64)
    if pred.shape != split.labels.shape:
        raise EvaluationError(f"stage {stage} predicted {pred.shape} for heldout labels {split.labels.shape}")
    record = {"metrics": store.evaluate(cfg["data"]["set"])(pred, split.labels), "stage": stage,
              "data_hash": split.meta["data_hash"], "n": len(split),
              "result": {"config_fp": result["config_fp"], "parent": result["parent"], "smoke": result["smoke"],
                         "sha256": rundir.file_sha256(run / stage / "result.json")}}
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1, default=float))
    try:
        os.link(tmp, out)                    # atomic, and fails if another call wrote it meanwhile
    finally:
        tmp.unlink()
    return record


def main(argv) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1])
        return 2
    try:
        record = evaluate(argv[0])
    except (FileExistsError, EvaluationError) as e:
        print(e)
        return 1
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
