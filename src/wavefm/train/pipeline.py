"""Run the stages of one run in order: ``stages: [ssl, ridge, student]`` -> ssl, then ridge, then student.

    <output>/<group>/<name>/config.json     the expanded config, written once; a later call with another config
                                            (another rundir.config_fp) under the same name is refused
    <output>/<group>/<name>/<stage>/        ``wavefm.train.<stage>.run(cfg, out, parent)``; ``parent`` is the
                                            directory of the stage before (None for the first)

What a stage hands to the next one is one file of its directory, HANDOVER[stage] (``handover(parent)``): the stages
start from it and record its sha256 (rundir.parent_record), and scripts/run_scan.py derives the expected parent of a
finished stage from it.

The pipeline only calls. Whether a stage is already finished, and whether a finished one is this stage, is decided by
the stage itself (rundir.check_identity, which also compares the sha256 of the parent weights), never by the presence
of a file here. A stage that raises stops the run and the error propagates; a stage that returns
``{"preempted": True}`` (deadline or SIGTERM, resumable) stops it cleanly, and scripts/train.py exits with the
trainer's PREEMPTED code so that the job is started again. A SIGTERM (or the deadline) that arrives while a stage
finishes (during its terminal evaluation, or during ridge) lets that stage complete, and the pipeline then starts no
further stage and returns preempted.

``compress`` lives in the other environment (TensorFlow/QKeras, no torch): the pipeline stops before it and prints
the command that runs it. It distils the student on the teacher labels of ridge, so ``check_stages`` refuses, before
any training, a ``compress`` that does not follow ridge and a teacher-label student.
"""
import importlib
import math
import signal
import time
from pathlib import Path

from wavefm import paths
from wavefm.train import rundir
from wavefm.utils import config, runtime
from wavefm.utils.io import save_json, load_json


HANDOVER = {"ssl": "weights.pt", "finetune": "weights.pt", "ridge": "teacher_labels.npz"}


class PipelineError(RuntimeError):
    pass


def handover(parent):
    """The file the stage directory ``parent`` hands to the next stage (None for no parent)."""
    if parent is None:
        return None
    if Path(parent).name not in HANDOVER:
        raise PipelineError(f"stage {Path(parent).name} hands nothing to a next stage (known: {sorted(HANDOVER)})")
    return Path(parent) / HANDOVER[Path(parent).name]


def check_stages(cfg: dict):
    """Stage combinations that cannot run to the end, refused at the start."""
    st = cfg["stages"]
    if "compress" in st and (not {"ridge", "student"} <= set(st[:st.index("compress")])
                             or cfg["student"]["targets"] != "teacher"):
        raise PipelineError(f"stages {st} with student.targets {cfg['student']['targets']!r}: compress distils on "
                            f"the teacher labels of ridge, so it needs stages [.., ridge, student, compress] and "
                            f"student.targets teacher")


def run_dir(cfg: dict):
    return paths.OUTPUT / cfg["run"]["group"] / cfg["run"]["name"]


def compress_command(directory) -> str:
    return f"cd $WAVEFM_ROOT && uv run --frozen --project compression python compression/compress.py {directory}"


def write_config(cfg: dict, directory):
    p = directory / "config.json"
    if p.exists():
        old = rundir.config_fp(load_json(p))
        if old != rundir.config_fp(cfg):
            raise PipelineError(f"{p} holds another config ({old} != {rundir.config_fp(cfg)}): use a new run.name; "
                                f"a run directory belongs to one config")
        return
    save_json(cfg, p)


def run(cfg: dict) -> dict:
    """Returns {"run": run directory, "results": {stage: what its run returned}, "preempted": bool,
    "stopped_before": "compress" or None}."""
    config.check(cfg)
    check_stages(cfg)
    runtime.setup_numerics()        # strict fp32 before any stage: ridge after ssl refuses to run otherwise
    directory = run_dir(cfg)
    write_config(cfg, directory)
    done = {"run": str(directory), "results": {}, "preempted": False, "stopped_before": None}
    parent, stop, deadline = None, [], cfg["run"]["deadline"] or math.inf
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    try:
        for stage in cfg["stages"]:
            if stage == "compress":
                print(f"[pipeline] stopping before compress (other environment); run:\n    "
                      f"{compress_command(directory)}", flush=True)
                done["stopped_before"] = "compress"
                break
            if stop or time.time() > deadline:
                print(f"[pipeline] {'SIGTERM' if stop else 'deadline'}: not starting stage {stage}; start the same "
                      f"command again to resume", flush=True)
                done["preempted"] = True
                break
            out = directory / stage
            print(f"[pipeline] stage {stage}: {out} (parent {parent})", flush=True)
            result = importlib.import_module(f"wavefm.train.{stage}").run(cfg, out, parent)
            done["results"][stage] = result
            if result.get("preempted"):
                print(f"[pipeline] stage {stage} preempted: start the same command again to resume", flush=True)
                done["preempted"] = True
                break
            parent = out
    finally:
        signal.signal(signal.SIGTERM, previous)
    return done
