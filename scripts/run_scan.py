"""Run a development scan: a set of configs, one process per run, the runs spread over the visible GPUs.

Usage: python scripts/run_scan.py <scan.yaml | configs.json> [--dry-run] [--gpus N]

A scan is development: the evidence for the fixed defaults of an implementation (configs/scans_dev/README.md). It
trains and validates on dev and never opens heldout: nothing here calls scripts/evaluate.py, and the record holds
only training and validation curves.

The scan file (YAML or JSON) is one of
    {base: <config, relative to the scan file>, group: <name>, grid: {key.path: [value, ..], ..}}
        every combination of the grid values (cartesian product, in the order written)
    {base: .., group: .., runs: [{key.path: value, ..}, ..]}
        one run per entry
or a JSON file holding a LIST of configs, complete or partial, each merged onto configs/teacher.yaml: the way to hand
the scan a set of configs chosen elsewhere (a researcher choosing the next rates and lengths). They share one run.group.

Each config is checked (utils.config.check: unknown keys and stages are errors) and named after its overrides,
``ssl.lr_tokenizer=0.0001_ssl.steps=2000`` (a list config without run.name: after its difference from teacher.yaml;
``run.name`` in the overrides names it explicitly). Two runs of one name are an error. The data set of every config
must be built and current (wavefm.data.dev).

Before anything starts, every run is classified from its directory, <output>/<group>/<name>/:
    done      every stage finished with THIS identity: the config fingerprint, the data hash and the parent, which
              must be the file the pipeline hands it (the stage before in ``stages``, pipeline.handover; none for the
              first) with its present sha256 (rundir.check_identity). Skipped.
    awaiting compression   the same for every stage before compress, which runs in the other environment: the
              record gives its command (a run with compress is never "finished" here). Skipped.
    refused   the directory holds another config, or a finished stage of another identity (including a recorded
              parent other than the expected one, or none where one is expected). Nothing is started and nothing is
              overwritten: the record says why; use another group or run name.
    todo      started; the pipeline skips its finished stages itself and resumes an interrupted one.
A file existing is never taken as "done".

``--dry-run`` expands, checks and classifies every config, prints the list, and starts nothing.

Inside a Slurm allocation each run gets ``run.deadline`` = the end of the allocation minus 10 min (a stage then
checkpoints and stops, and the next scan resumes it), and no run is started with less than 20 min left. A SIGTERM to
the scan is passed on to its runs. A run is a forked process (so it gets the checked config as it is) with its own
CUDA_VISIBLE_DEVICES; it logs to <output>/<group>/<name>/train.log.

The record ``<output>/<group>/scan.json`` is rewritten whenever a run changes state: for every run its overrides,
directory, status, and per stage the status, requested and actual length, final training and validation loss, hints
(analysis.curves), verdict and whether it is accepted. It is plain JSON, built with analysis.read from the run
directories, which analysis.read.runs reads in the same way.
"""
import argparse
import copy
import itertools
import multiprocessing
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

import wavefm.data
from analysis import read
from wavefm import paths
from wavefm.train import pipeline, rundir
from wavefm.train.trainer import PREEMPTED       # imports torch, creates no CUDA context in this process
from wavefm.utils import config
from wavefm.utils.io import load_json, save_json

MARGIN_S, MIN_LEFT_S = 600, 1200
SCAN_KEYS = {"base", "group", "grid", "runs"}


class ScanError(ValueError):
    pass


# ---- expansion ------------------------------------------------------------------------------------------------------

def flat(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def nested(over: dict) -> dict:
    """{"a.b": 1} -> {"a": {"b": 1}}, to be merged onto a config (check() then rejects unknown keys)."""
    out = {}
    for key, v in over.items():
        *path, last = key.split(".")
        d = out
        for k in path:
            d = d.setdefault(k, {})
        d[last] = v
    return out


def _text(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(map(_text, v))
    return str(v).replace("/", "-")


def run_name(over: dict) -> str:
    """``key=value`` of every override, joined by ``_``; ``base`` for none."""
    return "_".join(f"{k}={_text(v)}" for k, v in over.items()) or "base"


def _job(base: dict, over: dict, group: str) -> dict:
    if "run.group" in over:
        raise ScanError(f"overrides {over} set run.group: the group of a scan is its `group`")
    named = {k: v for k, v in over.items() if k != "run.name"}
    cfg = config.merge(base, nested(over))
    cfg["run"].update(group=group, name=over.get("run.name", run_name(named)))
    return {"name": cfg["run"]["name"], "overrides": named, "cfg": cfg}


def expand(path) -> tuple:
    """(group, [{"name", "overrides", "cfg"}]) of a scan file or a JSON list of configs."""
    path = Path(path)
    scan = yaml.safe_load(path.read_text())
    if isinstance(scan, list):
        teacher = config.load(paths.ROOT / "configs" / "teacher.yaml")
        groups = {(c.get("run") or {}).get("group") for c in scan}
        if len(groups) != 1 or None in groups:
            raise ScanError(f"{path}: the configs of a list must all set the same run.group, "
                            f"got {sorted(map(str, groups))}")
        jobs, defaults = [], flat(teacher)
        for c in scan:
            cfg = config.merge(teacher, c)
            diff = {k: v for k, v in flat(cfg).items() if not k.startswith("run.") and defaults.get(k, ...) != v}
            name = c["run"].get("name") or run_name(diff)
            cfg["run"]["name"] = name
            jobs.append({"name": name, "overrides": diff, "cfg": cfg})
        return groups.pop(), jobs
    if not isinstance(scan, dict) or SCAN_KEYS - {"grid", "runs"} - scan.keys() or scan.keys() - SCAN_KEYS \
            or ("grid" in scan) == ("runs" in scan):
        raise ScanError(f"{path}: a scan is {{base, group, grid}} or {{base, group, runs}}, or a JSON list of configs")
    base = config.load(path.parent / scan["base"])
    if "grid" in scan:
        keys = list(scan["grid"])
        overs = [dict(zip(keys, vals)) for vals in itertools.product(*(scan["grid"][k] for k in keys))]
    else:
        overs = scan["runs"]
    return scan["group"], [_job(base, o, scan["group"]) for o in overs]


# ---- checks and classification --------------------------------------------------------------------------------------

def data_hashes(jobs) -> dict:
    """{(set, config): data hash or the error that says why the data set cannot be used}."""
    out = {}
    for j in jobs:
        d = j["cfg"]["data"]
        key = (d["set"], d["config"])
        if key not in out:
            try:
                out[key] = wavefm.data.dev(j["cfg"]).meta["data_hash"]
            except Exception as e:                     # missing, stale or damaged: reported, never trained on
                out[key] = e
    return out


def problems(jobs, hashes) -> list:
    """Everything that stops the scan before any run starts: unknown keys or stages, two runs of one name, data."""
    out = []
    for j in jobs:
        try:
            config.check(j["cfg"])
        except config.ConfigError as e:
            out.append(f"{j['name']}: {e.args[0]}")
    names = [j["name"] for j in jobs]
    out += [f"run name {n} appears {names.count(n)} times" for n in sorted(set(names)) if names.count(n) > 1]
    out += [f"data set {s}/{c}: {e}" for (s, c), e in hashes.items() if isinstance(e, Exception)]
    return out


def classify(cfg, data_hash) -> tuple:
    """("done" | "todo" | "refused", why) of one run from its directory (see the module docstring)."""
    d = pipeline.run_dir(cfg)
    if (d / "config.json").exists() and rundir.config_fp(load_json(d / "config.json")) != rundir.config_fp(cfg):
        return "refused", f"{d}/config.json holds another config: use another run name or group"
    stages = cfg["stages"][:cfg["stages"].index("compress")] if "compress" in cfg["stages"] else cfg["stages"]
    finished, prev = True, None
    for s in stages:
        out = d / s
        if (out / "result.json").exists():
            try:
                r = rundir.read_result(out)
                f = pipeline.handover(prev)                        # what the pipeline hands this stage
                got = None if r["parent"] is None else Path(r["parent"]["file"])
                same = f is None if got is None else f is not None and f.is_file() and got.resolve() == f.resolve()
                if not same:
                    return "refused", f"{out} started from {got}, but the pipeline hands it {f}"
                rundir.check_identity(out, cfg, data_hash, rundir.parent_record(f))
            except (rundir.RunError, pipeline.PipelineError) as e:
                return "refused", str(e)
        else:
            finished = False
        prev = out
    if not finished:
        return "todo", ""
    if "compress" in cfg["stages"]:
        return "awaiting compression", f"the stages before it finished with this identity; run " \
                                       f"{pipeline.compress_command(d)}"
    return "done", "every stage finished with this identity"


def after(cfg) -> str:
    """Status of a run whose pipeline ended without error or preemption."""
    if "compress" in cfg["stages"]:
        return f"awaiting compression: {pipeline.compress_command(pipeline.run_dir(cfg))}"
    return "finished"


# ---- the record -----------------------------------------------------------------------------------------------------

def _stage_record(run, stage) -> dict:
    s = run["stages"].get(stage) if run else None
    if s is None:
        return {"status": "not started"}
    row, = read.table([run], stage)
    try:
        ok = rundir.accepted(s["path"])
    except rundir.RunError as e:
        ok = f"no: {e}"
    return {"status": "finished" if row["finished"] else "unfinished",
            "requested": (s["result"] or {}).get("requested"), "actual": (s["result"] or {}).get("actual"),
            "final_train_loss": row["final_train_loss"], "final_val_loss": row["final_val_loss"],
            "hints": {k[5:]: v for k, v in row.items() if k.startswith("hint_")}, "verdict": s["verdict"],
            "accepted": ok}


def record(source, group, jobs) -> dict:
    by_name = {r["name"]: r for r in read.runs(paths.OUTPUT) if r["group"] == group}
    rec = {"group": group, "source": str(source), "written": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    for j in jobs:
        run = by_name.get(j["name"])
        rec["runs"].append({"name": j["name"], "overrides": j["overrides"], "path": str(pipeline.run_dir(j["cfg"])),
                            "seed": j["cfg"]["seed"], "status": j["status"],
                            "stages": {s: _stage_record(run, s) for s in j["cfg"]["stages"]}})
    save_json(rec, paths.OUTPUT / group / "scan.json")
    return rec


# ---- running --------------------------------------------------------------------------------------------------------

def allocation_end():
    """Unix time at which the current Slurm allocation ends, or None outside Slurm."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True).stdout.strip()
    if not left or left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days, _, hms = left.rpartition("-")
    secs = 0
    for part in hms.split(":"):
        secs = 60 * secs + int(part)
    return time.time() + secs + 86400 * int(days or 0)


def _worker(cfg, gpu, log):
    """One run in a forked process: its GPU, its log, the default SIGTERM (the trainer installs its own)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2)
    sys.stdout = sys.stderr = os.fdopen(fd, "w", buffering=1)
    try:
        code = PREEMPTED if pipeline.run(cfg)["preempted"] else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)


def execute(source, group, jobs, gpus, end):
    """Run the jobs of status "todo", at most one per GPU at a time, rewriting the record at every change."""
    ctx = multiprocessing.get_context("fork")
    queue, free, running, stopping = [j for j in jobs if j["status"] == "todo"], list(gpus), {}, []

    def on_term(*_):
        stopping.append(True)
        for p in running:
            p.terminate()
    old = signal.signal(signal.SIGTERM, on_term)
    try:
        while queue or running:
            while queue and free and not stopping:
                if end is not None and end - time.time() < MIN_LEFT_S:
                    for j in queue:
                        j["status"] = f"not started: less than {MIN_LEFT_S // 60} min left in the allocation"
                    queue.clear()
                    break
                j, gpu = queue.pop(0), free.pop(0)
                cfg = copy.deepcopy(j["cfg"])
                if end is not None:
                    cfg["run"]["deadline"] = end - MARGIN_S
                log = pipeline.run_dir(cfg) / "train.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                p = ctx.Process(target=_worker, args=(cfg, gpu, str(log)))
                p.start()
                running[p] = (gpu, j)
                j["status"] = f"running on GPU {gpu}"
                print(f"[scan] start {j['name']} on GPU {gpu}", flush=True)
                record(source, group, jobs)
            if stopping:
                for j in queue:
                    j["status"] = "not started: the scan was stopped (SIGTERM)"
                queue.clear()
            if running:
                multiprocessing.connection.wait([p.sentinel for p in running], timeout=30)
            for p in [p for p in running if p.exitcode is not None]:
                gpu, j = running.pop(p)
                free.append(gpu)
                j["status"] = {0: after(j["cfg"]), PREEMPTED: "preempted: start the scan again to resume"}.get(
                    p.exitcode, f"failed (exit code {p.exitcode}): see {pipeline.run_dir(j['cfg']) / 'train.log'}")
                print(f"[scan] {j['name']}: {j['status']}", flush=True)
                record(source, group, jobs)
    finally:
        signal.signal(signal.SIGTERM, old)


def visible_gpus(n) -> list:
    ids = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    ids = ids or [str(i) for i in range(n or 1)]
    n = n or len(ids)
    if not 1 <= n <= len(ids):
        raise ScanError(f"--gpus {n}, but the visible GPUs are {ids}")
    return ids[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan")
    ap.add_argument("--dry-run", action="store_true", help="expand, check and classify every config; start nothing")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs to use (default: every visible one)")
    a = ap.parse_args(argv)
    group, jobs = expand(a.scan)
    hashes = data_hashes(jobs)
    bad = problems(jobs, hashes)
    if bad:
        print(f"[scan] {a.scan}: {len(bad)} problem(s), nothing started:\n  " + "\n  ".join(bad), flush=True)
        return 2
    for j in jobs:
        d = j["cfg"]["data"]
        j["status"], why = classify(j["cfg"], hashes[(d["set"], d["config"])])
        if j["status"] != "todo":
            j["status"] = f"{j['status']}: {why}" + (", skipped" if j["status"] != "refused" else "")
    print(f"[scan] {group}: {len(jobs)} runs", flush=True)
    for j in jobs:
        print(f"  {j['name']}: {j['status']}", flush=True)
    if a.dry_run:
        print("[scan] dry run: nothing started", flush=True)
        return 0
    gpus = visible_gpus(a.gpus)
    record(a.scan, group, jobs)
    execute(a.scan, group, jobs, gpus, allocation_end())
    record(a.scan, group, jobs)
    failed = [j["name"] for j in jobs if j["status"].startswith(("failed", "refused"))]
    print(f"[scan] {group}: record in {paths.OUTPUT / group / 'scan.json'}"
          + (f"; failed or refused: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
