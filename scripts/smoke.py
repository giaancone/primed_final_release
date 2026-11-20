"""The smoke test of the whole chain, across the two environments: one command per task.

Usage: python scripts/smoke.py <dro|dch> [run name]       (after ``source setup.sh``; any working directory)

It runs configs/smoke/<task>.yaml (the paper config on the small data set of the task, real TimesFM weights, every
trained stage 20 steps or fewer) as run ``<output>/smoke/<name>`` (default name: ``<task>-<date>-<time>``, so a
rerun never meets a finished run of another identity) in four parts:

    train      python scripts/train.py configs/smoke/<task>.yaml       ssl -> ridge -> student, stops before compress
    compress   uv run --frozen --project compression python compression/compress.py <run>     (other environment)
    evaluate   python scripts/evaluate.py <run>                        the student on heldout, once
    evaluate compressed   ... compression/compress.py --evaluate <run> the quantised model on heldout, once

and after each part checks, failing with exit code 1 otherwise:
  * every stage of the part finished: rundir.read_result accepts its result.json, ``smoke`` is true, every promised
    artifact exists with its recorded size and sha256, every number in result.json and every loss in history.json is
    finite;
  * the real summary gate refuses the run: a ``pass`` verdict is written on every stage that has curves (judge
    "scripts/smoke.py", it judges nothing) and analysis.summary must still leave the run out, because it is smoke;
  * reload gives the same predictions: student.predict on every dev row from student.npz, in two fresh processes,
    gives the same bytes, and the loss on the student's validation rows recomputed from those predictions equals the
    last val_loss the training loop recorded (torch); model.h5 reloaded in two fresh processes of the compression
    environment gives the same validation predictions, and their loss equals the val_loss compress recorded;
  * each heldout.json (<run>/heldout.json for the student, <run>/compress/heldout.json) names the evaluated stage, is
    smoke, and records the sha256 of that stage's present result.json.
Wall time of every part and the total are printed. Smoke numbers are not results.
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("WAVEFM_OUTPUT", ROOT / "outputs"))
GROUP = "smoke"
COMPRESSION = ["uv", "run", "--frozen", "--project", str(ROOT / "compression"), "python"]
TOL = 1e-5          # relative: the same loss computed in numpy and in torch (or TensorFlow), float32 predictions


class SmokeError(RuntimeError):
    pass


def sh(cmd) -> str:
    print("[smoke] $ " + " ".join(map(str, cmd)), flush=True)
    p = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, end="", flush=True)
    if p.returncode != 0:
        raise SmokeError(f"exit code {p.returncode}: {' '.join(map(str, cmd))}")
    return p.stdout


def numbers(x, where):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, f"{where}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, f"{where}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield where, x


def check_stage(run, stage):
    from wavefm.train import rundir
    out = run / stage
    r = rundir.read_result(out)
    if r["smoke"] is not True:
        raise SmokeError(f"{out}: result.json says smoke={r['smoke']}")
    for a, f in r["artifacts"].items():
        if not (out / a).is_file() or (out / a).stat().st_size != f["bytes"] or rundir.file_sha256(out / a) != f["sha256"]:
            raise SmokeError(f"{out}: promised artifact {a} is missing or differs from what result.json records")
    bad = [w for w, v in numbers(r, "result") if not math.isfinite(v)]
    if (out / "history.json").exists():
        hist = json.loads((out / "history.json").read_text())
        first = {i for i, h in enumerate(hist) if h["step"] == 0 or h.get("phase") != hist[i - 1].get("phase")}
        bad += [f"history[{i}].{k}" for i, h in enumerate(hist) for k, v in h.items()
                if "loss" in k and not (v is None and k == "train_loss" and i in first)   # null at a phase's start
                and not (v is not None and math.isfinite(v))]
    if bad:
        raise SmokeError(f"{out}: non-finite values {bad[:5]}")
    print(f"[smoke] ok {stage}: result.json read, smoke, {len(r['artifacts'])} artifacts, finite", flush=True)
    return r


def check_summary_refuses(run, stages):
    from analysis import read, summary
    from wavefm.train import rundir
    for s in stages:
        if s not in rundir.NO_CURVE and not (run / s / "verdict.json").exists():
            rundir.write_verdict(run / s, "pass", "gate check of the smoke test, not a judgement", "scripts/smoke.py")
    mine = [r for r in read.runs(run.parent.parent) if Path(r["path"]) == run]
    for s in stages:
        why = dict((str(r["path"]), w) for r, w in summary.rejected(mine, s)).get(str(run))
        if summary.select(mine, s) or why is None or rundir.accepted(run / s):
            raise SmokeError(f"{run}: the summary gate accepts the smoke stage {s}")
    print(f"[smoke] ok summary refuses {stages} (\"{why}\")", flush=True)


STUDENT_PASS = """
import hashlib, json, sys
import numpy as np
import wavefm.data
from wavefm.train import rundir, student
run = sys.argv[1]
cfg = json.load(open(f"{run}/config.json"))
dev = wavefm.data.dev(cfg)
pred = student.predict(f"{run}/student", dev)
_, _, spec = rundir.read_student(f"{run}/student/student.npz", dev.meta["data_hash"])
val = np.load(f"{run}/student/rows.npz")["val"]
y = (rundir.read_teacher_labels(f"{run}/ridge/teacher_labels.npz", dev)[val] - spec["output"]["offset"]) / spec["output"]["scale"]
p = (pred[val] - spec["output"]["offset"]) / spec["output"]["scale"]
w = np.asarray(cfg["student"]["loss_weights"] or [1.0] * y.shape[1])
print(json.dumps({"sha256": hashlib.sha256(pred.tobytes()).hexdigest(), "val_loss": float(((p - y) ** 2 * w).sum(1).mean())}))
"""

COMPRESS_PASS = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[2])
import numpy as np
import compress as c
run = sys.argv[1]
cfg = json.load(open(f"{run}/config.json"))
dev = c.split(cfg, "dev")
_, _, spec = c.read_student(f"{run}/student/student.npz", dev.meta["data_hash"])
_, val = c.read_rows(f"{run}/student/rows.npz", dev.n)
y = ((c.read_teacher_labels(f"{run}/ridge/teacher_labels.npz", dev)[val] - spec["output"]["offset"]) / spec["output"]["scale"]).astype(np.float32)
model = c.load_model(f"{run}/compress/model.h5")      # strict fp32, as --evaluate
p = c.predict(model, c.inputs(dev, val, spec))
print(json.dumps({"sha256": hashlib.sha256(p.tobytes()).hexdigest(), "val_loss": float(np.mean((p.astype(np.float64) - y) ** 2))}))
"""


def two_passes(cmd, recorded, what):
    check_reload(*(json.loads(sh(cmd).strip().splitlines()[-1]) for _ in range(2)), recorded, what)


def check_reload(a, b, recorded, what):
    """The outputs {sha256, val_loss} of two fresh reload processes against the loss the stage recorded."""
    if not all(math.isfinite(x) for x in (a["val_loss"], b["val_loss"], recorded)):
        raise SmokeError(f"{what}: non-finite validation loss (reloaded {a['val_loss']}, {b['val_loss']}; "
                         f"recorded {recorded})")
    if a["sha256"] != b["sha256"]:
        raise SmokeError(f"{what}: two fresh processes give different predictions")
    if abs(a["val_loss"] - recorded) > TOL * abs(recorded):
        raise SmokeError(f"{what}: reloaded validation loss {a['val_loss']!r} != recorded {recorded!r}")
    print(f"[smoke] ok reload {what}: identical predictions in two processes, val loss {a['val_loss']:.8g} "
          f"(recorded {recorded:.8g})", flush=True)


def check_heldout(run, path, stage):
    """A heldout record of scripts/evaluate.py's form names ``stage``, is smoke and identifies its present result."""
    from wavefm.train import rundir
    h = json.loads(Path(path).read_text())
    want = (stage, True, rundir.file_sha256(run / stage / "result.json"))
    if (h["stage"], h["result"]["smoke"], h["result"]["sha256"]) != want:
        raise SmokeError(f"{path} evaluated stage {h['stage']} (smoke {h['result']['smoke']}), not the present "
                         f"result.json of the smoke stage {stage}")
    print(f"[smoke] ok {path}: {stage}, n {h['n']}, {h['metrics']}", flush=True)
    return h


def main(argv) -> int:
    if len(argv) not in (1, 2) or argv[0] not in ("dro", "dch"):
        print(__doc__.split("\n\n")[1])
        return 2
    task = argv[0]
    name = argv[1] if len(argv) == 2 else f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
    run, py, wall, t_all = OUTPUT / GROUP / name, sys.executable, {}, time.time()

    def part(label, fn):
        t = time.time()
        print(f"[smoke] ---- {label}", flush=True)
        fn()
        wall[label] = time.time() - t
        print(f"[smoke] ---- {label} done in {wall[label]:.1f} s", flush=True)

    def train():
        out = sh([py, ROOT / "scripts/train.py", ROOT / f"configs/smoke/{task}.yaml", f"run.group={GROUP}", f"run.name={name}"])
        if "stopping before compress" not in out:
            raise SmokeError("the pipeline did not stop before compress")
        for s in ("ssl", "ridge", "student"):
            check_stage(run, s)
        check_summary_refuses(run, ["ssl", "ridge", "student"])
        hist = json.loads((run / "student/history.json").read_text())
        two_passes([py, "-c", STUDENT_PASS, run], hist[-1]["val_loss"], "student.npz")

    def compress():
        sh(COMPRESSION + [ROOT / "compression/compress.py", run])
        r = check_stage(run, "compress")
        check_summary_refuses(run, ["ssl", "ridge", "student", "compress"])
        two_passes(COMPRESSION + ["-c", COMPRESS_PASS, run, ROOT / "compression"], r["val_loss"]["final"], "model.h5")

    def evaluate():
        sh([py, ROOT / "scripts/evaluate.py", run])
        check_heldout(run, run / "heldout.json", "student")

    def evaluate_compressed():
        sh(COMPRESSION + [ROOT / "compression/compress.py", "--evaluate", run])
        check_heldout(run, run / "compress/heldout.json", "compress")

    try:
        for label, fn in (("train", train), ("compress", compress), ("evaluate", evaluate),
                          ("evaluate compressed", evaluate_compressed)):
            part(label, fn)
    except Exception as e:
        traceback.print_exc()
        print(f"[smoke] FAILED in part {len(wall) + 1} ({run}): {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[smoke] PASSED {task} {run}: " + ", ".join(f"{k} {v:.1f} s" for k, v in wall.items())
          + f"; total {time.time() - t_all:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
