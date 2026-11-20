"""scripts/run_scan.py with fake stages (no training): expansion and checks (--dry-run), skipping only what finished
with the same identity, refusing what finished with another, never opening heldout, the external list of configs,
the readable record, the deadline and SIGTERM.

The fake stages are modules in sys.modules, as in test_pipeline.py; the scan forks one process per run, which
inherits them. What happens inside a run is written to files, since a forked process shares no memory."""
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import types
from pathlib import Path

import pytest

import fakes
import wavefm.data
from analysis import read
from data import store
from wavefm import paths
from wavefm.train import pipeline, rundir
from wavefm.utils.io import load_json, save_json

HIST = [{"step": s, "train_loss": t, "val_loss": v, "lr": 1e-3} for s, t, v in
        zip([0, 1, 2, 3], [4.0, 2.0, 1.0, 1.0], [5.0, 3.0, 2.0, 2.5])]


def _load_scan():
    spec = importlib.util.spec_from_file_location("run_scan", paths.ROOT / "scripts" / "run_scan.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(tmp_path, what):
    with open(tmp_path / "events.txt", "a") as f:
        f.write(json.dumps(what) + "\n")


def events(tmp_path):
    p = tmp_path / "events.txt"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def fake_stage(tmp_path, name, behaviour=None):
    """A stage as the real ones are: opens dev, reuses itself only through check_identity, writes curves, weights and
    result.json last. ``behaviour(cfg, out)`` may replace the training."""
    def run(cfg, out, parent):
        dev = wavefm.data.dev(cfg)
        rec = rundir.parent_record(pipeline.handover(parent))
        done = rundir.check_identity(out, cfg, dev.meta["data_hash"], rec)
        if done:
            return done
        _log(tmp_path, {"start": f"{cfg['run']['name']}/{name}", "deadline": cfg["run"]["deadline"],
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        if behaviour:
            r = behaviour(cfg, out)
            if r is not None:
                return r
        out.mkdir(parents=True, exist_ok=True)
        save_json(HIST, out / "history.json")
        steps, hand = cfg[name].get("steps", 0), pipeline.HANDOVER.get(name, "weights.pt")
        (out / hand).write_bytes(f"{name} {steps}".encode())
        return rundir.write_result(out, name, cfg, dev.meta["data_hash"], rec, requested={"steps": steps},
                                   actual={"steps": steps}, hints={}, wall_s=0.1, artifacts=["history.json", hand])
    return types.SimpleNamespace(run=run)


@pytest.fixture
def scan(tmp_path, monkeypatch):
    """(run_scan module, scan-writing helper). A data set fake/tiny (fakes.py) is the data, the output is
    tmp_path/out, the fake stages ssl and finetune are installed, and there is no Slurm allocation."""
    root = fakes.make_root(tmp_path)
    path, *_ = fakes.make_split(tmp_path, "npy", root=root)
    (root / "fake" / "configs" / "tiny.yaml").write_text(f"out: {path.parent}\nseed: 3\n")
    monkeypatch.setattr(wavefm.data, "ROOT", root)
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "out")
    for s in ("ssl", "finetune"):
        monkeypatch.setitem(sys.modules, f"wavefm.train.{s}", fake_stage(tmp_path, s))
    m = _load_scan()
    monkeypatch.setattr(m, "allocation_end", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / "base.yaml").write_text(f"defaults: {paths.ROOT / 'configs' / 'teacher.yaml'}\n"
                                        "data: {set: fake, config: tiny}\nstages: [ssl, finetune]\n"
                                        "ssl: {steps: 3}\nfinetune: {steps: 3}\n")

    def write(body, name="scan.yaml"):
        p = tmp_path / name
        p.write_text(body if isinstance(body, str) else json.dumps(body))
        return str(p)
    return m, write


GRID = "base: base.yaml\ngroup: lr\ngrid:\n  finetune.lr_head: [0.001, 0.003]\n  finetune.steps: [3]\n"
NAMES = ["finetune.lr_head=0.001_finetune.steps=3", "finetune.lr_head=0.003_finetune.steps=3"]


def starts(tmp_path):
    return sorted(e["start"] for e in events(tmp_path) if "start" in e)


# ---- ⑤ dry run ------------------------------------------------------------------------------------------------------

def test_dry_run_expands_checks_and_starts_nothing(scan, tmp_path, capsys):
    m, write = scan
    assert m.main([write(GRID), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert all(f"  {n}: todo" in out for n in NAMES) and "nothing started" in out
    assert not (tmp_path / "out").exists() and events(tmp_path) == []


@pytest.mark.parametrize("body, problem", [
    ("base: base.yaml\ngroup: g\ngrid:\n  finetune.lr_haed: [0.001]\n", "unknown config key finetune.lr_haed"),
    ("base: base.yaml\ngroup: g\nruns:\n  - {ssl.steps: 5}\n  - {ssl.steps: 5}\n", "appears 2 times"),
    ("base: base.yaml\ngroup: g\nruns:\n  - {stages: [ssl, finetuen]}\n", "stages ['ssl', 'finetuen']"),
    ("base: base.yaml\ngroup: g\nruns:\n  - {data.config: nothere}\n", "data set fake/nothere"),
])
def test_dry_run_reports_every_problem(scan, tmp_path, capsys, body, problem):
    m, write = scan
    for dry in (["--dry-run"], []):                       # the same checks stop a real scan before anything starts
        assert m.main([write(body)] + dry) == 2
        assert problem in capsys.readouterr().out
    assert not (tmp_path / "out").exists() and events(tmp_path) == []


def test_malformed_scan_file_refused(scan):
    m, write = scan
    with pytest.raises(m.ScanError, match="grid"):
        m.main([write("base: base.yaml\ngroup: g\ngrid: {ssl.steps: [1]}\nruns: [{}]\n"), "--dry-run"])
    with pytest.raises(m.ScanError, match="run.group"):
        m.main([write("base: base.yaml\ngroup: g\nruns: [{run.group: x}]\n"), "--dry-run"])


# ---- ③ identity, not existence --------------------------------------------------------------------------------------

def test_skip_only_with_the_same_identity(scan, tmp_path, capsys):
    m, write = scan
    f = write(GRID)
    assert m.main([f]) == 0
    assert starts(tmp_path) == sorted(f"{n}/{s}" for n in NAMES for s in ("ssl", "finetune"))
    assert m.main([f]) == 0                                                   # unchanged: skipped
    assert len(starts(tmp_path)) == 4 and "every stage finished with this identity" in capsys.readouterr().out
    run = tmp_path / "out" / "lr" / NAMES[0]
    sha = rundir.file_sha256(run / "finetune" / "result.json")

    # the config changed under the same run names: refused, nothing started or overwritten
    base = (tmp_path / "base.yaml").read_text()
    (tmp_path / "base.yaml").write_text(base + "seed: 1\n")
    assert m.main([f]) == 1
    rec = load_json(tmp_path / "out" / "lr" / "scan.json")
    assert all(r["status"].startswith("refused") and "holds another config" in r["status"] for r in rec["runs"])
    (tmp_path / "base.yaml").write_text(base)

    # the weights a later stage started from changed
    (run / "ssl" / "weights.pt").write_bytes(b"other weights")
    assert m.main([f]) == 1
    rec = load_json(tmp_path / "out" / "lr" / "scan.json")
    assert "another parent weights" in rec["runs"][0]["status"] and rec["runs"][1]["status"].startswith("done")

    # the data set was rebuilt from changed code: another data hash
    shutil.rmtree(tmp_path / "npy")
    (wavefm.data.ROOT / "fake" / "build.py").write_text("# fake, changed\n")
    fakes.make_split(tmp_path, "npy", root=wavefm.data.ROOT)
    assert m.main([f]) == 1
    rec = load_json(tmp_path / "out" / "lr" / "scan.json")
    assert all("another data set" in r["status"] for r in rec["runs"])
    assert len(starts(tmp_path)) == 4 and rundir.file_sha256(run / "finetune" / "result.json") == sha


def test_a_stage_without_the_expected_parent_is_refused(scan, tmp_path):
    """The expected parent comes from the pipeline: the stage before and its hand-over file. A later stage that
    records no parent (copied from a standalone run) or another file of the stage before is refused, not done."""
    m, write = scan
    f = write(GRID)
    assert m.main([f]) == 0
    run = tmp_path / "out" / "lr" / NAMES[0]
    cfg = load_json(run / "config.json")
    r = rundir.read_result(run / "finetune")
    (run / "ssl" / "other.pt").write_bytes(b"x")
    for parent in (None, rundir.parent_record(run / "ssl" / "other.pt")):
        rundir.write_result(run / "finetune", "finetune", cfg, r["data_hash"], parent, requested=r["requested"],
                            actual=r["actual"], hints={}, wall_s=0.1, artifacts=["history.json", "weights.pt"])
        status, why = m.classify(cfg, r["data_hash"])
        assert status == "refused" and "the pipeline hands it" in why and str(run / "ssl" / "weights.pt") in why


def test_a_run_with_compress_awaits_compression(scan, tmp_path, monkeypatch):
    """compress runs in the other environment: a run that includes it is never reported finished by the scan."""
    m, write = scan
    for s in ("ridge", "student"):
        monkeypatch.setitem(sys.modules, f"wavefm.train.{s}", fake_stage(tmp_path, s))
    f = write("base: base.yaml\ngroup: c\nruns:\n  - {stages: [ridge, student, compress]}\n")
    for again in (False, True):
        assert m.main([f]) == 0
        st, = [r["status"] for r in load_json(tmp_path / "out" / "c" / "scan.json")["runs"]]
        assert st.startswith("awaiting compression") and "compression/compress.py" in st
        assert st.endswith("skipped") == again and len(starts(tmp_path)) == 2


def test_an_unfinished_run_is_started_again(scan, tmp_path):
    m, write = scan
    f = write(GRID)
    assert m.main([f]) == 0
    (tmp_path / "out" / "lr" / NAMES[1] / "finetune" / "result.json").unlink()
    assert m.main([f]) == 0
    assert starts(tmp_path).count(f"{NAMES[1]}/finetune") == 2 and starts(tmp_path).count(f"{NAMES[1]}/ssl") == 1


# ---- ④ heldout ------------------------------------------------------------------------------------------------------

def test_the_scan_never_opens_heldout(scan, tmp_path, monkeypatch):
    m, write = scan
    init = store.Split.__init__

    def recording(self, path):
        _log(tmp_path, {"opened": str(path)})
        init(self, path)
    monkeypatch.setattr(store.Split, "__init__", recording)
    monkeypatch.setattr(wavefm.data, "heldout", lambda cfg: _log(tmp_path, {"opened": "wavefm.data.heldout"}))
    assert m.main([write(GRID)]) == 0
    opened = [e["opened"] for e in events(tmp_path) if "opened" in e]
    assert len(opened) >= 5 and all(Path(p).name == "dev" for p in opened), opened    # the driver and 4 stages


# ---- ⑥ external configs and the record ------------------------------------------------------------------------------

def test_external_list_of_configs_and_the_record(scan, tmp_path):
    m, write = scan
    full = m.config.load(tmp_path / "base.yaml")
    full["run"]["group"] = "ext"
    del full["run"]["name"]                      # named after its difference from teacher.yaml
    full["finetune"]["steps"] = 7
    part = {"run": {"group": "ext", "name": "short"}, "data": {"set": "fake", "config": "tiny"},
            "stages": ["finetune"], "finetune": {"steps": 2, "lr_head": 0.01}}
    assert m.main([write([full, part], "configs.json")]) == 0
    rec = load_json(tmp_path / "out" / "ext" / "scan.json")
    long_name = "data.set=fake_data.config=tiny_stages=ssl-finetune_finetune.steps=7_ssl.steps=3"
    assert [r["name"] for r in rec["runs"]] == [long_name, "short"]
    assert rec["runs"][1]["overrides"] == {"data.set": "fake", "data.config": "tiny", "finetune.steps": 2,
                                           "finetune.lr_head": 0.01}     # stages [finetune] is the default
    a, b = rec["runs"]
    assert a["status"] == b["status"] == "finished" and a["path"] == str(tmp_path / "out" / "ext" / long_name)
    ft = b["stages"]["finetune"]
    assert ft["status"] == "finished" and ft["requested"] == ft["actual"] == {"steps": 2}
    assert (ft["final_train_loss"], ft["final_val_loss"]) == (1.0, 2.5)
    assert ft["hints"] == {"on": "val_loss", "final": 2.5, "min": 2.0, "min_step": 2, "final_over_min": 0.25,
                           "last10_drop": -0.25}
    assert ft["verdict"] is None and ft["accepted"] is False
    assert a["stages"]["ssl"]["actual"] == {"steps": 3} and a["stages"]["finetune"]["actual"] == {"steps": 7}
    runs = {r["name"]: r for r in read.runs(tmp_path / "out")}                # the run directories, read the same way
    assert sorted(runs) == sorted([long_name, "short"]) and runs["short"]["cfg"]["finetune"]["lr_head"] == 0.01
    rundir.write_verdict(tmp_path / "out" / "ext" / "short" / "finetune", "pass", "flat", "test")
    assert m.main([str(tmp_path / "configs.json")]) == 0                       # done: skipped, the record updated
    rec = load_json(tmp_path / "out" / "ext" / "scan.json")
    assert rec["runs"][1]["status"].startswith("done") and rec["runs"][1]["stages"]["finetune"]["accepted"] is True


# ---- deadline and SIGTERM -------------------------------------------------------------------------------------------

def test_deadline_from_the_allocation(scan, tmp_path, monkeypatch):
    m, write = scan
    end = time.time() + 3600
    monkeypatch.setattr(m, "allocation_end", lambda: end)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert m.main([write(GRID)]) == 0
    ev = [e for e in events(tmp_path) if "start" in e]
    assert {e["deadline"] for e in ev} == {end - m.MARGIN_S} and {e["gpu"] for e in ev} <= {"5", "7"}
    monkeypatch.setattr(m, "allocation_end", lambda: time.time() + 60)        # too little left: nothing starts
    shutil.rmtree(tmp_path / "out")
    assert m.main([write(GRID)]) == 0
    rec = load_json(tmp_path / "out" / "lr" / "scan.json")
    assert all(r["status"].startswith("not started: less than") for r in rec["runs"])
    assert len([e for e in events(tmp_path) if "start" in e]) == 4


def test_sigterm_is_passed_on(scan, tmp_path, monkeypatch):
    """The first run sends SIGTERM to the scan; the scan terminates it (the stage stops cleanly, as the trainer does)
    and starts nothing more."""
    m, write = scan
    before = signal.getsignal(signal.SIGTERM)

    def stop(cfg, out):
        got = []
        signal.signal(signal.SIGTERM, lambda *_: got.append(1))
        os.kill(os.getppid(), signal.SIGTERM)
        for _ in range(200):
            if got:
                return {"preempted": True}
            time.sleep(0.05)
        raise RuntimeError("SIGTERM was not passed on")
    monkeypatch.setitem(sys.modules, "wavefm.train.ssl", fake_stage(tmp_path, "ssl", stop))
    assert m.main([write(GRID), "--gpus", "1"]) == 0
    rec = load_json(tmp_path / "out" / "lr" / "scan.json")
    assert [r["status"] for r in rec["runs"]] == ["preempted: start the scan again to resume",
                                                  "not started: the scan was stopped (SIGTERM)"]
    assert signal.getsignal(signal.SIGTERM) is before and starts(tmp_path) == [f"{NAMES[0]}/ssl"]


def test_more_scan_guards(tmp_path, monkeypatch):
    """more GPUs asked for than visible; a list of configs of
    several groups."""
    rs = _load_scan()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert rs.visible_gpus(2) == ["0", "1"]
    with pytest.raises(rs.ScanError, match="--gpus 3"):
        rs.visible_gpus(3)
    f = tmp_path / "list.json"
    f.write_text(json.dumps([{"run": {"group": "a"}}, {"run": {"group": "b"}}]))
    with pytest.raises(rs.ScanError, match="same run.group"):
        rs.expand(f)
