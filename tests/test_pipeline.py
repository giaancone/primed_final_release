"""The stage pipeline with fake stages injected as wavefm.train.<stage> modules: order, out and parent, failure,
preemption, compress, the config guard, and the parent-weights identity across stages."""
import os
import signal
import sys
import types

import pytest
import torch

from wavefm import paths
from wavefm.train import pipeline, rundir
from wavefm.utils import config
from wavefm.utils.io import load_json

DATA_HASH = "fakedata"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "out")
    c = config.load(paths.ROOT / "configs" / "teacher.yaml")
    c["run"].update(group="test", name="pipe")
    c["stages"] = ["ssl", "ridge", "student"]
    return c


def install(monkeypatch, calls, **behaviour):
    """Fake stage modules: each records (stage, out, parent) and returns or raises what ``behaviour`` says."""
    for stage in ("ssl", "ridge", "student", "finetune", "compress"):
        def run(cfg, out, parent, stage=stage):
            calls.append((stage, out, parent))
            b = behaviour.get(stage, {"stage": stage})
            if isinstance(b, Exception):
                raise b
            return b
        monkeypatch.setitem(sys.modules, f"wavefm.train.{stage}", types.SimpleNamespace(run=run))


def test_stages_run_in_order_with_out_and_parent(cfg, monkeypatch):
    calls = []
    install(monkeypatch, calls)
    r = pipeline.run(cfg)
    d = paths.OUTPUT / "test" / "pipe"
    assert calls == [("ssl", d / "ssl", None), ("ridge", d / "ridge", d / "ssl"), ("student", d / "student", d / "ridge")]
    assert r == {"run": str(d), "results": {s: {"stage": s} for s in ("ssl", "ridge", "student")}, "preempted": False,
                 "stopped_before": None}
    assert rundir.config_fp(load_json(d / "config.json")) == rundir.config_fp(cfg)


def test_a_failing_stage_stops_the_pipeline(cfg, monkeypatch):
    calls = []
    install(monkeypatch, calls, ridge=RuntimeError("ridge broke"))
    with pytest.raises(RuntimeError, match="ridge broke"):
        pipeline.run(cfg)
    assert [c[0] for c in calls] == ["ssl", "ridge"]


def test_a_preempted_stage_stops_cleanly(cfg, monkeypatch):
    calls = []
    install(monkeypatch, calls, ssl={"preempted": True})
    r = pipeline.run(cfg)
    assert [c[0] for c in calls] == ["ssl"] and r["preempted"] and r["results"] == {"ssl": {"preempted": True}}


def test_a_stop_while_a_stage_finishes_starts_no_further_stage(cfg, monkeypatch):
    """A SIGTERM that a stage lets finish (the trainer passes on one that arrives in its terminal evaluation; ridge has
    no loop) or a deadline passed meanwhile: that stage completes, the next is not started, the run is preempted."""
    got = []
    previous = signal.signal(signal.SIGTERM, lambda *a: got.append(a[0]))      # the caller's handler, restored after
    try:
        calls = []
        install(monkeypatch, calls)
        ssl = sys.modules["wavefm.train.ssl"].run
        monkeypatch.setattr(sys.modules["wavefm.train.ssl"], "run",
                            lambda *a: (os.kill(os.getpid(), signal.SIGTERM), ssl(*a))[1])
        r = pipeline.run(cfg)
        assert [c[0] for c in calls] == ["ssl"] and r["preempted"] and r["results"] == {"ssl": {"stage": "ssl"}}
        assert got == []                                                           # the pipeline caught it
        calls.clear()
        monkeypatch.setattr(sys.modules["wavefm.train.ssl"], "run", ssl)
        cfg["run"]["deadline"] = 1.0                                               # long past
        r = pipeline.run(cfg)
        assert calls == [] and r["preempted"]
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_one_handover_file_per_stage(tmp_path):
    assert pipeline.handover(None) is None
    assert pipeline.handover(tmp_path / "ssl") == tmp_path / "ssl" / "weights.pt"
    assert pipeline.handover(tmp_path / "ridge") == tmp_path / "ridge" / "teacher_labels.npz"
    with pytest.raises(pipeline.PipelineError, match="hands nothing"):
        pipeline.handover(tmp_path / "student")


def test_compress_is_not_run_here(cfg, monkeypatch, capsys):
    calls = []
    install(monkeypatch, calls)
    cfg["stages"] = ["ridge", "student", "compress"]
    r = pipeline.run(cfg)
    assert [c[0] for c in calls] == ["ridge", "student"] and r["stopped_before"] == "compress"
    assert f"uv run --frozen --project compression python compression/compress.py {r['run']}" in capsys.readouterr().out


@pytest.mark.parametrize("stages, targets", [(["student", "compress"], "teacher"), (["ridge", "compress"], "teacher"),
                                             (["ridge", "student", "compress"], "truth")])
def test_compress_without_a_teacher_label_student_is_refused_first(cfg, monkeypatch, stages, targets):
    calls = []
    install(monkeypatch, calls)
    cfg["stages"], cfg["student"]["targets"] = stages, targets
    with pytest.raises(pipeline.PipelineError, match="compress distils on the teacher labels"):
        pipeline.run(cfg)
    assert calls == [] and not (paths.OUTPUT / "test" / "pipe").exists()


def test_another_config_under_the_same_name_is_refused(cfg, monkeypatch):
    calls = []
    install(monkeypatch, calls)
    pipeline.run(cfg)
    pipeline.run(cfg)                                    # the same config again: allowed (the stages decide what is done)
    assert len(calls) == 6
    cfg["seed"] = 1
    with pytest.raises(pipeline.PipelineError, match="another config"):
        pipeline.run(cfg)
    assert len(calls) == 6


def test_the_config_is_checked_first(cfg, monkeypatch):
    install(monkeypatch, [])
    cfg["model"]["typo"] = 1
    with pytest.raises(config.ConfigError):
        pipeline.run(cfg)
    assert not (paths.OUTPUT / "test" / "pipe").exists()


def test_changed_parent_weights_are_detected(cfg, monkeypatch):
    """The mechanism the real stages use: a stage records rundir.parent_record of the file it starts from, and
    rundir.check_identity refuses a finished stage whose parent file has changed since."""
    calls = []

    def ssl(cfg, out, parent):
        calls.append("ssl")
        done = rundir.check_identity(out, cfg, DATA_HASH, None)
        if done:
            return done
        out.mkdir(parents=True)
        torch.save({"encoder.w": torch.ones(2), "ssl_next.w": torch.ones(3)}, out / "weights.pt")
        return rundir.write_result(out, "ssl", cfg, DATA_HASH, None, requested={}, actual={}, hints={}, wall_s=0)

    def ridge(cfg, out, parent):
        calls.append("ridge")
        rec = rundir.parent_record(parent / "weights.pt")
        done = rundir.check_identity(out, cfg, DATA_HASH, rec)
        if done:
            return done
        out.mkdir(parents=True)
        return rundir.write_result(out, "ridge", cfg, DATA_HASH, rec, requested={}, actual={}, hints={}, wall_s=0)

    monkeypatch.setitem(sys.modules, "wavefm.train.ssl", types.SimpleNamespace(run=ssl))
    monkeypatch.setitem(sys.modules, "wavefm.train.ridge", types.SimpleNamespace(run=ridge))
    cfg["stages"] = ["ssl", "ridge"]
    first = pipeline.run(cfg)
    assert pipeline.run(cfg)["results"] == first["results"]          # finished and unchanged: reused
    w = paths.OUTPUT / "test" / "pipe" / "ssl" / "weights.pt"
    torch.save({"encoder.w": 2 * torch.ones(2), "ssl_next.w": torch.ones(3)}, w)
    with pytest.raises(rundir.RunError, match="another parent weights"):
        pipeline.run(cfg)
    assert calls == ["ssl", "ridge"] * 3
