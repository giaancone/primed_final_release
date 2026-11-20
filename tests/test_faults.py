"""Fault injection: every error below is introduced on purpose and must be caught, never silently trained through.
The trainer's guards run on the tiny synthetic problem of test_trainer.py, the stage guards on the fake data sets."""
import threading
import time

import numpy as np
import pytest
import torch

from test_trainer import N, base_cfg, cnn_cfg, problem, train
from wavefm import paths, tasks
from wavefm.data import splits
from wavefm.train import finetune, guards, rundir, trainer
from wavefm.utils import config, runtime
from wavefm.utils.io import load_json


@pytest.fixture(autouse=True)
def strict():
    runtime.setup_numerics()
    yield
    runtime.setup_numerics()


# ---- ② brakes: NaN / inf in the loss or a gradient, divergence; no result.json --------------------------------------

def nan_after(loss, k, where):
    n = [0]

    def f(b, gen):
        n[0] += 1
        out = loss(b, gen)
        if n[0] >= k and where == "loss":
            return out * float("nan")
        if n[0] >= k and where == "gradient":                 # finite loss, NaN gradient
            return out + 0 * torch.sqrt(out.detach() * 0 + 0 * out)
        if n[0] >= k and where == "divergence":
            return out * 1e6
        return out
    return f


@pytest.mark.parametrize("where", ["loss", "gradient", "divergence"])
def test_brakes_stop_the_run_without_a_result(tmp_path, where):
    model, groups, loss, validate = problem()
    cfg = base_cfg(steps=200, eval_points=2)
    with pytest.raises(trainer.TrainingBraked, match="diverged" if where == "divergence" else "non-finite"):
        trainer.train(tmp_path, cfg, "finetune", model, groups, nan_after(loss, 7, where), N, data_hash="d",
                      parent=None, validate=validate)
    assert not (tmp_path / "result.json").exists() and not (tmp_path / "weights.pt").exists()


def test_nan_validation_loss_brakes(tmp_path):
    with pytest.raises(trainer.TrainingBraked, match="validation"):
        train(tmp_path, base_cfg(), validate=lambda: float("nan"))


@pytest.mark.parametrize("where", ["model", "extra"])
def test_non_finite_weights_after_the_final_update_brake(tmp_path, where):
    """The last update makes a weight infinite while every loss and gradient norm stays finite, and there is no
    validation set: the run is braked, and nothing is saved as a success."""
    model, groups, loss, _ = problem(dropout=0.0)
    aux = torch.zeros(3)                                     # an extra tensor (not trained)
    n = [0]

    def last_breaks(b, gen):
        n[0] += 1
        out = loss(b, gen)
        if n[0] == 40:
            with torch.no_grad():
                (model[0].bias if where == "model" else aux).fill_(float("inf"))   # bias: not saved by autograd
        return out
    with pytest.raises(trainer.TrainingBraked, match="non-finite weights after step 40"):
        trainer.train(tmp_path, base_cfg(), "finetune", model, groups, last_breaks, N, data_hash="d", parent=None,
                      extra={"aux": aux})
    assert not (tmp_path / "weights.pt").exists() and not (tmp_path / "result.json").exists()


def test_braked_stage_writes_no_result(fake_dro, monkeypatch):
    real = trainer.train
    monkeypatch.setattr(trainer, "train", lambda out, cfg, stage, model, groups, loss, *a, **k: real(
        out, cfg, stage, model, groups, lambda b, g: loss(b, g) * float("inf"), *a, **k))
    cfg = cnn_cfg(fake_dro)
    with pytest.raises(trainer.TrainingBraked):
        finetune.run(cfg, paths.OUTPUT / "braked" / "finetune", None)
    assert not (paths.OUTPUT / "braked" / "finetune" / "result.json").exists()
    assert not (paths.OUTPUT / "braked" / "finetune" / "run.lock").exists()


# ---- ⑤ strict fp32 ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fault", ["matmul_tf32", "cudnn_tf32", "precision", "autocast", "bf16_parameter"])
def test_tf32_autocast_or_non_fp32_refused(tmp_path, fault):
    model, groups, loss, validate = problem()
    if fault == "matmul_tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif fault == "cudnn_tf32":
        torch.backends.cudnn.allow_tf32 = True
    elif fault == "precision":
        torch.set_float32_matmul_precision("high")
    elif fault == "bf16_parameter":
        model[3].to(torch.bfloat16)
    run = lambda: trainer.train(tmp_path, base_cfg(), "finetune", model, groups, loss, N, data_hash="d", parent=None,
                                validate=validate)
    with pytest.raises(guards.GuardError, match="TF32|autocast|float32"):
        if fault == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                run()
        else:
            run()


def test_setup_numerics_is_strict_and_the_stage_sets_it(fake_dro):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    finetune.run(cnn_cfg(fake_dro, steps=2, eval_every=1), paths.OUTPUT / "n" / "finetune", None)
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    guards.check_strict_fp32(torch.nn.Linear(2, 2))


# ---- ⑥ rows: train and validation disjoint, inside the label budget --------------------------------------------------

def test_rows_disjoint_and_inside_the_budget(fake_dro, monkeypatch):
    used = {}
    real_get = tasks.get

    def spy(name):
        t = real_get(name)
        prep = t.prepare

        def prepare(cfg, device):
            tr, va = prep(cfg, device)
            used.update(train=tr.idx.copy(), val=va.idx.copy())
            return tr, va
        t.prepare = prepare
        return t
    monkeypatch.setattr(tasks, "get", spy)
    cfg = cnn_cfg(fake_dro, steps=5)
    cfg["data"].update(label_frac=0.3, val_frac=0.2)
    out = paths.OUTPUT / "rows" / "finetune"
    r = finetune.run(cfg, out, None)
    rows = np.load(out / "rows.npz")
    assert (rows["train"] == used["train"]).all() and (rows["val"] == used["val"]).all()   # what the stage used
    budget = np.random.default_rng(cfg["seed"]).permutation(400)[:120]                    # 0.3 of 400 dev rows
    assert not set(rows["train"]) & set(rows["val"])
    assert set(rows["train"]) | set(rows["val"]) == set(budget) and len(rows["val"]) == 24
    assert r["n_train"] == 96 and r["n_val"] == 24
    tr, va = splits.labelled(400, 0.3, 0.2, cfg["seed"], 32, 8)
    assert (tr == rows["train"]).all() and (va == rows["val"]).all()


# ---- ⑧ resume refused when the config, the data or the parent weights changed ----------------------------------------

def test_resume_refused_when_identity_changed(tmp_path):
    cfg = base_cfg()
    early = config.merge(cfg, {"run": {"deadline": 1.0}})
    assert train(tmp_path, early)[0] == {"preempted": True} and (tmp_path / "ckpt.pt").exists()
    changed = [(config.merge(cfg, {"finetune": {"lr_head": 0.5}}), "d" * 64, None, "another config"),
               (cfg, "e" * 64, None, "another data set"),
               (cfg, "d" * 64, {"file": "w.pt", "sha256": "f" * 64}, "another parent weights")]
    for c, data_hash, parent, msg in changed:
        model, groups, loss, validate = problem()
        with pytest.raises(trainer.TrainerError, match=msg):
            trainer.train(tmp_path, c, "finetune", model, groups, loss, N, data_hash=data_hash, parent=parent,
                          validate=validate)
    names = config.merge(cfg, {"run": {"name": "renamed", "group": "g"}, "ridge": {"block": 3}})
    assert train(tmp_path, names)[0]["actual"]["steps"] == 40       # what is not this stage's identity: resumes


def test_finished_stage_with_another_identity_is_refused(fake_dro):
    cfg = cnn_cfg(fake_dro, steps=4, eval_every=2)
    out = paths.OUTPUT / "id" / "finetune"
    r = finetune.run(cfg, out, None)
    assert finetune.run(cfg, out, None) == r
    with pytest.raises(rundir.RunError, match="another config"):
        finetune.run(config.merge(cfg, {"finetune": {"lr_head": 0.5}}), out, None)


# ---- ⑨ and the controls: refused before any work ---------------------------------------------------------------------

def no_work(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("work started before the config was checked")
    monkeypatch.setattr(tasks, "get", boom)
    monkeypatch.setattr(finetune, "build_model", boom)


def test_cnn_with_dch_refused_before_any_work(fake_dch, monkeypatch):
    no_work(monkeypatch)
    cfg = config.merge(fake_dch, {"model": {"kind": "cnn"}, "finetune": {"steps": 5}})
    with pytest.raises(finetune.FinetuneError, match="cnn is implemented for task dro only"):
        finetune.run(cfg, paths.OUTPUT / "x" / "finetune", None)


@pytest.mark.parametrize("blocks", ["random", "none"])
def test_controls_with_trained_blocks_refused(fake_dro, monkeypatch, blocks):
    no_work(monkeypatch)
    cfg = config.merge(fake_dro, {"model": {"blocks": blocks}, "finetune": {"steps": 5, "train_layers": 2}})
    with pytest.raises(finetune.FinetuneError, match="control for ssl"):
        finetune.run(cfg, paths.OUTPUT / "x" / "finetune", None)


# ---- gradients reach every trained tensor ----------------------------------------------------------------------------

def test_frozen_or_disconnected_tensor_in_a_trained_group_refused(tmp_path):
    model, groups, loss, validate = problem()
    stray = torch.nn.Parameter(torch.zeros(3))
    groups.append({"params": [stray], "lr": 1e-3, "weight_decay": 0.0, "name": "stray"})
    with pytest.raises(guards.GuardError, match="no gradient"):
        trainer.train(tmp_path, base_cfg(), "finetune", model, groups, loss, N, data_hash="d", parent=None,
                      validate=validate, extra={"stray": stray})
    model, groups, loss, validate = problem()
    model[0].weight.requires_grad_(False)
    with pytest.raises(guards.GuardError, match="frozen"):
        trainer.train(tmp_path / "b", base_cfg(), "finetune", model, groups, loss, N, data_hash="d", parent=None,
                      validate=validate)


# ---- one process per stage directory ---------------------------------------------------------------------------------

def test_second_process_refused_and_stale_lock_taken_over(tmp_path):
    (tmp_path / "run.lock").write_text("othernode:1234:999")
    with pytest.raises(guards.GuardError, match="being trained"):
        guards.RunLock(tmp_path).acquire()
    old = time.time() - 3600
    import os
    os.utime(tmp_path / "run.lock", (old, old))                   # a dead process's lock is taken over
    with guards.RunLock(tmp_path) as lock:
        assert (tmp_path / "run.lock").read_text() == lock.owner
        with pytest.raises(guards.GuardError, match="being trained"):
            guards.RunLock(tmp_path).acquire()
    assert not (tmp_path / "run.lock").exists()


def test_acquisition_is_atomic(tmp_path):
    got, start = [], threading.Barrier(8)

    def grab():
        start.wait()
        try:
            got.append(guards.RunLock(tmp_path).acquire())
        except guards.GuardError:
            pass
    threads = [threading.Thread(target=grab) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(got) == 1
    got[0].release()


def test_heartbeat_is_shorter_than_staleness_and_takeover_is_detected(tmp_path):
    lock = guards.RunLock(tmp_path, stale_s=0.8).acquire()
    assert lock.interval < lock.stale_s
    time.sleep(2.0)                                                   # far longer than the staleness, no explicit beat
    assert time.time() - (tmp_path / "run.lock").stat().st_mtime < lock.stale_s
    with pytest.raises(guards.GuardError, match="being trained"):
        guards.RunLock(tmp_path, stale_s=0.8).acquire()
    (tmp_path / "run.lock").write_text("othernode:1:2")             # another process took it over
    with pytest.raises(guards.GuardError, match="taken over"):
        lock.check()
    lock.release()
    assert (tmp_path / "run.lock").read_text() == "othernode:1:2"   # never removes another owner's lock


# ---- configs ---------------------------------------------------------------------------------------------------------

def test_typo_and_type_errors_in_overrides():
    base = config.load(paths.ROOT / "configs" / "teacher.yaml")
    for bad in ("finetune.lr_haed=1e-3", "finetune.steps=1.5", "finetune.compile=yes", "dynamics.steps=3"):
        with pytest.raises(config.ConfigError):
            config.apply_overrides(base, [bad])
    c = config.apply_overrides(base, ["finetune.lr_head=1e-4", "model.version=3.0", "finetune.steps=6e4"])
    assert c["finetune"]["lr_head"] == 1e-4 and c["model"]["version"] == "3.0" and c["finetune"]["steps"] == 60000



def test_more_config_override_errors():
    base = config.load(paths.ROOT / "configs" / "teacher.yaml")
    with pytest.raises(config.ConfigError, match="is not key=value"):
        config.apply_overrides(base, ["finetune.lr_head"])
    with pytest.raises(config.ConfigError, match="needs a number"):
        config.apply_overrides(base, ["finetune.lr_head=fast"])
    with pytest.raises(config.ConfigError, match="a section in one place and a value in the other"):
        config.check({**base, "model": 3})


def test_bad_schedule_and_dynamics_settings_are_refused(tmp_path):
    from wavefm.train import schedule
    with pytest.raises(schedule.ScheduleError, match="warmup_frac"):
        schedule.make({**base_cfg()["finetune"], "warmup_frac": 1.0}, 10)
    with pytest.raises(schedule.ScheduleError, match="plateau: patience"):
        schedule.make({**base_cfg()["finetune"], "schedule": "plateau", "patience": 0}, 10)
    with pytest.raises(trainer.TrainerError, match="eval_points must be >= 1"):
        trainer.eval_steps({"eval_every": 0, "eval_points": 0}, 10)
    with pytest.raises(trainer.TrainerError, match="keep 'best'"):
        train(tmp_path / "a", base_cfg(keep="best"))
    model, groups, loss, validate = problem()
    with pytest.raises(trainer.TrainerError, match="clash with model keys"):
        trainer.train(tmp_path / "b", base_cfg(), "finetune", model, groups, loss, N, data_hash="d", parent=None,
                      validate=validate, extra={"0.weight": torch.zeros(1)})


def test_lock_staleness_and_missing_artifact_guards(tmp_path):
    with pytest.raises(guards.GuardError, match="heartbeat interval"):
        guards.RunLock(tmp_path, stale_s=0)
    with pytest.raises(guards.GuardError, match="did not promise this file"):
        guards.check_artifact(tmp_path, {"artifacts": {}}, "student.npz")


def test_model_building_guards(tmp_path):
    from wavefm.models import backbones, build, student
    with pytest.raises(ValueError, match="model.version"):
        backbones.TimesFMEncoder("9.9", load_weights=False)
    with pytest.raises(ValueError, match="has no encoder"):
        build.build_encoder({"kind": "cnn"})
    cfg = config.load(paths.ROOT / "configs" / "cnn.yaml")
    with pytest.raises(ValueError, match="no encoder to start from"):
        build.build_model({**cfg, "task": "dro"}, None, 3, init_from=tmp_path / "weights.pt")
    torch.save([torch.zeros(1)], tmp_path / "list.pt")
    with pytest.raises(build.InitFromError, match="is not a state dict"):
        build.encoder_state(tmp_path / "list.pt")
    with pytest.raises(student.StudentInputError, match="need an input and an output width"):
        student.mlp([640])
