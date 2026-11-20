"""The shared training loop (train.trainer) on a tiny synthetic problem, the finetune stage on the fake data sets, and
the three default configs. Fault injection of the guards is in test_faults.py.

The synthetic problem: 64 rows of 5 features, a small MLP with dropout (so the global CPU generator matters), and a
loss that also draws a random row mask from the trainer's generator (as SSL draws its patch masks)."""
import importlib.util
import math
import os
import signal

import numpy as np
import pytest
import torch
import torch.nn as nn

from wavefm import paths
from wavefm.data.batches import BatchSampler
from wavefm.train import finetune, guards, pipeline, rundir, trainer
from wavefm.utils import config, runtime
from wavefm.utils.io import load_json, save_json

N, NV = 64, 16


@pytest.fixture(autouse=True)
def strict():
    runtime.setup_numerics()
    yield
    runtime.setup_numerics()


def base_cfg(stage="finetune", **dyn):
    c = config.load(paths.ROOT / "configs" / "teacher.yaml")
    c[stage].update({"steps": 40, "epochs": 0, "batch_size": 8, "eval_points": 4, "compile": False,
                     "weight_decay": 0.01, **dyn})
    return c


def problem(device="cpu", dropout=0.1):
    """(model, groups, loss, validate) built from seed 0; loss and validate on fixed data."""
    g = torch.Generator().manual_seed(5)
    X, Xv = torch.randn(N, 5, generator=g), torch.randn(NV, 5, generator=g)
    w = torch.randn(5, 1, generator=g)
    Y, Yv = X @ w + 0.1 * torch.randn(N, 1, generator=g), Xv @ w
    X, Y, Xv, Yv = X.to(device), Y.to(device), Xv.to(device), Yv.to(device)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Dropout(dropout), nn.Linear(16, 1)).to(device)
    groups = [{"params": [p for p in model.parameters() if p.ndim > 1], "lr": 1e-2, "weight_decay": 0.01, "name": "w"},
              {"params": [p for p in model.parameters() if p.ndim <= 1], "lr": 3e-2, "weight_decay": 0.0, "name": "b"}]

    def loss(b, gen):
        keep = (torch.rand(len(b), generator=gen, device=gen.device) < 0.8).float()[:, None]
        return (((model(X[b]) - Y[b]) ** 2) * keep).sum() / keep.sum().clamp_min(1)

    def validate():
        return ((model(Xv) - Yv) ** 2).mean()

    return model, groups, loss, validate


def train(out, cfg, stage="finetune", prob=None, **kw):
    model, groups, loss, validate = prob or problem()
    kw.setdefault("validate", validate)
    s = trainer.train(out, cfg, stage, model, groups, kw.pop("loss", loss), N, data_hash="d" * 64, parent=None, **kw)
    return s, model


def manual(cfg, steps, clip):
    """The same training written out by hand: AdamW, sampler, generator, clipping; no schedule."""
    d = cfg["finetune"]
    model, groups, loss, _ = problem()
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=d["weight_decay"], fused=False)
    sampler = BatchSampler(N, d["batch_size"], cfg["seed"], "cpu")
    gen = torch.Generator().manual_seed(cfg["seed"] + trainer.GEN_OFFSET)
    params, norms = [p for g in groups for p in g["params"]], []
    for _ in range(steps):
        l = loss(sampler.next(), gen)
        opt.zero_grad(set_to_none=True)
        l.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(params, clip).item())
        opt.step()
    return model, norms


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---- ① the default: constant rate, exactly the configured steps, the last weights ----------------------------------

def test_default_is_constant_rate_fixed_steps_last_weights(tmp_path):
    cfg = base_cfg(grad_clip=0.0)
    calls = []
    model, groups, loss, validate = problem()
    s = trainer.train(tmp_path, cfg, "finetune", model, groups, lambda b, g: (calls.append(1), loss(b, g))[1], N,
                      data_hash="d" * 64, parent=None, validate=validate)
    hist = load_json(tmp_path / "history.json")
    assert len(calls) == 40 and s["actual"]["steps"] == s["requested"]["steps"] == 40 and s["ended_by"] == "steps"
    assert [h["lr"] for h in hist] == [3e-2] * 4 and [h["step"] for h in hist] == [10, 20, 30, 40]
    assert s["keep"] == "last" and not s["checkpoint_method"] and s["kept_step"] == 40
    ref, _ = manual(cfg, 40, math.inf)
    assert same(weights(tmp_path / "weights.pt"), ref.state_dict())


# ---- ③ a validation loss that rises and falls does not stop the run ------------------------------------------------

def test_validation_that_rises_then_falls_runs_to_the_end(tmp_path):
    values = iter([1.0, 2.0, 4.0, 8.0, 16.0, 3.0, 0.5, 0.4])
    cfg = base_cfg(steps=80, eval_points=8)
    s, _ = train(tmp_path, cfg, validate=lambda: next(values))
    hist = load_json(tmp_path / "history.json")
    assert s["actual"]["steps"] == 80 and [h["val_loss"] for h in hist] == [1, 2, 4, 8, 16, 3, 0.5, 0.4]
    assert s["final_val_loss"] == 0.4 and s["hints"]["on"] == "val_loss"


# ---- ④ result.json: requested / actual, the full curves, clipping statistics against hand counts ---------------------

def test_result_records_lengths_curves_and_clipping(tmp_path):
    cfg = base_cfg(grad_clip=0.0)
    _, free = manual(cfg, 40, math.inf)
    clip = float(np.median(free))
    cfg["finetune"]["grad_clip"] = clip
    ref, norms = manual(cfg, 40, clip)
    clipped = np.array(norms) > clip
    assert 0 < clipped.sum() < 40                                     # some, not all steps are clipped
    s, _ = train(tmp_path / "finetune", cfg)
    cfg["run"]["smoke"] = True
    r = trainer.finish(tmp_path / "finetune", cfg, "finetune", s)
    assert r == load_json(tmp_path / "finetune" / "result.json") and rundir.RESULT_KEYS <= r.keys()
    assert r["requested"] == r["actual"] == {"steps": 40, "epochs": 40 * 8 / N, "samples": 320}
    assert set(r["artifacts"]) == {"weights.pt", "history.json"} and not (tmp_path / "finetune" / "ckpt.pt").exists()
    hist = load_json(tmp_path / "finetune" / "history.json")
    assert all({"step", "train_loss", "val_loss", "lr", "grad_norm", "clip_frac"} <= h.keys() for h in hist)
    for k, h in enumerate(hist):
        sl = slice(10 * k, 10 * k + 10)
        assert h["clip_frac"] == clipped[sl].mean()
        assert h["grad_norm"] == pytest.approx(np.mean(norms[sl]), rel=1e-6)
    assert same(weights(tmp_path / "finetune" / "weights.pt"), ref.state_dict())
    assert r["hints"]["final"] == hist[-1]["val_loss"] and r["final_train_loss"] == hist[-1]["train_loss"]


def test_epochs_are_never_clipped(tmp_path):
    """The old step_budget clipped 10 epochs to max_steps; now ceil(epochs * n / batch) steps are run, all of them."""
    cfg = base_cfg(steps=0, epochs=10, batch_size=3, eval_points=3)
    s, _ = train(tmp_path, cfg)
    want = math.ceil(10 * N / 3)
    assert s["requested"] == {"steps": want, "epochs": 10.0, "samples": want * 3}
    assert s["actual"]["steps"] == want and s["actual"]["epochs"] == pytest.approx(want * 3 / N)


@pytest.mark.parametrize("steps, epochs", [(0, 0), (10, 1)])
def test_exactly_one_of_steps_and_epochs(tmp_path, steps, epochs):
    with pytest.raises(trainer.TrainerError, match="exactly one"):
        train(tmp_path, base_cfg(steps=steps, epochs=epochs))


def test_batch_larger_than_the_training_rows_is_refused(tmp_path):
    with pytest.raises(trainer.TrainerError, match="batch_size"):
        train(tmp_path, base_cfg(batch_size=N + 1))


# ---- advanced options ----------------------------------------------------------------------------------------------

def test_best_val_keeps_the_best_weights_and_is_a_checkpoint_method(tmp_path):
    values = iter([3.0, 2.0, 1.0, 5.0, 6.0, 7.0])
    s, _ = train(tmp_path / "best", base_cfg(steps=60, eval_every=10, keep="best_val"), validate=lambda: next(values))
    assert s["kept_step"] == 30 and s["checkpoint_method"] and s["actual"]["steps"] == 60
    t, _ = train(tmp_path / "short", base_cfg(steps=30))      # the same trajectory, stopped at step 30
    assert same(weights(tmp_path / "best" / "weights.pt"), weights(tmp_path / "short" / "weights.pt"))


def test_plateau_lowers_the_rate_and_stops(tmp_path):
    cfg = base_cfg(steps=1000, eval_every=5, schedule="plateau", patience=2, lr_factor=0.5, max_reductions=1)
    s, _ = train(tmp_path, cfg, validate=lambda: 1.0)        # never improves after the first evaluation
    hist = load_json(tmp_path / "history.json")
    # evaluation 1 sets the best; 2, 3 are a plateau -> rate x 0.5 from step 15; 4, 5 a plateau -> stop at step 25
    assert [h["step"] for h in hist] == [5, 10, 15, 20, 25] and s["ended_by"] == "plateau"
    assert [h["lr"] for h in hist] == [3e-2, 3e-2, 3e-2, 1.5e-2, 1.5e-2]
    assert s["actual"]["steps"] == 25 and s["requested"]["steps"] == 1000 and not s["checkpoint_method"]
    with pytest.raises(trainer.TrainerError, match="eval_every"):
        train(tmp_path / "b", base_cfg(schedule="plateau"))


def test_cosine_and_warmup_factors():
    from wavefm.train import schedule
    d = {"schedule": "cosine", "warmup_frac": 0.1, "final_lr_frac": 0.1, "patience": 1, "lr_factor": 0.5,
         "max_reductions": 0}
    s = schedule.make(d, 100)
    assert s.factor(0) == pytest.approx(0.1) and s.factor(9) == pytest.approx(1.0)
    assert s.factor(10) == pytest.approx(1.0) and s.factor(99) == pytest.approx(0.1)        # the last of 100 updates
    assert schedule.make({**d, "schedule": "constant", "warmup_frac": 0.0}, 100).factor(57) == 1.0
    with pytest.raises(schedule.ScheduleError):
        schedule.make({**d, "schedule": "step"}, 100)


def test_cosine_reaches_the_final_rate_on_the_last_update():
    """Factors of the updates 0..steps-1, by hand (schedule docstring): the cosine runs over the updates after the
    warm-up, j = 0..D-1, 0.5 (1 + cos(pi j / (D - 1))); the warm-up is round(warmup_frac * steps), at most steps - 1."""
    from wavefm.train import schedule
    d = {"schedule": "cosine", "warmup_frac": 0.0, "final_lr_frac": 0.0, "patience": 1, "lr_factor": 0.5,
         "max_reductions": 0}
    seq = lambda steps, **o: [schedule.make({**d, **o}, steps).factor(i) for i in range(steps)]
    assert seq(4) == pytest.approx([1.0, 0.75, 0.25, 0.0], abs=1e-15)
    assert seq(3, final_lr_frac=0.2) == pytest.approx([1.0, 0.6, 0.2])
    assert seq(4, warmup_frac=0.5) == pytest.approx([0.5, 1.0, 1.0, 0.0])           # W = 2, then D = 2
    assert seq(4, warmup_frac=0.9) == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])      # round(3.6) = 4 -> W = 3, D = 1
    assert seq(1, warmup_frac=0.9) == [1.0]                                         # one step: no warm-up, base rate
    assert seq(2, warmup_frac=0.9, schedule="constant") == [1.0, 1.0]               # W = 1: the full rate at once


def test_accumulators_are_float64(tmp_path):
    """Two steps with losses 2^24 and 1 in fp32: their mean is 8388608.5 exactly in float64 (fp32 sums lose the 1)."""
    model, groups, _, _ = problem()
    vals = iter([2.0 ** 24, 1.0] * 2)
    loss = lambda b, gen: model(torch.zeros(1, 5)).sum() * 0 + torch.tensor(next(vals), dtype=torch.float32)
    trainer.train(tmp_path, base_cfg(steps=4, eval_every=2), "finetune", model, groups, loss, N, data_hash="d" * 64,
                  parent=None)
    assert [h["train_loss"] for h in load_json(tmp_path / "history.json")] == [8388608.5, 8388608.5]


def test_sigterm_during_the_terminal_evaluation_completes_and_is_passed_on(tmp_path):
    """SIGTERM inside the last evaluation: the stage is complete (weights, full summary), and the signal reaches the
    handler installed before the trainer (the pipeline's), so no further stage starts."""
    got = []
    previous = signal.signal(signal.SIGTERM, lambda *a: got.append(a[0]))
    try:
        model, groups, loss, validate = problem()
        n = [0]

        def val():
            n[0] += 1
            if n[0] == 4:
                os.kill(os.getpid(), signal.SIGTERM)
            return validate()
        s = trainer.train(tmp_path, base_cfg(), "finetune", model, groups, loss, N, data_hash="d" * 64, parent=None,
                          validate=val)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert not s.get("preempted") and s["actual"]["steps"] == 40 and (tmp_path / "weights.pt").exists()
    assert got == [signal.SIGTERM]


@pytest.mark.parametrize("over", [{"keep": "best_val"}, {"schedule": "plateau", "eval_every": 5}])
def test_no_validation_refuses_best_val_and_plateau(tmp_path, over):
    with pytest.raises(trainer.TrainerError, match="no validation set"):
        train(tmp_path, base_cfg("ssl", **over), "ssl", validate=None)


def test_no_validation_named_losses_and_extra_tensors(tmp_path):
    """The ssl form: no validation, two named loss terms, a trained tensor outside the model."""
    model, groups, loss, _ = problem()
    token = nn.Parameter(torch.zeros(5))
    groups.append({"params": [token], "lr": 1e-2, "weight_decay": 0.0, "name": "mask_token"})
    first = []

    def two_terms(b, gen):
        a = loss(b, gen)
        m = ((model(token[None]) - 1) ** 2).mean()
        return {"loss": a + 0.5 * m, "next": a, "masked": m}

    s = trainer.train(tmp_path, base_cfg("ssl"), "ssl", model, groups, two_terms, N, data_hash="d" * 64, parent=None,
                      extra={"mask_token": token})
    hist = load_json(tmp_path / "history.json")
    assert all("val_loss" not in h and {"train_next", "train_masked"} <= h.keys() for h in hist)
    assert all(h["train_loss"] == pytest.approx(h["train_next"] + 0.5 * h["train_masked"], rel=1e-5) for h in hist)
    w = weights(tmp_path / "weights.pt")
    assert torch.equal(w["mask_token"], token.detach()) and w["mask_token"].abs().sum() > 0 and "3.weight" in w
    assert s["hints"]["on"] == "train_loss" and s["final_val_loss"] is None


def test_second_validation_curve_is_recorded_and_only_val_loss_selects(tmp_path):
    vals = iter([(3, 0.1), (1, 9.0), (2, 0.05), (4, 0.01)])
    s, _ = train(tmp_path, base_cfg(keep="best_val"),
                 validate=lambda: dict(zip(("val_loss", "val_truth_loss"), next(vals))))
    hist = load_json(tmp_path / "history.json")
    assert [h["val_truth_loss"] for h in hist] == [0.1, 9.0, 0.05, 0.01] and s["kept_step"] == 20


# ---- ⑦ exact resume: interrupted mid-interval, on an evaluation boundary, inside an evaluation -----------------------

def interrupted(out, cfg, prob_args=(), kill_loss=None, kill_val=None, deadline_first=False):
    """Run with SIGTERM sent from inside loss call ``kill_loss`` (or validate call ``kill_val``), then resume."""
    model, groups, loss, validate = problem(*prob_args)
    n = {"loss": 0, "val": 0}

    def k_loss(b, gen):
        n["loss"] += 1
        if n["loss"] == kill_loss:
            os.kill(os.getpid(), signal.SIGTERM)
        return loss(b, gen)

    def k_val():
        n["val"] += 1
        if n["val"] == kill_val:
            os.kill(os.getpid(), signal.SIGTERM)
        return validate()

    c = config.merge(cfg, {"run": {"deadline": 1.0}}) if deadline_first else cfg
    first = trainer.train(out, c, "finetune", model, groups, k_loss, N, data_hash="d" * 64, parent=None,
                          validate=k_val)
    assert first == {"preempted": True} and (out / "ckpt.pt").exists()
    model, groups, loss, validate = problem(*prob_args)               # a new process: fresh objects
    return trainer.train(out, cfg, "finetune", model, groups, loss, N, data_hash="d" * 64, parent=None,
                         validate=validate)


@pytest.mark.parametrize("kill", [dict(kill_loss=15), dict(kill_loss=20), dict(kill_val=2), dict(deadline_first=True)],
                         ids=["mid-interval", "evaluation-boundary", "inside-evaluation", "deadline"])
def test_exact_resume(tmp_path, kill):
    cfg = base_cfg(eval_every=10, grad_clip=0.5)
    ref, _ = train(tmp_path / "ref", cfg)
    s = interrupted(tmp_path / "int", cfg, **kill)
    assert same(weights(tmp_path / "int" / "weights.pt"), weights(tmp_path / "ref" / "weights.pt"))
    ha, hb = load_json(tmp_path / "int" / "history.json"), load_json(tmp_path / "ref" / "history.json")
    assert ha == hb and [h["step"] for h in ha] == [10, 20, 30, 40]   # no evaluation skipped
    assert s["actual"] == ref["actual"] and s["hints"] == ref["hints"]


@pytest.mark.gpu
def test_exact_resume_on_gpu_with_the_trainers_generator(tmp_path):
    """GPU-side random draws (the loss's mask from the trainer's CUDA generator) resume exactly."""
    assert torch.cuda.is_available(), "this test needs a GPU"
    cfg = base_cfg(eval_every=10, grad_clip=0.5)
    model, groups, loss, validate = problem("cuda")
    trainer.train(tmp_path / "ref", cfg, "finetune", model, groups, loss, N, data_hash="d" * 64, parent=None,
                  validate=validate)
    for name, kill in (("mid", dict(kill_loss=15)), ("edge", dict(kill_loss=20))):
        interrupted(tmp_path / name, cfg, ("cuda",), **kill)
        assert same(weights(tmp_path / name / "weights.pt"), weights(tmp_path / "ref" / "weights.pt"))
        assert load_json(tmp_path / name / "history.json") == load_json(tmp_path / "ref" / "history.json")
    other = base_cfg(eval_every=10, grad_clip=0.5)
    other["seed"] = 1                                                 # control: another generator seed differs
    model, groups, loss, validate = problem("cuda")
    trainer.train(tmp_path / "seed1", other, "finetune", model, groups, loss, N, data_hash="d" * 64, parent=None,
                  validate=validate)
    assert not same(weights(tmp_path / "seed1" / "weights.pt"), weights(tmp_path / "ref" / "weights.pt"))


# ---- the finetune stage on the fake data sets ----------------------------------------------------------------------

def cnn_cfg(fake, **over):
    c = config.merge(config.load(paths.ROOT / "configs" / "cnn.yaml"), {"data": fake["data"], "run": fake["run"]})
    c["model"]["widths"] = [4, 8, 8]
    c["run"]["smoke"] = True
    c["finetune"].update({"steps": 30, "batch_size": 16, "eval_every": 10, "compile": False, **over})
    config.check(c)
    return c


def script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", paths.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finetune_stage_end_to_end(fake_dro):
    cfg = cnn_cfg(fake_dro)
    r = pipeline.run(cfg)["results"]["finetune"]
    out = pipeline.run_dir(cfg) / "finetune"
    assert set(r["artifacts"]) == {"weights.pt", "history.json", "rows.npz", "constants.json"}
    assert r["keep"] == "best_val" and r["checkpoint_method"] and r["smoke"]
    assert not (out / "ckpt.pt").exists() and not (out / "run.lock").exists()
    assert pipeline.run(cfg)["results"]["finetune"] == r                   # finished: reused, not redone
    import wavefm.data
    dev = wavefm.data.dev(cfg)
    pred = finetune.predict(out, dev)
    assert pred.shape == dev.labels.shape and pred.dtype == np.float64 and np.isfinite(pred).all()
    model, task, _ = finetune.load(out, "cpu")                             # the kept weights give the recorded loss
    rows = np.load(out / "rows.npz")
    val = task.waveforms(dev, rows["val"], "cpu")
    best = min(load_json(out / "history.json"), key=lambda h: h["val_loss"])
    assert task.val_loss(model, val, 7) == pytest.approx(best["val_loss"], rel=1e-5)
    rec = script("evaluate").evaluate(pipeline.run_dir(cfg))               # heldout through predict, physical units
    assert rec["stage"] == "finetune" and rec["n"] == 100 and rec["metrics"]
    for name in ("constants.json", "weights.pt"):                          # a file edited after the stage finished
        keep = (out / name).read_bytes()
        (out / name).write_bytes(keep + b" ")
        with pytest.raises(guards.GuardError, match=f"{name} changed after the stage finished"):
            finetune.predict(out, dev)
        (out / name).write_bytes(keep)


# ---- the three default configs on the fake data: 20 updates, weights change, finite losses ---------------------

def smoke(fake, name, monkeypatch):
    c = config.merge(config.load(paths.ROOT / "configs" / name), {"data": fake["data"], "run": fake["run"]})
    c = config.apply_overrides(c, ["finetune.steps=20", "finetune.epochs=0", "finetune.batch_size=8",
                                   "finetune.compile=false", "finetune.eval_every=10", "run.smoke=true"])
    config.check(c)
    before = {}
    real = trainer.train

    def spy(out, cfg, stage, model, groups, *a, **k):
        before.update({n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad})
        return real(out, cfg, stage, model, groups, *a, **k)

    monkeypatch.setattr(trainer, "train", spy)
    r = pipeline.run(c)["results"]["finetune"]
    out = pipeline.run_dir(c) / "finetune"
    w = weights(out / "weights.pt")
    hist = load_json(out / "history.json")
    assert r["actual"]["steps"] == 20 and before
    changed = [n for n, p in before.items() if not torch.equal(w[n], p)]
    assert len(changed) >= 0.9 * len(before), f"only {len(changed)} of {len(before)} trained tensors changed"
    assert all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in hist)
    return r, before


def test_default_cnn_config_runs(fake_dro, monkeypatch):
    smoke(fake_dro, "cnn.yaml", monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("name, fake", [("dro_teacher.yaml", "fake_dro"), ("dch_teacher.yaml", "fake_dch")])
def test_default_two_layer_configs_run(name, fake, request, monkeypatch):
    assert torch.cuda.is_available(), "this test needs a GPU and the TimesFM weights"
    r, before = smoke(request.getfixturevalue(fake), name, monkeypatch)
    trained = {n.split(".")[2] for n in before if n.startswith("encoder.layers.")}
    assert trained == {"18", "19"} and any(n.startswith("head.") for n in before)      # the last two blocks + head


def test_finetune_load_refuses_an_edited_config(fake_dro, monkeypatch):
    """the stage is rebuilt only from the config it was trained with."""
    smoke(fake_dro, "cnn.yaml", monkeypatch)
    run = paths.OUTPUT / fake_dro["run"]["group"] / fake_dro["run"]["name"]
    cfg = load_json(run / "config.json")
    cfg["finetune"]["lr_head"] *= 2
    save_json(cfg, run / "config.json")
    with pytest.raises(finetune.FinetuneError, match="is not the config the finetune stage"):
        finetune.load(run / "finetune")
