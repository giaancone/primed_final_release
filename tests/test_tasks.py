"""The tasks (losses, exact validation loss, prediction in physical units, saved constants), the final evaluation
entry scripts/evaluate.py, and scripts/prepare_data.py."""
import importlib.util
import json
import shutil
import sys
import types

import numpy as np
import pytest
import torch

import wavefm.data
from data import store
from wavefm import paths, tasks
from wavefm.data import splits
from wavefm.train import rundir
from wavefm.utils import config
from wavefm.utils.io import load_json, save_json


def script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", paths.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dro_forward(x):
    """A fixed model of the scaled waveform: its value depends on the input scaling."""
    return torch.stack([x.mean(1), x.max(1).values, x[:, 70] - x[:, 50]], 1)


def dch_forward(x, pad_logits=None):
    logits = torch.stack([torch.sin(x), -x, 0.5 * x ** 2], -1)
    if pad_logits is not None:
        logits = torch.cat([logits[:, :250], pad_logits.expand(len(x), -1, -1)], 1)
    return {"logits": logits, "count": torch.tanh(x).sum(1) / 20}


# ---- prepare --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["fake_dro", "fake_dch"])
def test_prepare_draws_inside_the_budget(fixture, request):
    cfg = config.merge(request.getfixturevalue(fixture), {"seed": 3, "data": {"label_frac": 0.5}})
    task = tasks.get(cfg["task"])
    train, val = task.prepare(cfg, "cpu")
    d = cfg["data"]
    tr, va = splits.labelled(400, 0.5, d["val_frac"], 3, d["min_labelled"], d["min_val"])
    np.testing.assert_array_equal(train.idx, tr); np.testing.assert_array_equal(val.idx, va)
    dev = wavefm.data.dev(cfg)
    np.testing.assert_array_equal(train.t["raw"].numpy(), dev.labels[tr].astype(np.float64))
    st = dev.meta["waveform"]["stats"]
    c = task.constants
    assert c["input"] == {"kind": "global_minmax", "lo": st["min"], "hi": st["max"]}       # dev meta, not fitted
    np.testing.assert_allclose(c["target"]["mean"], dev.labels[tr].astype(np.float64).mean(0), rtol=1e-12)
    assert json.loads(json.dumps(c)) == c and c["labels"] == dev.meta["labels"]
    c2 = tasks.get(cfg["task"]); c2.prepare(config.merge(cfg, {"model": {"scaling": "native"}}), "cpu")
    assert c2.constants["input"] == {"kind": "native"} and c2.in_scaler is None


def test_dro_loss_is_the_weighted_mse(fake_dro):
    cfg = config.merge(fake_dro, {"finetune": {"loss_weights": [4.0, 1.0, 0.3], "target_scaling": "divisor",
                                               "divisors": [500, 220, 2.5]}})
    task = tasks.get("dro")
    train, val = task.prepare(cfg, "cpu")
    np.testing.assert_allclose(train.t["y"].numpy(), train.t["raw"].numpy() / [500, 220, 2.5], rtol=1e-6)
    x, t = train.batch(torch.arange(16))
    out = dro_forward(x)
    want = ((out - t["y"]) ** 2 * torch.tensor([4.0, 1.0, 0.3])).mean()
    assert task.loss(out, t).item() == pytest.approx(want.item(), rel=1e-6)
    with pytest.raises(ValueError, match="loss_weights"):
        tasks.get("dro").prepare(config.merge(cfg, {"finetune": {"loss_weights": [1.0]}}), "cpu")


# ---- ⑥ exact validation loss, padding out of the cross entropy ------------------------------------------------------

@pytest.mark.parametrize("fixture", ["fake_dro", "fake_dch"])
def test_val_loss_does_not_depend_on_the_batch_size(fixture, request):
    cfg = request.getfixturevalue(fixture)
    task = tasks.get(cfg["task"])
    _, val = task.prepare(cfg, "cpu")
    fwd = dch_forward if cfg["task"] == "dch" else dro_forward
    a, b, c = (task.val_loss(fwd, val, bs) for bs in (7, 64, len(val)))
    assert a == pytest.approx(b, rel=1e-6) and a == pytest.approx(c, rel=1e-6)


def test_dch_cross_entropy_leaves_padding_out(fake_dch):
    """The class-weighted CE is sum w nll / sum w over the 250 real samples of all rows, written out with numpy; the
    logits of the 6 padded samples do not change it."""
    task = tasks.get("dch")
    _, val = task.prepare(fake_dch, "cpu")
    assert val.x.shape[1] == 256 and task.constants["length"] == 250
    dev = wavefm.data.dev(fake_dch)
    y = dev.extra("classes")[val.idx].astype(np.int64)
    x = val.x.numpy().astype(np.float64)[:, :250]
    z = np.stack([np.sin(x), -x, 0.5 * x ** 2], -1)
    nll = np.log(np.exp(z).sum(-1)) - np.take_along_axis(z, y[..., None], -1)[..., 0]
    w = np.asarray(task.constants["class_weights"])[y]
    ce = (w * nll).sum() / w.sum()
    cnt = ((np.tanh(val.x.numpy().astype(np.float64)).sum(1) / 20 - val.t["count"].numpy()) ** 2).mean()
    assert task.val_loss(dch_forward, val, 9) == pytest.approx(ce + cnt, rel=1e-6)
    freq = np.bincount(dev.extra("classes")[task_train_idx(fake_dch)].ravel(), minlength=3)
    np.testing.assert_allclose(task.constants["class_frequency"], freq / freq.sum())
    for pad in (torch.full((1, 6, 3), 50.0), torch.tensor([[[-80.0, 30.0, 0.0]]]).expand(1, 6, 3)):
        assert task.val_loss(lambda v: dch_forward(v, pad), val, 9) == pytest.approx(ce + cnt, rel=1e-6)
    xb, t = val.batch(torch.arange(5))
    assert task.loss(dch_forward(xb), t).item() == pytest.approx(task.loss(dch_forward(xb, torch.ones(1, 6, 3) * 99), t).item())


def test_dch_nan_logits_in_padding_change_nothing(fake_dch):
    """The padded samples are dropped before the cross entropy: NaN logits there give the loss without them."""
    task = tasks.get("dch")
    _, val = task.prepare(fake_dch, "cpu")
    nan = torch.full((1, 6, 3), float("nan"))
    xb, t = val.batch(torch.arange(5))
    clean, dirty = task.loss(dch_forward(xb), t), task.loss(dch_forward(xb, nan), t)
    assert torch.isfinite(dirty) and dirty.item() == clean.item()
    assert task.val_loss(lambda v: dch_forward(v, nan), val, 9) == task.val_loss(dch_forward, val, 9)


def task_train_idx(cfg):
    d = cfg["data"]
    return splits.labelled(400, d["label_frac"], d["val_frac"], cfg["seed"], d["min_labelled"], d["min_val"])[0]


# ---- prediction in physical units, from saved constants -------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["fake_dro", "fake_dch"])
def test_predict_physical_units_and_restore(fixture, request):
    cfg = request.getfixturevalue(fixture)
    task = tasks.get(cfg["task"])
    _, val = task.prepare(cfg, "cpu")
    zero = (lambda x: torch.zeros(len(x), 3)) if cfg["task"] == "dro" else (lambda x: {"count": torch.zeros(len(x))})
    pred = task.predict(zero, val, 13)
    k = len(task.constants["labels"])
    assert pred.shape == (len(val), k) and pred.dtype == np.float64
    np.testing.assert_allclose(pred, np.broadcast_to(task.constants["target"]["mean"], pred.shape), rtol=1e-12)
    fwd = dch_forward if cfg["task"] == "dch" else dro_forward
    again = tasks.get(cfg["task"]).restore(json.loads(json.dumps(task.constants)), "cpu")
    dev = wavefm.data.dev(cfg)
    np.testing.assert_array_equal(again.predict(fwd, again.waveforms(dev, val.idx, "cpu"), 50), task.predict(fwd, val, 13))
    m = task.score(task.predict(fwd, val, 13), val)
    assert m["n"] == len(val)


# ---- ③ the final evaluation -----------------------------------------------------------------------------------------

def fake_stage(monkeypatch):
    """A stage that "trains" through the real task (constants fitted on dev) and predicts with a fixed forward from
    the constants it saved."""
    mod = types.ModuleType("wavefm.train.fake_stage")

    def run(cfg, out, parent):
        task = tasks.get(cfg["task"])
        task.prepare(cfg, "cpu")
        save_json(task.constants, out / "constants.json")
        return rundir.write_result(out, "fake_stage", cfg, wavefm.data.dev(cfg).meta["data_hash"], parent,
                                   requested={"steps": 0}, actual={"steps": 0}, hints={}, wall_s=0.0)

    def predict(stage_dir, split):
        c = load_json(stage_dir / "constants.json")
        task = tasks.get(c["task"]).restore(c, "cpu")
        return task.predict(dro_forward, task.waveforms(split, np.arange(len(split)), "cpu"), 32)

    mod.run, mod.predict = run, predict
    monkeypatch.setitem(sys.modules, "wavefm.train.fake_stage", mod)
    return mod


def make_run(cfg, name, mod, verdict="pass"):
    """A finished non-smoke run of the fake stage, judged ``verdict`` (None: not judged)."""
    cfg = config.merge(cfg, {"stages": ["fake_stage"], "run": {"name": name}})
    run = paths.OUTPUT / "test" / name
    (run / "fake_stage").mkdir(parents=True)
    save_json(cfg, run / "config.json")
    mod.run(cfg, run / "fake_stage", None)
    if verdict:
        rundir.write_verdict(run / "fake_stage", verdict, "by hand", "test")
    return run


def test_final_evaluation_uses_saved_constants(fake_dro, monkeypatch, capsys):
    """③ heldout is scored in physical units with the constants of training; rewriting dev after training changes
    a refit (control) but not the heldout predictions. heldout.json is written once."""
    ev, mod = script("evaluate"), fake_stage(monkeypatch)
    run = make_run(fake_dro, "a", mod)
    r = ev.evaluate(run)
    with wavefm.data.final_evaluation():
        h = wavefm.data.heldout(fake_dro)
    pred = mod.predict(run / "fake_stage", h)
    assert r["metrics"] == store.evaluate("dro_gen")(pred, h.labels) and r["n"] == 100 and r["stage"] == "fake_stage"
    assert r["data_hash"] == h.meta["data_hash"] and r["result"]["sha256"] == rundir.file_sha256(run / "fake_stage" / "result.json")
    assert load_json(run / "heldout.json") == json.loads(json.dumps(r))
    c = load_json(run / "fake_stage" / "constants.json")
    assert np.abs(pred[:, 0]).mean() > 10 and pred.dtype == np.float64            # photons, not scaled targets

    # rewrite dev with other content (same definition, so the same data hash): a refit would change
    dev = wavefm.data.dev(fake_dro)
    W, Y, rows, meta = dev.take(np.arange(len(dev))), dev.labels, dev.rows, dev.meta
    shutil.rmtree(dev.path)
    store.write_split(dev.path, store.load_config("dro_gen", "fake", wavefm.data.ROOT), "dev", Y * 2, rows,
                      meta["labels"], dt_ns=meta["dt_ns"], t_first_ns=0.0, unit="mV", waveform=W * 3,
                      root=wavefm.data.ROOT)
    refit = tasks.get("dro"); refit.prepare(fake_dro, "cpu")
    assert refit.constants["input"] != c["input"] and refit.constants["target"] != c["target"]   # control
    run_b = paths.OUTPUT / "test" / "b"
    shutil.copytree(run, run_b)
    (run_b / "heldout.json").unlink()
    assert ev.evaluate(run_b)["metrics"] == r["metrics"]

    # once: a second call refuses, prints the existing file, and leaves it as it was
    before = (run / "heldout.json").read_text()
    assert ev.main([str(run)]) == 1
    assert before in capsys.readouterr().out and (run / "heldout.json").read_text() == before
    with pytest.raises(wavefm.data.HeldoutError):                       # locked again afterwards
        wavefm.data.heldout(fake_dro)


def test_final_evaluation_guards(fake_dro, monkeypatch, capsys):
    ev, mod = script("evaluate"), fake_stage(monkeypatch)
    assert ev.main([]) == 2 and "Usage" in capsys.readouterr().out
    cfg = config.merge(fake_dro, {"stages": ["fake_stage"], "run": {"name": "none"}})
    (paths.OUTPUT / "none").mkdir(parents=True)
    save_json(cfg, paths.OUTPUT / "none" / "config.json")
    with pytest.raises(ev.EvaluationError, match="no stage"):
        ev.evaluate(paths.OUTPUT / "none")
    run = make_run(fake_dro, "other_data", mod)
    r = load_json(run / "fake_stage" / "result.json")
    save_json({**r, "data_hash": "0" * 64}, run / "fake_stage" / "result.json")
    rundir.write_verdict(run / "fake_stage", "pass", "judged again", "test")
    with pytest.raises(ev.EvaluationError, match="another definition"):
        ev.evaluate(run)
    assert not (run / "heldout.json").exists()


def test_evaluation_is_claimed_before_heldout_is_opened(fake_dro, monkeypatch):
    """A second evaluator that starts while the claim exists never opens heldout; a failed evaluation removes its
    claim, so it can be retried; a finished heldout.json refuses for ever."""
    ev, mod = script("evaluate"), fake_stage(monkeypatch)
    run = make_run(fake_dro, "claimed", mod)
    opened, real = [], wavefm.data.heldout
    monkeypatch.setattr(wavefm.data, "heldout", lambda cfg: (opened.append(1), real(cfg))[1])
    (run / ".heldout.claim").write_text("")                         # another evaluator is at work
    with pytest.raises(ev.EvaluationError, match="being evaluated"):
        ev.evaluate(run)
    assert opened == [] and not (run / "heldout.json").exists()
    (run / ".heldout.claim").unlink()
    good = mod.predict
    mod.predict = lambda *a: (_ for _ in ()).throw(RuntimeError("prediction failed"))
    with pytest.raises(RuntimeError, match="prediction failed"):
        ev.evaluate(run)
    assert opened == [1] and not (run / ".heldout.claim").exists() and not (run / "heldout.json").exists()
    mod.predict = good
    assert ev.evaluate(run)["n"] == 100 and not (run / ".heldout.claim").exists()
    with pytest.raises(FileExistsError):
        ev.evaluate(run)
    assert opened == [1, 1]


def test_a_run_whose_finished_stages_predict_nothing(fake_dro, monkeypatch, capsys):
    """ssl has no predict and compress is scored in the other environment: evaluating a run whose only finished
    stages are those two says so in one line and exits 1 (a finished compress made it crash: found by the smoke test)."""
    ev = script("evaluate")
    cfg = config.merge(fake_dro, {"stages": ["ssl", "compress"], "run": {"name": "ssl_only"}})
    run = paths.OUTPUT / "test" / "ssl_only"
    save_json(cfg, run / "config.json")
    for s in ("ssl", "compress"):
        (run / s).mkdir(parents=True)
        rundir.write_result(run / s, s, cfg, wavefm.data.dev(cfg).meta["data_hash"], None, requested={},
                            actual={}, hints={}, wall_s=0.0)
    monkeypatch.setattr(wavefm.data, "heldout", lambda cfg: pytest.fail("heldout opened"))
    assert ev.main([str(run)]) == 1
    out = capsys.readouterr().out
    assert "no stage of" in out and "predicts" in out and len(out.strip().splitlines()) == 1


def test_evaluation_waits_for_the_last_predicting_stage_and_the_judge(fake_dro, monkeypatch):
    """A later predicting stage configured but unfinished, or a non-smoke stage not judged 'pass' on its present
    result: refused before heldout is opened (the run's one evaluation would be spent on the wrong thing)."""
    ev, mod = script("evaluate"), fake_stage(monkeypatch)
    opened, real = [], wavefm.data.heldout
    monkeypatch.setattr(wavefm.data, "heldout", lambda cfg: (opened.append(1), real(cfg))[1])
    run = make_run(fake_dro, "later", mod)
    cfg = load_json(run / "config.json")
    save_json({**cfg, "stages": ["fake_stage", "student"]}, run / "config.json")
    with pytest.raises(ev.EvaluationError, match="stage student .* has not finished"):
        ev.evaluate(run)
    save_json(cfg, run / "config.json")
    for name, verdict in (("unjudged", None), ("overfit", "overfit")):
        with pytest.raises(ev.EvaluationError, match="not accepted"):
            ev.evaluate(make_run(fake_dro, name, mod, verdict))
    assert opened == [] and not list(paths.OUTPUT.glob("test/*/heldout.json"))
    smoke = make_run(config.merge(fake_dro, {"run": {"smoke": True}}), "smoke", mod, None)   # exempt: never summarised
    assert ev.evaluate(smoke)["result"]["smoke"] and ev.evaluate(run)["stage"] == "fake_stage" and opened == [1, 1]


# ---- prepare_data.py ------------------------------------------------------------------------------------------------

def test_prepare_data_script(fake_dch, capsys):
    pd = script("prepare_data")
    assert pd.main([]) == 2 and "Usage" in capsys.readouterr().out                 # no KeyError, the usage
    assert pd.main(["dch_wf", "fake"]) == 0
    assert "exists, digests agree" in capsys.readouterr().out
    f = wavefm.data.ROOT / "dch_wf" / "configs" / "fake.yaml"
    f.write_text(f.read_text().replace("seed: 1", "seed: 2"))
    assert pd.main(["dch_wf", "fake"]) == 1
    assert "another config" in capsys.readouterr().out
    assert pd.main(["dch_wf", "fake", "--workers", "4"]) == 2                      # dch_wf's builder has no workers



def test_a_restored_task_refuses_another_label_set(fake_dro, fake_dch):
    from wavefm.tasks import common
    c = {"labels": wavefm.data.dev(fake_dro).meta["labels"], "length": 256}
    with pytest.raises(ValueError, match="differ from those the task was trained with"):
        common.check_labels(c, wavefm.data.dev(fake_dch))
    with pytest.raises(ValueError, match="task dch predicts the label"):
        tasks.get("dch").prepare(config.merge(fake_dro, {"task": "dch"}), "cpu")


def test_a_prediction_of_the_wrong_shape_is_refused(fake_dro, monkeypatch):
    ev, mod = script("evaluate"), fake_stage(monkeypatch)
    run = make_run(fake_dro, "wrong_shape", mod)
    mod.predict = lambda stage_dir, split: np.zeros((len(split), 2))
    with pytest.raises(ev.EvaluationError, match=r"predicted \(100, 2\)"):
        ev.evaluate(run)
    assert not (run / "heldout.json").exists()
