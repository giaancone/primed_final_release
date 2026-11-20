"""The ssl stage (train/ssl.py) on the fake data sets.

CPU tests use the cheapest real encoders: ``blocks: none`` (tokenizer + adapter, no transformer) with a stand-in
tokenizer in place of the Hugging Face file, and ``blocks: random`` (the full 20-block 2.5 module at its seeded random
initialisation, nothing read) where the frozen transformer must be checked. The ``gpu`` tests run the real pretrained
TimesFM 2.5 in the two paper-like settings (DRO: global_minmax + zscore targets; DCH: trace_native + input targets);
they fail, never skip, without a GPU."""
import os
import pathlib
import signal

import numpy as np
import pytest
import torch
import torch.nn as nn

import wavefm.data as wdata
from data import store
from wavefm.models import backbones, build
from wavefm.train import pipeline, rundir, ssl
from wavefm.utils import config, runtime
from wavefm.utils.io import load_json

DRO = {"scaling": "global_minmax", "target_space": "zscore"}
DCH = {"scaling": "trace_native", "target_space": "input"}


@pytest.fixture(autouse=True)
def strict():
    runtime.setup_numerics()
    yield
    runtime.setup_numerics()


@pytest.fixture
def fake_tokenizer(monkeypatch):
    """A seeded stand-in for the pretrained tokenizer (the only pretrained part of ``blocks: none``)."""
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module as TFM
    from timesfm.torch.dense import ResidualBlock
    torch.manual_seed(11)
    sd = {f"tokenizer.{k}": v for k, v in ResidualBlock(TFM.config.tokenizer).state_dict().items()}
    monkeypatch.setattr(backbones, "hf_weights", lambda version: sd)
    return sd


def make(fake, setting=DRO, blocks="none", stages=("ssl",), name=None, **dyn):
    c = config.merge(fake, {"stages": list(stages), "run": {"smoke": True, **({"name": name} if name else {})},
                            "model": {"blocks": blocks, "scaling": setting["scaling"], "adapter": True},
                            "ssl": {"steps": 5, "epochs": 0, "batch_size": 8, "eval_every": 1, "compile": False,
                                    "target_space": setting["target_space"], **dyn}})
    config.check(c)
    return c


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def run(cfg, where):
    return ssl.run(cfg, where / "ssl", None)


# ---- ① five steps, both terms recorded, only the promised files -----------------------------------------------------

@pytest.mark.parametrize("setting", [DRO, DCH], ids=["dro", "dch"])
def test_five_steps_record_both_terms(fake_dro, fake_dch, fake_tokenizer, tmp_path, setting):
    fake = fake_dro if setting is DRO else fake_dch
    r = run(make(fake, setting), tmp_path)
    out = tmp_path / "ssl"
    hist = load_json(out / "history.json")
    assert [h["step"] for h in hist] == [1, 2, 3, 4, 5]
    for h in hist:
        assert {"train_loss", "train_next", "train_masked", "lr", "grad_norm", "clip_frac"} <= h.keys()
        assert "val_loss" not in h and all(np.isfinite(h[k]) for k in ("train_next", "train_masked"))
        assert h["train_loss"] == pytest.approx(h["train_next"] + h["train_masked"], rel=1e-5)
    assert r["final_next"] == hist[-1]["train_next"] and r["final_masked"] == hist[-1]["train_masked"]
    assert r["actual"]["steps"] == 5 and r["final_val_loss"] is None
    assert sorted(p.name for p in out.iterdir()) == ["history.json", "result.json", "weights.pt"]
    w = weights(out / "weights.pt")
    others = sorted(k for k in w if not k.startswith("encoder."))
    assert others == ["mask_token", "ssl_masked.bias", "ssl_masked.weight", "ssl_next.bias", "ssl_next.weight"]
    assert run(make(fake, setting), tmp_path) == r                       # finished: returned, not retrained


def test_loss_weights_and_mask_prob_are_used(fake_dro, fake_tokenizer, tmp_path):
    r = run(make(fake_dro, next_weight=0.5, masked_weight=2.0, steps=2), tmp_path)
    for h in load_json(tmp_path / "ssl" / "history.json"):
        assert h["train_loss"] == pytest.approx(0.5 * h["train_next"] + 2.0 * h["train_masked"], rel=1e-5)
    assert r["constants"]["space"] == "zscore"


# ---- ② what is trained and what is frozen ----------------------------------------------------------------------------

def test_trained_parts_change_and_the_transformer_does_not(fake_dro, tmp_path):
    cfg = make(fake_dro, DRO, blocks="random", steps=2)
    model0, token0, _ = ssl.build(cfg)
    before = {k: v.clone() for k, v in model0.state_dict().items()}
    run(cfg, tmp_path)
    w = weights(tmp_path / "ssl" / "weights.pt")
    layers = [k for k in before if k.startswith("encoder.layers.")]
    assert len(layers) > 100 and all(torch.equal(w[k], before[k]) for k in layers)
    for prefix in ("encoder.tokenizer.", "encoder.adapter.conv1.", "encoder.adapter.conv2.", "ssl_next.",
                   "ssl_masked."):
        keys = [k for k in before if k.startswith(prefix)]
        assert keys and all(not torch.equal(w[k], before[k]) for k in keys), prefix
    assert not torch.equal(w["mask_token"], token0.detach())

    # gradients at the trained state: every trained tensor gets a non-zero one, the transformer none
    model, token, groups = ssl.build(cfg)
    model.load_state_dict({k: v for k, v in w.items() if k != "mask_token"}, strict=True)
    token.data.copy_(w["mask_token"])
    dev = wdata.dev(cfg)
    c = ssl.constants(cfg, dev.meta["waveform"]["stats"])
    x = torch.from_numpy(dev.take(np.arange(8)))
    m = torch.rand(8, x.shape[1] // 32, generator=torch.Generator().manual_seed(0)) < 0.25
    nxt, msk = ssl.terms(model.encoder.forward_patches, model, *ssl.views(model.encoder, x, m, token, c), m)
    (nxt + msk).backward()
    trained = {id(t) for g in groups for t in g["params"]}
    names = {id(t): n for n, t in model.named_parameters()} | {id(token): "mask_token"}
    for g in groups:
        for t in g["params"]:
            assert t.grad is not None and t.grad.abs().sum() > 0, names[id(t)]
    for n, t in model.encoder.layers.named_parameters():
        assert not t.requires_grad and t.grad is None and id(t) not in trained, n
    assert {g["name"] for g in groups} == {"tokenizer", "adapter", "heads"}
    s = cfg["ssl"]
    assert [g["lr"] for g in groups] == [s["lr_tokenizer"], s["lr_adapter"], s["lr_heads"]]
    assert all("weight_decay" not in g for g in groups)        # the stage's weight_decay on all, as the old scripts


# ---- ③ the input and target spaces equal the hand-written formulas ------------------------------------------------

def hand_causal(p):
    out = torch.empty_like(p)
    for i in range(p.shape[1]):
        seen = p[:, :i + 1].reshape(len(p), -1)
        m, s = seen.mean(1), seen.std(1, unbiased=False)
        out[:, i] = (p[:, i] - m[:, None]) / torch.where(s < 1e-6, torch.ones_like(s), s)[:, None]
    return out


def hand_trace(x):
    mu = x.mean(1, keepdim=True)
    sd = ((x - mu) ** 2).sum(1, keepdim=True).div(x.shape[1] - 1).sqrt()
    return (x - mu) / torch.where(sd < 1e-6, torch.ones_like(sd), sd)


SUPPORTED = [("input", s) for s in backbones.SCALINGS] + [("zscore", "global_minmax")]


@pytest.mark.parametrize("space, scaling", SUPPORTED)
def test_spaces_equal_hand_written_formulas(fake_tokenizer, scaling, space):
    enc = build.build_encoder({**make_model(), "scaling": scaling}, seed=0).double()
    torch.manual_seed(3)
    B, N = 3, 4
    x = (torch.randn(B, N * 32) * 40 + 7).double()
    m = torch.rand(B, N) < 0.4
    m[0, 0] = True
    token = torch.randn(32, dtype=torch.float64)
    lo, hi, mu, sd = -50.0, 120.0, 6.0, 41.0
    stats = {"min": lo, "max": hi, "mean": mu, "std": sd}
    c = ssl.constants({"model": {"scaling": scaling}, "ssl": {"target_space": space}}, stats)

    def to_patches(raw):                                            # the encoder's input path, by hand
        z = (raw - lo) / (hi - lo) if scaling == "global_minmax" else raw
        z = hand_trace(z) if scaling == "trace_native" else z
        p = z.reshape(B, N, 32)
        return p if scaling == "global_minmax" else hand_causal(p)

    P_want = to_patches(x)
    if space == "input":
        T_want = P_want
        Pm_want = P_want.clone()
        Pm_want[m] = token
    else:
        T_want = ((x - mu) / sd).reshape(B, N, 32)
        zm = T_want.clone()
        zm[m] = token
        Pm_want = to_patches(zm.reshape(B, -1) * sd + mu)
    P, Pm, T = ssl.views(enc, x, m, token, c)
    for got, want in ((P, P_want), (Pm, Pm_want), (T, T_want)):
        assert torch.allclose(got, want, atol=1e-10, rtol=1e-10)

    # the two loss terms, written as loops over rows and patches
    model = nn.ModuleDict({"encoder": enc, "ssl_next": nn.Linear(1280, 32), "ssl_masked": nn.Linear(1280, 32)}).double()
    nxt, msk = ssl.terms(enc.forward_patches, model, P, Pm, T, m)
    with torch.no_grad():
        tok, tok_m = enc.forward_patches(P_want), enc.forward_patches(Pm_want)
        e_next = [((model.ssl_next(tok[b, i]) - T_want[b, i + 1]) ** 2).mean() for b in range(B) for i in range(N - 1)]
        e_mask = [((model.ssl_masked(tok_m[b, i]) - T_want[b, i]) ** 2).mean()
                  for b in range(B) for i in range(N) if m[b, i]]
    assert torch.allclose(nxt, torch.stack(e_next).mean(), rtol=1e-10)
    assert torch.allclose(msk, torch.stack(e_mask).sum() / len(e_mask), rtol=1e-10)


@pytest.mark.parametrize("space, scaling", SUPPORTED + [("zscore", "native")])
def test_finite_gradients_at_the_zero_mask_token(fake_tokenizer, space, scaling):
    """The start of every run: mask token 0, first patch masked. Every supported combination has finite gradients;
    zscore with a per-patch scaling (refused by check_config) is the control that has not."""
    enc = build.build_encoder({**make_model(), "scaling": scaling}, seed=0).set_trainable(0, True)
    model = nn.ModuleDict({"encoder": enc, "ssl_next": nn.Linear(1280, 32), "ssl_masked": nn.Linear(1280, 32)})
    token = nn.Parameter(torch.zeros(32))
    x = torch.randn(3, 4 * 32, generator=torch.Generator().manual_seed(4)) * 40 + 7
    m = torch.zeros(3, 4, dtype=torch.bool)
    m[:, 0], m[1, 2] = True, True
    stats = {"min": float(x.min()), "max": float(x.max()), "mean": 7.0, "std": 40.0}
    c = ssl.constants({"model": {"scaling": scaling}, "ssl": {"target_space": space}}, stats)
    nxt, msk = ssl.terms(enc.forward_patches, model, *ssl.views(enc, x, m, token, c), m)
    (nxt + msk).backward()
    grads = [t.grad for t in [*model.parameters(), token] if t.requires_grad]
    finite = all(g is not None and torch.isfinite(g).all() for g in grads)
    assert finite == ((space, scaling) in SUPPORTED) and torch.isfinite(nxt + msk)


def make_model():
    return {"kind": "timesfm", "version": "2.5", "blocks": "none", "scaling": "native", "adapter": True,
            "widths": [32, 64, 128], "head_hidden": 128}


def test_zscore_uses_the_dev_statistics(fake_dro):
    stats = wdata.dev(fake_dro).meta["waveform"]["stats"]
    c = ssl.constants(make(fake_dro, DRO), stats)
    assert (c["lo"], c["hi"], c["mu"], c["sd"]) == (stats["min"], stats["max"], stats["mean"], stats["std"])
    with pytest.raises(ssl.SslError, match="spread"):
        ssl.constants(make(fake_dro, DRO), {**stats, "std": 0.0})
    with pytest.raises(ssl.SslError, match="minimum"):
        ssl.constants(make(fake_dro, DRO), {**stats, "max": stats["min"]})
    assert "mu" not in ssl.constants(make(fake_dro, DCH), stats)       # input space: no data-set constants


# ---- ④ exact resume -------------------------------------------------------------------------------------------------

def interrupted_run(cfg, where, monkeypatch, at):
    real, n = ssl.terms, [0]

    def kill(*a):
        n[0] += 1
        if n[0] == at:
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*a)

    monkeypatch.setattr(ssl, "terms", kill)
    assert run(cfg, where) == {"preempted": True} and (where / "ssl" / "ckpt.pt").exists()
    monkeypatch.setattr(ssl, "terms", real)
    return run(cfg, where)


def test_resume_is_exact(fake_dch, fake_tokenizer, tmp_path, monkeypatch):
    cfg = make(fake_dch, DCH, steps=6, eval_every=2)
    run(cfg, tmp_path / "ref")
    for at in (3, 4):                                                 # mid-interval, on an evaluation boundary
        interrupted_run(cfg, tmp_path / f"int{at}", monkeypatch, at)
        assert same(weights(tmp_path / f"int{at}/ssl/weights.pt"), weights(tmp_path / "ref/ssl/weights.pt"))
        assert load_json(tmp_path / f"int{at}/ssl/history.json") == load_json(tmp_path / "ref/ssl/history.json")
    other = config.merge(cfg, {"seed": 1})                            # control: another seed gives other masks
    run(other, tmp_path / "seed1")
    assert not same(weights(tmp_path / "seed1/ssl/weights.pt"), weights(tmp_path / "ref/ssl/weights.pt"))


# ---- ⑤ labels are never read ----------------------------------------------------------------------------------------

def test_labels_change_nothing(fake_dro, fake_tokenizer, tmp_path, monkeypatch):
    cfg = make(fake_dro, DRO)
    run(cfg, tmp_path / "ref")
    real = wdata.dev
    for kind in ("garbage", "absent"):
        def dev(c, kind=kind):
            s = real(c)
            if kind == "garbage":
                s.labels = np.full_like(s.labels, np.nan)
            else:
                del s.labels                                          # any read raises AttributeError
            return s
        monkeypatch.setattr(wdata, "dev", dev)
        run(cfg, tmp_path / kind)
        assert same(weights(tmp_path / kind / "ssl/weights.pt"), weights(tmp_path / "ref/ssl/weights.pt")), kind
        assert load_json(tmp_path / kind / "ssl/history.json") == load_json(tmp_path / "ref/ssl/history.json"), kind


# ---- the next stage, refusals, the adapter at zero ------------------------------------------------------------------

def test_ssl_then_ridge_through_the_pipeline(fake_dch, fake_tokenizer):
    cfg = make(fake_dch, DCH, stages=("ssl", "ridge"), steps=3)
    done = pipeline.run(cfg)
    d = pipeline.run_dir(cfg)
    assert set(done["results"]) == {"ssl", "ridge"}
    assert done["results"]["ridge"]["parent"]["file"] == str(d / "ssl/weights.pt")
    dev = wdata.dev(cfg)
    pred = rundir.read_teacher_labels(d / "ridge/teacher_labels.npz", dev)
    assert pred.shape == (len(dev), 1) and np.isfinite(pred).all()
    enc = build.build_encoder(cfg["model"], init_from=d / "ssl/weights.pt", seed=5)
    w = weights(d / "ssl/weights.pt")
    assert all(torch.equal(v, w[f"encoder.{k}"]) for k, v in enc.state_dict().items())


@pytest.mark.parametrize("over, match", [({"ssl": {"keep": "best_val"}}, "no validation set"),
                                         ({"ssl": {"schedule": "plateau", "eval_every": 5}}, "no validation set"),
                                         ({"ssl": {"target_space": "raw"}}, "target_space"),
                                         ({"model": {"scaling": "native"}}, "zscore with model.scaling"),
                                         ({"model": {"scaling": "native_stats"}}, "zscore with model.scaling"),
                                         ({"model": {"scaling": "trace_native"}}, "zscore with model.scaling"),
                                         ({"ssl": {"mask_prob": 0.0}}, "mask_prob"),
                                         ({"model": {"kind": "cnn"}}, "kind")])
def test_refused_before_any_work(fake_dro, tmp_path, monkeypatch, over, match):
    def no_data(c):
        raise AssertionError("the data were opened before the config was checked")
    monkeypatch.setattr(wdata, "dev", no_data)
    with pytest.raises(ssl.SslError, match=match):
        run(config.merge(make(fake_dro), over), tmp_path)


def test_training_never_opens_heldout(fake_dro, fake_tokenizer, tmp_path, monkeypatch):
    """Every split opened, array loaded and file read by the stage is recorded; none is under heldout/."""
    opened, init, load, read = [], store.Split.__init__, np.load, pathlib.Path.read_text
    monkeypatch.setattr(store.Split, "__init__", lambda self, path: (opened.append(str(path)), init(self, path))[1])
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (opened.append(str(f)), load(f, *a, **k))[1])
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: (opened.append(str(self)), read(self, *a, **k))[1])
    run(make(fake_dro, steps=2), tmp_path)
    assert any(p.endswith("dev") for p in opened) and not [p for p in opened if "heldout" in p], opened


def test_mode_dependent_modules_are_refused():
    ssl.check_modes(nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.GELU()}))
    for bad in (nn.Dropout(0.1), nn.BatchNorm1d(3)):
        with pytest.raises(ssl.SslError, match="train"):
            ssl.check_modes(nn.ModuleDict({"enc": nn.Sequential(nn.Linear(3, 3), bad)}))


def test_zero_initialised_adapter_leaves_the_encoder_unchanged(fake_dro):
    a = build.build_encoder({**make_model(), "blocks": "random", "adapter": True}, seed=2).eval()
    b = build.build_encoder({**make_model(), "blocks": "random", "adapter": False}, seed=2).eval()
    x = torch.randn(3, 128) * 5
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_ssl_has_no_predict():
    assert not hasattr(ssl, "predict")                 # scripts/evaluate.py skips it (tests/test_tasks.py)


def test_a_parent_gives_the_starting_encoder(fake_dch, fake_tokenizer, tmp_path):
    cfg = make(fake_dch, DCH, steps=2)
    run(cfg, tmp_path / "first")
    w = weights(tmp_path / "first/ssl/weights.pt")
    model, token, _ = ssl.build(cfg, tmp_path / "first/ssl")
    assert all(torch.equal(v, w[f"encoder.{k}"]) for k, v in model.encoder.state_dict().items())
    assert not token.detach().any()                                   # heads and token are not passed on
    r = ssl.run(cfg, tmp_path / "second/ssl", tmp_path / "first/ssl")
    assert r["parent"]["sha256"] == rundir.file_sha256(tmp_path / "first/ssl/weights.pt")


# ---- GPU: the real pretrained TimesFM 2.5 ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("setting", [DRO, DCH], ids=["dro", "dch"])
def test_pretrained_on_gpu(fake_dro, fake_dch, tmp_path, monkeypatch, setting):
    assert torch.cuda.is_available(), "this test needs a GPU"
    fake = fake_dro if setting is DRO else fake_dch
    cfg = make(fake, setting, blocks="pretrained", batch_size=16, steps=5, eval_every=2, compile=setting is DRO)
    r = run(cfg, tmp_path / "ref")
    hist = load_json(tmp_path / "ref/ssl/history.json")
    assert all(np.isfinite(h["train_next"]) and np.isfinite(h["train_masked"]) for h in hist)
    w, hf = weights(tmp_path / "ref/ssl/weights.pt"), backbones.hf_weights("2.5")
    layers = [k for k in hf if k.startswith("stacked_xf.")]
    assert len(layers) > 100 and all(torch.equal(w["encoder.layers." + k[len("stacked_xf."):]], hf[k]) for k in layers)
    tok = [k for k in hf if k.startswith("tokenizer.")]
    assert all(not torch.equal(w["encoder." + k], hf[k]) for k in tok)
    assert r["peak_mem_gb"] is not None
    print(f"\n[ssl gpu] {setting}: {r['steps_per_s']:.2f} steps/s at batch 16 (fixture length), peak "
          f"{r['peak_mem_gb']:.2f} GB, compile {cfg['ssl']['compile']}")
    if setting is DCH:                                                # ④ on the GPU (CUDA generator for the masks)
        cfg = make(fake, setting, blocks="pretrained", batch_size=16, steps=5, eval_every=2)
        interrupted_run(cfg, tmp_path / "int", monkeypatch, 3)
        assert same(weights(tmp_path / "int/ssl/weights.pt"), w)
        assert load_json(tmp_path / "int/ssl/history.json") == hist
