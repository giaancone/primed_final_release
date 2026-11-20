"""The ridge stage (train/ridge.py): the closed-form mathematics against direct solutions, and the stage on a tiny
fake data set with a fake encoder (CPU, seconds)."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

import fakes
import wavefm.data as wdata
from data import store
from wavefm import paths
from wavefm.models import backbones
from wavefm.train import guards, ridge, rundir
from wavefm.utils import config

TOL = 1e-8


def problem(n, d, k=3, seed=0, noise=0.1):
    """Features with offsets and spreads far from 0 and 1, targets linear in them plus noise."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, d)) * rng.uniform(0.1, 20, d) + rng.uniform(-50, 50, d)
    W, b = rng.normal(size=(d, k)), rng.normal(size=k) * 10
    return F, F @ W + b + noise * rng.normal(size=(n, k)), W, b


def direct(F, Y, alpha, mean, scale):
    """Normal equations of [1, Z] with the penalty diag(0, alpha I), solved directly: (coef, intercept)."""
    Z = (F - mean) / scale
    A = np.hstack([np.ones((len(Z), 1)), Z])
    P = alpha * np.eye(A.shape[1]); P[0, 0] = 0
    s = np.linalg.solve(A.T @ A + P, A.T @ Y)
    return s[1:] / scale[:, None], s[0] - (mean / scale) @ s[1:]


# ---- mathematics ----------------------------------------------------------------------------------------------------

def test_alpha_grid():
    g = ridge.alpha_grid(config.load(paths.ROOT / "configs" / "teacher.yaml")["ridge"])
    assert len(g) == 27 and g[0] == 1e-8 and g[-1] == 1e5 and np.allclose(np.diff(np.log10(g)), 0.5)
    with pytest.raises(ridge.RidgeError, match="whole"):
        ridge.alpha_grid({"log10_alpha_min": -8.0, "log10_alpha_max": 5.2, "alphas_per_decade": 2})


def test_recovers_coefficients_without_regularisation():                                     # acceptance 1
    F, Y, W, b = problem(50, 6, noise=0.0)
    m = ridge.fit(F, Y, [0.0])
    assert m["solver"] == "primal" and m["rank"] == 6
    assert np.abs(m["coef"] - W).max() < TOL and np.abs(m["intercept"] - b).max() < TOL


@pytest.mark.parametrize("n, d", [(40, 6), (20, 50)])
@pytest.mark.parametrize("alpha", [1e-3, 0.7, 30.0, 1e4])
def test_equals_normal_equations(n, d, alpha):                                               # acceptance 1 and 3
    F, Y, _, _ = problem(n, d)
    m = ridge.fit(F, Y, [alpha])
    coef, icpt = direct(F, Y, alpha, m["mean"], m["scale"])
    assert m["solver"] == ("primal" if n > d else "dual")
    assert np.abs(m["coef"] - coef).max() < TOL and np.abs(m["intercept"] - icpt).max() < TOL
    assert np.abs((F @ m["coef"] + m["intercept"]) - (F @ coef + icpt)).max() < TOL


@pytest.mark.parametrize("alpha", [0.0, 1e-2, 5.0])
def test_fewer_rows_than_features_is_finite_and_the_dual_solution(alpha):                     # acceptance 3
    F, Y, _, _ = problem(12, 200)
    m = ridge.fit(F, Y, [alpha])
    assert m["solver"] == "dual" and all(np.isfinite(m[k]).all() for k in ("coef", "intercept"))
    assert np.isfinite(m["loo"]).all() == (alpha > 0)                # alpha 0 interpolates: leave-one-out is inf
    Z = (F - m["mean"]) / m["scale"]
    Zc, Yc = Z - Z.mean(0), Y - Y.mean(0)
    w = Zc.T @ np.linalg.pinv(Zc @ Zc.T + alpha * np.eye(len(Zc))) @ Yc                     # dual (min-norm at 0)
    assert np.abs(m["coef"] - w / m["scale"][:, None]).max() < TOL


@pytest.mark.parametrize("d", [5, 60])
def test_closed_form_leave_one_out_equals_refits(d):                                          # acceptance 2
    F, Y, _, _ = problem(30, d, noise=1.0)
    grid = np.array([1e-2, 0.3, 1.0, 10.0, 300.0])
    m = ridge.fit(F, Y, grid)
    for j, a in enumerate(grid):
        err = np.zeros_like(Y)
        for i in range(len(F)):
            keep = np.arange(len(F)) != i                    # the standardisation is held fixed, as documented
            coef, icpt = direct(F[keep], Y[keep], a, m["mean"], m["scale"])
            err[i] = Y[i] - (F[i] @ coef + icpt)
        assert np.abs(m["loo"][j] - (err ** 2).mean(0)).max() < TOL, (d, a)
    assert (m["index"] == m["loo"].argmin(0)).all() and (m["alpha"] == grid[m["index"]]).all()


def test_alpha_at_the_grid_edge_is_flagged():                                                 # acceptance 4
    rng = np.random.default_rng(1)
    F = rng.normal(size=(60, 10))
    Y = np.stack([F @ rng.normal(size=10),                       # noiseless: the smallest alpha wins
                  F @ rng.normal(size=10) + 3 * rng.normal(size=60)], 1)     # noisy: an alpha of order 10
    m = ridge.fit(F, Y, 10.0 ** np.arange(-6, 7))
    assert m["index"][0] == 0 and 0 < m["index"][1] < 12 and m["edge"] == ["min", None]
    m = ridge.fit(F, Y, 10.0 ** np.arange(-9, -5))                # a grid that stops below the noisy optimum
    assert m["index"][1] == 3 and m["edge"][1] == "max"


def test_equivariant_under_affine_targets():
    F, Y, _, _ = problem(40, 8, noise=2.0)
    c, s = np.array([-37.5, 1e-3, 4.0]), np.array([1e3, -0.2, 7.0])
    grid = 10.0 ** np.arange(-3, 4, 0.5)
    m, m2 = ridge.fit(F, Y, grid), ridge.fit(F, Y * c + s, grid)
    np.testing.assert_allclose(F @ m2["coef"] + m2["intercept"], (F @ m["coef"] + m["intercept"]) * c + s,
                               rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(m2["loo"], m["loo"] * c ** 2, rtol=1e-10)
    assert (m2["index"] == m["index"]).all()


@pytest.mark.parametrize("on_dev", [False, True])
def test_alpha_convention_invariant_under_affine_features(on_dev):
    """Alpha acts on the standardised features: rescaling and shifting a feature changes nothing."""
    F, Y, _, _ = problem(40, 8, noise=2.0)
    extra = problem(100, 8, seed=5)[0]
    c, s = np.random.default_rng(2).uniform(-30, 30, 8), np.random.default_rng(3).uniform(-1e3, 1e3, 8)
    grid = 10.0 ** np.arange(-3, 4, 0.5)
    rows = np.vstack([F, extra]) if on_dev else None                 # standardised on more rows than are fitted
    m = ridge.fit(F, Y, grid, F_standardise=rows)
    m2 = ridge.fit(F * c + s, Y, grid, F_standardise=None if rows is None else rows * c + s)
    np.testing.assert_allclose(m2["loo"], m["loo"], rtol=1e-9)
    np.testing.assert_allclose((F * c + s) @ m2["coef"] + m2["intercept"], F @ m["coef"] + m["intercept"], rtol=1e-9)
    assert (m2["index"] == m["index"]).all()


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_constant_features_get_zero_coefficients(alpha):
    """A column constant everywhere (2) and one constant on the fitted rows only (5) change nothing."""
    F, Y, _, _ = problem(30, 4)
    G = problem(20, 4, seed=9)[0]                                             # rows used only for standardisation
    on_fit_only = np.r_[np.full(30, -2.5), np.linspace(0, 9, 20)][:, None]
    with_const = lambda X, c: np.hstack([X[:, :2], np.full((len(X), 1), 3.7), X[:, 2:], c])
    for fs in (None, np.vstack([F, G])):
        m = ridge.fit(with_const(F, on_fit_only[:30]), Y, [alpha],
                      F_standardise=None if fs is None else with_const(fs, on_fit_only))
        m0 = ridge.fit(F, Y, [alpha], F_standardise=fs)
        assert all(np.isfinite(m[k]).all() for k in ("coef", "intercept", "mean", "scale"))
        assert (m["coef"][[2, 5]] == 0).all() and m["scale"][2] == 1.0 and m["mean"][2] == 3.7
        np.testing.assert_allclose(m["coef"][[0, 1, 3, 4]], m0["coef"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(m["intercept"], m0["intercept"], rtol=1e-10)
    m = ridge.fit(np.full((10, 3), 1.5), Y[:10], [0.0, 1.0])                                    # nothing to learn
    assert m["rank"] == 0 and (m["coef"] == 0).all() and np.allclose(m["intercept"], Y[:10].mean(0))


def test_interpolating_fit_at_alpha_zero_has_infinite_leave_one_out():
    """3 rows, 3 features: the fit at alpha 0 interpolates (n <= rank + 1), so its leave-one-out error is inf by rule;
    round-off leaves 1 - h_ii slightly above 0 here, which made it finite and chosen. A non-interpolating fit at 0
    keeps a finite value."""
    rng = np.random.default_rng(8)
    F, Y = rng.normal(size=(3, 3)), rng.normal(size=(3, 1))
    m = ridge.fit(F, Y, [0.0, 1e-8])
    assert m["rank"] >= 2 and np.isinf(m["loo"][0]).all() and np.isfinite(m["loo"][1]).all() and m["index"][0] == 1
    F, Y, _, _ = problem(30, 4)
    assert np.isfinite(ridge.fit(F, Y, [0.0, 1.0])["loo"][0]).all()


def test_fit_refuses_bad_input():
    F, Y, _, _ = problem(10, 3)
    for args in ((F, Y[:9], [1.0]), (F[:2], Y[:2], [1.0]), (F, Y, [1.0, 0.1]), (F, Y, [-1.0]),
                 (np.where(np.eye(10, 3) > 0, np.nan, F), Y, [1.0])):
        with pytest.raises(ridge.RidgeError):
            ridge.fit(*args)


# ---- the stage ------------------------------------------------------------------------------------------------------

D = 48                    # features of the fake encoder: more than the 40 labelled rows, so the stage takes the dual


class FakeEncoder(torch.nn.Module):
    """The interface of wavefm.models.build.build_encoder: forward(x, block=-1) -> (B, N, D), patches(x)."""

    def __init__(self, d=D, n_blocks=2):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.proj = torch.nn.Parameter(torch.randn(32, d, generator=g) / 4)
        self.blocks = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)
                                              for _ in range(n_blocks)])

    def patches(self, x):
        return x.reshape(len(x), -1, 32)

    def forward(self, x, block=-1):
        h = torch.tanh(self.patches(x) @ self.proj)
        for w in self.blocks[:len(self.blocks) if block == -1 else block + 1]:
            h = h + torch.tanh(h @ w)
        return h


@pytest.fixture
def stage(fake_dro, tmp_path, monkeypatch):
    """The fake DRO dev split (conftest), the fake encoder, an ssl parent directory, strict fp32."""
    from wavefm.models import build
    real_dev = wdata.dev

    class S:
        calls, nan_outside, tmp = [], False, tmp_path
        cfg = config.merge(fake_dro, {"stages": ["ssl", "ridge"], "data": {"label_frac": 0.1},
                                      "ridge": {"batch_size": 7}})

    def dev(cfg):
        s = real_dev(cfg)
        if S.nan_outside:                                     # every label outside the budget is unreadable
            s.labels = s.labels.copy()                        # the split's own array is read-only
            s.labels[np.setdiff1d(np.arange(len(s)), ridge.labelled_rows(cfg, len(s)))] = np.nan
        return s

    def build_encoder(model_cfg, init_from=None, seed=0):
        S.calls.append((init_from, seed))
        enc = FakeEncoder()
        if init_from is not None:
            enc.load_state_dict(torch.load(init_from))
        return enc

    monkeypatch.setattr(wdata, "dev", dev)
    monkeypatch.setattr(build, "build_encoder", build_encoder, raising=False)
    S.dev, S.parent = dev, tmp_path / "run" / "ssl"
    S.parent.mkdir(parents=True)
    w = FakeEncoder().state_dict(); w["proj"] = w["proj"] * 1.5
    torch.save(w, S.parent / "weights.pt")
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32   # strict fp32, restored after
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield S
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def run_in(S, name, cfg=None):
    cfg = cfg or S.cfg
    (S.tmp / name).mkdir(exist_ok=True)
    (S.tmp / name / "config.json").write_text(json.dumps(cfg))
    return ridge.run(cfg, S.tmp / name / "ridge", S.parent), S.tmp / name / "ridge"


def test_run_and_predict_end_to_end(stage, fake_dch):
    r, out = run_in(stage, "run")
    dev, lab = stage.dev(stage.cfg), ridge.labelled_rows(stage.cfg, 400)
    assert len(lab) == 40 and r["stage"] == "ridge" and r["requested"] == r["actual"] == {"rows": 40}
    assert r["n_features"] == D and len(r["alpha"]) == 3 and set(r["hints"]) == {"edge", "loo_over_target_var"}
    assert r["parent"]["sha256"] == rundir.file_sha256(stage.parent / "weights.pt")
    assert set(r["artifacts"]) == {"ridge.npz", "loo.json", "teacher_labels.npz"}
    assert stage.calls == [(stage.parent / "weights.pt", stage.cfg["seed"])]
    loo = json.loads((out / "loo.json").read_text())
    assert np.array(loo["loo_mse"]).shape == (27, 3) and loo["outputs"] == ["C", "S", "t0"] and loo["rows"] == 40
    assert loo["solver"] == "dual" and np.isfinite(loo["loo_mse"]).all()
    assert not (out / "history.json").exists()                                               # no faked curve
    pred = rundir.read_teacher_labels(out / "teacher_labels.npz", dev)                        # acceptance 5: contract
    assert pred.shape == (400, 3) and pred.dtype == np.float64
    np.testing.assert_array_equal(ridge.predict(out, dev), pred)
    assert stage.calls[-1] == (stage.parent / "weights.pt", stage.cfg["seed"])
    # the exported labels are fit() on the labelled rows' features, applied to every dev row
    enc = FakeEncoder(); enc.load_state_dict(torch.load(stage.parent / "weights.pt"))
    F = ridge.features(enc.to("cuda" if torch.cuda.is_available() else "cpu"), dev, stage.cfg,   # the stage's device
                       ridge.input_range(stage.cfg, dev))
    m = ridge.fit(F[lab], dev.labels[lab].astype(np.float64), ridge.alpha_grid(stage.cfg["ridge"]))
    np.testing.assert_allclose(pred, F @ m["coef"] + m["intercept"], rtol=1e-12)
    n = len(stage.calls)
    assert ridge.run(stage.cfg, out, stage.parent) == r and len(stage.calls) == n          # finished: not redone
    with pytest.raises(rundir.RunError, match="another"):
        ridge.run(config.merge(stage.cfg, {"ridge": {"block": 0}}), out, stage.parent)
    for other in (store.Split(fakes.make_split(stage.tmp / "o", "npy")[0]), wdata.dev(fake_dch)):
        with pytest.raises(rundir.RunError, match="another data set"):                       # acceptance 5: other dev
            rundir.read_teacher_labels(out / "teacher_labels.npz", other)
    assert r["encoder_from"] is None                                                          # it has a parent
    with np.load(out / "ridge.npz") as z:
        z = dict(z)
    np.savez(out / "ridge.npz", **{**z, "intercept": z["intercept"] + 1})                      # edited after the stage
    with pytest.raises(guards.GuardError, match="changed after the stage finished"):
        ridge.predict(out, dev)
    np.savez(out / "ridge.npz", **z)
    torch.save(FakeEncoder().state_dict(), stage.parent / "weights.pt")
    with pytest.raises(ridge.RidgeError, match="changed"):
        ridge.predict(out, dev)


class OtherStats:
    """The rows of ``split`` with other waveform minimum and maximum in its meta.json (as heldout has)."""

    def __init__(self, split):
        self.split, self.meta = split, copy.deepcopy(split.meta)
        st = self.meta["waveform"]["stats"]
        st["min"], st["max"] = st["min"] - 50.0, st["max"] * 3.0

    def __len__(self):
        return len(self.split)

    def take(self, idx):
        return self.split.take(idx)


def test_global_minmax_uses_the_dev_constants_saved_with_the_fit(stage):
    assert stage.cfg["model"]["scaling"] == "global_minmax"
    _, out = run_in(stage, "run")
    dev = stage.dev(stage.cfg)
    st = dev.meta["waveform"]["stats"]
    with np.load(out / "ridge.npz") as z:
        assert z["input_lo_hi"].tolist() == [st["min"], st["max"]]
    labels = rundir.read_teacher_labels(out / "teacher_labels.npz", dev)
    np.testing.assert_array_equal(ridge.predict(out, OtherStats(dev)), labels)    # identical waveforms, same answer


def test_first_stage_records_the_pretrained_file_or_the_seed(stage, monkeypatch):
    """Without a parent, the sha256 of the pretrained file actually resolved is recorded and checked by predict;
    the random control records its seed."""
    f = stage.tmp / "model.safetensors"
    f.write_bytes(b"pretrained v1")
    monkeypatch.setattr(backbones, "pretrained_file", lambda version: str(f))
    cfg = config.merge(stage.cfg, {"stages": ["ridge"]})
    (stage.tmp / "solo").mkdir()
    (stage.tmp / "solo" / "config.json").write_text(json.dumps(cfg))
    out = stage.tmp / "solo" / "ridge"
    r = ridge.run(cfg, out, None)
    want = {"pretrained_sha256": rundir.file_sha256(f)}
    assert r["parent"] is None and r["encoder_from"] == want and stage.calls[-1] == (None, cfg["seed"])
    with np.load(out / "ridge.npz") as z:
        assert json.loads(str(z["encoder_from"])) == want
    dev = stage.dev(cfg)
    np.testing.assert_array_equal(ridge.predict(out, dev), rundir.read_teacher_labels(out / "teacher_labels.npz", dev))
    f.write_bytes(b"pretrained v2")                                   # the cache now resolves to other weights
    with pytest.raises(ridge.RidgeError, match="pretrained_sha256"):
        ridge.predict(out, dev)
    rnd = config.merge(cfg, {"model": {"blocks": "random"}, "run": {"name": "rnd"}})
    assert ridge.run(rnd, stage.tmp / "rnd" / "ridge", None)["encoder_from"] == {"seed": cfg["seed"]}


def test_predict_in_a_fresh_process_sets_strict_fp32(stage):
    """A new interpreter has TF32 at torch's defaults (cudnn on): predict sets strict fp32 itself."""
    _, out = run_in(stage, "run")
    code = f"""
import sys, torch
sys.path[:0] = [{str(paths.ROOT)!r}, {str(paths.ROOT / 'tests')!r}]
assert torch.backends.cudnn.allow_tf32                     # torch's default in a fresh process
from test_ridge import FakeEncoder
from data import store
from wavefm.models import build
from wavefm.train import ridge

def build_encoder(model_cfg, init_from=None, seed=0):
    enc = FakeEncoder()
    enc.load_state_dict(torch.load(init_from))
    return enc

build.build_encoder = build_encoder
print(ridge.predict({str(out)!r}, store.Split({str(stage.dev(stage.cfg).path)!r})).shape)
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0 and "(400, 3)" in p.stdout, p.stderr[-2000:]


@pytest.mark.parametrize("on", ["labelled", "dev"])
def test_reads_no_label_outside_the_budget(stage, on):                                        # acceptance 5
    cfg = config.merge(stage.cfg, {"ridge": {"standardise_on": on}})
    _, a = run_in(stage, "clean", cfg)
    stage.nan_outside = True
    assert np.isnan(stage.dev(cfg).labels).any()
    _, b = run_in(stage, "nan", cfg)
    for f in ("ridge.npz", "teacher_labels.npz"):
        with np.load(a / f) as x, np.load(b / f) as y:
            assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
    assert json.loads((a / "loo.json").read_text()) == json.loads((b / "loo.json").read_text())


def test_strict_fp32_is_set_by_the_stage_and_checked_on_the_features(stage):
    """run and predict set strict fp32 (utils.runtime.setup_numerics) as every stage does; features refuses TF32,
    autocast and non-fp32 encoders (guards.check_strict_fp32)."""
    dev = stage.dev(stage.cfg)
    for flags in ((False, True), (True, False)):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        with pytest.raises(guards.GuardError, match="TF32"):
            ridge.features(FakeEncoder(), dev, stage.cfg, None)
    run_in(stage, "tf32")                                             # the stage turns TF32 off itself
    assert not (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(guards.GuardError, match="autocast"):
        run_in(stage, "autocast")
    with pytest.raises(guards.GuardError, match="float32"):
        ridge.features(FakeEncoder().double(), dev, stage.cfg, None)
    with pytest.raises(ridge.RidgeError, match="standardise_on"):
        run_in(stage, "bad", config.merge(stage.cfg, {"ridge": {"standardise_on": "train"}}))
    assert not any((stage.tmp / n / "ridge" / "result.json").exists() for n in ("autocast", "bad"))



class _Broken(FakeEncoder):
    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, x, block=-1):
        t = super().forward(x, block)
        return t.double() if self.how == "float64" else t * float("nan")


def test_features_refuse_a_wrong_dtype_or_non_finite_tokens(stage):
    dev = stage.dev(stage.cfg)
    with pytest.raises(ridge.RidgeError, match="expected"):
        ridge.features(_Broken("float64"), dev, stage.cfg, None)
    with pytest.raises(ridge.RidgeError, match="non-finite encoder features"):
        ridge.features(_Broken("nan"), dev, stage.cfg, None)


def test_predict_refuses_an_edited_config(stage):
    _, out = run_in(stage, "edited")
    (out.parent / "config.json").write_text(json.dumps(config.merge(stage.cfg, {"ridge": {"block": 0}})))
    with pytest.raises(ridge.RidgeError, match="is not the config the ridge stage"):
        ridge.predict(out, stage.dev(stage.cfg))
