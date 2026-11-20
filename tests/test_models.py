"""Model tests. CPU tests build TimesFM 2.5 without its weights (random, or a stand-in "pretrained" file); the ``gpu``
ones load the real TimesFM weights from $HF_HOME and compare with the library's own forward. They fail, never skip,
without a GPU."""
import copy

import pytest
import torch

from wavefm.models import backbones, build
from wavefm.models.adapters import ConvAdapter
from wavefm.models.backbones import TimesFMEncoder, causal_patch_stats, causal_scale
from wavefm.models.heads import MeanPoolHead

MODEL = {"kind": "timesfm", "version": "2.5", "blocks": "pretrained", "scaling": "native", "adapter": False,
         "widths": [32, 64, 128], "head_hidden": 128}
FINETUNE = {"train_layers": 2, "train_tokenizer": False, "lr_backbone": 1e-4, "layer_decay": 0.5, "lr_tokenizer": 3e-5,
            "lr_adapter": 2e-4, "lr_head": 1e-3, "weight_decay": 0.01}


def model(**kw):
    return {**MODEL, **kw}


@pytest.fixture(scope="module")
def stand_in_weights():
    """A state dict of the right keys and shapes (a fresh 2.5 module), in place of the Hugging Face file."""
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    torch.manual_seed(123)
    return TimesFM_2p5_200M_torch_module().state_dict()


@pytest.fixture
def fake_hf(monkeypatch, stand_in_weights):
    monkeypatch.setattr(backbones, "hf_weights", lambda version: stand_in_weights)
    return stand_in_weights


@pytest.fixture
def no_hf(monkeypatch):
    """Any attempt to read pretrained weights fails."""
    def boom(*a, **k):
        raise AssertionError("pretrained weights were read")
    for name in ("hf_weights", "load_file", "hf_hub_download"):
        monkeypatch.setattr(backbones, name, boom)


@pytest.fixture(scope="module")
def random_enc():
    return build.build_encoder(model(blocks="random", adapter=True), seed=7).eval()


def rms_norms(mod):
    from timesfm.torch.normalization import RMSNorm
    return [m for m in mod.modules() if isinstance(m, RMSNorm)]


def run_blocks(enc, p, k):
    """Hand-written: tokenizer on (patches, zero flags), adapter, then blocks 0..k-1."""
    tok = enc.tokenizer(torch.cat([p, torch.zeros_like(p)], -1))
    tok = enc.adapter(tok) if enc.adapter is not None else tok
    mask = torch.zeros(p.shape[:2], dtype=torch.bool)
    for layer in enc.layers[:k]:
        tok = layer(tok, mask, None)[0]
    return tok


# ---- scaling --------------------------------------------------------------------------------------------------------

def test_causal_stats_match_timesfm_25():
    from timesfm.torch import util as u25
    p = torch.randn(3, 196, 32, dtype=torch.float64) * 50 + 7
    n = mu = sg = torch.zeros(3, dtype=torch.float64)
    ref_mu, ref_sg = [], []
    for i in range(p.shape[1]):
        (n, mu, sg), _ = u25.update_running_stats(n, mu, sg, p[:, i], torch.zeros(3, 32, dtype=torch.bool))
        ref_mu.append(mu); ref_sg.append(sg)
    m, s = causal_patch_stats(p)
    assert torch.allclose(m, torch.stack(ref_mu, 1), rtol=1e-9) and torch.allclose(s, torch.stack(ref_sg, 1), rtol=1e-7)


def test_causal_stats_match_timesfm_3():
    from timesfm3.torch import util as u3
    p = torch.randn(2, 1, 196, 32) * 50 + 7
    _, rmu, rsg = u3.get_running_stats(p, torch.zeros_like(p, dtype=torch.bool))
    m, s = causal_patch_stats(p)
    assert torch.allclose(m, rmu, rtol=1e-4, atol=1e-3) and torch.allclose(s, rsg, rtol=1e-4, atol=1e-3)


def test_causal_scale_is_amplitude_free():
    p = torch.randn(2, 10, 32).abs() + 1
    assert torch.allclose(causal_scale(p), causal_scale(30 * p), atol=1e-5)


def test_scalings_equal_hand_written_formulas(random_enc):
    torch.manual_seed(0)
    x = torch.randn(3, 96, dtype=torch.float64) * 40 + 5
    x[2] = 4.0                                                        # a flat trace: spread 0 -> divided by 1
    enc = copy.copy(random_enc)
    want = {}
    for scaling in ("global_minmax", "native", "native_stats", "trace_native"):
        z = x.clone()
        if scaling == "trace_native":                                 # unbiased spread over the whole trace
            mu = z.mean(1, keepdim=True)
            sd = ((z - mu) ** 2).sum(1, keepdim=True).div(z.shape[1] - 1).sqrt()
            z = (z - mu) / torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        p = z.reshape(3, 3, 32)
        if scaling != "global_minmax":                                # patch i: population stats of patches 0..i
            out = torch.empty_like(p)
            for i in range(3):
                seen = p[:, :i + 1].reshape(3, -1)
                m, s = seen.mean(1), seen.std(1, unbiased=False)
                out[:, i] = (p[:, i] - m[:, None]) / torch.where(s < 1e-6, torch.ones_like(s), s)[:, None]
            p = out
        enc.scaling = scaling
        want[scaling] = p
        assert torch.allclose(enc.patches(x), p, atol=1e-12), scaling
    assert torch.equal(want["global_minmax"], x.reshape(3, 3, 32))


def test_trace_native_matches_the_dch_paper_code(random_enc):
    """dch_ssl.Encoder.normalize (legacy_double_revin): timesfm revin with torch.std, then Welford running stats."""
    from timesfm.torch.util import revin, update_running_stats
    torch.manual_seed(1)
    x = torch.randn(4, 3008) * 3 + torch.linspace(0, 20, 3008)
    xn = revin(x, x.mean(-1, keepdim=True), x.std(-1, keepdim=True), reverse=False)
    pi = xn.reshape(4, -1, 32)
    n = mu = sg = torch.zeros(4)
    cm, cs = [], []
    for i in range(pi.shape[1]):
        (n, mu, sg), _ = update_running_stats(n, mu, sg, pi[:, i], torch.zeros(4, 32, dtype=torch.bool))
        cm.append(mu); cs.append(sg)
    ref = revin(pi, torch.stack(cm, 1), torch.stack(cs, 1), reverse=False)
    enc = copy.copy(random_enc); enc.scaling = "trace_native"
    assert torch.allclose(enc.patches(x), ref, atol=2e-5)


def test_patches_are_what_the_tokenizer_receives(random_enc):
    seen = []
    h = random_enc.tokenizer.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    try:
        x = torch.randn(2, 128) * 9 + 3
        with torch.no_grad():
            random_enc(x)
    finally:
        h.remove()
    assert len(seen) == 1 and seen[0].shape == (2, 4, 64)
    assert torch.equal(seen[0][..., :32], random_enc.patches(x)) and not seen[0][..., 32:].any()   # flags all 0


# ---- forward, blocks ------------------------------------------------------------------------------------------------

def test_forward_patches_and_block_index(random_enc):
    torch.manual_seed(2)
    x = torch.randn(2, 160) * 5
    p = random_enc.patches(x)
    with torch.no_grad():
        assert torch.equal(random_enc.forward_patches(p), random_enc(x))
        for block, k in ((0, 1), (3, 4), (-1, 20), (-20, 1), (19, 20)):
            assert torch.allclose(random_enc(x, block=block), run_blocks(random_enc, p, k), atol=1e-5), block
    for bad in (20, -21):
        with pytest.raises(ValueError, match="20 blocks"):
            random_enc(x, block=bad)
    with pytest.raises(ValueError, match="multiple of 32"):
        random_enc(torch.randn(2, 100))


def test_adapter_starts_as_identity():
    h = torch.randn(2, 196, 1280)
    assert torch.equal(ConvAdapter(1280)(h), h)
    assert sum(p.numel() for p in ConvAdapter(1280).parameters()) == 6_556_160


def test_head_shapes():
    tok, stats = torch.randn(4, 196, 1280), torch.rand(4, 2) + 0.1
    assert MeanPoolHead(1280, 3)(tok).shape == (4, 3)
    assert MeanPoolHead(1280, 3, n_extra=2)(tok, stats).shape == (4, 3)


# ---- controls -------------------------------------------------------------------------------------------------------

def test_timesfm_rmsnorm_starts_at_zero_and_makes_blocks_dead():
    """The claim behind the random control, on the real module: every RMSNorm scale is 0 at construction, and then a
    block returns its input bit for bit."""
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    net = TimesFM_2p5_200M_torch_module()
    norms = rms_norms(net.stacked_xf)
    assert len(norms) == 6 * 20 and all(not n.scale.detach().any() for n in norms)
    h = torch.randn(2, 5, 1280)
    with torch.no_grad():
        assert torch.equal(net.stacked_xf[0](h, torch.zeros(2, 5, dtype=torch.bool), None)[0], h)


def test_random_blocks(no_hf):
    a = build.build_encoder(model(blocks="random", adapter=True), seed=3)
    b = build.build_encoder(model(blocks="random", adapter=True), seed=3)
    c = build.build_encoder(model(blocks="random", adapter=True), seed=4)
    norms = rms_norms(a.layers)
    assert len(norms) == 120 and all(torch.equal(n.scale, torch.ones_like(n.scale)) for n in norms)
    assert all(not p.requires_grad for p in a.layers.parameters())
    assert all(p.requires_grad for p in list(a.tokenizer.parameters()) + list(a.adapter.parameters()))
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not torch.equal(sa["tokenizer.hidden_layer.weight"], sc["tokenizer.hidden_layer.weight"])
    assert not torch.equal(sa["layers.5.ff0.weight"], sc["layers.5.ff0.weight"])
    assert not torch.equal(sa["adapter.conv1.weight"], sc["adapter.conv1.weight"])
    torch.manual_seed(5)
    x = torch.randn(3, 128) * 4
    with torch.no_grad():
        out, front = a(x), run_blocks(a, a.patches(x), 0)
    assert out[0].std() > 0 and (out[0] - out[1]).abs().max() > 1e-3             # depends on the input
    assert (out - front).abs().max() > 1e-2                                       # the blocks change the tokens: alive
    with pytest.raises(ValueError, match="train_layers 0"):
        a.set_trainable(2, True)


def test_random_control_does_not_touch_the_callers_generator(no_hf):
    torch.manual_seed(9)
    want = torch.rand(3)
    torch.manual_seed(9)
    build.build_encoder(model(blocks="random"), seed=1)
    assert torch.equal(torch.rand(3), want)


def test_no_blocks(fake_hf, monkeypatch):
    from timesfm.torch import transformer

    def boom(*a, **k):
        raise AssertionError("the transformer was called")
    monkeypatch.setattr(transformer.Transformer, "forward", boom)
    enc = build.build_encoder(model(blocks="none", adapter=True), seed=0)
    assert len(enc.layers) == 0 and not any(k.startswith("layers.") for k in enc.state_dict())
    tok_sd = {k[len("tokenizer."):]: v for k, v in fake_hf.items() if k.startswith("tokenizer.")}
    assert all(torch.equal(enc.tokenizer.state_dict()[k], v) for k, v in tok_sd.items())
    with torch.no_grad():
        enc.adapter.conv2.weight.normal_(0, 1e-2)                   # a non-trivial adapter
    x = torch.randn(2, 96) * 3
    with torch.no_grad():
        assert torch.equal(enc(x), run_blocks(enc, enc.patches(x), 0))
    with pytest.raises(ValueError, match="no transformer"):
        enc(x, block=3)


def test_controls_are_for_25_only():
    for blocks in ("random", "none"):
        with pytest.raises(ValueError, match="2.5 only"):
            TimesFMEncoder("3.0", blocks=blocks)


# ---- init_from ------------------------------------------------------------------------------------------------------

def test_init_from_loads_only_the_encoder_strictly(tmp_path, no_hf):
    src = build.build_encoder(model(blocks="random", adapter=True), seed=1)
    sd = {f"encoder.{k}": v for k, v in src.state_dict().items()}
    head = {"head.weight": torch.randn(3, 1280), "ssl_next.weight": torch.randn(32, 1280), "mask_token": torch.ones(32)}
    torch.save({**sd, **head}, tmp_path / "weights.pt")
    enc = build.build_encoder(model(blocks="random", adapter=True), init_from=tmp_path / "weights.pt", seed=99)
    got = enc.state_dict()
    assert got.keys() == src.state_dict().keys() and all(torch.equal(got[k], v) for k, v in src.state_dict().items())
    assert not any(k.startswith(("head", "ssl", "mask")) for k in got)

    cases = {"missing": {k: v for k, v in sd.items() if k != "encoder.adapter.conv1.bias"},
             "unexpected": {**sd, "encoder.extra.weight": torch.zeros(2)},
             "shape": {**sd, "encoder.adapter.conv2.bias": torch.zeros(7)},
             "no encoder": head}
    for what, bad in cases.items():
        torch.save(bad, tmp_path / f"{what}.pt")
        with pytest.raises(build.InitFromError) as e:
            build.build_encoder(model(blocks="random", adapter=True), init_from=tmp_path / f"{what}.pt")
        assert {"missing": "adapter.conv1.bias", "unexpected": "extra.weight", "shape": "conv2.bias (7,)",
                "no encoder": "no 'encoder.' keys"}[what] in str(e.value)
    with pytest.raises(build.InitFromError, match="unexpected"):    # an adapter file into an encoder without one
        build.build_encoder(model(blocks="random", adapter=False), init_from=tmp_path / "weights.pt")


def test_timesfm_3_with_init_from_reads_no_pretrained_weights(tmp_path, no_hf, monkeypatch):
    """TimesFM 3.0 built from a stage's weights.pt reads only the model's config.json (from $HF_HOME), never the
    pretrained weights; CPU only."""
    from huggingface_hub import hf_hub_download
    from timesfm3.torch.model import TimesFM3Torch

    def config_only(repo, name):
        assert name == "config.json", f"{name} was read"
        return hf_hub_download(repo, name)
    monkeypatch.setattr(backbones, "hf_hub_download", config_only)
    monkeypatch.setattr(TimesFM3Torch, "from_pretrained", classmethod(lambda *a, **k: pytest.fail("from_pretrained")))
    torch.manual_seed(4)
    src = TimesFMEncoder("3.0", load_weights=False)
    torch.save({f"encoder.{k}": v for k, v in src.state_dict().items()}, tmp_path / "weights.pt")
    enc = build.build_encoder(model(version="3.0"), init_from=tmp_path / "weights.pt")
    assert all(torch.equal(v, src.state_dict()[k]) for k, v in enc.state_dict().items())


@pytest.mark.gpu
def test_building_leaves_the_callers_cuda_generator_untouched(no_hf):
    """build_encoder and ssl.build draw from a forked CPU generator; the caller's CUDA generator is not reseeded."""
    from wavefm import paths
    from wavefm.train import ssl
    from wavefm.utils import config
    assert torch.cuda.is_available(), "this test needs a GPU"
    torch.cuda.manual_seed(5)
    before = torch.cuda.get_rng_state()
    build.build_encoder(model(blocks="random", adapter=True), seed=1)
    cfg = config.merge(config.load(paths.ROOT / "configs" / "teacher.yaml"), {"model": {"blocks": "random"}})
    ssl.build(cfg)
    assert torch.equal(torch.cuda.get_rng_state(), before)


# ---- build_model, param_groups --------------------------------------------------------------------------------------

def cfg_for(task="dro", **m):
    return {"task": task, "seed": 0, "model": model(**m), "finetune": dict(FINETUNE)}


def test_cnn_is_dro_only():
    cnn = build.build_model(cfg_for(kind="cnn"), None, n_out=3)
    assert cnn(torch.randn(2, 640)).shape == (2, 3)
    assert [g["name"] for g in build.param_groups(cnn, FINETUNE)] == ["all", "all"]
    assert all(g["lr"] == FINETUNE["lr_head"] for g in build.param_groups(cnn, FINETUNE))
    with pytest.raises(ValueError, match="task dro only"):
        build.build_model(cfg_for("dch", kind="cnn"), None, n_out=1)


def test_two_layer_param_groups(fake_hf):
    cfg = cfg_for(adapter=True)
    m = build.build_model(cfg, MeanPoolHead(1280, 3), n_out=3)
    groups = build.param_groups(m, FINETUNE)
    lr = {g["name"]: g["lr"] for g in groups}
    assert lr == {"layer18": 5e-5, "layer19": 1e-4, "adapter": 2e-4, "head": 1e-3}
    assert all(g["weight_decay"] == (0.01 if g["params"][0].ndim > 1 else 0.0) for g in groups)
    n_trained = sum(p.numel() for g in groups for p in g["params"])
    assert n_trained == sum(p.numel() for p in m.parameters() if p.requires_grad)
    cfg["finetune"].update(train_tokenizer=True, train_layers=-1)
    groups = build.param_groups(build.build_model(cfg, MeanPoolHead(1280, 3), n_out=3), cfg["finetune"])
    lr = {g["name"]: g["lr"] for g in groups}
    assert lr["tokenizer"] == 3e-5 and lr["layer0"] == pytest.approx(1e-4 * 0.5 ** 19) and len(lr) == 23


def test_teacher_head_gets_trace_stats(fake_hf):
    m = build.build_model(cfg_for(scaling="native_stats"), MeanPoolHead(1280, 3, n_extra=2), n_out=3)
    x = torch.randn(2, 64) * 7 + 3
    seen = []
    m.head.register_forward_pre_hook(lambda mod, args: seen.append(args[1]))
    with torch.no_grad():
        m(x)
    assert torch.allclose(seen[0], torch.stack([x.mean(1), x.std(1, unbiased=False)], 1), atol=1e-5)


# ---- GPU: real weights, library equivalence -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wave():
    torch.manual_seed(0)
    t = torch.arange(6272, dtype=torch.float32)
    pulse = torch.where(t > 900, torch.exp(-(t - 900) / 300), torch.zeros_like(t))
    return (torch.stack([500 * pulse, 80 * pulse]) + torch.randn(2, 6272)).cuda()


@pytest.fixture(autouse=True)
def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def library_25():
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    ref = TimesFM_2p5_200M_torch_module()
    ref.load_state_dict(backbones.hf_weights("2.5"))
    return ref.cuda().eval()


@pytest.mark.gpu
def test_encoder_25_matches_library(wave):
    ref = library_25()
    enc = TimesFMEncoder("2.5", scaling="native").cuda().eval()
    with torch.no_grad():
        normed = causal_scale(wave.reshape(2, 196, 32))
        (_, out_ref, _, _), _ = ref(normed, torch.zeros_like(normed, dtype=torch.bool))
        out = enc(wave)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_trace_native_matches_dch_paper_forward():
    """The DCH paper's Encoder: legacy_double_revin (library revin + Welford running stats), then the module's
    output embeddings ([0][1]), on a 3008-sample trace."""
    from timesfm.torch.util import revin, update_running_stats
    torch.manual_seed(3)
    x = (torch.randn(2, 3008) * 0.3 + torch.exp(-((torch.arange(3008.) - 1500) / 40) ** 2) * 50).cuda()
    xn = revin(x, x.mean(-1, keepdim=True), x.std(-1, keepdim=True), reverse=False)
    pi = xn.reshape(2, -1, 32)
    n = mu = sg = torch.zeros(2, device="cuda")
    cm, cs = [], []
    for i in range(pi.shape[1]):
        (n, mu, sg), _ = update_running_stats(n, mu, sg, pi[:, i], torch.zeros(2, 32, dtype=torch.bool, device="cuda"))
        cm.append(mu); cs.append(sg)
    normed = revin(pi, torch.stack(cm, 1), torch.stack(cs, 1), reverse=False)
    ref = library_25()
    enc = TimesFMEncoder("2.5", scaling="trace_native").cuda().eval()
    with torch.no_grad():
        out_ref = ref(normed, torch.zeros_like(normed, dtype=torch.bool), None)[0][1]
        out = enc(x)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4), (out - out_ref).abs().max()


@pytest.mark.gpu
def test_encoder_3_matches_library(wave):
    from timesfm3.torch.model import TimesFM3Torch
    ref = TimesFM3Torch.from_pretrained("google/timesfm-3.0-pytorch").cuda().eval()
    enc = TimesFMEncoder("3.0", scaling="native").cuda().eval()
    v = wave.reshape(2, 1, 196, 32)
    with torch.no_grad():
        o = ref({"values": v, "masks": torch.zeros_like(v, dtype=torch.bool),
                 "patch_is_target": torch.ones(2, 1, 196, dtype=torch.bool, device=v.device)}, return_aux_outputs=True)
        out = enc(wave)
    assert torch.allclose(out, o["__call__:transformer_output"][:, 0], atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_no_blocks_loads_the_pretrained_tokenizer(wave):
    full = TimesFMEncoder("2.5", scaling="native").cuda().eval()
    none = build.build_encoder(model(blocks="none")).cuda().eval()
    with torch.no_grad():
        assert torch.equal(none(wave), full._tokens(full.patches(wave)))


@pytest.mark.gpu
@pytest.mark.parametrize("version", ["2.5", "3.0"])
def test_encoder_amplitude_and_gradients(version, wave):
    native = TimesFMEncoder(version, scaling="native").cuda()
    with torch.no_grad():   # TimesFM's scaling: a 3x brighter trace gives the same tokens
        assert torch.allclose(native(wave), native(3 * wave), atol=1e-3)
    none = TimesFMEncoder(version, scaling="global_minmax", adapter=True).cuda()
    x = wave / wave.abs().max()
    with torch.no_grad():   # without it the tokens see the amplitude
        assert (none(x) - none(3 * x)).abs().max() > 1e-2
    none(x).pow(2).mean().backward()
    for mod in (none.tokenizer, none.layers[0], none.layers[-1], none.adapter):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())


@pytest.mark.gpu
@pytest.mark.parametrize("version", ["2.5", "3.0"])
def test_compiled_model_matches_eager(version, wave):
    m = build.build_model(cfg_for(version=version, scaling="global_minmax", adapter=True), MeanPoolHead(1280, 3),
                          n_out=3).cuda().eval()
    x = (wave / wave.abs().max()).repeat(16, 1)
    with torch.no_grad():
        assert torch.allclose(torch.compile(m)(x), m(x), atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_library_comparison_detects_unloaded_weights(wave):
    """Power of the library tests: an encoder without the pretrained weights must fail the same comparison."""
    ref = library_25()
    enc = TimesFMEncoder("2.5", scaling="native", blocks="random").cuda().eval()
    with torch.no_grad():
        normed = causal_scale(wave.reshape(2, 196, 32))
        (_, out_ref, _, _), _ = ref(normed, torch.zeros_like(normed, dtype=torch.bool))
        assert not torch.allclose(enc(wave), out_ref, atol=1e-2)


@pytest.mark.gpu
def test_frozen_prefix_is_the_same_function(wave):
    """Training only the last 2 layers changes what gets gradients, not what the encoder computes."""
    full = TimesFMEncoder("2.5", scaling="global_minmax").cuda()
    part = TimesFMEncoder("2.5", scaling="global_minmax").set_trainable(2, False).cuda()
    x = wave / wave.abs().max()
    a, b = full(x), part(x)
    assert torch.allclose(a, b, atol=1e-5)
    b.pow(2).mean().backward()
    trained = {n.split(".")[1] if n.startswith("layers") else n.split(".")[0] for n, p in part.named_parameters() if p.grad is not None}
    assert trained == {"18", "19"}
    assert all(not p.requires_grad for p in part.tokenizer.parameters())
    with torch.no_grad():    # block selection runs only the blocks asked for, with or without the frozen prefix
        assert torch.allclose(part(x, block=5), full(x, block=5), atol=1e-5)
