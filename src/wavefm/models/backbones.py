"""TimesFM 2.5 and 3.0 as differentiable waveform encoders with one interface.

    encoder.patches(x)                  (B, L) waveforms, L a multiple of 32 -> (B, L/32, 32) scaled patches: exactly
                                        what the tokenizer receives (with all-zero mask flags)
    encoder.forward_patches(p, block)   scaled patches -> (B, N, 1280) tokens, the output of transformer block
                                        ``block`` (python indexing over the blocks, -1 = the last, the paper's choice)
    encoder(x, block=-1)                forward_patches(patches(x), block)
    encoder.trace_stats(x)              (B, 2) mean and population spread of each raw trace: the amplitude that the
                                        native scalings divide out, for the head of ``native_stats``

Both versions are called below their forecasting wrappers, so gradients reach every weight, the forecast heads are
deleted, and no padding mask is used (traces are padded at the end by repeating the last sample, data.batches).
The tokens are the raw output of the block: no final norm, no output projection (as the old scripts used them).

Input scaling ``scaling`` (a model-level hyper-parameter, see configs/teacher.yaml):
    native, native_stats  TimesFM's causal per-patch scaling: patch i is scaled by the mean and population spread of
                          all samples of patches 0..i (the same definition in 2.5 and 3.0), spread < 1e-6 -> 1
    trace_native          the DCH paper's double scaling: each (padded) trace standardised by its own mean and
                          UNBIASED spread (timesfm.torch.util.revin: spread < 1e-6 -> 1), then the causal scaling
    global_minmax         the input is already scaled by the data layer; the encoder applies nothing

Transformer blocks ``blocks``:
    pretrained  the Hugging Face weights, loaded strictly
    random      the control: nothing pretrained is loaded (the tokenizer is random too), the weights are PyTorch's
                default initialisation drawn from the seed, every RMSNorm scale of the transformer is set to 1, and
                the transformer is frozen. TimesFM initialises RMSNorm scales to 0: every norm then outputs 0, both
                residual branches add 0 and every block is the identity (checked on the 2.5 module: a fresh block
                returns its input bit for bit, tests/test_models.py). The attention's per_dim_scale also starts at 0
                but is left alone: softplus(0) * 1.4427 = 1, the standard 1/sqrt(d) scaling.
    none        the control without a transformer: only the tokenizer is built and loaded, ``forward`` returns the
                tokenizer (+ adapter) output, and ``block`` must be -1.
The controls are built for TimesFM 2.5 (the paper's model) only.

What is trained is set by ``set_trainable``; frozen modules get requires_grad False, and the frozen part in front of
the first trained module runs without autograd (no stored activations, no backward).
"""
import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from wavefm.models.adapters import ConvAdapter

PATCH, DIM = 32, 1280
HF_REPO = {"2.5": "google/timesfm-2.5-200m-pytorch", "3.0": "google/timesfm-3.0-pytorch"}
SCALINGS = ("global_minmax", "native", "native_stats", "trace_native")
BLOCKS = ("pretrained", "random", "none")
_TOL = 1e-6  # TimesFM's revin tolerance


def pretrained_file(version: str) -> str:
    """The path of the pretrained weight file in the Hugging Face cache ($HF_HOME, offline on compute nodes): the file
    hf_weights reads, so a stage can record its sha256."""
    return hf_hub_download(HF_REPO[version], "model.safetensors")


def hf_weights(version: str) -> dict:
    """The pretrained state dict, read from pretrained_file(version)."""
    return load_file(pretrained_file(version))


def causal_patch_stats(patches: torch.Tensor):
    """Mean and population spread of all samples in patches 0..i, for every patch i, accumulated in float64.
    patches (..., N, P) -> (..., N), (..., N)."""
    p = patches.double()
    n = torch.arange(1, p.shape[-2] + 1, device=p.device, dtype=p.dtype) * p.shape[-1]
    mu = p.sum(-1).cumsum(-1) / n
    var = (p * p).sum(-1).cumsum(-1) / n - mu * mu
    return mu.to(patches.dtype), var.clamp_min(0).sqrt().to(patches.dtype)


def causal_scale(patches: torch.Tensor) -> torch.Tensor:
    mu, sigma = causal_patch_stats(patches)
    sigma_safe = torch.where(sigma < _TOL, torch.ones_like(sigma), sigma)
    return (patches - mu[..., None]) / sigma_safe[..., None]


def trace_standardise(x: torch.Tensor) -> torch.Tensor:
    """(B, L) -> each trace minus its mean over its own samples, divided by its unbiased spread (< 1e-6 -> 1), as
    timesfm.torch.util.revin with torch.std in the DCH paper code."""
    mu, sigma = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
    return (x - mu) / torch.where(sigma < _TOL, torch.ones_like(sigma), sigma)


def _mixing_layer_one_channel(layer, h, pmask):
    """One TimesFM 3.0 layer for a single channel, h (B, N, D). Same function as MixingTransformer.forward with
    v = 1: attention over one channel puts weight 1 on that channel, so it reduces exactly to out_proj(value_proj(.)).
    This skips 6272 one-key attentions and the unused query and key projections per batch (3.0 is 4x slower without)."""
    a, _, _ = layer.seq_attn(layer.pre_seq_attn_ln(h), patch_mask=pmask)
    h = layer.post_seq_attn_ln(a) + h
    if layer.use_variate_attention:
        va = layer.var_attn
        h = layer.post_var_attn_ln(va.out_proj(va.value_proj(layer.pre_var_attn_ln(h)))) + h
    return layer.post_ff_ln(layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(h))))) + h


class TimesFMEncoder(nn.Module):
    """``load_weights`` False builds the structure without reading the pretrained file (the caller loads a stage's
    weights over every key, models.build.build_encoder). Construct it under a fixed seed: random parts (the adapter's
    first convolution, the random control) are drawn from the global generator."""

    def __init__(self, version: str = "2.5", scaling: str = "native", blocks: str = "pretrained",
                 adapter: bool = False, load_weights: bool = True):
        super().__init__()
        version = str(version)       # YAML reads 3.0 as a float
        if version not in HF_REPO or scaling not in SCALINGS or blocks not in BLOCKS:
            raise ValueError(f"model.version {version!r} / scaling {scaling!r} / blocks {blocks!r}: known "
                             f"{tuple(HF_REPO)}, {SCALINGS}, {BLOCKS}")
        if version != "2.5" and blocks != "pretrained":
            raise ValueError(f"model.blocks {blocks!r} is built for TimesFM 2.5 only (the paper's model)")
        self.version, self.scaling, self.blocks = version, scaling, blocks
        load = load_weights and blocks != "random"
        if version == "2.5":
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module as TFM
            if blocks == "none":     # the tokenizer alone: the transformer is never built
                from timesfm.torch.dense import ResidualBlock
                self.tokenizer, self.layers = ResidualBlock(TFM.config.tokenizer), nn.ModuleList()
                if load:
                    sd = {k[len("tokenizer."):]: v for k, v in hf_weights(version).items() if k.startswith("tokenizer.")}
                    self.tokenizer.load_state_dict(sd, strict=True)
            else:
                net = TFM()
                if load:
                    net.load_state_dict(hf_weights(version), strict=True)
                del net.output_projection_point, net.output_projection_quantiles
                self.tokenizer, self.layers = net.tokenizer, net.stacked_xf
                if blocks == "random":
                    from timesfm.torch.normalization import RMSNorm
                    for m in self.layers.modules():
                        if isinstance(m, RMSNorm):
                            nn.init.ones_(m.scale)
        else:
            from timesfm3.torch.model import TimesFM3Torch
            if load:
                net = TimesFM3Torch.from_pretrained(HF_REPO[version])     # builds the model from config.json
                # from_pretrained loads with strict=False: reload strictly so a renamed key cannot stay random
                net.load_state_dict(hf_weights(version), strict=True)
            else:                                                         # the structure alone, from config.json
                with open(hf_hub_download(HF_REPO[version], "config.json")) as f:
                    net = TimesFM3Torch(**json.load(f))
            del net.output_head
            self.tokenizer, self.layers = net.pre_transformer_resblock, net.transformer_stack.layers
            if any(l.use_variate_attention and l.var_attn.value_ln is not None for l in self.layers):
                raise NotImplementedError("single-channel shortcut assumes no value norm in the channel attention")
        self.adapter = ConvAdapter(DIM) if adapter else None
        self.set_trainable(-1 if blocks == "pretrained" else 0, True)

    def set_trainable(self, train_layers: int, train_tokenizer: bool):
        """Train the top ``train_layers`` transformer blocks (-1: all) and the tokenizer if ``train_tokenizer``; the
        adapter is always trained. The random control keeps its transformer frozen: asking to train it is an error."""
        n = len(self.layers)
        n_frozen = 0 if train_layers < 0 else n - min(int(train_layers), n)
        if self.blocks == "random" and n_frozen != n:
            raise ValueError("model.blocks random is the frozen random-transformer control: set train_layers 0")
        self.n_frozen, self.train_tokenizer = n_frozen, bool(train_tokenizer)
        self.tokenizer.requires_grad_(self.train_tokenizer)
        for i, layer in enumerate(self.layers):
            layer.requires_grad_(i >= n_frozen)
        # blocks in front of the first trained module run under no_grad
        self.no_grad_layers = n_frozen if (not self.train_tokenizer and self.adapter is None) else 0
        return self

    def trace_stats(self, x: torch.Tensor) -> torch.Tensor:
        d = x.double()
        return torch.stack([d.mean(-1), d.std(-1, unbiased=False)], -1).to(x.dtype)

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L % PATCH:
            raise ValueError(f"waveforms of {L} samples: pad to a multiple of {PATCH} first (data.batches.pad_to_patch)")
        if self.scaling == "trace_native":
            x = trace_standardise(x)
        p = x.reshape(B, L // PATCH, PATCH)
        return p if self.scaling == "global_minmax" else causal_scale(p)

    def _n_run(self, block: int) -> int:
        n = len(self.layers)
        if n == 0:
            if block != -1:
                raise ValueError(f"block {block}: model.blocks none has no transformer, only block -1 (the tokens)")
            return 0
        if not -n <= block < n:
            raise ValueError(f"block {block}: the encoder has {n} blocks (python indexing, -1 = the last)")
        return block % n + 1

    def _tokens(self, patches):
        if self.version == "2.5":
            return self.tokenizer(torch.cat([patches, torch.zeros_like(patches)], -1))   # 32 values + 32 mask flags
        # 3.0: 32 values + 64-sample future slot (empty for a target series) + 32 + 64 mask flags, one variate
        v = patches[:, None]
        fut = torch.zeros(*v.shape[:-1], 2 * PATCH, device=v.device, dtype=v.dtype)
        return self.tokenizer(torch.cat([v, fut, torch.zeros_like(v), torch.ones_like(fut)], -1))[:, 0]

    def _layer(self, layer, tok, pmask):
        if self.version == "2.5":
            return layer(tok, pmask, None)[0]
        return _mixing_layer_one_channel(layer, tok, pmask)

    def forward_patches(self, patches: torch.Tensor, block: int = -1) -> torch.Tensor:
        """Tokens from patches that are already scaled, (B, N, 32) -> (B, N, 1280). Skips only the input scaling of
        ``patches(x)``: nothing is re-normalised, so patches edited after scaling (a mask token in place of some
        patches, SSL) reach the tokenizer as given, with all mask flags 0 and positions 0..N-1."""
        k = self._n_run(block)
        pmask = torch.zeros(patches.shape[:2], dtype=torch.bool, device=patches.device)
        frozen_front = not self.train_tokenizer and self.adapter is None and not patches.requires_grad
        n_ng = min(self.no_grad_layers, k) if frozen_front else 0
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen_front):
            tok = self._tokens(patches)
            for layer in self.layers[:n_ng]:
                tok = self._layer(layer, tok, pmask)
        if self.adapter is not None:
            tok = self.adapter(tok)
        for layer in self.layers[n_ng:k]:
            tok = self._layer(layer, tok, pmask)
        return tok

    def forward(self, x: torch.Tensor, block: int = -1) -> torch.Tensor:
        return self.forward_patches(self.patches(x), block)
