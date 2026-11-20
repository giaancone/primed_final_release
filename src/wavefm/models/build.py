"""Build the encoder, a model, and the optimiser parameter groups of the finetune stage from the config.

State-dict convention between stages. A stage that passes weights on (ssl, finetune) saves one flat state dict in
``<stage>/weights.pt``: the encoder's keys prefixed ``encoder.`` (``encoder.tokenizer.*``, ``encoder.layers.<i>.*``,
``encoder.adapter.*``) and anything stage-specific under any other prefix (``head.*``, ``ssl_next.*``, ``mask_token``).
``Teacher.state_dict()`` already has this form. The next stage loads it with ``build_encoder(.., init_from=path)``:
only the ``encoder.`` keys, strictly; the stage-specific ones are never passed on.

Seeds. Everything this module initialises (the random control, the adapter, the CNN) is built on the CPU and drawn
from the CPU generator seeded with ``seed`` inside a fork of it: two builds with one seed are identical, and the
caller's random state is not touched (CPU, and CUDA, which is never seeded here). The task head is built by the caller
and follows the caller's generator.
"""
import torch
import torch.nn as nn

from wavefm.models.backbones import TimesFMEncoder
from wavefm.models.cnn import StreamingCNN

PREFIX = "encoder."


class InitFromError(RuntimeError):
    pass


def _seeded(seed, make):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))      # the CPU generator only: torch.manual_seed seeds CUDA too
        return make()


def encoder_state(path) -> dict:
    """The ``encoder.`` keys of a stage's weights.pt, without the prefix."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise InitFromError(f"{path} is not a state dict: a stage saves weights.pt as one flat state dict")
    enc = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    if not enc:
        raise InitFromError(f"{path} has no '{PREFIX}' keys: it holds no encoder (keys {sorted(sd)[:5]} ..)")
    return enc


def load_encoder_strict(encoder: nn.Module, path):
    """Load the encoder keys of ``path`` into ``encoder``; any missing key, unexpected encoder key or shape mismatch
    raises InitFromError naming the keys (the model config of this stage does not build what the file holds)."""
    sd, own = encoder_state(path), encoder.state_dict()
    missing, unexpected = sorted(own.keys() - sd.keys()), sorted(sd.keys() - own.keys())
    shapes = sorted(f"{k} {tuple(sd[k].shape)} != {tuple(own[k].shape)}" for k in own.keys() & sd.keys()
                    if sd[k].shape != own[k].shape)
    if missing or unexpected or shapes:
        cut = lambda ks: ks[:8] + ([f".. {len(ks) - 8} more"] if len(ks) > 8 else [])
        raise InitFromError(f"{path} does not fit the encoder of model config: missing {cut(missing)}, unexpected "
                            f"{cut(unexpected)}, shape {cut(shapes)}. Start from the stage written with this model "
                            f"config (same version, blocks, adapter)")
    encoder.load_state_dict(sd, strict=True)


def build_encoder(model_cfg: dict, init_from=None, seed: int = 0) -> TimesFMEncoder:
    """The encoder of ``model`` (kind timesfm). ``init_from``: an earlier stage's weights.pt; then the pretrained file
    is not read and every encoder weight comes from ``init_from``, strictly. Everything is trainable except the
    transformer of the random control; a stage narrows this with ``set_trainable``."""
    m = model_cfg
    if m["kind"] != "timesfm":
        raise ValueError(f"model.kind {m['kind']!r} has no encoder")
    enc = _seeded(seed, lambda: TimesFMEncoder(m["version"], scaling=m["scaling"], blocks=m["blocks"],
                                               adapter=m["adapter"], load_weights=init_from is None))
    if init_from is not None:
        load_encoder_strict(enc, init_from)
    return enc


class Teacher(nn.Module):
    """Encoder + task head; the head gets the tokens of the last block and the raw trace mean and spread."""

    def __init__(self, encoder: TimesFMEncoder, head: nn.Module):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x):
        return self.head(self.encoder(x), self.encoder.trace_stats(x))


def build_model(cfg: dict, head: nn.Module, n_out: int, init_from=None) -> nn.Module:
    """The finetune model: TimesFM encoder + the task's ``head``, trained as ``cfg["finetune"]`` says
    (``train_layers``, ``train_tokenizer``), or the CNN baseline with ``n_out`` outputs (DRO only: the CNN has no
    per-sample head for DCH, on purpose)."""
    m = cfg["model"]
    if m["kind"] == "cnn":
        if cfg["task"] != "dro":
            raise ValueError(f"model.kind cnn is implemented for task dro only, not {cfg['task']!r} (on purpose)")
        if init_from is not None:
            raise ValueError("model.kind cnn has no encoder to start from: it is trained from scratch")
        return _seeded(cfg["seed"], lambda: StreamingCNN(n_out, widths=tuple(m["widths"])))
    enc = build_encoder(m, init_from, seed=cfg["seed"])
    f = cfg["finetune"]
    enc.set_trainable(f["train_layers"], f["train_tokenizer"])
    return Teacher(enc, head)


def unused_parameters(model: nn.Module) -> tuple:
    """Name fragments of parameters without gradient by construction: in TimesFM 3.0 with one channel the channel
    attention uses only its value and output projections (backbones._mixing_layer_one_channel)."""
    if isinstance(model, Teacher) and model.encoder.version == "3.0":
        return ("var_attn.query_proj", "var_attn.key_proj", "var_attn.query_ln", "var_attn.key_ln", "var_attn.per_dim_scale")
    return ()


def param_groups(model: nn.Module, f: dict) -> list:
    """AdamW groups of the finetune stage from ``f = cfg["finetune"]``: the tokenizer at ``lr_tokenizer``; block i of
    n at ``lr_backbone * layer_decay ** (n - 1 - i)`` (the last block at lr_backbone); the adapter at ``lr_adapter``;
    the head at ``lr_head`` (also the single rate of the CNN). Frozen parameters are left out. ``weight_decay`` on
    matrices only: none on biases and norm scales. Every group has a ``name``."""
    if isinstance(model, Teacher):
        enc, n = model.encoder, len(model.encoder.layers)
        groups = [("tokenizer", enc.tokenizer, f["lr_tokenizer"])]
        groups += [(f"layer{i}", l, f["lr_backbone"] * f["layer_decay"] ** (n - 1 - i)) for i, l in enumerate(enc.layers)]
        groups += [("adapter", enc.adapter, f["lr_adapter"])] if enc.adapter is not None else []
        groups += [("head", model.head, f["lr_head"])]
    else:
        groups = [("all", model, f["lr_head"])]
    out = []
    for name, mod, lr in groups:
        ps = [p for p in mod.parameters() if p.requires_grad]
        for decay, sel in ((True, [p for p in ps if p.ndim > 1]), (False, [p for p in ps if p.ndim <= 1])):
            if sel:
                out.append({"params": sel, "lr": float(lr), "name": name, "weight_decay": f["weight_decay"] if decay else 0.0})
    return out
