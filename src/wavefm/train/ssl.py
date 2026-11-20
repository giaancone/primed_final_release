"""The label-free stage of the teacher: ``run(cfg, out, parent)`` trains the tokenizer and the adapter of the frozen
TimesFM on every dev waveform, with no label, so that ridge (the next stage) reads better tokens.

As run by the paper scripts (work_dro/ssl_adapt2.py for DRO, dch_norm_aux_v2/src/dch_ssl.py for DCH), every
difference between the two a config item:
  trained    the tokenizer (``ssl.lr_tokenizer``), the adapter if ``model.adapter`` (``ssl.lr_adapter``), two temporary
             heads ``Linear(1280, 32)`` (next-patch, masked-patch) and one mask token ``zeros(32)`` (``ssl.lr_heads``);
             weight decay ``ssl.weight_decay`` on every one of them, biases and the mask token included (both old
             scripts gave one AdamW weight_decay=0.01 to all groups). The transformer is frozen: pretrained, the
             random control at its seeded initialisation, or absent (``blocks: none``).
  rows       all dev waveforms, read once to the device and padded (data.batches); ``dev.labels`` is never read.
  objective  for raw padded waveforms x (B, L), N = L / 32 patches, a mask m (B, N) drawn from the trainer's generator
             (each patch masked independently with probability ``mask_prob``), token t:
                 next   = mean_{b, i < N-1, k} (head_next(tok(P)[:, :-1]) - T[:, 1:])^2          (clean input)
                 masked = sum_{b, i} m_bi mean_k (head_masked(tok(P_m)) - T)^2_bik / max(sum m, 1)
                 loss   = next_weight * next + masked_weight * masked
             with tok = enc.forward_patches (block -1), P = enc.patches(s(x)) the clean scaled patches, s the
             model-level input scaling (global_minmax: (x - lo) / (hi - lo) with the dev minimum and maximum; else
             none, the encoder scales inside), and the target space ``ssl.target_space``:
                 input   T = P;                       P_m = where(m, t, T)                       (DCH as run)
                 zscore  T = patches((x - mu) / sd);  P_m = enc.patches(s(where(m, t, T) * sd + mu))   (DRO as run)
             mu, sd: mean and spread of the dev waveforms (dev meta.json). In ``input`` the masked patches go straight
             to the tokenizer (nothing is re-normalised: with a causal scaling the running statistics are those of the
             clean trace); in ``zscore`` the masked trace goes back to raw units and through the encoder's normal
             input path. Supported: ``input`` with every scaling, ``zscore`` with ``global_minmax`` only (each
             backward-tested at the zero mask token). ``zscore`` with a per-trace or per-patch scaling (native,
             native_stats, trace_native) is refused: the zero token makes a masked first patch constant, its spread
             0, and the square root of the renormalisation gives non-finite gradients.
  output     weights.pt: one flat state dict, the encoder under ``encoder.`` (what ridge and finetune load, strictly),
             the heads under ``ssl_next.`` / ``ssl_masked.`` and ``mask_token`` (never passed on); history.json
             (training curves only: ``train_loss``, ``train_next``, ``train_masked``, lr, grad norm, clip fraction);
             result.json with the final values of both terms.

Known and kept on purpose (the paper ran it so):
  * the symmetric conv3 adapter lets token i see patch i+1, so the next-patch target leaks into its input;
  * padded samples (the repeated last sample) are part of the targets of both terms.

cudnn runs deterministic algorithms (utils.runtime.setup_numerics; here it matters for the adapter's convolutions):
without it two identical GPU runs differ in the last bits from the second step on, and a resumed run would not equal
the uninterrupted one.

ssl has no validation set (plan, section 2): ``keep: best_val`` and ``schedule: plateau`` are refused. It has no
``predict`` either: scripts/evaluate.py evaluates the last finished stage that has one. The modules run in train()
mode (the trainer's); that equals eval() because the encoder has no dropout or batch norm, which is checked.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import wavefm.data as wdata
from wavefm.data.batches import WaveformSet
from wavefm.models.backbones import DIM, PATCH
from wavefm.models.build import build_encoder
from wavefm.train import guards, pipeline, rundir, trainer
from wavefm.utils import runtime
from wavefm.utils.io import load_json

SPACES = ("input", "zscore")
MODE_DEPENDENT = (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._NormBase)


class SslError(ValueError):
    pass


def check_config(cfg: dict):
    m, s = cfg["model"], cfg["ssl"]
    if m["kind"] != "timesfm":
        raise SslError(f"model.kind {m['kind']!r}: ssl trains the tokenizer of the TimesFM encoder, use kind timesfm")
    if s["target_space"] not in SPACES:
        raise SslError(f"ssl.target_space {s['target_space']!r}: known {SPACES}")
    if s["target_space"] == "zscore" and m["scaling"] != "global_minmax":
        raise SslError(f"ssl.target_space zscore with model.scaling {m['scaling']!r}: the masked trace would be "
                       f"renormalised per trace or patch, which has non-finite gradients at the zero mask token; use "
                       f"zscore with global_minmax, or target_space input")
    if not 0 < s["mask_prob"] < 1:
        raise SslError(f"ssl.mask_prob {s['mask_prob']} must be in (0, 1)")
    if s["keep"] != "last" or s["schedule"] == "plateau":
        raise SslError(f"ssl.keep {s['keep']!r} / ssl.schedule {s['schedule']!r}: ssl has no validation set, so "
                       f"best_val and plateau are undefined; use keep: last with constant or cosine")


def check_modes(model: nn.Module):
    """train() and eval() must be the same function: no dropout or batch norm anywhere in the trained model."""
    bad = [n for n, mod in model.named_modules() if isinstance(mod, MODE_DEPENDENT)]
    if bad:
        raise SslError(f"{bad[:4]} behave differently in train() and eval(): ssl assumes the encoder has none")


def constants(cfg: dict, stats: dict) -> dict:
    """The model-level numbers of the objective, from the dev waveform statistics (meta.json)."""
    c = {"scaling": cfg["model"]["scaling"], "space": cfg["ssl"]["target_space"]}
    if c["scaling"] == "global_minmax":
        c["lo"], c["hi"] = float(stats["min"]), float(stats["max"])
        if not c["hi"] > c["lo"]:
            raise SslError(f"dev waveforms have minimum {c['lo']} >= maximum {c['hi']}: global_minmax is undefined")
    if c["space"] == "zscore":
        c["mu"], c["sd"] = float(stats["mean"]), float(stats["std"])
        if not (math.isfinite(c["sd"]) and c["sd"] > 0):
            raise SslError(f"dev waveform spread {c['sd']}: the zscore target space is undefined")
    return c


def scale(x: torch.Tensor, c: dict) -> torch.Tensor:
    """Raw padded waveforms -> the encoder's input (global_minmax is applied here, the other scalings inside)."""
    return (x - c["lo"]) / (c["hi"] - c["lo"]) if c["scaling"] == "global_minmax" else x


def views(enc, x: torch.Tensor, m: torch.Tensor, token: torch.Tensor, c: dict):
    """(P, P_m, T) for raw padded waveforms x (B, L) and mask m (B, N): the clean and the masked patches the tokenizer
    receives, and the reconstruction targets (module docstring)."""
    P = enc.patches(scale(x, c))
    if c["space"] == "input":
        return P, torch.where(m[..., None], token, P), P
    T = ((x - c["mu"]) / c["sd"]).reshape(len(x), -1, PATCH)
    raw = torch.where(m[..., None], token, T).reshape(len(x), -1) * c["sd"] + c["mu"]
    return P, enc.patches(scale(raw, c)), T


def terms(fwd, model, P, Pm, T, m) -> tuple:
    """(next, masked) losses from the views; ``fwd`` = the encoder's forward_patches (maybe compiled)."""
    tok, tok_m = fwd(P), fwd(Pm)
    nxt = ((model.ssl_next(tok[:, :-1]) - T[:, 1:]) ** 2).mean()
    msk = (((model.ssl_masked(tok_m) - T) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1)
    return nxt, msk


def build(cfg: dict, parent=None):
    """(model, mask token, groups): ModuleDict {encoder, ssl_next, ssl_masked} on the CPU, heads and token seeded from
    cfg["seed"], the transformer frozen."""
    s = cfg["ssl"]
    init = pipeline.handover(parent)
    enc = build_encoder(cfg["model"], init_from=init, seed=cfg["seed"]).set_trainable(0, True)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg["seed"])    # the CPU generator only: the caller's CUDA state is kept
        model = nn.ModuleDict({"encoder": enc, "ssl_next": nn.Linear(DIM, PATCH), "ssl_masked": nn.Linear(DIM, PATCH)})
    token = nn.Parameter(torch.zeros(PATCH))
    groups = [{"params": list(enc.tokenizer.parameters()), "lr": float(s["lr_tokenizer"]), "name": "tokenizer"}]
    if enc.adapter is not None:
        groups.append({"params": list(enc.adapter.parameters()), "lr": float(s["lr_adapter"]), "name": "adapter"})
    groups.append({"params": [*model.ssl_next.parameters(), *model.ssl_masked.parameters(), token],
                   "lr": float(s["lr_heads"]), "name": "heads"})
    return model, token, groups


def run(cfg: dict, out, parent) -> dict:
    runtime.setup_numerics()                       # strict fp32 before anything else runs
    out = Path(out)
    check_config(cfg)
    s = cfg["ssl"]
    prec = rundir.parent_record(pipeline.handover(parent))
    dev = wdata.dev(cfg)
    data_hash = dev.meta["data_hash"]
    done = rundir.check_identity(out, cfg, data_hash, prec)
    if done is not None:
        return done
    out.mkdir(parents=True, exist_ok=True)
    with guards.RunLock(out) as lock:
        device = runtime.device()
        c = constants(cfg, dev.meta["waveform"]["stats"])
        model, token, groups = build(cfg, parent)
        check_modes(model)
        model.to(device)
        token.data = token.data.to(device)                                    # the same Parameter, in groups
        enc = model.encoder
        fwd = torch.compile(enc.forward_patches) if s["compile"] and device.type == "cuda" else enc.forward_patches
        X = WaveformSet(dev, np.arange(len(dev)), device).x        # raw, padded, on the device
        wn, wm, p = float(s["next_weight"]), float(s["masked_weight"]), float(s["mask_prob"])

        def loss(b, gen):
            x = X[b]
            m = torch.rand(x.shape[0], x.shape[1] // PATCH, generator=gen, device=gen.device) < p
            nxt, msk = terms(fwd, model, *views(enc, x, m, token, c), m)
            return {"loss": wn * nxt + wm * msk, "next": nxt, "masked": msk}

        r = trainer.train(out, cfg, "ssl", model, groups, loss, len(dev), data_hash=data_hash, parent=prec,
                          extra={"mask_token": token}, lock=lock)
        if r.get("preempted"):
            return r
        hist = load_json(out / "history.json")
        n_trained = sum(t.numel() for g in groups for t in g["params"])
        return trainer.finish(out, cfg, "ssl", r, final_next=hist[-1]["train_next"],
                              final_masked=hist[-1]["train_masked"], constants=c, n_trained=n_trained,
                              n_params=sum(t.numel() for t in model.parameters()) + token.numel())

