"""The one training loop of every trained stage (finetune, ssl, student): a function the stage calls with plain pieces.

    summary = train(out, cfg, stage, model, groups, loss, n_train, data_hash=.., parent=..,
                    validate=None, extra=None, unused=(), lock=None)
    if summary.get("preempted"):
        return summary                              # checkpointed; the next start resumes exactly
    ... the stage writes its own files from the trained ``model`` ...
    return finish(out, cfg, stage, summary, artifacts=("rows.npz",), **stage_values)

The stage owns everything task-specific; the loop owns the dynamics of ``cfg[stage]`` (configs/teacher.yaml):
  model      nn.Module; its state dict (plus ``extra``) is what weights.pt holds and what a resume restores
  groups     AdamW parameter groups (``params``, ``lr``, ``weight_decay``, optional ``name``); the base rates
  loss(b, gen) -> scalar tensor, or {"loss": the optimised total, <name>: tensor, ..}: one training batch. ``b`` is a
             device tensor of positions 0..n_train-1 (data.batches.BatchSampler: without replacement, a new
             permutation every epoch, from ``cfg["seed"]``). ``gen`` is a torch.Generator on the model's device,
             owned and checkpointed here, for every random draw of the step (augmentation, SSL masks): never draw from
             the global generators in a step, use ``gen``. Named terms are recorded as ``train_<name>``.
  n_train    rows of the training set: epochs <-> steps, and the sampler
  validate() -> float, or {"val_loss": float, <name>: float, ..}: called in eval mode under no_grad at every
             evaluation. ``val_loss`` is the primary value: the stage's own loss on its validation rows, the only value
             model selection (``keep: best_val``) and ``schedule: plateau`` look at; the others are recorded only (the
             student's loss on true labels). None: the stage has no validation set (ssl); then ``keep: best_val`` and
             ``schedule: plateau`` are refused and the curves are the training losses alone.
  extra      {name: tensor} trained tensors outside ``model`` (a mask token, temporary heads as
             ``head.state_dict(keep_vars=True)`` under a prefix): saved in weights.pt under their names next to the
             model's keys, restored in place on resume. They must be in ``groups`` to be trained.
  data_hash, parent   identity of the stage (rundir): the data set's hash and rundir.parent_record of the file the
             stage starts from (None for a first stage)
  unused     name fragments of trained tensors that legitimately get no gradient (models.build.unused_parameters)
  lock       the stage's guards.RunLock, checked at every evaluation and checkpoint

Length (plan, section 2): exactly one of ``steps`` / ``epochs`` is positive, else TrainerError. With epochs the run
has ``ceil(epochs * n_train / batch_size)`` steps; nothing is clipped to a maximum. An "epoch" is n_train samples.
Requested and actual steps, epochs and samples are printed at the start and recorded.

Defaults: constant rate, no warm-up, ``steps`` updates, the LAST weights. Advanced and off by default (train.schedule):
cosine, plateau (rate drops and early stopping on the validation loss), warm-up, ``keep: best_val`` (the weights of
the evaluation with the lowest ``val_loss``: a checkpoint method, recorded as such). Model selection and plateau use
the stage's validation LOSS, never a metric. AdamW (betas 0.9, 0.999), gradient clipping at ``grad_clip`` (0: none)
of the global norm of all trained tensors.

Brakes. Mid-run the loop stops ONLY on
  * a non-finite training loss, gradient norm or validation loss, or a non-finite weight (model or ``extra``) at an
    evaluation (the last step is always one, so weights broken by the final update are never saved), and
  * divergence: the mean training loss of an interval exceeds DIVERGE times the loss of the first step,
checked every CHECK_EVERY steps and at every evaluation. A rising validation loss never stops a run (a run that gets
worse and recovers runs to the end); only ``schedule: plateau``, chosen explicitly, stops early. A braked stage
raises TrainingBraked and writes no result.json.

Curves. history.json, one row per evaluation: {"step", "train_loss" (mean over the steps since the last row),
"val_loss" (absent without validation), the other validation values, "lr" (the largest rate among the groups at
the last step of the interval), "grad_norm" (mean pre-clip global norm since the last row), "clip_frac" (fraction of
those steps that were clipped), "train_<name>" (named loss terms)}. Evaluations: every ``eval_every`` steps when it is
> 0, else ``eval_points`` equally spaced ones; the last step always.

Resume (exact; it is not a checkpoint method). ckpt.pt holds the model, ``extra``, the optimiser, the schedule, the
sampler, the CPU and CUDA generator states, ``gen``, the loss / gradient accumulators of the open interval, the best
validation state, and the history. It is written every ``ckpt_minutes`` and when the run stops on SIGTERM or at
``run.deadline``; the signal is acted on at the next step boundary, after the evaluation of a step that has one, so
no evaluation is skipped. A SIGTERM that arrives during the terminal evaluation (the last step, or the one that ends
a plateau run) does not interrupt the finished stage: it is passed on to the SIGTERM handler installed before (the
pipeline's, which then starts no further stage; a deadline passed then is caught by the pipeline too). A resume is
refused (TrainerError) when the stage's config fingerprint (rundir.config_fp), the data hash or the parent's sha256
changed. ``finish`` removes ckpt.pt.

Numerics: the loss and gradient-norm accumulators are float64 (sums of per-step fp32 values). Strict fp32 is checked
before the first step (guards.check_strict_fp32); the stage sets it first (utils.runtime.setup_numerics).
"""
import math
import signal
import time
from pathlib import Path

import torch

from analysis.curves import hints
from wavefm.data.batches import BatchSampler
from wavefm.train import guards, rundir, schedule
from wavefm.utils.io import save_json, save_torch, remove_tmp_files

PREEMPTED = 3             # exit code of scripts/train.py when a stage stopped at its deadline or on SIGTERM
CHECK_EVERY = 50          # steps between host syncs for the brakes
DIVERGE = 100.0           # an interval's mean training loss above DIVERGE x the first step's loss is divergence
GEN_OFFSET = 1_000_003    # the step generator is seeded with cfg["seed"] + GEN_OFFSET (the sampler with cfg["seed"])
KEEP = ("last", "best_val")


class TrainerError(ValueError):
    pass


class TrainingBraked(guards.GuardError):
    pass


def length(d: dict, n_train: int) -> dict:
    """{"steps", "epochs", "samples"} requested by the dynamics section ``d`` for ``n_train`` training rows."""
    steps, epochs, bs = d["steps"], d["epochs"], d["batch_size"]
    if (steps > 0) == (epochs > 0):
        raise TrainerError(f"steps = {steps}, epochs = {epochs}: set exactly one of them positive (the other 0)")
    if not 1 <= bs <= n_train:
        raise TrainerError(f"batch_size {bs} for {n_train} training rows: it must be between 1 and the row count")
    s = int(steps) if steps > 0 else math.ceil(epochs * n_train / bs)
    return {"steps": s, "epochs": float(epochs) if epochs > 0 else s * bs / n_train, "samples": s * bs}


def eval_steps(d: dict, steps: int) -> set:
    """The steps after which the curves get a row: every eval_every (> 0), else eval_points equally spaced; the last."""
    if d["eval_every"] > 0:
        return set(range(d["eval_every"], steps + 1, d["eval_every"])) | {steps}
    if d["eval_points"] < 1:
        raise TrainerError("eval_points must be >= 1 when eval_every is 0")
    n = min(d["eval_points"], steps)
    return {math.ceil(k * steps / n) for k in range(1, n + 1)}


def _check_options(d, validate):
    if d["keep"] not in KEEP:
        raise TrainerError(f"keep {d['keep']!r}: known {KEEP}")
    if validate is None and (d["keep"] == "best_val" or d["schedule"] == "plateau"):
        raise TrainerError(f"keep {d['keep']!r} / schedule {d['schedule']!r}: this stage has no validation set, so "
                           f"best_val and plateau are undefined; use keep: last with constant or cosine")
    if d["schedule"] == "plateau" and d["eval_every"] <= 0:
        raise TrainerError("schedule plateau counts its patience in evaluations: set eval_every > 0 (a fixed interval)")


def _identity(cfg, stage, data_hash, parent) -> dict:
    return {"config": rundir.config_fp(cfg, stage), "data set": data_hash,
            "parent weights": (parent or {}).get("sha256")}


def _cpu(state: dict) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in state.items()}


def train(out, cfg, stage, model, groups, loss, n_train, *, data_hash, parent, validate=None, extra=None, unused=(),
          lock=None) -> dict:
    """Train (or resume) ``model`` as ``cfg[stage]`` says. Returns {"preempted": True} after a clean stop, else the
    summary that ``finish`` turns into result.json; ``model`` and ``extra`` then hold the kept weights, and
    weights.pt and history.json are written."""
    out, d, t_start = Path(out), cfg[stage], time.time()
    extra = dict(extra or {})
    req = length(d, n_train)
    steps, bs = req["steps"], d["batch_size"]
    _check_options(d, validate)
    if extra.keys() & model.state_dict().keys():
        raise TrainerError(f"extra tensors {sorted(extra.keys() & model.state_dict().keys())} clash with model keys")
    guards.check_strict_fp32(model, extra)
    trained = [p for g in groups for p in g["params"]]
    dev = trained[0].device
    out.mkdir(parents=True, exist_ok=True)
    remove_tmp_files(out)

    plan = schedule.make(d, steps)
    evals = eval_steps(d, steps)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=d["weight_decay"], fused=dev.type == "cuda")
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    sampler = BatchSampler(n_train, bs, cfg["seed"], dev)
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"] + GEN_OFFSET)
    clip = d["grad_clip"] if d["grad_clip"] > 0 else math.inf
    names = {**{id(p): n for n, p in model.named_parameters()}, **{id(t): k for k, t in extra.items()}}
    ident = _identity(cfg, stage, data_hash, parent)
    zero = lambda: torch.zeros((), device=dev, dtype=torch.float64)
    st = {"step": 0, "history": [], "loss0": None, "best": math.inf, "best_step": None, "best_state": None,
          "wall_s": 0.0, "acc": {"loss": zero(), "grad_norm": zero(), "clipped": zero()}, "n": 0}

    ck = out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        for what, want in ident.items():
            if s["identity"][what] != want:
                raise TrainerError(f"{ck} was written with another {what} ({s['identity'][what]} != {want}): refusing "
                                   f"to resume. Use a new run.name, or delete {ck} to start this stage again")
        model.load_state_dict(s["model"])
        with torch.no_grad():
            for k, t in extra.items():
                t.copy_(s["extra"][k])
        opt.load_state_dict(s["opt"]); plan.load_state_dict(s["plan"]); sampler.load_state_dict(s["sampler"])
        torch.set_rng_state(s["cpu_rng"]); gen.set_state(s["gen"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(s["cuda_rng"], dev)
        st = s["state"]
        st["acc"] = {k: v.to(dev) for k, v in st["acc"].items()}
        print(f"[train] {stage}: resumed at step {st['step']} of {steps}", flush=True)
    else:
        print(f"[train] {stage}: {n_train} training rows, batch {bs}: requested {steps} steps = {req['epochs']:.4g} "
              f"epochs = {req['samples']} samples; schedule {d['schedule']}, keep {d['keep']}, "
              f"{len(evals)} evaluations, "
              f"{'no validation set' if validate is None else 'validation loss'}", flush=True)

    def save_ckpt():
        save_torch({"identity": ident, "model": model.state_dict(), "extra": _cpu(extra), "opt": opt.state_dict(),
                    "plan": plan.state_dict(), "sampler": sampler.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None, "gen": gen.get_state(),
                    "state": {**st, "wall_s": st["wall_s"] + time.time() - t_start}}, ck)
        if lock is not None:
            lock.check()

    def brakes(step):
        acc, n = st["acc"], max(st["n"], 1)
        bad = [k for k in ("loss", "grad_norm") if not torch.isfinite(acc[k])]
        if bad:
            raise TrainingBraked(f"{stage}: non-finite training {' and '.join(bad)} before step {step}: the run is "
                                 f"stopped and writes no result. Lower the learning rate or check the inputs; "
                                 f"then use a new run.name or delete {ck}")
        mean = acc["loss"].item() / n
        if st["loss0"] is not None and st["loss0"] > 0 and mean > DIVERGE * st["loss0"]:
            raise TrainingBraked(f"{stage}: diverged before step {step}: mean training loss {mean:.4g} > {DIVERGE:g} x "
                                 f"the first step's {st['loss0']:.4g}. Lower the learning rate; then use a new "
                                 f"run.name or delete {ck}")

    def finite_weights(step):
        bad = [k for k, v in {**model.state_dict(), **extra}.items()
               if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise TrainingBraked(f"{stage}: non-finite weights after step {step} ({len(bad)} tensors, e.g. {bad[0]}): "
                                 f"the run is stopped and writes no result. Lower the learning rate or check the "
                                 f"inputs; then use a new run.name or delete {ck}")

    stop = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    deadline = cfg["run"]["deadline"] or math.inf
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    step0, t_loop, t_ckpt, first = st["step"], time.time(), time.time(), True
    try:
        model.train()
        while st["step"] < steps and not plan.done:
            if stop or time.time() > deadline:
                save_ckpt()
                why = "SIGTERM" if stop else "deadline"
                print(f"[train] {stage}: {why}: checkpoint at step {st['step']}, stopping", flush=True)
                return {"preempted": True}
            f = plan.factor(st["step"])
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            terms = loss(sampler.next(), gen)
            terms = terms if isinstance(terms, dict) else {"loss": terms}
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            if first:
                guards.check_gradients(names, groups, unused)
                first = False
            gn = torch.nn.utils.clip_grad_norm_(trained, clip)
            opt.step()
            st["step"] += 1
            step, acc = st["step"], st["acc"]
            acc["loss"] += terms["loss"].detach(); acc["grad_norm"] += gn.detach(); acc["clipped"] += gn.detach() > clip
            for k, v in terms.items():
                if k != "loss":
                    acc[f"train_{k}"] = acc.get(f"train_{k}", zero()) + v.detach()
            st["n"] += 1
            if st["loss0"] is None:
                st["loss0"] = terms["loss"].item()
            if step in evals:
                brakes(step)
                finite_weights(step)
                n = st["n"]
                row = {"step": step, "train_loss": acc["loss"].item() / n}
                if validate is not None:
                    model.eval()
                    with torch.no_grad():
                        v = validate()
                    model.train()
                    v = dict(v) if isinstance(v, dict) else {"val_loss": v}
                    v = {k: float(x) for k, x in v.items()}
                    if not math.isfinite(v["val_loss"]):
                        raise TrainingBraked(f"{stage}: non-finite validation loss at step {step}: the run is stopped "
                                             f"and writes no result")
                    row.update(v)
                    plan.observe(v["val_loss"])
                    if d["keep"] == "best_val" and v["val_loss"] < st["best"]:
                        st["best"], st["best_step"] = v["val_loss"], step
                        st["best_state"] = (_cpu(model.state_dict()), _cpu(extra))
                row.update(lr=max(g["lr"] for g in opt.param_groups), grad_norm=acc["grad_norm"].item() / n,
                           clip_frac=acc["clipped"].item() / n)
                row.update({k: a.item() / n for k, a in acc.items() if k.startswith("train_")})
                st["history"].append(row)
                st["acc"], st["n"] = {"loss": zero(), "grad_norm": zero(), "clipped": zero()}, 0
                save_json(st["history"], out / "history.json")
                if lock is not None:
                    lock.check()
                print(f"[train] {stage} step {step}/{steps} loss {row['train_loss']:.5g}"
                      + (f" val {row['val_loss']:.5g}" if "val_loss" in row else "")
                      + f" lr {row['lr']:.3g} clip {row['clip_frac']:.2f}", flush=True)
            elif step % CHECK_EVERY == 0:
                brakes(step)
            if time.time() - t_ckpt > 60 * d["ckpt_minutes"] and st["step"] < steps and not plan.done:
                save_ckpt(); t_ckpt = time.time()
    finally:
        signal.signal(signal.SIGTERM, previous)
    if stop and callable(previous):         # arrived during the terminal evaluation: the stage is complete; pass it on
        previous(signal.SIGTERM, None)

    step, hist = st["step"], st["history"]
    kept = step
    if d["keep"] == "best_val":
        model.load_state_dict(st["best_state"][0])
        with torch.no_grad():
            for k, t in extra.items():
                t.copy_(st["best_state"][1][k])
        kept = st["best_step"]
    save_torch({**_cpu(model.state_dict()), **_cpu(extra)}, out / "weights.pt")
    save_json(hist, out / "history.json")
    rate = (step - step0) / (time.time() - t_loop) if step > step0 else None
    summary = {"data_hash": data_hash, "parent": parent, "requested": req,
               "actual": {"steps": step, "epochs": step * bs / n_train, "samples": step * bs},
               "hints": hints(hist), "wall_s": st["wall_s"] + time.time() - t_start,
               "ended_by": "plateau" if plan.done else "steps", "keep": d["keep"],
               "checkpoint_method": d["keep"] != "last", "kept_step": kept,
               "final_train_loss": hist[-1]["train_loss"], "final_val_loss": hist[-1].get("val_loss"),
               "steps_per_s": rate, "n_train": n_train, "batch_size": bs,
               "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None}
    print(f"[train] {stage}: done, {step} of {steps} steps (ended by {summary['ended_by']}), kept step {kept}, "
          f"final train loss {summary['final_train_loss']:.5g}", flush=True)
    return summary


def finish(out, cfg, stage, summary, artifacts=(), **more) -> dict:
    """Write result.json of a trained stage (rundir.write_result) from the summary of ``train`` and the stage's own
    files and values, then remove the resume checkpoint. Call it last."""
    s = dict(summary)
    r = rundir.write_result(out, stage, cfg, s.pop("data_hash"), s.pop("parent"), requested=s.pop("requested"),
                            actual=s.pop("actual"), hints=s.pop("hints"), wall_s=s.pop("wall_s"),
                            artifacts=("weights.pt", "history.json", *artifacts), **s, **more)
    (Path(out) / "ckpt.pt").unlink(missing_ok=True)
    return r
