# Waveform feature extraction: adaptation, distillation, compression

Adapt TimesFM 2.5 to detector waveforms, fit a ridge teacher, distil it into a small MLP, and compress the student with QKeras.

- **DCH:** primary ionisation cluster counts from drift-chamber waveforms.
- **DRO:** Cherenkov and scintillation photon counts and signal timing from dual-readout waveforms.

The main pipeline is `ssl → ridge → student → compress`. Random-transformer and no-transformer controls,
last-two-block fine-tuning, and a from-scratch CNN are also available.

## Quick start

The supported environment is Linux x86-64 with NVIDIA GPUs. Training needs a GPU, the detector data,
and cached TimesFM weights; this checkout does not include them.

```bash
source setup.sh
uv sync --frozen
(cd compression && uv sync --frozen)

python scripts/prepare_data.py dro_gen pwo_smoke
python scripts/train.py configs/smoke/dro.yaml run.group=smoke run.name=example
uv run --frozen --project compression python compression/compress.py outputs/smoke/example
```

The main environment uses Python 3.12 and PyTorch; compression uses Python 3.10, TensorFlow 2.15 and QKeras.
Both have locked dependencies. Replace `/path/to/wavefm-data` in the dataset YAML files with your data location.
Set `HF_HOME` to your cached TimesFM weights (default: `.cache/huggingface` in this checkout).
Runs go to `outputs/` unless `WAVEFM_OUTPUT` is set. Adapt the example SLURM account, partition and constraint
to your cluster, and run computation on a compute node. The `wavefm` package name is retained for imports.

## Repository

| Directory | Purpose |
|---|---|
| `src/wavefm/` | Models, preprocessing, training and checkpoint handling |
| `data/` | Dataset builders, split validation and metrics |
| `configs/paper/` | Paper recipes and random/no-transformer controls |
| `configs/smoke/` | Small end-to-end checks |
| `scripts/` | Data preparation, training, evaluation, scans and summaries |
| `compression/` | Pruning and quantisation in a separate environment |
| `analysis/` | Run inspection, training curves and comparison plots |
| `tests/` | Pipeline, numerical and data-integrity checks |

## Running and resuming

```bash
sbatch -t 0:15:00 scripts/example_job.sh configs/smoke/dro.yaml run.group=smoke run.name=job
```

Repeat the same command to resume interrupted training or skip completed stages. A run with different inputs
or configuration needs a new name. Training exits with code 3 after a checkpointed deadline or SIGTERM.
Compression restarts from the beginning if interrupted.

Choose settings using development data only. Before final evaluation, inspect the training curves and record a
verdict with `scripts/judge.py`. Held-out evaluation is recorded once; summaries accept completed, verified,
non-smoke runs with a passing verdict.

## Documentation

- [Workflow reference](docs/workflow.md): data contracts, evaluation, scans, compression and test commands.
- [Paper recipes](configs/paper/README.md): original settings, differences from the submitted runs and numerical checks.
- [DCH dataset](data/dch_wf/README.md) and [DRO dataset](data/dro_gen/README.md).
- [Curve review](analysis/JUDGING.md): how training runs qualify for summaries.

## Validation

```bash
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu
uv run --frozen --project compression python -m pytest -q compression/tests
```

GPU tests require a GPU and cached TimesFM weights. `python scripts/smoke.py dro` (or `dch`) checks the full
pipeline using a small dataset; smoke outputs are not scientific results.

## Scope

This implementation is not an exact replay of every submitted run: see the [paper recipe notes](configs/paper/README.md).
It does not include DCH kaon–pion separation, raw ROOT dataset reconstruction, HLS conversion, or hardware measurements.
