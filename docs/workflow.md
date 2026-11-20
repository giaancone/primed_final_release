# Workflow reference

Paths and commands below are relative to the repository root.

## Data sets

A data set is a problem: its source, two splits `dev` and `heldout` with their metadata, the metric `evaluate` and
the result of the reference method on `heldout`. Each data config (`data/<set>/configs/<config>.yaml`) has one
choice for everything and is the same for every method. See `data/dro_gen/README.md` (paper set `pwo_realistic`,
smoke set `pwo_smoke`) and `data/dch_wf/README.md` (`pion`, smoke set `pion_smoke`).

```bash
python scripts/prepare_data.py dro_gen pwo_smoke     # builds it, or verifies the one that exists and leaves it
```

A split is checked when it is opened: content digests, and the data hash (the expanded data config plus the code of
`data/`), so a result is never computed on a data set that has changed since it was built.

**The heldout rule.** A method sees only `dev`: every choice (scaling, rates, lengths, alpha, the kept weights) is
made on `dev`. Validation against true labels uses rows inside the label budget; the student's validation against
teacher labels uses dev rows that may lie outside it, with their teacher labels only. `heldout` is predicted once, in physical
units, and scored by the data set. Two entries open it: `scripts/evaluate.py` (`wavefm.data.heldout` raises
`HeldoutError` anywhere else) and `compression/compress.py --evaluate` for the compressed model.

## Running a pipeline

A run is one config and one seed. Keys can be overridden on the command line (`key.path=value`); unknown keys are
errors. The stages of `stages` run in order, each in `<output>/<group>/<name>/<stage>/`, each starting from the file
the stage before hands over. The pipeline stops before `compress` and prints its command; `compress` needs `ridge`
and a teacher-label `student` before it, which the pipeline checks before any training.

```bash
python scripts/train.py configs/smoke/dro.yaml run.group=smoke run.name=readme
uv run --frozen --project compression python compression/compress.py outputs/smoke/readme
```

A stage's `result.json` is written last and records its identity (config fingerprint, data hash, sha256 of the file
it started from). Starting the same command again skips finished stages and resumes an interrupted one exactly (the
stages of the shared trainer: ssl, student, finetune); a finished stage of another identity is refused (use a new
`run.name`); nothing is overwritten. A stage stopped by SIGTERM or at `run.deadline` checkpoints and `train.py` exits
with code 3: start it again. Compression has no resume: an interrupted `compress` starts again from the beginning
(its partial `history.json` stays readable).
`scripts/example_job.sh` is a minimal SLURM job for one run (one GPU; supply your cluster account with `sbatch -A <account>`):

```bash
sbatch -t 0:15:00 scripts/example_job.sh configs/smoke/dro.yaml run.group=smoke run.name=job
```

**Smoke test.** One command per task chains every part through both environments on the small data set, with the
real TimesFM weights and at most 20 steps per trained stage (about 100 s on one A100), and checks each part (results
readable, promised files present, finite losses, the summary refuses the run, reload gives the same predictions and
the recorded loss, each heldout record names the present result of its stage):

```bash
python scripts/smoke.py dro
python scripts/smoke.py dch
```

Smoke runs carry `run.smoke: true`: their numbers are not results and they never enter a summary.

## Judging and summarising

Training defaults to a constant rate, fixed steps and the last weights. Every loss curve is kept and judged before a
result counts (`analysis/JUDGING.md`): `pass`, `overfit`, `undertrained` or `unclear`, written by a reviewer.

```bash
python scripts/judge.py --pending outputs            # curves.png and a table for every stage still to judge
python scripts/judge.py outputs/smoke/readme/student pass "flat at the end" --judge me
python scripts/summarize.py outputs student --seeds 0
```

The summary takes only accepted stages (finished, not smoke, files intact, verdict `pass` on that very result; ridge
needs none) and averages only over seeds of one recipe (same data, label budget and config but the seed).

## Scans

Rates and lengths are scanned once, on one seed, and then fixed as the defaults of the implementation. A scan is
development: its configs and curves are evidence, not part of the method (`configs/scans_dev/README.md`). Format in
the docstring of `scripts/run_scan.py`; `--dry-run` expands and checks every config without training.

```bash
python scripts/run_scan.py <scan.yaml> --dry-run
python scripts/run_scan.py <scan.yaml> --gpus 4
```

## Final evaluation

Once per run, after the curves are judged (a non-smoke run is refused unless every stage up to the evaluated one is
accepted): the last stage of `stages` that predicts (ridge, finetune or student) must have finished; it is asked for
its heldout predictions with the constants it saved, and the data set scores them. A second call refuses. The
compressed model is scored the same way in its own environment, into `<run>/compress/heldout.json`; the summary of
each stage reads its own record.

```bash
python scripts/evaluate.py outputs/smoke/readme                      # writes <run>/heldout.json
uv run --frozen --project compression python compression/compress.py --evaluate outputs/smoke/readme
```

## Compression

`compression/compress.py` reads the student (`student/student.npz`), the teacher labels and the student's train and
validation rows (each checked against the size and sha256 its stage recorded), builds the same MLP in QKeras, prunes it (`compress.sparsity`, `prune_steps`) and trains it with
quantisation (`compress.kernel`, `bias`, `activation`, `steps`), and writes `compress/model.h5` with the parameter
count and sparsity in `result.json`. It ends there: HLS and hardware conversion are not part of WaveFM.

## Tests

On a GPU node (the GPU tests need a GPU and the TimesFM weights, and fail rather than skip without them):

```bash
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu
uv run --frozen --project compression python -m pytest -q compression/tests
```

From another directory: `python -m pytest -q $WAVEFM_ROOT/tests -c $WAVEFM_ROOT/pyproject.toml --rootdir $WAVEFM_ROOT`.

## Not included

- the K/pi separation metric of DCH, and a DCH data set rebuilt from the raw ROOT files;
- cross-validation at low label counts;
- ablations of the method: a causal adapter (the symmetric one lets token i see patch i+1), the choice of block;
- re-tuned training dynamics: `configs/paper/` holds the values the paper ran with (`configs/paper/README.md`), and
  the `reference` results of the data sets are empty;
- HLS and hardware conversion are out of scope.
