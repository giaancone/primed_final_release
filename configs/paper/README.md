# Paper configs: what "as run" means

`dro.yaml` and `dch.yaml` hold, for every key, the value the old paper scripts ran with (DRO: work_dro/ssl_adapt2.py,
feat_ds.py, ridge_ds.py, mlp_multi.py, compress_dro.py; DCH: dch_norm_aux_v2/src/dch_ssl.py and its student and
compression scripts), where the keys can express it. The training dynamics are those values too: they were not
re-tuned under the present protocol (`configs/scans_dev/README.md` says how that is done). The `*_no_blocks` / `*_random_blocks` files are the
two controls of each task.

## As-run behaviours the keys cannot express (the new code does the plain thing instead)

- Data: the old DRO scripts used an 80 000-row pool and preset constants (input min/max -305.34 / 9083.94, target
  mean/sd, SSL z-score mean/sd from the first 5000 pool rows); the new code takes the statistics of the dev split
  (`meta.json`) of `dro_gen/pwo_realistic`. DCH SSL used 500 000 rows; DRO dropped the last partial batch.
- SSL: DRO trained one SSL per seed with TF32 matmul on; DCH one SSL (seed 0) shared by every seed. New: one SSL per
  run, strict fp32. DRO loaded the pretrained weights with `strict=False`, DCH strictly; new: strictly.
- Ridge: DRO standardised features with eps 1e-12 over all pool rows, scaled targets with preset constants, chose
  alpha by validation error on seed 0 and reused it; DCH used eps 1e-6, scanned per run by validation MAE. New: alpha
  per output by closed-form leave-one-out on the labelled rows, every run.
- Label budget: DRO had no minimum; DCH `max(32, .)` labelled and `max(8, .)` validation rows (`data.min_labelled`,
  `data.min_val`).
- Student: DRO drew batches with replacement from a generator seeded per process, ran 800 k steps and kept the best
  EMA (0.7/0.3) of the validation loss on true labels, checked every 4000 steps; DCH held its rate, then halved it
  every 40 k steps to 1e-5 with patience 20 k, clipped gradients at 1, validated every 50 steps on an EMA of the MAE.
  New: epoch permutations, the shared trainer's dynamics keys, selection (if `keep: best_val`) on the plain
  teacher-label validation loss.
- Compression: DRO equalised units, conditioned activations (cap 4 - 2^-8), used cosine schedules with early
  stopping and removed dead units afterwards; DCH pruned with patience 8, trained QAT with patience 10. New: fixed
  step counts, constant rate, last weights (`compression/compress.py`); the epoch counts are upper bounds there.
- Metrics: DRO err68 of C/S and t0; DCH separation (x 7.454), MAE, RMSE: defined once by `data/<set>/evaluate.py`.

## Known as-run quirks kept on purpose

- The symmetric conv3 adapter lets token i see patch i+1, so the next-patch target is visible in its input.
- Padded samples are part of the SSL targets and of the token mean of ridge.
- DRO's SSL targets and mask token live in the z-scored waveform while the encoder sees min-max inputs
  (`ssl.target_space: zscore`).

## Numerical check against the old code (2026-11-20, A100, strict fp32 on both sides)

The new encoder, scalings, SSL views and terms were run against verbatim copies of the old code on the same inputs
(real dev rows and the synthetic batch of `tests/test_paper.py`), the same pretrained file, tokenizer, adapter, heads,
mask token and mask; once with the adapter as built (last convolution zero) and once with a trained-looking tokenizer
and a non-zero adapter. Result: DRO bit-identical in every case; DCH within 1.8e-6 relative (tokens of the last
block). The DCH residual is float32 round-off: the new causal per-patch statistics accumulate in float64, the old
TimesFM code merges them in float32 (Welford). `tests/test_paper.py` holds the new code's values on the synthetic
batch (the old code agrees to at most 1.3e-7 relative on each of them) and fails if any moves by more than 1e-5.
