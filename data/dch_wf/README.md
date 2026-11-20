# dch_wf: primary-cluster counting in drift-chamber waveforms

## Source
Simulated waveforms of a CEPC 4th-concept drift-chamber cell (Tian et al., "Cluster counting algorithm for the CEPC
drift chamber using LSTM and DGCNN", arXiv:2402.16493; samples published by the authors on scidb.cn, dataset V2).
Heed ionisation, Garfield++-parameterised drift and amplification, the measured preamplifier response and noise taken
from test-beam data (noise level 5 %). Each waveform has 3000 samples at 1.5 GS/s (the paper's 2000 ns window), so
`dt_ns` = 1/1.5 and the first sample is at 0 ns.

The ROOT files were converted to npz by `CFS/DCH_wf/processor.ipynb` (members `wf_i (n, 3000) float32`,
`mom (n,) float32` in GeV/c, `tag_times (n, 300) int32`, `tag_values (n, 300) int32`; 100 000 events per file):

| split | files | paper sample (Table 1) |
|---|---|---|
| `dev` | `CFS/DCH_wf/processed_data_train/batch_{0..4}.npz` | peak-finding training, 500 000 pi, 0.2-20 GeV/c (`signal_noise05_PFtrain_*.root`) |
| `heldout` | `CFS/DCH_wf/processed_data_test/batch_{0..4}.npz` | peak-finding testing, 500 000 pi, 0.2-20 GeV/c (`signal_noise05_PFtest_*.root`) |

Nothing is filtered: the rows of a split are its events in file order. The waveforms are not copied; the split
records each file, its member `wf_i` and the sha256 of its float32 content. A member of an npz cannot be memory-mapped,
so reading a split decompresses each file once (about 6 GB of waveforms per split held in memory).

Smoke set `pion_smoke`: the first 2500 events of `processed_data_train/batch_0.npz` (dev) and the first 500 of
`processed_data_test/batch_0.npz` (heldout), for smoke tests only. It uses the optional config key
`first_rows: {dev: N, heldout: M}` (absent in `pion`: every row): a split is then its first N events in file order, and
because that is only part of a member its waveforms are copied into the split as `waveform.npy`.

Unit: `a.u.`. The paper plots the waveform as "Amplitude" of the induced current with no unit, and the conversion
keeps the simulated values; the unit is certainly not a calibrated physical one (confidence: high that it is
arbitrary, none about any scale behind it).

## Labels and extras
A tag is a (time, value) pair: time is a sample index, value 1 a primary and 2 a secondary ionisation peak, 0 an
unused slot (the arrays are zero-padded to 300).
* `n_primary` (clusters, float32) = number of tags with value 1 and time in [0, 3000). The time condition is the
  definition of the old repository and of the paper's code; on these files it removes nothing (every used tag time
  lies in [1, 2998]).
* extra `classes (N, 3000) int8`: 0 noise, 1 primary, 2 secondary at the tagged samples (same time condition); a
  sample tagged both is 2. The auxiliary per-sample target of the 2-layer method.

## Metric (`evaluate.py`)
For predicted counts p and true counts y of N events, in clusters, in float64:
`count_mae = mean |p - y|`, `count_rmse = sqrt(mean (p - y)^2)`. A non-finite prediction is an error of +inf, so both
metrics become inf; it is never dropped and never NaN.

## Known issues
1. **300-tag truncation.** The conversion keeps at most 300 tags per event. An event whose 300 slots are all used
   may have lost tags and then under-counts `n_primary`: 92 of 500 000 events in `dev` (1.8e-4) and 67 of 500 000 in
   `heldout` (1.3e-4) fill all 300 slots (measured 2026-11-19). Accepted as is.
2. **`heldout` is not a pristine test set.** `processed_data_test` was evaluated repeatedly during the earlier
   development of the paper's DCH methods (its record says: "Development included repeated model/test
   comparisons"). Results on it are not a single untouched test evaluation.

K/pi separation (the 10 GeV/c pion and kaon samples in `CFS/DCH_wf/{pion,kaon}`) is not part of this data
set.
