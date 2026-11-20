# dro_gen: simulated dual-readout crystal waveforms

One SiPM channel reading a crystal of a dual-readout calorimeter. Each event is a waveform of 6246 samples at
0.32 ns (window -400 ns to 1598.4 ns) and three labels: the numbers of Cherenkov photons C and scintillation
photons S put into it, and the event time offset t0 (ns). The task is to recover (C, S, t0); the score is the
resolution of the ratio C/S and of t0.

Build: `python scripts/prepare_data.py dro_gen <config>` (calls `data.store.prepare` -> `build.build`). The splits
are written to `out` of the config, under `/path/to/wavefm-data/dro_gen/<config>/{dev,heldout}`.
Generation runs at about 1500-2000 rows/s per core (measured 2026-11-19 on a GPU-cluster node, 1000 rows); 100 000
rows of `pwo_realistic` take 14 s with 8 workers and 2.5 GB of memory.

## Configs

| config | crystal | sensor | C range | S range |
|---|---|---|---|---|
| `pwo_realistic` | PbWO4, 7.5 ns | realistic | 0..1500 | 0..900 |
| `pwo_ideal` | PbWO4 | ideal | 0..1500 | 0..900 |
| `bso_realistic`, `bso_ideal` | BSO, 22 ns (fast) + 98 ns | | 0..1500 | 0..900 |
| `bgo_realistic`, `bgo_ideal` | BGO, 50 ns (fast) + 320 ns | | 0..1500 | 0..5100 |

`pwo_realistic` is the data set of the paper: its unfiltered 100 000 rows are bit for bit the paper file
PWO_real3_100k (2026-11-18). Its config holds the digests of that file (`expect`); `build` refuses to write anything
else. All six configs use the same seed and draws.

- **realistic**: dark counts at 10 MHz (Poisson over the window, uniform in time), photoelectron gain spread
  Normal(1, 0.10), optical crosstalk 7 % (one extra cell, for signal and dark counts, not in the labels), white
  Gaussian noise of 0.10 photoelectron peak amplitude r.m.s. per sample.
- **ideal**: the same with `noise: {dark_rate_hz: 0, spe_sigma: 0, crosstalk: 0, electronic_spe: 0}`: no dark
  counts, every photoelectron exactly one response of peak `spe_amplitude`, no crosstalk, no white noise. What
  stays: the photon statistics (arrival times, fast/slow component), the sub-sample placement on the grid, and the
  measured response itself, which is an average of laser waveforms and keeps their small residual noise. The draws
  follow the same order, but switching the effects off changes how many numbers are drawn (no dark-count times),
  so after the first row of each chunk the ideal and realistic sets are different samples: an ideal row is NOT the
  realistic row of the same number without noise (measured: 2 of the first 1000 rows share C, S, t0).

Every crystal constant and sensor parameter is in `configs/_base.yaml`; the crystal constants are those of the
original generator (`dro_wf_generator`, `config/setup_config.yaml`).

## Generator (generate.py)

Chunk k (500 rows) draws from `numpy.random.default_rng([run.seed, k])`, so rows do not depend on the number of
workers. Per row, in this order (the order is part of the definition):
1. C, S uniform integers in 0..c_max, 0..s_max (source `root`: given, no draw); t0 uniform in +-2.5 ns (or given)
2. number of fast scintillation photons: Binomial(S, a1 t1 / (a1 t1 + a2 t2)); number of dark counts: Poisson
3. arrival times: Cherenkov Exp(0.1 ns), scintillation Exp(t1) or Exp(t2), all + onset (-100 ns) + t0; dark
   counts uniform in the window
4. crosstalk flags; amplitudes Normal(1, spe_sigma), plus a second Normal(1, spe_sigma) where crosstalk fired
5. linear sub-sample deposit on the grid (photons in the last sample or outside are lost), FFT convolution with
   the response, white noise Normal(0, electronic_spe * spe_amplitude); stored as float32

The response is the mean laser-calibration waveform of run 477 (`templates/spr_laser_run477.csv`, 0.16 ns,
12500 samples), from 0 ns on, interpolated to 0.32 ns over 10000 samples, the last 10 file samples dropped, and
scaled so its peak is `spe_amplitude` = 3.621.

Parallel mode spawns worker processes: a script calling `build(cfg, workers > 1)` must do so under
`if __name__ == "__main__":` (the backup generator's recursion bug came from a caller without it).

## Waveform unit: mV (fairly sure, not certain)

The scale is set by one number, the peak of one photoelectron, 3.621, which is the 28 dB front-end gain of the
6x6 mm2 SiPM in `dro_wf_generator/config/setup_config.yaml`. That generator's README and plots call its waveforms
and baseline offset mV. Against it: the template file header says "ADC units", and some of its training plots
label the same waveforms "ADC". Since the template is peak-normalised, its own unit does not matter; the unit
rests on the meaning of 3.621. Only the overall scale is affected; no result in physical units of C, S, t0 is.

## Split and filter

Split first, by original row number: rows < round((1 - split.heldout_frac) n) = 80 000 are `dev`, the rest
`heldout`. Then inside each split rows with C == 0 or S == 0 are dropped (the ratio is undefined): for
`pwo_realistic` 157 of `dev` and 32 of `heldout`, leaving 79 843 and 19 968. `rows.npy` holds the original row
numbers. Filtering never changes the random stream or moves a row between splits.

## Score (evaluate.py)

`evaluate(pred, truth)`, both (N, 3) = C, S (photons), t0 (ns), rows in the same order:
- `err68_ratio_pct` = 100 p68(|Rp - Rt| / Rt), R = C / S
- `err68_t0_ns` = p68(|t0p - t0t|)
- `n`, and `n_invalid` = events whose ratio error is +inf

p68 is `data.score.p68` (numpy's default linear interpolation). A prediction with S <= 0 or non-finite C or S has a
ratio error of +inf, a non-finite t0 a t0 error of +inf; they are kept, never dropped. The score is finite, or inf
when 32 % or more of the events are invalid; never NaN. Truth with C <= 0 or S <= 0 raises.

## Sources

- `random` (all six configs): above.
- `root` (technical path only): replace `source` by
  `{kind: root, file: <.root>, tree: tree, c_branch: <b>, c_factor: <x>, s_branch: <b>, s_factor: <x>,
  t0_branch: null}`.
  The first run.n entries are read with uproot; C = rint(c_branch * c_factor), S = rint(s_branch * s_factor)
  (round half to even), t0 from `t0_branch` or drawn as for `random`. Every other draw follows, so a `root` row is
  not the `random` row of the same number. Example for the demo file
  `/path/to/wavefm-data/dro_root_demo/merged_HCAL_DREAM_only+solenoid+ECAL_1GeV_e-.root` (20 000 entries):
  `c_branch: tot_phot_cer_ECAL_r`, `s_branch: depositedEnergyECAL_r` (GeV) with `s_factor` the scintillation
  light yield in photons per GeV.

## Known issues

- The ROOT path is a technical demonstration. The existing ROOT files have no per-channel Cherenkov count (only the
  sum over the ECAL `tot_phot_cer_ECAL_r` and per-channel deposited energy), so they are not reliable physics
  input for one channel; new ROOT files will be provided.
- The waveform unit (above).
- The crosstalk and dark-count photoelectrons are in the waveform but not in the labels, by design.
