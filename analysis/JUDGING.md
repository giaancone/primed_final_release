# Judging a stage

Every trained stage is judged from its curves before its result may enter a summary (`rundir.accepted`). The judge
is a reviewer; the code only draws and lists (`python scripts/judge.py --pending <output>` writes
`<stage>/curves.png` and prints the table), and the judge writes the verdict:

    python scripts/judge.py <stage dir> <pass|overfit|undertrained|unclear> "<one sentence why>" --judge <who>

A verdict belongs to one `result.json` (its sha256 is recorded): if the stage is run again, it is judged again.

**What the verdicts mean.** Look at the whole curve at the length the stage was run for (default: constant rate,
fixed steps, last weights); never decide by running twice as long.
- `pass`: the training loss has flattened by the end, and the validation loss at the end is at or near its minimum.
- `undertrained`: the training loss is still clearly falling over the last part of the run.
- `overfit`: with `keep: last`, the validation loss at the end is clearly above its minimum. (With `keep: best_val`
  the weights of the minimum are kept; judge whether that minimum is a real one and not noise.)
- `unclear`: too noisy, or too few evaluation points, to tell. An unreliable conclusion is no conclusion: say
  `unclear` rather than guess, and the result stays out of the summary.

**Per stage.** Supervised fine-tuning and the student have training and validation curves. `ssl` trains on every dev
waveform and has no validation set: judge its training loss only (under-training is what can be seen there). `ridge`
is closed form and needs no verdict: read its leave-one-out table; an alpha at the edge of the grid is reported, it
is not a failure by itself.

**Hints are hints.** The table gives `final_over_min` (how far the end lies above the lowest point) and `last10_drop`
(how much the curve still fell over the last 10 % of the steps). They point the eye; they are not a threshold. As a
starting tolerance, about 3 % in either is worth a close look; calibrate the tolerance on the curves of real runs.

**What never counts.** Smoke runs are never judged into a summary (`accepted` refuses them whatever their verdict).
The formal batch of results is judged a second time, independently, and every disagreement is settled by a person.
