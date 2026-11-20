Development scans live here, apart from `configs/paper/`. A scan (`python scripts/run_scan.py <file>`, format in the
docstring of `scripts/run_scan.py`) is the evidence for the fixed defaults of an implementation, such as the learning
rate and length of a trained stage: it is run once, on one seed, judged from its curves (`analysis/JUDGING.md`), and
its result is written into the defaults. It is development, not part of the method: whoever uses the method neither
needs nor should repeat it, and a scan is done again only when the recipe changes in substance (preprocessing, loss,
scaling, data, structure). Scans never open `heldout`.
