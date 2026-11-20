"""Build the dev and heldout splits of a data config, or check the ones that exist.

Usage: python scripts/prepare_data.py <set> <config> [--workers N]
    e.g.  python scripts/prepare_data.py dro_gen pwo_realistic --workers 32

Existing splits are verified (digests, data hash) and left untouched; splits built from another config or other
data-set code are refused with what to do, and the exit code is 1. ``--workers`` is passed to the builder of the set
(an error if that builder has no workers).
"""
import importlib
import inspect
import json
import sys
from pathlib import Path

import wavefm.data
from data import store

USAGE = __doc__.split("\n\n")[1]


def main(argv) -> int:
    args, workers = list(argv), None
    if "--workers" in args:
        i = args.index("--workers")
        workers, args = int(args[i + 1]), args[:i] + args[i + 2:]
    if len(args) != 2:
        print(USAGE)
        return 2
    root = wavefm.data.ROOT
    cfg = store.load_config(*args, root)
    build = None
    if workers is not None:
        builder = importlib.import_module(f"data.{cfg['set']}.build").build
        if "workers" not in inspect.signature(builder).parameters:
            print(f"data/{cfg['set']}/build.py has no workers argument: run without --workers")
            return 2
        build = lambda c: builder(c, workers=workers)
    try:
        state = store.prepare(cfg, build, root)
    except (store.StaleDataError, store.DigestError) as e:
        print(f"{args[0]}/{args[1]}: {e}")
        return 1
    n = {s: json.loads((Path(cfg["out"]) / s / "meta.json").read_text())["n"] for s in store.SPLITS}
    print(f"{args[0]}/{args[1]}: {'built' if state == 'built' else 'exists, digests agree'} in {cfg['out']}: "
          + ", ".join(f"{s} {k} rows" for s, k in n.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
