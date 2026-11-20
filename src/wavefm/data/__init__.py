"""The two splits of the data set a run names, as ``data.store.Split``: ``dev`` for every method, ``heldout`` for the
final evaluation only.

``dev(cfg)`` opens ``<out>/dev`` of ``cfg["data"]["set"]`` / ``["config"]`` after checking that it was built from the
present data config and data-set code (the data hash): a result on a data set that has changed since is not
comparable, so it is refused (StaleDataError), never silently used.

``heldout(cfg)`` is locked. A method never sees heldout, not even its statistics (meta.json): every choice is made on
dev. Only ``scripts/evaluate.py`` opens it, once per run, inside ``with final_evaluation():``; anywhere else
heldout() raises HeldoutError, and no other code in wavefm opens ``<out>/heldout``. (The compressed model is scored
by compression/compress.py --evaluate in its own environment, under the same rules.)

``ROOT`` is the data package root; tests point it at a temporary data root (tests/conftest.py). There is no other
way to change where data sets are found.
"""
from contextlib import contextmanager
from pathlib import Path

from data import store

ROOT = store.ROOT
_final = False


class HeldoutError(RuntimeError):
    pass


def _open(cfg: dict, split: str) -> store.Split:
    d = cfg["data"]
    data_cfg = store.load_config(d["set"], d["config"], ROOT)
    s = store.Split(Path(data_cfg["out"]) / split)
    if s.meta["data_hash"] != store.data_hash(data_cfg, ROOT):
        raise store.StaleDataError(
            f"{s.path} was built from another config or other data-set code than the present data/{d['set']}: run "
            f"`python scripts/prepare_data.py {d['set']} {d['config']}` (it says what to move away), or restore them")
    return s


def dev(cfg: dict) -> store.Split:
    return _open(cfg, "dev")


@contextmanager
def final_evaluation():
    """The one context in which heldout() may be called: scripts/evaluate.py, after the method is fixed."""
    global _final
    _final = True
    try:
        yield
    finally:
        _final = False


def heldout(cfg: dict) -> store.Split:
    if not _final:
        raise HeldoutError("heldout is opened only by scripts/evaluate.py, once, after every choice was made on dev: "
                           "use wavefm.data.dev(cfg) and a validation set inside the label budget instead")
    return _open(cfg, "heldout")
