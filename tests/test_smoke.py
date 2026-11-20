"""The checks of scripts/smoke.py fed wrong values, without training: a NaN reload loss, two reload processes with
different predictions, a heldout record of another stage, of a non-smoke result or of another result."""
import importlib.util

import pytest

from wavefm import paths
from wavefm.train import rundir
from wavefm.utils.io import save_json

GOOD = {"sha256": "a", "val_loss": 1.0}
NAN = float("nan")


def smoke():
    spec = importlib.util.spec_from_file_location("smoke_script", paths.ROOT / "scripts" / "smoke.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reload_check():
    m = smoke()
    m.check_reload(GOOD, dict(GOOD), 1.0 + 1e-6, "x")
    for a, b, recorded, match in ((dict(GOOD, val_loss=NAN), GOOD, 1.0, "non-finite"),
                                  (GOOD, dict(GOOD, val_loss=NAN), 1.0, "non-finite"),
                                  (GOOD, GOOD, NAN, "non-finite"),
                                  (GOOD, dict(GOOD, sha256="b"), 1.0, "different predictions"),
                                  (GOOD, GOOD, 1.1, "reloaded validation loss")):
        with pytest.raises(m.SmokeError, match=match):
            m.check_reload(a, b, recorded, "x")


@pytest.mark.parametrize("stage, where", [("student", "heldout.json"), ("compress", "compress/heldout.json")])
def test_heldout_check(tmp_path, stage, where):
    m, run = smoke(), tmp_path / "run"
    for s in ("student", "compress"):
        (run / s).mkdir(parents=True)
        (run / s / "result.json").write_text(f'{{"stage": "{s}"}}')
    sha = rundir.file_sha256(run / stage / "result.json")
    record = {"metrics": {"m": 1.0}, "stage": stage, "n": 3, "result": {"smoke": True, "sha256": sha}}
    save_json(record, run / where)
    assert m.check_heldout(run, run / where, stage) == record
    other = "compress" if stage == "student" else "student"
    for wrong in ({"stage": other}, {"result": {"smoke": False, "sha256": sha}},
                  {"result": {"smoke": True, "sha256": rundir.file_sha256(run / other / "result.json")}}):
        save_json({**record, **wrong}, run / where)
        with pytest.raises(m.SmokeError, match="not the present"):
            m.check_heldout(run, run / where, stage)
