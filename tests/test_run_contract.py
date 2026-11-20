"""The contract between stages (train/rundir.py) and of the config keys (configs/teacher.yaml): write, read back,
and every way a reader must refuse."""
import json

import numpy as np
import pytest

import fakes
from data import store
from wavefm import paths
from wavefm.train import rundir
from wavefm.utils import config

STAGES = ("finetune", "ssl", "ridge", "student", "compress")


def base():
    return config.load(paths.ROOT / "configs" / "teacher.yaml")


def test_teacher_yaml_has_every_stage_and_rejects_typos():
    cfg = base()
    assert set(STAGES) <= cfg.keys() and {"task", "seed", "run", "data", "stages", "model"} <= cfg.keys()
    assert STAGES == config.STAGES and "dynamics" not in cfg       # the template is used up when the file is parsed
    assert cfg["run"]["smoke"] is False and set(cfg["stages"]) <= set(STAGES)
    for s in ("finetune", "ssl", "student"):                       # the plainest training is the default
        assert (cfg[s]["schedule"], cfg[s]["keep"], cfg[s]["warmup_frac"]) == ("constant", "last", 0.0)
    config.check(cfg)
    for bad in ({"ssl": {"lr_tokeniser": 1e-4}}, {"sll": {}}, {"stages": ["ssl", "rigde"]}, {"data": {"name": "x"}}):
        with pytest.raises(config.ConfigError):
            config.check(config.merge(cfg, bad))
    for override in ("ridge.alpha=1", "dynamics.batch_size=8"):    # the second would silently change nothing
        with pytest.raises(config.ConfigError):
            config.apply_overrides(cfg, [override])
    for f in sorted((paths.ROOT / "configs").glob("*.yaml")) + sorted((paths.ROOT / "configs").glob("paper/*.yaml")):
        config.check(config.load(f))                               # every shipped config uses known keys only


def _result(out, cfg, stage="ssl", parent=None, data_hash="d" * 64, artifacts=()):
    out.mkdir(parents=True, exist_ok=True)
    return rundir.write_result(out, stage, cfg, data_hash, parent, requested={"steps": 20, "epochs": 0.1, "samples": 320},
                               actual={"steps": 20, "epochs": 0.1, "samples": 320}, hints={"last_over_min": 1.0},
                               wall_s=1.5, artifacts=artifacts, loss_next=0.1)


def test_result_identity(tmp_path):
    cfg = base()
    w = tmp_path / "w.pt"; w.write_bytes(b"weights")
    parent = rundir.parent_record(w)
    assert rundir.check_identity(tmp_path / "ssl", cfg, "d" * 64, parent) is None            # not finished
    r = _result(tmp_path / "ssl", cfg, parent=parent)
    assert rundir.RESULT_KEYS <= r.keys() and r["loss_next"] == 0.1 and r["parent"]["sha256"] == rundir.file_sha256(w)
    assert rundir.read_result(tmp_path / "ssl") == json.loads((tmp_path / "ssl" / "result.json").read_text())
    same = config.merge(cfg, {"run": {"name": "other", "group": "g", "deadline": 1.7e9}, "student": {"lr": 0.5},
                              "ridge": {"block": 3}, "stages": ["ssl", "ridge", "student"]})
    assert rundir.check_identity(tmp_path / "ssl", same, "d" * 64, parent) == r    # names, deadline, other stages
    assert rundir.config_fp(same) != rundir.config_fp(cfg)                          # but another run as a whole
    assert rundir.config_fp(same, "ssl") == rundir.config_fp(cfg, "ssl") != rundir.config_fp(cfg, "ridge")
    w2 = tmp_path / "w2.pt"; w2.write_bytes(b"other weights")
    for c, d, p in ((config.merge(cfg, {"ssl": {"batch_size": 8}}), "d" * 64, parent), (cfg, "e" * 64, parent),
                    (cfg, "d" * 64, rundir.parent_record(w2)), (cfg, "d" * 64, None),
                    (config.merge(cfg, {"run": {"smoke": True}}), "d" * 64, parent),
                    (config.merge(cfg, {"model": {"adapter": True}}), "d" * 64, parent), ({**cfg, "seed": 1}, "d" * 64, parent)):
        with pytest.raises(rundir.RunError, match="another"):
            rundir.check_identity(tmp_path / "ssl", c, d, p)
    with pytest.raises(rundir.RunError, match="reserved"):
        rundir.write_result(tmp_path / "ssl", "ssl", cfg, "d", None, requested={}, actual={}, hints={}, wall_s=0, smoke=False)
    with pytest.raises(rundir.RunError, match="directory of its name"):
        rundir.write_result(tmp_path / "ssl", "ridge", cfg, "d", None, requested={}, actual={}, hints={}, wall_s=0)
    (tmp_path / "ridge").mkdir()                                  # a result copied into another stage's directory
    (tmp_path / "ridge" / "result.json").write_bytes((tmp_path / "ssl" / "result.json").read_bytes())
    for refused in (rundir.read_result, rundir.accepted, lambda o: rundir.check_identity(o, cfg, "d" * 64, parent)):
        with pytest.raises(rundir.RunError, match="copied or moved"):
            refused(tmp_path / "ridge")
    (tmp_path / "old").mkdir(); (tmp_path / "old" / "result.json").write_text('{"stage": "ssl"}')
    with pytest.raises(rundir.RunError, match="lacks"):
        rundir.read_result(tmp_path / "old")
    with pytest.raises(rundir.RunError, match="did not finish"):
        rundir.read_result(tmp_path / "none")


def test_verdict_gate(tmp_path):
    """Only a finished, non-smoke stage judged "pass" on its present results is accepted; ridge needs no verdict."""
    cfg, out = base(), tmp_path / "ssl"
    assert not rundir.accepted(out)
    _result(out, cfg)
    assert not rundir.accepted(out)                                                           # no verdict
    for v in ("overfit", "undertrained", "unclear"):
        rundir.write_verdict(out, v, "validation loss rises from step 10", "reviewer")
        assert not rundir.accepted(out)
    v = rundir.write_verdict(out, "pass", "both curves flat over the last third", "reviewer")
    assert rundir.accepted(out) and set(v) == {"verdict", "reason", "judge", "result_sha256"}
    _result(out, cfg)                                             # same content: the verdict still holds
    assert rundir.accepted(out)
    rundir.write_result(out, "ssl", cfg, "d" * 64, None, requested={}, actual={}, hints={}, wall_s=9.0)
    assert not rundir.accepted(out)                               # results changed after the verdict
    for bad in (("good", "r", "j"), ("pass", " ", "j"), ("pass", "r", "")):
        with pytest.raises(rundir.RunError):
            rundir.write_verdict(out, *bad)
    smoke = tmp_path / "smoke_run" / "ssl"
    _result(smoke, config.merge(cfg, {"run": {"smoke": True}}))
    rundir.write_verdict(smoke, "pass", "fine", "reviewer")
    assert not rundir.accepted(smoke)                             # a smoke run never enters a summary
    ridge = tmp_path / "ridge"; ridge.mkdir()
    with pytest.raises(rundir.RunError, match="did not write"):
        _result(ridge, cfg, stage="ridge", artifacts=["teacher_labels.npz"])
    (ridge / "teacher_labels.npz").write_bytes(b"0123456789")
    r = _result(ridge, cfg, stage="ridge", artifacts=["teacher_labels.npz"])
    assert r["artifacts"]["teacher_labels.npz"] == {"sha256": rundir.file_sha256(ridge / "teacher_labels.npz"), "bytes": 10}
    assert rundir.accepted(ridge)                                 # closed form: no verdict asked for
    (ridge / "teacher_labels.npz").write_bytes(b"01234")
    assert not rundir.accepted(ridge)                             # a promised file changed
    (ridge / "teacher_labels.npz").unlink()
    assert not rundir.accepted(ridge)                             # or is gone


def test_teacher_labels(tmp_path):
    cfg = base()
    dev = store.Split(fakes.make_split(tmp_path, "npy")[0])
    teacher = _result(tmp_path / "ridge", cfg, stage="ridge", data_hash=dev.meta["data_hash"])
    pred = dev.labels.astype(np.float64) + 0.25
    f = tmp_path / "ridge" / "teacher_labels.npz"
    rundir.write_teacher_labels(f, dev, pred, teacher)
    np.testing.assert_array_equal(rundir.read_teacher_labels(f, dev), pred)
    with np.load(f) as z:
        meta = json.loads(str(z["meta"]))
        np.testing.assert_array_equal(z["rows"], dev.rows)
    assert meta["labels"] == dev.meta["labels"] and meta["data_hash"] == dev.meta["data_hash"] and meta["content"] == dev.content
    assert meta["teacher"] == {"config_fp": teacher["config_fp"], "parent": None}
    for bad in (pred[:-1], pred[:, :2], np.where(np.arange(len(pred))[:, None] == 3, np.nan, pred)):
        with pytest.raises(rundir.RunError):
            rundir.write_teacher_labels(tmp_path / "bad.npz", dev, bad, teacher)
    heldout = store.Split(fakes.make_split(tmp_path / "h", "npy", "heldout")[0])
    for d, t in ((heldout, teacher), (dev, {**teacher, "data_hash": "0" * 64})):
        with pytest.raises(rundir.RunError, match="dev split the teacher was fitted on"):
            rundir.write_teacher_labels(tmp_path / "bad.npz", d, pred, t)
    with pytest.raises(rundir.RunError, match="another build"):   # same config and code, other content
        rundir.read_teacher_labels(f, heldout)

    def rewrite(**change):
        with np.load(f) as z:
            a = {"rows": z["rows"], "pred": z["pred"], "meta": json.loads(str(z["meta"]))}
        a.update({k: v(a[k]) for k, v in change.items()})
        np.savez(tmp_path / "t.npz", rows=a["rows"], pred=a["pred"], meta=json.dumps(a["meta"]))
        return tmp_path / "t.npz"

    units = lambda m: {**m, "labels": [{**m["labels"][0], "unit": "mV"}] + m["labels"][1:]}
    order = lambda m: {**m, "labels": m["labels"][::-1]}
    for change in (dict(rows=lambda r: r + 1), dict(rows=lambda r: r[::-1].copy()), dict(pred=lambda p: p[:, :2]),
                   dict(meta=units), dict(meta=order), dict(meta=lambda m: {**m, "data_hash": "0" * 64}),
                   dict(meta=lambda m: {**m, "split": "heldout"}), dict(meta=lambda m: {**m, "content": "0" * 64}),
                   dict(pred=lambda q: q.astype(np.float32)), dict(pred=lambda q: np.where(q > 1e9, q, np.inf))):
        with pytest.raises(rundir.RunError):
            rundir.read_teacher_labels(rewrite(**change), dev)


SPEC = {"sizes": [8, 4, 3], "activation": "relu", "labels": fakes.LABELS, "data_hash": "d" * 64,
        "input": {"length": 30, "pad_to": 32, "every": 5, "lo": -2.0, "hi": 6.0},
        "output": {"scale": [433.0, 260.0, 1.44], "offset": [750.0, 450.0, 0.0]}}


def test_student_file(tmp_path):
    rng = np.random.default_rng(0)
    W = [rng.normal(size=(4, 8)).astype(np.float32), rng.normal(size=(3, 4)).astype(np.float32)]
    B = [rng.normal(size=4).astype(np.float32), rng.normal(size=3).astype(np.float32)]
    f = tmp_path / "student.npz"
    rundir.write_student(f, W, B, SPEC)
    w, b, spec = rundir.read_student(f, "d" * 64)
    with pytest.raises(rundir.RunError, match="another data set"):
        rundir.read_student(f, "e" * 64)
    assert spec == SPEC and all((x == y).all() for x, y in zip(w + b, W + B))

    x = np.arange(30, dtype=np.float32)[None] + np.array([[0.0], [100.0]], np.float32)       # an increasing ramp
    kept = rundir.student_input(x, SPEC)
    want = np.array([0, 5, 10, 15, 20, 25, 29, 29], np.float32)   # 30 -> 32 by the last sample, every 5th: 7, pad to 8
    np.testing.assert_allclose(kept, (want[None] + np.array([[0.0], [100.0]]) + 2.0) / 8.0, rtol=1e-6)
    h = np.maximum(kept @ W[0].T + B[0], 0) @ W[1].T + B[1]
    np.testing.assert_allclose(rundir.student_predict(w, b, spec, x), h * SPEC["output"]["scale"] + SPEC["output"]["offset"], rtol=1e-6)
    with pytest.raises(rundir.RunError, match="30 samples"):
        rundir.student_predict(w, b, spec, x[:, :29])

    inp = SPEC["input"]
    for bad_spec in ({**SPEC, "sizes": [8, 5, 3]}, {k: v for k, v in SPEC.items() if k != "output"},
                     {k: v for k, v in SPEC.items() if k != "data_hash"}, {**SPEC, "activation": "gelu"},
                     {**SPEC, "labels": fakes.LABELS[:2]}, {**SPEC, "input": {"length": 30, "every": 5}},
                     {**SPEC, "input": {**inp, "hi": -2.0}}, {**SPEC, "input": {**inp, "every": 0}},
                     {**SPEC, "input": {**inp, "pad_to": 29}}, {**SPEC, "input": {**inp, "every": 2}},
                     {**SPEC, "input": {**inp, "lo": float("nan")}}, {**SPEC, "output": {"scale": [1.0, 2.0, float("inf")], "offset": [0.0] * 3}}):
        with pytest.raises(rundir.RunError):
            rundir.write_student(tmp_path / "bad.npz", W, B, bad_spec)
        np.savez(tmp_path / "t.npz", spec=json.dumps(bad_spec), w0=W[0], w1=W[1], b0=B[0], b1=B[1])
        with pytest.raises(rundir.RunError):                      # the reader refuses what the writer refuses
            rundir.read_student(tmp_path / "t.npz", "d" * 64)
    with pytest.raises(rundir.RunError):
        rundir.write_student(tmp_path / "bad.npz", [W[0].T, W[1]], B, SPEC)                   # Keras layout
    for arrays in (dict(w0=W[0].T, w1=W[1], b0=B[0], b1=B[1]), dict(w0=W[0], w1=W[1], b0=B[0], b1=B[1][:1]),
                   dict(w0=W[0], w1=W[1], b0=B[0].astype(np.float64), b1=B[1]), dict(w0=W[0], w1=W[1], b0=B[0]),
                   dict(w0=W[0] * np.nan, w1=W[1], b0=B[0], b1=B[1])):
        np.savez(tmp_path / "t.npz", spec=json.dumps(SPEC), **arrays)
        with pytest.raises(rundir.RunError):
            rundir.read_student(tmp_path / "t.npz", "d" * 64)
