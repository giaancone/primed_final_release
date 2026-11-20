"""YAML configs with ``defaults`` inheritance and ``section.key=value`` command-line overrides."""
import copy
from pathlib import Path

import yaml


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path) -> dict:
    """Load a YAML file; a top-level ``defaults: other.yaml`` is loaded first (relative path) and overridden.
    ``dynamics`` is dropped: it is the YAML template of the stage sections, used up when the file is parsed, so
    setting it later would change nothing (and check() rejects it as an unknown key)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("defaults", None)
    cfg.pop("dynamics", None)
    return merge(load(path.parent / parent), cfg) if parent else cfg


class ConfigError(KeyError):
    pass


STAGES = ("finetune", "ssl", "ridge", "student", "compress")


def check(cfg: dict, base: dict | None = None):
    """Every key of ``cfg`` exists in configs/teacher.yaml and every stage is known: a typo is an error, never a
    silent default. Call it on every config before it is used."""
    if base is None:
        from wavefm import paths
        base = load(paths.ROOT / "configs" / "teacher.yaml")

    def walk(c, b, where):
        for k, v in c.items():
            if k not in b:
                raise ConfigError(f"unknown config key {where}{k}: every key and its default is in configs/teacher.yaml")
            if isinstance(b[k], dict) != isinstance(v, dict):
                raise ConfigError(f"config key {where}{k} is a section in one place and a value in the other")
            if isinstance(v, dict):
                walk(v, b[k], f"{where}{k}.")

    walk(cfg, base, "")
    unknown = [s for s in cfg.get("stages", []) if s not in STAGES]
    if unknown or not cfg.get("stages"):
        raise ConfigError(f"stages {cfg.get('stages')}: need at least one, each one of {STAGES}")


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply ``a.b=value`` items. The key must exist in cfg (typos are errors) and the value is converted to the
    type of the existing value: "3.0" stays a string for model.version, "1e-3" becomes a float for a float key."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ConfigError(f"override {item!r} is not key=value")
        *path, last = key.split(".")
        d = cfg
        for k in path:
            if not isinstance(d.get(k), dict):
                raise ConfigError(f"unknown config section {key!r}")
            d = d[k]
        if last not in d:
            raise ConfigError(f"unknown config key {key!r}")
        d[last] = _typed(val, d.get(last), key)
    return cfg


def _typed(val: str, old, key):
    if isinstance(old, bool):
        if val.lower() not in ("true", "false"):
            raise ConfigError(f"{key} needs true or false, got {val!r}")
        return val.lower() == "true"
    if isinstance(old, (int, float)):
        try:
            x = float(val)
        except ValueError:
            raise ConfigError(f"{key} needs a number, got {val!r}") from None
        if isinstance(old, int):
            if not x.is_integer():
                raise ConfigError(f"{key} needs an integer, got {val!r}")
            return int(x)
        return x
    if isinstance(old, str):
        return val
    return yaml.safe_load(val)
