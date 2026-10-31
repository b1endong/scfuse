import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _env(m: re.Match) -> str:
    value = os.environ.get(m.group(1), m.group(2))
    if value is None:
        raise KeyError(f"environment variable {m.group(1)} is not set and has no default")
    return value


def _expand(v):
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return _ENV.sub(_env, v) if isinstance(v, str) else v


def _raw(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = _merge(_raw(path.parent / cfg.pop("base")), cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _raw(Path(path))
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return _expand(cfg)


def run_name(cfg: dict, seed: int | None = None) -> str:
    return f"{cfg['run']['name']}_s{cfg['seed'] if seed is None else seed}"
