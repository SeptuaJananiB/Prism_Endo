"""YAML configuration loading with deep-merge and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` recursively merged into `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    # yaml.safe_load gives ints, floats, bools, null and lists for free.
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply `key.sub=value` overrides, e.g. ``train.epochs=3``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.strip().split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override '{dotted}': '{k}' is not a mapping")
        node[keys[-1]] = _parse_scalar(raw)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config. A top-level ``_base_`` key names a file (relative
    to this one) whose contents are merged underneath."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        cfg = deep_merge(load_config(path.parent / base_name), cfg)
    return apply_overrides(cfg, overrides)


def resolve_path(p: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root."""
    if p is None or p == "":
        return None
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
