"""Plain-YAML config with 'extends' merging and dot-path CLI overrides.

A config file may list other files under `extends:` (paths relative to the
file); they are deep-merged in order, own keys win. Overrides look like
`trainer.lr=1e-3`; values are parsed as YAML scalars.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _coerce(value: Any) -> Any:
    """YAML 1.1 parses `1e-4` as a string (it wants `1.0e-4`), which then flows
    into e.g. the optimiser lr as text. Recover any numeric-looking string."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict[str, Any] = {}
    for parent in raw.pop("extends", []):
        merged = _deep_merge(merged, load_config(path.parent / parent))
    merged = _deep_merge(merged, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = merged
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(yaml.safe_load(value))
    return merged
