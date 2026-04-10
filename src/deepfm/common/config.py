from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    base_entries = config.pop("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]

    merged: dict[str, Any] = {}
    for entry in base_entries:
        base_path = (config_path.parent / entry).resolve()
        merged = _merge_dicts(merged, load_config(base_path))

    merged = _merge_dicts(merged, config)
    merged.setdefault("meta", {})
    merged["meta"]["config_path"] = str(config_path)
    return merged
