"""Utilities for running DEQN entrypoints with external JSON config files."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of ``overrides`` into ``base``."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object at top level of config file: {config_path}")
    return loaded


def split_run_config(raw_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Split a run config into train overrides, analysis overrides, and run flag."""
    train_overrides = raw_config.get("train")
    analysis_overrides = raw_config.get("analysis")
    run_analysis = bool(raw_config.get("run_analysis", analysis_overrides is not None))

    if train_overrides is None:
        train_overrides = {
            key: value
            for key, value in raw_config.items()
            if key not in {"analysis", "run_analysis"}
        }

    if not isinstance(train_overrides, Mapping):
        raise ValueError("The 'train' section must be a JSON object.")
    if analysis_overrides is not None and not isinstance(analysis_overrides, Mapping):
        raise ValueError("The 'analysis' section must be a JSON object.")

    return dict(train_overrides), dict(analysis_overrides) if analysis_overrides is not None else None, run_analysis
