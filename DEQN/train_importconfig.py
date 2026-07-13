#!/usr/bin/env python3
"""Run DEQN training with overrides loaded from a JSON config file."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from DEQN.import_config import deep_merge, load_json_config, split_run_config  # noqa: E402
from DEQN.econ_models import load_model_class  # noqa: E402


def _write_temp_analysis_config(analysis_overrides: dict) -> Path:
    import json

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="deqn_analysis_",
        delete=False,
        encoding="utf-8",
    )
    with temp_file:
        json.dump(analysis_overrides, temp_file, indent=2)
        temp_file.write("\n")
    return Path(temp_file.name)


def _configure_train_module(train_module, overrides: dict) -> None:
    train_module.config = deep_merge(train_module.config, overrides)
    train_module._set_derived_training_config(train_module.config)

    train_module.Model = load_model_class(
        train_module.config["model_dir"],
        train_module.config.get("exact_cobb_douglas", False),
    )
    train_module.analysis_hooks = train_module.load_model_analysis_hooks(
        train_module.config["model_dir"]
    )


def _run_analysis(analysis_config_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "DEQN.analysis_importconfig",
        "--config",
        str(analysis_config_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON run config.")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not run analysis even if the config has an analysis section.",
    )
    args = parser.parse_args()

    raw_config = load_json_config(args.config)
    train_overrides, analysis_overrides, run_analysis = split_run_config(raw_config)

    import DEQN.train as train_module

    _configure_train_module(train_module, train_overrides)
    result = train_module.main()

    if result is None:
        raise RuntimeError("Training did not return a result; skipping analysis.")

    if args.skip_analysis or not run_analysis:
        return

    if analysis_overrides is None:
        experiment_name = train_module.config["exper_name"]
        analysis_overrides = {
            "model_dir": train_module.config["model_dir"],
            "exact_cobb_douglas": train_module.config.get("exact_cobb_douglas", False),
            "analysis_name": experiment_name,
            "model_data_file": train_module.config.get("model_data_file"),
            "experiment_to_analyze": {experiment_name: experiment_name},
        }

    analysis_config_path = _write_temp_analysis_config(analysis_overrides)
    try:
        _run_analysis(analysis_config_path)
    finally:
        try:
            os.remove(analysis_config_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
