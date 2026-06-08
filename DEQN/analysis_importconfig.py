#!/usr/bin/env python3
"""Run DEQN analysis with overrides loaded from a JSON config file."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from DEQN.import_config import deep_merge, load_json_config  # noqa: E402


def _load_analysis_script_module():
    analysis_path = Path(__file__).resolve().with_name("analysis.py")
    spec = importlib.util.spec_from_file_location("_deqn_analysis_script", analysis_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load analysis script from {analysis_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_analysis_module(analysis_module, overrides: dict) -> None:
    analysis_module.config = deep_merge(analysis_module.config, overrides)

    model_module = importlib.import_module(
        f"DEQN.econ_models.{analysis_module.config['model_dir']}.model"
    )
    analysis_module.Model = model_module.Model
    analysis_module.analysis_hooks = analysis_module.load_model_analysis_hooks(
        analysis_module.config["model_dir"]
    )
    analysis_module.analysis_reporting.analysis_hooks = analysis_module.analysis_hooks
    analysis_module.config = analysis_module.apply_model_config_defaults(
        analysis_module.config,
        analysis_module.analysis_hooks,
    )

    plots_module_name = f"DEQN.econ_models.{analysis_module.config['model_dir']}.plots"
    try:
        plots_module = importlib.import_module(plots_module_name)
    except ModuleNotFoundError as exc:
        if exc.name == plots_module_name:
            plots_module = None
        else:
            raise
    analysis_module.MODEL_SPECIFIC_PLOTS = (
        getattr(plots_module, "MODEL_SPECIFIC_PLOTS", [])
        if plots_module is not None
        else []
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON analysis config.")
    args = parser.parse_args()

    raw_config = load_json_config(args.config)
    overrides = raw_config.get("analysis", raw_config)
    if not isinstance(overrides, dict):
        raise ValueError("The 'analysis' section must be a JSON object.")

    analysis_module = _load_analysis_script_module()

    _configure_analysis_module(analysis_module, overrides)
    analysis_module.main()


if __name__ == "__main__":
    main()
