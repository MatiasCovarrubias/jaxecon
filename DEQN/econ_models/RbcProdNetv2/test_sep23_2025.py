#!/usr/bin/env python3
"""
RBC ProdNet Model Testing Script

This script provides comprehensive testing routines for trained neural network policies
for the RBC Production Network model using the DEQN solver. It loads experiment results
and performs diagnostic tests including:

- Seed-length grid diagnostics for convergence assessment
- Simulation quality tests across different episode lengths
- Cross-seed dispersion analysis
- Stationarity and drift diagnostics
- Out-of-distribution behavior assessment

The testing framework uses grid simulation analysis to systematically evaluate
model behavior across different simulation parameters and random seeds.

Usage:
    # Method 1: Run as module (from repository root):
    python -m DEQN.econ_models.RbcProdNetv2.test_sep23_2025

    # Method 2: Run directly as script (from repository root):
    python DEQN/econ_models/RbcProdNetv2/test_sep23_2025.py

    Both methods require you to be in the repository root directory.
"""

import os
import sys

import scipy.io as sio
from jax import config as jax_config
from jax import numpy as jnp

# Add repository root to path for absolute imports when run directly
# This MUST come before any DEQN imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
model_dir = os.path.join(repo_root, "DEQN", "econ_models", "RbcProdNetv2")

# DEQN imports (use absolute imports that work both as module and script)
from DEQN.econ_models.RbcProdNetv2.plots import (  # noqa: E402
    plot_grid_test_diagnostics,
    plot_grid_test_scaling,
)
from DEQN.econ_models.RbcProdNetv2.RbcProdNet_Sept23_2025 import Model  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.tables import (  # noqa: E402
    create_grid_test_detailed_table,
    create_grid_test_summary_table,
)
from DEQN.tests.grid_simulation_analysis import (  # noqa: E402
    _print_grid_summary,
    run_seed_length_grid,
)
from DEQN.utils import load_experiment_data, load_trained_model_GPU  # noqa: E402

# Configure JAX debugging
jax_config.update("jax_debug_nans", True)


def create_test_config():
    """Create configuration for testing."""
    return {
        # Test identification
        "test_name": "baseline_diagnostics",  # Name for this specific test run
        # Experiments to test
        "experiments_to_test": {
            # "High Volatility": "baseline_nostateaug_high",
            "Baseline": "baseline_nostateaug_finetunev2",
            # "Low Volatility": "baseline_nostateaug_lower",
        },
        # Grid test configuration
        "test_lengths": [2000, 5000, 10000],  # Episode lengths to test
        "test_n_seeds": 5,  # Number of random seeds per length
        "test_burnin_fracs": [0.1, 0.2],  # Burn-in fractions to test
        "test_iact_batches": 20,  # Number of batches for IACT estimation
        # Base simulation parameters (for individual grid runs)
        "init_range": 0,
        "simul_vol_scale": 1,
        # JAX configuration
        "double_precision": True,
    }


def main():
    print("RBC Production Network Model Testing")
    print("=" * 60)

    # Configuration
    test_config = create_test_config()
    double_precision = test_config["double_precision"]
    precision = jnp.float64 if double_precision else jnp.float32

    # Configure JAX precision if needed
    if double_precision:
        jax_config.update("jax_enable_x64", True)

    # Set up paths relative to script location
    data_dir = os.path.join(model_dir, "Model_Data")
    save_dir = os.path.join(model_dir, "Experiments")
    plots_dir = os.path.join(model_dir, "Test_Plots")
    tables_dir = os.path.join(model_dir, "Test_Tables")
    model_name = "Feb21_24_baselinev3.mat"

    model_file = f"RbcProdNet_SolData_{model_name}"
    model_path = os.path.join(data_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using data directory: {data_dir}")
    print(f"Using experiments directory: {save_dir}")
    print(f"Using model file: {model_file}")

    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Get experiments to test from configuration
    experiments_to_test = test_config["experiments_to_test"]

    # Load model data and create economic model
    print("Loading economic model data...")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    # Create state_ss from k_ss and a_ss (zeros) and initialize model directly
    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    econ_model = Model(
        parameters=model_data["SolData"]["parameters"],
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=double_precision,
    )

    # Load experiment data
    print("Loading experiment data...")
    experiments_data = load_experiment_data(experiments_to_test, save_dir)

    # Define shared nn_config using model_data (features will be set per experiment)
    nn_config_base = {
        "C": model_data["SolData"]["C"],
        "policies_sd": econ_model.policies_sd,
        "params_dtype": precision,
    }

    # Run testing routines
    print("Running grid simulation analysis tests...")

    # Storage for test results
    grid_test_results = {}

    # Testing loop - run grid simulation analysis for each experiment
    for experiment_label, exp_data in experiments_data.items():
        print(f"Running grid tests for experiment: {experiment_label}")

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        # Build nn_config with experiment-specific features
        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        # Load trained model
        train_state = load_trained_model_GPU(experiment_name, save_dir, nn_config)

        # Run grid simulation analysis
        print("  Running seed-length grid diagnostics...")
        grid_results = run_seed_length_grid(
            econ_model=econ_model,
            train_state=train_state,
            base_config=test_config,
            lengths=test_config["test_lengths"],
            n_seeds=test_config["test_n_seeds"],
            burnin_fracs=test_config["test_burnin_fracs"],
            iact_num_batches=test_config["test_iact_batches"],
        )

        # Store results
        grid_test_results[experiment_label] = grid_results

        # Print summary for this experiment
        print(f"  Grid test results for {experiment_label}:")
        _print_grid_summary(grid_results)

        print(f"  Testing completed for {experiment_label}")

    print("\nAll testing completed successfully!")

    # ===================================================================
    # GENERATE TEST TABLES AND FIGURES
    # ===================================================================

    print("Generating test diagnostics tables and plots...")

    # 1. Generate summary table with key diagnostics
    print("Creating grid test summary table...")
    summary_table = create_grid_test_summary_table(
        grid_test_results=grid_test_results,
        save_path=os.path.join(tables_dir, "grid_test_summary.tex"),
        test_name=test_config["test_name"],
    )
    print("Grid Test Summary Table:")
    print(summary_table)
    print("-" * 80)

    # 2. Generate detailed table across episode lengths
    print("Creating grid test detailed table...")
    detailed_table = create_grid_test_detailed_table(
        grid_test_results=grid_test_results,
        save_path=os.path.join(tables_dir, "grid_test_detailed.tex"),
        test_name=test_config["test_name"],
    )
    print("Grid Test Detailed Table:")
    print(detailed_table)
    print("-" * 80)

    # 3. Generate SD vs T scaling plots
    print("Creating SD vs T scaling plots...")
    plot_grid_test_scaling(
        grid_test_results=grid_test_results,
        save_dir=plots_dir,
        test_name=test_config["test_name"],
    )
    print(f"SD vs T scaling plots saved to: {plots_dir}")

    # 4. Generate diagnostic plots for IACT, OOD, and trends
    diagnostic_types = ["iact", "ood", "trend"]
    for diag_type in diagnostic_types:
        print(f"Creating {diag_type.upper()} diagnostic plots...")
        plot_grid_test_diagnostics(
            grid_test_results=grid_test_results,
            diagnostic_type=diag_type,
            save_dir=plots_dir,
            test_name=test_config["test_name"],
        )
    print(f"Diagnostic plots saved to: {plots_dir}")

    # 5. Save raw grid results for further analysis (optional)
    import json

    results_file = os.path.join(tables_dir, f"grid_test_results_{test_config['test_name']}.json")

    # Convert JAX arrays to lists for JSON serialization
    serializable_results = {}
    for exp_label, grid_data in grid_test_results.items():
        serializable_results[exp_label] = {}
        for key, value in grid_data.items():
            if isinstance(key, (int, float)):
                # Length key
                serializable_results[exp_label][str(key)] = {}
                for bfrac, bfrac_data in value.items():
                    serializable_results[exp_label][str(key)][str(bfrac)] = {}
                    for metric, metric_value in bfrac_data.items():
                        if isinstance(metric_value, list):
                            serializable_results[exp_label][str(key)][str(bfrac)][metric] = metric_value
                        elif hasattr(metric_value, "tolist"):
                            serializable_results[exp_label][str(key)][str(bfrac)][metric] = metric_value.tolist()
                        else:
                            serializable_results[exp_label][str(key)][str(bfrac)][metric] = (
                                float(metric_value)
                                if isinstance(metric_value, (int, float, jnp.number))
                                else metric_value
                            )
            else:
                # Other keys like "sd_vs_T_slope"
                if isinstance(value, dict):
                    serializable_results[exp_label][key] = {
                        k: float(v) if isinstance(v, (int, float, jnp.number)) else v for k, v in value.items()
                    }
                else:
                    serializable_results[exp_label][key] = (
                        float(value) if isinstance(value, (int, float, jnp.number)) else value
                    )

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Raw grid test results saved to: {results_file}")
    print(f"Test plots saved to: {plots_dir}")
    print(f"Test tables saved to: {tables_dir}")

    print("\nTest Summary:")
    print("-" * 40)
    for exp_label, grid_data in grid_test_results.items():
        print(f"\nExperiment: {exp_label}")
        slopes = grid_data.get("sd_vs_T_slope", {})
        if slopes:
            state_slope = slopes.get("state_logsd_logT_slope", float("nan"))
            policies_slope = slopes.get("policies_logsd_logT_slope", float("nan"))
            agg_slope = slopes.get("aggregates_logsd_logT_slope", float("nan"))
            print(
                f"  SD vs T slopes (log-log): state={state_slope:.3f}, policies={policies_slope:.3f}, aggregates={agg_slope:.3f}"
            )
            print("  Expected slope for pure sampling error: ~-0.5")

            # Interpretation
            if abs(state_slope + 0.5) < 0.2:
                print("  ✓ State scaling indicates good sampling behavior")
            else:
                print("  ⚠ State scaling deviates from expected sampling behavior")

            if abs(agg_slope + 0.5) < 0.2:
                print("  ✓ Aggregate scaling indicates good sampling behavior")
            else:
                print("  ⚠ Aggregate scaling deviates from expected sampling behavior")

    return {
        "grid_test_results": grid_test_results,
        "test_config": test_config,
    }


if __name__ == "__main__":
    main()
