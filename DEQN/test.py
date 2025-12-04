#!/usr/bin/env python3
"""
Testing script for DEQN trained models.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.test

        # Method 2: Run directly as script (from repository root):
        python DEQN/test.py

        Both methods require you to be in the repository root directory.

    COLAB:
        Simply run all cells in order. The script will automatically detect the Colab
        environment, install dependencies, clone the repository, and mount Google Drive.
"""

import os
import sys

# ============================================================================
# ENVIRONMENT DETECTION AND SETUP
# ============================================================================

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    print("Installing JAX with CUDA support...")
    import subprocess

    subprocess.run(["pip", "install", "--upgrade", "jax[cuda12]"], check=True)

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    sys.path.insert(0, "/content/jaxecon")

    print("Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    base_dir = "/content/drive/MyDrive/Jaxecon/DEQN"

else:
    # Configure JAX for multi-threaded CPU execution BEFORE importing JAX
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import importlib  # noqa: E402
import json  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.tests.grid_simulation_analysis import (  # noqa: E402
    _print_grid_summary,
    run_seed_length_grid,
)
from DEQN.tests.plots import (  # noqa: E402
    plot_grid_test_diagnostics,
    plot_grid_test_scaling,
)
from DEQN.training.checkpoints import load_experiment_data, load_trained_model_GPU  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "model_dir": "RbcProdNet_Oct2025",
    "test_name": "baseline_diagnostics",
    # Experiments to test
    "experiments_to_test": {
        # "High Volatility": "baseline_nostateaug_high",
        "test": "test_local",
        # "Low Volatility": "baseline_nostateaug_lower",
    },
    # Grid test configuration
    "test_lengths": [2000, 5000, 10000],
    "test_n_seeds": 5,
    "test_burnin_fracs": [0.1, 0.2],
    "test_iact_batches": 20,
    # Base simulation parameters
    "init_range": 0,
    "simul_vol_scale": 1,
    # JAX configuration
    "double_precision": True,
}

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model

# Import model-specific table functions
tables_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.tables")
create_grid_test_detailed_table = tables_module.create_grid_test_detailed_table
create_grid_test_summary_table = tables_module.create_grid_test_summary_table


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    print(f"Testing: {config['test_name']}", flush=True)

    # Environment and precision setup
    print("Setting up precision...", flush=True)
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)
    print("Precision setup complete.", flush=True)

    model_dir = os.path.join(base_dir, config["model_dir"])
    save_dir = os.path.join(model_dir, "experiments/")
    plots_dir = os.path.join(model_dir, "test_plots/")
    tables_dir = os.path.join(model_dir, "test_tables/")

    # Load model data
    print("Loading model data...", flush=True)
    model_path = os.path.join(model_dir, "model_data.mat")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    print("Model data loaded successfully.", flush=True)
    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    # Create economic model
    print("Creating economic model...", flush=True)
    econ_model = Model(
        parameters=model_data["SolData"]["parameters"],
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=config["double_precision"],
    )
    print("Economic model created successfully.", flush=True)

    # Load experiment data
    print("Loading experiment data...", flush=True)
    experiments_to_test = config["experiments_to_test"]
    experiments_data = load_experiment_data(experiments_to_test, save_dir)
    print("Experiment data loaded successfully.", flush=True)

    # Define shared nn_config using model_data (features will be set per experiment)
    nn_config_base = {
        "C": model_data["SolData"]["C"],
        "policies_sd": model_data["SolData"]["policies_sd"],
        "params_dtype": precision,
    }

    # Storage for test results
    grid_test_results = {}

    # Testing loop
    print("Running grid simulation tests...", flush=True)
    for experiment_label, exp_data in experiments_data.items():
        print(f"  Processing: {experiment_label}", flush=True)

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        # Build nn_config with experiment-specific features
        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        # Load trained model
        train_state = load_trained_model_GPU(experiment_name, save_dir, nn_config)

        # Run grid simulation analysis
        grid_results = run_seed_length_grid(
            econ_model=econ_model,
            train_state=train_state,
            base_config=config,
            lengths=config["test_lengths"],
            n_seeds=config["test_n_seeds"],
            burnin_fracs=config["test_burnin_fracs"],
            iact_num_batches=config["test_iact_batches"],
        )

        # Store results
        grid_test_results[experiment_label] = grid_results

        # Print summary for this experiment
        _print_grid_summary(grid_results)

    print("Grid simulation tests completed successfully.", flush=True)

    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Generate tables and figures
    print("Generating test tables and figures...", flush=True)

    # Summary table
    create_grid_test_summary_table(
        grid_test_results=grid_test_results,
        save_path=os.path.join(tables_dir, "grid_test_summary.tex"),
        test_name=config["test_name"],
    )

    # Detailed table
    create_grid_test_detailed_table(
        grid_test_results=grid_test_results,
        save_path=os.path.join(tables_dir, "grid_test_detailed.tex"),
        test_name=config["test_name"],
    )

    # SD vs T scaling plots
    plot_grid_test_scaling(
        grid_test_results=grid_test_results,
        save_dir=plots_dir,
        test_name=config["test_name"],
    )

    # Diagnostic plots
    diagnostic_types = ["iact", "ood", "trend"]
    for diag_type in diagnostic_types:
        plot_grid_test_diagnostics(
            grid_test_results=grid_test_results,
            diagnostic_type=diag_type,
            save_dir=plots_dir,
            test_name=config["test_name"],
        )

    # Save raw grid results
    results_file = os.path.join(tables_dir, f"grid_test_results_{config['test_name']}.json")

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

    print("Testing completed successfully.", flush=True)

    # Print final summary
    for exp_label, grid_data in grid_test_results.items():
        slopes = grid_data.get("sd_vs_T_slope", {})
        if slopes:
            state_slope = slopes.get("state_logsd_logT_slope", float("nan"))
            policies_slope = slopes.get("policies_logsd_logT_slope", float("nan"))
            agg_slope = slopes.get("aggregates_logsd_logT_slope", float("nan"))
            print(
                f"{exp_label} - SD slopes: state={state_slope:.3f}, policies={policies_slope:.3f}, agg={agg_slope:.3f}",
                flush=True,
            )

    return {
        "grid_test_results": grid_test_results,
        "test_config": config,
    }


if __name__ == "__main__":
    main()
