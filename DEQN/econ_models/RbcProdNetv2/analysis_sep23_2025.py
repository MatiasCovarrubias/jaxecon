#!/usr/bin/env python3
"""
RBC ProdNet Model Analysis Script

This script provides a general analysis framework for trained neural network policies
for the RBC Production Network model using the DEQN solver. It loads experiment results
and performs comprehensive analyses including:

- Simulation statistics for all states, policies, and aggregates
- Stochastic steady state calculations for all model variables
- Comparative analysis across different experiments
- Optional seed-length grid diagnostics for convergence assessment

The analysis is general and relies on the economic model's get_aggregates function
to compute all relevant aggregate variables rather than focusing on specific measures.

Converted from RbcProdNet_Analysis_Sep23_2025.ipynb for local development.

Usage:
    # Method 1: Run as module (from repository root):
    python -m DEQN.econ_models.RbcProdNetv2.RbcProdNet_Analysis_Sep23_2025

    # Method 2: Run directly as script (from repository root):
    python DEQN/econ_models/RbcProdNetv2/RbcProdNet_Analysis_Sep23_2025.py

    Both methods require you to be in the repository root directory.
"""

import os
import sys

import jax
import scipy.io as sio
from jax import config as jax_config
from jax import numpy as jnp
from jax import random
from scipy.stats import kurtosis, skew

# Add repository root to path for absolute imports when run directly
# This MUST come before any DEQN imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# DEQN imports (use absolute imports that work both as module and script)
from DEQN.algorithm.simulation import create_episode_simulation_fn_verbose  # noqa: E402
from DEQN.analysis.stochastic_ss import create_stochss_fn  # noqa: E402
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.plots import plot_sectoral_capital_mean  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.RbcProdNet_Sept23_2025 import Model  # noqa: E402
from DEQN.tests.grid_simulation_analysis import (  # noqa: E402
    _print_grid_summary,
    run_seed_length_grid,
)
from DEQN.utils import load_experiment_data, load_trained_model_GPU  # noqa: E402

# Configure JAX debugging
jax_config.update("jax_debug_nans", True)


def create_analysis_config():
    """Create configuration for analysis."""
    return {
        # Simulation configuration
        "init_range": 0,
        "periods_per_epis": 6000,
        "burn_in_periods": 1000,
        "simul_vol_scale": 1,
        "simul_seed": 0,
        # Stochastic steady state configuration
        "n_draws": 500,
        "time_to_converge": 200,
        "seed": 0,
        # Welfare configuration
        "welfare_n_trajects": 100,
        "welfare_traject_length": 500,
        "welfare_seed": 0,
        # JAX configuration
        "double_precision": True,
        # Grid diagnostics configuration
        "run_seed_length_grid": True,
        "grid_lengths": [2000, 5000, 10000],
        "grid_n_seeds": 16,
        "grid_burnin_fracs": [0.2],
        "grid_experiment_labels": ["baseline"],
        "iact_num_batches": 20,
    }


def main():
    model_dir = os.path.join(repo_root, "DEQN", "econ_models", "RbcProdNetv2")
    print("RBC Production Network Model Analysis")
    print("=" * 60)

    # Configuration
    analysis_config = create_analysis_config()
    double_precision = analysis_config["double_precision"]
    precision = jnp.float64 if double_precision else jnp.float32

    # Configure JAX precision if needed
    if double_precision:
        jax_config.update("jax_enable_x64", True)

    # Set up paths relative to script location
    data_dir = os.path.join(model_dir, "Model_Data")
    save_dir = os.path.join(model_dir, "Experiments")
    plots_dir = os.path.join(model_dir, "Plots")
    model_name = "Feb21_24_baselinev3.mat"

    model_file = f"RbcProdNet_SolData_{model_name}"
    model_path = os.path.join(data_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using data directory: {data_dir}")
    print(f"Using experiments directory: {save_dir}")
    print(f"Using model file: {model_file}")

    # Define experiments to analyze
    experiments_to_analyze = {
        # "High Volatility": "baseline_nostateaug_high",
        "Baseline": "baseline_nostateaug_finetunev2",
        # "Low Volatility": "baseline_nostateaug_lower",
    }

    # Load model data and create economic model
    print("Loading economic model data...")
    model_data = sio.loadmat(model_path, simplify_cells=True)

    econ_model = Model(
        parameters=model_data["SolData"]["parameters"],
        state_ss=model_data["SolData"]["state_ss"],
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=double_precision,
    )

    # Load experiment data
    print("Loading experiment data...")
    experiments_data = load_experiment_data(experiments_to_analyze, save_dir)

    # Define shared nn_config using model_data (features will be set per experiment)
    nn_config_base = {
        "C": model_data["SolData"]["C"],
        "policies_sd": econ_model.policies_sd,
        "params_dtype": precision,
    }

    # Run comparative analysis (integrated into main)
    print("Running comparative analysis across experiments...")

    # Create simulation and stochastic steady state functions once
    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, analysis_config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, analysis_config))
    welfare_fn = jax.jit(get_welfare_fn(econ_model, analysis_config))

    analysis_results = {}

    for experiment_label, exp_data in experiments_data.items():
        print(f"Analyzing experiment: {experiment_label}")

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        # Build nn_config with experiment-specific features
        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        # Load trained model
        train_state = load_trained_model_GPU(experiment_name, save_dir, nn_config)

        # Generate simulation data using verbose simulation (one long episode)
        print(f"  Generating simulation data for {experiment_label}...")
        episode_rng = random.PRNGKey(analysis_config["simul_seed"])
        simul_state, simul_policies = simulation_fn(train_state, episode_rng)
        # Extract burn-in period
        simul_state = simul_state[analysis_config["burn_in_periods"] :]
        simul_policies = simul_policies[analysis_config["burn_in_periods"] :]

        # Test 1: the last observation of the simulation should be between -10 and 10
        max_dev = jnp.max(jnp.abs(simul_state[-1, :]))
        print(f"The max standadized dev in last obs of simulation is: {max_dev:.6f}")
        assert max_dev < 10, f"Last observation too large: {max_dev:.6f}"

        # Get mean states and policies from ergodic distribution to construct aggregates.
        simul_state_mean = jnp.mean(simul_state, axis=0)
        simul_policies_mean = jnp.mean(simul_policies, axis=0)
        P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
        Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
        Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

        # Get aggregates for simulation data
        simul_aggregates = jax.vmap(econ_model.get_aggregates, in_axes=(0, 0, None, None, None))(
            simul_state, simul_policies, P_mean, Pk_mean, Pm_mean
        )

        simul_aggregates_descstats = {
            "mean": jnp.mean(simul_aggregates, axis=0).tolist(),
            "std": jnp.std(simul_aggregates, axis=0).tolist(),
            "skewness": skew(simul_aggregates, axis=0).tolist(),
            "kurtosis": kurtosis(simul_aggregates, axis=0).tolist(),
        }

        # Calculate stochastic steady state from simulation data
        print(f"  Calculating stochastic steady state for {experiment_label}...")
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_state, train_state)

        # Test 2: Assert that stochastic steady state standard deviation is close to zero
        max_std = jnp.max(stoch_ss_obs_std)
        print(f"The max std is:{max_std:.6f}")
        assert max_std < 0.01, f"Stochastic steady state std too large: {max_std:.6f}"

        # Get stochastic steady state aggregates
        stoch_ss_aggregates = econ_model.get_aggregates(stoch_ss_obs, stoch_ss_policy, P_mean, Pk_mean, Pm_mean)

        # Get deterministic steady state aggregates as reference
        deterministic_ss_aggregates = econ_model.get_aggregates(
            jnp.zeros_like(econ_model.state_ss),
            jnp.zeros_like(econ_model.policies_ss),
            P_mean,
            Pk_mean,
            Pm_mean,
        )

        print("Stochastic SS aggregates:", stoch_ss_aggregates)
        print("Deterministic SS aggregates:", deterministic_ss_aggregates)

        # Store results including simulation data
        # Convert JAX arrays to Python lists for JSON serialization
        experiment_analysis_json = {
            "simul_state_mean": simul_state_mean.tolist(),
            "simul_policies_mean": simul_policies_mean.tolist(),
            "simul_aggregates_mean": simul_aggregates_descstats["mean"],
            "simul_aggregates_std": simul_aggregates_descstats["std"],
            "simul_aggregates_skewness": simul_aggregates_descstats["skewness"],
            "simul_aggregates_kurtosis": simul_aggregates_descstats["kurtosis"],
            "stoch_ss_state": stoch_ss_obs.tolist(),
            "stoch_ss_policy": stoch_ss_policy.tolist(),
            "stoch_ss_aggregates": stoch_ss_aggregates.tolist(),
            "deterministic_ss_aggregates": deterministic_ss_aggregates.tolist(),
        }

        print(f"  Experiment analysis for {experiment_label}: {experiment_analysis_json}")

        analysis_results[experiment_label] = experiment_analysis_json

        # Optional: run seed × length grid diagnostics
        if analysis_config.get("run_seed_length_grid", False):
            grid_labels = analysis_config.get("grid_experiment_labels", [])
            if not grid_labels or (experiment_label in grid_labels):
                print(f"  Running seed-length grid diagnostics for {experiment_label}...")
                grid_results = run_seed_length_grid(
                    econ_model,
                    train_state,
                    base_config=analysis_config,
                    lengths=analysis_config.get("grid_lengths", [2000, 5000, 10000]),
                    n_seeds=int(analysis_config.get("grid_n_seeds", 16)),
                    burnin_fracs=analysis_config.get("grid_burnin_fracs", [0.2]),
                    iact_num_batches=int(analysis_config.get("iact_num_batches", 20)),
                )
                analysis_results[experiment_label]["grid_results"] = grid_results
                _print_grid_summary(grid_results)

        # store in save_dir/experiment_label/analysis_results.json
        # experiment_dir = os.path.join(save_dir, experiment_name)
        # os.makedirs(experiment_dir, exist_ok=True)
        # with open(os.path.join(experiment_dir, "analysis_results.json"), "w") as f:
        #     json.dump(experiment_analysis, f)

        print(f"  Analysis completed for {experiment_label}")

    print("\nAnalysis completed successfully!")

    # Generate sectoral capital plot
    print("Generating mean sectoral capital plot...")
    # Extract sectoral capital means from the general state means for backward compatibility
    plot_results = {}
    for experiment_label, results in analysis_results.items():
        plot_results[experiment_label] = {"sectoral_capital_mean": results["simul_state_mean"][: econ_model.n_sectors]}

    fig, ax = plot_sectoral_capital_mean(
        analysis_results=plot_results,
        sector_labels=econ_model.labels,
        save_path=os.path.join(save_dir, "sectoral_capital_mean.png"),
    )
    print(f"Plot saved to: {os.path.join(save_dir, 'sectoral_capital_mean.png')}")

    return analysis_results


if __name__ == "__main__":
    main()
