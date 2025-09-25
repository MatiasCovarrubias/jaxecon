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

# Add repository root to path for absolute imports when run directly
# This MUST come before any DEQN imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# DEQN imports (use absolute imports that work both as module and script)
from DEQN.algorithm.simulation import create_episode_simulation_fn_verbose  # noqa: E402
from DEQN.analysis.simul_analysis import simulation_analysis  # noqa: E402
from DEQN.analysis.stochastic_ss import create_stochss_fn  # noqa: E402
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.plots import (  # noqa: E402
    plot_ergodic_histograms,
    plot_sectoral_capital_mean,
)
from DEQN.econ_models.RbcProdNetv2.RbcProdNet_Sept23_2025 import Model  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.tables import (  # noqa: E402
    create_comparative_stats_table,
    create_descriptive_stats_table,
    create_stochastic_ss_table,
    create_welfare_table,
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
        "n_simul_seeds": 10,  # Number of parallel simulation seeds
        # Welfare configuration
        "welfare_n_trajects": 100,
        "welfare_traject_length": 500,
        "welfare_seed": 0,
        # Stochastic steady state configuration
        "n_draws": 500,
        "time_to_converge": 200,
        "seed": 0,
        # JAX configuration
        "double_precision": True,
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

    # Create simulation, welfare, and stochastic steady state functions
    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, analysis_config))
    welfare_fn = jax.jit(get_welfare_fn(econ_model, analysis_config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, analysis_config))

    # Storage for the three types of data we collect
    simulation_data = {}  # aggregates, sectoral capital means
    welfare_costs = {}  # welfare losses
    stochastic_ss_data = {}  # stochastic steady state obs and policies

    # Data collection loop - collect three types of data
    for experiment_label, exp_data in experiments_data.items():
        print(f"Collecting data for experiment: {experiment_label}")

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        # Build nn_config with experiment-specific features
        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        # Load trained model
        train_state = load_trained_model_GPU(experiment_name, save_dir, nn_config)

        # Generate simulation data
        simul_obs, simul_policies, simul_aggregates = simulation_analysis(
            train_state, econ_model, analysis_config, simulation_fn
        )

        # 1. Store simulation data (aggregates + sectoral capital)
        simulation_data[experiment_label] = {
            "aggregates": simul_aggregates,
            "sectoral_capital_mean": jnp.mean(simul_obs, axis=0)[: econ_model.n_sectors].tolist(),
        }

        # 2. Calculate and store welfare cost
        simul_utilities = simul_aggregates[:, -1]  # Extract utility levels
        welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)
        welfare = welfare_fn(
            simul_utilities,
            welfare_ss,
            random.PRNGKey(analysis_config["welfare_seed"]),
        )
        welfare_loss = (1 - welfare / welfare_ss) * 100  # Convert to percentage
        welfare_costs[experiment_label] = welfare_loss
        print(f"    Welfare loss: {welfare_loss:.4f}%")

        # 3. Calculate and store stochastic steady state
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)

        # Validate convergence
        max_std = jnp.max(stoch_ss_obs_std)
        print(f"    Max stochastic SS std: {max_std:.6f}")
        if max_std > 0.01:
            print(f"    Warning: Stochastic SS std may be large: {max_std:.6f}")

        # Get average prices from simulation policies
        simul_policies_mean = jnp.mean(simul_policies, axis=0)
        P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
        Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
        Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

        # Calculate stochastic steady state aggregates
        stoch_ss_aggregates = econ_model.get_aggregates(stoch_ss_obs, stoch_ss_policy, P_mean, Pk_mean, Pm_mean)

        # Store stochastic steady state data (first 7 aggregates)
        stochastic_ss_data[experiment_label] = stoch_ss_aggregates[:7].tolist()

        print(f"  Data collection completed for {experiment_label}")

    print("\nData collection completed successfully!")

    # ===================================================================
    # GENERATE TABLES AND FIGURES
    # ===================================================================

    # Create output directories
    plots_dir = os.path.join(model_dir, "Plots")
    tables_dir = os.path.join(model_dir, "Tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Extract aggregates data for table/plot functions
    aggregates_data = {exp_label: data["aggregates"] for exp_label, data in simulation_data.items()}
    sectoral_capital_data = {
        exp_label: {"sectoral_capital_mean": data["sectoral_capital_mean"]}
        for exp_label, data in simulation_data.items()
    }

    # 1. Generate descriptive statistics tables
    print("Generating descriptive statistics tables...")
    create_descriptive_stats_table(
        aggregates_data=aggregates_data, save_path=os.path.join(tables_dir, "descriptive_stats_table.tex")
    )

    if len(aggregates_data) > 1:
        create_comparative_stats_table(
            aggregates_data=aggregates_data,
            save_path=os.path.join(tables_dir, "descriptive_stats_comparative.tex"),
        )

    print(f"Descriptive statistics tables saved to: {tables_dir}")

    # 2. Generate welfare table
    print("Generating welfare table...")
    create_welfare_table(welfare_data=welfare_costs, save_path=os.path.join(tables_dir, "welfare_table.tex"))
    print(f"Welfare table saved to: {tables_dir}")

    # 3. Generate stochastic steady state table
    print("Generating stochastic steady state table...")
    create_stochastic_ss_table(
        stochastic_ss_data=stochastic_ss_data, save_path=os.path.join(tables_dir, "stochastic_ss_table.tex")
    )
    print(f"Stochastic steady state table saved to: {tables_dir}")

    # 4. Generate aggregate histograms
    print("Generating aggregate histograms...")
    plot_ergodic_histograms(aggregates_data=aggregates_data, save_dir=plots_dir)
    print(f"Histogram plots saved to: {plots_dir}")

    # 5. Generate sectoral capital bar plot
    print("Generating sectoral capital bar plot...")
    plot_sectoral_capital_mean(
        analysis_results=sectoral_capital_data,
        sector_labels=econ_model.labels,
        save_path=os.path.join(plots_dir, "sectoral_capital_analysis.png"),
    )
    print(f"Sectoral capital plot saved to: {plots_dir}")

    print("\nAll analysis completed successfully!")

    return {
        "simulation_data": simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
    }


if __name__ == "__main__":
    main()
