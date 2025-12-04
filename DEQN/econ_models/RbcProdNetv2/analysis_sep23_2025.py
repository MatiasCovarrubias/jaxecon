#!/usr/bin/env python3
"""
Analysis script for DEQN trained models.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.econ_models.RbcProdNetv2.Analysis_Sep23_2025

        # Method 2: Run directly as script (from repository root):
        python DEQN/econ_models/RbcProdNetv2/Analysis_Sep23_2025.py

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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import importlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402

from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    simulation_analysis,
)
from DEQN.analysis.stochastic_ss import create_stochss_fn  # noqa: E402
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.plots import (  # noqa: E402
    plot_ergodic_histograms,
    plot_gir_responses,
    plot_sectoral_capital_mean,
)
from DEQN.econ_models.RbcProdNetv2.tables import (  # noqa: E402
    create_comparative_stats_table,
    create_descriptive_stats_table,
    create_stochastic_ss_table,
    create_welfare_table,
)
from DEQN.training.checkpoints import load_experiment_data, load_trained_model_GPU  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and experiment names
MODEL_DIR = "RbcProdNet_Oct2025"
ANALYSIS_NAME = "baseline_analysis"

# Configuration dictionary
config = {
    # Analysis identification
    "analysis_name": ANALYSIS_NAME,
    # Model and path configuration
    "model_dir": MODEL_DIR,
    # Experiments to analyze
    "experiments_to_analyze": {
        # "High Volatility": "baseline_nostateaug_high",
        "Baseline": "baseline_nostateaug_finetunev2",
        # "Low Volatility": "baseline_nostateaug_lower",
    },
    # Simulation configuration
    "init_range": 0,
    "periods_per_epis": 6000,
    "burn_in_periods": 1000,
    "simul_vol_scale": 1,
    "simul_seed": 0,
    "n_simul_seeds": 10,
    # Welfare configuration
    "welfare_n_trajects": 100,
    "welfare_traject_length": 500,
    "welfare_seed": 0,
    # Stochastic steady state configuration
    "n_draws": 500,
    "time_to_converge": 200,
    "seed": 0,
    # GIR configuration
    "gir_n_draws": 100,
    "gir_trajectory_length": 50,
    "gir_tfp_shock_size": 0.2,
    "gir_sectors_to_shock": None,
    "gir_aggregate_indices": [0, 3, 5],
    "gir_seed": 42,
    # JAX configuration
    "double_precision": True,
}

# ============================================================================
# DYNAMIC IMPORTS (based on MODEL_DIR)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{MODEL_DIR}.model")
Model = model_module.Model


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    print(f"Analysis: {config['analysis_name']}", flush=True)

    # Environment and precision setup
    print("Setting up precision...", flush=True)
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)
    print("Precision setup complete.", flush=True)

    model_dir = os.path.join(base_dir, config["model_dir"])
    save_dir = os.path.join(model_dir, "experiments/")
    plots_dir = os.path.join(model_dir, "plots/")
    tables_dir = os.path.join(model_dir, "tables/")

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
    experiments_to_analyze = config["experiments_to_analyze"]
    experiments_data = load_experiment_data(experiments_to_analyze, save_dir)
    print("Experiment data loaded successfully.", flush=True)

    # Define shared nn_config using model_data (features will be set per experiment)
    nn_config_base = {
        "C": model_data["SolData"]["C"],
        "policies_sd": model_data["SolData"]["policies_sd"],
        "params_dtype": precision,
    }

    # Create analysis functions
    print("Creating analysis functions...", flush=True)
    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config))
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, config))
    gir_fn = jax.jit(create_GIR_fn(econ_model, config))
    print("Analysis functions created successfully.", flush=True)

    # Storage for analysis results
    simulation_data = {}
    welfare_costs = {}
    stochastic_ss_data = {}
    gir_data = {}

    # Data collection loop
    print("Collecting analysis data...", flush=True)
    for experiment_label, exp_data in experiments_data.items():
        print(f"  Processing: {experiment_label}", flush=True)

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        # Build nn_config with experiment-specific features
        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        # Load trained model
        train_state = load_trained_model_GPU(experiment_name, save_dir, nn_config)

        # Generate simulation data
        simul_obs, simul_policies, simul_aggregates = simulation_analysis(
            train_state, econ_model, config, simulation_fn
        )

        # Store simulation data (aggregates + sectoral capital)
        simulation_data[experiment_label] = {
            "aggregates": simul_aggregates,
            "sectoral_capital_mean": jnp.mean(simul_obs, axis=0)[: econ_model.n_sectors].tolist(),
        }

        # Calculate and store welfare cost
        simul_utilities = simul_aggregates[:, -1]
        welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)
        welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config["welfare_seed"]))
        welfare_loss = (1 - welfare / welfare_ss) * 100
        welfare_costs[experiment_label] = welfare_loss

        # Calculate and store stochastic steady state
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)

        # Get average prices from simulation policies
        simul_policies_mean = jnp.mean(simul_policies, axis=0)
        P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
        Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
        Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

        # Calculate stochastic steady state aggregates
        stoch_ss_aggregates = econ_model.get_aggregates(stoch_ss_obs, stoch_ss_policy, P_mean, Pk_mean, Pm_mean)

        # Store stochastic steady state data (first 7 aggregates)
        stochastic_ss_data[experiment_label] = stoch_ss_aggregates[:7].tolist()

        # Calculate and store GIR
        gir_results = gir_fn(simul_obs, train_state, simul_policies)
        gir_data[experiment_label] = gir_results

    print("Data collection completed successfully.", flush=True)

    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Extract aggregates data for table/plot functions
    aggregates_data = {exp_label: data["aggregates"] for exp_label, data in simulation_data.items()}
    sectoral_capital_data = {
        exp_label: {"sectoral_capital_mean": data["sectoral_capital_mean"]}
        for exp_label, data in simulation_data.items()
    }

    # Generate tables and figures
    print("Generating tables and figures...", flush=True)

    # Descriptive statistics tables
    create_descriptive_stats_table(
        aggregates_data=aggregates_data,
        save_path=os.path.join(tables_dir, "descriptive_stats_table.tex"),
        analysis_name=config["analysis_name"],
    )

    if len(aggregates_data) > 1:
        create_comparative_stats_table(
            aggregates_data=aggregates_data,
            save_path=os.path.join(tables_dir, "descriptive_stats_comparative.tex"),
            analysis_name=config["analysis_name"],
        )

    # Welfare table
    create_welfare_table(
        welfare_data=welfare_costs,
        save_path=os.path.join(tables_dir, "welfare_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # Stochastic steady state table
    create_stochastic_ss_table(
        stochastic_ss_data=stochastic_ss_data,
        save_path=os.path.join(tables_dir, "stochastic_ss_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # Aggregate histograms
    plot_ergodic_histograms(aggregates_data=aggregates_data, save_dir=plots_dir, analysis_name=config["analysis_name"])

    # Sectoral capital bar plot
    plot_sectoral_capital_mean(
        analysis_results=sectoral_capital_data,
        sector_labels=econ_model.labels,
        save_path=os.path.join(plots_dir, "sectoral_capital_analysis.png"),
        analysis_name=config["analysis_name"],
    )

    # GIR plots
    first_experiment = list(gir_data.keys())[0]
    sectors_shocked = list(gir_data[first_experiment].keys())

    plot_gir_responses(
        gir_data=gir_data,
        aggregate_indices=config["gir_aggregate_indices"],
        sectors_to_plot=sectors_shocked,
        save_dir=plots_dir,
        analysis_name=config["analysis_name"],
    )

    print("Analysis completed successfully.", flush=True)

    return {
        "simulation_data": simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
        "gir_data": gir_data,
    }


if __name__ == "__main__":
    main()
