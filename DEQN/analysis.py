#!/usr/bin/env python3
"""
Analysis script for DEQN trained models.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.analysis

        # Method 2: Run directly as script (from repository root):
        python DEQN/analysis.py

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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import glob  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402

from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
from DEQN.analysis.plots import (  # noqa: E402
    plot_ergodic_histograms,
    plot_gir_responses,
)
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    simulation_analysis,
)
from DEQN.analysis.stochastic_ss import create_stochss_fn  # noqa: E402
from DEQN.analysis.tables import (  # noqa: E402
    create_comparative_stats_table,
    create_descriptive_stats_table,
    create_stochastic_ss_table,
    create_welfare_table,
)
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.utils import load_experiment_data, load_trained_model_orbax  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "model_dir": "RbcProdNet_Oct2025",
    "analysis_name": "baseline_analysis",
    # Experiments to analyze
    "experiments_to_analyze": {
        # "High Volatility": "baseline_nostateaug_high",
        "test": "test_local_1thread",
        # "Low Volatility": "baseline_nostateaug_lower",
    },
    # Simulation configuration
    "init_range": 0,
    "periods_per_epis": 8000,
    "burn_in_periods": 1000,
    "simul_vol_scale": 1,
    "simul_seed": 0,
    "n_simul_seeds": 10,
    # Welfare configuration
    "welfare_n_trajects": 200,
    "welfare_traject_length": 500,
    "welfare_seed": 0,
    # Stochastic steady state configuration
    "n_draws": 500,
    "time_to_converge": 200,
    "seed": 0,
    # GIR configuration
    "gir_n_draws": 100,
    "gir_trajectory_length": 50,
    "shock_size": 0.2,
    "states_to_shock": None,
    "gir_seed": 42,
    # JAX configuration
    "double_precision": True,
}

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model

# Import model-specific plots module and registry
plots_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.plots")
MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", [])


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

    # Create analysis directory structure
    analysis_dir = os.path.join(model_dir, "analysis", config["analysis_name"])
    simulation_dir = os.path.join(analysis_dir, "simulation")
    irs_dir = os.path.join(analysis_dir, "IRs")

    # Create all directories
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(irs_dir, exist_ok=True)

    # Save analysis configuration as JSON
    config_path = os.path.join(analysis_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Analysis configuration saved to: {config_path}", flush=True)

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
    analysis_variables_data = {}
    raw_simulation_data = {}
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

        # Load trained model (using same initialization approach as training)
        train_state = load_trained_model_orbax(experiment_name, save_dir, nn_config, econ_model.state_ss)

        # Generate simulation data
        simul_obs, simul_policies, simul_analysis_variables = simulation_analysis(
            train_state, econ_model, config, simulation_fn
        )

        # Store raw simulation data for model-specific plots
        raw_simulation_data[experiment_label] = {
            "simul_obs": simul_obs,
            "simul_policies": simul_policies,
            "simul_analysis_variables": simul_analysis_variables,
        }

        # Store analysis variables for general analysis
        analysis_variables_data[experiment_label] = simul_analysis_variables

        # Calculate utilities separately using the new utility method
        simul_utilities = jax.vmap(econ_model.utility_from_policies)(simul_policies)

        # Calculate and store welfare cost
        welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)
        welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config["welfare_seed"]))
        welfare_loss = (1 - welfare / welfare_ss) * 100
        welfare_costs[experiment_label] = welfare_loss

        # Calculate and store stochastic steady state
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
        if stoch_ss_obs_std.max() > 0.001:
            raise ValueError("Stochastic steady state standard deviation too large")

        # Get average prices from simulation policies
        simul_policies_mean = jnp.mean(simul_policies, axis=0)
        P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
        Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
        Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

        # Calculate stochastic steady state analysis variables (returns dictionary)
        stoch_ss_analysis_variables = econ_model.get_analysis_variables(
            stoch_ss_obs, stoch_ss_policy, P_mean, Pk_mean, Pm_mean
        )

        # Store stochastic steady state data as dictionary
        stochastic_ss_data[experiment_label] = stoch_ss_analysis_variables

        # Calculate and store GIR
        gir_results = gir_fn(simul_obs, train_state, simul_policies)
        gir_data[experiment_label] = gir_results

    print("Data collection completed successfully.", flush=True)

    # ============================================================================
    # GENERAL ANALYSIS: Tables and Plots
    # ============================================================================
    print("Generating general analysis tables and figures...", flush=True)

    # Descriptive statistics tables (in simulation folder)
    descriptive_stats_path = os.path.join(simulation_dir, "descriptive_stats_table.tex")
    create_descriptive_stats_table(
        analysis_variables_data=analysis_variables_data,
        save_path=descriptive_stats_path,
        analysis_name=config["analysis_name"],
    )

    # Print descriptive statistics table
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS TABLE")
    print("=" * 80)
    with open(descriptive_stats_path, "r") as f:
        print(f.read())
    print("=" * 80 + "\n")

    if len(analysis_variables_data) > 1:
        create_comparative_stats_table(
            analysis_variables_data=analysis_variables_data,
            save_path=os.path.join(simulation_dir, "descriptive_stats_comparative.tex"),
            analysis_name=config["analysis_name"],
        )

    # Welfare table (in analysis directory)
    create_welfare_table(
        welfare_data=welfare_costs,
        save_path=os.path.join(analysis_dir, "welfare_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # Stochastic steady state table (in analysis directory)
    stochastic_ss_path = os.path.join(analysis_dir, "stochastic_ss_table.tex")
    create_stochastic_ss_table(
        stochastic_ss_data=stochastic_ss_data,
        save_path=stochastic_ss_path,
        analysis_name=config["analysis_name"],
    )

    # Print stochastic steady state table
    print("\n" + "=" * 80)
    print("STOCHASTIC STEADY STATE TABLE")
    print("=" * 80)
    with open(stochastic_ss_path, "r") as f:
        print(f.read())
    print("=" * 80 + "\n")

    # Analysis variable histograms (in simulation folder)
    plot_ergodic_histograms(
        analysis_variables_data=analysis_variables_data, save_dir=simulation_dir, analysis_name=config["analysis_name"]
    )

    # Display histogram plots
    print("Displaying ergodic histograms...", flush=True)
    histogram_files = glob.glob(os.path.join(simulation_dir, f"histogram_*_{config['analysis_name']}.png"))
    for hist_file in sorted(histogram_files):
        img = plt.imread(hist_file)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # GIR plots (in IRs folder) - save but don't display (too many)
    first_experiment = list(gir_data.keys())[0]
    states_shocked = list(gir_data[first_experiment].keys())

    plot_gir_responses(
        gir_data=gir_data,
        states_to_plot=states_shocked,
        save_dir=irs_dir,
        analysis_name=config["analysis_name"],
    )
    print(f"GIR plots saved to {irs_dir} (not displayed - too many plots)", flush=True)

    # ============================================================================
    # MODEL-SPECIFIC ANALYSIS: Plots
    # ============================================================================
    print("Generating model-specific plots...", flush=True)

    model_specific_plot_files = []
    if MODEL_SPECIFIC_PLOTS:
        for plot_spec in MODEL_SPECIFIC_PLOTS:
            plot_name = plot_spec["name"]
            plot_function = plot_spec["function"]
            print(f"  - Running model-specific plot: {plot_name}", flush=True)

            # Run the plot for each experiment (save in simulation folder)
            for experiment_label, sim_data in raw_simulation_data.items():
                try:
                    plot_path = os.path.join(simulation_dir, f"{plot_name}_{experiment_label}.png")
                    plot_function(
                        simul_obs=sim_data["simul_obs"],
                        simul_policies=sim_data["simul_policies"],
                        simul_analysis_variables=sim_data["simul_analysis_variables"],
                        save_path=plot_path,
                        analysis_name=config["analysis_name"],
                        econ_model=econ_model,
                        experiment_label=experiment_label,
                    )
                    print(f"    ✓ {plot_name} for {experiment_label}", flush=True)
                    model_specific_plot_files.append(plot_path)
                except Exception as e:
                    print(f"    ✗ Failed to create {plot_name} for {experiment_label}: {e}", flush=True)

        # Display model-specific plots
        print("Displaying model-specific plots...", flush=True)
        for plot_file in sorted(model_specific_plot_files):
            if os.path.exists(plot_file):
                img = plt.imread(plot_file)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(os.path.basename(plot_file))
                plt.tight_layout()
                plt.show()
    else:
        print("  No model-specific plots registered.", flush=True)

    print("Analysis completed successfully.", flush=True)

    return {
        "analysis_variables_data": analysis_variables_data,
        "raw_simulation_data": raw_simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
        "gir_data": gir_data,
    }


if __name__ == "__main__":
    main()
