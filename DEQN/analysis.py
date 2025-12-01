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

    base_dir = "/content/drive/MyDrive/Jaxecontemp"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import importlib  # noqa: E402
import json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402

from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
from DEQN.analysis.matlab_irs import load_matlab_irs  # noqa: E402
from DEQN.analysis.plots import (  # noqa: E402
    plot_ergodic_histograms,
    plot_sector_ir_by_shock_size,
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
# DYNARE SIMULATION PROCESSING UTILITIES
# ============================================================================


def get_dynare_variable_indices(n_sectors: int) -> dict:
    """Get variable indices for Dynare simulation data (0-indexed for Python)."""
    n = n_sectors
    return {
        "k": (0, n),
        "a": (n, 2 * n),
        "c": (2 * n, 3 * n),
        "l": (3 * n, 4 * n),
        "pk": (4 * n, 5 * n),
        "pm": (5 * n, 6 * n),
        "m": (6 * n, 7 * n),
        "mout": (7 * n, 8 * n),
        "i": (8 * n, 9 * n),
        "iout": (9 * n, 10 * n),
        "p": (10 * n, 11 * n),
        "q": (11 * n, 12 * n),
        "y": (12 * n, 13 * n),
        "cagg": 13 * n,
        "lagg": 13 * n + 1,
        "yagg": 13 * n + 2,
        "iagg": 13 * n + 3,
        "magg": 13 * n + 4,
    }


def process_dynare_simulation(
    simul_data: jnp.ndarray,
    n_sectors: int,
    P_weights: jnp.ndarray,
    Pk_weights: jnp.ndarray,
    Pm_weights: jnp.ndarray,
    state_ss: jnp.ndarray,
    policies_ss: jnp.ndarray,
    burn_in: int = 0,
) -> dict:
    """
    Process Dynare simulation data and compute analysis variables using nonlinear aggregators.

    Args:
        simul_data: Dynare simulation output (n_vars, T) in log levels
        n_sectors: Number of sectors
        P_weights: Price weights for output aggregation (from nonlinear ergodic dist)
        Pk_weights: Capital price weights for investment/capital aggregation
        Pm_weights: Intermediate price weights for intermediate aggregation
        state_ss: Steady state states (log)
        policies_ss: Steady state policies (log)
        burn_in: Number of initial periods to discard

    Returns:
        Dictionary with analysis variables (log deviations from steady state)
    """
    idx = get_dynare_variable_indices(n_sectors)

    # Extract simulation data (after burn-in) - data is in log levels
    simul = simul_data[:, burn_in:]
    T = simul.shape[1]

    # Extract sectoral variables in levels
    K = jnp.exp(simul[idx["k"][0] : idx["k"][1], :])  # (n_sectors, T)
    Y = jnp.exp(simul[idx["y"][0] : idx["y"][1], :])
    I = jnp.exp(simul[idx["i"][0] : idx["i"][1], :])
    M = jnp.exp(simul[idx["m"][0] : idx["m"][1], :])

    # Extract aggregate variables directly from Dynare (in log levels)
    Cagg = jnp.exp(simul[idx["cagg"], :])
    Lagg = jnp.exp(simul[idx["lagg"], :])

    # Aggregate using nonlinear model's price weights
    # These weights are already in levels (from ergodic distribution)
    Kagg = K.T @ Pk_weights  # (T,)
    Yagg = Y.T @ P_weights
    Iagg = I.T @ Pk_weights
    Magg = M.T @ Pm_weights

    # Calculate steady state aggregates using same weights
    K_ss = jnp.exp(state_ss[:n_sectors])
    policies_ss_levels = jnp.exp(policies_ss)

    Y_ss = policies_ss_levels[10 * n_sectors : 11 * n_sectors]
    I_ss = policies_ss_levels[6 * n_sectors : 7 * n_sectors]
    M_ss = policies_ss_levels[4 * n_sectors : 5 * n_sectors]
    Cagg_ss = policies_ss_levels[11 * n_sectors]
    Lagg_ss = policies_ss_levels[11 * n_sectors + 1]

    Kagg_ss = K_ss @ Pk_weights
    Yagg_ss = Y_ss @ P_weights
    Iagg_ss = I_ss @ Pk_weights
    Magg_ss = M_ss @ Pm_weights

    # Compute log deviations from steady state
    Cagg_logdev = jnp.log(Cagg) - jnp.log(Cagg_ss)
    Lagg_logdev = jnp.log(Lagg) - jnp.log(Lagg_ss)
    Kagg_logdev = jnp.log(Kagg) - jnp.log(Kagg_ss)
    Yagg_logdev = jnp.log(Yagg) - jnp.log(Yagg_ss)
    Iagg_logdev = jnp.log(Iagg) - jnp.log(Iagg_ss)
    Magg_logdev = jnp.log(Magg) - jnp.log(Magg_ss)

    return {
        "Agg. Consumption": Cagg_logdev,
        "Agg. Labor": Lagg_logdev,
        "Agg. Capital": Kagg_logdev,
        "Agg. Output": Yagg_logdev,
        "Agg. Intermediates": Magg_logdev,
        "Agg. Investment": Iagg_logdev,
    }


def compute_simulation_statistics(analysis_vars: dict) -> dict:
    """Compute mean and std for each analysis variable."""
    stats = {}
    for var_name, values in analysis_vars.items():
        stats[var_name] = {
            "mean": float(jnp.mean(values)),
            "std": float(jnp.std(values)),
            "min": float(jnp.min(values)),
            "max": float(jnp.max(values)),
        }
    return stats


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "model_dir": "RbcProdNet_nonlinear",
    "analysis_name": "npnlinearv2_lowvol",
    # Experiments to analyze
    "experiments_to_analyze": {
        # "higher volatility": "3vol",
        # "x0.075 volatility": "nonlinearv2_volx0dot075",
        # "x0.05 volatility": "nonlinearv2_vol0dot05",
        # "x0.025 volatility": "nonlinearv2_volx0dot025",
        "x1.5 volatility": "nonlinearv2_volx1.5",
        "baseline": "nonlinearv2",
        "x0.1 volatility": "nonlinearv2_volx0dot1",
        # "x0.75 volatility": "x0dot75vol_newNN",
        # "x0.5 volatility": "x0dot5_newNN",
        # "x0.1 volatility": "newcalib_volx0dot1",
    },
    # Simulation configuration
    "init_range": 6,
    "periods_per_epis": 64000,
    "burn_in_periods": 3200,
    "simul_vol_scale": 1,
    "simul_seed": 0,
    "n_simul_seeds": 16,
    # Welfare configuration
    "welfare_n_trajects": 16000,
    "welfare_traject_length": 100,
    "welfare_seed": 0,
    # Stochastic steady state configuration
    "n_draws": 2000,
    "time_to_converge": 500,
    "seed": 0,
    # GIR configuration
    "gir_n_draws": 1000,
    "gir_trajectory_length": 100,
    "shock_size": 0.2,
    "gir_seed": 42,
    # Combined IR analysis configuration
    # Sectors to analyze: specify sector indices (0-based).
    # GIRs shock the TFP/productivity state (state index = n_sectors + sector_idx).
    # For example, sector 0 TFP is at state index 37 (for n_sectors=37).
    "ir_sectors_to_plot": [0, 2, 23],
    "ir_variable_to_plot": "Agg. Consumption",
    "ir_shock_sizes": [5, 10, 20],
    "ir_max_periods": 80,
    "matlab_ir_file_pattern": "AllSectors_IRS__Oct_25nonlinear_{sign}_{size}.mat",
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

    # Load model data (supports both old and new structure)
    print("Loading model data...", flush=True)
    # Try new naming convention first, fall back to old
    model_path = os.path.join(model_dir, "ModelData.mat")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model_data.mat")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Tried: ModelData.mat and model_data.mat in {model_dir}")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    print("Model data loaded successfully.", flush=True)

    # Detect structure and extract data
    dynare_simul_loglin = None
    dynare_simul_determ = None

    def safe_get(d, *keys, default=None):
        """Safely navigate nested dict/struct, returning default if any key is missing."""
        for key in keys:
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d

    if "ModelData" in model_data:
        # New structure: ModelData.SteadyState, ModelData.Simulation, ModelData.Solution
        print("Detected new ModelData structure.", flush=True)
        md = model_data["ModelData"]

        # Required: SteadyState (always needed)
        if "SteadyState" not in md:
            raise ValueError("ModelData missing required 'SteadyState' field")
        ss = md["SteadyState"]
        n_sectors = ss["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(ss["endostates_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params = ss["parameters"]
        policies_ss = jnp.array(ss["policies_ss"], dtype=precision)

        # Required: Solution.StateSpace.C (needed for neural net)
        C_matrix = safe_get(md, "Solution", "StateSpace", "C")
        if C_matrix is None:
            raise ValueError("ModelData missing required 'Solution.StateSpace.C' field")

        # Required: Simulation stats (states_sd, policies_sd)
        state_sd = safe_get(md, "Simulation", "states_sd")
        policies_sd = safe_get(md, "Simulation", "policies_sd")
        if state_sd is None or policies_sd is None:
            raise ValueError("ModelData missing required 'Simulation.states_sd' or 'Simulation.policies_sd'")
        state_sd = jnp.array(state_sd, dtype=precision)
        policies_sd = jnp.array(policies_sd, dtype=precision)

        # Optional: Full Dynare simulations (for comparison analysis)
        loglin_simul = safe_get(md, "Simulation", "Loglin", "full_simul")
        if loglin_simul is not None:
            dynare_simul_loglin = jnp.array(loglin_simul, dtype=precision)
            print(f"  ✓ Log-linear simulation available: {dynare_simul_loglin.shape}", flush=True)
        else:
            print("  - Log-linear simulation: not included", flush=True)

        determ_simul = safe_get(md, "Simulation", "Determ", "full_simul")
        if determ_simul is not None:
            dynare_simul_determ = jnp.array(determ_simul, dtype=precision)
            print(f"  ✓ Deterministic simulation available: {dynare_simul_determ.shape}", flush=True)
        else:
            print("  - Deterministic simulation: not included", flush=True)

        # Optional: IRFs (report availability)
        if safe_get(md, "IRFs") is not None:
            n_shocks = len(safe_get(md, "IRFs", "by_shock", default=[]))
            print(f"  ✓ IRFs available: {n_shocks} shock configurations", flush=True)
        else:
            print("  - IRFs: not included", flush=True)

    elif "SolData" in model_data:
        # Old structure: SolData contains everything
        print("Detected old SolData structure.", flush=True)
        soldata = model_data["SolData"]
        n_sectors = soldata["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(soldata["k_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params = soldata["parameters"]
        state_sd = jnp.array(soldata["states_sd"], dtype=precision)
        policies_sd = jnp.array(soldata["policies_sd"], dtype=precision)
        policies_ss = jnp.array(soldata["policies_ss"], dtype=precision)
        C_matrix = soldata["C"]
    else:
        raise ValueError("Unknown model_data structure. Expected 'ModelData' or 'SolData' key.")

    # Print parameters
    print("Parameters:\n", params)

    # Create economic model
    print("Creating economic model...", flush=True)
    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
    )
    print("Economic model created successfully.", flush=True)

    # Load experiment data
    print("Loading experiment data...", flush=True)
    experiments_to_analyze = config["experiments_to_analyze"]
    experiments_data = load_experiment_data(experiments_to_analyze, save_dir)
    print("Experiment data loaded successfully.", flush=True)

    # Define shared nn_config (features will be set per experiment)
    nn_config_base = {
        "C": C_matrix,
        "states_sd": state_sd,
        "policies_sd": policies_sd,
        "params_dtype": precision,
    }

    # Compute states_to_shock from ir_sectors_to_plot
    # GIRs always shock TFP/productivity: state_idx = n_sectors + sector_idx
    ir_sectors = config.get("ir_sectors_to_plot", [0])
    states_to_shock = [n_sectors + sector_idx for sector_idx in ir_sectors]

    config["states_to_shock"] = states_to_shock
    print(f"GIR: Shocking TFP for sectors {ir_sectors} (state indices: {states_to_shock})", flush=True)

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
    # PROCESS DYNARE SIMULATIONS (if available)
    # ============================================================================
    # Use average prices from the first experiment's ergodic distribution
    # to aggregate Dynare simulations consistently with the nonlinear model
    first_experiment_label = list(analysis_variables_data.keys())[0]
    first_sim_data = raw_simulation_data[first_experiment_label]

    # Get average prices from nonlinear simulation (in log deviations, convert to levels)
    simul_policies = first_sim_data["simul_policies"]
    simul_policies_mean = jnp.mean(simul_policies, axis=0)

    # Prices are stored as log deviations in policies, need to convert to levels for weights
    # P_mean is the mean log deviation, so P_weights = P_ss * exp(P_mean) gives mean prices
    P_ss = jnp.exp(policies_ss[8 * n_sectors : 9 * n_sectors])
    Pk_ss = jnp.exp(policies_ss[2 * n_sectors : 3 * n_sectors])
    Pm_ss = jnp.exp(policies_ss[3 * n_sectors : 4 * n_sectors])

    # simul_policies_mean is already denormalized (log deviation from SS)
    # So mean prices in levels are: P_ss * exp(mean_logdev)
    P_weights = P_ss * jnp.exp(simul_policies_mean[8 * n_sectors : 9 * n_sectors])
    Pk_weights = Pk_ss * jnp.exp(simul_policies_mean[2 * n_sectors : 3 * n_sectors])
    Pm_weights = Pm_ss * jnp.exp(simul_policies_mean[3 * n_sectors : 4 * n_sectors])

    print(f"Using price weights from experiment: {first_experiment_label}", flush=True)

    # Process Dynare simulations if available
    if dynare_simul_loglin is not None:
        print("Processing log-linear (Dynare) simulation...", flush=True)
        loglin_analysis_vars = process_dynare_simulation(
            simul_data=dynare_simul_loglin,
            n_sectors=n_sectors,
            P_weights=P_weights,
            Pk_weights=Pk_weights,
            Pm_weights=Pm_weights,
            state_ss=state_ss,
            policies_ss=policies_ss,
            burn_in=config["burn_in_periods"],
        )
        # Add to analysis data for comparison
        analysis_variables_data["Log-Linear (Dynare)"] = loglin_analysis_vars
        print("  Log-linear simulation processed and added to analysis.", flush=True)

    if dynare_simul_determ is not None:
        print("Processing deterministic (Dynare) simulation...", flush=True)
        determ_analysis_vars = process_dynare_simulation(
            simul_data=dynare_simul_determ,
            n_sectors=n_sectors,
            P_weights=P_weights,
            Pk_weights=Pk_weights,
            Pm_weights=Pm_weights,
            state_ss=state_ss,
            policies_ss=policies_ss,
            burn_in=min(config["burn_in_periods"], dynare_simul_determ.shape[1] // 10),
        )
        # Add to analysis data for comparison
        analysis_variables_data["Deterministic (Dynare)"] = determ_analysis_vars
        print("  Deterministic simulation processed and added to analysis.", flush=True)

    # ============================================================================
    # GENERAL ANALYSIS: Tables and Plots
    # ============================================================================
    print("Generating general analysis tables and figures...", flush=True)

    # Descriptive statistics tables (in simulation folder)
    create_descriptive_stats_table(
        analysis_variables_data=analysis_variables_data,
        save_path=os.path.join(simulation_dir, "descriptive_stats_table.tex"),
        analysis_name=config["analysis_name"],
    )

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
    create_stochastic_ss_table(
        stochastic_ss_data=stochastic_ss_data,
        save_path=os.path.join(analysis_dir, "stochastic_ss_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # Analysis variable histograms (in simulation folder)
    plot_ergodic_histograms(
        analysis_variables_data=analysis_variables_data, save_dir=simulation_dir, analysis_name=config["analysis_name"]
    )

    # Note: plot_gir_responses uses old GIR data structure.
    # Combined IR plots below handle both positive and negative shocks.

    # ============================================================================
    # COMBINED IR ANALYSIS: MATLAB + JAX GIRs
    # ============================================================================
    print("\n" + "=" * 60, flush=True)
    print("COMBINED IR ANALYSIS: MATLAB + JAX GIRs", flush=True)
    print("=" * 60, flush=True)

    matlab_ir_dir = os.path.join(model_dir, "MATLAB", "IRs")
    matlab_ir_data = {}

    print(f"\nChecking MATLAB IR directory: {matlab_ir_dir}", flush=True)

    if os.path.exists(matlab_ir_dir):
        print("  ✓ Directory exists", flush=True)
        matlab_ir_data = load_matlab_irs(
            matlab_ir_dir=matlab_ir_dir,
            shock_sizes=config.get("ir_shock_sizes", [5, 10, 20]),
            file_pattern=config.get("matlab_ir_file_pattern", "AllSectors_IRS__Oct_25nonlinear_{sign}_{size}.mat"),
        )

        if matlab_ir_data:
            print(f"\n  ✓ Successfully loaded {len(matlab_ir_data)} shock configurations", flush=True)
        else:
            print("\n  ✗ No MATLAB IR files were loaded", flush=True)
    else:
        print(f"  ✗ Directory NOT FOUND: {matlab_ir_dir}", flush=True)
        print("    To use MATLAB IRs, create this directory and add the .mat files", flush=True)

    # Generate combined IR plots (with or without MATLAB data)
    print("\nGenerating IR comparison plots...", flush=True)

    sectors_to_plot = config.get("ir_sectors_to_plot", [0, 2, 23])
    ir_variable = config.get("ir_variable_to_plot", "Agg. Consumption")
    shock_sizes = config.get("ir_shock_sizes", [5, 10, 20])
    max_periods = config.get("ir_max_periods", 80)

    print(f"  Variable: {ir_variable}", flush=True)
    print(f"  Shock sizes: {shock_sizes}", flush=True)
    print(f"  Sectors to plot: {sectors_to_plot}", flush=True)

    for i, sector_idx in enumerate(sectors_to_plot):
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        print(f"\n  [{i + 1}/{len(sectors_to_plot)}] Processing sector {sector_idx}: {sector_label}...", flush=True)

        plot_sector_ir_by_shock_size(
            gir_data=gir_data,
            matlab_ir_data=matlab_ir_data,
            sector_idx=sector_idx,
            sector_label=sector_label,
            variable_to_plot=ir_variable,
            shock_sizes=shock_sizes,
            save_dir=irs_dir,
            analysis_name=config["analysis_name"],
            max_periods=max_periods,
            n_sectors=n_sectors,
        )

    print("\n  ✓ Combined IR analysis completed.", flush=True)

    # ============================================================================
    # MODEL-SPECIFIC ANALYSIS: Plots
    # ============================================================================
    print("Generating model-specific plots...", flush=True)

    if MODEL_SPECIFIC_PLOTS:
        for plot_spec in MODEL_SPECIFIC_PLOTS:
            plot_name = plot_spec["name"]
            plot_function = plot_spec["function"]
            print(f"  - Running model-specific plot: {plot_name}", flush=True)

            # Run the plot for each experiment (save in simulation folder)
            for experiment_label, sim_data in raw_simulation_data.items():
                try:
                    plot_function(
                        simul_obs=sim_data["simul_obs"],
                        simul_policies=sim_data["simul_policies"],
                        simul_analysis_variables=sim_data["simul_analysis_variables"],
                        save_path=os.path.join(simulation_dir, f"{plot_name}_{experiment_label}.png"),
                        analysis_name=config["analysis_name"],
                        econ_model=econ_model,
                        experiment_label=experiment_label,
                    )
                    print(f"    ✓ {plot_name} for {experiment_label}", flush=True)
                except Exception as e:
                    print(f"    ✗ Failed to create {plot_name} for {experiment_label}: {e}", flush=True)
    else:
        print("  No model-specific plots registered.", flush=True)

    print("Analysis completed successfully.", flush=True)

    return {
        "analysis_variables_data": analysis_variables_data,
        "raw_simulation_data": raw_simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
        "gir_data": gir_data,
        "matlab_ir_data": matlab_ir_data,
    }


if __name__ == "__main__":
    main()
