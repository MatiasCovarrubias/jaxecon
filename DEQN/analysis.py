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

from DEQN.analysis.aggregation_correction import (  # noqa: E402
    compute_ergodic_prices_from_simulation,
    compute_ergodic_steady_state,
    process_simulation_with_consistent_aggregation,
    recenter_analysis_variables,
)
from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
from DEQN.analysis.matlab_irs import load_matlab_irs  # noqa: E402
from DEQN.analysis.plots import (  # noqa: E402
    plot_ergodic_histograms,
    plot_sector_ir_by_shock_size,
    plot_sectoral_capital_comparison,
    plot_sectoral_capital_stochss,
    plot_sectoral_variable_ergodic,
    plot_sectoral_variable_stochss,
)
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    simulation_analysis,
)
from DEQN.analysis.stochastic_ss import (  # noqa: E402
    create_stochss_fn,
    create_stochss_loss_fn,
)
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
    "ir_variables_to_plot": ["Agg. Consumption", "Agg. Labor"],
    "ir_shock_sizes": [5, 10, 20],
    "ir_max_periods": 80,
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

    # Load model data (core)
    print("Loading model data...", flush=True)
    model_path = os.path.join(model_dir, "ModelData.mat")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    print("  ✓ ModelData.mat loaded", flush=True)

    # Extract data from ModelData structure
    if "ModelData" not in model_data:
        raise ValueError("Expected 'ModelData' key in model file.")

    md = model_data["ModelData"]

    # Extract from SteadyState
    ss = md["SteadyState"]
    n_sectors = ss["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    k_ss = jnp.array(ss["endostates_ss"], dtype=precision)
    state_ss = jnp.concatenate([k_ss, a_ss])
    params = ss["parameters"]
    policies_ss = jnp.array(ss["policies_ss"], dtype=precision)

    # Extract from Statistics
    stats = md["Statistics"]
    state_sd = jnp.array(stats["states_sd"], dtype=precision)
    policies_sd = jnp.array(stats["policies_sd"], dtype=precision)

    # Extract from Solution
    C_matrix = md["Solution"]["StateSpace"]["C"]

    # Ensure policies_ss and policies_sd have matching dimensions
    # (Old checkpoints were trained with truncated policies_ss to match policies_sd)
    if len(policies_ss) != len(policies_sd):
        n_policies = len(policies_sd)
        print(f"    ⚠ Aligning policies_ss ({len(policies_ss)}) to policies_sd ({n_policies})", flush=True)
        policies_ss = policies_ss[:n_policies]

    print(f"    n_sectors: {n_sectors}", flush=True)
    print(f"    policies_ss shape: {policies_ss.shape}", flush=True)
    print(f"    policies_sd shape: {policies_sd.shape}", flush=True)

    # Load simulation data (optional - for Dynare comparison)
    dynare_simul_loglin = None
    dynare_simul_determ = None
    simul_path = os.path.join(model_dir, "ModelData_simulation.mat")

    if os.path.exists(simul_path):
        print("  ✓ ModelData_simulation.mat found", flush=True)
        simul_data = sio.loadmat(simul_path, simplify_cells=True)
        simul = simul_data.get("ModelData_simulation", {})

        if "Loglin" in simul and "full_simul" in simul["Loglin"]:
            dynare_simul_loglin = jnp.array(simul["Loglin"]["full_simul"], dtype=precision)
            print(f"    ✓ Log-linear simulation: {dynare_simul_loglin.shape}", flush=True)
        else:
            print("    - Log-linear simulation: not included", flush=True)

        if "Determ" in simul and "full_simul" in simul["Determ"]:
            dynare_simul_determ = jnp.array(simul["Determ"]["full_simul"], dtype=precision)
            print(f"    ✓ Deterministic simulation: {dynare_simul_determ.shape}", flush=True)
        else:
            print("    - Deterministic simulation: not included", flush=True)
    else:
        print("  - ModelData_simulation.mat: not found (Dynare comparison disabled)", flush=True)

    # Load IRF data (optional - for MATLAB IR comparison)
    irs_path = os.path.join(model_dir, "ModelData_IRs.mat")
    if os.path.exists(irs_path):
        irs_data = sio.loadmat(irs_path, simplify_cells=True)
        irs = irs_data.get("ModelData_IRs", {})
        n_ir_shocks = irs.get("n_shocks", 0)
        print(f"  ✓ ModelData_IRs.mat found: {n_ir_shocks} shock configurations", flush=True)
    else:
        print("  - ModelData_IRs.mat: not found", flush=True)

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
    stoch_ss_loss_fn = create_stochss_loss_fn(econ_model, mc_draws=32)
    gir_fn = jax.jit(create_GIR_fn(econ_model, config))
    print("Analysis functions created successfully.", flush=True)

    # Storage for analysis results
    analysis_variables_data = {}
    raw_simulation_data = {}
    welfare_costs = {}
    stochastic_ss_data = {}
    stochastic_ss_states = {}  # Store stochastic SS states for GIR computation
    stochastic_ss_policies = {}  # Store stochastic SS policies for sectoral plots
    stochastic_ss_loss = {}  # Store loss evaluation at stochastic SS
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

        # Calculate welfare from simulation
        welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)
        welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config["welfare_seed"]))

        # Compute consumption-equivalent welfare cost (Vc < 0 means loss)
        Vc = econ_model.consumption_equivalent(welfare)
        welfare_cost_ce = -Vc * 100  # Convert to positive percentage
        welfare_costs[experiment_label] = welfare_cost_ce

        # Verification: Vc at steady state should be ~0
        Vc_ss = econ_model.consumption_equivalent(welfare_ss)
        print(f"    Welfare: Vc = {Vc:.6f}, Vc_ss = {Vc_ss:.6f} (should be ~0)", flush=True)
        print(f"    Consumption-equivalent welfare cost: {welfare_cost_ce:.4f}%", flush=True)

        # Calculate and store stochastic steady state
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
        if stoch_ss_obs_std.max() > 0.001:
            raise ValueError("Stochastic steady state standard deviation too large")

        # Store stochastic SS state for GIR computation (in logdev form)
        stochastic_ss_states[experiment_label] = stoch_ss_obs

        # Store stochastic SS policies for sectoral plots (in logdev form)
        stochastic_ss_policies[experiment_label] = stoch_ss_policy

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

        # Evaluate equilibrium condition loss at stochastic steady state
        loss_results = stoch_ss_loss_fn(stoch_ss_obs, stoch_ss_policy, train_state, random.PRNGKey(config["seed"]))
        stochastic_ss_loss[experiment_label] = loss_results

        # Print loss diagnostics
        print("    Stochastic SS Loss Diagnostics:", flush=True)
        print(f"      Mean Loss (MSE): {loss_results['mean_loss']:.6e}", flush=True)
        print(f"      Mean Accuracy: {loss_results['mean_accuracy']:.6f}", flush=True)
        print(f"      Min Accuracy: {loss_results['min_accuracy']:.6f}", flush=True)

        # Calculate and store GIR (including IRs from stochastic steady state)
        gir_results = gir_fn(simul_obs, train_state, simul_policies, stoch_ss_obs)
        gir_data[experiment_label] = gir_results

    print("Data collection completed successfully.", flush=True)

    # ============================================================================
    # COMPUTE ERGODIC PRICES AND STEADY STATE CORRECTIONS
    # ============================================================================
    # Use average prices from the first experiment's ergodic distribution
    # to aggregate all simulations consistently
    first_experiment_label = list(analysis_variables_data.keys())[0]
    first_sim_data = raw_simulation_data[first_experiment_label]

    # Get ergodic prices from nonlinear simulation
    simul_policies = first_sim_data["simul_policies"]
    P_ergodic, Pk_ergodic, Pm_ergodic = compute_ergodic_prices_from_simulation(simul_policies, policies_ss, n_sectors)

    print(f"Using ergodic prices from experiment: {first_experiment_label}", flush=True)

    # Compute steady state correction factors for price-weighted aggregates
    ss_corrections = compute_ergodic_steady_state(
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        Pm_ergodic=Pm_ergodic,
        n_sectors=n_sectors,
    )

    # Print correction diagnostics
    print("  Steady state corrections (log scale):", flush=True)
    print(f"    Yagg: {ss_corrections['Yagg_correction']:.6f}", flush=True)
    print(f"    Kagg: {ss_corrections['Kagg_correction']:.6f}", flush=True)
    print(f"    Iagg: {ss_corrections['Iagg_correction']:.6f}", flush=True)
    print(f"    Magg: {ss_corrections['Magg_correction']:.6f}", flush=True)

    # ============================================================================
    # RECENTER NONLINEAR SIMULATION ANALYSIS VARIABLES
    # ============================================================================
    # Apply corrections to nonlinear simulation data so all use consistent SS
    print("\nRecentering nonlinear simulation data with ergodic-price SS...", flush=True)
    for exp_label in list(analysis_variables_data.keys()):
        analysis_variables_data[exp_label] = recenter_analysis_variables(
            analysis_variables_data[exp_label], ss_corrections
        )

    # ============================================================================
    # PROCESS DYNARE SIMULATIONS (if available)
    # ============================================================================
    # Process Dynare simulations using consistent ergodic-price aggregation
    if dynare_simul_loglin is not None:
        print("\nProcessing log-linear (Dynare) simulation with consistent aggregation...", flush=True)
        print(f"    Dynare simulation shape: {dynare_simul_loglin.shape}", flush=True)

        loglin_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_loglin,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
            burn_in=config["burn_in_periods"],
            source_label="Log-Linear (Dynare)",
        )

        # Check for NaN values in results
        for var_name, var_values in loglin_analysis_vars.items():
            n_nan = jnp.sum(jnp.isnan(var_values))
            if n_nan > 0:
                print(f"    ⚠ WARNING: {var_name} has {n_nan} NaN values!", flush=True)

        # Add to analysis data for comparison
        analysis_variables_data["Log-Linear (Dynare)"] = loglin_analysis_vars
        print("  ✓ Log-linear simulation processed with consistent aggregation.", flush=True)

    if dynare_simul_determ is not None:
        print("\nProcessing deterministic (Dynare) simulation with consistent aggregation...", flush=True)

        determ_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_determ,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
            burn_in=min(config["burn_in_periods"], dynare_simul_determ.shape[1] // 10),
            source_label="Deterministic (Dynare)",
        )

        # Add to analysis data for comparison
        analysis_variables_data["Deterministic (Dynare)"] = determ_analysis_vars
        print("  ✓ Deterministic simulation processed with consistent aggregation.", flush=True)

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
    # Filter out deterministic solution for histograms
    histogram_data = {k: v for k, v in analysis_variables_data.items() if "Deterministic" not in k}
    plot_ergodic_histograms(
        analysis_variables_data=histogram_data, save_dir=simulation_dir, analysis_name=config["analysis_name"]
    )

    # Note: plot_gir_responses uses old GIR data structure.
    # Combined IR plots below handle both positive and negative shocks.

    # ============================================================================
    # COMBINED IR ANALYSIS: MATLAB + JAX GIRs
    # ============================================================================
    print("\n" + "=" * 60, flush=True)
    print("COMBINED IR ANALYSIS: MATLAB + JAX GIRs", flush=True)
    print("=" * 60, flush=True)

    # The load_matlab_irs function automatically searches for:
    # 1. ModelData_IRs.mat in the model directory
    # 2. ModelData_IRs.mat in experiment subfolders
    # 3. Legacy format files in MATLAB/IRs folder (fallback)
    matlab_ir_dir = os.path.join(model_dir, "MATLAB", "IRs")
    matlab_ir_data = {}

    print("\nLooking for MATLAB IR data...", flush=True)
    matlab_ir_data = load_matlab_irs(
        matlab_ir_dir=matlab_ir_dir,
        shock_sizes=config.get("ir_shock_sizes", [5, 10, 20]),
    )

    if matlab_ir_data:
        print(f"\n  ✓ Successfully loaded {len(matlab_ir_data)} shock configurations", flush=True)
        for key in matlab_ir_data.keys():
            n_sectors_loaded = len(matlab_ir_data[key].get("sectors", {}))
            print(f"    - {key}: {n_sectors_loaded} sectors", flush=True)
    else:
        print("\n  ✗ No MATLAB IR data was loaded", flush=True)
        print("    To use MATLAB IRs, either:", flush=True)
        print("    1. Save ModelData_IRs.mat in the model directory, or", flush=True)
        print("    2. Run MATLAB main.m with config.save_results = true", flush=True)

    # Generate combined IR plots (with or without MATLAB data)
    print("\nGenerating IR comparison plots...", flush=True)

    sectors_to_plot = config.get("ir_sectors_to_plot", [0, 2, 23])
    ir_variables = config.get("ir_variables_to_plot", ["Agg. Consumption"])
    shock_sizes = config.get("ir_shock_sizes", [5, 10, 20])
    max_periods = config.get("ir_max_periods", 80)

    print(f"  Variables: {ir_variables}", flush=True)
    print(f"  Shock sizes: {shock_sizes}", flush=True)
    print(f"  Sectors to plot: {sectors_to_plot}", flush=True)

    total_plots = len(sectors_to_plot) * len(ir_variables)
    plot_idx = 0

    for sector_idx in sectors_to_plot:
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )

        for ir_variable in ir_variables:
            plot_idx += 1
            print(
                f"\n  [{plot_idx}/{total_plots}] Sector {sector_idx} ({sector_label}), Variable: {ir_variable}...",
                flush=True,
            )

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

    # ============================================================================
    # STOCHASTIC STEADY STATE ANALYSIS: Capital Distribution Plots
    # ============================================================================
    print("\nGenerating stochastic steady state plots...", flush=True)

    # Plot stochastic SS sectoral capital distribution (comparison across experiments)
    if stochastic_ss_states:
        try:
            plot_sectoral_capital_stochss(
                stochastic_ss_states=stochastic_ss_states,
                save_dir=simulation_dir,
                analysis_name=config["analysis_name"],
                econ_model=econ_model,
            )
            print("  ✓ Stochastic SS sectoral capital plot generated", flush=True)
        except Exception as e:
            print(f"  ✗ Failed to create stochastic SS capital plot: {e}", flush=True)

    # Plot stochastic SS sectoral distributions for L, Y, M, Q
    if stochastic_ss_policies:
        for var_name in ["L", "Y", "M", "Q"]:
            try:
                plot_sectoral_variable_stochss(
                    stochastic_ss_states=stochastic_ss_states,
                    stochastic_ss_policies=stochastic_ss_policies,
                    variable_name=var_name,
                    save_dir=simulation_dir,
                    analysis_name=config["analysis_name"],
                    econ_model=econ_model,
                )
                print(f"  ✓ Stochastic SS sectoral {var_name} plot generated", flush=True)
            except Exception as e:
                print(f"  ✗ Failed to create stochastic SS {var_name} plot: {e}", flush=True)

    # ============================================================================
    # ERGODIC DISTRIBUTION ANALYSIS: Sectoral Variable Plots
    # ============================================================================
    print("\nGenerating ergodic distribution plots...", flush=True)

    # Plot ergodic mean sectoral distributions for K, L, Y, M, Q
    if raw_simulation_data:
        for var_name in ["K", "L", "Y", "M", "Q"]:
            try:
                plot_sectoral_variable_ergodic(
                    raw_simulation_data=raw_simulation_data,
                    variable_name=var_name,
                    save_dir=simulation_dir,
                    analysis_name=config["analysis_name"],
                    econ_model=econ_model,
                )
                print(f"  ✓ Ergodic sectoral {var_name} plot generated", flush=True)
            except Exception as e:
                print(f"  ✗ Failed to create ergodic {var_name} plot: {e}", flush=True)

    # Plot comparison of ergodic mean vs stochastic SS for each experiment
    for experiment_label, sim_data in raw_simulation_data.items():
        if experiment_label in stochastic_ss_states:
            try:
                plot_sectoral_capital_comparison(
                    simul_obs=sim_data["simul_obs"],
                    stochastic_ss_state=stochastic_ss_states[experiment_label],
                    save_dir=simulation_dir,
                    analysis_name=config["analysis_name"],
                    econ_model=econ_model,
                    experiment_label=experiment_label,
                )
                print(f"  ✓ Capital comparison plot for {experiment_label}", flush=True)
            except Exception as e:
                print(f"  ✗ Failed to create capital comparison for {experiment_label}: {e}", flush=True)

    print("Analysis completed successfully.", flush=True)

    return {
        "analysis_variables_data": analysis_variables_data,
        "raw_simulation_data": raw_simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
        "stochastic_ss_states": stochastic_ss_states,
        "stochastic_ss_policies": stochastic_ss_policies,
        "stochastic_ss_loss": stochastic_ss_loss,
        "gir_data": gir_data,
        "matlab_ir_data": matlab_ir_data,
    }


if __name__ == "__main__":
    main()
