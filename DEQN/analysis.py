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
from DEQN.analysis.model_hooks import (  # noqa: E402
    apply_model_config_defaults,
    compute_analysis_variables,
    get_states_to_shock,
    load_model_analysis_hooks,
    run_model_postprocess,
)
from DEQN.analysis.plots import plot_ergodic_histograms  # noqa: E402
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    simulation_analysis,
)
from DEQN.analysis.stochastic_ss import (  # noqa: E402
    create_stochss_fn,
    create_stochss_loss_fn,
)
from DEQN.analysis.tables import (  # noqa: E402
    create_calibration_table,
    create_comparative_stats_table,
    create_descriptive_stats_table,
    create_ergodic_aggregate_stats_table,
    create_stochastic_ss_aggregates_table,
    create_stochastic_ss_table,
    create_welfare_table,
)
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.training.checkpoints import (  # noqa: E402
    load_experiment_data,
    load_trained_model_orbax,
)

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "model_dir": "RbcProdNet_Dec2025",
    "analysis_name": "benchmarkMarch",
    # MATLAB data files (relative to model_dir)
    # Set to None to use defaults: "ModelData.mat", "ModelData_IRs.mat", "ModelData_simulation.mat"
    "model_data_file": "ModelData_benchMar.mat",
    "model_data_irs_file": "ModelData_IRs_benchMar.mat",
    "model_data_simulation_file": "ModelData_simulation_benchMar.mat",  # Set to None to skip MATLAB simulation comparison
    # Experiments to analyze
    "experiments_to_analyze": {
        "Benchmark March": "sigl0dot5capadj3epsl0dot05",
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
    "welfare_traject_length": 200,
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
    # IR method(s): choose any of ["GIR", "IR_stoch_ss"].
    # - GIR: average over ergodic draws
    # - IR_stoch_ss: single IR from stochastic steady state
    "ir_methods": ["IR_stoch_ss"],
    # MATLAB benchmark used in IR figures. Options:
    # "FirstOrder", "SecondOrder", "PerfectForesight"
    "ir_benchmark_method": "PerfectForesight",
    # Combined IR analysis configuration
    # Sectors to analyze: specify sector indices (0-based).
    # GIRs shock the TFP/productivity state (state index = n_sectors + sector_idx).
    # For example, sector 0 TFP is at state index 37 (for n_sectors=37).
    "ir_sectors_to_plot": [0],
    "ir_variables_to_plot": ["Agg. Consumption", "Agg. Investment", "Agg. GDP", "Agg. Capital"],
    "sectoral_ir_variables_to_plot": [
        "Cj",
        "Pj",
        "Ioutj",
        "Moutj",
        "Lj",
        "Ij",
        "Mj",
        "Yj",
        "Qj",
        "Kj",
        "Cj_client",
        "Pj_client",
        "Ioutj_client",
        "Moutj_client",
        "Lj_client",
        "Ij_client",
        "Mj_client",
        "Yj_client",
        "Qj_client",
        "Pmj_client",
        "gammaij_client",
    ],
    "ir_shock_sizes": [10, 20, 30],
    "ir_max_periods": 40,
    # Aggregate reporting controls
    "aggregate_variables": ["Agg. Consumption", "Agg. Investment", "Agg. GDP", "Agg. Capital"],
    "descriptive_stats_variables": ["Agg. Consumption", "Agg. Investment", "Agg. GDP", "Agg. Capital"],
    # Benchmark methods included in ergodic exercises (descriptive table, aggregate stats, histograms).
    # Nonlinear experiment methods are always included when always_include_nonlinear_methods=True.
    # Canonical names: "Log-Linear", "SecondOrder", "PerfectForesight", "MITShocks".
    # Alias supported: "FirstOrder" -> "Log-Linear".
    "ergodic_methods_to_include": [],
    "always_include_nonlinear_methods": True,
    # Methods included in stochastic-SS aggregate table.
    # Use None to include all analyzed experiments.
    "stochss_methods_to_include": [],
    # JAX configuration
    "double_precision": True,
}

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model
analysis_hooks = load_model_analysis_hooks(config["model_dir"])
config = apply_model_config_defaults(config, analysis_hooks)

# Import model-specific plots module and registry if available
plots_module_name = f"DEQN.econ_models.{config['model_dir']}.plots"
try:
    plots_module = importlib.import_module(plots_module_name)
except ModuleNotFoundError as exc:
    if exc.name == plots_module_name:
        plots_module = None
    else:
        raise

MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", []) if plots_module is not None else []


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    model_dir = os.path.join(base_dir, config["model_dir"])
    save_dir = os.path.join(model_dir, "experiments/")

    analysis_dir = os.path.join(model_dir, "analysis", config["analysis_name"])
    simulation_dir = os.path.join(analysis_dir, "simulation")
    irs_dir = os.path.join(analysis_dir, "IRs")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(irs_dir, exist_ok=True)

    config_path = os.path.join(analysis_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD MODEL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    # Determine file names from config (with defaults)
    model_data_file = config.get("model_data_file", "ModelData.mat")
    model_data_irs_file = config.get("model_data_irs_file", "ModelData_IRs.mat")
    model_data_simulation_file = config.get("model_data_simulation_file", "ModelData_simulation.mat")

    model_path = os.path.join(model_dir, model_data_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading ModelData from: {model_data_file}")
    model_data = sio.loadmat(model_path, simplify_cells=True)

    if "ModelData" not in model_data:
        raise ValueError("Expected 'ModelData' key in model file.")

    md = model_data["ModelData"]

    ss = md["SteadyState"]
    n_sectors = ss["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    k_ss = jnp.array(ss["endostates_ss"], dtype=precision)
    state_ss = jnp.concatenate([k_ss, a_ss])
    params = ss["parameters"]
    policies_ss = jnp.array(ss["policies_ss"], dtype=precision)

    stats = md["Statistics"]
    state_sd = jnp.array(stats["states_sd"], dtype=precision)
    policies_sd = jnp.array(stats["policies_sd"], dtype=precision)

    C_matrix = md["Solution"]["StateSpace"]["C"]

    if len(policies_ss) != len(policies_sd):
        n_policies = len(policies_sd)
        policies_ss = policies_ss[:n_policies]

    # Load simulation data (optional - for Dynare comparison)
    dynare_simul_1storder = None
    dynare_simul_so = None
    dynare_simul_pf = None
    dynare_simul_mit = None

    if model_data_simulation_file is not None:
        simul_path = os.path.join(model_dir, model_data_simulation_file)
        if os.path.exists(simul_path):
            print(f"  Loading simulation data from: {model_data_simulation_file}")
            simul_data = sio.loadmat(simul_path, simplify_cells=True)
            simul = simul_data.get("ModelData_simulation", {})

            # First-order simulation (new: FirstOrder, legacy: Loglin)
            if "FirstOrder" in simul and "full_simul" in simul["FirstOrder"]:
                dynare_simul_1storder = jnp.array(simul["FirstOrder"]["full_simul"], dtype=precision)
            elif "Loglin" in simul and "full_simul" in simul["Loglin"]:
                dynare_simul_1storder = jnp.array(simul["Loglin"]["full_simul"], dtype=precision)

            # Perfect foresight simulation (new: PerfectForesight, legacy: Determ)
            if "PerfectForesight" in simul and "full_simul" in simul["PerfectForesight"]:
                dynare_simul_pf = jnp.array(simul["PerfectForesight"]["full_simul"], dtype=precision)
            elif "Determ" in simul and "full_simul" in simul["Determ"]:
                dynare_simul_pf = jnp.array(simul["Determ"]["full_simul"], dtype=precision)

            # Second-order simulation
            if "SecondOrder" in simul and "full_simul" in simul["SecondOrder"]:
                dynare_simul_so = jnp.array(simul["SecondOrder"]["full_simul"], dtype=precision)
            # MIT shocks simulation
            if "MITShocks" in simul and "full_simul" in simul["MITShocks"]:
                dynare_simul_mit = jnp.array(simul["MITShocks"]["full_simul"], dtype=precision)
        else:
            print(f"  ⚠ Simulation file not found: {model_data_simulation_file} (skipping)")

    # Load IRF data path for MATLAB IR comparison (actual loading happens later via load_matlab_irs)
    irs_path = os.path.join(model_dir, model_data_irs_file) if model_data_irs_file else None
    if irs_path and os.path.exists(irs_path):
        print(f"  Found IRs file: {model_data_irs_file}")
    elif model_data_irs_file:
        print(f"  ⚠ IRs file not found: {model_data_irs_file} (will try legacy format)")

    # Create economic model
    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
    )

    # Load experiment data
    experiments_to_analyze = config["experiments_to_analyze"]
    experiments_data = load_experiment_data(experiments_to_analyze, save_dir)

    nn_config_base = {
        "C": C_matrix,
        "states_sd": state_sd,
        "policies_sd": policies_sd,
        "params_dtype": precision,
    }

    config["states_to_shock"] = get_states_to_shock(
        config=config,
        econ_model=econ_model,
        analysis_hooks=analysis_hooks,
    )

    # Keep welfare baseline available for all methods.
    welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)

    def _normalize_dynare_full_simul(simul_data, state_ss_vec, policies_ss_vec):
        """Return simulation in log deviations with shape (n_vars, T)."""
        expected_n_vars = state_ss_vec.shape[0] + policies_ss_vec.shape[0]
        if simul_data.shape[0] == expected_n_vars:
            simul_matrix = simul_data
        elif simul_data.shape[1] == expected_n_vars:
            simul_matrix = simul_data.T
        else:
            raise ValueError(
                f"Unexpected Dynare simulation shape {simul_data.shape}; expected one axis = {expected_n_vars}."
            )

        ss_full = jnp.concatenate([state_ss_vec, policies_ss_vec])
        dist_to_zero = jnp.mean(jnp.abs(simul_matrix[:, 0]))
        dist_to_ss = jnp.mean(jnp.abs(simul_matrix[:, 0] - ss_full))
        if dist_to_ss < dist_to_zero:
            simul_matrix = simul_matrix - ss_full[:, None]
        return simul_matrix

    def _welfare_cost_from_dynare_simul(simul_data, method_name):
        """Compute consumption-equivalent welfare cost from Dynare full_simul."""
        if simul_data is None:
            return None
        simul_matrix = _normalize_dynare_full_simul(simul_data, state_ss, policies_ss)
        n_periods = simul_matrix.shape[1]
        burn_in = min(config["burn_in_periods"], max(0, n_periods // 10))
        if burn_in >= n_periods:
            burn_in = 0
        policies_logdev = simul_matrix[2 * n_sectors :, burn_in:].T

        if policies_logdev.shape[0] <= config["welfare_traject_length"]:
            print(
                f"  ⚠ Skipping welfare for {method_name}: not enough periods ({policies_logdev.shape[0]}) "
                f"for welfare_traject_length={config['welfare_traject_length']}"
            )
            return None

        method_seed = sum(ord(c) for c in method_name)
        welfare = welfare_fn(
            jax.vmap(econ_model.utility_from_policies)(policies_logdev),
            welfare_ss,
            random.PRNGKey(config["welfare_seed"] + method_seed),
        )
        Vc = econ_model.consumption_equivalent(welfare)
        return -Vc * 100

    # Create analysis functions
    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config))
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, config))
    stoch_ss_loss_fn = create_stochss_loss_fn(econ_model, mc_draws=32)
    gir_fn = jax.jit(create_GIR_fn(econ_model, config, analysis_hooks=analysis_hooks))

    # Storage for analysis results
    analysis_variables_data = {}
    raw_simulation_data = {}
    welfare_costs = {}
    stochastic_ss_data = {}
    stochastic_ss_states = {}
    stochastic_ss_policies = {}
    stochastic_ss_loss = {}
    gir_data = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION & WELFARE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SIMULATION & WELFARE RESULTS")
    print("═" * 72)
    print(f"  Analysis: {config['analysis_name']}")
    print(f"  Model: {config['model_dir']}")
    print(f"  Experiments: {list(experiments_to_analyze.keys())}")
    print(f"  Sectors: {n_sectors}")
    print("─" * 72, flush=True)

    for experiment_label, exp_data in experiments_data.items():
        print(f"\n  ▶ {experiment_label}", flush=True)

        experiment_config = exp_data["config"]
        experiment_name = exp_data["results"]["exper_name"]

        nn_config = nn_config_base.copy()
        nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

        train_state = load_trained_model_orbax(experiment_name, save_dir, nn_config, econ_model.state_ss)

        simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
            train_state,
            econ_model,
            config,
            simulation_fn,
            analysis_hooks=analysis_hooks,
        )

        raw_simulation_data[experiment_label] = {
            "simul_obs": simul_obs,
            "simul_policies": simul_policies,
            "simul_analysis_variables": simul_analysis_variables,
            "analysis_context": analysis_context,
        }

        analysis_variables_data[experiment_label] = simul_analysis_variables

        simul_utilities = jax.vmap(econ_model.utility_from_policies)(simul_policies)

        welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config["welfare_seed"]))

        Vc = econ_model.consumption_equivalent(welfare)
        welfare_cost_ce = -Vc * 100
        welfare_costs[experiment_label] = welfare_cost_ce

        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
        if stoch_ss_obs_std.max() > 0.001:
            raise ValueError("Stochastic steady state standard deviation too large")

        stochastic_ss_states[experiment_label] = stoch_ss_obs
        stochastic_ss_policies[experiment_label] = stoch_ss_policy

        stoch_ss_analysis_variables = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=stoch_ss_obs,
            policy_logdev=stoch_ss_policy,
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )

        stochastic_ss_data[experiment_label] = stoch_ss_analysis_variables

        loss_results = stoch_ss_loss_fn(stoch_ss_obs, stoch_ss_policy, train_state, random.PRNGKey(config["seed"]))
        stochastic_ss_loss[experiment_label] = loss_results

        print(f"    Welfare cost (CE): {welfare_cost_ce:.4f}%")
        print(
            f"    Equilibrium accuracy: {loss_results['mean_accuracy']:.4f} (min: {loss_results['min_accuracy']:.4f})",
            flush=True,
        )

        gir_results = gir_fn(simul_obs, train_state, simul_policies, stoch_ss_obs)
        gir_data[experiment_label] = gir_results

    # Add welfare costs from Dynare simulation methods (if available).
    dynare_welfare_inputs = {
        "FirstOrder": dynare_simul_1storder,
        "SecondOrder": dynare_simul_so,
        "PerfectForesight": dynare_simul_pf,
        "MITShocks": dynare_simul_mit,
    }
    for method_name, simul_data in dynare_welfare_inputs.items():
        welfare_cost = _welfare_cost_from_dynare_simul(simul_data, method_name)
        if welfare_cost is not None:
            welfare_costs[method_name] = welfare_cost
            print(f"    Welfare cost ({method_name}): {float(welfare_cost):.4f}%")

    theoretical_stats = {}
    histogram_theo_params = {}
    matlab_ir_data = None
    upstreamness_data = None

    model_postprocess = run_model_postprocess(
        analysis_hooks=analysis_hooks,
        config=config,
        model_dir=model_dir,
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        irs_dir=irs_dir,
        econ_model=econ_model,
        model_data=md,
        stats=stats,
        policies_ss=policies_ss,
        state_ss=state_ss,
        raw_simulation_data=raw_simulation_data,
        analysis_variables_data=analysis_variables_data,
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
        stochastic_ss_data=stochastic_ss_data,
        gir_data=gir_data,
        dynare_simulations=dynare_welfare_inputs,
        irs_path=irs_path,
    )

    analysis_variables_data = model_postprocess.get("analysis_variables_data", analysis_variables_data)
    theoretical_stats = model_postprocess.get("theoretical_stats", theoretical_stats)
    histogram_theo_params = model_postprocess.get("histogram_theo_params", histogram_theo_params)
    matlab_ir_data = model_postprocess.get("matlab_ir_data", matlab_ir_data)
    upstreamness_data = model_postprocess.get("upstreamness_data", upstreamness_data)

    # ═══════════════════════════════════════════════════════════════════════════
    # CALIBRATION TABLE (First-Order Model vs Empirical Targets)
    # ═══════════════════════════════════════════════════════════════════════════
    calibration_emp = (
        md.get("EmpiricalTargets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("empirical_targets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("EmpiricalTargets")
    )
    fo_stats = stats.get("FirstOrder") or stats.get("firstorder")
    calibration_model_stats = (
        (fo_stats.get("ModelStats") or fo_stats.get("modelstats")) if isinstance(fo_stats, dict) else None
    )
    if calibration_emp is not None:
        create_calibration_table(
            empirical_targets=calibration_emp,
            first_order_model_stats=calibration_model_stats,
            save_path=os.path.join(analysis_dir, "calibration_table.tex"),
            analysis_name=config["analysis_name"],
        )
    else:
        print(
            "  ⚠ Calibration table skipped: no empirical targets in ModelData (EmpiricalTargets / calibration.empirical_targets)."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STOCHASTIC STEADY STATE
    # ═══════════════════════════════════════════════════════════════════════════
    create_stochastic_ss_aggregates_table(
        stochastic_ss_data=stochastic_ss_data,
        save_path=os.path.join(analysis_dir, "stochastic_ss_aggregates_table.tex"),
        analysis_name=config["analysis_name"],
        methods_to_include=config.get("stochss_methods_to_include"),
    )

    create_stochastic_ss_table(
        stochastic_ss_data=stochastic_ss_data,
        save_path=os.path.join(analysis_dir, "stochastic_ss_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # ERGODIC DISTRIBUTION MOMENTS
    # ═══════════════════════════════════════════════════════════════════════════
    method_aliases = {
        "FirstOrder": "Log-Linear",
        "LogLinear": "Log-Linear",
        "Second-Order": "SecondOrder",
        "Perfect Foresight": "PerfectForesight",
        "MITShock": "MITShocks",
    }

    def _normalize_method_name(method_name: str) -> str:
        return method_aliases.get(method_name, method_name)

    available_methods = sorted(
        {_normalize_method_name(k) for k in analysis_variables_data.keys()}
        | {_normalize_method_name(k) for k in theoretical_stats.keys()}
    )
    print(f"  Available methods (canonical): {available_methods}")

    benchmark_methods_cfg = config.get("ergodic_methods_to_include")
    benchmark_methods = None
    if benchmark_methods_cfg is not None:
        if isinstance(benchmark_methods_cfg, str):
            benchmark_methods_cfg = [benchmark_methods_cfg]
        benchmark_methods = {_normalize_method_name(m) for m in benchmark_methods_cfg}

    selected_methods = set(available_methods) if benchmark_methods is None else set(benchmark_methods)
    if config.get("always_include_nonlinear_methods", True):
        selected_methods.update(experiments_to_analyze.keys())

    filtered_analysis_variables_data = {
        _normalize_method_name(k): v
        for k, v in analysis_variables_data.items()
        if _normalize_method_name(k) in selected_methods
    }
    filtered_theoretical_stats = {
        _normalize_method_name(k): v
        for k, v in theoretical_stats.items()
        if _normalize_method_name(k) in selected_methods
    }

    desc_vars = config.get("descriptive_stats_variables")
    if desc_vars:
        desc_analysis_data = {
            method: {k: v for k, v in variables.items() if k in desc_vars}
            for method, variables in filtered_analysis_variables_data.items()
        }
        desc_analysis_data = {k: v for k, v in desc_analysis_data.items() if v}
        desc_theo_stats = None
        if filtered_theoretical_stats:
            desc_theo_stats = {
                method: {k: v for k, v in variables.items() if k in desc_vars}
                for method, variables in filtered_theoretical_stats.items()
            }
            desc_theo_stats = {k: v for k, v in desc_theo_stats.items() if v} or None
    else:
        desc_analysis_data = filtered_analysis_variables_data
        desc_theo_stats = filtered_theoretical_stats if filtered_theoretical_stats else None

    create_descriptive_stats_table(
        analysis_variables_data=desc_analysis_data,
        save_path=os.path.join(simulation_dir, "descriptive_stats_table.tex"),
        analysis_name=config["analysis_name"],
        theoretical_stats=desc_theo_stats,
    )

    # Aggregate-only ergodic descriptive statistics (C, I, GDP, K)
    aggregate_vars = config.get(
        "aggregate_variables", ["Agg. Consumption", "Agg. Investment", "Agg. GDP", "Agg. Capital"]
    )
    aggregate_ergodic_data = {}
    for method_name, variables in filtered_analysis_variables_data.items():
        filtered = {k: v for k, v in variables.items() if k in aggregate_vars}
        if filtered:
            aggregate_ergodic_data[method_name] = filtered

    create_ergodic_aggregate_stats_table(
        analysis_variables_data=aggregate_ergodic_data,
        save_path=os.path.join(simulation_dir, "ergodic_aggregate_stats_table.tex"),
        analysis_name=config["analysis_name"],
        methods_to_include=list(selected_methods),
    )

    if len(filtered_analysis_variables_data) > 1:
        create_comparative_stats_table(
            analysis_variables_data=filtered_analysis_variables_data,
            save_path=os.path.join(simulation_dir, "descriptive_stats_comparative.tex"),
            analysis_name=config["analysis_name"],
        )

    # Generate histograms:
    # - keep model simulation distributions
    # - keep theoretical log-linear distribution
    histogram_data = {k: v for k, v in filtered_analysis_variables_data.items() if "Deterministic" not in k}
    histogram_data = {
        k: {var: arr for var, arr in v.items() if var in aggregate_vars} for k, v in histogram_data.items()
    }
    histogram_data = {k: v for k, v in histogram_data.items() if v}
    filtered_histogram_theo_params = {
        _normalize_method_name(k): v
        for k, v in histogram_theo_params.items()
        if _normalize_method_name(k) in selected_methods
    }
    plot_ergodic_histograms(
        analysis_variables_data=histogram_data,
        save_dir=simulation_dir,
        analysis_name=config["analysis_name"],
        theo_dist_params=filtered_histogram_theo_params if filtered_histogram_theo_params else None,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # WELFARE COSTS
    # ═══════════════════════════════════════════════════════════════════════════
    create_welfare_table(
        welfare_data=welfare_costs,
        save_path=os.path.join(analysis_dir, "welfare_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # Model-specific plots
    if MODEL_SPECIFIC_PLOTS:
        for plot_spec in MODEL_SPECIFIC_PLOTS:
            plot_name = plot_spec["name"]
            plot_function = plot_spec["function"]

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
                except Exception as e:
                    print(f"    ✗ Failed to create {plot_name} for {experiment_label}: {e}", flush=True)

    print("\n" + "═" * 72)
    print("  ANALYSIS COMPLETE")
    print("═" * 72, flush=True)

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
        "upstreamness_data": upstreamness_data,
        "theoretical_stats": theoretical_stats,  # Pre-computed stats for log-linear and perfect foresight
    }


if __name__ == "__main__":
    main()
