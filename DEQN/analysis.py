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
    get_shock_dimension,
    get_states_to_shock,
    load_model_analysis_hooks,
    run_model_postprocess,
)
from DEQN.analysis.plots import plot_ergodic_histograms  # noqa: E402
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    create_shock_path_simulation_fn,
    simulation_analysis,
    simulation_analysis_with_shocks,
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
    "model_dir": "RbcProdNet_March2026",
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
# ANALYSIS HELPERS
# ============================================================================


def _write_analysis_config(config_dict, analysis_dir):
    config_path = os.path.join(analysis_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def _resolve_data_file(model_dir, configured_name, fallback_names, *, label, required):
    candidate_names = []
    for name in [configured_name, *fallback_names]:
        if name is None or name in candidate_names:
            continue
        candidate_names.append(name)

    for filename in candidate_names:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            if configured_name is not None and filename != configured_name:
                print(f"  Using fallback {label} file: {filename} (configured '{configured_name}' not found)")
            return filename, path

    if required:
        raise FileNotFoundError(
            f"{label} file not found in {model_dir}. Tried: {candidate_names}"
        )

    if configured_name is not None:
        print(f"  ⚠ {label} file not found. Tried: {candidate_names} (skipping)")
    return None, None


def _create_nonlinear_simulation_runner(
    *,
    econ_model,
    config_dict,
    analysis_hooks,
    matlab_common_shock_schedule,
):
    if matlab_common_shock_schedule is not None:
        shock_path_simulation_fn = jax.jit(create_shock_path_simulation_fn(econ_model))
        print(
            "  Nonlinear simulation mode: shared MATLAB shock path "
            f"({matlab_common_shock_schedule['reference_method']})"
        )

        def run_simulation(train_state):
            return simulation_analysis_with_shocks(
                train_state=train_state,
                econ_model=econ_model,
                shock_path=matlab_common_shock_schedule["full_shocks"],
                simulation_fn=shock_path_simulation_fn,
                active_start=matlab_common_shock_schedule["active_start"],
                active_end=matlab_common_shock_schedule["active_end"],
                analysis_config=config_dict,
                analysis_hooks=analysis_hooks,
                label=(f"Common-shock nonlinear simulation ({matlab_common_shock_schedule['reference_method']})"),
            )

        return run_simulation

    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config_dict))
    print("  Nonlinear simulation mode: ergodic fallback (no MATLAB common-shock schedule found)")

    def run_simulation(train_state):
        simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
            train_state=train_state,
            econ_model=econ_model,
            analysis_config=config_dict,
            simulation_fn=simulation_fn,
            analysis_hooks=analysis_hooks,
        )
        return (
            simul_obs,
            simul_policies,
            simul_analysis_variables,
            analysis_context,
            simul_obs,
            simul_policies,
        )

    return run_simulation


def _run_experiment_analysis(
    *,
    exp_data,
    save_dir,
    nn_config_base,
    econ_model,
    run_nonlinear_simulation,
    welfare_fn,
    welfare_ss,
    stoch_ss_fn,
    stoch_ss_loss_fn,
    gir_fn,
    config_dict,
    analysis_hooks,
):
    experiment_config = exp_data["config"]
    experiment_name = exp_data["results"]["exper_name"]

    nn_config = nn_config_base.copy()
    nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

    train_state = load_trained_model_orbax(experiment_name, save_dir, nn_config, econ_model.state_ss)

    (
        simul_obs,
        simul_policies,
        simul_analysis_variables,
        analysis_context,
        simul_obs_full,
        simul_policies_full,
    ) = run_nonlinear_simulation(train_state)

    simul_utilities = jax.vmap(econ_model.utility_from_policies)(simul_policies)
    welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config_dict["welfare_seed"]))
    welfare_cost_ce = -econ_model.consumption_equivalent(welfare) * 100

    stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
    if stoch_ss_obs_std.max() > 0.001:
        raise ValueError("Stochastic steady state standard deviation too large")

    stoch_ss_analysis_variables = compute_analysis_variables(
        econ_model=econ_model,
        state_logdev=stoch_ss_obs,
        policy_logdev=stoch_ss_policy,
        analysis_context=analysis_context,
        analysis_hooks=analysis_hooks,
    )

    loss_results = stoch_ss_loss_fn(stoch_ss_obs, stoch_ss_policy, train_state, random.PRNGKey(config_dict["seed"]))
    gir_results = gir_fn(simul_obs, train_state, simul_policies, stoch_ss_obs)

    print(f"    Welfare cost (CE): {welfare_cost_ce:.4f}%")
    print(
        f"    Equilibrium accuracy: {loss_results['mean_accuracy']:.4f} (min: {loss_results['min_accuracy']:.4f})",
        flush=True,
    )

    return {
        "raw_simulation_data": {
            "simul_obs": simul_obs,
            "simul_obs_full": simul_obs_full,
            "simul_policies": simul_policies,
            "simul_policies_full": simul_policies_full,
            "simul_analysis_variables": simul_analysis_variables,
            "analysis_context": analysis_context,
        },
        "analysis_variables": simul_analysis_variables,
        "welfare_cost": welfare_cost_ce,
        "stochastic_ss_state": stoch_ss_obs,
        "stochastic_ss_policy": stoch_ss_policy,
        "stochastic_ss_data": stoch_ss_analysis_variables,
        "stochastic_ss_loss": loss_results,
        "gir_data": gir_results,
    }


def _normalize_dynare_simulation_orientation(simul_matrix, expected_n_vars, precision):
    arr = jnp.array(simul_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((expected_n_vars, 0), dtype=precision)
        if arr.size == expected_n_vars:
            return arr.reshape(expected_n_vars, 1)
        raise ValueError(
            "Unexpected 1D Dynare simulation vector with "
            f"length={arr.size}; expected length {expected_n_vars} for a single-period slice."
        )
    if arr.ndim != 2:
        raise ValueError(f"Unexpected Dynare simulation ndim={arr.ndim}; expected 1 or 2.")
    if arr.shape[0] == expected_n_vars:
        return arr
    if arr.shape[1] == expected_n_vars:
        return arr.T
    raise ValueError(f"Unexpected Dynare simulation shape {arr.shape}; expected one axis = {expected_n_vars}.")


def _extract_dynare_simulation_artifact(simul, method_names, expected_n_vars, precision):
    """Load active/full simulation paths for one Dynare method."""
    for method_name in method_names:
        method_block = simul.get(method_name)
        if not isinstance(method_block, dict):
            continue

        full_simul = None
        active_simul = None

        full_simul_raw = method_block.get("full_simul")
        if full_simul_raw is not None:
            full_simul = _normalize_dynare_simulation_orientation(full_simul_raw, expected_n_vars, precision)

        burnin_simul = method_block.get("burnin_simul")
        shocks_simul = method_block.get("shocks_simul")
        burnout_simul = method_block.get("burnout_simul")

        if shocks_simul is not None:
            active_simul = _normalize_dynare_simulation_orientation(shocks_simul, expected_n_vars, precision)

        if full_simul is None:
            windows = []
            for window in (burnin_simul, shocks_simul, burnout_simul):
                if window is None:
                    continue
                window_arr = _normalize_dynare_simulation_orientation(window, expected_n_vars, precision)
                if window_arr.shape[1] > 0:
                    windows.append(window_arr)
            if windows:
                full_simul = jnp.concatenate(windows, axis=1)

        if active_simul is None and full_simul is not None:
            burn_in = int(method_block.get("burn_in", 0))
            t_active = method_block.get("T_active")
            if t_active is not None:
                t_active = int(t_active)
                active_simul = full_simul[:, burn_in : burn_in + t_active]
            else:
                active_simul = full_simul

        if active_simul is not None or full_simul is not None:
            return {
                "active_simul": active_simul if active_simul is not None else full_simul,
                "full_simul": full_simul if full_simul is not None else active_simul,
            }

    return {"active_simul": None, "full_simul": None}


def _normalize_shock_matrix(shocks_matrix, shock_dimension, precision):
    arr = jnp.array(shocks_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((0, shock_dimension), dtype=precision)
        if arr.size == shock_dimension:
            return arr.reshape(1, shock_dimension)
        raise ValueError(
            "Unexpected 1D shock vector with "
            f"length={arr.size}; expected length {shock_dimension} for a single-period shock path."
        )
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D shock matrix, got shape {arr.shape}")
    if arr.shape[1] == shock_dimension:
        return arr
    if arr.shape[0] == shock_dimension:
        return arr.T
    raise ValueError(f"Unexpected shock matrix shape {arr.shape} for shock_dimension={shock_dimension}")


def _extract_matlab_common_shock_schedule(simul, shock_dimension, precision):
    shocks_block = simul.get("Shocks")
    if not isinstance(shocks_block, dict) or "data" not in shocks_block:
        return None

    active_shocks_full = _normalize_shock_matrix(shocks_block["data"], shock_dimension, precision)
    usage = shocks_block.get("usage", {})

    candidate_methods = [
        ("FirstOrder", ["FirstOrder", "Loglin", "LogLinear"]),
        ("SecondOrder", ["SecondOrder"]),
        ("PerfectForesight", ["PerfectForesight", "Determ"]),
        ("MITShocks", ["MITShocks", "MITShock"]),
    ]

    selected_method = None
    method_block = None
    usage_block = None
    for canonical_name, aliases in candidate_methods:
        for alias in aliases:
            block = simul.get(alias)
            if not isinstance(block, dict):
                continue
            usage_candidate = usage.get(alias) or usage.get(canonical_name)
            if usage_candidate is not None or "T_active" in block or "burn_in" in block:
                selected_method = canonical_name
                method_block = block
                usage_block = usage_candidate
                break
        if selected_method is not None:
            break

    if method_block is None:
        return None

    active_shocks = active_shocks_full
    if isinstance(usage_block, dict) and "start" in usage_block and "end" in usage_block:
        start_idx = max(int(usage_block["start"]) - 1, 0)
        end_idx = min(int(usage_block["end"]), active_shocks_full.shape[0])
        active_shocks = active_shocks_full[start_idx:end_idx]

    burn_in = int(method_block.get("burn_in", 0))
    burn_out = int(method_block.get("burn_out", 0))
    zero_burnin = jnp.zeros((burn_in, shock_dimension), dtype=precision)
    zero_burnout = jnp.zeros((burn_out, shock_dimension), dtype=precision)
    full_shocks = jnp.concatenate([zero_burnin, active_shocks, zero_burnout], axis=0)

    return {
        "reference_method": selected_method,
        "active_shocks": active_shocks,
        "full_shocks": full_shocks,
        "burn_in": burn_in,
        "burn_out": burn_out,
        "active_start": burn_in,
        "active_end": burn_in + active_shocks.shape[0],
    }


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


def _welfare_cost_from_dynare_simul(
    simul_data,
    method_name,
    state_ss,
    policies_ss,
    econ_model,
    welfare_fn,
    welfare_ss,
    config_dict,
):
    """Compute consumption-equivalent welfare cost from the canonical active Dynare sample."""
    if simul_data is None:
        return None
    simul_matrix = _normalize_dynare_full_simul(simul_data, state_ss, policies_ss)
    n_state_vars = state_ss.shape[0]
    policies_logdev = simul_matrix[n_state_vars:, :].T

    if policies_logdev.shape[0] == 0:
        print(f"  ⚠ Skipping welfare for {method_name}: active sample is empty.")
        return None

    method_seed = sum(ord(c) for c in method_name)
    welfare = welfare_fn(
        jax.vmap(econ_model.utility_from_policies)(policies_logdev),
        welfare_ss,
        random.PRNGKey(config_dict["welfare_seed"] + method_seed),
    )
    Vc = econ_model.consumption_equivalent(welfare)
    return -Vc * 100


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

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD MODEL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    model_data_file, model_path = _resolve_data_file(
        model_dir,
        config.get("model_data_file"),
        ["ModelData.mat", "model_data.mat"],
        label="ModelData",
        required=True,
    )
    model_data_irs_file, irs_path = _resolve_data_file(
        model_dir,
        config.get("model_data_irs_file"),
        ["ModelData_IRs.mat"],
        label="IR benchmark",
        required=False,
    )
    model_data_simulation_file, simul_path = _resolve_data_file(
        model_dir,
        config.get("model_data_simulation_file"),
        ["ModelData_simulation.mat"],
        label="Simulation benchmark",
        required=False,
    )

    config["model_data_file"] = model_data_file
    config["model_data_irs_file"] = model_data_irs_file
    config["model_data_simulation_file"] = model_data_simulation_file
    _write_analysis_config(config, analysis_dir)

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

    expected_n_vars = state_ss.shape[0] + policies_ss.shape[0]

    # Load simulation data (optional - for Dynare comparison)
    dynare_simul_1storder = None
    dynare_simul_so = None
    dynare_simul_pf = None
    dynare_simul_mit = None
    matlab_common_shock_schedule = None
    matlab_simulation_block = None
    dynare_1st_artifact = {"active_simul": None, "full_simul": None}
    dynare_so_artifact = {"active_simul": None, "full_simul": None}
    dynare_pf_artifact = {"active_simul": None, "full_simul": None}
    dynare_mit_artifact = {"active_simul": None, "full_simul": None}

    if simul_path is not None:
        print(f"  Loading simulation data from: {model_data_simulation_file}")
        simul_data = sio.loadmat(simul_path, simplify_cells=True)
        matlab_simulation_block = simul_data.get("ModelData_simulation", {})

        dynare_1st_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["FirstOrder", "Loglin"], expected_n_vars, precision
        )
        dynare_pf_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["PerfectForesight", "Determ"], expected_n_vars, precision
        )
        dynare_so_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["SecondOrder"], expected_n_vars, precision
        )
        dynare_mit_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["MITShocks", "MITShock"], expected_n_vars, precision
        )

        dynare_simul_1storder = dynare_1st_artifact["active_simul"]
        dynare_simul_pf = dynare_pf_artifact["active_simul"]
        dynare_simul_so = dynare_so_artifact["active_simul"]
        dynare_simul_mit = dynare_mit_artifact["active_simul"]

    if irs_path:
        print(f"  Found IRs file: {model_data_irs_file}")
    elif config.get("model_data_irs_file"):
        print(f"  ⚠ IRs file not found: {config['model_data_irs_file']} (will try legacy format)")

    # Create economic model
    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
    )
    shock_dimension = get_shock_dimension(econ_model, analysis_hooks)

    if matlab_simulation_block is not None:
        matlab_common_shock_schedule = _extract_matlab_common_shock_schedule(
            matlab_simulation_block,
            shock_dimension,
            precision,
        )
        if matlab_common_shock_schedule is not None:
            print(
                "  Loaded shared MATLAB shock path "
                f"({matlab_common_shock_schedule['reference_method']}: "
                f"{matlab_common_shock_schedule['burn_in']} burn-in, "
                f"{matlab_common_shock_schedule['active_shocks'].shape[0]} active, "
                f"{matlab_common_shock_schedule['burn_out']} burn-out)"
            )

    # Load experiment data
    experiments_to_analyze = config["experiments_to_analyze"]
    experiments_data = load_experiment_data(
        experiments_to_analyze,
        save_dir,
        expected_model_dir=config["model_dir"],
    )

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

    # Create analysis functions
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, config, analysis_hooks=analysis_hooks))
    stoch_ss_loss_fn = create_stochss_loss_fn(econ_model, mc_draws=32)
    gir_fn = jax.jit(create_GIR_fn(econ_model, config, analysis_hooks=analysis_hooks))
    run_nonlinear_simulation = _create_nonlinear_simulation_runner(
        econ_model=econ_model,
        config_dict=config,
        analysis_hooks=analysis_hooks,
        matlab_common_shock_schedule=matlab_common_shock_schedule,
    )

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
        experiment_results = _run_experiment_analysis(
            exp_data=exp_data,
            save_dir=save_dir,
            nn_config_base=nn_config_base,
            econ_model=econ_model,
            run_nonlinear_simulation=run_nonlinear_simulation,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            stoch_ss_fn=stoch_ss_fn,
            stoch_ss_loss_fn=stoch_ss_loss_fn,
            gir_fn=gir_fn,
            config_dict=config,
            analysis_hooks=analysis_hooks,
        )

        raw_simulation_data[experiment_label] = experiment_results["raw_simulation_data"]
        analysis_variables_data[experiment_label] = experiment_results["analysis_variables"]
        welfare_costs[experiment_label] = experiment_results["welfare_cost"]
        stochastic_ss_states[experiment_label] = experiment_results["stochastic_ss_state"]
        stochastic_ss_policies[experiment_label] = experiment_results["stochastic_ss_policy"]
        stochastic_ss_data[experiment_label] = experiment_results["stochastic_ss_data"]
        stochastic_ss_loss[experiment_label] = experiment_results["stochastic_ss_loss"]
        gir_data[experiment_label] = experiment_results["gir_data"]

    # Add welfare costs from Dynare simulation methods (if available).
    dynare_welfare_inputs = {
        "FirstOrder": dynare_1st_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "SecondOrder": dynare_so_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "PerfectForesight": dynare_pf_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "MITShocks": dynare_mit_artifact["active_simul"] if model_data_simulation_file is not None else None,
    }
    for method_name, simul_data in dynare_welfare_inputs.items():
        welfare_cost = _welfare_cost_from_dynare_simul(
            simul_data,
            method_name,
            state_ss,
            policies_ss,
            econ_model,
            welfare_fn,
            welfare_ss,
            config,
        )
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
    calibration_method_stats = model_postprocess.get("calibration_method_stats")
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
            method_model_stats=calibration_method_stats,
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
        "MIT Shocks": "MITShocks",
        "MIT shocks": "MITShocks",
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


# Legacy long-ergodic simulation branch kept for later reuse:
# simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config))
# simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
#     train_state,
#     econ_model,
#     config,
#     simulation_fn,
#     analysis_hooks=analysis_hooks,
# )
# simul_obs_full = simul_obs
# simul_policies_full = simul_policies


if __name__ == "__main__":
    main()
