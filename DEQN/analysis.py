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
import importlib
import math

# ============================================================================
# CONFIGURATION — edit this block before running in Colab
# ============================================================================

config = {
    # Core selection
    "model_dir": "RbcProdNet_April2026",
    "exact_cobb_douglas": False,
    "analysis_name": "baseline_newmetrics",
    # A checkpoint name string is the simplest form. Use {"label": "checkpoint"}
    # only when the displayed label should differ from the checkpoint directory.
    "experiments_to_analyze": "finercal_July",
    # Optional checkpoint used only for DEQN IR/GIR trajectories.
    # Leave as "" or None to use the normal experiment checkpoint.
    "ir_experiment_to_analyze": "finercal_July_IR",
    # Expensive analysis modules. The nonlinear simulation and descriptive
    # outputs still run because they supply the shared analysis sample.
    "run_stochastic_ss": True,
    "run_impulse_responses": True,
    "run_welfare": True,
    # MATLAB data files (relative to model_dir).
    # Set to None to use the standard filename or skip an optional comparison.
    "model_data_file": "ModelData_newmetrics.mat",
    "model_data_irs_file": "ModelData_IRs_finercal.mat",
    "model_data_simulation_file": "ModelData_simulation_finercal.mat",
    # Aggregation convention
    # False: use aggregate endogenous policy variables directly.
    # True: re-aggregate with fixed prices from a long ergodic DEQN sample.
    "ergodic_price_aggregation": False,
    # Main nonlinear simulation sample
    # False: replay the exact shock sequence saved by MATLAB, so DEQN and the
    # benchmark methods are compared over the same realizations and dates.
    # True: simulate a long DEQN sample from fresh random shock draws.
    "long_simulation": True,
    "init_range": 6,
    "periods_per_epis": 64000,
    "burn_in_periods": 3200,
    "simul_vol_scale": 1,
    "simul_seed": 0,
    "n_simul_seeds": 16,
    # Welfare
    "welfare_n_trajects": 16000,
    "welfare_traject_length": 200,
    "welfare_seed": 0,
    # Stochastic steady state
    "n_draws": 2000,
    "time_to_converge": 500,
    "seed": 0,
    # Impulse responses
    "gir_n_draws": 1000,
    "gir_trajectory_length": 100,
    "gir_seed": 0,
    # False: responses start from the stochastic steady state.
    # True: generalized responses average over simulated starting points.
    # A stochastic-steady-state IR automatically enables run_stochastic_ss.
    "use_gir": True,
    # Shock sizes are discovered from model_data_irs_file. If that file is
    # unavailable, provide percentages explicitly, for example [5, 10, 20].
    "ir_shock_sizes": None,
    "show_ir_plots": False,
    "ir_benchmark_methods": ["PerfectForesight", "FirstOrder"],
    # Sector indices are zero-based. A sector-j IR shocks its TFP state.
    "ir_sectors_to_plot": list(range(37)),
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
    "ir_max_periods": 20,
    "double_precision": True,
}

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
    import subprocess

    def _colab_package_stack_is_usable() -> bool:
        try:
            import numpy
            import scipy
            import scipy.io  # noqa: F401
            import jax
        except Exception as exc:
            print(f"Package stack check failed in current kernel: {exc!r}")
            return False
        print(
            "Package stack OK: "
            f"numpy={numpy.__version__} scipy={scipy.__version__} jax={jax.__version__}"
        )
        return True

    _COLAB_REPAIR_MARKER = "/content/.jaxecon_colab_numpy_repair_attempted"
    if _colab_package_stack_is_usable():
        if os.path.exists(_COLAB_REPAIR_MARKER):
            os.remove(_COLAB_REPAIR_MARKER)
    else:
        if os.path.exists(_COLAB_REPAIR_MARKER):
            raise RuntimeError(
                "The Colab NumPy/SciPy/JAX stack is still inconsistent after a repair restart. "
                "Use Runtime > Disconnect and delete runtime, then rerun the notebook."
            )
        print("Installing pinned NumPy/SciPy/JAX stack, then restarting Colab.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "numpy==2.0.2",
                "scipy==1.15.3",
                "jax[cuda12]",
            ],
            check=True,
        )
        with open(_COLAB_REPAIR_MARKER, "w", encoding="utf-8") as marker:
            marker.write("numpy==2.0.2 scipy==1.15.3 jax[cuda12]\n")
        print("Package stack repaired. Restarting Colab runtime; rerun this cell after reconnecting.")
        os.kill(os.getpid(), 9)

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

from typing import cast  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
import DEQN.analysis.reporting as analysis_reporting  # noqa: E402
from DEQN.analysis.artifacts import save_analysis_artifacts  # noqa: E402
from DEQN.analysis.io import (  # noqa: E402
    _extract_dynare_simulation_artifact,
    _extract_matlab_common_shock_schedule,
    _resolve_data_file,
    _write_analysis_config,
)
from DEQN.analysis.output_labels import (  # noqa: E402
    apply_display_labels_to_mapping,
    apply_display_labels_to_postprocess_context,
    apply_display_labels_to_sequence,
    build_display_welfare_costs,
    build_output_display_label_map,
)
from DEQN.analysis.reporting import (  # noqa: E402
    _build_output_note_context,
    _write_analysis_results_latex,
)
from DEQN.analysis.single_experiment import (  # noqa: E402
    SingleExperimentResult,
    _create_nonlinear_simulation_runners,
    _run_experiment_analysis,
)
from DEQN.analysis.simul_analysis import simulate_loglinear_ergodic_analysis  # noqa: E402
from DEQN.analysis.welfare_outputs import (  # noqa: E402
    _compute_welfare_cost_from_utilities,
    _welfare_cost_from_dynare_simul,
)
from DEQN.analysis.model_hooks import (  # noqa: E402
    apply_model_config_defaults,
    get_shock_dimension,
    get_states_to_shock,
    load_model_analysis_hooks,
    run_model_postprocess,
)
from DEQN.econ_models import load_model_class  # noqa: E402
from DEQN.analysis.stochastic_ss import (  # noqa: E402
    create_stochss_fn,
    create_stochss_loss_fn,
)
from DEQN.analysis.tables import (  # noqa: E402
    create_calibration_table,
    create_descriptive_stats_table,
    create_stochastic_ss_aggregates_table,
    create_targeted_moments_table,
    create_welfare_table,
)
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.training.checkpoints import load_experiment_data  # noqa: E402

jax_config.update("jax_debug_nans", True)

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory/regime
Model = load_model_class(config["model_dir"], config.get("exact_cobb_douglas", False))
analysis_hooks = load_model_analysis_hooks(config["model_dir"])
analysis_reporting.analysis_hooks = analysis_hooks
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


def _resolve_experiment_config(config_dict):
    experiments = config_dict.get("experiments_to_analyze")
    if isinstance(experiments, str) and experiments:
        return {experiments: experiments}
    if not isinstance(experiments, dict) or len(experiments) != 1:
        raise ValueError(
            "config['experiments_to_analyze'] must be a checkpoint-name string "
            "or a single-entry {label: checkpoint} dictionary."
        )
    return cast("dict[str, str]", experiments)


def _validate_ir_shock_sizes(shock_sizes):
    if isinstance(shock_sizes, (str, bytes)) or not isinstance(shock_sizes, (list, tuple)):
        raise ValueError("config['ir_shock_sizes'] must be a list of percentages.")

    normalized = []
    for raw_size in shock_sizes:
        if isinstance(raw_size, bool):
            raise ValueError("IR shock sizes must be numeric percentages.")
        try:
            size = float(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("IR shock sizes must be numeric percentages.") from exc
        if not math.isfinite(size) or not 0 < size < 100:
            raise ValueError(f"IR shock sizes must be finite percentages strictly between 0 and 100; got {raw_size}.")
        if size not in normalized:
            normalized.append(size)

    if not normalized:
        raise ValueError("config['ir_shock_sizes'] must contain at least one percentage.")
    return [int(size) if size.is_integer() else size for size in normalized]


def _resolve_analysis_modules(config_dict):
    run_impulse_responses = bool(config_dict.get("run_impulse_responses", True))
    run_stochastic_ss = bool(config_dict.get("run_stochastic_ss", True))
    if run_impulse_responses and not bool(config_dict.get("use_gir", True)) and not run_stochastic_ss:
        run_stochastic_ss = True
        print("  Enabling stochastic steady state because use_gir=False requires it.")

    resolved = {
        "run_stochastic_ss": run_stochastic_ss,
        "run_impulse_responses": run_impulse_responses,
        "run_welfare": bool(config_dict.get("run_welfare", True)),
    }
    config_dict.update(resolved)
    enabled = [name.removeprefix("run_") for name, is_enabled in resolved.items() if is_enabled]
    print(f"  Enabled analysis modules: {', '.join(enabled) if enabled else 'none'}")
    return resolved


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
    modules = _resolve_analysis_modules(config)

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
    if modules["run_impulse_responses"]:
        model_data_irs_file, irs_path = _resolve_data_file(
            model_dir,
            config.get("model_data_irs_file"),
            ["ModelData_IRs.mat"],
            label="IR benchmark",
            required=False,
        )
    else:
        model_data_irs_file = config.get("model_data_irs_file")
        irs_path = None
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

    state_space = md["Solution"]["StateSpace"]
    A_matrix = jnp.array(state_space["A"], dtype=precision)
    B_matrix = jnp.array(state_space["B"], dtype=precision)
    C_matrix = jnp.array(state_space["C"], dtype=precision)
    D_matrix = jnp.array(state_space["D"], dtype=precision)

    if len(policies_ss) != len(policies_sd):
        n_policies = len(policies_sd)
        policies_ss = policies_ss[:n_policies]
        C_matrix = C_matrix[:n_policies, :]
        D_matrix = D_matrix[:n_policies, :]

    expected_n_vars = state_ss.shape[0] + policies_ss.shape[0]

    # Load simulation data (optional - for Dynare comparison)
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

    if modules["run_impulse_responses"] and irs_path:
        print(f"  Found IRs file: {model_data_irs_file}")
    elif modules["run_impulse_responses"] and config.get("model_data_irs_file"):
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

    if (
        modules["run_impulse_responses"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "discover_ir_shock_sizes")
    ):
        discovered_ir_shock_sizes = analysis_hooks.discover_ir_shock_sizes(
            config=config,
            model_dir=model_dir,
            irs_path=irs_path,
        )
        if discovered_ir_shock_sizes:
            config["ir_shock_sizes"] = list(discovered_ir_shock_sizes)
            print(
                "  Using IR shock sizes discovered from MATLAB objects for DEQN IR computation: "
                f"{config['ir_shock_sizes']}"
            )
        elif not config.get("ir_shock_sizes"):
            raise ValueError(
                "Impulse responses are enabled, but no shock sizes were found in model_data_irs_file. "
                "Set config['ir_shock_sizes'] to an explicit percentage list when benchmark IR objects "
                "are unavailable."
            )
    if modules["run_impulse_responses"] and config.get("ir_shock_sizes") is not None:
        config["ir_shock_sizes"] = _validate_ir_shock_sizes(config["ir_shock_sizes"])
    _write_analysis_config(config, analysis_dir)

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
    experiments_to_analyze = _resolve_experiment_config(config)
    experiment_label = next(iter(experiments_to_analyze))
    experiments_data = load_experiment_data(
        experiments_to_analyze,
        save_dir,
        expected_model_dir=config["model_dir"],
    )
    exp_data = experiments_data[experiment_label]
    ir_experiment_name = config.get("ir_experiment_to_analyze")
    ir_exp_data = None
    if modules["run_impulse_responses"] and ir_experiment_name:
        ir_experiments_data = load_experiment_data(
            {"ir": ir_experiment_name},
            save_dir,
            expected_model_dir=config["model_dir"],
        )
        ir_exp_data = ir_experiments_data["ir"]
        print(f"  IR trajectories will use experiment checkpoint: {ir_experiment_name}")
    elif modules["run_impulse_responses"]:
        print("  IR trajectories will use the normal experiment checkpoint.")

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
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config)) if modules["run_welfare"] else None
    stoch_ss_fn = (
        jax.jit(create_stochss_fn(econ_model, config, analysis_hooks=analysis_hooks))
        if modules["run_stochastic_ss"]
        else None
    )
    stoch_ss_loss_fn = (
        create_stochss_loss_fn(econ_model, mc_draws=32) if modules["run_stochastic_ss"] else None
    )
    gir_fn = (
        jax.jit(create_GIR_fn(econ_model, config, analysis_hooks=analysis_hooks))
        if modules["run_impulse_responses"]
        else None
    )
    nonlinear_simulation_runners = _create_nonlinear_simulation_runners(
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
    nonlinear_method_labels = []

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION & WELFARE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SIMULATION & WELFARE RESULTS" if modules["run_welfare"] else "  SIMULATION RESULTS")
    print("═" * 72)
    print(f"  Analysis: {config['analysis_name']}")
    print(f"  Model: {config['model_dir']}")
    print(f"  Experiment: {experiment_label}")
    print(f"  Sectors: {n_sectors}")
    print("─" * 72, flush=True)

    print(f"\n  ▶ {experiment_label}", flush=True)
    experiment_results = SingleExperimentResult.from_mapping(
        experiment_label,
        _run_experiment_analysis(
            experiment_label=experiment_label,
            exp_data=exp_data,
            save_dir=save_dir,
            nn_config_base=nn_config_base,
            econ_model=econ_model,
            nonlinear_simulation_runners=nonlinear_simulation_runners,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            stoch_ss_fn=stoch_ss_fn,
            stoch_ss_loss_fn=stoch_ss_loss_fn,
            gir_fn=gir_fn,
            config_dict=config,
            analysis_hooks=analysis_hooks,
            ir_exp_data=ir_exp_data,
        ),
    )

    raw_simulation_data.update(experiment_results.raw_simulation_data)
    analysis_variables_data.update(experiment_results.analysis_variables)
    welfare_costs.update(experiment_results.welfare_costs)
    stochastic_ss_states.update(experiment_results.stochastic_ss_states)
    stochastic_ss_policies.update(experiment_results.stochastic_ss_policies)
    stochastic_ss_data.update(experiment_results.stochastic_ss_data)
    stochastic_ss_loss.update(experiment_results.stochastic_ss_loss)
    gir_data[experiment_label] = experiment_results.gir_data
    nonlinear_method_labels.extend(experiment_results.nonlinear_method_labels)

    long_loglinear_welfare_cost = None
    if bool(config.get("long_simulation", False)):
        print("  Computing matched long ergodic FirstOrder simulation.", flush=True)
        long_firstorder_analysis, long_firstorder_utilities = simulate_loglinear_ergodic_analysis(
            state_transition_matrix=A_matrix,
            state_shock_matrix=B_matrix,
            policy_state_matrix=C_matrix,
            policy_shock_matrix=D_matrix,
            econ_model=econ_model,
            analysis_config=config,
            analysis_context=experiment_results.raw_simulation_data[experiment_label]["analysis_context"],
            analysis_hooks=analysis_hooks,
            include_utilities=modules["run_welfare"],
        )
        analysis_variables_data["Log-Linear"] = long_firstorder_analysis
        if modules["run_welfare"] and long_firstorder_utilities is not None:
            method_seed = sum(ord(char) for char in "FirstOrder")
            long_loglinear_welfare_cost = _compute_welfare_cost_from_utilities(
                econ_model=econ_model,
                welfare_fn=welfare_fn,
                welfare_ss=welfare_ss,
                utilities=long_firstorder_utilities,
                rng_key=jax.random.PRNGKey(config["welfare_seed"] + method_seed),
            )
            welfare_costs["FirstOrder"] = long_loglinear_welfare_cost
            print(f"    Welfare cost (FirstOrder, long ergodic): {float(long_loglinear_welfare_cost):.4f}%")

    # Add welfare costs from Dynare simulation methods (if available).
    dynare_welfare_inputs = {
        "FirstOrder": dynare_1st_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "SecondOrder": dynare_so_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "PerfectForesight": dynare_pf_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "MITShocks": dynare_mit_artifact["active_simul"] if model_data_simulation_file is not None else None,
    }
    if modules["run_welfare"]:
        for method_name, simul_data in dynare_welfare_inputs.items():
            if method_name == "FirstOrder" and long_loglinear_welfare_cost is not None:
                continue
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
    matlab_ir_data = None
    upstreamness_data = None
    postprocess_context = None

    if analysis_hooks is not None and hasattr(analysis_hooks, "prepare_postprocess_analysis"):
        model_postprocess = analysis_hooks.prepare_postprocess_analysis(
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
            matlab_common_shock_schedule=matlab_common_shock_schedule,
        )
    else:
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
    matlab_ir_data = model_postprocess.get("matlab_ir_data", matlab_ir_data)
    upstreamness_data = model_postprocess.get("upstreamness_data", upstreamness_data)
    postprocess_context = model_postprocess.get("postprocess_context")
    save_analysis_artifacts(
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        analysis_name=config["analysis_name"],
        config=config,
        raw_simulation_data=raw_simulation_data,
        analysis_variables_data=analysis_variables_data,
        welfare_costs=welfare_costs,
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
        stochastic_ss_data=stochastic_ss_data,
    )
    output_display_label_map = build_output_display_label_map(config)
    display_postprocess_context = apply_display_labels_to_postprocess_context(
        postprocess_context,
        output_display_label_map,
    )
    display_gir_data = apply_display_labels_to_mapping(gir_data, output_display_label_map)
    display_stochastic_ss_states = apply_display_labels_to_mapping(stochastic_ss_states, output_display_label_map)
    display_stochastic_ss_policies = apply_display_labels_to_mapping(stochastic_ss_policies, output_display_label_map)
    display_stochastic_ss_data = apply_display_labels_to_mapping(stochastic_ss_data, output_display_label_map)
    display_welfare_costs = build_display_welfare_costs(welfare_costs, output_display_label_map)
    display_raw_simulation_data = apply_display_labels_to_mapping(raw_simulation_data, output_display_label_map)
    display_stochss_methods_to_include = cast(
        "list[str] | None",
        apply_display_labels_to_sequence(
            config.get("stochss_methods_to_include"),
            output_display_label_map,
        ),
    )
    if not display_stochss_methods_to_include:
        display_stochss_methods_to_include = cast("list[str] | None", list(display_stochastic_ss_data.keys()))
    output_note_context = _build_output_note_context(config, matlab_common_shock_schedule)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL VS DATA MOMENTS TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  MODEL VS DATA MOMENTS TABLE")
    print("═" * 72, flush=True)
    calibration_emp = (
        md.get("EmpiricalTargets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("empirical_targets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("EmpiricalTargets")
    )
    fo_stats = stats.get("FirstOrder") or stats.get("firstorder")
    calibration_model_stats = (
        (fo_stats.get("ModelStats") or fo_stats.get("modelstats")) if isinstance(fo_stats, dict) else None
    )
    model_vs_data_aliases = {
        "FirstOrder": "1st",
        "Log-Linear": "1st",
        "LogLinear": "1st",
        "SecondOrder": "2nd",
        "Second-Order": "2nd",
        "PerfectForesight": "PF",
        "Perfect Foresight": "PF",
        "MITShock": "MITShocks",
        "MIT Shocks": "MITShocks",
        "MIT shocks": "MITShocks",
        "NonlinearCS": "Nonlinear-CS",
        "Nonlinear_CS": "Nonlinear-CS",
    }

    def _normalize_model_vs_data_method_name(method_name: str) -> str:
        return model_vs_data_aliases.get(method_name, method_name)

    model_vs_data_methods_cfg = config.get("model_vs_data_methods_to_include")
    filtered_calibration_method_stats = calibration_method_stats
    if model_vs_data_methods_cfg and calibration_method_stats is not None:
        if isinstance(model_vs_data_methods_cfg, str):
            model_vs_data_methods_cfg = [model_vs_data_methods_cfg]
        requested_methods = [_normalize_model_vs_data_method_name(name) for name in model_vs_data_methods_cfg]
        requested_method_set = set(requested_methods)
        filtered_calibration_method_stats = {
            method_name: stats_dict
            for method_name, stats_dict in calibration_method_stats.items()
            if method_name in requested_method_set
        }
        if filtered_calibration_method_stats:
            print(f"  Model-vs-data methods: {list(filtered_calibration_method_stats.keys())}")
        else:
            print(
                "  ⚠ model_vs_data_methods_to_include did not match any available methods; "
                f"using all available methods {list(calibration_method_stats.keys())}."
            )
            filtered_calibration_method_stats = calibration_method_stats
    if calibration_emp is not None:
        create_calibration_table(
            empirical_targets=calibration_emp,
            first_order_model_stats=calibration_model_stats,
            method_model_stats=filtered_calibration_method_stats,
            save_path=os.path.join(analysis_dir, "calibration_table.tex"),
            analysis_name=config["analysis_name"],
            note_context=output_note_context,
        )
        if filtered_calibration_method_stats:
            create_targeted_moments_table(
                empirical_targets=calibration_emp,
                method_model_stats=filtered_calibration_method_stats,
                save_path=os.path.join(analysis_dir, "targeted_moments.tex"),
                analysis_name=config["analysis_name"],
            )
    else:
        print(
            "  ⚠ Calibration table skipped: no empirical targets in ModelData (EmpiricalTargets / calibration.empirical_targets)."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATE IMPULSE RESPONSES
    # ═══════════════════════════════════════════════════════════════════════════
    if modules["run_impulse_responses"]:
        print("\n" + "═" * 72)
        print("  AGGREGATE IMPULSE RESPONSES")
        print("═" * 72, flush=True)
    if (
        modules["run_impulse_responses"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "render_aggregate_ir_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_aggregate_ir_outputs(
            config=config,
            irs_dir=irs_dir,
            econ_model=econ_model,
            gir_data=display_gir_data,
            postprocess_context=display_postprocess_context,
        )
    if (
        modules["run_impulse_responses"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "render_cir_analysis_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_cir_analysis_outputs(
            config=config,
            irs_dir=irs_dir,
            econ_model=econ_model,
            gir_data=display_gir_data,
            postprocess_context=display_postprocess_context,
            raw_simulation_data=raw_simulation_data,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTORAL VARIABLES IN STOCHASTIC SS
    # ═══════════════════════════════════════════════════════════════════════════
    if modules["run_stochastic_ss"]:
        print("\n" + "═" * 72)
        print("  SECTORAL VARIABLES IN STOCHASTIC SS")
        print("═" * 72, flush=True)
    if (
        modules["run_stochastic_ss"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "render_sectoral_stochss_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_sectoral_stochss_outputs(
            config=config,
            simulation_dir=simulation_dir,
            econ_model=econ_model,
            stochastic_ss_states=display_stochastic_ss_states,
            stochastic_ss_policies=display_stochastic_ss_policies,
            postprocess_context=display_postprocess_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTORAL UPSTREAMNESS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SECTORAL UPSTREAMNESS")
    print("═" * 72, flush=True)
    if (
        analysis_hooks is not None
        and hasattr(analysis_hooks, "render_upstreamness_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_upstreamness_outputs(
            config=config,
            simulation_dir=simulation_dir,
            econ_model=econ_model,
            postprocess_context=display_postprocess_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATE STOCHASTIC STEADY STATE
    # ═══════════════════════════════════════════════════════════════════════════
    if modules["run_stochastic_ss"]:
        print("\n" + "═" * 72)
        print("  AGGREGATE STOCHASTIC STEADY STATE")
        print("═" * 72, flush=True)
        create_stochastic_ss_aggregates_table(
            stochastic_ss_data=display_stochastic_ss_data,
            save_path=os.path.join(analysis_dir, "stochastic_ss_aggregates_table.tex"),
            analysis_name=config["analysis_name"],
            methods_to_include=display_stochss_methods_to_include,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # DESCRIPTIVE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  DESCRIPTIVE STATISTICS")
    print("═" * 72, flush=True)
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

    available_methods = sorted({_normalize_method_name(k) for k in analysis_variables_data.keys()})
    print(f"  Available methods (canonical): {available_methods}")

    benchmark_methods_cfg = config.get("ergodic_methods_to_include")
    benchmark_methods = None
    if benchmark_methods_cfg:
        if isinstance(benchmark_methods_cfg, str):
            benchmark_methods_cfg = [benchmark_methods_cfg]
        benchmark_methods = {_normalize_method_name(m) for m in benchmark_methods_cfg}

    selected_methods = set(available_methods) if benchmark_methods is None else set(benchmark_methods)
    if config.get("always_include_nonlinear_methods", True):
        selected_methods.update(_normalize_method_name(label) for label in nonlinear_method_labels)

    filtered_analysis_variables_data = {
        _normalize_method_name(k): v
        for k, v in analysis_variables_data.items()
        if _normalize_method_name(k) in selected_methods
    }
    desc_vars = config.get("descriptive_stats_variables")
    if desc_vars:
        desc_analysis_data = {
            method: {k: v for k, v in variables.items() if k in desc_vars}
            for method, variables in filtered_analysis_variables_data.items()
        }
        desc_analysis_data = {k: v for k, v in desc_analysis_data.items() if v}
    else:
        desc_analysis_data = filtered_analysis_variables_data

    desc_display_label_map = dict(output_display_label_map)
    desc_display_label_map.update(
        {
            "Log-Linear": "1st Order Approximation",
            "MITShocks": "MIT shocks",
        }
    )
    display_desc_analysis_data = apply_display_labels_to_mapping(
        desc_analysis_data,
        desc_display_label_map,
    )
    display_theoretical_stats = apply_display_labels_to_mapping(
        theoretical_stats,
        desc_display_label_map,
    )

    create_descriptive_stats_table(
        analysis_variables_data=display_desc_analysis_data,
        save_path=os.path.join(simulation_dir, "descriptive_stats_table.tex"),
        analysis_name=config["analysis_name"],
        theoretical_stats=display_theoretical_stats,
        note_context=output_note_context,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATE HISTOGRAMS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  AGGREGATE HISTOGRAMS")
    print("═" * 72, flush=True)
    if (
        analysis_hooks is not None
        and hasattr(analysis_hooks, "render_aggregate_histogram_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_aggregate_histogram_outputs(
            config=config,
            simulation_dir=simulation_dir,
            analysis_variables_data=analysis_variables_data,
            postprocess_context=postprocess_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # WELFARE COSTS
    # ═══════════════════════════════════════════════════════════════════════════
    if modules["run_welfare"]:
        print("\n" + "═" * 72)
        print("  WELFARE COSTS")
        print("═" * 72, flush=True)
        create_welfare_table(
            welfare_data=display_welfare_costs,
            save_path=os.path.join(analysis_dir, "welfare_table.tex"),
            analysis_name=config["analysis_name"],
            note_context=output_note_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTORAL IMPULSE RESPONSES
    # ═══════════════════════════════════════════════════════════════════════════
    if modules["run_impulse_responses"]:
        print("\n" + "═" * 72)
        print("  SECTORAL IMPULSE RESPONSES")
        print("═" * 72, flush=True)
    if (
        modules["run_impulse_responses"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "render_sectoral_ir_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_sectoral_ir_outputs(
            config=config,
            irs_dir=irs_dir,
            econ_model=econ_model,
            gir_data=display_gir_data,
            postprocess_context=display_postprocess_context,
        )

    if (
        modules["run_stochastic_ss"]
        and analysis_hooks is not None
        and hasattr(analysis_hooks, "render_ergodic_sectoral_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_ergodic_sectoral_outputs(
            config=config,
            simulation_dir=simulation_dir,
            econ_model=econ_model,
            raw_simulation_data=display_raw_simulation_data,
            postprocess_context=display_postprocess_context,
            stochastic_ss_states=display_stochastic_ss_states,
            stochastic_ss_policies=display_stochastic_ss_policies,
        )

    # Model-specific plots
    if MODEL_SPECIFIC_PLOTS:
        for plot_spec in MODEL_SPECIFIC_PLOTS:
            plot_name = plot_spec["name"]
            plot_function = plot_spec["function"]

            for simulation_label, sim_data in raw_simulation_data.items():
                if sim_data.get("simulation_kind", "ergodic") != "ergodic":
                    continue
                try:
                    plot_function(
                        simul_obs=sim_data["simul_obs"],
                        simul_policies=sim_data["simul_policies"],
                        simul_analysis_variables=sim_data["simul_analysis_variables"],
                        save_path=os.path.join(simulation_dir, f"{plot_name}_{simulation_label}.png"),
                        analysis_name=config["analysis_name"],
                        econ_model=econ_model,
                        experiment_label=simulation_label,
                    )
                except Exception as e:
                    print(f"    ✗ Failed to create {plot_name} for {simulation_label}: {e}", flush=True)

    _write_analysis_results_latex(
        config_dict=config,
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        irs_dir=irs_dir,
        econ_model=econ_model,
    )

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
