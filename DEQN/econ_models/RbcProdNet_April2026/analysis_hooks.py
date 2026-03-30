import os
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from DEQN.econ_models.RbcProdNet_April2026.aggregation import (
    compute_model_moments_from_dynare_simulation,
    compute_ergodic_prices_from_simulation,
    compute_model_moments_with_consistent_aggregation,
    create_theoretical_descriptive_stats,
    get_loglinear_distribution_params,
    process_simulation_with_consistent_aggregation,
    reaggregate_aggregates,
)
from DEQN.econ_models.RbcProdNet_April2026.matlab_irs import get_available_shock_sizes, load_matlab_irs
from DEQN.econ_models.RbcProdNet_April2026.plot_helpers import plot_ergodic_histograms, _write_figure_note_tex
from DEQN.econ_models.RbcProdNet_April2026.plots import (
    plot_sector_ir_by_shock_size,
    plot_sectoral_variable_ergodic,
    plot_sectoral_variable_stochss,
)

DEFAULT_ANALYSIS_CONFIG = {
    "ergodic_price_aggregation": False,
}

DEFAULT_IR_BENCHMARK_METHODS = ["PerfectForesight", "FirstOrder"]

DEFAULT_AGGREGATE_IR_LABELS = [
    "Agg. Consumption",
    "Agg. Investment",
    "Agg. GDP",
    "Agg. Capital",
    "Agg. Labor",
    "Intratemporal Utility",
]

AGGREGATE_HISTOGRAM_BENCHMARKS = [
    ("Log-Linear", "1st Order Approx."),
    ("MITShocks", "MIT shocks"),
]

CORE_SECTORAL_IR_LABELS = [
    "Cj",
    "Ioutj",
    "Yj",
    "Kj",
    "Lj",
    "Qj",
]

SUPPORTED_SECTORAL_IR_LABELS = [
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
]

_SECTORAL_POLICY_BLOCKS = {
    "Cj": 0,
    "Lj": 1,
    "Pj": 8,
    "Mj": 4,
    "Moutj": 5,
    "Ij": 6,
    "Ioutj": 7,
    "Qj": 9,
    "Yj": 10,
    "Pmj_client": 3,
    "Cj_client": 0,
    "Lj_client": 1,
    "Pj_client": 8,
    "Mj_client": 4,
    "Moutj_client": 5,
    "Ij_client": 6,
    "Ioutj_client": 7,
    "Qj_client": 9,
    "Yj_client": 10,
}

_WARNED_UNSUPPORTED_IR_LABELS: set[str] = set()

SECTORAL_VAR_DESC = {
    "Cj": ("Row  3", "Consumption (own sector)"),
    "Pj": ("Row  4", "Output price (own sector)"),
    "Ioutj": ("Row  5", "Investment output (own sector)"),
    "Moutj": ("Row  6", "Intermediate output (own sector)"),
    "Lj": ("Row  7", "Labor (own sector)"),
    "Ij": ("Row  8", "Investment input (own sector)"),
    "Mj": ("Row  9", "Intermediate input (own sector)"),
    "Yj": ("Row 10", "Value added (own sector)"),
    "Qj": ("Row 11", "Gross output (own sector)"),
    "Kj": ("Row 22", "Capital (own sector)"),
    "Cj_client": ("Row 13", "Consumption (client sector)"),
    "Pj_client": ("Row 14", "Output price (client sector)"),
    "Ioutj_client": ("Row 15", "Investment output (client sector)"),
    "Moutj_client": ("Row 16", "Intermediate output (client sector)"),
    "Lj_client": ("Row 17", "Labor (client sector)"),
    "Ij_client": ("Row 18", "Investment input (client sector)"),
    "Mj_client": ("Row 19", "Intermediate input (client sector)"),
    "Yj_client": ("Row 20", "Value added (client sector)"),
    "Qj_client": ("Row 21", "Gross output (client sector)"),
    "Pmj_client": ("Row 24", "Intermediate input price (client sector)"),
    "gammaij_client": ("Row 25", "Expenditure share deviation (client sector)"),
}


def prepare_analysis_context(econ_model, simul_obs, simul_policies, config) -> Dict[str, Any]:
    del simul_obs
    n = econ_model.n_sectors
    P_ss = jnp.exp(econ_model.policies_ss[8 * n : 9 * n])
    Pk_ss = jnp.exp(econ_model.policies_ss[2 * n : 3 * n])
    use_ergodic_prices = bool(config.get("ergodic_price_aggregation", False))
    if use_ergodic_prices:
        P_ergodic, Pk_ergodic, _ = compute_ergodic_prices_from_simulation(
            simul_policies,
            econ_model.policies_ss,
            n,
        )
    else:
        P_ergodic, Pk_ergodic = P_ss, Pk_ss
    return {
        "P_weights": jnp.log(P_ergodic) - jnp.log(P_ss),
        "Pk_weights": jnp.log(Pk_ergodic) - jnp.log(Pk_ss),
        "ergodic_price_aggregation": use_ergodic_prices,
    }


def compute_analysis_variables(econ_model, state_logdev, policy_logdev, analysis_context) -> Dict[str, Any]:
    if analysis_context["ergodic_price_aggregation"]:
        return reaggregate_aggregates(
            state_logdev=state_logdev,
            policies_logdev=policy_logdev,
            policies_ss=econ_model.policies_ss,
            state_ss=econ_model.state_ss,
            log_policy_count=econ_model.log_policy_count,
            utility_intratemp_idx=econ_model.utility_intratemp_idx,
            P_weights=analysis_context["P_weights"],
            Pk_weights=analysis_context["Pk_weights"],
        )
    return econ_model.get_aggregates(policy_logdev)


def get_states_to_shock(config, econ_model) -> list[int]:
    ir_sectors = config.get("ir_sectors_to_plot")
    if ir_sectors:
        return [econ_model.n_sectors + sector_idx for sector_idx in ir_sectors]
    return list(range(econ_model.n_sectors, 2 * econ_model.n_sectors))


def _get_requested_sectoral_ir_variables(config) -> list[str]:
    requested = config.get("sectoral_ir_variables_to_plot", CORE_SECTORAL_IR_LABELS)
    if isinstance(requested, str):
        return [requested]
    return list(requested)


def _resolve_ir_benchmark_methods(config) -> list[str]:
    configured_methods = config.get("ir_benchmark_methods")
    if configured_methods is None:
        legacy_method = config.get("ir_benchmark_method")
        configured_methods = [legacy_method] if legacy_method else list(DEFAULT_IR_BENCHMARK_METHODS)
    elif isinstance(configured_methods, str):
        configured_methods = [configured_methods]

    resolved_methods = []
    for method in configured_methods:
        if method and method not in resolved_methods:
            resolved_methods.append(method)
    return resolved_methods or list(DEFAULT_IR_BENCHMARK_METHODS)


def _resolve_ir_response_source(config) -> str:
    use_gir = config.get("use_gir")
    if use_gir is not None:
        return "GIR" if bool(use_gir) else "IR_stoch_ss"

    configured_ir_methods = config.get("ir_methods")
    if configured_ir_methods is None:
        return "IR_stoch_ss"

    if isinstance(configured_ir_methods, str):
        configured_ir_methods = [configured_ir_methods]

    return "GIR" if "GIR" in configured_ir_methods else "IR_stoch_ss"


def _warn_unsupported_sectoral_ir_variables(requested_labels) -> None:
    unsupported = [label for label in requested_labels if label not in SUPPORTED_SECTORAL_IR_LABELS]
    new_labels = [label for label in unsupported if label not in _WARNED_UNSUPPORTED_IR_LABELS]
    if new_labels:
        _WARNED_UNSUPPORTED_IR_LABELS.update(new_labels)
        print(
            "  Warning: skipping unsupported sectoral IR variables "
            f"{new_labels}. Supported labels: {SUPPORTED_SECTORAL_IR_LABELS}"
        )


def _get_client_indices(econ_model) -> list[int]:
    gamma_m = np.asarray(econ_model.Gamma_M, dtype=float)
    client_indices = []
    for sector_idx in range(econ_model.n_sectors):
        row = gamma_m[sector_idx].copy()
        row[sector_idx] = -np.inf
        client_indices.append(int(np.argmax(row)))
    return client_indices


def _policy_value_for_sector(policy_logdev, block_idx, sector_idx, n_sectors):
    return policy_logdev[block_idx * n_sectors + sector_idx]


def extend_gir_var_labels(var_labels, econ_model, config) -> list[str]:
    del econ_model
    extended = list(var_labels)
    requested_labels = _get_requested_sectoral_ir_variables(config)
    _warn_unsupported_sectoral_ir_variables(requested_labels)
    for label in requested_labels:
        if label in SUPPORTED_SECTORAL_IR_LABELS and label not in extended:
            extended.append(label)
    return extended


def augment_gir_analysis_variables(analysis_vars_dict, obs_logdev, policy_logdev, state_idx, econ_model, config):
    sector_idx = state_idx - econ_model.n_sectors
    if sector_idx < 0 or sector_idx >= econ_model.n_sectors:
        return analysis_vars_dict

    requested_labels = _get_requested_sectoral_ir_variables(config)
    _warn_unsupported_sectoral_ir_variables(requested_labels)

    n = econ_model.n_sectors
    j = sector_idx
    client_idx = _get_client_indices(econ_model)[j]

    supported_requested = [label for label in requested_labels if label in SUPPORTED_SECTORAL_IR_LABELS]
    if not supported_requested:
        return analysis_vars_dict

    own_sector_values = {
        "Kj": obs_logdev[j],
    }
    client_sector_values = {}

    for label, block_idx in _SECTORAL_POLICY_BLOCKS.items():
        if label.endswith("_client"):
            client_sector_values[label] = _policy_value_for_sector(policy_logdev, block_idx, client_idx, n)
        elif label != "Pmj_client":
            own_sector_values[label] = _policy_value_for_sector(policy_logdev, block_idx, j, n)

    client_sector_values["gammaij_client"] = (1 - econ_model.sigma_m) * (
        own_sector_values["Pj"] - client_sector_values["Pmj_client"]
    )

    for label in supported_requested:
        if label in own_sector_values:
            analysis_vars_dict[label] = own_sector_values[label]
        elif label in client_sector_values:
            analysis_vars_dict[label] = client_sector_values[label]

    return analysis_vars_dict


def _resolve_reference_experiment_label(config, raw_simulation_data) -> str:
    configured_label = config.get("aggregation_reference_experiment")
    if configured_label is not None:
        if configured_label in raw_simulation_data:
            return configured_label
        ergodic_alias = f"{configured_label} (ergodic)"
        if ergodic_alias in raw_simulation_data:
            return ergodic_alias
        available = list(raw_simulation_data.keys())
        raise ValueError(
            "aggregation_reference_experiment must match an analyzed experiment label. "
            f"Got '{configured_label}', available labels: {available}"
        )

    for label, sim_data in raw_simulation_data.items():
        if sim_data.get("simulation_kind") == "ergodic":
            return label
    return next(iter(raw_simulation_data))


def discover_ir_shock_sizes(*, config, model_dir, irs_path):
    if not irs_path:
        return None

    matlab_ir_dir = os.path.join(model_dir, "MATLAB", "IRs")
    matlab_ir_data = load_matlab_irs(
        matlab_ir_dir=matlab_ir_dir,
        irs_file_path=irs_path,
    )
    shock_sizes = get_available_shock_sizes(matlab_ir_data)
    if not shock_sizes:
        configured_shocks = config.get("ir_shock_sizes")
        if configured_shocks:
            return list(configured_shocks)
        return None
    return shock_sizes


def _build_ir_render_context(*, config, model_dir, irs_path, policies_ss, state_ss, P_ergodic, Pk_ergodic, econ_model, n_sectors):
    matlab_ir_dir = os.path.join(model_dir, "MATLAB", "IRs")
    matlab_ir_data = load_matlab_irs(
        matlab_ir_dir=matlab_ir_dir,
        irs_file_path=irs_path,
    )
    shock_sizes = get_available_shock_sizes(matlab_ir_data)
    if not shock_sizes:
        configured_shocks = config.get("ir_shock_sizes")
        if configured_shocks:
            shock_sizes = list(configured_shocks)
            print(f"  Falling back to configured IR shock sizes: {shock_sizes}")
        else:
            raise ValueError("Could not infer IR shock sizes from MATLAB IR objects.")
    else:
        print(f"  Using IR shock sizes discovered from MATLAB objects: {shock_sizes}")
        config["ir_shock_sizes"] = shock_sizes

    sectors_to_plot = config.get("ir_sectors_to_plot", [0, 2, 23])
    ir_variables = list(DEFAULT_AGGREGATE_IR_LABELS)

    sectoral_ir_variables = [
        label for label in _get_requested_sectoral_ir_variables(config) if label in SUPPORTED_SECTORAL_IR_LABELS
    ]
    max_periods = config.get("ir_max_periods", 80)
    ir_response_source = _resolve_ir_response_source(config)

    return {
        "matlab_ir_data": matlab_ir_data,
        "sectors_to_plot": sectors_to_plot,
        "ir_variables": ir_variables,
        "sectoral_ir_variables": sectoral_ir_variables,
        "shock_sizes": shock_sizes,
        "largest_shock": max(shock_sizes),
        "max_periods": max_periods,
        "ir_response_source": ir_response_source,
        "policies_ss_np": np.asarray(policies_ss),
        "state_ss_np": np.asarray(state_ss),
        "P_ergodic_np": np.asarray(P_ergodic),
        "Pk_ergodic_np": np.asarray(Pk_ergodic),
        "ergodic_price_aggregation": bool(config.get("ergodic_price_aggregation", False)),
        "n_sectors": n_sectors,
    }


def prepare_postprocess_analysis(
    *,
    config,
    model_dir,
    analysis_dir,
    simulation_dir,
    irs_dir,
    econ_model,
    model_data,
    stats,
    policies_ss,
    state_ss,
    raw_simulation_data,
    analysis_variables_data,
    stochastic_ss_states,
    stochastic_ss_policies,
    stochastic_ss_data,
    gir_data,
    dynare_simulations,
    irs_path,
    matlab_common_shock_schedule=None,
):
    del model_data
    n_sectors = econ_model.n_sectors
    analysis_variables_data = dict(analysis_variables_data)
    theoretical_stats: Dict[str, Any] = {}
    use_ergodic_prices = bool(config.get("ergodic_price_aggregation", False))

    reference_experiment_label = _resolve_reference_experiment_label(config, raw_simulation_data)
    reference_sim_data = raw_simulation_data[reference_experiment_label]
    simul_policies = reference_sim_data.get("simul_policies_full", reference_sim_data["simul_policies"])
    P_ss = jnp.exp(policies_ss[8 * n_sectors : 9 * n_sectors])
    Pk_ss = jnp.exp(policies_ss[2 * n_sectors : 3 * n_sectors])
    if use_ergodic_prices:
        print(
            f"  Using '{reference_experiment_label}' as the ergodic-price aggregation reference.",
            flush=True,
        )
        P_ergodic, Pk_ergodic, _ = compute_ergodic_prices_from_simulation(simul_policies, policies_ss, n_sectors)
    else:
        print("  Using model-implied aggregate policy variables (no ergodic-price reaggregation).", flush=True)
        P_ergodic, Pk_ergodic = P_ss, Pk_ss

    dynare_simul_1storder = dynare_simulations.get("FirstOrder")
    dynare_simul_so = dynare_simulations.get("SecondOrder")
    dynare_simul_pf = dynare_simulations.get("PerfectForesight")
    dynare_simul_mit = dynare_simulations.get("MITShocks")
    theo_stats = stats.get("TheoStats") if isinstance(stats, dict) else None

    if isinstance(theo_stats, dict):
        theoretical_stats.update(create_theoretical_descriptive_stats(theo_stats, label="Log-Linear"))

    if dynare_simul_1storder is not None:
        firstorder_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_1storder,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            n_sectors=n_sectors,
            ergodic_price_aggregation=use_ergodic_prices,
            burn_in=0,
            source_label="First-Order (Dynare)",
        )

        for var_name, var_values in firstorder_analysis_vars.items():
            n_nan = jnp.sum(jnp.isnan(var_values))
            if n_nan > 0:
                print(f"    WARNING: {var_name} has {n_nan} NaN values!", flush=True)
        analysis_variables_data["Log-Linear"] = firstorder_analysis_vars
        print("  Loaded First-Order (Dynare) simulation with consistent aggregation.")

    if dynare_simul_so is not None:
        secondorder_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_so,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            n_sectors=n_sectors,
            ergodic_price_aggregation=use_ergodic_prices,
            burn_in=0,
            source_label="Second-Order",
        )
        analysis_variables_data["SecondOrder"] = secondorder_analysis_vars
        print("  Loaded Second-Order simulation series.")

    pf_stats_key = "PerfectForesight" if "PerfectForesight" in stats else ("Determ" if "Determ" in stats else None)
    if pf_stats_key:
        pf_stats = stats[pf_stats_key]
        pf_model_stats = pf_stats.get("ModelStats") if isinstance(pf_stats, dict) else None
        if dynare_simul_pf is None:
            if isinstance(pf_model_stats, dict):
                c_sd = pf_model_stats.get("sigma_C_agg")
                i_sd = pf_model_stats.get("sigma_I_agg")
                y_sd = pf_model_stats.get("sigma_VA_agg")
                if c_sd is not None:
                    print(f"    Agg. Consumption (exp): sigma={float(c_sd)*100:.4f}%", flush=True)
                if i_sd is not None:
                    print(f"    Agg. Investment (exp): sigma={float(i_sd)*100:.4f}%", flush=True)
                if y_sd is not None:
                    print(f"    Agg. Output/GDP (exp): sigma={float(y_sd)*100:.4f}%", flush=True)
            elif "policies_std" in pf_stats:
                policies_std = pf_stats["policies_std"]
                n = n_sectors
                if len(policies_std) > 11 * n + 6:
                    print(f"    Agg. Consumption: sigma={float(policies_std[11*n+2])*100:.4f}%", flush=True)
                    print(f"    Agg. Labor: sigma={float(policies_std[11*n+3])*100:.4f}%", flush=True)
        else:
            print("  Perfect Foresight moments will use Perfect Foresight (Dynare) simulation series.")

    if dynare_simul_pf is not None:
        pf_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_pf,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            n_sectors=n_sectors,
            ergodic_price_aggregation=use_ergodic_prices,
            burn_in=0,
            source_label="Perfect Foresight",
        )
        analysis_variables_data["PerfectForesight"] = pf_analysis_vars

    if dynare_simul_mit is not None:
        mit_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_mit,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            n_sectors=n_sectors,
            ergodic_price_aggregation=use_ergodic_prices,
            burn_in=0,
            source_label="MITShocks",
        )
        analysis_variables_data["MITShocks"] = mit_analysis_vars
        print("  Loaded MITShocks simulation series.")

    calibration_method_stats = _build_calibration_method_stats(
        stats=stats,
        dynare_simulations=dynare_simulations,
        analysis_variables_data=analysis_variables_data,
        raw_simulation_data=raw_simulation_data,
        reference_experiment_label=reference_experiment_label,
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        n_sectors=n_sectors,
        ergodic_price_aggregation=use_ergodic_prices,
    )

    upstreamness_data = econ_model.upstreamness()
    ergodic_experiment_labels = [
        label for label, sim_data in raw_simulation_data.items() if sim_data.get("simulation_kind", "ergodic") == "ergodic"
    ]
    if not ergodic_experiment_labels and reference_experiment_label in raw_simulation_data:
        ergodic_experiment_labels = [reference_experiment_label]
    ir_render_context = _build_ir_render_context(
        config=config,
        model_dir=model_dir,
        irs_path=irs_path,
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        econ_model=econ_model,
        n_sectors=n_sectors,
    )
    aggregate_histogram_context = _build_aggregate_histogram_context(
        config=config,
        simulation_dir=simulation_dir,
        raw_simulation_data=raw_simulation_data,
        reference_experiment_label=reference_experiment_label,
        matlab_common_shock_schedule=matlab_common_shock_schedule,
    )
    if isinstance(theo_stats, dict):
        loglinear_hist_params = get_loglinear_distribution_params(theo_stats)
        if loglinear_hist_params:
            aggregate_histogram_context["theoretical_distribution_params"] = {
                "1st Order Approx.": loglinear_hist_params
            }

    return {
        "analysis_variables_data": analysis_variables_data,
        "calibration_method_stats": calibration_method_stats,
        "theoretical_stats": theoretical_stats,
        "matlab_ir_data": ir_render_context["matlab_ir_data"],
        "upstreamness_data": upstreamness_data,
        "stochastic_ss_data": stochastic_ss_data,
        "postprocess_context": {
            "ir_render_context": ir_render_context,
            "upstreamness_data": upstreamness_data,
            "ergodic_experiment_labels": ergodic_experiment_labels,
            "reference_experiment_label": reference_experiment_label,
            "aggregate_histogram_context": aggregate_histogram_context,
        },
    }


def _build_aggregate_histogram_context(
    *,
    config,
    simulation_dir,
    raw_simulation_data,
    reference_experiment_label,
    matlab_common_shock_schedule,
):
    note_anchor_path = os.path.join(
        simulation_dir,
        f"aggregate_histograms_{config['analysis_name']}.png",
    )
    context = {
        "note_anchor_path": note_anchor_path,
        "note_path": os.path.splitext(note_anchor_path)[0] + "_note.tex",
        "long_simulation": bool(config.get("long_simulation", False)),
        "benchmark_labels": [display_label for _, display_label in AGGREGATE_HISTOGRAM_BENCHMARKS],
    }

    if context["long_simulation"]:
        reference_sim_data = raw_simulation_data.get(reference_experiment_label, {})
        active_obs = reference_sim_data.get("simul_obs")
        periods_per_episode = int(config.get("periods_per_epis", 0))
        burn_in = int(config.get("burn_in_periods", 0))
        kept_periods_per_seed = max(periods_per_episode - burn_in, 0)
        context.update(
            {
                "mode": "long_ergodic",
                "active_observations": int(active_obs.shape[0]) if active_obs is not None else 0,
                "kept_periods_per_seed": kept_periods_per_seed,
                "total_periods": periods_per_episode,
                "burn_in": burn_in,
                "burn_out": 0,
                "n_simul_seeds": int(config.get("n_simul_seeds", 0)),
                "periods_per_episode": periods_per_episode,
            }
        )
        return context

    schedule = matlab_common_shock_schedule or {}
    active_shocks = schedule.get("active_shocks")
    active_periods = int(active_shocks.shape[0]) if active_shocks is not None else 0
    full_shocks = schedule.get("full_shocks")
    total_periods = int(full_shocks.shape[0]) if full_shocks is not None else active_periods
    context.update(
        {
            "mode": "common_shock",
            "reference_method": schedule.get("reference_method", "MATLAB benchmark"),
            "burn_in": int(schedule.get("burn_in", 0)),
            "active_periods": active_periods,
            "burn_out": int(schedule.get("burn_out", 0)),
            "total_periods": total_periods,
        }
    )
    return context


def _build_aggregate_histogram_note(histogram_context):
    benchmark_labels = histogram_context.get("benchmark_labels", [])
    if not benchmark_labels:
        benchmark_text = "MATLAB benchmarks"
    elif len(benchmark_labels) == 1:
        benchmark_text = benchmark_labels[0]
    elif len(benchmark_labels) == 2:
        benchmark_text = f"{benchmark_labels[0]} and {benchmark_labels[1]}"
    else:
        benchmark_text = ", ".join(benchmark_labels[:-1]) + f", and {benchmark_labels[-1]}"
    base_text = (
        "The reported histograms summarize the distribution of the simulated series for each displayed aggregate. "
        "Each panel plots percent log deviations from deterministic steady state. "
        f"The solid colored line reports the Global solution; dashed gray lines report {benchmark_text}. "
    )
    if histogram_context.get("theoretical_distribution_params"):
        benchmark_source_text = (
            "The 1st Order Approx. benchmark plots the Gaussian density implied by its theoretical moments, while the remaining benchmark histograms use the MATLAB `shocks_simul` active windows stored in `ModelData_simulation`."
        )
    else:
        benchmark_source_text = (
            "Benchmark histograms use the MATLAB `shocks_simul` active windows stored in `ModelData_simulation`."
        )
    if histogram_context.get("mode") == "long_ergodic":
        return (
            base_text
            + "The global-solution histogram is computed from the retained ergodic simulation sample after discarding "
            f"{histogram_context.get('burn_in', 0)} burn-in periods from each simulation path. "
            f"The reported nonlinear sample contains {histogram_context.get('active_observations', 0)} observations. "
            + benchmark_source_text
        )
    return (
        base_text
        + "The global-solution histogram is computed from the active window of the common-shock simulation, with "
        f"{histogram_context.get('burn_in', 0)} burn-in periods, "
        f"{histogram_context.get('active_periods', 0)} active-shock periods, and "
        f"{histogram_context.get('burn_out', 0)} burn-out periods "
        f"({histogram_context.get('total_periods', 0)} total). "
        f"The reported nonlinear sample contains {histogram_context.get('active_periods', 0)} observations. "
        + benchmark_source_text
    )


def render_aggregate_ir_outputs(*, config, irs_dir, econ_model, gir_data, postprocess_context):
    ir_render_context = postprocess_context.get("ir_render_context") if postprocess_context else None
    if not ir_render_context:
        return

    for sector_idx in ir_render_context["sectors_to_plot"]:
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        print(f"\n  Aggregate IRs: {sector_label} (sector {sector_idx + 1})")
        for ir_variable in ir_render_context["ir_variables"]:
            print(f"    Plotting aggregate variable: {ir_variable}")
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=ir_render_context["matlab_ir_data"],
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=ir_render_context["shock_sizes"],
                save_dir=irs_dir,
                analysis_name=config["analysis_name"],
                max_periods=ir_render_context["max_periods"],
                n_sectors=ir_render_context["n_sectors"],
                benchmark_methods=_resolve_ir_benchmark_methods(config),
                response_source=ir_render_context["ir_response_source"],
                agg_consumption_mode=True,
                negative_only=False,
                policies_ss=ir_render_context["policies_ss_np"],
                state_ss=ir_render_context["state_ss_np"],
                P_ergodic=ir_render_context["P_ergodic_np"],
                Pk_ergodic=ir_render_context["Pk_ergodic_np"],
                ergodic_price_aggregation=ir_render_context["ergodic_price_aggregation"],
            )


def render_aggregate_histogram_outputs(*, config, simulation_dir, analysis_variables_data, postprocess_context):
    if not analysis_variables_data or not postprocess_context:
        return

    reference_experiment_label = postprocess_context.get("reference_experiment_label")
    if not reference_experiment_label or reference_experiment_label not in analysis_variables_data:
        return
    histogram_context = postprocess_context.get("aggregate_histogram_context") or {}
    histogram_theoretical_params = histogram_context.get("theoretical_distribution_params")

    selected_methods = [
        (reference_experiment_label, "Global solution"),
        *AGGREGATE_HISTOGRAM_BENCHMARKS,
    ]
    ordered_histogram_data = {}
    missing_methods = []

    for source_label, display_label in selected_methods:
        series = analysis_variables_data.get(source_label)
        if series is None:
            if histogram_theoretical_params and display_label in histogram_theoretical_params:
                ordered_histogram_data[display_label] = {}
                continue
            missing_methods.append(source_label)
            continue
        filtered_series = {
            variable_label: series[variable_label]
            for variable_label in DEFAULT_AGGREGATE_IR_LABELS
            if variable_label in series
        }
        if filtered_series:
            ordered_histogram_data[display_label] = filtered_series

    if not ordered_histogram_data:
        return

    if missing_methods:
        print(
            "  Aggregate histograms skipped missing benchmarks: " + ", ".join(missing_methods),
            flush=True,
        )

    print("  Aggregate histograms: Global solution vs 1st-order and MIT benchmarks", flush=True)
    plot_ergodic_histograms(
        analysis_variables_data=ordered_histogram_data,
        save_dir=simulation_dir,
        analysis_name=config["analysis_name"],
        theo_dist_params=histogram_theoretical_params,
        benchmark_order=[display_label for _, display_label in AGGREGATE_HISTOGRAM_BENCHMARKS],
    )
    if histogram_context.get("note_anchor_path"):
        _write_figure_note_tex(
            histogram_context["note_anchor_path"],
            _build_aggregate_histogram_note(histogram_context),
        )


def render_sectoral_stochss_outputs(
    *,
    config,
    simulation_dir,
    econ_model,
    stochastic_ss_states,
    stochastic_ss_policies,
    postprocess_context,
):
    if not stochastic_ss_policies:
        return

    ergodic_experiment_labels = postprocess_context.get("ergodic_experiment_labels") if postprocess_context else None
    if not ergodic_experiment_labels:
        return
    long_simulation_stochastic_ss_states = {
        label: stochastic_ss_states[label] for label in ergodic_experiment_labels if label in stochastic_ss_states
    }
    long_simulation_stochastic_ss_policies = {
        label: stochastic_ss_policies[label] for label in ergodic_experiment_labels if label in stochastic_ss_policies
    }
    if not long_simulation_stochastic_ss_policies:
        return

    upstreamness_data = postprocess_context.get("upstreamness_data") if postprocess_context else None
    for var_name in ["K", "L", "Y", "M", "Q"]:
        try:
            plot_sectoral_variable_stochss(
                stochastic_ss_states=long_simulation_stochastic_ss_states,
                stochastic_ss_policies=long_simulation_stochastic_ss_policies,
                variable_name=var_name,
                save_dir=simulation_dir,
                analysis_name=config["analysis_name"],
                econ_model=econ_model,
                upstreamness_data=upstreamness_data,
            )
        except Exception as exc:
            print(f"    Failed to create stochastic SS {var_name} plot: {exc}", flush=True)


def render_sectoral_ir_outputs(*, config, irs_dir, econ_model, gir_data, postprocess_context):
    ir_render_context = postprocess_context.get("ir_render_context") if postprocess_context else None
    if not ir_render_context:
        return

    sectoral_ir_variables = ir_render_context["sectoral_ir_variables"]
    for sector_idx in ir_render_context["sectors_to_plot"]:
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        if sectoral_ir_variables:
            print(f"\n  Sectoral IRs: {sector_label} (sector {sector_idx + 1})")
            for variable_name in sectoral_ir_variables:
                row_ref, desc = SECTORAL_VAR_DESC.get(variable_name, ("", variable_name))
                print(f"    {row_ref}  {variable_name:<20}  {desc}")
            print()

        for ir_variable in sectoral_ir_variables:
            row_ref, desc = SECTORAL_VAR_DESC.get(ir_variable, ("", ir_variable))
            print(f"    Plotting [{row_ref}] {ir_variable}: {desc} | {sector_label}")
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=ir_render_context["matlab_ir_data"],
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=[ir_render_context["largest_shock"]],
                save_dir=irs_dir,
                analysis_name=config["analysis_name"],
                max_periods=ir_render_context["max_periods"],
                n_sectors=ir_render_context["n_sectors"],
                benchmark_methods=_resolve_ir_benchmark_methods(config),
                response_source=ir_render_context["ir_response_source"],
                negative_only=True,
                policies_ss=ir_render_context["policies_ss_np"],
                state_ss=ir_render_context["state_ss_np"],
                P_ergodic=ir_render_context["P_ergodic_np"],
                Pk_ergodic=ir_render_context["Pk_ergodic_np"],
            )


def render_ergodic_sectoral_outputs(*, config, simulation_dir, econ_model, raw_simulation_data, postprocess_context):
    if not raw_simulation_data:
        return

    upstreamness_data = postprocess_context.get("upstreamness_data") if postprocess_context else None
    ergodic_raw_simulation_data = {
        label: sim_data
        for label, sim_data in raw_simulation_data.items()
        if sim_data.get("simulation_kind", "ergodic") == "ergodic"
    }
    if not ergodic_raw_simulation_data:
        return

    for var_name in ["K", "L", "Y", "M", "Q"]:
        try:
            plot_sectoral_variable_ergodic(
                raw_simulation_data=ergodic_raw_simulation_data,
                variable_name=var_name,
                save_dir=simulation_dir,
                analysis_name=config["analysis_name"],
                econ_model=econ_model,
                upstreamness_data=upstreamness_data,
            )
        except Exception as exc:
            print(f"    Failed to create ergodic {var_name} plot: {exc}", flush=True)


def postprocess_analysis(
    *,
    config,
    model_dir,
    analysis_dir,
    simulation_dir,
    irs_dir,
    econ_model,
    model_data,
    stats,
    policies_ss,
    state_ss,
    raw_simulation_data,
    analysis_variables_data,
    stochastic_ss_states,
    stochastic_ss_policies,
    stochastic_ss_data,
    gir_data,
    dynare_simulations,
    irs_path,
):
    prepared = prepare_postprocess_analysis(
        config=config,
        model_dir=model_dir,
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        irs_dir=irs_dir,
        econ_model=econ_model,
        model_data=model_data,
        stats=stats,
        policies_ss=policies_ss,
        state_ss=state_ss,
        raw_simulation_data=raw_simulation_data,
        analysis_variables_data=analysis_variables_data,
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
        stochastic_ss_data=stochastic_ss_data,
        gir_data=gir_data,
        dynare_simulations=dynare_simulations,
        irs_path=irs_path,
    )
    postprocess_context = prepared.get("postprocess_context")

    render_aggregate_ir_outputs(
        config=config,
        irs_dir=irs_dir,
        econ_model=econ_model,
        gir_data=gir_data,
        postprocess_context=postprocess_context,
    )
    render_sectoral_stochss_outputs(
        config=config,
        simulation_dir=simulation_dir,
        econ_model=econ_model,
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
        postprocess_context=postprocess_context,
    )
    render_sectoral_ir_outputs(
        config=config,
        irs_dir=irs_dir,
        econ_model=econ_model,
        gir_data=gir_data,
        postprocess_context=postprocess_context,
    )
    render_ergodic_sectoral_outputs(
        config=config,
        simulation_dir=simulation_dir,
        econ_model=econ_model,
        raw_simulation_data=raw_simulation_data,
        postprocess_context=postprocess_context,
    )

    return prepared


# Legacy long-ergodic price averaging path kept for later reuse:
# first_sim_data = raw_simulation_data[first_experiment_label]
# simul_policies = first_sim_data.get("simul_policies_full", first_sim_data["simul_policies"])
# P_ergodic, Pk_ergodic, Pm_ergodic = compute_ergodic_prices_from_simulation(
#     simul_policies,
#     policies_ss,
#     n_sectors,
# )


def _build_calibration_method_stats(
    *,
    stats,
    dynare_simulations,
    analysis_variables_data,
    raw_simulation_data,
    reference_experiment_label,
    policies_ss,
    state_ss,
    P_ergodic,
    Pk_ergodic,
    n_sectors,
    ergodic_price_aggregation,
):
    method_stats = {
        "1st": _copy_model_stats((stats.get("FirstOrder") or {}).get("ModelStats")),
        "2nd": _copy_model_stats((stats.get("SecondOrder") or {}).get("ModelStats")),
        "PF": _copy_model_stats((stats.get("PerfectForesight") or stats.get("Determ") or {}).get("ModelStats")),
        "MITShocks": _copy_model_stats((stats.get("MITShocks") or stats.get("MITShock") or {}).get("ModelStats")),
    }

    dynare_method_map = {
        "1st": ("FirstOrder", "First-Order (Dynare)"),
        "2nd": ("SecondOrder", "Second-Order (Dynare)"),
        "PF": ("PerfectForesight", "Perfect Foresight (Dynare)"),
        "MITShocks": ("MITShocks", "MIT Shocks (Dynare)"),
    }
    for column_label, (dynare_key, source_label) in dynare_method_map.items():
        dynare_simul = dynare_simulations.get(dynare_key)
        if dynare_simul is not None:
            method_stats[column_label] = compute_model_moments_from_dynare_simulation(
                dynare_simul,
                policies_ss=policies_ss,
                state_ss=state_ss,
                P_ergodic=P_ergodic,
                Pk_ergodic=Pk_ergodic,
                n_sectors=n_sectors,
                ergodic_price_aggregation=ergodic_price_aggregation,
                source_label=source_label,
            )

    aggregate_method_map = {
        "1st": "Log-Linear",
        "2nd": "SecondOrder",
        "PF": "PerfectForesight",
        "MITShocks": "MITShocks",
    }
    for column_label, method_name in aggregate_method_map.items():
        stats_dict = method_stats.get(column_label)
        analysis_vars = analysis_variables_data.get(method_name)
        if stats_dict is not None and analysis_vars is not None:
            _override_aggregate_rows_from_analysis_vars(stats_dict, analysis_vars)

    nonlinear_sim_data = raw_simulation_data[reference_experiment_label]
    method_stats["Nonlinear"] = compute_model_moments_with_consistent_aggregation(
        simul_obs=nonlinear_sim_data["simul_obs"],
        simul_policies=nonlinear_sim_data["simul_policies"],
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        n_sectors=n_sectors,
        ergodic_price_aggregation=ergodic_price_aggregation,
    )

    if reference_experiment_label.endswith(" (ergodic)"):
        common_shock_label = reference_experiment_label[: -len(" (ergodic)")] + " (common shocks)"
    else:
        common_shock_label = f"{reference_experiment_label} (common shocks)"
    common_shock_sim_data = raw_simulation_data.get(common_shock_label)
    if common_shock_sim_data is not None:
        method_stats["Nonlinear-CS"] = compute_model_moments_with_consistent_aggregation(
            simul_obs=common_shock_sim_data["simul_obs"],
            simul_policies=common_shock_sim_data["simul_policies"],
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            n_sectors=n_sectors,
            ergodic_price_aggregation=ergodic_price_aggregation,
        )

    return {label: stats_dict for label, stats_dict in method_stats.items() if stats_dict is not None}


def _copy_model_stats(model_stats):
    if not isinstance(model_stats, dict):
        return None
    return dict(model_stats)


def _override_aggregate_rows_from_analysis_vars(model_stats, analysis_vars):
    required_keys = {"Agg. Consumption", "Agg. Investment", "Agg. GDP", "Agg. Labor"}
    if not required_keys.issubset(analysis_vars):
        return

    c_series = _as_float_array(analysis_vars.get("Agg. Consumption"))
    i_series = _as_float_array(analysis_vars.get("Agg. Investment"))
    y_series = _as_float_array(analysis_vars.get("Agg. GDP"))
    l_series = _as_float_array(analysis_vars.get("Agg. Labor"))

    if c_series.size:
        model_stats["sigma_C_agg"] = _matlab_std(c_series)
    if i_series.size:
        model_stats["sigma_I_agg"] = _matlab_std(i_series)
    if y_series.size:
        model_stats["sigma_VA_agg"] = _matlab_std(y_series)
    if l_series.size:
        labor_sigma = _matlab_std(l_series)
        model_stats["sigma_L_agg"] = labor_sigma
        model_stats["sigma_L_hc_agg"] = labor_sigma
    if c_series.size and l_series.size:
        model_stats["corr_L_C_agg"] = _safe_corr(l_series, c_series)
    if c_series.size and i_series.size:
        model_stats["corr_I_C_agg"] = _safe_corr(i_series, c_series)


def _as_float_array(values):
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _matlab_std(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return 0.0
    return float(np.std(x, ddof=1))
