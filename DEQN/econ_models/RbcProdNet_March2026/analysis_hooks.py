import os
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from DEQN.econ_models.RbcProdNet_March2026.aggregation import (
    compute_model_moments_with_consistent_aggregation,
    compute_ergodic_prices_from_simulation,
    compute_ergodic_steady_state,
    create_perfect_foresight_descriptive_stats,
    create_theoretical_descriptive_stats,
    get_loglinear_distribution_params,
    process_simulation_with_consistent_aggregation,
    recenter_analysis_variables,
)
from DEQN.econ_models.RbcProdNet_March2026.matlab_irs import load_matlab_irs
from DEQN.econ_models.RbcProdNet_March2026.plots import (
    plot_sector_ir_by_shock_size,
    plot_sectoral_variable_ergodic,
    plot_sectoral_variable_stochss,
)

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
    del simul_obs, config
    simul_policies_mean = jnp.mean(simul_policies, axis=0)
    n = econ_model.n_sectors
    return {
        "P_weights": simul_policies_mean[8 * n : 9 * n],
        "Pk_weights": simul_policies_mean[2 * n : 3 * n],
        "Pm_weights": simul_policies_mean[3 * n : 4 * n],
    }


def compute_analysis_variables(econ_model, state_logdev, policy_logdev, analysis_context) -> Dict[str, Any]:
    return econ_model.get_analysis_variables(
        state_logdev,
        policy_logdev,
        analysis_context["P_weights"],
        analysis_context["Pk_weights"],
        analysis_context["Pm_weights"],
    )


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
        if configured_label not in raw_simulation_data:
            available = list(raw_simulation_data.keys())
            raise ValueError(
                "aggregation_reference_experiment must match an analyzed experiment label. "
                f"Got '{configured_label}', available labels: {available}"
            )
        return configured_label

    return next(iter(raw_simulation_data))


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
    del model_data
    n_sectors = econ_model.n_sectors
    analysis_variables_data = dict(analysis_variables_data)
    theoretical_stats: Dict[str, Any] = {}
    histogram_theo_params: Dict[str, Any] = {}

    reference_experiment_label = _resolve_reference_experiment_label(config, raw_simulation_data)
    print(
        f"  Using '{reference_experiment_label}' as the fixed-price aggregation reference.",
        flush=True,
    )
    reference_sim_data = raw_simulation_data[reference_experiment_label]
    simul_policies = reference_sim_data["simul_policies"]

    P_ergodic, Pk_ergodic, Pm_ergodic = compute_ergodic_prices_from_simulation(simul_policies, policies_ss, n_sectors)

    ss_corrections = compute_ergodic_steady_state(
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        Pm_ergodic=Pm_ergodic,
        n_sectors=n_sectors,
    )

    for exp_label in list(analysis_variables_data.keys()):
        analysis_variables_data[exp_label] = recenter_analysis_variables(
            analysis_variables_data[exp_label],
            ss_corrections,
        )

    dynare_simul_1storder = dynare_simulations.get("FirstOrder")
    dynare_simul_so = dynare_simulations.get("SecondOrder")
    dynare_simul_pf = dynare_simulations.get("PerfectForesight")
    dynare_simul_mit = dynare_simulations.get("MITShocks")

    if dynare_simul_1storder is not None:
        firstorder_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_1storder,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
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
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
            burn_in=0,
            source_label="Second-Order",
        )
        analysis_variables_data["SecondOrder"] = secondorder_analysis_vars
        print("  Loaded Second-Order simulation series.")

    if "TheoStats" in stats:
        theo_stats = stats["TheoStats"]
        print("  Using TheoStats for Log-Linear moments (mean=0, skew=0, kurt=0)", flush=True)

        loglin_theo_stats = create_theoretical_descriptive_stats(theo_stats=theo_stats, label="Log-Linear")
        theoretical_stats.update(loglin_theo_stats)

        theo_dist_params = get_loglinear_distribution_params(theo_stats)
        histogram_theo_params["Log-Linear"] = theo_dist_params

        if "Log-Linear" not in analysis_variables_data:
            analysis_variables_data["Log-Linear"] = {var_name: jnp.zeros(10) for var_name in theo_dist_params.keys()}

        for var_name, params_dict in theo_dist_params.items():
            print(f"    {var_name}: sigma={params_dict['std']*100:.4f}%", flush=True)

    pf_stats_key = "PerfectForesight" if "PerfectForesight" in stats else ("Determ" if "Determ" in stats else None)
    if pf_stats_key:
        pf_stats = stats[pf_stats_key]
        pf_model_stats = pf_stats.get("ModelStats") if isinstance(pf_stats, dict) else None
        if dynare_simul_pf is None:
            if pf_model_stats is not None:
                print(f"  Using Statistics.{pf_stats_key} (+ ModelStats) for Perfect Foresight moments", flush=True)
            else:
                print(f"  Using Statistics.{pf_stats_key} for Perfect Foresight moments", flush=True)

            pf_theo_stats = create_perfect_foresight_descriptive_stats(
                determ_stats=pf_stats,
                label="PerfectForesight",
                n_sectors=n_sectors,
                model_stats=pf_model_stats,
                policies_ss=policies_ss,
            )
            theoretical_stats.update(pf_theo_stats)

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
                if len(policies_std) > 11 * n + 1:
                    print(f"    Agg. Consumption (utility): sigma={float(policies_std[11*n])*100:.4f}%", flush=True)
                    print(
                        f"    Agg. Labor (CES fallback): sigma={float(policies_std[11*n+1])*100:.4f}%",
                        flush=True,
                    )
        else:
            print("  Perfect Foresight moments will use Perfect Foresight (Dynare) simulation series.")

    if dynare_simul_pf is not None:
        pf_analysis_vars = process_simulation_with_consistent_aggregation(
            simul_data=dynare_simul_pf,
            policies_ss=policies_ss,
            state_ss=state_ss,
            P_ergodic=P_ergodic,
            Pk_ergodic=Pk_ergodic,
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
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
            Pm_ergodic=Pm_ergodic,
            n_sectors=n_sectors,
            burn_in=0,
            source_label="MITShocks",
        )
        analysis_variables_data["MITShocks"] = mit_analysis_vars
        print("  Loaded MITShocks simulation series.")

    calibration_method_stats = _build_calibration_method_stats(
        stats=stats,
        analysis_variables_data=analysis_variables_data,
        raw_simulation_data=raw_simulation_data,
        reference_experiment_label=reference_experiment_label,
        policies_ss=policies_ss,
        state_ss=state_ss,
        P_ergodic=P_ergodic,
        Pk_ergodic=Pk_ergodic,
        Pm_ergodic=Pm_ergodic,
        n_sectors=n_sectors,
    )

    matlab_ir_dir = os.path.join(model_dir, "MATLAB", "IRs")
    matlab_ir_data = load_matlab_irs(
        matlab_ir_dir=matlab_ir_dir,
        shock_sizes=config.get("ir_shock_sizes", [5, 10, 20]),
        irs_file_path=irs_path,
    )

    sectors_to_plot = config.get("ir_sectors_to_plot", [0, 2, 23])
    ir_variables = config.get("ir_variables_to_plot", ["Agg. Consumption"])
    if isinstance(ir_variables, str):
        ir_variables = [ir_variables]

    unsupported_aggregate_ir_vars = {"Agg. Capital"}
    filtered_ir_variables = [v for v in ir_variables if v not in unsupported_aggregate_ir_vars]
    dropped_ir_variables = [v for v in ir_variables if v in unsupported_aggregate_ir_vars]
    if dropped_ir_variables:
        print(
            "  Note: aggregate capital IR benchmark is not available; "
            f"skipping {dropped_ir_variables} from ir_variables_to_plot."
        )
    ir_variables = filtered_ir_variables

    sectoral_ir_variables = [
        label
        for label in _get_requested_sectoral_ir_variables(config)
        if label in SUPPORTED_SECTORAL_IR_LABELS
    ]
    shock_sizes = config.get("ir_shock_sizes", [5, 10, 20])
    if not shock_sizes:
        raise ValueError("ir_shock_sizes must contain at least one shock size.")
    max_periods = config.get("ir_max_periods", 80)
    configured_ir_methods = config.get("ir_methods", ["IR_stoch_ss"])
    if isinstance(configured_ir_methods, str):
        configured_ir_methods = [configured_ir_methods]
    if "GIR" in configured_ir_methods and "IR_stoch_ss" in configured_ir_methods:
        ir_response_source = "both"
    elif "GIR" in configured_ir_methods:
        ir_response_source = "GIR"
    else:
        ir_response_source = "IR_stoch_ss"

    largest_shock = max(shock_sizes)
    policies_ss_np = np.asarray(policies_ss)
    P_ergodic_np = np.asarray(P_ergodic)

    for sector_idx in sectors_to_plot:
        sector_label = econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        for ir_variable in ir_variables:
            is_agg_consumption = ir_variable == "Agg. Consumption"
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=matlab_ir_data,
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=shock_sizes if is_agg_consumption else [largest_shock],
                save_dir=irs_dir,
                analysis_name=config["analysis_name"],
                max_periods=max_periods,
                n_sectors=n_sectors,
                benchmark_method=config.get("ir_benchmark_method", "PerfectForesight"),
                response_source=ir_response_source,
                agg_consumption_mode=is_agg_consumption,
                negative_only=not is_agg_consumption,
                policies_ss=policies_ss_np,
                P_ergodic=P_ergodic_np,
            )

    upstreamness_data = econ_model.upstreamness()
    if stochastic_ss_policies:
        for var_name in ["K", "L", "Y", "M", "Q"]:
            try:
                plot_sectoral_variable_stochss(
                    stochastic_ss_states=stochastic_ss_states,
                    stochastic_ss_policies=stochastic_ss_policies,
                    variable_name=var_name,
                    save_dir=simulation_dir,
                    analysis_name=config["analysis_name"],
                    econ_model=econ_model,
                    upstreamness_data=upstreamness_data,
                )
            except Exception as exc:
                print(f"    Failed to create stochastic SS {var_name} plot: {exc}", flush=True)

    for sector_idx in sectors_to_plot:
        sector_label = econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
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
                matlab_ir_data=matlab_ir_data,
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=[largest_shock],
                save_dir=irs_dir,
                analysis_name=config["analysis_name"],
                max_periods=max_periods,
                n_sectors=n_sectors,
                benchmark_method=config.get("ir_benchmark_method", "PerfectForesight"),
                response_source=ir_response_source,
                negative_only=True,
                policies_ss=policies_ss_np,
                P_ergodic=P_ergodic_np,
            )

    if raw_simulation_data:
        for var_name in ["K", "L", "Y", "M", "Q"]:
            try:
                plot_sectoral_variable_ergodic(
                    raw_simulation_data=raw_simulation_data,
                    variable_name=var_name,
                    save_dir=simulation_dir,
                    analysis_name=config["analysis_name"],
                    econ_model=econ_model,
                    upstreamness_data=upstreamness_data,
                )
            except Exception as exc:
                print(f"    Failed to create ergodic {var_name} plot: {exc}", flush=True)

    return {
        "analysis_variables_data": analysis_variables_data,
        "calibration_method_stats": calibration_method_stats,
        "theoretical_stats": theoretical_stats,
        "histogram_theo_params": histogram_theo_params,
        "matlab_ir_data": matlab_ir_data,
        "upstreamness_data": upstreamness_data,
        "stochastic_ss_data": stochastic_ss_data,
    }


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
    analysis_variables_data,
    raw_simulation_data,
    reference_experiment_label,
    policies_ss,
    state_ss,
    P_ergodic,
    Pk_ergodic,
    Pm_ergodic,
    n_sectors,
):
    method_stats = {
        "1st": _copy_model_stats((stats.get("FirstOrder") or {}).get("ModelStats")),
        "2nd": _copy_model_stats((stats.get("SecondOrder") or {}).get("ModelStats")),
        "PF": _copy_model_stats((stats.get("PerfectForesight") or stats.get("Determ") or {}).get("ModelStats")),
        "MITShocks": _copy_model_stats((stats.get("MITShocks") or stats.get("MITShock") or {}).get("ModelStats")),
    }

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
        Pm_ergodic=Pm_ergodic,
        n_sectors=n_sectors,
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
        model_stats["sigma_C_agg"] = float(np.std(c_series))
    if i_series.size:
        model_stats["sigma_I_agg"] = float(np.std(i_series))
    if y_series.size:
        model_stats["sigma_VA_agg"] = float(np.std(y_series))
    if l_series.size:
        labor_sigma = float(np.std(l_series))
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
