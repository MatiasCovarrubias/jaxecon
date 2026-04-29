import os
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from DEQN.analysis.shock_keys import build_shock_key
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
from DEQN.analysis.welfare_outputs import (
    WELFARE_BOTH_RECENTERED_LABEL,
    WELFARE_L_FIXED_AT_DSS_LABEL,
    _compute_counterfactual_welfare_cost_from_sample,
)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    scalar = float(arr.ravel()[0])
    return scalar if np.isfinite(scalar) else None


def _safe_corr(x_values: Any, y_values: Any) -> Optional[float]:
    try:
        x = np.asarray(x_values, dtype=float).ravel()
        y = np.asarray(y_values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _format_table_value(value: Any) -> str:
    scalar = _as_float(value)
    if scalar is None:
        return "--"
    return f"{scalar:.3f}"


def _nanmean_or_none(values: Any) -> Optional[float]:
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _latex_label_token(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(text))

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
    ("Log-Linear", "1st Order Approximation"),
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


def compute_welfare_outputs(*, experiment_label, selected_results, econ_model, welfare_fn, welfare_ss, config) -> Dict[str, Any]:
    welfare_outputs = {}
    welfare_specs = [
        (
            WELFARE_BOTH_RECENTERED_LABEL,
            {"recenter_consumption": True, "recenter_labor": True},
        ),
        (
            WELFARE_L_FIXED_AT_DSS_LABEL,
            {"fix_labor_at_dss": True},
        ),
    ]
    for label, options in welfare_specs:
        full_label = f"{experiment_label} ({label})"
        try:
            welfare_outputs[full_label] = _compute_counterfactual_welfare_cost_from_sample(
                econ_model=econ_model,
                welfare_fn=welfare_fn,
                welfare_ss=welfare_ss,
                policies_logdev=selected_results["simul_policies"],
                config_dict=config,
                **options,
            )
        except ValueError as exc:
            print(f"    Warning: welfare counterfactual {full_label} skipped ({exc}).", flush=True)
    return welfare_outputs


def get_report_sections(*, config, analysis_dir, simulation_dir, irs_dir, econ_model, helpers=None):
    helpers = helpers or {}
    analysis_named_path = helpers["analysis_named_path"]
    build_simple_figure_spec = helpers["build_simple_figure_spec"]
    make_safe_plot_label = helpers["make_safe_plot_label"]
    caption_label = helpers["caption_label"]
    join_labels = helpers["join_labels"]
    format_percent_list = helpers["format_percent_list"]
    describe_ir_benchmark_methods = helpers["describe_ir_benchmark_methods"]
    describe_deqn_ir_note = helpers["describe_deqn_ir_note"]
    existing_subfigures = helpers["existing_subfigures"]
    existing_subfigure_groups = helpers["existing_subfigure_groups"]

    analysis_name = config.get("analysis_name") or "analysis"
    sections = []

    def add_table_section(title, tex_paths):
        existing_paths = [path for path in tex_paths if os.path.exists(path)]
        if existing_paths:
            sections.append({"title": title, "tables": existing_paths, "figures": []})

    def add_figure_section(title, figure_paths):
        existing_figures = []
        for figure in figure_paths:
            figure_path = figure["path"] if isinstance(figure, dict) else figure
            if os.path.exists(figure_path):
                existing_figures.append(figure)
        if existing_figures:
            sections.append({"title": title, "tables": [], "figures": existing_figures})

    def add_grouped_figure_section(title, figure_groups):
        existing_groups = []
        for figure_group in figure_groups:
            subfigures = existing_subfigures(figure_group.get("subfigures", []))
            if subfigures:
                existing_group = dict(figure_group)
                existing_group["subfigures"] = subfigures
                caption_builder = existing_group.get("caption_builder")
                if caption_builder is not None:
                    existing_group["caption"] = caption_builder(subfigures)
                note_builder = existing_group.get("note_builder")
                if note_builder is not None:
                    existing_group["note_text"] = note_builder(subfigures)
                existing_groups.append(existing_group)
        if existing_groups:
            sections.append({"title": title, "tables": [], "figures": existing_groups})

    def add_nested_grouped_figure_section(title, figure_specs):
        existing_figures = []
        for figure_spec in figure_specs:
            subfigure_groups = existing_subfigure_groups(figure_spec.get("subfigure_groups", []))
            if subfigure_groups:
                existing_figure = dict(figure_spec)
                existing_figure["subfigure_groups"] = subfigure_groups
                existing_figures.append(existing_figure)
        if existing_figures:
            sections.append({"title": title, "tables": [], "figures": existing_figures})

    add_table_section(
        "1. Model vs. Data Moments",
        [os.path.join(analysis_dir, f"calibration_table_{analysis_name}.tex")],
    )

    aggregate_ir_figures = []
    aggregate_variable_captions = {
        "Agg. Consumption": "Consumption",
        "Agg. Investment": "Investment",
        "Agg. GDP": "GDP",
        "Agg. Labor": "Labor",
        "Agg. Capital": "Capital",
        "Intratemporal Utility": "Intratemporal Utility",
    }
    aggregate_variable_note_labels = {
        "Agg. Consumption": "consumption",
        "Agg. Investment": "investment",
        "Agg. GDP": "GDP",
        "Agg. Labor": "labor",
        "Agg. Capital": "capital",
        "Intratemporal Utility": "intratemporal utility",
    }
    ir_shock_sizes = list(config.get("ir_shock_sizes", []))
    largest_ir_shock = max(ir_shock_sizes) if ir_shock_sizes else None
    aggregate_benchmark_labels = describe_ir_benchmark_methods(config)
    deqn_ir_note = describe_deqn_ir_note(config)
    aggregate_ir_variables = list(DEFAULT_AGGREGATE_IR_LABELS)
    aggregate_ir_main_text_figures = []
    aggregate_ir_appendix_figures = []
    paper_main_aggregate_variables = ["Agg. Consumption", "Agg. GDP"]
    paper_appendix_aggregate_variables = ["Agg. Investment", "Agg. Capital", "Agg. Labor"]

    def _aggregate_ir_largest_negative_path(variable_name, safe_sector):
        safe_variable = make_safe_plot_label(variable_name)
        return analysis_named_path(
            irs_dir,
            f"IR_{safe_variable}_{safe_sector}_largest_negative",
            analysis_name,
            ".png",
        )

    def _aggregate_note_label(variable_name):
        return aggregate_variable_note_labels.get(variable_name, caption_label(variable_name))

    def _aggregate_caption_label(variable_name):
        return aggregate_variable_captions.get(variable_name, variable_name)

    def _build_grouped_aggregate_ir_note(*, sector_label, variable_names):
        displayed_labels = [_aggregate_note_label(variable_name) for variable_name in variable_names]
        shock_text = (
            f"The panels show responses to the largest discovered negative TFP shock in {sector_label} "
            f"({largest_ir_shock} percent). "
            if largest_ir_shock is not None
            else f"The panels show responses to a negative TFP shock in {sector_label}. "
        )
        return (
            f"{shock_text}"
            f"The panels report aggregate {join_labels(displayed_labels)}. "
            f"{deqn_ir_note}"
            "The horizontal axis reports periods after impact. "
            "The vertical axis reports impulse responses in percent. "
            "Dashed lines report comparison IRs from the "
            f"{aggregate_benchmark_labels}; these comparison IRs are anchored at the deterministic "
            "steady state."
        )
    for sector_idx in config.get("ir_sectors_to_plot", []):
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        safe_sector = make_safe_plot_label(sector_label)
        for variable_name in aggregate_ir_variables:
            safe_variable = make_safe_plot_label(variable_name)
            figure_caption = aggregate_variable_captions.get(variable_name, variable_name)
            note_label = aggregate_variable_note_labels.get(variable_name, variable_name)
            shock_layout_text = (
                f"The rows correspond to {format_percent_list(ir_shock_sizes)} percent TFP shocks in {sector_label}; "
                "the left column shows negative shocks and the right column positive shocks. "
                if ir_shock_sizes
                else ""
            )
            aggregate_ir_figures.append(
                build_simple_figure_spec(
                    analysis_named_path(irs_dir, f"IR_{safe_variable}_{safe_sector}", analysis_name, ".png"),
                    f"Aggregate {caption_label(figure_caption)} response to a TFP shock in {sector_label}.",
                    note_text=(
                        f"{shock_layout_text}"
                        f"The figure plots the response of aggregate {note_label}. "
                        f"{deqn_ir_note}"
                        "The horizontal axis reports periods after impact. "
                        "The vertical axis reports impulse responses in percent. "
                        "Dashed lines report comparison IRs from the "
                        f"{aggregate_benchmark_labels}; these comparison IRs are anchored at the deterministic "
                        "steady state."
                    ),
                )
            )
        main_subfigures = [
            {
                "path": _aggregate_ir_largest_negative_path(variable_name, safe_sector),
                "caption": _aggregate_caption_label(variable_name),
            }
            for variable_name in paper_main_aggregate_variables
            if variable_name in aggregate_ir_variables
        ]
        main_exists = [os.path.exists(subfigure["path"]) for subfigure in main_subfigures]
        if len(main_subfigures) == len(paper_main_aggregate_variables) and all(main_exists):
            aggregate_ir_main_text_figures.append(
                {
                    "caption": f"Aggregate consumption and GDP responses to the largest negative TFP shock in {sector_label}.",
                    "note_text": _build_grouped_aggregate_ir_note(
                        sector_label=sector_label,
                        variable_names=paper_main_aggregate_variables,
                    ),
                    "subfigure_groups": [{"subfigures": main_subfigures}],
                }
            )

        appendix_subfigures = [
            {
                "path": _aggregate_ir_largest_negative_path(variable_name, safe_sector),
                "caption": _aggregate_caption_label(variable_name),
            }
            for variable_name in paper_appendix_aggregate_variables
            if variable_name in aggregate_ir_variables
        ]
        appendix_exists = [os.path.exists(subfigure["path"]) for subfigure in appendix_subfigures]
        if len(appendix_subfigures) == len(paper_appendix_aggregate_variables) and all(appendix_exists):
            aggregate_ir_appendix_figures.append(
                {
                    "caption": f"Aggregate investment, capital, and labor responses to the largest negative TFP shock in {sector_label}.",
                    "note_text": _build_grouped_aggregate_ir_note(
                        sector_label=sector_label,
                        variable_names=paper_appendix_aggregate_variables,
                    ),
                    "subfigure_groups": [{"subfigures": appendix_subfigures}],
                }
            )
    add_figure_section("2. Aggregate Impulse Responses", aggregate_ir_figures)
    add_nested_grouped_figure_section(
        "2A. Paper Aggregate Impulse Responses",
        aggregate_ir_main_text_figures,
    )
    add_nested_grouped_figure_section(
        "2B. Appendix Aggregate Impulse Responses",
        aggregate_ir_appendix_figures,
    )
    add_table_section(
        "2C. Impulse Response Nonlinearity Summary",
        [os.path.join(analysis_dir, f"ir_nonlinearity_summary_{analysis_name}.tex")],
    )
    add_table_section(
        "2D. CIR Optimal Attenuation Summary",
        [os.path.join(irs_dir, f"cir_analysis_{analysis_name}.tex")],
    )

    add_figure_section(
        "3. Sectoral Variables in Stochastic Steady State",
        [
            build_simple_figure_spec(
                analysis_named_path(simulation_dir, f"sectoral_{variable_name}_stochss", analysis_name, ".png"),
                f"Sectoral {variable_caption} at the stochastic steady state.",
            )
            for variable_name, variable_caption in [
                ("k", "capital"),
                ("l", "labor"),
                ("y", "value added"),
                ("m", "intermediates"),
                ("q", "gross output"),
            ]
        ],
    )

    add_table_section(
        "4. Aggregate Stochastic Steady State",
        [
            os.path.join(analysis_dir, f"stochastic_ss_aggregates_{analysis_name}.tex"),
        ],
    )

    add_table_section(
        "5. Descriptive Statistics",
        [
            os.path.join(simulation_dir, f"descriptive_stats_{analysis_name}.tex"),
        ],
    )

    histogram_variable_groups = [
        (
            "Expenditure Aggregates",
            [
                ("Agg. Consumption", "Consumption"),
                ("Agg. Investment", "Investment"),
                ("Agg. GDP", "GDP"),
            ],
        ),
        (
            "Inputs and Utility",
            [
                ("Agg. Capital", "Capital"),
                ("Agg. Labor", "Labor"),
                ("Intratemporal Utility", "Utility"),
            ],
        ),
    ]

    def _histogram_filename(variable_name):
        return variable_name.replace(" ", "_").replace(".", "").replace("/", "_")

    histogram_group_note_path = os.path.join(simulation_dir, f"aggregate_histograms_{analysis_name}_note.tex")
    aggregate_histogram_figures = []
    for _, variable_specs in histogram_variable_groups:
        for variable_name, variable_caption in variable_specs:
            aggregate_histogram_figures.append(
                build_simple_figure_spec(
                    analysis_named_path(
                        simulation_dir,
                        f"Histogram_{_histogram_filename(variable_name)}",
                        analysis_name,
                        ".png",
                    ),
                    f"Aggregate {caption_label(variable_caption)} distribution: Global Solution, 1st Order Approximation, and MIT shocks.",
                    note_text=(
                        f"The figure compares the distribution of aggregate {caption_label(variable_caption)} across "
                        "the global solution, the 1st-order approximation, and MIT shocks."
                    ),
                    note_path=histogram_group_note_path,
                )
            )
    add_figure_section("6. Aggregate Distribution Histograms", aggregate_histogram_figures)

    add_table_section(
        "7. Welfare Cost of Business Cycles",
        [os.path.join(analysis_dir, f"welfare_{analysis_name}.tex")],
    )

    sectoral_ir_groups = []
    largest_sectoral_shock = max(config.get("ir_shock_sizes", [0])) if config.get("ir_shock_sizes") else None
    sectoral_benchmark_labels = describe_ir_benchmark_methods(config)
    sectoral_group_specs = [
        (
            "Shocked Sector Inputs",
            [("Lj", "Labor"), ("Ij", "Investment"), ("Mj", "Intermediates"), ("Yj", "Value Added"), ("Kj", "Capital")],
        ),
        (
            "Shocked Sector Outputs",
            [
                ("Cj", "Consumption"),
                ("Pj", "Price"),
                ("Moutj", "Intermediate Sales"),
                ("Ioutj", "Investment Sales"),
                ("Qj", "Gross Output"),
            ],
        ),
        (
            "Client Sector Inputs",
            [
                ("Lj_client", "Labor"),
                ("Ij_client", "Investment"),
                ("Mj_client", "Intermediates"),
                ("Yj_client", "Value Added"),
                ("Pmj_client", "Intermediate Price"),
                ("gammaij_client", "Expenditure Share"),
            ],
        ),
        (
            "Client Sector Outputs",
            [
                ("Cj_client", "Consumption"),
                ("Pj_client", "Price"),
                ("Moutj_client", "Intermediate Sales"),
                ("Ioutj_client", "Investment Sales"),
                ("Qj_client", "Gross Output"),
            ],
        ),
    ]
    for sector_idx in config.get("ir_sectors_to_plot", []):
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        safe_sector = make_safe_plot_label(sector_label)
        configured_sectoral_variables = set(config.get("sectoral_ir_variables_to_plot", []))
        for group_title, variable_specs in sectoral_group_specs:
            subfigures = []
            for variable_name, variable_caption in variable_specs:
                if variable_name not in configured_sectoral_variables:
                    continue
                safe_variable = make_safe_plot_label(variable_name)
                subfigures.append(
                    {
                        "path": analysis_named_path(
                            irs_dir,
                            f"IR_{safe_variable}_{safe_sector}",
                            analysis_name,
                            ".png",
                        ),
                        "caption": variable_caption,
                    }
                )

            if subfigures:
                shock_text = (
                    f"The panels plot responses to a negative {largest_sectoral_shock} percent TFP shock in {sector_label}."
                    if largest_sectoral_shock
                    else f"The panels plot responses to the sectoral TFP shock used in the analysis for {sector_label}."
                )
                sectoral_ir_groups.append(
                    {
                        "caption": f"{group_title} for {sector_label}.",
                        "note_text": (
                            f"{shock_text} "
                            f"{deqn_ir_note}"
                            "The horizontal axis reports periods after impact. "
                            "The vertical axis reports impulse responses in percent. "
                            "Dashed lines report comparison IRs from the "
                            f"{sectoral_benchmark_labels}; these comparison IRs are anchored at the deterministic "
                            "steady state."
                        ),
                        "subfigures": subfigures,
                    }
                )
    add_grouped_figure_section("8. Sectoral Impulse Responses", sectoral_ir_groups)

    add_figure_section(
        "9. Ergodic Mean Sectoral Variables",
        [
            build_simple_figure_spec(
                analysis_named_path(simulation_dir, f"sectoral_{variable_name}_ergodic", analysis_name, ".png"),
                f"Ergodic mean sectoral {variable_caption}.",
            )
            for variable_name, variable_caption in [
                ("k", "capital"),
                ("l", "labor"),
                ("y", "value added"),
                ("m", "intermediates"),
                ("q", "gross output"),
            ]
        ],
    )

    return sections


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


def _extract_matlab_upstreamness(model_data, fallback_upstreamness):
    diagnostics = model_data.get("Diagnostics") if isinstance(model_data, dict) else None
    upstreamness = diagnostics.get("upstreamness") if isinstance(diagnostics, dict) else None
    if not isinstance(upstreamness, dict):
        return fallback_upstreamness or {}

    result = {}
    for source_key, target_key in [
        ("U_M", "U_M"),
        ("U_I", "U_I"),
        ("U_simple", "U_simple"),
        ("sectoral_shock_std", "shock_volatility"),
        ("shock_volatility", "shock_volatility"),
    ]:
        value = upstreamness.get(source_key)
        if value is not None and target_key not in result:
            result[target_key] = np.asarray(value, dtype=float).ravel()
    for key, value in (fallback_upstreamness or {}).items():
        result.setdefault(key, value)
    return result


def _get_gir_state_name_for_sector(gir_data, sector_idx, n_sectors):
    if not gir_data:
        return None
    if len(gir_data) != 1:
        raise ValueError(
            "CIR analysis expects exactly one nonlinear experiment in gir_data; "
            f"got {list(gir_data.keys())}."
        )
    first_exp_data = next(iter(gir_data.values()), {})
    for state_name, state_data in first_exp_data.items():
        state_idx = state_data.get("state_idx") if isinstance(state_data, dict) else None
        if state_idx in (sector_idx, n_sectors + sector_idx):
            return state_name
    for state_idx in (n_sectors + sector_idx, sector_idx):
        state_name = f"state_{state_idx}"
        if state_name in first_exp_data:
            return state_name
    return None


def _resolve_shock_data_by_key(state_data, shock_key):
    shock_data = state_data.get(shock_key)
    return shock_data if isinstance(shock_data, dict) else None


def _get_global_cir_for_sector(
    gir_data,
    *,
    experiment_name,
    sector_idx,
    shock_key,
    variable_name,
    n_sectors,
    max_periods,
    response_source,
):
    exp_data = gir_data.get(experiment_name, {})
    state_name = _get_gir_state_name_for_sector({experiment_name: exp_data}, sector_idx, n_sectors)
    if state_name is None:
        return None
    state_data = exp_data.get(state_name, {})
    candidate_key = f"{shock_key}_stochss" if response_source == "IR_stoch_ss" else shock_key
    shock_data = _resolve_shock_data_by_key(state_data, candidate_key)
    if not isinstance(shock_data, dict):
        return None
    variables = shock_data.get("gir_analysis_variables", {})
    series = variables.get(variable_name)
    if series is None:
        return None
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        return None
    horizon = min(int(max_periods), arr.size)
    return float(np.nansum(arr[:horizon]))


def _get_matlab_cir_horizon_for_sector(matlab_ir_data, *, shock_key, sector_idx):
    shock_data = matlab_ir_data.get(shock_key, {})
    sector_entry = (shock_data.get("sectors", {}) or {}).get(sector_idx, {})
    if not isinstance(sector_entry, dict):
        return None
    for aggregate_key in ("aggregate_perfect_foresight", "aggregate_first_order", "aggregate_second_order"):
        aggregate = sector_entry.get(aggregate_key, {})
        if isinstance(aggregate, dict) and aggregate.get("C_exp") is not None:
            arr = np.asarray(aggregate["C_exp"]).ravel()
            if arr.size > 0:
                return int(arr.size)
    return None


def _get_matlab_cir_for_sector(matlab_ir_data, *, shock_key, sector_idx, method):
    shock_data = matlab_ir_data.get(shock_key, {})
    sector_entry = (shock_data.get("sectors", {}) or {}).get(sector_idx, {})
    cir = sector_entry.get("cir", {}) if isinstance(sector_entry, dict) else {}
    cumulative = cir.get("cumulative_responses", {}) if isinstance(cir, dict) else {}
    if method in cumulative:
        return _as_float(cumulative.get(method))

    series_key = {
        "first_order": "aggregate_first_order",
        "second_order": "aggregate_second_order",
        "perfect_foresight": "aggregate_perfect_foresight",
    }.get(method)
    aggregate = sector_entry.get(series_key, {}) if isinstance(sector_entry, dict) else {}
    if isinstance(aggregate, dict) and aggregate.get("C_exp") is not None:
        return float(np.nansum(np.asarray(aggregate["C_exp"], dtype=float).ravel()))
    return None


def _build_cir_analysis_table(*, config, gir_data, matlab_ir_data, upstreamness_data, n_sectors):
    shock_sizes = get_available_shock_sizes(matlab_ir_data)
    if not shock_sizes or not gir_data:
        return None
    if len(gir_data) != 1:
        raise ValueError(
            "CIR analysis expects exactly one nonlinear experiment in gir_data; "
            f"got {list(gir_data.keys())}."
        )
    experiment_name = next(iter(gir_data))
    max_periods = int(config.get("ir_max_periods", 80))
    response_source = _resolve_ir_response_source(config)
    variable_name = "Agg. Consumption"

    rows = []
    for shock_size in shock_sizes:
        pos_key = build_shock_key("pos", shock_size)
        neg_key = build_shock_key("neg", shock_size)
        global_pos = []
        global_neg = []
        pf_pos = []
        pf_neg = []
        fo_neg = []
        fo_pos = []
        for sector_idx in range(n_sectors):
            g_pos = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=pos_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=_get_matlab_cir_horizon_for_sector(
                    matlab_ir_data, shock_key=pos_key, sector_idx=sector_idx
                )
                or max_periods,
                response_source=response_source,
            )
            g_neg = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=neg_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=_get_matlab_cir_horizon_for_sector(
                    matlab_ir_data, shock_key=neg_key, sector_idx=sector_idx
                )
                or max_periods,
                response_source=response_source,
            )
            p_pos = _get_matlab_cir_for_sector(
                matlab_ir_data, shock_key=pos_key, sector_idx=sector_idx, method="perfect_foresight"
            )
            p_neg = _get_matlab_cir_for_sector(
                matlab_ir_data, shock_key=neg_key, sector_idx=sector_idx, method="perfect_foresight"
            )
            f_pos = _get_matlab_cir_for_sector(
                matlab_ir_data, shock_key=pos_key, sector_idx=sector_idx, method="first_order"
            )
            f_neg = _get_matlab_cir_for_sector(
                matlab_ir_data, shock_key=neg_key, sector_idx=sector_idx, method="first_order"
            )
            global_pos.append(g_pos)
            global_neg.append(g_neg)
            pf_pos.append(p_pos)
            pf_neg.append(p_neg)
            fo_pos.append(f_pos)
            fo_neg.append(f_neg)

        global_pos_arr = np.asarray([np.nan if v is None else v for v in global_pos], dtype=float)
        global_neg_arr = np.asarray([np.nan if v is None else v for v in global_neg], dtype=float)
        pf_pos_arr = np.asarray([np.nan if v is None else v for v in pf_pos], dtype=float)
        pf_neg_arr = np.asarray([np.nan if v is None else v for v in pf_neg], dtype=float)
        fo_pos_arr = np.asarray([np.nan if v is None else v for v in fo_pos], dtype=float)
        fo_neg_arr = np.asarray([np.nan if v is None else v for v in fo_neg], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            attenuation_neg = global_neg_arr / pf_neg_arr
            attenuation_pos = global_pos_arr / pf_pos_arr
            global_asymmetry = global_neg_arr / global_pos_arr
            matlab_amp_neg = pf_neg_arr / fo_neg_arr
            matlab_amp_pos = pf_pos_arr / fo_pos_arr
            matlab_pf_asymmetry = pf_neg_arr / pf_pos_arr

        rows.append(
            {
                "shock_size": shock_size,
                "values": {
                    "Global optimal attenuation, negative shocks": _nanmean_or_none(attenuation_neg),
                    "Global optimal attenuation, positive shocks": _nanmean_or_none(attenuation_pos),
                    "Global-solution asymmetry": _nanmean_or_none(global_asymmetry),
                    "MATLAB nonlinear amplification, negative shocks": _nanmean_or_none(matlab_amp_neg),
                    "MATLAB nonlinear amplification, positive shocks": _nanmean_or_none(matlab_amp_pos),
                    "MATLAB PF asymmetry": _nanmean_or_none(matlab_pf_asymmetry),
                    "corr(global negative attenuation, IO upstreamness)": _safe_corr(
                        attenuation_neg, upstreamness_data.get("U_M")
                    ),
                    "corr(global asymmetry, IO upstreamness)": _safe_corr(
                        global_asymmetry, upstreamness_data.get("U_M")
                    ),
                    "corr(global negative attenuation, investment upstreamness)": _safe_corr(
                        attenuation_neg, upstreamness_data.get("U_I")
                    ),
                    "corr(global asymmetry, investment upstreamness)": _safe_corr(
                        global_asymmetry, upstreamness_data.get("U_I")
                    ),
                    "corr(global negative attenuation, sectoral shock volatility)": _safe_corr(
                        attenuation_neg, upstreamness_data.get("shock_volatility")
                    ),
                    "corr(global asymmetry, sectoral shock volatility)": _safe_corr(
                        global_asymmetry, upstreamness_data.get("shock_volatility")
                    ),
                },
            }
        )
    return rows


def _write_cir_analysis_table(*, rows, save_path, analysis_name, response_source):
    if not rows:
        return
    measure_order = list(rows[0]["values"].keys())
    with open(save_path, "w") as table_file:
        table_file.write("\\begin{table}[htbp]\n\\centering\n")
        table_file.write("\\caption{Cumulative impulse-response analysis}\n")
        table_file.write(f"\\label{{tab:cir_analysis_{_latex_label_token(analysis_name)}}}\n")
        table_file.write("\\begin{tabular}{l" + "r" * len(rows) + "}\n\\hline\n")
        headers = ["Measure"] + [f"{row['shock_size']:g}\\%" for row in rows]
        table_file.write(" & ".join(_latex_escape(header) for header in headers) + " \\\\\n\\hline\n")
        for measure in measure_order:
            values = [_format_table_value(row["values"].get(measure)) for row in rows]
            table_file.write(_latex_escape(measure) + " & " + " & ".join(values) + " \\\\\n")
        table_file.write("\\hline\n\\end{tabular}\n")
        table_file.write(
            "\\begin{minipage}{0.92\\textwidth}\n\\footnotesize\n"
            "\\textit{Notes:} CIR is the sum over the displayed IR horizon of the aggregate consumption response. "
            f"The Python global-solution CIR uses the selected IR source: {_latex_escape(response_source)}. "
            "MATLAB nonlinear amplification is perfect-foresight CIR divided by first-order CIR. "
            "Global optimal attenuation is global-solution CIR divided by perfect-foresight CIR. "
            "Asymmetry ratios divide the negative-shock CIR by the positive-shock CIR. "
            "Missing MATLAB CIR or diagnostic fields are left blank.\n"
            "\\end{minipage}\n"
        )
        table_file.write("\\end{table}\n")


def render_cir_analysis_outputs(*, config, irs_dir, econ_model, gir_data, postprocess_context):
    ir_render_context = postprocess_context.get("ir_render_context") if postprocess_context else None
    if not ir_render_context:
        return
    rows = _build_cir_analysis_table(
        config=config,
        gir_data=gir_data,
        matlab_ir_data=ir_render_context["matlab_ir_data"],
        upstreamness_data=postprocess_context.get("upstreamness_data", {}),
        n_sectors=econ_model.n_sectors,
    )
    if not rows:
        print("  CIR analysis skipped: no compatible global or MATLAB IR objects found.", flush=True)
        return
    output_path = os.path.join(irs_dir, f"cir_analysis_{config['analysis_name']}.tex")
    _write_cir_analysis_table(
        rows=rows,
        save_path=output_path,
        analysis_name=config["analysis_name"],
        response_source=_resolve_ir_response_source(config),
    )
    print(f"  Saved CIR analysis table: {os.path.basename(output_path)}", flush=True)


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
    n_sectors = econ_model.n_sectors
    analysis_variables_data = dict(analysis_variables_data)
    theoretical_stats: Dict[str, Any] = {}
    use_ergodic_prices = bool(config.get("ergodic_price_aggregation", False))

    reference_experiment_label = _resolve_reference_experiment_label(config, raw_simulation_data)
    reference_sim_data = raw_simulation_data[reference_experiment_label]
    reference_analysis_variables = analysis_variables_data.get(reference_experiment_label)
    if reference_analysis_variables is None:
        reference_analysis_variables = reference_sim_data.get("simul_analysis_variables")
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
        loglinear_theoretical_stats = create_theoretical_descriptive_stats(theo_stats, label="Log-Linear")
        if loglinear_theoretical_stats.get("Log-Linear"):
            theoretical_stats.update(loglinear_theoretical_stats)
        else:
            print("  TheoStats not usable for 1st Order Approx.; falling back to simulation moments.", flush=True)

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

    if dynare_simul_pf is not None:
        print("  Perfect Foresight moments will use Perfect Foresight (Dynare) simulation series.")
    elif isinstance(stats.get("PerfectForesight") or stats.get("Determ"), dict):
        print(
            "  Perfect Foresight benchmark moments are unavailable because "
            "ModelData_simulation has no PerfectForesight block.",
            flush=True,
        )

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

    upstreamness_data = _extract_matlab_upstreamness(model_data, econ_model.upstreamness())
    ergodic_experiment_labels = [
        label for label, sim_data in raw_simulation_data.items() if sim_data.get("simulation_kind", "ergodic") == "ergodic"
    ]
    ergodic_labels_with_stochss = [label for label in ergodic_experiment_labels if label in stochastic_ss_policies]
    if ergodic_labels_with_stochss:
        ergodic_experiment_labels = ergodic_labels_with_stochss
    elif reference_experiment_label in stochastic_ss_policies:
        ergodic_experiment_labels = [reference_experiment_label]
    elif reference_experiment_label.endswith(" (ergodic)"):
        base_reference_label = reference_experiment_label[: -len(" (ergodic)")]
        if base_reference_label in stochastic_ss_policies:
            ergodic_experiment_labels = [base_reference_label]
    elif not ergodic_experiment_labels and reference_experiment_label in raw_simulation_data:
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
        reference_sim_data=reference_sim_data,
        reference_experiment_label=reference_experiment_label,
        matlab_common_shock_schedule=matlab_common_shock_schedule,
    )
    if isinstance(theo_stats, dict):
        loglinear_hist_params = get_loglinear_distribution_params(theo_stats)
        if loglinear_hist_params:
            aggregate_histogram_context["theoretical_distribution_params"] = {
                "1st Order Approximation": loglinear_hist_params
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
            "reference_analysis_variables": reference_analysis_variables,
            "aggregate_histogram_context": aggregate_histogram_context,
        },
    }


def _build_aggregate_histogram_context(
    *,
    config,
    simulation_dir,
    raw_simulation_data,
    reference_sim_data,
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
        "common_shock_burn_in": None,
        "common_shock_active_periods": None,
        "common_shock_burn_out": None,
        "common_shock_total_periods": None,
        "uses_auxiliary_ergodic_reference": False,
    }

    schedule = matlab_common_shock_schedule or {}
    active_shocks = schedule.get("active_shocks")
    full_shocks = schedule.get("full_shocks")
    context.update(
        {
            "common_shock_burn_in": int(schedule.get("burn_in", 0)),
            "common_shock_active_periods": int(active_shocks.shape[0]) if active_shocks is not None else None,
            "common_shock_burn_out": int(schedule.get("burn_out", 0)),
            "common_shock_total_periods": int(full_shocks.shape[0]) if full_shocks is not None else None,
        }
    )

    reference_sim_data = reference_sim_data or raw_simulation_data.get(reference_experiment_label, {})
    if reference_sim_data.get("simulation_kind") == "ergodic":
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
                "uses_auxiliary_ergodic_reference": not bool(config.get("long_simulation", False)),
            }
        )
        return context

    active_periods = int(active_shocks.shape[0]) if active_shocks is not None else 0
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


def _format_histogram_count(value):
    if value is None:
        return None
    return f"{int(value):,}"


def _common_shock_window_text(histogram_context):
    burn_in = _format_histogram_count(histogram_context.get("common_shock_burn_in"))
    active_periods = _format_histogram_count(histogram_context.get("common_shock_active_periods"))
    burn_out = _format_histogram_count(histogram_context.get("common_shock_burn_out"))
    total_periods = _format_histogram_count(histogram_context.get("common_shock_total_periods"))
    if burn_in and active_periods and burn_out and total_periods:
        return (
            "the shorter common-shock window "
            f"({burn_in} burn-in, {active_periods} active, {burn_out} burn-out; {total_periods} total)"
        )
    return "the shorter common-shock window"


def _build_aggregate_histogram_note(histogram_context):
    comparison_labels = histogram_context.get("benchmark_labels", [])
    if not comparison_labels:
        comparison_text = "the comparison methods"
    elif len(comparison_labels) == 1:
        comparison_text = comparison_labels[0]
    elif len(comparison_labels) == 2:
        comparison_text = f"{comparison_labels[0]} and {comparison_labels[1]}"
    else:
        comparison_text = ", ".join(comparison_labels[:-1]) + f", and {comparison_labels[-1]}"
    base_text = (
        "The reported histograms summarize the distribution of the simulated series for each displayed aggregate. "
        "Each panel plots percent log deviations from the deterministic steady state. "
        f"The solid colored line reports the global solution; dashed gray lines report {comparison_text}. "
    )
    if histogram_context.get("theoretical_distribution_params"):
        comparison_source_text = (
            "For the 1st Order Approximation, the dashed curve is the Gaussian density implied by its theoretical moments rather than a simulated histogram. "
            f"The remaining comparison distributions use {_common_shock_window_text(histogram_context)}."
        )
    else:
        comparison_source_text = (
            f"The comparison distributions use {_common_shock_window_text(histogram_context)}."
        )
    if histogram_context.get("mode") == "long_ergodic":
        periods_per_episode = _format_histogram_count(histogram_context.get("periods_per_episode"))
        burn_in = _format_histogram_count(histogram_context.get("burn_in"))
        kept_periods_per_seed = _format_histogram_count(histogram_context.get("kept_periods_per_seed"))
        active_observations = _format_histogram_count(histogram_context.get("active_observations"))
        n_simul_seeds = _format_histogram_count(histogram_context.get("n_simul_seeds"))
        reference_prefix = ""
        if histogram_context.get("uses_auxiliary_ergodic_reference"):
            reference_prefix = (
                "Because fixed-price aggregation is anchored to ergodic prices, "
                "the global-solution histogram uses the auxiliary long ergodic reference sample rather "
                "than the shorter common-shock window. "
            )
        if periods_per_episode and burn_in and kept_periods_per_seed and active_observations and n_simul_seeds:
            nonlinear_text = (
                "The global-solution histogram uses the long simulation, pooling "
                f"{n_simul_seeds} simulated paths with different seeds. Each path has {periods_per_episode} periods; "
                f"after dropping {burn_in} burn-in periods, {kept_periods_per_seed} observations per seed are retained "
                f"({active_observations} total). "
            )
        else:
            nonlinear_text = (
                "The global-solution histogram uses the retained long simulation sample. "
            )
        return (
            base_text
            + reference_prefix
            + nonlinear_text
            + comparison_source_text
        )
    return (
        base_text
        + "The global-solution histogram is computed from the shorter common-shock sample, with "
        f"{histogram_context.get('burn_in', 0)} burn-in periods, "
        f"{histogram_context.get('active_periods', 0)} active-shock periods, and "
        f"{histogram_context.get('burn_out', 0)} burn-out periods "
        f"({histogram_context.get('total_periods', 0)} total). "
        f"The reported nonlinear sample contains {histogram_context.get('active_periods', 0)} observations. "
        + comparison_source_text
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
            print(f"    Plotting aggregate variable: {ir_variable} [full panel]")
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
            print(f"    Plotting aggregate variable: {ir_variable} [largest negative shock]")
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
                filename_suffix="largest_negative",
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
    reference_analysis_variables = postprocess_context.get("reference_analysis_variables")
    if not reference_experiment_label:
        return
    histogram_context = postprocess_context.get("aggregate_histogram_context") or {}
    histogram_theoretical_params = histogram_context.get("theoretical_distribution_params")

    selected_methods = [("__reference__", "Global solution"), *AGGREGATE_HISTOGRAM_BENCHMARKS]
    ordered_histogram_data = {}
    missing_methods = []

    for source_label, display_label in selected_methods:
        if source_label == "__reference__":
            series = analysis_variables_data.get(reference_experiment_label, reference_analysis_variables)
        else:
            series = analysis_variables_data.get(source_label)
        if series is None:
            if histogram_theoretical_params and display_label in histogram_theoretical_params:
                ordered_histogram_data[display_label] = {}
                continue
            missing_methods.append(reference_experiment_label if source_label == "__reference__" else source_label)
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
    del stats
    method_stats = {}

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
    if common_shock_sim_data is None and reference_experiment_label.endswith(" (ergodic)"):
        base_label = reference_experiment_label[: -len(" (ergodic)")]
        fallback_common_shock = raw_simulation_data.get(base_label)
        if fallback_common_shock is not None and fallback_common_shock.get("simulation_kind") == "common_shock":
            common_shock_sim_data = fallback_common_shock
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
