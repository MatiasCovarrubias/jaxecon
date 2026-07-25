import csv
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
    process_simulation_with_consistent_aggregation,
    reaggregate_aggregates,
)
from DEQN.econ_models.RbcProdNet_April2026.matlab_irs import get_available_shock_sizes, load_matlab_irs
from DEQN.econ_models.RbcProdNet_April2026.plot_helpers import (
    plot_cir_shock_size_profiles,
    plot_ergodic_histograms,
    plot_sectoral_diagnostic_bar,
    plot_upstreamness,
    _sectoral_levels_from_logdev,
    _sectoral_share_change,
    _sectoral_share_weights,
    _sectoral_variable_info,
    _single_experiment_name,
    _write_figure_note_tex,
)
from DEQN.econ_models.RbcProdNet_April2026.plots import (
    plot_sector_ir_by_shock_size,
    plot_sectoral_variable_composition_ergodic,
    plot_sectoral_variable_composition_stochss,
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
    if x.shape != y.shape:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _is_cir_correlation_measure(measure: str) -> bool:
    return measure.startswith("corr(")


def _cir_display_value(value: Any, measure: str) -> Optional[float]:
    scalar = _as_float(value)
    if scalar is None:
        return None
    return scalar if _is_cir_correlation_measure(measure) else scalar * 100.0


def _format_table_value(value: Any, measure: str) -> str:
    scalar = _cir_display_value(value, measure)
    if scalar is None:
        return ""
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


# Mean CIR rows: levels first, then correlations grouped by covariate.
_CIR_MEAN_MEASURES = [
    {
        "key": "Nonlin. ampl. (-)",
        "label": "Nonlin. ampl. (-)",
        "formula": r"$100\times(GIR_{MIT}(-)/GIR_{1or}(-)-1)$",
        "note": "100 times the negative-shock MIT CIR divided by the negative-shock first-order CIR, minus 100",
        "panel": "A. MIT shock vs 1st-order approx.",
        "figure_title": "Nonlinear amplification, negative shocks",
        "filename_stem": "cir_nonlin_ampl_negative",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
    {
        "key": "Nonlin. ampl. (+)",
        "label": "Nonlin. ampl. (+)",
        "formula": r"$100\times(GIR_{MIT}(+)/GIR_{1or}(+)-1)$",
        "note": "100 times the positive-shock MIT CIR divided by the positive-shock first-order CIR, minus 100",
        "panel": "A. MIT shock vs 1st-order approx.",
        "figure_title": "Nonlinear amplification, positive shocks",
        "filename_stem": "cir_nonlin_ampl_positive",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
    {
        "key": "MIT asym.",
        "label": "MIT asym.",
        "formula": r"$100\times(1-|GIR_{MIT}(-)|/|GIR_{MIT}(+)|)$",
        "note": (
            "100 times one minus the absolute negative-shock MIT CIR divided by the absolute positive-shock MIT CIR"
        ),
        "panel": "A. MIT shock vs 1st-order approx.",
        "figure_title": "MIT shock asymmetry, negative vs positive shocks",
        "filename_stem": "cir_mit_asymmetry",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
    {
        "key": "Opt. atten. (-)",
        "label": "Attenuation (-)",
        "formula": r"$100\times(1-GIR_{GS}(-)/GIR_{MIT}(-))$",
        "note": "100 times one minus the negative-shock global-solution CIR divided by the negative-shock MIT CIR",
        "panel": "B. Global solution vs MIT shock",
        "figure_title": "Optimal attenuation, negative shocks",
        "filename_stem": "cir_opt_atten_negative",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
    {
        "key": "Opt. atten. (+)",
        "label": "Attenuation (+)",
        "formula": r"$100\times(1-GIR_{GS}(+)/GIR_{MIT}(+))$",
        "note": "100 times one minus the positive-shock global-solution CIR divided by the positive-shock MIT CIR",
        "panel": "B. Global solution vs MIT shock",
        "figure_title": "Optimal attenuation, positive shocks",
        "filename_stem": "cir_opt_atten_positive",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
    {
        "key": "Global asym.",
        "label": "Global asym.",
        "formula": r"$100\times(1-|GIR_{GS}(-)/GIR_{GS}(+)|)$",
        "note": (
            "100 times one minus the absolute negative-shock global-solution CIR divided by the absolute "
            "positive-shock global-solution CIR"
        ),
        "panel": "B. Global solution vs MIT shock",
        "figure_title": "Global asymmetry, negative vs positive shocks",
        "filename_stem": "cir_global_asymmetry",
        "ylabel": "Percent difference",
        "scale": 100.0,
    },
]

# Within each covariate panel, keep this outcome order.
_CIR_CORR_OUTCOMES = [
    {
        "outcome_key": "Nonlin. ampl. (-)",
        "short": "nonlin. ampl. (-)",
        "label_stem": "Nonlin. ampl. (-)",
        "note_stem": "negative-shock nonlinear amplification",
        "filename_stem": "nonlin_ampl_neg",
        "title_stem": "nonlinear amplification for negative shocks",
    },
    {
        "outcome_key": "Nonlin. ampl. (+)",
        "short": "nonlin. ampl. (+)",
        "label_stem": "Nonlin. ampl. (+)",
        "note_stem": "positive-shock nonlinear amplification",
        "filename_stem": "nonlin_ampl_pos",
        "title_stem": "nonlinear amplification for positive shocks",
    },
    {
        "outcome_key": "Opt. atten. (-)",
        "short": "opt. atten. (-)",
        "label_stem": "Attenuation (-)",
        "note_stem": "negative-shock optimal attenuation",
        "filename_stem": "opt_atten_neg",
        "title_stem": "optimal attenuation for negative shocks",
    },
    {
        "outcome_key": "Opt. atten. (+)",
        "short": "opt. atten. (+)",
        "label_stem": "Attenuation (+)",
        "note_stem": "positive-shock optimal attenuation",
        "filename_stem": "opt_atten_pos",
        "title_stem": "optimal attenuation for positive shocks",
    },
    {
        "outcome_key": "MIT asym.",
        "short": "MIT asym.",
        "label_stem": "MIT asym.",
        "note_stem": "MIT-shock asymmetry",
        "filename_stem": "mit_asym",
        "title_stem": "MIT shock asymmetry",
    },
    {
        "outcome_key": "Global asym.",
        "short": "global asym.",
        "label_stem": "Global asym.",
        "note_stem": "global asymmetry",
        "filename_stem": "global_asym",
        "title_stem": "global asymmetry",
    },
]

_CIR_CORR_COVARIATES = [
    {
        "key": "U_M",
        "symbol": "U_M",
        "latex_symbol": r"U\_M",
        "name": "IO upstreamness",
        "panel": "C. Correlations with IO upstreamness (U_M)",
    },
    {
        "key": "U_I",
        "symbol": "U_I",
        "latex_symbol": r"U\_I",
        "name": "investment upstreamness",
        "panel": "D. Correlations with investment upstreamness (U_I)",
    },
    {
        "key": "sigA",
        "symbol": "sigA",
        "latex_symbol": "sigA",
        "name": "sectoral TFP shock volatility",
        "panel": "E. Correlations with shock volatility (sigA)",
    },
    {
        "key": "rho",
        "symbol": "rho",
        "latex_symbol": r"$\rho$",
        "name": "sectoral TFP persistence",
        "panel": "F. Correlations with shock persistence (rho)",
    },
]


def _cir_corr_measure_key(outcome_short: str, covariate_symbol: str) -> str:
    return f"corr({outcome_short}, {covariate_symbol})"


def _build_cir_correlation_measures() -> list[dict[str, Any]]:
    measures = []
    for covariate in _CIR_CORR_COVARIATES:
        for outcome in _CIR_CORR_OUTCOMES:
            measure_key = _cir_corr_measure_key(outcome["short"], covariate["symbol"])
            measures.append(
                {
                    "key": measure_key,
                    "label": f"corr({outcome['label_stem']}, {covariate['symbol']})",
                    "formula": f"corr({outcome['label_stem']}, {covariate['latex_symbol']})",
                    "note": f"the correlation between {outcome['note_stem']} and {covariate['name']}",
                    "panel": covariate["panel"],
                    "figure_title": f"Correlation: {outcome['title_stem']} and {covariate['symbol']}",
                    "filename_stem": f"cir_corr_{outcome['filename_stem']}_{covariate['key']}",
                    "ylabel": "Correlation",
                    "scale": 1.0,
                    "outcome_key": outcome["outcome_key"],
                    "covariate_key": covariate["key"],
                }
            )
    return measures


_CIR_CORR_MEASURES = _build_cir_correlation_measures()
_CIR_ALL_MEASURES = _CIR_MEAN_MEASURES + _CIR_CORR_MEASURES

CIR_MEASURE_DESCRIPTIONS = {measure["key"]: measure["formula"] for measure in _CIR_ALL_MEASURES}
CIR_TABLE_MEASURE_LABELS = {measure["key"]: measure["label"] for measure in _CIR_ALL_MEASURES}
CIR_MEASURE_NOTE_DESCRIPTIONS = {measure["key"]: measure["note"] for measure in _CIR_ALL_MEASURES}

_CIR_PANEL_ORDER = [
    "A. MIT shock vs 1st-order approx.",
    "B. Global solution vs MIT shock",
    "C. Correlations with IO upstreamness (U_M)",
    "D. Correlations with investment upstreamness (U_I)",
    "E. Correlations with shock volatility (sigA)",
    "F. Correlations with shock persistence (rho)",
]

CIR_TABLE_PANELS = [
    (panel_title, [measure["key"] for measure in _CIR_ALL_MEASURES if measure["panel"] == panel_title])
    for panel_title in _CIR_PANEL_ORDER
]

CIR_SECTOR_VALUE_MEASURES = [measure["key"] for measure in _CIR_MEAN_MEASURES]

CIR_FIGURE_SPECS = [
    {
        "title": measure["figure_title"],
        "measures": [measure["key"]],
        "filename_stem": measure["filename_stem"],
        "ylabel": measure["ylabel"],
        "scale": measure["scale"],
    }
    for measure in _CIR_ALL_MEASURES
]


def _magnitude_ratio(numerator: Any, denominator: Any) -> np.ndarray:
    return np.abs(np.asarray(numerator, dtype=float)) / np.abs(np.asarray(denominator, dtype=float))


def _signed_ratio(numerator: Any, denominator: Any) -> np.ndarray:
    return np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)


def _sectoral_shock_volatility(value: Any, n_sectors: Optional[int] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).squeeze()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        if arr.shape[0] == arr.shape[1]:
            arr = np.sqrt(np.maximum(np.diag(arr), 0.0))
        elif 1 in arr.shape:
            arr = arr.ravel()
        else:
            return None
    else:
        arr = arr.ravel()
    if n_sectors is not None and arr.size == n_sectors * n_sectors:
        arr = np.sqrt(np.maximum(np.diag(arr.reshape(n_sectors, n_sectors)), 0.0))
    if n_sectors is not None and arr.size != n_sectors:
        return None
    return arr if np.isfinite(arr).any() else None


def _sectoral_shock_persistence(value: Any, n_sectors: Optional[int] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).squeeze()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    arr = arr.ravel()
    if n_sectors is not None and arr.size == 1:
        arr = np.full(n_sectors, float(arr[0]), dtype=float)
    if n_sectors is not None and arr.size != n_sectors:
        return None
    return arr if np.isfinite(arr).any() else None


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


def _coerce_struct(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _coerce_struct(value) for key, value in obj.items()}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: _coerce_struct(obj[name]) for name in obj.dtype.names}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return [_coerce_struct(value) for value in obj.ravel()]
        if obj.dtype == object:
            return [_coerce_struct(value) for value in obj.ravel()]
        return obj
    field_names = getattr(obj, "_fieldnames", None)
    if field_names:
        return {name: _coerce_struct(getattr(obj, name)) for name in field_names}
    return obj


def _as_dict(value: Any) -> Dict[str, Any]:
    coerced = _coerce_struct(value)
    return coerced if isinstance(coerced, dict) else {}


def _as_dict_list(value: Any) -> list[Dict[str, Any]]:
    coerced = _coerce_struct(value)
    if isinstance(coerced, dict):
        return [coerced]
    if isinstance(coerced, list):
        return [item for item in coerced if isinstance(item, dict)]
    return []

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
    aggregate_irs_dir = os.path.join(irs_dir, "IR_aggregate")
    sectoral_irs_dir = os.path.join(irs_dir, "IR_sectoral")
    ir_tables_dir = os.path.join(irs_dir, "IR_tables")
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
        "1. Untargeted Model vs. Data Moments",
        [os.path.join(analysis_dir, f"calibration_table_{analysis_name}.tex")],
    )
    add_table_section(
        "2. Targeted Moments",
        [os.path.join(analysis_dir, f"targeted_moments_{analysis_name}.tex")],
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
    ir_max_periods = int(config.get("ir_max_periods", 80))
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
            aggregate_irs_dir,
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
            f"The panels show aggregate {join_labels(displayed_labels)} responses to a negative "
            f"{format_percent_list([largest_ir_shock])} percent TFP shock in {sector_label}. "
        )
        return (
            f"{shock_text}"
            f"{deqn_ir_note}"
            f"The horizontal axis reports periods 0 through {ir_max_periods - 1} after impact. "
            "The vertical axis reports 100 times the log deviation from the deterministic steady state. "
            "Dashed and dash-dotted lines report comparison IRs under the "
            f"{aggregate_benchmark_labels}; these comparison IRs are anchored at the deterministic "
            "steady state."
        )
    for sector_idx in config.get("ir_sectors_to_plot", []):
        if not ir_shock_sizes:
            continue
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
            )
            aggregate_ir_figures.append(
                build_simple_figure_spec(
                    analysis_named_path(aggregate_irs_dir, f"IR_{safe_variable}_{safe_sector}", analysis_name, ".png"),
                    f"Aggregate {caption_label(figure_caption)} response to a TFP shock in {sector_label}.",
                    note_text=(
                        f"{shock_layout_text}"
                        f"The figure plots the response of aggregate {note_label}. "
                        f"{deqn_ir_note}"
                        f"The horizontal axis reports periods 0 through {ir_max_periods - 1} after impact. "
                        "The vertical axis reports 100 times the log deviation from the deterministic steady state. "
                        "Dashed and dash-dotted lines report comparison IRs under the "
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
        "2D. CIR Analysis",
        [os.path.join(ir_tables_dir, f"cir_analysis_{analysis_name}.tex")],
    )
    add_table_section(
        "2D2. CIR Cross-Sector Regressions",
        [os.path.join(ir_tables_dir, f"cir_regressions_{analysis_name}.tex")],
    )
    cir_figures_dir = os.path.join(irs_dir, "IR_CIR")
    add_figure_section(
        "2E. CIR Shock-Size Profiles",
        [
            build_simple_figure_spec(
                analysis_named_path(cir_figures_dir, figure_spec["filename_stem"], analysis_name, ".png"),
                f"{figure_spec['title']} by shock size.",
                note_text=_build_cir_profile_note(
                    figure_spec,
                    n_sectors=econ_model.n_sectors,
                    rows=[],
                ),
            )
            for figure_spec in CIR_FIGURE_SPECS
        ],
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

    add_figure_section(
        "3A. Sectoral Composition in Stochastic Steady State",
        [
            build_simple_figure_spec(
                analysis_named_path(
                    simulation_dir,
                    f"sectoral_{variable_name}_composition_stochss",
                    analysis_name,
                    ".png",
                ),
                f"Sectoral {variable_caption} deterministic-price share change at the stochastic steady state.",
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

    add_figure_section(
        "3B. Sectoral Upstreamness",
        [
            build_simple_figure_spec(
                analysis_named_path(simulation_dir, "sectoral_upstreamness", analysis_name, ".png"),
                "Sector-level upstreamness measures.",
                note_text=(
                    "Sectors are sorted by intermediate-input upstreamness. U_M solves (I - Delta_M) U_M = 1 using "
                    "steady-state intermediate-input expenditure shares, and U_I solves (I - Delta_I) U_I = 1 using "
                    "steady-state investment-flow expenditure shares. Mout/Q is steady-state intermediate sales "
                    "divided by gross output."
                ),
            )
        ],
    )

    add_figure_section(
        "3C. Sectoral TFP Shock Diagnostics",
        [
            build_simple_figure_spec(
                analysis_named_path(simulation_dir, "sectoral_shock_volatility", analysis_name, ".png"),
                "Sector-level TFP shock volatility.",
                note_text=(
                    "Each bar reports a sector's TFP innovation standard deviation. For a covariance-matrix input, "
                    "the value is the square root of the corresponding diagonal element."
                ),
            ),
            build_simple_figure_spec(
                analysis_named_path(simulation_dir, "sectoral_tfp_persistence", analysis_name, ".png"),
                "Sector-level TFP shock persistence.",
                note_text=(
                    "Each bar reports the sectoral AR(1) persistence coefficient rho_j. A common scalar rho produces "
                    "the same bar value for every sector."
                ),
            ),
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
                            sectoral_irs_dir,
                            f"IR_{safe_variable}_{safe_sector}",
                            analysis_name,
                            ".png",
                        ),
                        "caption": variable_caption,
                        "variable_name": variable_name,
                    }
                )

            if subfigures and largest_sectoral_shock is not None:
                shock_text = (
                    f"The panels show responses to a negative "
                    f"{format_percent_list([largest_sectoral_shock])} percent TFP shock in "
                    f"{sector_label}."
                )
                vertical_axis_text = (
                    "The vertical axis reports 100 times the log deviation from the deterministic steady state; "
                    "the expenditure-share panel reports percentage-point deviations. "
                    if any(subfigure["variable_name"] == "gammaij_client" for subfigure in subfigures)
                    else "The vertical axis reports 100 times the log deviation from the deterministic steady state. "
                )
                sectoral_ir_groups.append(
                    {
                        "caption": f"{group_title} for {sector_label}.",
                        "note_text": (
                            f"{shock_text} "
                            f"{deqn_ir_note}"
                            f"The horizontal axis reports periods 0 through {ir_max_periods - 1} after impact. "
                            f"{vertical_axis_text}"
                            "Dashed and dash-dotted lines report comparison IRs under the "
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

    add_figure_section(
        "9A. Ergodic Mean Sectoral Composition",
        [
            build_simple_figure_spec(
                analysis_named_path(
                    simulation_dir,
                    f"sectoral_{variable_name}_composition_ergodic",
                    analysis_name,
                    ".png",
                ),
                f"Ergodic mean sectoral {variable_caption} deterministic-price share change.",
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
            raise ValueError("Could not infer IR shock sizes from benchmark IR objects.")
    else:
        print(f"  Using IR shock sizes discovered from benchmark IR objects: {shock_sizes}")
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


def _build_upstreamness_data(model_data, model_upstreamness, *, n_sectors: Optional[int] = None):
    model_data = _as_dict(model_data)
    result = {}

    for key, value in (model_upstreamness or {}).items():
        if key in {"U_M", "U_I", "U_simple"}:
            result[key] = np.asarray(value, dtype=float).ravel()
        else:
            result[key] = value

    steady_state = _as_dict(model_data.get("SteadyState") or model_data.get("steadystate"))
    parameters = _as_dict(steady_state.get("parameters") or model_data.get("parameters"))
    shock_volatility = _sectoral_shock_volatility(parameters.get("parSigma_A"), n_sectors)
    if shock_volatility is not None:
        result["shock_volatility"] = shock_volatility
    shock_persistence = _sectoral_shock_persistence(parameters.get("parrho"), n_sectors)
    if shock_persistence is not None:
        result["shock_persistence"] = shock_persistence

    return result


def _extract_matlab_irf_breakdown_rows(model_data) -> list[Dict[str, Any]]:
    model_data = _as_dict(model_data)
    diagnostics = _as_dict(model_data.get("Diagnostics") or model_data.get("diagnostics"))
    breakdown = _as_dict(diagnostics.get("irf_sector_breakdown"))
    return _as_dict_list(breakdown.get("rows"))


def _find_matlab_breakdown_row(rows: list[Dict[str, Any]], shock_size: float) -> Dict[str, Any]:
    for row in rows:
        size = _as_float(row.get("size_pct"))
        if size is not None and abs(size - float(shock_size)) <= 1e-6:
            return row
    return {}


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


def _build_cir_analysis_table(*, config, gir_data, matlab_ir_data, upstreamness_data, matlab_breakdown_rows, n_sectors):
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
        matlab_breakdown_row = _find_matlab_breakdown_row(matlab_breakdown_rows, shock_size)
        shock_volatility = _sectoral_shock_volatility(upstreamness_data.get("shock_volatility"), n_sectors)
        if shock_volatility is None and matlab_breakdown_row.get("shock_volatility") is not None:
            shock_volatility = _sectoral_shock_volatility(matlab_breakdown_row["shock_volatility"], n_sectors)
        shock_persistence = _sectoral_shock_persistence(upstreamness_data.get("shock_persistence"), n_sectors)
        if shock_persistence is None and matlab_breakdown_row.get("shock_persistence") is not None:
            shock_persistence = _sectoral_shock_persistence(matlab_breakdown_row["shock_persistence"], n_sectors)
        covariate_vectors = {
            "U_M": upstreamness_data.get("U_M"),
            "U_I": upstreamness_data.get("U_I"),
            "sigA": shock_volatility,
            "rho": shock_persistence,
        }
        pos_key = build_shock_key("pos", shock_size)
        neg_key = build_shock_key("neg", shock_size)
        global_pos = []
        global_neg = []
        pf_pos = []
        pf_neg = []
        fo_neg = []
        fo_pos = []
        horizon_periods = []
        for sector_idx in range(n_sectors):
            pos_horizon = (
                _get_matlab_cir_horizon_for_sector(matlab_ir_data, shock_key=pos_key, sector_idx=sector_idx)
                or max_periods
            )
            neg_horizon = (
                _get_matlab_cir_horizon_for_sector(matlab_ir_data, shock_key=neg_key, sector_idx=sector_idx)
                or max_periods
            )
            horizon_periods.extend([pos_horizon, neg_horizon])
            g_pos = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=pos_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=pos_horizon,
                response_source=response_source,
            )
            g_neg = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=neg_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=neg_horizon,
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
            outcome_vectors = {
                "Opt. atten. (-)": 1.0 - _signed_ratio(global_neg_arr, pf_neg_arr),
                "Opt. atten. (+)": 1.0 - _signed_ratio(global_pos_arr, pf_pos_arr),
                "Global asym.": 1.0 - _magnitude_ratio(global_neg_arr, global_pos_arr),
                "Nonlin. ampl. (-)": _signed_ratio(pf_neg_arr, fo_neg_arr) - 1.0,
                "Nonlin. ampl. (+)": _signed_ratio(pf_pos_arr, fo_pos_arr) - 1.0,
                "MIT asym.": 1.0 - _magnitude_ratio(pf_neg_arr, pf_pos_arr),
            }

        values = {
            measure_key: _nanmean_or_none(outcome_vector)
            for measure_key, outcome_vector in outcome_vectors.items()
        }
        for corr_measure in _CIR_CORR_MEASURES:
            values[corr_measure["key"]] = _safe_corr(
                outcome_vectors.get(corr_measure["outcome_key"]),
                covariate_vectors.get(corr_measure["covariate_key"]),
            )

        rows.append(
            {
                "shock_size": shock_size,
                "horizon_periods": sorted(set(horizon_periods)),
                "values": values,
            }
        )
    return rows


def _cir_panel_title(measure: str) -> str:
    for panel_title, measures in CIR_TABLE_PANELS:
        if measure in measures:
            return panel_title
    return "Other"


def _format_cir_horizon_note(rows) -> str:
    horizon_parts = []
    for row in rows:
        horizons = sorted({int(value) for value in row.get("horizon_periods", []) if int(value) > 0})
        if not horizons:
            continue
        shock_label = f"{float(row['shock_size']):g} percent shocks"
        if len(horizons) == 1:
            horizon = horizons[0]
            horizon_parts.append(f"{shock_label} sum periods 0 through {horizon - 1} ({horizon} periods)")
        else:
            horizon_list = ", ".join(str(horizon) for horizon in horizons)
            horizon_parts.append(
                f"{shock_label} use {horizon_list} periods across sector-sign pairs, each beginning at period 0"
            )
    if not horizon_parts:
        return ""
    return " CIR horizons are: " + "; ".join(horizon_parts) + "."


def _build_cir_profile_note(figure_spec, *, n_sectors: int, rows) -> str:
    descriptions = [
        CIR_MEASURE_NOTE_DESCRIPTIONS[measure]
        for measure in figure_spec.get("measures", [])
        if measure in CIR_MEASURE_NOTE_DESCRIPTIONS
    ]
    unit_text = (
        "The vertical axis is a unitless correlation."
        if figure_spec.get("ylabel") == "Correlation"
        else "The vertical axis reports the cross-sector mean in percent."
    )
    return (
        f"The horizontal axis reports TFP shock size in percent. Each point reports {'; '.join(descriptions)} "
        f"across {n_sectors} shocked sectors. {unit_text}"
        + _format_cir_horizon_note(rows)
    )


def _nan_if_none(value: Any) -> float:
    scalar = _as_float(value)
    return np.nan if scalar is None else scalar


def _build_cir_sector_value_rows(*, config, gir_data, matlab_ir_data, n_sectors, sector_labels=None):
    shock_sizes = get_available_shock_sizes(matlab_ir_data)
    if not shock_sizes or not gir_data:
        return []
    if len(gir_data) != 1:
        raise ValueError(
            "CIR sector value export expects exactly one nonlinear experiment in gir_data; "
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

        for sector_idx in range(n_sectors):
            pos_horizon = (
                _get_matlab_cir_horizon_for_sector(matlab_ir_data, shock_key=pos_key, sector_idx=sector_idx)
                or max_periods
            )
            neg_horizon = (
                _get_matlab_cir_horizon_for_sector(matlab_ir_data, shock_key=neg_key, sector_idx=sector_idx)
                or max_periods
            )
            g_pos = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=pos_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=pos_horizon,
                response_source=response_source,
            )
            g_neg = _get_global_cir_for_sector(
                gir_data,
                experiment_name=experiment_name,
                sector_idx=sector_idx,
                shock_key=neg_key,
                variable_name=variable_name,
                n_sectors=n_sectors,
                max_periods=neg_horizon,
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
            if all(value is None for value in [g_pos, g_neg, p_pos, p_neg, f_pos, f_neg]):
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                values = {
                    "Opt. atten. (-)": 1.0 - _signed_ratio(_nan_if_none(g_neg), _nan_if_none(p_neg)),
                    "Opt. atten. (+)": 1.0 - _signed_ratio(_nan_if_none(g_pos), _nan_if_none(p_pos)),
                    "Global asym.": 1.0 - _magnitude_ratio(_nan_if_none(g_neg), _nan_if_none(g_pos)),
                    "Nonlin. ampl. (-)": _signed_ratio(_nan_if_none(p_neg), _nan_if_none(f_neg)) - 1.0,
                    "Nonlin. ampl. (+)": _signed_ratio(_nan_if_none(p_pos), _nan_if_none(f_pos)) - 1.0,
                    "MIT asym.": 1.0 - _magnitude_ratio(_nan_if_none(p_neg), _nan_if_none(p_pos)),
                }

            sector_label = (
                sector_labels[sector_idx]
                if sector_labels is not None and sector_idx < len(sector_labels)
                else f"Sector {sector_idx + 1}"
            )
            for measure in CIR_SECTOR_VALUE_MEASURES:
                value = _as_float(values.get(measure))
                rows.append(
                    {
                        "analysis_name": config["analysis_name"],
                        "experiment_name": experiment_name,
                        "response_source": response_source,
                        "sector_idx": sector_idx,
                        "sector_label": sector_label,
                        "shock_size": shock_size,
                        "positive_horizon_periods": pos_horizon,
                        "negative_horizon_periods": neg_horizon,
                        "measure_panel": _cir_panel_title(measure),
                        "measure": measure,
                        "formula": CIR_MEASURE_DESCRIPTIONS.get(measure, ""),
                        "value": value,
                        "value_percent": _cir_display_value(value, measure),
                    }
                )
    return rows


def _format_csv_float(value: Any) -> str:
    scalar = _as_float(value)
    return "" if scalar is None else f"{scalar:.12g}"


def _write_cir_sector_values_csv(*, rows, save_path):
    if not rows:
        return
    fieldnames = [
        "analysis_name",
        "experiment_name",
        "response_source",
        "sector_idx",
        "sector_label",
        "shock_size",
        "positive_horizon_periods",
        "negative_horizon_periods",
        "measure_panel",
        "measure",
        "formula",
        "value",
        "value_percent",
    ]
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["value"] = _format_csv_float(row.get("value"))
            csv_row["value_percent"] = _format_csv_float(row.get("value_percent"))
            writer.writerow({field: csv_row.get(field, "") for field in fieldnames})
    print(f"  Saved CIR sector value artifact: {os.path.basename(save_path)}", flush=True)


def _write_cir_analysis_table(*, rows, save_path, analysis_name, response_source):
    if not rows:
        return
    available_measures = set(rows[0]["values"].keys())
    panel_rows = [
        (panel_title, [measure for measure in measures if measure in available_measures])
        for panel_title, measures in CIR_TABLE_PANELS
    ]
    panel_rows = [(panel_title, measures) for panel_title, measures in panel_rows if measures]
    panel_measure_set = {measure for _, measures in panel_rows for measure in measures}
    ungrouped_measures = [measure for measure in rows[0]["values"].keys() if measure not in panel_measure_set]
    if ungrouped_measures:
        panel_rows.append(("Other", ungrouped_measures))
    column_count = 2 + len(rows)
    with open(save_path, "w") as table_file:
        table_file.write("\\begin{table}[htbp]\n\\centering\n")
        table_file.write("\\caption{Cumulative impulse-response analysis}\n")
        table_file.write(f"\\label{{tab:cir_analysis_{_latex_label_token(analysis_name)}}}\n")
        table_file.write("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
        table_file.write("\\resizebox{\\textwidth}{!}{%\n")
        table_file.write("\\begin{tabular}{ll" + "r" * len(rows) + "}\n\\hline\n")
        headers = ["Metric", "Formula"] + [f"{row['shock_size']:g}%" for row in rows]
        table_file.write(" & ".join(_latex_escape(header) for header in headers) + " \\\\\n\\hline\n")
        for panel_title, measures in panel_rows:
            table_file.write(
                f"\\multicolumn{{{column_count}}}{{l}}{{\\textit{{{_latex_escape(panel_title)}}}}} \\\\\n"
            )
            for measure in measures:
                description = CIR_MEASURE_DESCRIPTIONS.get(measure, "")
                values = [_format_table_value(row["values"].get(measure), measure) for row in rows]
                table_file.write(
                    _latex_escape(CIR_TABLE_MEASURE_LABELS.get(measure, measure))
                    + " & "
                    + description
                    + " & "
                    + " & ".join(values)
                    + " \\\\\n"
                )
        table_file.write("\\hline\n\\end{tabular}\n}\n")
        table_file.write(
            "\\begin{minipage}{0.92\\textwidth}\n\\footnotesize\n"
            "\\textit{Notes:} CIR is the sum over the displayed impulse-response horizon of the aggregate "
            "consumption response. The global-solution CIR uses the selected IR source: "
            + _latex_escape(response_source)
            + ". "
            + r"Notation: $GIR_{GS}$ is the global-solution CIR, $GIR_{MIT}$ is the MIT shock CIR, "
            r"and $GIR_{1or}$ is the 1st-order approximation CIR; + and - denote positive and negative shocks. "
            "Opt. atten., nonlin. ampl., asym., and corr denote optimal attenuation, nonlinear amplification, "
            "asymmetry, and correlation. "
            r"$U_M$ and $U_I$ are IO and investment upstreamness, sigA is sectoral shock volatility, "
            r"and $\rho$ is sectoral TFP persistence. "
            "Panels A--B report cross-sector means; panels C--F report cross-sector correlations for the same "
            "outcome ordering within each covariate. "
            "Amplification, attenuation, and asymmetry rows are reported in percent, while correlation rows are "
            "unitless. Missing benchmark CIR or diagnostic fields are left blank.\n"
            "\\end{minipage}\n"
        )
        table_file.write("\\end{table}\n")


def _print_cir_analysis_table(rows) -> None:
    if not rows:
        return
    shock_headers = [f"{row['shock_size']:g}%" for row in rows]
    available_measures = set(rows[0]["values"].keys())
    panel_rows = [
        (panel_title, [measure for measure in measures if measure in available_measures])
        for panel_title, measures in CIR_TABLE_PANELS
    ]
    panel_rows = [(panel_title, measures) for panel_title, measures in panel_rows if measures]
    panel_measure_set = {measure for _, measures in panel_rows for measure in measures}
    ungrouped_measures = [measure for measure in rows[0]["values"].keys() if measure not in panel_measure_set]
    if ungrouped_measures:
        panel_rows.append(("Other", ungrouped_measures))
    measure_order = [measure for _, measures in panel_rows for measure in measures]
    measure_width = max(16, max(len(measure) for measure in measure_order))
    description_width = max(
        18,
        max(len(CIR_MEASURE_DESCRIPTIONS.get(measure, "")) for measure in measure_order),
    )
    value_width = 12
    print("\n  CIR ANALYSIS SUMMARY", flush=True)
    total_width = measure_width + description_width + value_width * len(shock_headers) + 4
    print("  " + "-" * total_width, flush=True)
    header = (
        "  "
        + "Metric".ljust(measure_width)
        + "Formula".ljust(description_width)
        + "".join(header.rjust(value_width) for header in shock_headers)
    )
    print(header, flush=True)
    print("  " + "-" * total_width, flush=True)
    for panel_title, measures in panel_rows:
        print("  " + panel_title, flush=True)
        for measure in measures:
            description = CIR_MEASURE_DESCRIPTIONS.get(measure, "")
            values = [_format_table_value(row["values"].get(measure), measure).rjust(value_width) for row in rows]
            print(
                "  " + measure.ljust(measure_width) + description.ljust(description_width) + "".join(values),
                flush=True,
            )
    print("  " + "-" * total_width, flush=True)


_CIR_REGRESSION_COVARIATES = [
    {"key": "U_M", "label": r"$U_M$", "source": "U_M"},
    {"key": "U_I", "label": r"$U_I$", "source": "U_I"},
    {"key": "sigA", "label": "sigA", "source": "shock_volatility"},
    {"key": "rho", "label": r"$\rho$", "source": "shock_persistence"},
]

_CIR_REGRESSION_OUTCOME_PANELS = [
    {
        "panel_title": "Panel A. Dependent variable: Nonlinear amplification (-)",
        "measure": "Nonlin. ampl. (-)",
        "kind": "cir",
    },
    {
        "panel_title": "Panel B. Dependent variable: Optimal attenuation (-)",
        "measure": "Opt. atten. (-)",
        "kind": "cir",
    },
]


def _significance_stars(p_value: Optional[float]) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def _ols_hc1(y_values: Any, x_columns: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Cross-section OLS with HC1 robust standard errors."""
    from scipy import stats

    try:
        y = np.asarray(y_values, dtype=float).ravel()
        named_cols = []
        for name, values in x_columns.items():
            col = np.asarray(values, dtype=float).ravel()
            if col.shape != y.shape:
                return None
            named_cols.append((name, col))
    except (TypeError, ValueError):
        return None
    if not named_cols:
        return None

    x_stack = np.column_stack([col for _, col in named_cols])
    mask = np.isfinite(y) & np.all(np.isfinite(x_stack), axis=1)
    y = y[mask]
    x_stack = x_stack[mask]
    n_obs, n_vars = x_stack.shape
    if n_obs <= n_vars:
        return None

    # Drop zero-variance covariates (e.g. common rho); keep at least the constant if present.
    keep = []
    for idx, (name, _) in enumerate(named_cols):
        col = x_stack[:, idx]
        if name == "Constant" or float(np.nanstd(col)) > 1e-12:
            keep.append(idx)
    if not keep:
        return None
    names = [named_cols[idx][0] for idx in keep]
    x_stack = x_stack[:, keep]
    n_obs, n_vars = x_stack.shape
    if n_obs <= n_vars:
        return None

    xtx_inv = np.linalg.pinv(x_stack.T @ x_stack)
    beta = xtx_inv @ (x_stack.T @ y)
    resid = y - x_stack @ beta
    meat = x_stack.T @ ((resid * resid)[:, None] * x_stack)
    cov = (n_obs / max(n_obs - n_vars, 1)) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=max(n_obs - n_vars, 1))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n_obs": int(n_obs),
        "r_squared": float(r_squared),
        "coefficients": {
            name: {
                "coef": float(beta[idx]),
                "se": float(se[idx]),
                "p_value": float(p_values[idx]) if np.isfinite(p_values[idx]) else None,
            }
            for idx, name in enumerate(names)
        },
        "dropped_covariates": [named_cols[idx][0] for idx in range(len(named_cols)) if idx not in keep],
    }


def _ergodic_capital_share_change_pp(*, raw_simulation_data, econ_model) -> Optional[np.ndarray]:
    if not raw_simulation_data:
        return None
    ergodic_data = {
        label: sim_data
        for label, sim_data in raw_simulation_data.items()
        if sim_data.get("simulation_kind", "ergodic") == "ergodic"
    }
    if not ergodic_data:
        ergodic_data = raw_simulation_data
    try:
        experiment_name = next(iter(ergodic_data))
        sim_data = ergodic_data[experiment_name]
        n_sectors = econ_model.n_sectors
        variable_info = _sectoral_variable_info("K", n_sectors)
        logdev_values = np.asarray(sim_data["simul_obs"], dtype=float)[:, :n_sectors]
        ss_log_values = np.asarray(econ_model.state_ss, dtype=float)[:n_sectors]
        current_levels = _sectoral_levels_from_logdev(logdev_values, ss_log_values)
        ss_levels = np.exp(ss_log_values)
        weights, _ = _sectoral_share_weights(econ_model.policies_ss, variable_info, n_sectors)
        share_changes = _sectoral_share_change(current_levels, ss_levels, weights)
    except Exception:
        return None
    if share_changes is None or not np.isfinite(share_changes).any():
        return None
    return np.asarray(share_changes, dtype=float) * 100.0


def _sector_measure_vector(sector_rows, *, measure: str, shock_size: float, n_sectors: int) -> np.ndarray:
    values = np.full(n_sectors, np.nan, dtype=float)
    for row in sector_rows:
        if row.get("measure") != measure:
            continue
        size = _as_float(row.get("shock_size"))
        if size is None or abs(size - float(shock_size)) > 1e-8:
            continue
        sector_idx = int(row["sector_idx"])
        if 0 <= sector_idx < n_sectors:
            values[sector_idx] = _as_float(row.get("value_percent"))
    return values


def _regression_design_matrix(upstreamness_data: Dict[str, Any], n_sectors: int) -> Dict[str, np.ndarray]:
    columns = {"Constant": np.ones(n_sectors, dtype=float)}
    for covariate in _CIR_REGRESSION_COVARIATES:
        source = covariate["source"]
        if source == "shock_volatility":
            values = _sectoral_shock_volatility(upstreamness_data.get(source), n_sectors)
        elif source == "shock_persistence":
            values = _sectoral_shock_persistence(upstreamness_data.get(source), n_sectors)
        else:
            raw = upstreamness_data.get(source)
            values = None if raw is None else np.asarray(raw, dtype=float).ravel()
            if values is not None and values.size != n_sectors:
                values = None
        if values is not None:
            columns[covariate["key"]] = np.asarray(values, dtype=float).ravel()
    return columns


def _build_cir_regression_results(
    *,
    sector_rows,
    shock_sizes,
    upstreamness_data,
    capital_share_change_pp,
    n_sectors: int,
) -> Dict[str, Any]:
    x_columns = _regression_design_matrix(upstreamness_data, n_sectors)
    panels = []
    for panel_spec in _CIR_REGRESSION_OUTCOME_PANELS:
        columns = []
        for shock_size in shock_sizes:
            y = _sector_measure_vector(
                sector_rows,
                measure=panel_spec["measure"],
                shock_size=shock_size,
                n_sectors=n_sectors,
            )
            fit = _ols_hc1(y, x_columns)
            columns.append({"shock_size": float(shock_size), "fit": fit})
        panels.append({**panel_spec, "columns": columns})

    capital_fit = None
    if capital_share_change_pp is not None:
        capital_fit = _ols_hc1(capital_share_change_pp, x_columns)
    panels.append(
        {
            "panel_title": "Panel C. Dependent variable: Ergodic capital share change",
            "measure": "Capital share change",
            "kind": "capital",
            "columns": [{"shock_size": None, "fit": capital_fit}],
        }
    )
    return {
        "shock_sizes": [float(size) for size in shock_sizes],
        "covariate_keys": [cov["key"] for cov in _CIR_REGRESSION_COVARIATES if cov["key"] in x_columns],
        "panels": panels,
    }


def _format_regression_coef(fit: Optional[Dict[str, Any]], covariate_key: str) -> str:
    if not fit:
        return ""
    entry = fit.get("coefficients", {}).get(covariate_key)
    if not entry:
        return ""
    stars = _significance_stars(entry.get("p_value"))
    coef = f"{entry['coef']:.3f}"
    return coef if not stars else f"{coef}$^{{{stars}}}$"


def _format_regression_se(fit: Optional[Dict[str, Any]], covariate_key: str) -> str:
    if not fit:
        return ""
    entry = fit.get("coefficients", {}).get(covariate_key)
    if not entry:
        return ""
    return f"({entry['se']:.3f})"


def _write_cir_regression_table(*, results: Dict[str, Any], save_path: str, analysis_name: str) -> None:
    shock_sizes = results.get("shock_sizes") or []
    panels = results.get("panels") or []
    if not panels:
        return

    cir_panels = [panel for panel in panels if panel.get("kind") == "cir"]
    capital_panels = [panel for panel in panels if panel.get("kind") == "capital"]
    used_covariates = {
        name
        for panel in panels
        for column in panel.get("columns", [])
        for name in ((column.get("fit") or {}).get("coefficients") or {})
    }
    display_rows = [
        (cov["key"], cov["label"])
        for cov in _CIR_REGRESSION_COVARIATES
        if cov["key"] in used_covariates
    ]
    if "Constant" in used_covariates:
        display_rows.append(("Constant", "Constant"))

    with open(save_path, "w") as table_file:
        table_file.write("\\begin{table}[htbp]\n\\centering\n")
        table_file.write(
            "\\caption{Cross-sector regressions of CIR measures and capital reallocation}\n"
        )
        table_file.write(f"\\label{{tab:cir_regressions_{_latex_label_token(analysis_name)}}}\n")
        table_file.write("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")

        if cir_panels and shock_sizes:
            n_shock = len(shock_sizes)
            table_file.write("\\resizebox{\\textwidth}{!}{%\n")
            table_file.write("\\begin{tabular}{l" + "c" * n_shock + "}\n\\hline\n")
            table_file.write(
                " & \\multicolumn{"
                + str(n_shock)
                + "}{c}{Shock size} \\\\\n\\cline{2-"
                + str(n_shock + 1)
                + "}\n"
            )
            table_file.write(
                " & " + " & ".join(f"{size:g}\\%" for size in shock_sizes) + " \\\\\n\\hline\n"
            )
            for panel in cir_panels:
                table_file.write(
                    f"\\multicolumn{{{n_shock + 1}}}{{l}}{{\\textit{{{_latex_escape(panel['panel_title'])}}}}} \\\\\n"
                )
                fits = [column.get("fit") for column in panel.get("columns", [])]
                for cov_key, cov_label in display_rows:
                    coef_cells = [_format_regression_coef(fit, cov_key) for fit in fits]
                    se_cells = [_format_regression_se(fit, cov_key) for fit in fits]
                    table_file.write(cov_label + " & " + " & ".join(coef_cells) + " \\\\\n")
                    table_file.write(" & " + " & ".join(se_cells) + " \\\\\n")
                n_cells = ["" if fit is None else str(fit["n_obs"]) for fit in fits]
                r2_cells = [
                    "" if fit is None or not np.isfinite(fit["r_squared"]) else f"{fit['r_squared']:.3f}"
                    for fit in fits
                ]
                table_file.write("Observations & " + " & ".join(n_cells) + " \\\\\n")
                table_file.write("$R^{2}$ & " + " & ".join(r2_cells) + " \\\\\n")
                table_file.write("\\hline\n")
            table_file.write("\\end{tabular}\n}\n")

        if capital_panels:
            table_file.write("\\vspace{0.6em}\n")
            table_file.write("\\begin{tabular}{lc}\n\\hline\n")
            for panel in capital_panels:
                table_file.write(
                    f"\\multicolumn{{2}}{{l}}{{\\textit{{{_latex_escape(panel['panel_title'])}}}}} \\\\\n"
                )
                fit = panel.get("columns", [{}])[0].get("fit")
                for cov_key, cov_label in display_rows:
                    table_file.write(
                        cov_label
                        + " & "
                        + _format_regression_coef(fit, cov_key)
                        + " \\\\\n"
                    )
                    table_file.write(" & " + _format_regression_se(fit, cov_key) + " \\\\\n")
                n_cell = "" if fit is None else str(fit["n_obs"])
                r2_cell = (
                    ""
                    if fit is None or not np.isfinite(fit["r_squared"])
                    else f"{fit['r_squared']:.3f}"
                )
                table_file.write(f"Observations & {n_cell} \\\\\n")
                table_file.write(f"$R^{{2}}$ & {r2_cell} \\\\\n")
            table_file.write("\\hline\n\\end{tabular}\n")

        dropped = sorted(
            {
                name
                for panel in panels
                for column in panel.get("columns", [])
                for name in ((column.get("fit") or {}).get("dropped_covariates") or [])
                if name != "Constant"
            }
        )
        dropped_note = ""
        if dropped:
            dropped_note = (
                " Covariates without cross-sector variation are omitted ("
                + ", ".join(dropped)
                + ")."
            )
        table_file.write(
            "\\begin{minipage}{0.92\\textwidth}\n\\footnotesize\n"
            "\\textit{Notes:} Each column is a cross-sector OLS regression with HC1 robust standard errors "
            "in parentheses. Panels A--B use sector-level CIR outcomes for negative TFP shocks at the "
            "indicated size; the dependent variables are in percent. Panel C uses the ergodic-mean sectoral "
            "capital composition share relative to the deterministic steady state, also in percent, and does "
            "not vary with shock size. Covariates are IO upstreamness $U_M$, investment upstreamness $U_I$, "
            r"sectoral TFP shock volatility sigA, and sectoral TFP persistence $\rho$. "
            "Stars denote $p<0.10$, $p<0.05$, and $p<0.01$."
            + dropped_note
            + "\n\\end{minipage}\n"
        )
        table_file.write("\\end{table}\n")


def _print_cir_regression_table(results: Dict[str, Any]) -> None:
    print("\n  CIR / CAPITAL CROSS-SECTOR REGRESSIONS", flush=True)
    for panel in results.get("panels", []):
        print(f"  {panel['panel_title']}", flush=True)
        for column in panel.get("columns", []):
            fit = column.get("fit")
            size = column.get("shock_size")
            size_label = "capital" if size is None else f"{size:g}%"
            if not fit:
                print(f"    {size_label}: unavailable", flush=True)
                continue
            coef_bits = []
            for cov in _CIR_REGRESSION_COVARIATES + [{"key": "Constant"}]:
                entry = fit["coefficients"].get(cov["key"])
                if not entry:
                    continue
                coef_bits.append(
                    f"{cov['key']}={entry['coef']:.3f}{_significance_stars(entry.get('p_value'))}"
                )
            print(
                f"    {size_label}: "
                + ", ".join(coef_bits)
                + f"; N={fit['n_obs']}, R2={fit['r_squared']:.3f}",
                flush=True,
            )


def _write_cir_regression_csv(*, results: Dict[str, Any], save_path: str, analysis_name: str) -> None:
    fieldnames = [
        "analysis_name",
        "panel",
        "measure",
        "kind",
        "shock_size",
        "covariate",
        "coefficient",
        "std_error",
        "p_value",
        "n_obs",
        "r_squared",
    ]
    rows = []
    for panel in results.get("panels", []):
        for column in panel.get("columns", []):
            fit = column.get("fit")
            if not fit:
                continue
            for cov_key, entry in fit["coefficients"].items():
                rows.append(
                    {
                        "analysis_name": analysis_name,
                        "panel": panel.get("panel_title", ""),
                        "measure": panel.get("measure", ""),
                        "kind": panel.get("kind", ""),
                        "shock_size": "" if column.get("shock_size") is None else column["shock_size"],
                        "covariate": cov_key,
                        "coefficient": entry["coef"],
                        "std_error": entry["se"],
                        "p_value": entry.get("p_value"),
                        "n_obs": fit["n_obs"],
                        "r_squared": fit["r_squared"],
                    }
                )
    if not rows:
        return
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ("coefficient", "std_error", "p_value", "r_squared"):
                csv_row[key] = _format_csv_float(row.get(key))
            writer.writerow(csv_row)
    print(f"  Saved CIR regression artifact: {os.path.basename(save_path)}", flush=True)


def render_cir_analysis_outputs(
    *,
    config,
    irs_dir,
    econ_model,
    gir_data,
    postprocess_context,
    raw_simulation_data=None,
):
    ir_render_context = postprocess_context.get("ir_render_context") if postprocess_context else None
    if not ir_render_context:
        return
    rows = _build_cir_analysis_table(
        config=config,
        gir_data=gir_data,
        matlab_ir_data=ir_render_context["matlab_ir_data"],
        upstreamness_data=postprocess_context.get("upstreamness_data", {}),
        matlab_breakdown_rows=postprocess_context.get("matlab_irf_sector_breakdown_rows", []),
        n_sectors=econ_model.n_sectors,
    )
    if not rows:
        print("  CIR analysis skipped: no compatible global or benchmark IR objects found.", flush=True)
        return
    ir_tables_dir = os.path.join(irs_dir, "IR_tables")
    os.makedirs(ir_tables_dir, exist_ok=True)
    output_path = os.path.join(ir_tables_dir, f"cir_analysis_{config['analysis_name']}.tex")
    _write_cir_analysis_table(
        rows=rows,
        save_path=output_path,
        analysis_name=config["analysis_name"],
        response_source=_resolve_ir_response_source(config),
    )
    _print_cir_analysis_table(rows)
    sector_rows = _build_cir_sector_value_rows(
        config=config,
        gir_data=gir_data,
        matlab_ir_data=ir_render_context["matlab_ir_data"],
        n_sectors=econ_model.n_sectors,
        sector_labels=econ_model.labels,
    )
    _write_cir_sector_values_csv(
        rows=sector_rows,
        save_path=os.path.join(ir_tables_dir, f"cir_sector_values_{config['analysis_name']}.csv"),
    )

    capital_share_change_pp = _ergodic_capital_share_change_pp(
        raw_simulation_data=raw_simulation_data,
        econ_model=econ_model,
    )
    regression_results = _build_cir_regression_results(
        sector_rows=sector_rows,
        shock_sizes=[row["shock_size"] for row in rows],
        upstreamness_data=postprocess_context.get("upstreamness_data", {}),
        capital_share_change_pp=capital_share_change_pp,
        n_sectors=econ_model.n_sectors,
    )
    regression_path = os.path.join(ir_tables_dir, f"cir_regressions_{config['analysis_name']}.tex")
    _write_cir_regression_table(
        results=regression_results,
        save_path=regression_path,
        analysis_name=config["analysis_name"],
    )
    _write_cir_regression_csv(
        results=regression_results,
        save_path=os.path.join(ir_tables_dir, f"cir_regressions_{config['analysis_name']}.csv"),
        analysis_name=config["analysis_name"],
    )
    _print_cir_regression_table(regression_results)
    print(f"  Saved CIR regression table: {os.path.basename(regression_path)}", flush=True)

    cir_figures_dir = os.path.join(irs_dir, "IR_CIR")
    print("  CIR shock-size profile figures", flush=True)
    plot_cir_shock_size_profiles(
        rows=rows,
        figure_specs=[
            {
                **figure_spec,
                "note_text": _build_cir_profile_note(
                    figure_spec,
                    n_sectors=econ_model.n_sectors,
                    rows=rows,
                ),
            }
            for figure_spec in CIR_FIGURE_SPECS
        ],
        save_dir=cir_figures_dir,
        analysis_name=config["analysis_name"],
        show_plot=bool(config.get("show_ir_plots", False)),
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
            source_label="Log-Linear",
        )

        for var_name, var_values in firstorder_analysis_vars.items():
            n_nan = jnp.sum(jnp.isnan(var_values))
            if n_nan > 0:
                print(f"    WARNING: {var_name} has {n_nan} NaN values!", flush=True)
        analysis_variables_data["Log-Linear"] = firstorder_analysis_vars
        print("  Loaded log-linear simulation with consistent aggregation.")

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
        print("  Perfect Foresight moments will use the perfect-foresight simulation series.")
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

    upstreamness_data = _build_upstreamness_data(
        model_data,
        econ_model.upstreamness(),
        n_sectors=n_sectors,
    )
    if upstreamness_data.get("shock_persistence") is None:
        shock_persistence = _sectoral_shock_persistence(getattr(econ_model, "rho", None), n_sectors)
        if shock_persistence is not None:
            upstreamness_data["shock_persistence"] = shock_persistence
    matlab_irf_sector_breakdown_rows = _extract_matlab_irf_breakdown_rows(model_data)
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
            "matlab_irf_sector_breakdown_rows": matlab_irf_sector_breakdown_rows,
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
            "reference_method": schedule.get("reference_method", "benchmark"),
            "burn_in": int(schedule.get("burn_in", 0)),
            "active_periods": active_periods,
            "burn_out": int(schedule.get("burn_out", 0)),
            "total_periods": total_periods,
        }
    )
    return context


def _build_aggregate_histogram_note(histogram_context):
    del histogram_context
    return (
        "The panels show the ergodic distributions of aggregate variables as percent log deviations from the "
        "deterministic steady state (DSS). The global solution and first-order approximation are simulated using "
        "the same structure and identical shock realizations. The solid blue lines report the global solution, "
        "the dashed black lines report the first-order approximation, and the vertical dashed lines mark the DSS. "
        "The vertical axis reports the fraction of observations in each bin."
    )


def render_aggregate_ir_outputs(*, config, irs_dir, econ_model, gir_data, postprocess_context):
    ir_render_context = postprocess_context.get("ir_render_context") if postprocess_context else None
    if not ir_render_context:
        return

    aggregate_irs_dir = os.path.join(irs_dir, "IR_aggregate")
    os.makedirs(aggregate_irs_dir, exist_ok=True)
    show_ir_plots = bool(config.get("show_ir_plots", False))
    for sector_idx in ir_render_context["sectors_to_plot"]:
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        print(f"\n  Aggregate IRs: {sector_label} (sector {sector_idx + 1})")
        for ir_variable in ir_render_context["ir_variables"]:
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=ir_render_context["matlab_ir_data"],
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=ir_render_context["shock_sizes"],
                save_dir=aggregate_irs_dir,
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
                show_plot=show_ir_plots,
            )
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=ir_render_context["matlab_ir_data"],
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=[ir_render_context["largest_shock"]],
                save_dir=aggregate_irs_dir,
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
                show_plot=show_ir_plots,
            )


def render_upstreamness_outputs(*, config, simulation_dir, econ_model, postprocess_context):
    upstreamness_data = postprocess_context.get("upstreamness_data") if postprocess_context else None
    if not upstreamness_data:
        return

    show_plot = bool(config.get("show_upstreamness_plot", False))
    try:
        plot_upstreamness(
            upstreamness_data=upstreamness_data,
            save_dir=simulation_dir,
            analysis_name=config["analysis_name"],
            sector_labels=econ_model.labels,
            show_plot=show_plot,
        )
        plot_sectoral_diagnostic_bar(
            values=upstreamness_data.get("shock_volatility"),
            save_dir=simulation_dir,
            analysis_name=config["analysis_name"],
            filename_stem="sectoral_shock_volatility",
            ylabel="TFP Shock Volatility",
            note_text=(
                "Each bar reports a sector's TFP innovation standard deviation. For a covariance-matrix input, "
                "the value is the square root of the corresponding diagonal element."
            ),
            sector_labels=econ_model.labels,
            show_plot=show_plot,
        )
        plot_sectoral_diagnostic_bar(
            values=getattr(econ_model, "rho", None),
            save_dir=simulation_dir,
            analysis_name=config["analysis_name"],
            filename_stem="sectoral_tfp_persistence",
            ylabel="TFP Persistence",
            note_text=(
                "Each bar reports the sectoral AR(1) persistence coefficient rho_j. A common scalar rho produces "
                "the same bar value for every sector."
            ),
            sector_labels=econ_model.labels,
            show_plot=show_plot,
        )
    except Exception as exc:
        print(f"    Failed to create sectoral TFP diagnostic plots: {exc}", flush=True)


def render_aggregate_histogram_outputs(*, config, simulation_dir, analysis_variables_data, postprocess_context):
    if not analysis_variables_data or not postprocess_context:
        return

    reference_experiment_label = postprocess_context.get("reference_experiment_label")
    reference_analysis_variables = postprocess_context.get("reference_analysis_variables")
    if not reference_experiment_label:
        return
    histogram_context = postprocess_context.get("aggregate_histogram_context") or {}

    selected_methods = [("__reference__", "Global solution"), *AGGREGATE_HISTOGRAM_BENCHMARKS]
    ordered_histogram_data = {}
    missing_methods = []

    for source_label, display_label in selected_methods:
        if source_label == "__reference__":
            series = analysis_variables_data.get(reference_experiment_label, reference_analysis_variables)
        else:
            series = analysis_variables_data.get(source_label)
        if series is None:
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

    print("  Aggregate histograms: Global solution vs 1st-order approximation", flush=True)
    plot_ergodic_histograms(
        analysis_variables_data=ordered_histogram_data,
        save_dir=simulation_dir,
        analysis_name=config["analysis_name"],
        theo_dist_params=None,
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
            plot_sectoral_variable_composition_stochss(
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

    sectoral_irs_dir = os.path.join(irs_dir, "IR_sectoral")
    os.makedirs(sectoral_irs_dir, exist_ok=True)
    show_ir_plots = bool(config.get("show_ir_plots", False))
    sectoral_ir_variables = ir_render_context["sectoral_ir_variables"]
    for sector_idx in ir_render_context["sectors_to_plot"]:
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        if sectoral_ir_variables:
            print(f"\n  Sectoral IRs: {sector_label} (sector {sector_idx + 1})")

        for ir_variable in sectoral_ir_variables:
            plot_sector_ir_by_shock_size(
                gir_data=gir_data,
                matlab_ir_data=ir_render_context["matlab_ir_data"],
                sector_idx=sector_idx,
                sector_label=sector_label,
                variable_to_plot=ir_variable,
                shock_sizes=[ir_render_context["largest_shock"]],
                save_dir=sectoral_irs_dir,
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
                show_plot=show_ir_plots,
            )


def _write_sectoral_allocation_comparison(
    *,
    config,
    simulation_dir,
    econ_model,
    ergodic_raw_simulation_data,
    stochastic_ss_states,
    stochastic_ss_policies,
):
    if not ergodic_raw_simulation_data or not stochastic_ss_policies:
        return

    ergodic_name = _single_experiment_name(
        ergodic_raw_simulation_data,
        "_write_sectoral_allocation_comparison",
    )
    stochss_name = _single_experiment_name(
        stochastic_ss_policies,
        "_write_sectoral_allocation_comparison",
    )
    rows = []
    n_sectors = econ_model.n_sectors

    for variable_name in ["K", "L", "Y", "M", "Q"]:
        variable_info = _sectoral_variable_info(variable_name, n_sectors)
        idx_start = variable_info["index_start"]
        idx_end = idx_start + n_sectors

        if variable_info["source"] == "state":
            ergodic_logdev = ergodic_raw_simulation_data[ergodic_name]["simul_obs"][:, idx_start:idx_end]
            stochss_logdev = stochastic_ss_states[stochss_name][idx_start:idx_end]
            ss_log_values = econ_model.state_ss[idx_start:idx_end]
        else:
            ergodic_logdev = ergodic_raw_simulation_data[ergodic_name]["simul_policies"][:, idx_start:idx_end]
            stochss_logdev = stochastic_ss_policies[stochss_name][idx_start:idx_end]
            ss_log_values = econ_model.policies_ss[idx_start:idx_end]

        ss_levels = np.exp(np.asarray(ss_log_values, dtype=float))
        weights, _ = _sectoral_share_weights(econ_model.policies_ss, variable_info, n_sectors)
        ergodic_changes = _sectoral_share_change(
            _sectoral_levels_from_logdev(ergodic_logdev, ss_log_values),
            ss_levels,
            weights,
        )
        stochss_changes = _sectoral_share_change(
            _sectoral_levels_from_logdev(stochss_logdev, ss_log_values),
            ss_levels,
            weights,
        )
        correlation = _safe_corr(ergodic_changes, stochss_changes)

        for sector_idx, sector_label in enumerate(econ_model.labels):
            rows.append(
                {
                    "variable": variable_name,
                    "sector_index": sector_idx,
                    "sector": sector_label,
                    "ergodic_excess_allocation_percent": ergodic_changes[sector_idx] * 100.0,
                    "stochastic_ss_excess_allocation_percent": stochss_changes[sector_idx] * 100.0,
                    "cross_sector_correlation": correlation,
                }
            )

    output_path = os.path.join(
        simulation_dir,
        f"sectoral_allocation_comparison_{config['analysis_name']}.csv",
    )
    fieldnames = [
        "variable",
        "sector_index",
        "sector",
        "ergodic_excess_allocation_percent",
        "stochastic_ss_excess_allocation_percent",
        "cross_sector_correlation",
    ]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}", flush=True)


def render_ergodic_sectoral_outputs(
    *,
    config,
    simulation_dir,
    econ_model,
    raw_simulation_data,
    postprocess_context,
    stochastic_ss_states=None,
    stochastic_ss_policies=None,
):
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

    _write_sectoral_allocation_comparison(
        config=config,
        simulation_dir=simulation_dir,
        econ_model=econ_model,
        ergodic_raw_simulation_data=ergodic_raw_simulation_data,
        stochastic_ss_states=stochastic_ss_states or {},
        stochastic_ss_policies=stochastic_ss_policies or {},
    )

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
            plot_sectoral_variable_composition_ergodic(
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
    render_upstreamness_outputs(
        config=config,
        simulation_dir=simulation_dir,
        econ_model=econ_model,
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
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
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
        "1st": ("FirstOrder", "Log-Linear"),
        "2nd": ("SecondOrder", "Second-Order"),
        "PF": ("PerfectForesight", "Perfect Foresight"),
        "MITShocks": ("MITShocks", "MIT Shocks"),
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
