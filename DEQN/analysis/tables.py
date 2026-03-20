"""
General Analysis Tables Module

This module provides table generation functions for standard analysis results.
These functions are model-agnostic and work with any analysis results following
the standard structure using labels from econ_model.get_analysis_variables().
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

_CALIBRATION_TARGETED_ROWS = [
    ("$\\bar{\\sigma}(L_j)$", "sigma_L_avg_empweighted", "sigma_L_avg_empweighted"),
    ("$\\bar{\\sigma}(I_j)$", "sigma_I_avg_invweighted", "sigma_I_avg_invweighted"),
]

_CALIBRATION_UNTARGETED_ROWS = [
    ("$\\sigma(Y_{\\text{agg}})$", "sigma_VA_agg", "sigma_VA_agg"),
    ("$\\sigma(C^{\\text{exp}_{\\text{agg}}})$", "sigma_C_agg", "sigma_C_agg"),
    ("$\\sigma(I^{\\text{exp}_{\\text{agg}}})$", "sigma_I_agg", "sigma_I_agg"),
]

_CALIBRATION_TARGETED_CONSOLE_LABELS = [
    "σ̄(Lⱼ)",
    "σ̄(Iⱼ)",
]

_CALIBRATION_UNTARGETED_CONSOLE_LABELS = [
    "σ(Y_agg)",
    "σ(C^exp_agg)",
    "σ(I^exp_agg)",
]

_MODEL_VS_DATA_PANELS = [
    (
        "Targeted moments",
        [
            (
                "$\\sum_j \\omega_j^{I}\\sigma(I_{jt})$",
                "sigma_I_avg_invweighted",
                "sigma_I_avg_invweighted",
                "sum w^I sigma(I_jt)",
            ),
            (
                "$\\sum_j \\omega_j^{VA}\\mathrm{corr}(L_{jt},A_{jt})$",
                "corr_L_TFP_sectoral_avg_vashare",
                ("correlations", "L_TFP_sectoral_avg_vashare"),
                "sum w^VA corr(L_jt,A_jt)",
            ),
        ],
    ),
    (
        "Volatility of aggregates",
        [
            ("$\\sigma(\\mathrm{GDP}_t)$", "sigma_VA_agg", "sigma_VA_agg", "sigma(GDP_t)"),
            ("$\\sigma(C_t)$", "sigma_C_agg", "sigma_C_agg", "sigma(C_t)"),
            ("$\\sigma(I_t)$", "sigma_I_agg", "sigma_I_agg", "sigma(I_t)"),
            ("$\\sigma(L_t)$", "sigma_L_hc_agg", "sigma_L_agg", "sigma(L_t)"),
        ],
    ),
    (
        "Correlation of aggregates",
        [
            ("$\\mathrm{corr}(C_t,L_t)$", "corr_L_C_agg", ("correlations", "L_C_agg"), "corr(C_t,L_t)"),
            ("$\\mathrm{corr}(C_t,I_t)$", "corr_I_C_agg", ("correlations", "I_C_agg"), "corr(C_t,I_t)"),
            (
                "$\\mathrm{corr}(L_t,A_t)$",
                "corr_L_TFP_agg",
                ("correlations", "L_TFP_agg"),
                "corr(L_t,A_t)",
            ),
        ],
    ),
    (
        "Sectoral comovement",
        [
            (
                "avg pairwise $\\mathrm{corr}(\\mathbf{C}_t,\\mathbf{C}_t)$",
                "avg_pairwise_corr_C",
                "avg_pairwise_corr_C",
                "avg pairwise corr(C_t,C_t)",
            ),
            (
                "avg pairwise $\\mathrm{corr}(\\mathbf{Y}_t,\\mathbf{Y}_t)$",
                "avg_pairwise_corr_VA",
                "avg_pairwise_corr_VA",
                "avg pairwise corr(Y_t,Y_t)",
            ),
            (
                "avg pairwise $\\mathrm{corr}(\\mathbf{L}_t,\\mathbf{L}_t)$",
                "avg_pairwise_corr_L",
                "avg_pairwise_corr_L",
                "avg pairwise corr(L_t,L_t)",
            ),
            (
                "avg pairwise $\\mathrm{corr}(\\mathbf{I}_t,\\mathbf{I}_t)$",
                "avg_pairwise_corr_I",
                "avg_pairwise_corr_I",
                "avg pairwise corr(I_t,I_t)",
            ),
        ],
    ),
    (
        "Sectoral weighted-average volatilities",
        [
            ("$\\sum_j \\omega_j^{VA}\\sigma(Y_{jt})$", "sigma_VA_avg", "sigma_VA_avg", "sum w^VA sigma(Y_jt)"),
            ("$\\sum_j \\omega_j^{VA}\\sigma(L_{jt})$", "sigma_L_avg", "sigma_L_avg", "sum w^VA sigma(L_jt)"),
            ("$\\sum_j \\omega_j^{VA}\\sigma(I_{jt})$", "sigma_I_avg", "sigma_I_avg", "sum w^VA sigma(I_jt)"),
            (
                "$\\sum_j \\omega_j^{L,emp}\\sigma(L_{jt})$",
                "sigma_L_avg_empweighted",
                "sigma_L_avg_empweighted",
                "sum w^L,emp sigma(L_jt)",
            ),
            (
                "$\\sum_j \\omega_j^{Q,ss}\\sigma(\\mathrm{Domar}_{jt})$",
                "sigma_Domar_avg",
                "sigma_Domar_avg",
                "sum w^Q,ss sigma(Domar_jt)",
            ),
        ],
    ),
]

_MODEL_VS_DATA_ROW_DEFS = [
    (row_label, model_key, data_key)
    for _, panel_rows in _MODEL_VS_DATA_PANELS
    for row_label, model_key, data_key, _ in panel_rows
]

_MODEL_VS_DATA_CONSOLE_LABELS = [
    console_label
    for _, panel_rows in _MODEL_VS_DATA_PANELS
    for _, _, _, console_label in panel_rows
]

_MODEL_VS_DATA_METHOD_ORDER = ["1st", "Nonlinear", "Nonlinear-CS"]

_MODEL_VS_DATA_METHOD_HEADERS = {
    "1st": r"\shortstack{1st Order\\Approx.}",
    "Nonlinear": r"\shortstack{Global Solution\\(long simul)}",
    "Nonlinear-CS": r"\shortstack{Global Solution\\(common shocks)}",
}

_MODEL_VS_DATA_METHOD_CONSOLE_HEADERS = {
    "1st": "1st Order Approx.",
    "Nonlinear": "Global Solution (long simul)",
    "Nonlinear-CS": "Global Solution (common shocks)",
}

_LOGDEV_PERCENT_NOTE = (
    r"Reported log-difference objects are measured relative to the deterministic steady state; "
    r"for small changes, a value such as $-0.1$ means approximately $0.1$ percent below the deterministic steady state."
)

_DESCRIPTIVE_SHAPE_NOTE = (
    r"Skewness is positive when the distribution has a longer right tail and negative when it has a longer left tail. "
    r"Excess kurtosis is reported relative to the Gaussian benchmark, so positive values indicate fatter tails and more extreme events, while negative values indicate thinner tails."
)


def _format_method_display_name(method_name: str, method_names: Optional[list[str]] = None) -> str:
    if method_names and "Global Solution (Common Shocks)" in method_names and method_name == "Global Solution":
        return "Global Solution (Long Simulation)"
    return method_name


def _nonlinear_method_note(method_names: list[str]) -> str:
    if "Global Solution" in method_names and "Global Solution (Common Shocks)" in method_names:
        return (
            r" Global Solution (Long Simulation) uses the long ergodic simulation sample, while "
            r"Global Solution (Common Shocks) uses the short simulation driven by the MATLAB common-shock path."
        )
    return ""


def _wrap_table_environment(tabular_code: str, *, caption: str, label: str, note_text: str, note_width: str = "0.92") -> str:
    return (
        r"\begin{table}[H]" + "\n"
        + r"\centering" + "\n"
        + rf"\caption{{{caption}}}" + "\n"
        + rf"\label{{{label}}}" + "\n"
        + tabular_code
        + rf"\begin{{minipage}}{{{note_width}\textwidth}}" + "\n"
        + r"\vspace{0.5em}" + "\n"
        + r"\footnotesize" + "\n"
        + r"\textit{Notes:} "
        + note_text
        + "\n"
        + r"\end{minipage}" + "\n"
        + r"\end{table}" + "\n"
    )


def _scalar_from_matlab_value(x: Any) -> Optional[float]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return float(arr.ravel()[0])


def _first_available_scalar(source: Optional[Dict[str, Any]], keys: list[str]) -> Optional[float]:
    """Return first available scalar value from a list of candidate keys."""
    if source is None:
        return None
    for key in keys:
        if key in source:
            val = _scalar_from_matlab_value(source[key])
            if val is not None:
                return val
    return None


def create_calibration_table(
    empirical_targets: Dict[str, Any],
    first_order_model_stats: Optional[Dict[str, Any]] = None,
    method_model_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table comparing first-order model moments to empirical targets.

    Structure: targeted sectoral volatilities first, then untargeted aggregates.
    No ratio column.

    Parameters:
    -----------
    empirical_targets : dict
        From ModelData.calibration.empirical_targets or ModelData.EmpiricalTargets.
    first_order_model_stats : dict, optional
        From ModelData.Statistics.FirstOrder.ModelStats. If None, model column shows N/A.
    save_path : str, optional
        If provided, save the LaTeX table to this path.
    analysis_name : str, optional
        Name of the analysis to include in the filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    if method_model_stats:
        latex_code = _create_model_vs_data_moments_table(
            empirical_targets=empirical_targets,
            method_model_stats=method_model_stats,
        )
        console_output = _generate_model_vs_data_console_table(
            empirical_targets=empirical_targets,
            method_model_stats=method_model_stats,
            analysis_name=analysis_name,
        )
        if save_path:
            if analysis_name:
                save_dir = os.path.dirname(save_path)
                ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
                new_filename = f"calibration_table_{analysis_name}{ext}"
                final_save_path = os.path.join(save_dir, new_filename)
            else:
                final_save_path = save_path
            with open(final_save_path, "w") as f:
                f.write(latex_code)

        print(console_output)
        return latex_code

    model_key_aliases = {
        "sigma_VA_agg": ["sigma_VA_agg", "sigma_GDP_agg"],
        "sigma_C_agg": ["sigma_C_agg", "sigma_C_exp_agg", "sigma_C_expenditure_agg"],
        "sigma_I_agg": ["sigma_I_agg", "sigma_I_exp_agg", "sigma_I_expenditure_agg"],
        "sigma_L_hc_agg": ["sigma_L_hc_agg", "sigma_L_headcount_agg", "sigma_L_agg"],
        "sigma_L_avg_empweighted": ["sigma_L_avg_empweighted"],
        "sigma_I_avg_invweighted": ["sigma_I_avg_invweighted"],
    }
    data_key_aliases = {
        "sigma_VA_agg": ["sigma_VA_agg", "sigma_GDP_agg"],
        "sigma_C_agg": ["sigma_C_agg", "sigma_C_exp_agg", "sigma_C_expenditure_agg"],
        "sigma_I_agg": ["sigma_I_agg", "sigma_I_exp_agg", "sigma_I_expenditure_agg"],
        "sigma_L_agg": ["sigma_L_agg", "sigma_L_hc_agg", "sigma_L_headcount_agg"],
        "sigma_L_avg_empweighted": ["sigma_L_avg_empweighted"],
        "sigma_I_avg_invweighted": ["sigma_I_avg_invweighted"],
    }

    def _resolve_rows(row_defs):
        rows = []
        for row_label, model_key, data_key in row_defs:
            model_val = _first_available_scalar(
                first_order_model_stats, model_key_aliases.get(model_key, [model_key])
            )
            data_val = _first_available_scalar(empirical_targets, data_key_aliases.get(data_key, [data_key]))
            rows.append((row_label, model_val, data_val))
        return rows

    targeted_rows = _resolve_rows(_CALIBRATION_TARGETED_ROWS)
    untargeted_rows = _resolve_rows(_CALIBRATION_UNTARGETED_ROWS)

    latex_code = _generate_calibration_latex_table(targeted_rows, untargeted_rows)
    console_output = _generate_calibration_console_table(targeted_rows, untargeted_rows, analysis_name)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"calibration_table_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path
        with open(final_save_path, "w") as f:
            f.write(latex_code)

    print(console_output)
    return latex_code


def _nested_lookup(source: Optional[Dict[str, Any]], path: tuple[str, str]) -> Optional[float]:
    if source is None:
        return None
    block = source.get(path[0])
    if isinstance(block, dict):
        return _scalar_from_matlab_value(block.get(path[1]))
    return None


def _resolve_empirical_value(empirical_targets: Dict[str, Any], data_key: Any) -> Optional[float]:
    if isinstance(data_key, tuple):
        nested_val = _nested_lookup(empirical_targets, data_key)
        if nested_val is not None:
            return nested_val
        if data_key[1] == "L_TFP_sectoral_avg_vashare":
            sectoral = _nested_lookup_array(empirical_targets, ("correlations", "L_TFP_sectoral"))
            weights = _array_from_value(empirical_targets.get("va_weights"))
            if sectoral is not None and weights is not None and sectoral.size == weights.size:
                return _weighted_mean_ignore_nan(sectoral, weights)
        return _first_available_scalar(
            empirical_targets,
            [data_key[1], f"corr_{data_key[1]}", f"avg_{data_key[1]}"],
        )
    return _first_available_scalar(
        empirical_targets,
        [data_key, f"corr_{data_key}", f"avg_{data_key}"],
    )


def _nested_lookup_array(source: Optional[Dict[str, Any]], path: tuple[str, str]) -> Optional[np.ndarray]:
    if source is None:
        return None
    block = source.get(path[0])
    if not isinstance(block, dict):
        return None
    return _array_from_value(block.get(path[1]))


def _array_from_value(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _weighted_mean_ignore_nan(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return None
    values = values[mask]
    weights = weights[mask]
    weights = weights / np.sum(weights)
    return float(np.sum(weights * values))


def _create_model_vs_data_moments_table(
    empirical_targets: Dict[str, Any],
    method_model_stats: Dict[str, Dict[str, Any]],
) -> str:
    method_order = [name for name in _MODEL_VS_DATA_METHOD_ORDER if name in method_model_stats]
    method_order.extend(name for name in method_model_stats if name not in method_order)
    method_order = [name for name in method_order if name in _MODEL_VS_DATA_METHOD_ORDER]
    n_methods = len(method_order)
    latex_code = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\caption{Model vs. data business-cycle moments}" + "\n"
        r"\label{tab:model_vs_data_moments}" + "\n"
        + f"\\begin{{tabular}}{{l *{{{n_methods + 1}}}{{r}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Moment}"
    )
    for method_name in method_order:
        latex_code += " & " + _MODEL_VS_DATA_METHOD_HEADERS.get(method_name, method_name)
    latex_code += r" & \textbf{Data} \\" + "\n" + r"\midrule" + "\n"

    total_columns = n_methods + 2
    panel_labels = ["Panel A", "Panel B", "Panel C", "Panel D", "Panel E"]
    for panel_index, (panel_title, panel_rows) in enumerate(_MODEL_VS_DATA_PANELS):
        panel_prefix = panel_labels[panel_index] if panel_index < len(panel_labels) else f"Panel {panel_index + 1}"
        panel_title_display = panel_title
        if panel_title == "Sectoral weighted-average volatilities":
            panel_title_display = panel_title + r"$^{a}$"
        latex_code += (
            rf"\multicolumn{{{total_columns}}}{{l}}{{\textit{{{panel_prefix}. {panel_title_display}}}}} \\"
            + "\n"
        )
        for row_label, model_key, data_key, _ in panel_rows:
            latex_code += row_label
            for method_name in method_order:
                method_stats = method_model_stats.get(method_name)
                value = _first_available_scalar(method_stats, [model_key]) if method_stats else None
                latex_code += f" & {value:.4f}" if value is not None else " & ---"
            data_value = _resolve_empirical_value(empirical_targets, data_key)
            latex_code += f" & {data_value:.4f}" if data_value is not None else " & ---"
            latex_code += r" \\" + "\n"
        if panel_index < len(_MODEL_VS_DATA_PANELS) - 1:
            latex_code += r"\addlinespace[0.4em]" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    latex_code += (
        r"\\" + "\n"
        r"{\footnotesize $^{a}$ Panel E reports sectoral weighted averages."
        r" Unless otherwise indicated, weights are value-added shares."
        r" $\omega_j^{L,emp}$ denotes employment shares, and $\omega_j^{Q,ss}=Q_{j,ss}/\sum_i Q_{i,ss}$ denotes steady-state gross-output shares.}" + "\n"
        r"\begin{minipage}{0.92\textwidth}" + "\n"
        r"\vspace{0.5em}" + "\n"
        r"\footnotesize" + "\n"
        r"\textit{Notes:} Entries are business-cycle moments. Volatility rows report standard deviations of log differences from the deterministic steady state; correlations are unit-free."
        r" For small changes, a value such as $-0.1$ means approximately $0.1$ percent below the deterministic steady state."
        r" Boldface objects such as $\mathbf{C}_t$ denote the full sectoral vector."
        r" Comovement refers to the average pairwise correlation across sectors for the corresponding sectoral vector."
        r" Aggregate rows are re-aggregated in Python using fixed ergodic-price weights so the aggregate definition is consistent across methods."
        r" The 1st Order Approx. and Global Solution (common shocks) columns use simulations of 5{,}000 periods."
        r" The Global Solution (long simul) column uses 16 parallel simulations with 64{,}000 periods each."
        r" In targeted moments, $\omega_j^{I}$ denotes investment-expenditure weights."
        + r" Data moments come from the empirical targets loaded with the MATLAB objects."
        + "\n"
        r"\end{minipage}" + "\n"
    )
    latex_code += r"\end{table}" + "\n"
    return latex_code


def _generate_model_vs_data_console_table(
    empirical_targets: Dict[str, Any],
    method_model_stats: Dict[str, Dict[str, Any]],
    analysis_name: Optional[str] = None,
) -> str:
    method_order = [name for name in _MODEL_VS_DATA_METHOD_ORDER if name in method_model_stats]
    method_order.extend(name for name in method_model_stats if name not in method_order)
    method_order = [name for name in method_order if name in _MODEL_VS_DATA_METHOD_ORDER]
    method_col_width = max(
        11,
        len("Data"),
        *(len(_MODEL_VS_DATA_METHOD_CONSOLE_HEADERS.get(method_name, method_name)) for method_name in method_order),
    )
    row_label_width = max(
        22,
        len("Moment"),
        *(len(label) for label in _MODEL_VS_DATA_CONSOLE_LABELS),
    )
    output = []
    output.append("")
    output.append("[1] MODEL VS DATA MOMENTS")
    output.append("")
    if analysis_name:
        output.append(f"Analysis: {analysis_name}")
        output.append("")

    header = f"{'Moment':<{row_label_width}}"
    for method_name in method_order:
        display_name = _MODEL_VS_DATA_METHOD_CONSOLE_HEADERS.get(method_name, method_name)
        header += f"{display_name:>{method_col_width}}"
    header += f"{'Data':>{method_col_width}}"
    output.append(header)
    output.append("-" * len(header))

    for (row_label, model_key, data_key), console_label in zip(_MODEL_VS_DATA_ROW_DEFS, _MODEL_VS_DATA_CONSOLE_LABELS):
        del row_label
        row = f"{console_label:<{row_label_width}}"
        for method_name in method_order:
            method_stats = method_model_stats.get(method_name)
            value = _first_available_scalar(method_stats, [model_key]) if method_stats else None
            row += f"{value:{method_col_width}.4f}" if value is not None else f"{'---':>{method_col_width}}"
        data_value = _resolve_empirical_value(empirical_targets, data_key)
        row += f"{data_value:{method_col_width}.4f}" if data_value is not None else f"{'---':>{method_col_width}}"
        output.append(row)

    output.append("")
    return "\n".join(output)


def _generate_calibration_console_table(
    targeted_rows: list, untargeted_rows: list, analysis_name: Optional[str] = None
) -> str:
    line_len = 80
    output = []
    output.append("")
    output.append("=" * line_len)
    output.append(" " * ((line_len - 42) // 2) + "calibration table (model vs data)")
    output.append("=" * line_len)
    if analysis_name:
        output.append(f"Experiment: {analysis_name}")
    output.append("-" * line_len)
    output.append("")
    output.append("    targeted moments (sectoral volatilities)")
    output.append(f"    {'':18s} {'Model':>7s}   {'Data':>7s}")
    output.append("    " + "-" * 36)

    for (_, model_val, data_val), console_label in zip(targeted_rows, _CALIBRATION_TARGETED_CONSOLE_LABELS):
        ms = f"{model_val:7.4f}" if model_val is not None else "    N/A"
        ds = f"{data_val:7.4f}" if data_val is not None else "    N/A"
        output.append(f"    {console_label:<18} {ms}   {ds}")

    output.append("")
    output.append("    untargeted moments (aggregates)")
    output.append(f"    {'':18s} {'Model':>7s}   {'Data':>7s}")
    output.append("    " + "-" * 36)

    for (_, model_val, data_val), console_label in zip(untargeted_rows, _CALIBRATION_UNTARGETED_CONSOLE_LABELS):
        ms = f"{model_val:7.4f}" if model_val is not None else "    N/A"
        ds = f"{data_val:7.4f}" if data_val is not None else "    N/A"
        output.append(f"    {console_label:<18} {ms}   {ds}")

    output.append("")
    output.append("=" * line_len)
    output.append("")
    return "\n".join(output)


def _generate_calibration_latex_table(targeted_rows: list, untargeted_rows: list) -> str:
    latex_code = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\caption{Model calibration: targeted and untargeted moments}" + "\n"
        r"\label{tab:calibration}" + "\n"
        r"\begin{tabular}{l r r}" + "\n"
        r"\toprule" + "\n"
        r" & \textbf{Model} & \textbf{Data} \\" + "\n"
        r"\midrule" + "\n"
    )

    latex_code += r"\multicolumn{3}{l}{\textit{Targeted moments}} \\" + "\n"
    latex_code += r"\addlinespace[0.3em]" + "\n"
    for row_label, model_val, data_val in targeted_rows:
        model_cell = f"{model_val:.4f}" if model_val is not None else "---"
        data_cell = f"{data_val:.4f}" if data_val is not None else "---"
        latex_code += f"\\quad {row_label} & {model_cell} & {data_cell} \\\\\n"

    latex_code += r"\addlinespace[0.6em]" + "\n"
    latex_code += r"\multicolumn{3}{l}{\textit{Untargeted moments}} \\" + "\n"
    latex_code += r"\addlinespace[0.3em]" + "\n"
    for row_label, model_val, data_val in untargeted_rows:
        model_cell = f"{model_val:.4f}" if model_val is not None else "---"
        data_cell = f"{data_val:.4f}" if data_val is not None else "---"
        latex_code += f"\\quad {row_label} & {model_cell} & {data_cell} \\\\\n"

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += (
        r"\begin{minipage}{0.85\textwidth}" + "\n"
        r"\vspace{0.5em}" + "\n"
        r"\footnotesize" + "\n"
        r"\textit{Notes:} All volatilities are standard deviations of HP-filtered ($\lambda=100$) log series. "
        r"\textit{Targeted moments:} "
        r"$\bar{\sigma}(L_j)$ is the employment-share-weighted average of sectoral employment volatilities; "
        r"$\bar{\sigma}(I_j)$ is the investment-expenditure-share-weighted average of sectoral investment volatilities. "
        r"Both use BEA annual data (1963--2018). "
        r"\textit{Untargeted moments:} "
        r"$\sigma(Y_{\text{agg}})$ is the volatility of aggregate GDP; "
        r"$\sigma(C^{\text{exp}_{\text{agg}}})$ and $\sigma(I^{\text{exp}_{\text{agg}}})$ "
        r"are the volatilities of aggregate consumption and investment. "
        r"In the data, GDP and investment are Tornqvist chain-weighted aggregates of BEA sectoral series; "
        r"consumption is NIPA real personal consumption expenditure. "
        r"In the model, aggregates are fixed-price expenditure sums at steady-state prices: "
        r"$\text{GDP}^{\bar{P}}_t = \sum_j \bar{P}_j(Q_{jt} - M^{\text{out}}_{jt})$, "
        r"$C^{\bar{P}}_t = \sum_j \bar{P}_j C_{jt}$, and $I^{\bar{P}}_t = \sum_j \bar{P}_j I^{\text{out}}_{jt}$ "
        r"(see Appendix~\ref{sec:app_aggregation}). "
        r"Model moments are from the first-order (log-linear) perturbation solution." + "\n"
        r"\end{minipage}" + "\n"
    )
    latex_code += r"\end{table}" + "\n"
    return latex_code


def create_descriptive_stats_table(
    analysis_variables_data: Dict[str, Any],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    theoretical_stats: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> str:
    """
    Create a LaTeX table with descriptive statistics organized by variable.

    Parameters:
    -----------
    analysis_variables_data : dict
        Dictionary where keys are experiment names and values are dictionaries mapping
        variable labels to arrays: {exp_name: {var_label: array}}
    save_path : str, optional
        If provided, save the LaTeX table to this path.
    analysis_name : str, optional
        Name of the analysis to include in the filename.
    theoretical_stats : dict, optional
        Pre-computed statistics for certain experiments (e.g., log-linear theoretical).
        Format: {exp_name: {var_label: {"Mean": val, "Sd": val, "Skewness": val, "Excess Kurtosis": val}}}
        For log-linear: mean=0, skewness=0, kurtosis=0 (normal distribution)

    Returns:
    --------
    str : The LaTeX table code
    """
    excluded_vars = ["Utility"]

    # Combine experiment names from both sources
    experiment_names = list(analysis_variables_data.keys())
    if theoretical_stats:
        for exp_name in theoretical_stats.keys():
            if exp_name not in experiment_names:
                experiment_names.append(exp_name)

    # Get variable labels from first available source
    first_exp = experiment_names[0]
    if first_exp in analysis_variables_data:
        var_labels = [v for v in analysis_variables_data[first_exp].keys() if v not in excluded_vars]
    elif theoretical_stats and first_exp in theoretical_stats:
        var_labels = [v for v in theoretical_stats[first_exp].keys() if v not in excluded_vars]
    else:
        var_labels = []

    stats_data: Dict[str, Dict[str, Dict[str, float]]] = {}

    min_samples = 100

    for var_label in var_labels:
        stats_data[var_label] = {}
        for exp_name in experiment_names:
            used_simulation = False
            if exp_name in analysis_variables_data:
                analysis_vars_dict = analysis_variables_data[exp_name]
                if var_label in analysis_vars_dict:
                    var_values = np.asarray(analysis_vars_dict[var_label])
                    if var_values.size >= min_samples:
                        stats_data[var_label][exp_name] = {
                            "Mean": float(np.mean(var_values) * 100),
                            "Sd": float(np.std(var_values) * 100),
                            "Skewness": float(skew(var_values)),
                            "Excess Kurtosis": float(kurtosis(var_values)),
                        }
                        used_simulation = True

            if not used_simulation and theoretical_stats and exp_name in theoretical_stats:
                if var_label in theoretical_stats[exp_name]:
                    stats_data[var_label][exp_name] = theoretical_stats[exp_name][var_label]

    if len(experiment_names) == 1:
        latex_code = _generate_single_method_latex_table(stats_data, experiment_names[0])
        console_output = _generate_single_method_console_table(stats_data, experiment_names[0])
    else:
        latex_code = _generate_variable_organized_latex_table(stats_data, experiment_names)
        console_output = _generate_console_table(stats_data, experiment_names)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"descriptive_stats_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    print(console_output)

    return latex_code


def _generate_console_table(stats_data: Dict[str, Dict[str, Dict[str, float]]], experiment_names: list) -> str:
    """Generate formatted console output for descriptive statistics."""
    output = []
    output.append("\n" + "═" * 72)
    output.append("  DESCRIPTIVE STATISTICS (% deviations from steady state)")
    output.append("═" * 72)

    for var_label, exp_stats in stats_data.items():
        output.append(f"\n  {var_label}")
        output.append("  " + "─" * 68)

        header = f"    {'Method':<30} {'Mean':>10} {'Sd':>10} {'Skew':>10} {'Ex.Kurt':>10}"
        output.append(header)
        output.append("  " + "─" * 68)

        for exp_name in experiment_names:
            if exp_name in exp_stats:
                stats = exp_stats[exp_name]
                row = f"    {exp_name:<30} {stats['Mean']:>10.3f} {stats['Sd']:>10.3f} {stats['Skewness']:>10.3f} {stats['Excess Kurtosis']:>10.3f}"
                output.append(row)

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_single_method_console_table(
    stats_data: Dict[str, Dict[str, Dict[str, float]]], method_name: str
) -> str:
    """Console table for the single-method case: variables as rows, metrics as columns."""
    output = []
    output.append("\n" + "═" * 72)
    output.append("  DESCRIPTIVE STATISTICS (% deviations from steady state)")
    output.append(f"  Method: {method_name}")
    output.append("═" * 72)
    output.append(f"  {'Variable':<30} {'Mean':>10} {'Sd':>10} {'Skew':>10} {'Ex.Kurt':>10}")
    output.append("  " + "─" * 68)

    for var_label, exp_stats in stats_data.items():
        if method_name in exp_stats:
            s = exp_stats[method_name]
            output.append(
                f"  {var_label:<30} {s['Mean']:>10.3f} {s['Sd']:>10.3f}"
                f" {s['Skewness']:>10.3f} {s['Excess Kurtosis']:>10.3f}"
            )

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_single_method_latex_table(
    stats_data: Dict[str, Dict[str, Dict[str, float]]], method_name: str
) -> str:
    """LaTeX table for single-method case: variables as rows, metrics as columns."""
    tabular_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Variable} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Excess Kurtosis} \\"
        + "\n"
        r"\midrule" + "\n"
    )

    for var_label, exp_stats in stats_data.items():
        if method_name in exp_stats:
            s = exp_stats[method_name]
            var_display = var_label.replace("_", r"\_")
            tabular_code += (
                f"{var_display} & {s['Mean']:.3f} & {s['Sd']:.3f}"
                f" & {s['Skewness']:.3f} & {s['Excess Kurtosis']:.3f} \\\\\n"
            )

    tabular_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    return _wrap_table_environment(
        tabular_code,
        caption="Descriptive statistics",
        label="tab:descriptive_statistics",
        note_text=(
            r"Mean and standard deviation are reported for log differences from the deterministic steady state; "
            + _LOGDEV_PERCENT_NOTE
            + r" Skewness and excess kurtosis are unit-free, and kurtosis is reported as excess kurtosis. "
            + _DESCRIPTIVE_SHAPE_NOTE
        ),
    )


def _generate_variable_organized_latex_table(
    stats_data: Dict[str, Dict[str, Dict[str, float]]], experiment_names: list
) -> str:
    """Generate LaTeX table organized by variable with experiments as rows within each variable section."""
    method_names = list(experiment_names)
    n_experiments = len(method_names)

    tabular_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Method} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Excess Kurtosis} \\"
        + "\n"
        r"\midrule" + "\n"
    )

    for var_idx, (var_label, exp_stats) in enumerate(stats_data.items()):
        tabular_code += f"\\multicolumn{{5}}{{l}}{{\\textbf{{{var_label}}}}} \\\\\n"

        for exp_name in method_names:
            if exp_name in exp_stats:
                stats = exp_stats[exp_name]
                exp_display = _format_method_display_name(exp_name, method_names).replace("_", r"\_")
                tabular_code += (
                    f"\\quad {exp_display} & {stats['Mean']:.3f} & {stats['Sd']:.3f} "
                    f"& {stats['Skewness']:.3f} & {stats['Excess Kurtosis']:.3f} \\\\\n"
                )

        if var_idx < len(stats_data) - 1:
            tabular_code += r"\addlinespace[0.5em]" + "\n"

    tabular_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    return _wrap_table_environment(
        tabular_code,
        caption="Descriptive statistics",
        label="tab:descriptive_statistics",
        note_text=(
            r"For each variable, rows compare the reported simulation methods."
            + _nonlinear_method_note(method_names)
            + r" Mean and standard deviation are reported for log differences from the deterministic steady state; "
            + _LOGDEV_PERCENT_NOTE
            + r" Skewness and excess kurtosis are unit-free, and kurtosis is reported as excess kurtosis. "
            + _DESCRIPTIVE_SHAPE_NOTE
            + r" The Log-Linear and Global Solution (Common Shocks) rows use simulation samples of 5{,}000 periods."
            + r" The Global Solution (Long Simulation) row uses 16 parallel simulations with 64{,}000 periods each."
            r" For nonlinear methods the moments are computed from the Python simulation sample; for benchmark methods they use the simulation blocks loaded from \texttt{ModelData\_simulation.mat} when those blocks are available."
        ),
    )


def create_comparative_stats_table(
    analysis_variables_data: Dict[str, Any],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a comparative LaTeX table with descriptive statistics organized by variable
    with experiments as columns.

    Parameters:
    -----------
    analysis_variables_data : dict
        Dictionary where keys are experiment names and values are dictionaries mapping
        variable labels to arrays: {exp_name: {var_label: array}}
    save_path : str, optional
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    experiment_names = list(analysis_variables_data.keys())

    first_exp = experiment_names[0]
    # TEMPORARY: Skip Utility - Dynare simulations don't have this variable
    excluded_vars = ["Utility"]
    var_labels = [v for v in analysis_variables_data[first_exp].keys() if v not in excluded_vars]

    stats_labels = ["Mean", "Sd", "Skewness", "Excess Kurtosis"]

    table_data = []

    for var_label in var_labels:
        for stat_label in stats_labels:
            row_data: Dict[str, Any] = {"Variable": f"{var_label} ({stat_label})"}

            for exp_name in experiment_names:
                analysis_vars_dict = analysis_variables_data[exp_name]
                if var_label in analysis_vars_dict:
                    var_values = analysis_vars_dict[var_label]

                    if stat_label == "Mean":
                        value = float(np.mean(var_values) * 100)
                    elif stat_label == "Sd":
                        value = float(np.std(var_values) * 100)
                    elif stat_label == "Skewness":
                        value = float(skew(var_values))
                    elif stat_label == "Excess Kurtosis":
                        value = float(kurtosis(var_values))  # scipy.stats.kurtosis returns excess kurtosis
                    else:
                        value = float(np.nan)

                    row_data[exp_name] = value
                else:
                    row_data[exp_name] = float(np.nan)

            table_data.append(row_data)

    df = pd.DataFrame(table_data)
    df.set_index("Variable", inplace=True)

    latex_code = _generate_comparative_latex_table(df, experiment_names)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"descriptive_stats_comparative_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    return latex_code


def _generate_comparative_latex_table(df: pd.DataFrame, experiment_names: list) -> str:
    """Generate comparative LaTeX table code from descriptive statistics DataFrame."""
    n_experiments = len(experiment_names)

    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Variable}"
    )

    for exp_name in experiment_names:
        latex_code += f" & \\textbf{{{exp_name}}}"

    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    for index, row in df.iterrows():
        formatted_row = [f"{float(value):.4f}" if not np.isnan(value) else "—" for value in row]
        latex_code += r"\textbf{" + str(index) + r"} & " + " & ".join(formatted_row) + r" \\" + "\n"

        if any(keyword in str(index) for keyword in ["Excess Kurtosis"]):
            latex_code += r"\cmidrule(lr){1-" + str(n_experiments + 1) + "}\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Notes:} Rows report descriptive moments for each variable-statistic pair across methods."
        r" Mean and standard deviation are reported in percent relative to the deterministic steady state; "
        + _LOGDEV_PERCENT_NOTE
        + r" Skewness and excess kurtosis are unit-free."
        + "\n"
    )

    return latex_code


def create_welfare_table(
    welfare_data: Dict[str, float],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table with welfare costs for different experiments.

    Parameters:
    -----------
    welfare_data : dict
        Dictionary where keys are experiment names and values are welfare losses (in percentage)
    save_path : str, optional
        If provided, save the LaTeX table to this path.
    analysis_name : str, optional
        Name of the analysis to include in the filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    latex_code = _generate_welfare_latex_table(welfare_data)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"welfare_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    console_output = _generate_welfare_console_table(welfare_data)
    print(console_output)

    return latex_code


def _generate_welfare_console_table(welfare_data: Dict[str, float]) -> str:
    """Generate formatted console output for welfare costs."""
    output = []
    output.append("\n" + "═" * 72)
    output.append("  WELFARE COSTS (consumption-equivalent %)")
    output.append("═" * 72)
    output.append(f"\n    {'Method':<40} {'Welfare Cost':>15}")
    output.append("  " + "─" * 68)

    for exp_name, welfare_cost in welfare_data.items():
        output.append(f"    {exp_name:<40} {welfare_cost:>15.4f}%")

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_welfare_latex_table(welfare_data: Dict[str, float]) -> str:
    """Generate LaTeX table code for welfare costs."""
    method_names = list(welfare_data.keys())
    tabular_code = (
        r"\begin{tabular}{l r}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Method} & \textbf{Welfare Cost ($V_c$, \%)} \\" + "\n"
        r"\midrule" + "\n"
    )

    for exp_name, welfare_cost in welfare_data.items():
        exp_display = _format_method_display_name(exp_name, method_names).replace("_", r"\_")
        tabular_code += f"{exp_display} & {welfare_cost:.4f} \\\\\n"

    tabular_code += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    return _wrap_table_environment(
        tabular_code,
        caption="Welfare cost of business cycles",
        label="tab:welfare_costs",
        note_text=(
            r"$V_c$ is the consumption-equivalent amount of consumption agents would be willing to give up in order to eliminate shocks and remain forever at the deterministic steady state."
            + _nonlinear_method_note(method_names)
            + r" A positive value means business cycles reduce welfare. A value of 1.0 means agents would give up 1\% of consumption to remove business-cycle risk."
        ),
    )


def create_stochastic_ss_table(
    stochastic_ss_data: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table with stochastic steady state values for key analysis variables across experiments.

    Parameters:
    -----------
    stochastic_ss_data : dict
        Dictionary where keys are experiment names and values are dictionaries mapping
        variable labels to stochastic steady state values: {exp_name: {var_label: value}}
    save_path : str, optional
        If provided, save the LaTeX table to this path.
    analysis_name : str, optional
        Name of the analysis to include in the filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    latex_code = _generate_stochastic_ss_latex_table(stochastic_ss_data)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"stochastic_ss_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    console_output = _generate_stochastic_ss_console_table(stochastic_ss_data)
    print(console_output)

    return latex_code


_STOCHSS_AGGREGATE_ORDER = [
    ("Agg. Consumption", "C"),
    ("Agg. Investment", "I"),
    ("Agg. GDP", "GDP"),
    ("Agg. Labor", "L"),
    ("Agg. Capital", "K"),
]


def _generate_stochastic_ss_console_table(stochastic_ss_data: Dict[str, Dict[str, float]]) -> str:
    """Generate formatted console output for stochastic steady state (four main aggregates)."""
    output = []
    output.append("\n" + "═" * 72)
    output.append("  STOCHASTIC STEADY STATE (% deviations from deterministic SS)")
    output.append("═" * 72)

    experiment_names = list(stochastic_ss_data.keys())

    header_parts = [f"{'Variable':<25}"]
    for exp_name in experiment_names:
        exp_short = exp_name[:12] if len(exp_name) > 12 else exp_name
        header_parts.append(f"{exp_short:>12}")
    output.append("\n    " + " ".join(header_parts))
    output.append("  " + "─" * 68)

    for var_key, var_short in _STOCHSS_AGGREGATE_ORDER:
        row_parts = [f"{var_short:<25}"]
        for exp_name in experiment_names:
            ss_vars_dict = stochastic_ss_data[exp_name]
            if var_key in ss_vars_dict:
                value = float(ss_vars_dict[var_key]) * 100
                row_parts.append(f"{value:>12.3f}")
            else:
                row_parts.append(f"{'—':>12}")
        output.append("    " + " ".join(row_parts))

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_stochastic_ss_latex_table(stochastic_ss_data: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table code for stochastic steady state (four main aggregates)."""
    experiment_names = list(stochastic_ss_data.keys())
    n_experiments = len(experiment_names)

    tabular_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Variable}"
    )

    for exp_name in experiment_names:
        exp_display = exp_name.replace("_", r"\_")
        tabular_code += f" & \\textbf{{{exp_display}}}"

    tabular_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_key, var_short in _STOCHSS_AGGREGATE_ORDER:
        tabular_code += var_short

        for exp_name in experiment_names:
            ss_vars_dict = stochastic_ss_data[exp_name]
            if var_key in ss_vars_dict:
                value = float(ss_vars_dict[var_key]) * 100
                tabular_code += f" & {value:.3f}"
            else:
                tabular_code += " & —"

        tabular_code += r" \\" + "\n"

    tabular_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    return _wrap_table_environment(
        tabular_code,
        caption="Aggregate stochastic steady state",
        label="tab:stochastic_ss",
        note_text=(
            r"Entries report stochastic steady-state values as log differences from the deterministic steady state."
            r" The stochastic steady state is computed by taking draws from the ergodic distribution, simulating forward with zero shocks, and taking the point to which those paths converge irrespective of the initial draw; this convergence condition is checked."
            r" For small changes, a value such as $-0.1$ means approximately $0.1$ percent below the deterministic steady state."
        ),
    )


def create_stochastic_ss_aggregates_table(
    stochastic_ss_data: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    methods_to_include: Optional[list[str]] = None,
) -> str:
    """
    Create a compact stochastic steady-state table for aggregates C, I, GDP, L, K.

    Values are reported in percentage deviations from deterministic steady state.
    """
    aggregate_order = [
        ("Agg. Consumption", "C"),
        ("Agg. Investment", "I"),
        ("Agg. GDP", "GDP"),
        ("Agg. Labor", "L"),
        ("Agg. Capital", "K"),
    ]

    all_methods = list(stochastic_ss_data.keys())
    if methods_to_include:
        method_names = [m for m in methods_to_include if m in stochastic_ss_data]
    else:
        method_names = all_methods

    n_methods = len(method_names)
    use_compact_layout = n_methods <= 2
    if use_compact_layout:
        tabular_code = (
            f"\\begin{{tabular}}{{l *{{{n_methods}}}{{r}}}}\n"
            + r"\toprule"
            + "\n"
            + r"\textbf{Aggregate}"
        )
    else:
        tabular_code = (
            f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_methods}}}{{X}}}}\n"
            + r"\toprule"
            + "\n"
            + r"\textbf{Aggregate}"
        )
    for method in method_names:
        method_display = _format_method_display_name(method, method_names).replace("_", r"\_")
        tabular_code += f" & \\textbf{{{method_display}}}"
    tabular_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_key, var_short in aggregate_order:
        tabular_code += var_short
        for method in method_names:
            value = stochastic_ss_data.get(method, {}).get(var_key)
            if value is None:
                tabular_code += " & —"
            else:
                tabular_code += f" & {float(value) * 100:.3f}"
        tabular_code += r" \\" + "\n"

    tabular_code += r"\bottomrule" + "\n"
    tabular_code += (r"\end{tabular}" if use_compact_layout else r"\end{tabularx}") + "\n"
    latex_code = _wrap_table_environment(
        tabular_code,
        caption="Aggregate stochastic steady state",
        label="tab:stochastic_ss_aggregates",
        note_text=(
            r"Entries report aggregate stochastic steady-state values for consumption, investment, GDP, labor, and capital as log differences from the deterministic steady state."
            + _nonlinear_method_note(method_names)
            + r" The stochastic steady state is computed by taking draws from the ergodic distribution, simulating forward with zero shocks, and taking the point to which those paths converge irrespective of the initial draw; this convergence condition is checked."
            + r" For small changes, a value such as $-0.1$ means approximately $0.1$ percent below the deterministic steady state."
        ),
    )

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"stochastic_ss_aggregates_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path
        with open(final_save_path, "w") as file:
            file.write(latex_code)

    return latex_code


def create_ergodic_aggregate_stats_table(
    analysis_variables_data: Dict[str, Any],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    methods_to_include: Optional[list[str]] = None,
) -> str:
    """
    Create ergodic descriptive statistics table (Mean, Sd, Skewness, Excess Kurtosis)
    for aggregates C, I, GDP, K.
    """
    aggregate_order = [
        ("Agg. Consumption", "C"),
        ("Agg. Investment", "I"),
        ("Agg. GDP", "GDP"),
        ("Agg. Capital", "K"),
    ]
    stat_labels = ["Mean", "Sd", "Skewness", "Excess Kurtosis"]

    all_methods = list(analysis_variables_data.keys())
    if methods_to_include:
        method_names = [m for m in methods_to_include if m in analysis_variables_data]
    else:
        method_names = all_methods

    n_methods = len(method_names)
    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{ll *{{{n_methods}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Aggregate} & \textbf{Statistic}"
    )
    for method in method_names:
        method_display = method.replace("_", r"\_")
        latex_code += f" & \\textbf{{{method_display}}}"
    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_key, var_short in aggregate_order:
        for idx, stat_name in enumerate(stat_labels):
            if idx == 0:
                latex_code += var_short
            else:
                latex_code += " "
            latex_code += f" & {stat_name}"

            for method in method_names:
                values = analysis_variables_data.get(method, {}).get(var_key)
                if values is None:
                    latex_code += " & —"
                    continue

                values_np = np.asarray(values, dtype=float)
                if values_np.size == 0:
                    latex_code += " & —"
                    continue

                if stat_name == "Mean":
                    stat_value = float(np.mean(values_np) * 100)
                elif stat_name == "Sd":
                    stat_value = float(np.std(values_np) * 100)
                elif stat_name == "Skewness":
                    stat_value = float(skew(values_np))
                else:
                    stat_value = float(kurtosis(values_np))

                latex_code += f" & {stat_value:.3f}"

            latex_code += r" \\" + "\n"

        latex_code += r"\addlinespace[0.35em]" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Notes:} Rows report descriptive moments for the ergodic distribution of aggregate consumption, investment, GDP, and capital."
        r" Mean and standard deviation are in percent relative to the deterministic steady state; "
        + _LOGDEV_PERCENT_NOTE
        + r" Skewness and excess kurtosis are unit-free."
        + "\n"
    )

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            ext = os.path.splitext(os.path.basename(save_path))[1] or ".tex"
            new_filename = f"ergodic_aggregate_stats_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path
        with open(final_save_path, "w") as file:
            file.write(latex_code)

    return latex_code
