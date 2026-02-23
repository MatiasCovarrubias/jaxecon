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
        r"\begin{table}[htbp]" + "\n"
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


def _generate_variable_organized_latex_table(
    stats_data: Dict[str, Dict[str, Dict[str, float]]], experiment_names: list
) -> str:
    """Generate LaTeX table organized by variable with experiments as rows within each variable section."""
    n_experiments = len(experiment_names)

    latex_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Method} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Excess Kurtosis} \\"
        + "\n"
        r"\midrule" + "\n"
    )

    for var_idx, (var_label, exp_stats) in enumerate(stats_data.items()):
        latex_code += f"\\multicolumn{{5}}{{l}}{{\\textbf{{{var_label}}}}} \\\\\n"

        for exp_name in experiment_names:
            if exp_name in exp_stats:
                stats = exp_stats[exp_name]
                exp_display = exp_name.replace("_", r"\_")
                latex_code += f"\\quad {exp_display} & {stats['Mean']:.3f} & {stats['Sd']:.3f} & {stats['Skewness']:.3f} & {stats['Excess Kurtosis']:.3f} \\\\\n"

        if var_idx < len(stats_data) - 1:
            latex_code += r"\addlinespace[0.5em]" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: Mean and Sd are reported as percentage deviations from deterministic steady state.}" + "\n"
    )

    return latex_code


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
        r"\textit{Note: Mean and Sd are reported as percentage deviations from deterministic steady state.}" + "\n"
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
    output.append(f"\n    {'Experiment':<40} {'Welfare Cost':>15}")
    output.append("  " + "─" * 68)

    for exp_name, welfare_cost in welfare_data.items():
        output.append(f"    {exp_name:<40} {welfare_cost:>15.4f}%")

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_welfare_latex_table(welfare_data: Dict[str, float]) -> str:
    """Generate LaTeX table code for welfare costs."""
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l X}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Experiment} & \textbf{Welfare Cost ($V_c$, \%)} \\" + "\n"
        r"\midrule" + "\n"
    )

    for exp_name, welfare_cost in welfare_data.items():
        exp_display = exp_name.replace("_", r"\_")
        latex_code += f"{exp_display} & {welfare_cost:.4f} \\\\\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: $V_c$ is the consumption-equivalent welfare cost of business cycles. "
        + r"A value of 1.0 means agents would need 1\% higher steady-state consumption to be compensated.}"
        + "\n"
    )

    return latex_code


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


def _generate_stochastic_ss_console_table(stochastic_ss_data: Dict[str, Dict[str, float]]) -> str:
    """Generate formatted console output for stochastic steady state."""
    output = []
    output.append("\n" + "═" * 72)
    output.append("  STOCHASTIC STEADY STATE (% deviations from deterministic SS)")
    output.append("═" * 72)

    experiment_names = list(stochastic_ss_data.keys())
    excluded_vars = ["Utility"]
    first_exp = experiment_names[0]
    var_labels = [v for v in stochastic_ss_data[first_exp].keys() if v not in excluded_vars]

    header_parts = [f"{'Variable':<25}"]
    for exp_name in experiment_names:
        exp_short = exp_name[:12] if len(exp_name) > 12 else exp_name
        header_parts.append(f"{exp_short:>12}")
    output.append("\n    " + " ".join(header_parts))
    output.append("  " + "─" * 68)

    for var_label in var_labels:
        row_parts = [f"{var_label:<25}"]
        for exp_name in experiment_names:
            ss_vars_dict = stochastic_ss_data[exp_name]
            if var_label in ss_vars_dict:
                value = float(ss_vars_dict[var_label]) * 100
                row_parts.append(f"{value:>12.3f}")
            else:
                row_parts.append(f"{'—':>12}")
        output.append("    " + " ".join(row_parts))

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_stochastic_ss_latex_table(stochastic_ss_data: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table code for stochastic steady state analysis variables."""
    experiment_names = list(stochastic_ss_data.keys())
    n_experiments = len(experiment_names)

    excluded_vars = ["Utility"]

    first_exp = experiment_names[0]
    var_labels = [v for v in stochastic_ss_data[first_exp].keys() if v not in excluded_vars]

    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Variable}"
    )

    for exp_name in experiment_names:
        exp_display = exp_name.replace("_", r"\_")
        latex_code += f" & \\textbf{{{exp_display}}}"

    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_label in var_labels:
        latex_code += f"{var_label}"

        for exp_name in experiment_names:
            ss_vars_dict = stochastic_ss_data[exp_name]
            if var_label in ss_vars_dict:
                value = float(ss_vars_dict[var_label]) * 100
                latex_code += f" & {value:.3f}"
            else:
                latex_code += " & —"

        latex_code += r" \\" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += r"\textit{Note: Values show percentage deviations from deterministic steady state.}" + "\n"

    return latex_code


def create_stochastic_ss_aggregates_table(
    stochastic_ss_data: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    methods_to_include: Optional[list[str]] = None,
) -> str:
    """
    Create a compact stochastic steady-state table for aggregates C, I, GDP, K.

    Values are reported in percentage deviations from deterministic steady state.
    """
    aggregate_order = [
        ("Agg. Consumption", "C"),
        ("Agg. Investment", "I"),
        ("Agg. GDP", "GDP"),
        ("Agg. Capital", "K"),
    ]

    all_methods = list(stochastic_ss_data.keys())
    if methods_to_include:
        method_names = [m for m in methods_to_include if m in stochastic_ss_data]
    else:
        method_names = all_methods

    n_methods = len(method_names)
    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_methods}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Aggregate}"
    )
    for method in method_names:
        method_display = method.replace("_", r"\_")
        latex_code += f" & \\textbf{{{method_display}}}"
    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_key, var_short in aggregate_order:
        latex_code += var_short
        for method in method_names:
            value = stochastic_ss_data.get(method, {}).get(var_key)
            if value is None:
                latex_code += " & —"
            else:
                latex_code += f" & {float(value) * 100:.3f}"
        latex_code += r" \\" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += r"\textit{Note: Values are percentage deviations from deterministic steady state.}" + "\n"

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
        r"\textit{Note: Mean and Sd are in percentage deviations from deterministic steady state.}"
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
