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


def create_descriptive_stats_table(
    analysis_variables_data: Dict[str, Any],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
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

    Returns:
    --------
    str : The LaTeX table code
    """
    excluded_vars = ["Utility"]

    experiment_names = list(analysis_variables_data.keys())
    first_exp = experiment_names[0]
    var_labels = [v for v in analysis_variables_data[first_exp].keys() if v not in excluded_vars]

    stats_data: Dict[str, Dict[str, Dict[str, float]]] = {}

    for var_label in var_labels:
        stats_data[var_label] = {}
        for exp_name in experiment_names:
            analysis_vars_dict = analysis_variables_data[exp_name]
            if var_label in analysis_vars_dict:
                var_values = analysis_vars_dict[var_label]
                stats_data[var_label][exp_name] = {
                    "Mean": float(np.mean(var_values) * 100),
                    "Sd": float(np.std(var_values) * 100),
                    "Skewness": float(skew(var_values)),
                    "Kurtosis": float(kurtosis(var_values)),
                }

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

        header = f"    {'Method':<30} {'Mean':>10} {'Sd':>10} {'Skew':>10} {'Kurt':>10}"
        output.append(header)
        output.append("  " + "─" * 68)

        for exp_name in experiment_names:
            if exp_name in exp_stats:
                stats = exp_stats[exp_name]
                row = f"    {exp_name:<30} {stats['Mean']:>10.3f} {stats['Sd']:>10.3f} {stats['Skewness']:>10.3f} {stats['Kurtosis']:>10.3f}"
                output.append(row)

    output.append("\n" + "═" * 72)
    return "\n".join(output)


def _generate_variable_organized_latex_table(stats_data: Dict[str, Dict[str, Dict[str, float]]], experiment_names: list) -> str:
    """Generate LaTeX table organized by variable with experiments as rows within each variable section."""
    n_experiments = len(experiment_names)

    latex_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Method} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Kurtosis} \\" + "\n"
        r"\midrule" + "\n"
    )

    for var_idx, (var_label, exp_stats) in enumerate(stats_data.items()):
        latex_code += f"\\multicolumn{{5}}{{l}}{{\\textbf{{{var_label}}}}} \\\\\n"

        for exp_name in experiment_names:
            if exp_name in exp_stats:
                stats = exp_stats[exp_name]
                exp_display = exp_name.replace("_", r"\_")
                latex_code += f"\\quad {exp_display} & {stats['Mean']:.3f} & {stats['Sd']:.3f} & {stats['Skewness']:.3f} & {stats['Kurtosis']:.3f} \\\\\n"

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

    stats_labels = ["Mean", "Sd", "Skewness", "Kurtosis"]

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
                    elif stat_label == "Kurtosis":
                        value = float(kurtosis(var_values))
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

        if any(keyword in str(index) for keyword in ["Kurtosis"]):
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
    latex_code += (
        r"\textit{Note: Values show percentage deviations from deterministic steady state.}"
        + "\n"
    )

    return latex_code
