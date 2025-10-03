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
    Create a LaTeX table with descriptive statistics for analysis variables across experiments.

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
    table_data = []

    for exp_name, analysis_vars_dict in analysis_variables_data.items():
        for var_label, var_values in analysis_vars_dict.items():
            mean_val = np.mean(var_values) * 100
            std_val = np.std(var_values) * 100
            skew_val = skew(var_values)
            kurt_val = kurtosis(var_values)

            row_label = f"{var_label} ({exp_name})"

            table_data.append(
                {"Variable": row_label, "Mean": mean_val, "Sd": std_val, "Skewness": skew_val, "Kurtosis": kurt_val}
            )

    df = pd.DataFrame(table_data)
    df.set_index("Variable", inplace=True)

    latex_code = _generate_latex_table(df)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"descriptive_stats_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS TABLE")
    print("=" * 80)
    print(latex_code)
    print("=" * 80 + "\n")

    return latex_code


def _generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table code from descriptive statistics DataFrame."""
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Variable} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Kurtosis} \\" + "\n"
        r"\midrule" + "\n"
        r"\cmidrule(lr){1-1} \cmidrule(lr){2-5}" + "\n"
    )

    for index, row in df.iterrows():
        formatted_row = [f"{float(value):.2f}" for value in row]
        latex_code += r"\textbf{" + str(index) + r"} & " + " & ".join(formatted_row) + r" \\" + "\n"

        if any(keyword in str(index) for keyword in ["Utility"]):
            latex_code += r"\cmidrule(lr){1-5}" + "\n"

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
    var_labels = list(analysis_variables_data[first_exp].keys())

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
        formatted_row = [f"{float(value):.2f}" if not np.isnan(value) else "—" for value in row]
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
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    latex_code = _generate_welfare_latex_table(welfare_data)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"welfare_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    print("\n" + "=" * 80)
    print("WELFARE TABLE")
    print("=" * 80)
    print(latex_code)
    print("=" * 80 + "\n")

    return latex_code


def _generate_welfare_latex_table(welfare_data: Dict[str, float]) -> str:
    """Generate LaTeX table code for welfare costs."""
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l X}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Experiment} & \textbf{Welfare Loss (\%)} \\" + "\n"
        r"\midrule" + "\n"
    )

    for exp_name, welfare_loss in welfare_data.items():
        latex_code += f"\\textbf{{{exp_name}}} & {welfare_loss:.4f} \\\\\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"

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
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    latex_code = _generate_stochastic_ss_latex_table(stochastic_ss_data)

    if save_path:
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"stochastic_ss_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    print("\n" + "=" * 80)
    print("STOCHASTIC STEADY STATE TABLE")
    print("=" * 80)
    print(latex_code)
    print("=" * 80 + "\n")

    return latex_code


def _generate_stochastic_ss_latex_table(stochastic_ss_data: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table code for stochastic steady state analysis variables."""
    experiment_names = list(stochastic_ss_data.keys())
    n_experiments = len(experiment_names)

    first_exp = experiment_names[0]
    var_labels = list(stochastic_ss_data[first_exp].keys())

    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Analysis Variable (Stochastic SS, \% dev.)}"
    )

    for exp_name in experiment_names:
        latex_code += f" & \\textbf{{{exp_name}}}"

    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    for var_label in var_labels:
        latex_code += f"\\textbf{{{var_label}}}"

        for exp_name in experiment_names:
            ss_vars_dict = stochastic_ss_data[exp_name]
            if var_label in ss_vars_dict:
                value = float(ss_vars_dict[var_label])
                latex_code += f" & {value:.4f}"
            else:
                latex_code += " & —"

        latex_code += r" \\" + "\n"

    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: Values show percentage deviations from deterministic steady state. For example, 0.1 = 0.1\%.}"
        + "\n"
    )

    return latex_code
