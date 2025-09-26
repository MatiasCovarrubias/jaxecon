import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def create_descriptive_stats_table(
    aggregates_data: Dict[str, Any],
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Production",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility",
    ],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table with descriptive statistics for aggregate variables across experiments.

    Parameters:
    -----------
    aggregates_data : dict
        Dictionary where keys are experiment names and values are arrays of simulated aggregates
        Each array should have shape (n_periods, n_aggregates) where n_aggregates >= 7
    aggregate_labels : list, optional
        Labels for the aggregate variables
    save_path : str, optional
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    # Initialize data storage for the table
    table_data = []

    # Process each experiment
    for exp_name, simul_aggregates in aggregates_data.items():
        # Calculate descriptive statistics for each aggregate variable
        for agg_idx, agg_label in enumerate(aggregate_labels):
            if agg_idx < simul_aggregates.shape[1]:  # Check if aggregate exists
                agg_values = simul_aggregates[:, agg_idx]

                # Calculate statistics and convert to percentages
                mean_val = np.mean(agg_values) * 100
                std_val = np.std(agg_values) * 100
                skew_val = skew(agg_values)
                kurt_val = kurtosis(agg_values)

                # Create row label combining experiment and variable
                row_label = f"{agg_label} ({exp_name})"

                table_data.append(
                    {"Variable": row_label, "Mean": mean_val, "Sd": std_val, "Skewness": skew_val, "Kurtosis": kurt_val}
                )

    # Create DataFrame
    df = pd.DataFrame(table_data)
    df.set_index("Variable", inplace=True)

    # Generate LaTeX table
    latex_code = _generate_latex_table(df)

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
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

    return latex_code


def _generate_latex_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table code from descriptive statistics DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with descriptive statistics

    Returns:
    --------
    str : LaTeX table code
    """
    # LaTeX preamble
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l *{4}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Variable} & \textbf{Mean} & \textbf{Sd} & \textbf{Skewness} & \textbf{Kurtosis} \\" + "\n"
        r"\midrule" + "\n"
        r"\cmidrule(lr){1-1} \cmidrule(lr){2-5}" + "\n"
    )

    # Add data rows
    for index, row in df.iterrows():
        # Format numbers with two decimals
        formatted_row = [f"{float(value):.2f}" for value in row]
        latex_code += r"\textbf{" + str(index) + r"} & " + " & ".join(formatted_row) + r" \\" + "\n"

        # Add separator lines for different experiments (optional)
        # You can customize this logic based on your variable naming convention
        if any(keyword in str(index) for keyword in ["Utility"]):
            latex_code += r"\cmidrule(lr){1-5}" + "\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"

    return latex_code


def create_comparative_stats_table(
    aggregates_data: Dict[str, Any],
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Production",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility",
    ],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a comparative LaTeX table with descriptive statistics organized by variable
    with experiments as columns.

    Parameters:
    -----------
    aggregates_data : dict
        Dictionary where keys are experiment names and values are arrays of simulated aggregates
    aggregate_labels : list, optional
        Labels for the aggregate variables
    save_path : str, optional
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    experiment_names = list(aggregates_data.keys())

    # Create multi-index for statistics x experiments
    stats_labels = ["Mean", "Sd", "Skewness", "Kurtosis"]

    # Initialize data storage
    table_data = []

    # Process each aggregate variable
    for agg_idx, agg_label in enumerate(aggregate_labels):
        for stat_label in stats_labels:
            row_data: Dict[str, Any] = {"Variable": f"{agg_label} ({stat_label})"}

            # Calculate statistic for each experiment
            for exp_name in experiment_names:
                simul_aggregates = aggregates_data[exp_name]
                if agg_idx < simul_aggregates.shape[1]:
                    agg_values = simul_aggregates[:, agg_idx]

                    if stat_label == "Mean":
                        value = float(np.mean(agg_values) * 100)
                    elif stat_label == "Sd":
                        value = float(np.std(agg_values) * 100)
                    elif stat_label == "Skewness":
                        value = float(skew(agg_values))
                    elif stat_label == "Kurtosis":
                        value = float(kurtosis(agg_values))

                    row_data[exp_name] = value
                else:
                    row_data[exp_name] = float(np.nan)

            table_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(table_data)
    df.set_index("Variable", inplace=True)

    # Generate LaTeX table
    latex_code = _generate_comparative_latex_table(df, experiment_names)

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
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
    """
    Generate comparative LaTeX table code from descriptive statistics DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with descriptive statistics
    experiment_names : list
        List of experiment names for column headers

    Returns:
    --------
    str : LaTeX table code
    """
    n_experiments = len(experiment_names)

    # LaTeX preamble with dynamic column count
    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Variable}"
    )

    # Add experiment headers
    for exp_name in experiment_names:
        latex_code += f" & \\textbf{{{exp_name}}}"

    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    # Add data rows
    for index, row in df.iterrows():
        # Format numbers with two decimals
        formatted_row = [f"{float(value):.2f}" if not np.isnan(value) else "—" for value in row]
        latex_code += r"\textbf{" + str(index) + r"} & " + " & ".join(formatted_row) + r" \\" + "\n"

        # Add separator lines between variables
        if any(keyword in str(index) for keyword in ["Kurtosis"]):
            latex_code += r"\cmidrule(lr){1-" + str(n_experiments + 1) + "}\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"

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
    # Generate LaTeX table
    latex_code = _generate_welfare_latex_table(welfare_data)

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
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

    return latex_code


def _generate_welfare_latex_table(welfare_data: Dict[str, float]) -> str:
    """
    Generate LaTeX table code for welfare costs.

    Parameters:
    -----------
    welfare_data : dict
        Dictionary with experiment names as keys and welfare losses as values

    Returns:
    --------
    str : LaTeX table code
    """
    # LaTeX preamble
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l X}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Experiment} & \textbf{Welfare Loss (\%)} \\" + "\n"
        r"\midrule" + "\n"
    )

    # Add data rows
    for exp_name, welfare_loss in welfare_data.items():
        latex_code += f"\\textbf{{{exp_name}}} & {welfare_loss:.4f} \\\\\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"

    return latex_code


def create_stochastic_ss_table(
    stochastic_ss_data: Dict[str, list],
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Production",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility",
    ],
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table with stochastic steady state values for key aggregates across experiments.

    Parameters:
    -----------
    stochastic_ss_data : dict
        Dictionary where keys are experiment names and values are lists of stochastic steady state aggregates
    aggregate_labels : list, optional
        Labels for the aggregate variables
    save_path : str, optional
        If provided, save the LaTeX table to this path. If analysis_name is provided, it will
        modify the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    # Generate LaTeX table
    latex_code = _generate_stochastic_ss_latex_table(stochastic_ss_data, aggregate_labels)

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
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

    return latex_code


def _generate_stochastic_ss_latex_table(stochastic_ss_data: Dict[str, list], aggregate_labels: list) -> str:
    """
    Generate LaTeX table code for stochastic steady state aggregates.

    Parameters:
    -----------
    stochastic_ss_data : dict
        Dictionary with experiment names as keys and stochastic steady state aggregates as values
    aggregate_labels : list
        Labels for the aggregate variables

    Returns:
    --------
    str : LaTeX table code
    """
    experiment_names = list(stochastic_ss_data.keys())
    n_experiments = len(experiment_names)

    # LaTeX preamble with dynamic column count
    latex_code = (
        f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{n_experiments}}}{{X}}}}\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Aggregate}"
    )

    # Add experiment headers
    for exp_name in experiment_names:
        latex_code += f" & \\textbf{{{exp_name}}}"

    latex_code += r" \\" + "\n" + r"\midrule" + "\n"

    # Add data rows
    for agg_idx, agg_label in enumerate(aggregate_labels):
        latex_code += f"\\textbf{{{agg_label}}}"

        # Add values for each experiment
        for exp_name in experiment_names:
            ss_values = stochastic_ss_data[exp_name]
            if agg_idx < len(ss_values):
                value = ss_values[agg_idx]
                latex_code += f" & {value:.4f}"
            else:
                latex_code += " & —"

        latex_code += r" \\" + "\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"

    return latex_code
