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
        r"\textbf{Variable} & \textbf{Mean (\%)} & \textbf{Sd (\%)} & \textbf{Skewness} & \textbf{Kurtosis} \\" + "\n"
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
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: Mean and Sd are reported as percentage deviations from deterministic steady state.}" + "\n"
    )

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
    stats_labels = ["Mean (%)", "Sd (%)", "Skewness", "Kurtosis"]

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
        + r"\textbf{Aggregate (Stochastic SS, \% dev.)}"
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
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: Values show percentage deviations from deterministic steady state. For example, 0.1 = 0.1\%.}"
        + "\n"
    )

    return latex_code


def create_grid_test_summary_table(
    grid_test_results: Dict[str, Any],
    save_path: Optional[str] = None,
    test_name: Optional[str] = None,
) -> str:
    """
    Create a LaTeX table summarizing key grid test diagnostics across experiments.

    Parameters:
    -----------
    grid_test_results : dict
        Dictionary containing grid test results from run_seed_length_grid
    save_path : str, optional
        If provided, save the LaTeX table to this path. If test_name is provided, it will
        modify the filename to include the test name.
    test_name : str, optional
        Name of the test to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    # Initialize data storage for the table
    table_data = []

    # Process each experiment
    for exp_name, grid_data in grid_test_results.items():
        # Extract SD vs T slope diagnostics
        slopes = grid_data.get("sd_vs_T_slope", {})
        state_slope = slopes.get("state_logsd_logT_slope", float("nan"))
        policies_slope = slopes.get("policies_logsd_logT_slope", float("nan"))
        agg_slope = slopes.get("aggregates_logsd_logT_slope", float("nan"))

        # Get diagnostics from the longest episode with first burn-in fraction
        lengths = sorted([k for k in grid_data.keys() if isinstance(k, (int, float))])
        if lengths:
            max_length = max(lengths)
            burnin_fracs = list(grid_data[max_length].keys())
            if burnin_fracs:
                b0 = burnin_fracs[0]
                data = grid_data[max_length][b0]

                # Average IACT across aggregates
                avg_iact = np.mean(data.get("avg_iact_aggregates", [1.0]))

                # Average OOD fraction (threshold 3.0)
                ood_dict = data.get("avg_ood_fraction", {})
                avg_ood = ood_dict.get(3.0, 0.0)

                # Average absolute trend slope across aggregates
                trend_slopes = data.get("avg_trend_slope_aggregates", [0.0])
                avg_trend = np.mean([abs(slope) for slope in trend_slopes])

                # Average shock diagnostics
                shock_mean_norm = data.get("avg_shock_mean_norm", 0.0)
                shock_cov_diff = data.get("avg_shock_cov_diff", 0.0)
        else:
            avg_iact = avg_ood = avg_trend = shock_mean_norm = shock_cov_diff = float("nan")

        table_data.append(
            {
                "Experiment": exp_name,
                "State Slope": state_slope,
                "Policies Slope": policies_slope,
                "Aggregates Slope": agg_slope,
                "Avg IACT": avg_iact,
                "OOD Frac (3σ)": avg_ood,
                "Avg |Trend|": avg_trend,
                "Shock Mean": shock_mean_norm,
                "Shock Cov": shock_cov_diff,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)
    df.set_index("Experiment", inplace=True)

    # Generate LaTeX table
    latex_code = _generate_grid_test_latex_table(df)

    # Save if path provided
    if save_path:
        # Modify filename to include test name if provided
        if test_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"grid_test_summary_{test_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    return latex_code


def _generate_grid_test_latex_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table code for grid test summary.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with grid test summary statistics

    Returns:
    --------
    str : LaTeX table code
    """
    # LaTeX preamble
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l *{8}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Experiment} & \textbf{State Slope} & \textbf{Policy Slope} & \textbf{Agg Slope} & "
        r"\textbf{Avg IACT} & \textbf{OOD 3$\sigma$} & \textbf{Avg |Trend|} & "
        r"\textbf{Shock Mean} & \textbf{Shock Cov} \\" + "\n"
        r"\midrule" + "\n"
    )

    # Add data rows
    for index, row in df.iterrows():
        # Format numbers appropriately
        formatted_values = []
        for i, value in enumerate(row):
            if np.isnan(value):
                formatted_values.append("—")
            elif i < 3:  # Slopes
                formatted_values.append(f"{value:.3f}")
            elif i == 3:  # IACT
                formatted_values.append(f"{value:.2f}")
            elif i == 4:  # OOD fraction
                formatted_values.append(f"{value:.4f}")
            else:  # Others
                formatted_values.append(f"{value:.3e}")

        latex_code += r"\textbf{" + str(index) + r"} & " + " & ".join(formatted_values) + r" \\" + "\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: Expected slope for pure sampling error is -0.5. "
        r"IACT > 1 indicates autocorrelation. OOD measures fraction |z| > 3.}" + "\n"
    )

    return latex_code


def create_grid_test_detailed_table(
    grid_test_results: Dict[str, Any],
    save_path: Optional[str] = None,
    test_name: Optional[str] = None,
) -> str:
    """
    Create a detailed LaTeX table showing grid test results across different episode lengths.

    Parameters:
    -----------
    grid_test_results : dict
        Dictionary containing grid test results from run_seed_length_grid
    save_path : str, optional
        If provided, save the LaTeX table to this path. If test_name is provided, it will
        modify the filename to include the test name.
    test_name : str, optional
        Name of the test to include in the filename. If None, uses default filename.

    Returns:
    --------
    str : The LaTeX table code
    """
    # Initialize data storage for the table
    table_data = []

    # Process each experiment
    for exp_name, grid_data in grid_test_results.items():
        # Get episode lengths
        lengths = sorted([k for k in grid_data.keys() if isinstance(k, (int, float))])

        for T in lengths:
            burnin_fracs = list(grid_data[T].keys())
            if burnin_fracs:
                b0 = burnin_fracs[0]  # Use first burn-in fraction
                data = grid_data[T][b0]

                # Extract key metrics
                sd_state = data.get("sd_state_mean", 0.0)
                sd_policies = data.get("sd_policies_mean", 0.0)
                sd_agg = data.get("sd_aggregates_mean", 0.0)

                # IACT for first aggregate (consumption)
                iact_aggs = data.get("avg_iact_aggregates", [1.0])
                iact_c = iact_aggs[0] if len(iact_aggs) > 0 else 1.0

                # OOD fraction for 3σ threshold
                ood_dict = data.get("avg_ood_fraction", {})
                ood_3sigma = ood_dict.get(3.0, 0.0)

                # Trend slope for first aggregate
                trend_aggs = data.get("avg_trend_slope_aggregates", [0.0])
                trend_c = abs(trend_aggs[0]) if len(trend_aggs) > 0 else 0.0

                table_data.append(
                    {
                        "Experiment": exp_name,
                        "Length T": int(T),
                        "SD State": sd_state,
                        "SD Policies": sd_policies,
                        "SD Aggregates": sd_agg,
                        "IACT (C)": iact_c,
                        "OOD 3σ": ood_3sigma,
                        "|Trend| (C)": trend_c,
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Generate LaTeX table
    latex_code = _generate_grid_test_detailed_latex_table(df)

    # Save if path provided
    if save_path:
        # Modify filename to include test name if provided
        if test_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".tex"
            new_filename = f"grid_test_detailed_{test_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path

        with open(final_save_path, "w") as file:
            file.write(latex_code)

    return latex_code


def _generate_grid_test_detailed_latex_table(df: pd.DataFrame) -> str:
    """
    Generate detailed LaTeX table code for grid test results.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with detailed grid test results

    Returns:
    --------
    str : LaTeX table code
    """
    # LaTeX preamble
    latex_code = (
        r"\begin{tabularx}{\textwidth}{l c *{6}{X}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Experiment} & \textbf{Length T} & \textbf{SD State} & \textbf{SD Policies} & "
        r"\textbf{SD Agg} & \textbf{IACT (C)} & \textbf{OOD 3$\sigma$} & \textbf{|Trend| (C)} \\" + "\n"
        r"\midrule" + "\n"
    )

    # Add data rows grouped by experiment
    current_exp = None
    for index, row in df.iterrows():
        exp_name = row["Experiment"]

        # Add separator between experiments
        if current_exp is not None and exp_name != current_exp:
            latex_code += r"\cmidrule(lr){1-8}" + "\n"
        current_exp = exp_name

        # Format values
        formatted_values = [
            str(int(row["Length T"])),
            f"{row['SD State']:.4e}",
            f"{row['SD Policies']:.4e}",
            f"{row['SD Aggregates']:.4e}",
            f"{row['IACT (C)']:.2f}",
            f"{row['OOD 3σ']:.4f}",
            f"{row['|Trend| (C)']:.3e}",
        ]

        latex_code += r"\textbf{" + exp_name + r"} & " + " & ".join(formatted_values) + r" \\" + "\n"

    # LaTeX closing
    latex_code += r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n"
    latex_code += r"\\" + "\n"
    latex_code += (
        r"\textit{Note: SD = cross-seed standard deviation. IACT = Integrated Autocorrelation Time. "
        r"OOD = Out-of-Distribution fraction. All results for consumption (C) aggregate.}" + "\n"
    )

    return latex_code
