"""
General Analysis Visualization Module

This module provides general plotting functions for visualizing analysis results from trained
neural network policies. These functions are model-agnostic and work with any analysis
results using labels from econ_model.get_analysis_variables().

Standard analysis data structures:
    - analysis_variables_data: {experiment_name: {var_label: array}}
    - gir_data: {experiment_name: {state_name: {"gir_analysis_variables": {var_label: array}, "state_idx": int}}}

For model-specific plots, see the plots.py file in the model directory.
"""

import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Plot styling configuration
sns.set_style("whitegrid")
palette = "dark"
sns.set_palette(palette)
colors = sns.color_palette(palette, 10)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

# Set font family and sizes globally
plt.rc("font", family="sans-serif", size=SMALL_SIZE)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

plt.rc("axes", titlesize=LARGE_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=LARGE_SIZE)

plt.rc("mathtext", fontset="dejavusans")


def plot_ergodic_histograms(
    analysis_variables_data: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 10),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality histograms of ergodic distributions for analysis variables.

    Parameters:
    -----------
    analysis_variables_data : dict
        Dictionary where keys are experiment names and values are dictionaries mapping
        variable labels to arrays of values: {exp_name: {var_label: array}}
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, no analysis name is added.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    list of (fig, ax) tuples for each analysis variable
    """
    # Get experiment names
    experiment_names = list(analysis_variables_data.keys())
    n_experiments = len(experiment_names)

    # Extract variable labels from first experiment
    first_exp = experiment_names[0]
    var_labels = list(analysis_variables_data[first_exp].keys())

    # Use colors from the global palette
    plot_colors = colors[:n_experiments]

    # Create safe file names from labels
    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    var_filenames = [make_safe_filename(label) for label in var_labels]

    figures = []

    for var_label, var_filename in zip(var_labels, var_filenames):
        # Extract data for this analysis variable across all experiments and convert to percentages
        var_data = {}
        for exp_name in experiment_names:
            var_data[exp_name] = analysis_variables_data[exp_name][var_label] * 100

        # Use fixed range from -10 to 10 (percentages)
        bin_range = (-10, 10)

        # Create bins using the fixed range
        bins = np.linspace(bin_range[0], bin_range[1], 31)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=display_dpi)

        # Plot histogram for each experiment
        for i, exp_name in enumerate(experiment_names):
            # Calculate histogram
            counts, _ = np.histogram(var_data[exp_name], bins=bins)
            freqs = counts / len(var_data[exp_name])

            # Plot the frequency line
            ax.plot(bin_centers, freqs, label=exp_name, color=plot_colors[i], linewidth=2, alpha=0.9)

        # Add vertical line at deterministic steady state (x=0)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Deterministic SS", alpha=0.7)

        # Styling
        ax.set_xlabel(
            f"{var_label} (% deviations from deterministic SS)",
            fontweight="bold",
            fontsize=MEDIUM_SIZE,
        )
        ax.set_ylabel("Frequency", fontweight="bold", fontsize=MEDIUM_SIZE)

        # Legend
        ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

        # Grid
        ax.grid(True, alpha=0.3)

        # Apply consistent styling
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

        # Adjust layout
        plt.tight_layout()

        # Save if directory provided
        if save_dir:
            # Create filename with analysis name if provided
            if analysis_name:
                filename = f"Histogram_{var_filename}_{analysis_name}.png"
            else:
                filename = f"Histogram_{var_filename}_comparative.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

        figures.append((fig, ax))

    plt.show()

    return figures


def plot_gir_responses(
    gir_data: Dict[str, Any],
    states_to_plot: Optional[list] = None,
    variables_to_plot: Optional[list] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of Generalized Impulse Responses over time.
    Each state gets its own separate plot for each analysis variable.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from analysis
        Structure: {experiment_name: {state_name: {"gir_analysis_variables": {var_label: array}, "state_idx": int}}}
    states_to_plot : list, optional
        Which states to plot. If None, plots all states.
    variables_to_plot : list, optional
        Which variable labels to plot. If None, plots all variables.
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    list of (fig, ax) tuples for each state-analysis variable combination
    """
    # Get experiment names
    experiment_names = list(gir_data.keys())
    n_experiments = len(experiment_names)

    # Get first experiment to determine states and variables
    first_experiment = experiment_names[0]
    first_exp_data = gir_data[first_experiment]

    # Get state names
    all_state_names = list(first_exp_data.keys())
    if states_to_plot is None:
        # Default to all states
        states_to_plot = all_state_names
    else:
        # Filter to requested states
        states_to_plot = [s for s in states_to_plot if s in all_state_names]

    # Get variable labels from first state's GIR data
    first_state = states_to_plot[0]
    gir_vars_dict = first_exp_data[first_state]["gir_analysis_variables"]
    all_var_labels = list(gir_vars_dict.keys())

    if variables_to_plot is None:
        variables_to_plot = all_var_labels
    else:
        # Filter to requested variables
        variables_to_plot = [v for v in variables_to_plot if v in all_var_labels]

    # Get time length from first variable
    first_var = variables_to_plot[0]
    time_length = len(gir_vars_dict[first_var])
    time_periods = np.arange(time_length)

    # Use colors from the global palette
    plot_colors = colors

    # Create safe file names from labels
    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    figures = []

    # Loop through each state first, then each analysis variable
    for state_name in states_to_plot:
        for var_label in variables_to_plot:
            # Create figure for this state-analysis variable combination
            fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

            # Plot for each experiment (if multiple)
            for j, exp_name in enumerate(experiment_names):
                gir_analysis_variables = gir_data[exp_name][state_name]["gir_analysis_variables"]

                # Convert to percentages
                response_pct = gir_analysis_variables[var_label] * 100

                # Create label
                if n_experiments > 1:
                    label = exp_name
                    color = plot_colors[j % len(plot_colors)]
                    linestyle = "-" if j == 0 else "--"
                else:
                    label = f"{state_name} Response"
                    color = plot_colors[0]
                    linestyle = "-"

                # Plot the impulse response
                ax.plot(
                    time_periods, response_pct, label=label, color=color, linewidth=2, linestyle=linestyle, alpha=0.8
                )

            # Add horizontal line at zero
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

            # Styling
            ax.set_xlabel("Time Periods", fontweight="bold", fontsize=MEDIUM_SIZE)
            ax.set_ylabel(f"{var_label} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

            # Legend
            if n_experiments > 1:
                ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=SMALL_SIZE)

            # Grid
            ax.grid(True, alpha=0.3)

            # Apply consistent styling
            ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

            # Set x-axis to start from 0
            ax.set_xlim(0, time_length - 1)

            # Adjust layout
            plt.tight_layout()

            # Save if directory provided
            if save_dir:
                # Create filename with state and analysis variable names
                safe_state_name = make_safe_filename(state_name)
                safe_var_name = make_safe_filename(var_label)
                if analysis_name:
                    filename = f"GIR_{safe_var_name}_{safe_state_name}_{analysis_name}.png"
                else:
                    filename = f"GIR_{safe_var_name}_{safe_state_name}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

            figures.append((fig, ax))

    plt.show()

    return figures


def plot_combined_impulse_responses(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sectors_to_plot: list,
    sector_labels: list,
    variables_to_plot: Optional[list] = None,
    shock_sizes_to_plot: list = [20],
    figsize: Tuple[float, float] = (14, 10),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
):
    """
    Create combined plots showing GIR responses alongside MATLAB perfect foresight and loglinear IRs.
    Shows both positive and negative shocks on the same graph for each shock size.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from JAX analysis
        Structure: {experiment_name: {state_name: {"gir_analysis_variables": {var_label: array}, "state_idx": int}}}
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs
    sectors_to_plot : list
        List of sector indices (0-based) to plot
    sector_labels : list
        List of sector labels for display
    variables_to_plot : list, optional
        Which variable labels to plot. If None, plots all variables.
    shock_sizes_to_plot : list
        List of shock sizes to include in plots (e.g., [5, 10, 20])
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.
    max_periods : int
        Maximum number of periods to plot

    Returns:
    --------
    list of (fig, ax) tuples for each sector-variable-shock_size combination
    """
    from DEQN.analysis.matlab_irs import get_matlab_ir_for_analysis_variable

    experiment_names = list(gir_data.keys())
    n_experiments = len(experiment_names)

    first_experiment = experiment_names[0]
    first_exp_data = gir_data[first_experiment]

    all_state_names = list(first_exp_data.keys())

    first_state = all_state_names[0]
    gir_vars_dict = first_exp_data[first_state]["gir_analysis_variables"]
    all_var_labels = list(gir_vars_dict.keys())

    if variables_to_plot is None:
        variables_to_plot = all_var_labels
    else:
        variables_to_plot = [v for v in variables_to_plot if v in all_var_labels]

    time_periods = np.arange(max_periods)

    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    figures = []

    n_sectors = len(sector_labels)

    for sector_idx in sectors_to_plot:
        sector_label = sector_labels[sector_idx] if sector_idx < len(sector_labels) else f"Sector {sector_idx + 1}"

        state_name = None
        for candidate in all_state_names:
            gir_info = first_exp_data[candidate]
            candidate_state_idx = gir_info.get("state_idx")
            if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors + sector_idx:
                state_name = candidate
                break

        if state_name is None:
            for possible_idx in [sector_idx, n_sectors + sector_idx]:
                possible_name = f"state_{possible_idx}"
                if possible_name in all_state_names:
                    state_name = possible_name
                    break

        if state_name is None:
            print(f"Warning: No GIR data found for sector {sector_idx}")
            print(f"  Available states: {all_state_names}")
            continue

        for var_label in variables_to_plot:
            for shock_size in shock_sizes_to_plot:
                fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

                pos_key = f"pos_{shock_size}"
                neg_key = f"neg_{shock_size}"

                matlab_irs_pos = get_matlab_ir_for_analysis_variable(
                    matlab_ir_data, sector_idx, var_label, max_periods
                )
                matlab_irs_neg = get_matlab_ir_for_analysis_variable(
                    matlab_ir_data, sector_idx, var_label, max_periods
                )

                if matlab_irs_pos and pos_key in matlab_irs_pos:
                    pos_loglin = matlab_irs_pos[pos_key]["loglin"][:max_periods] * 100
                    pos_determ = matlab_irs_pos[pos_key]["determ"][:max_periods] * 100
                    t_loglin = np.arange(len(pos_loglin))
                    t_determ = np.arange(len(pos_determ))

                    ax.plot(
                        t_loglin,
                        pos_loglin,
                        label=f"Loglinear (+{shock_size}%)",
                        color=colors[4],
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                    )
                    ax.plot(
                        t_determ,
                        pos_determ,
                        label=f"Perfect Foresight (+{shock_size}%)",
                        color=colors[2],
                        linewidth=2,
                        linestyle="-.",
                        alpha=0.8,
                    )

                if matlab_irs_neg and neg_key in matlab_irs_neg:
                    neg_loglin = matlab_irs_neg[neg_key]["loglin"][:max_periods] * 100
                    neg_determ = matlab_irs_neg[neg_key]["determ"][:max_periods] * 100
                    t_loglin = np.arange(len(neg_loglin))
                    t_determ = np.arange(len(neg_determ))

                    ax.plot(
                        t_loglin,
                        neg_loglin,
                        label=f"Loglinear (-{shock_size}%)",
                        color=colors[5],
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                    )
                    ax.plot(
                        t_determ,
                        neg_determ,
                        label=f"Perfect Foresight (-{shock_size}%)",
                        color=colors[3],
                        linewidth=2,
                        linestyle="-.",
                        alpha=0.8,
                    )

                for j, exp_name in enumerate(experiment_names):
                    if state_name in gir_data[exp_name]:
                        gir_analysis_variables = gir_data[exp_name][state_name]["gir_analysis_variables"]
                        if var_label in gir_analysis_variables:
                            response_pct = gir_analysis_variables[var_label][:max_periods] * 100

                            label = f"GIR ({exp_name})"
                            color = colors[j % len(colors)]

                            ax.plot(
                                time_periods[: len(response_pct)],
                                response_pct,
                                label=label,
                                color=color,
                                linewidth=2.5,
                                linestyle="-",
                                alpha=0.9,
                            )

                ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

                ax.set_xlabel("Time Periods", fontweight="bold", fontsize=MEDIUM_SIZE)
                ax.set_ylabel(f"{var_label} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)
                ax.set_title(
                    f"{sector_label}: {var_label} Response ({shock_size}% shock)",
                    fontweight="bold",
                    fontsize=LARGE_SIZE,
                )

                ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=SMALL_SIZE - 1, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
                ax.set_xlim(0, max_periods - 1)

                plt.tight_layout()

                if save_dir:
                    safe_sector = make_safe_filename(sector_label)
                    safe_var = make_safe_filename(var_label)
                    if analysis_name:
                        filename = f"CombinedIR_{safe_var}_{safe_sector}_shock{shock_size}_{analysis_name}.png"
                    else:
                        filename = f"CombinedIR_{safe_var}_{safe_sector}_shock{shock_size}.png"
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

                figures.append((fig, ax))

    plt.show()

    return figures


def plot_ir_comparison_panel(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sector_idx: int,
    sector_label: str,
    variables_to_plot: list,
    shock_sizes: list = [5, 10, 20],
    figsize: Tuple[float, float] = (18, 12),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
):
    """
    Create a panel figure showing all shock sizes for a single sector.
    Each row is a variable, each column is a shock size, with pos/neg on same plot.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from JAX analysis
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs
    sector_idx : int
        Sector index (0-based) to plot
    sector_label : str
        Label for the sector
    variables_to_plot : list
        List of variable labels to plot
    shock_sizes : list
        List of shock sizes (e.g., [5, 10, 20])
    figsize : tuple
        Figure size
    save_dir : str, optional
        Directory to save the figure
    analysis_name : str, optional
        Name for the analysis
    display_dpi : int
        DPI for display
    max_periods : int
        Maximum periods to plot

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    from DEQN.analysis.matlab_irs import get_matlab_ir_for_analysis_variable

    n_vars = len(variables_to_plot)
    n_sizes = len(shock_sizes)

    fig, axes = plt.subplots(n_vars, n_sizes, figsize=figsize, dpi=display_dpi, squeeze=False)

    experiment_names = list(gir_data.keys())
    first_experiment = experiment_names[0]
    first_exp_data = gir_data[first_experiment]
    all_state_names = list(first_exp_data.keys())

    n_sectors_in_model = 37

    state_name = None
    for candidate in all_state_names:
        gir_info = first_exp_data[candidate]
        candidate_state_idx = gir_info.get("state_idx")
        if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors_in_model + sector_idx:
            state_name = candidate
            break

    if state_name is None:
        for possible_idx in [sector_idx, n_sectors_in_model + sector_idx]:
            possible_name = f"state_{possible_idx}"
            if possible_name in all_state_names:
                state_name = possible_name
                break

    if state_name is None and all_state_names:
        print(f"Warning: No exact match for sector {sector_idx}, using first available state")
        state_name = all_state_names[0]

    time_periods = np.arange(max_periods)

    for i, var_label in enumerate(variables_to_plot):
        for j, shock_size in enumerate(shock_sizes):
            ax = axes[i, j]

            pos_key = f"pos_{shock_size}"
            neg_key = f"neg_{shock_size}"

            matlab_irs = get_matlab_ir_for_analysis_variable(matlab_ir_data, sector_idx, var_label, max_periods)

            if matlab_irs:
                if pos_key in matlab_irs:
                    pos_loglin = matlab_irs[pos_key]["loglin"][:max_periods] * 100
                    pos_determ = matlab_irs[pos_key]["determ"][:max_periods] * 100
                    ax.plot(
                        np.arange(len(pos_loglin)), pos_loglin, color=colors[4], linewidth=1.5, linestyle="--", alpha=0.7
                    )
                    ax.plot(
                        np.arange(len(pos_determ)),
                        pos_determ,
                        color=colors[2],
                        linewidth=1.5,
                        linestyle="-.",
                        alpha=0.7,
                    )

                if neg_key in matlab_irs:
                    neg_loglin = matlab_irs[neg_key]["loglin"][:max_periods] * 100
                    neg_determ = matlab_irs[neg_key]["determ"][:max_periods] * 100
                    ax.plot(
                        np.arange(len(neg_loglin)), neg_loglin, color=colors[5], linewidth=1.5, linestyle="--", alpha=0.7
                    )
                    ax.plot(
                        np.arange(len(neg_determ)),
                        neg_determ,
                        color=colors[3],
                        linewidth=1.5,
                        linestyle="-.",
                        alpha=0.7,
                    )

            for k, exp_name in enumerate(experiment_names):
                if state_name in gir_data[exp_name]:
                    gir_vars = gir_data[exp_name][state_name]["gir_analysis_variables"]
                    if var_label in gir_vars:
                        response = gir_vars[var_label][:max_periods] * 100
                        ax.plot(
                            time_periods[: len(response)],
                            response,
                            color=colors[k % len(colors)],
                            linewidth=2,
                            alpha=0.9,
                        )

            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.grid(True, alpha=0.2)

            if i == 0:
                ax.set_title(f"±{shock_size}% shock", fontweight="bold", fontsize=MEDIUM_SIZE)
            if j == 0:
                ax.set_ylabel(var_label, fontweight="bold", fontsize=SMALL_SIZE)
            if i == n_vars - 1:
                ax.set_xlabel("Periods", fontsize=SMALL_SIZE)

            ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE - 2)
            ax.set_xlim(0, max_periods - 1)

    fig.suptitle(f"{sector_label}: Impulse Response Comparison", fontweight="bold", fontsize=LARGE_SIZE, y=1.02)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors[4], linewidth=1.5, linestyle="--", label="Loglinear (+)"),
        Line2D([0], [0], color=colors[5], linewidth=1.5, linestyle="--", label="Loglinear (-)"),
        Line2D([0], [0], color=colors[2], linewidth=1.5, linestyle="-.", label="Perfect Foresight (+)"),
        Line2D([0], [0], color=colors[3], linewidth=1.5, linestyle="-.", label="Perfect Foresight (-)"),
    ]

    for k, exp_name in enumerate(experiment_names):
        legend_elements.append(Line2D([0], [0], color=colors[k % len(colors)], linewidth=2, label=f"GIR ({exp_name})"))

    fig.legend(
        handles=legend_elements, loc="lower center", ncol=len(legend_elements), fontsize=SMALL_SIZE - 1, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()

    if save_dir:

        def make_safe_filename(label):
            return label.replace(" ", "_").replace(".", "").replace("/", "_")

        safe_sector = make_safe_filename(sector_label)
        if analysis_name:
            filename = f"IRPanel_{safe_sector}_{analysis_name}.png"
        else:
            filename = f"IRPanel_{safe_sector}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, axes


def plot_sector_ir_by_shock_size(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sector_idx: int,
    sector_label: str,
    variable_to_plot: str = "Agg. Consumption",
    shock_sizes: list = [5, 10, 20],
    figsize: Tuple[float, float] = (16, 5),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
    n_sectors: int = 37,
):
    """
    Create a figure with subplots for each shock size, showing positive and negative IRs.
    Each subplot shows positive IRs in upper quadrant and negative IRs in lower quadrant.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from JAX analysis
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs
    sector_idx : int
        Sector index (0-based) to plot
    sector_label : str
        Label for the sector
    variable_to_plot : str
        Analysis variable to plot (default: "Agg. Consumption")
    shock_sizes : list
        List of shock sizes (e.g., [5, 10, 20])
    figsize : tuple
        Figure size
    save_dir : str, optional
        Directory to save the figure
    analysis_name : str, optional
        Name for the analysis
    display_dpi : int
        DPI for display
    max_periods : int
        Maximum periods to plot
    n_sectors : int
        Number of sectors in the model

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    from DEQN.analysis.matlab_irs import get_matlab_ir_for_analysis_variable

    n_sizes = len(shock_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=figsize, dpi=display_dpi, sharey=True)

    if n_sizes == 1:
        axes = [axes]

    experiment_names = list(gir_data.keys()) if gir_data else []
    first_exp_data = gir_data[experiment_names[0]] if experiment_names else {}
    all_state_names = list(first_exp_data.keys()) if first_exp_data else []

    state_name = None
    for candidate in all_state_names:
        gir_info = first_exp_data[candidate]
        candidate_state_idx = gir_info.get("state_idx")
        if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors + sector_idx:
            state_name = candidate
            break

    if state_name is None:
        for possible_idx in [sector_idx, n_sectors + sector_idx]:
            possible_name = f"state_{possible_idx}"
            if possible_name in all_state_names:
                state_name = possible_name
                break

    time_periods = np.arange(max_periods)

    for j, shock_size in enumerate(shock_sizes):
        ax = axes[j]

        pos_key = f"pos_{shock_size}"
        neg_key = f"neg_{shock_size}"

        matlab_irs = get_matlab_ir_for_analysis_variable(matlab_ir_data, sector_idx, variable_to_plot, max_periods)

        if matlab_irs:
            if pos_key in matlab_irs:
                pos_loglin = matlab_irs[pos_key]["loglin"][:max_periods] * 100
                pos_determ = matlab_irs[pos_key]["determ"][:max_periods] * 100
                ax.plot(
                    np.arange(len(pos_loglin)),
                    pos_loglin,
                    color=colors[4],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.8,
                    label="Loglinear" if j == 0 else None,
                )
                ax.plot(
                    np.arange(len(pos_determ)),
                    pos_determ,
                    color=colors[2],
                    linewidth=1.5,
                    linestyle="-.",
                    alpha=0.8,
                    label="Perfect Foresight" if j == 0 else None,
                )

            if neg_key in matlab_irs:
                neg_loglin = matlab_irs[neg_key]["loglin"][:max_periods] * 100
                neg_determ = matlab_irs[neg_key]["determ"][:max_periods] * 100
                ax.plot(
                    np.arange(len(neg_loglin)),
                    neg_loglin,
                    color=colors[4],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.8,
                )
                ax.plot(
                    np.arange(len(neg_determ)),
                    neg_determ,
                    color=colors[2],
                    linewidth=1.5,
                    linestyle="-.",
                    alpha=0.8,
                )

        for k, exp_name in enumerate(experiment_names):
            if state_name and state_name in gir_data[exp_name]:
                state_gir_data = gir_data[exp_name][state_name]

                if pos_key in state_gir_data:
                    gir_vars_pos = state_gir_data[pos_key].get("gir_analysis_variables", {})
                    if variable_to_plot in gir_vars_pos:
                        response_pos = gir_vars_pos[variable_to_plot][:max_periods] * 100
                        label = f"GIR ({exp_name})" if j == 0 else None
                        ax.plot(
                            time_periods[: len(response_pos)],
                            response_pos,
                            color=colors[k % len(colors)],
                            linewidth=2.5,
                            alpha=0.9,
                            label=label,
                        )

                if neg_key in state_gir_data:
                    gir_vars_neg = state_gir_data[neg_key].get("gir_analysis_variables", {})
                    if variable_to_plot in gir_vars_neg:
                        response_neg = gir_vars_neg[variable_to_plot][:max_periods] * 100
                        ax.plot(
                            time_periods[: len(response_neg)],
                            response_neg,
                            color=colors[k % len(colors)],
                            linewidth=2.5,
                            alpha=0.9,
                        )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"±{shock_size}% shock", fontweight="bold", fontsize=MEDIUM_SIZE)
        ax.set_xlabel("Periods", fontsize=SMALL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
        ax.set_xlim(0, max_periods - 1)

    axes[0].set_ylabel(f"{variable_to_plot} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

    fig.suptitle(
        f"{sector_label}: {variable_to_plot} Impulse Responses",
        fontweight="bold",
        fontsize=LARGE_SIZE,
        y=1.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=SMALL_SIZE,
            bbox_to_anchor=(0.5, -0.08),
        )

    plt.tight_layout()

    if save_dir:

        def make_safe_filename(label):
            return label.replace(" ", "_").replace(".", "").replace("/", "_")

        safe_sector = make_safe_filename(sector_label)
        safe_var = make_safe_filename(variable_to_plot)
        if analysis_name:
            filename = f"IR_{safe_var}_{safe_sector}_{analysis_name}.png"
        else:
            filename = f"IR_{safe_var}_{safe_sector}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
        print(f"      Saved: {filename}")

    return fig, axes


def plot_gir_heatmap(
    gir_data: Dict[str, Any],
    aggregate_idx: int = 0,
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Output",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility Welfare",
    ],
    time_slice: int = 10,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create a heatmap showing GIR responses across sectors at a specific time period.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from analysis
    aggregate_idx : int, optional
        Which aggregate variable to plot (index 0-6). Default is consumption.
    aggregate_labels : list, optional
        Labels for the aggregate variables
    time_slice : int, optional
        Which time period to show in the heatmap
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path
    analysis_name : str, optional
        Name of the analysis to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get experiment names (assuming single experiment for heatmap)
    experiment_names = list(gir_data.keys())
    first_experiment = experiment_names[0]
    exp_data = gir_data[first_experiment]

    # Get all sector names and their responses
    sector_names = list(exp_data.keys())
    n_sectors = len(sector_names)

    # Extract responses at the specified time slice
    responses = []
    sector_labels = []

    for sector_name in sector_names:
        gir_aggregates = exp_data[sector_name]["gir_aggregates"]
        # Convert to percentage and get specific aggregate at time slice
        response_pct = gir_aggregates[time_slice, aggregate_idx] * 100
        responses.append(response_pct)
        sector_labels.append(sector_name)

    # Create matrix for heatmap (single row)
    response_matrix = np.array(responses).reshape(1, -1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    # Create heatmap
    im = ax.imshow(response_matrix, cmap="RdBu_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(n_sectors))
    ax.set_xticklabels(sector_labels, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([f"Period {time_slice}"])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{aggregate_labels[aggregate_idx]} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

    # Apply consistent styling
    ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
        var_names = ["C_agg", "L_agg", "K_agg", "Y_agg", "M_agg", "I_agg", "Utility_welfare"]
        if analysis_name:
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".png"
            new_filename = f"GIR_heatmap_{var_names[aggregate_idx]}_period{time_slice}_{analysis_name}{ext}"
            final_save_path = save_path.replace(os.path.basename(save_path), new_filename)
        else:
            final_save_path = save_path.replace(
                os.path.basename(save_path), f"GIR_heatmap_{var_names[aggregate_idx]}_period{time_slice}.png"
            )
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax
