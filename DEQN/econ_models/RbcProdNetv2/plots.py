import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Production parameters
gamma_q_default = 0.5  # share of intermediate in production downstream
sigma_q_default = 0.25  # default elast of substitution (only relevant for ces_agg=True)

# 2. Risk aversion
epsc_min1_default = 2

# stochastic process
pl = 0.5
Al = 0.5
Ah = 1.5

# Plot parameters
sns.set_style("whitegrid")  # Set seaborn style and color palette

palette = "dark"  # or try "colorblind", "bright", "dark", "pastel"
sns.set_palette(palette)
colors = sns.color_palette(palette, 10)
print(colors)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

# Set font family and sizes globally
plt.rc("font", family="sans-serif", size=SMALL_SIZE)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

plt.rc("axes", titlesize=LARGE_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=LARGE_SIZE)

# For math rendering - using DejaVu which is always available
plt.rc("mathtext", fontset="dejavusans")  # DejaVu math fonts


def plot_upstreamness(
    upstreamness_data: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
):
    """
    Create a publication-quality bar graph of upstreamness measures.

    Parameters:
    -----------
    upstreamness_data : dict
        Dictionary containing 'sectors', 'U_M', 'U_I', and 'U_simple' as returned by the upstreamness() method
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Use the global color palette
    plot_colors = colors[:3]  # Use first 3 colors from the seaborn palette
    # Extract data
    sectors = upstreamness_data["sectors"]
    U_M = upstreamness_data["U_M"]
    U_I = upstreamness_data["U_I"]
    U_simple = upstreamness_data["U_simple"]

    # Sort sectors by U_M upstreamness for better visualization
    sorted_indices = np.argsort(U_M)[::-1]  # Sort in descending order
    sorted_sectors = [sectors[i] for i in sorted_indices]
    sorted_U_M = U_M[sorted_indices]
    sorted_U_I = U_I[sorted_indices]
    sorted_U_simple = U_simple[sorted_indices]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Set bar width and positions
    bar_width = 0.25
    x = np.arange(len(sorted_sectors))

    # Create bars with colors from the global palette
    ax.bar(
        x - bar_width,
        sorted_U_M,
        bar_width,
        label="Intermediate Inputs (U$^M$)",
        color=plot_colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        sorted_U_I,
        bar_width,
        label="Investment Flows (U$^I$)",
        color=plot_colors[1],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + bar_width,
        sorted_U_simple,
        bar_width,
        label="Simple (Mout/Q)",
        color=plot_colors[2],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add sector labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha="right")

    # Set labels and title
    ax.set_xlabel("Sector", fontweight="bold")
    ax.set_ylabel("Upstreamness Measure", fontweight="bold")

    # Add legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    # Set title
    ax.set_title("Sector Upstreamness Measures", fontweight="bold", pad=20)

    # Adjust layout and save if requested
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_sectoral_capital_mean(
    analysis_results: Dict[str, Any],
    sector_labels: list,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
):
    """
    Create a publication-quality bar graph of mean sectoral capital across experiments.

    Sectors are automatically sorted in descending order by their capital values
    from the first experiment in the analysis results.

    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing experiment results with 'sectoral_capital_mean' for each experiment
    sector_labels : list
        List of sector labels (should match the number of sectors)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path. If analysis_name is provided, it will modify
        the filename to include the analysis name.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, uses default filename.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Extract data for plotting
    experiments = list(analysis_results.keys())
    n_experiments = len(experiments)
    n_sectors = len(sector_labels)

    # Get the capital data for each experiment
    capital_data = {}
    for exp_name in experiments:
        capital_data[exp_name] = np.array(analysis_results[exp_name]["sectoral_capital_mean"])

    # Sort sectors by capital values from the first experiment
    first_experiment = experiments[0]
    first_exp_capital = capital_data[first_experiment]
    sorted_indices = np.argsort(first_exp_capital)[::-1]  # Sort in descending order

    # Apply sorting to sector labels and all capital data
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]
    sorted_capital_data = {}
    for exp_name in experiments:
        sorted_capital_data[exp_name] = capital_data[exp_name][sorted_indices]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Set bar width and positions
    bar_width = 0.8 / n_experiments  # Adjust width based on number of experiments
    x = np.arange(n_sectors)

    # Use colors from the global palette
    plot_colors = colors[:n_experiments]

    # Create bars for each experiment
    for i, exp_name in enumerate(experiments):
        offset = (i - (n_experiments - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            sorted_capital_data[exp_name],
            bar_width,
            label=exp_name,
            color=plot_colors[i],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )

    # Add sector labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    # Consistent tick styling
    ax.tick_params(axis="both", which="major")

    # Set labels and title using predefined font sizes
    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Average Capital (Log Deviations from SS)", fontweight="bold", fontsize=MEDIUM_SIZE)

    # Add legend using predefined font size
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

    # Set title using predefined font size
    ax.set_title("Mean Sectoral Capital Across Experiments", fontweight="bold", pad=20, fontsize=LARGE_SIZE)

    # Add horizontal line at y=0 (steady state) - consistent with other plots
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    # Add grid consistent with other plots
    ax.grid(True, alpha=0.3)

    # Adjust layout and save if requested
    plt.tight_layout()
    if save_path:
        # Modify filename to include analysis name if provided
        if analysis_name:
            # Extract directory and extension
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".png"
            new_filename = f"sectoral_capital_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax


def plot_ergodic_histograms(
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
    figsize: Tuple[float, float] = (15, 10),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
):
    """
    Create publication-quality histograms of ergodic distributions for aggregate variables.

    Parameters:
    -----------
    aggregates_data : dict
        Dictionary where keys are experiment names and values are arrays of simulated aggregates
        Each array should have shape (n_periods, n_aggregates) where n_aggregates >= 7
    aggregate_labels : list, optional
        Labels for the aggregate variables (first 7 will be used: C, L, K, Y, M, I, Utility)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, no analysis name is added.

    Returns:
    --------
    list of (fig, ax) tuples for each aggregate variable
    """
    # Get experiment names and aggregate variable names
    experiment_names = list(aggregates_data.keys())
    n_experiments = len(experiment_names)
    n_aggregates = min(7, len(aggregate_labels))  # Focus on first 7 aggregates (including utility)

    # Use colors from the global palette
    plot_colors = colors[:n_experiments]

    # Variable short names for file saving
    var_names = ["C_agg", "L_agg", "K_agg", "Y_agg", "M_agg", "I_agg", "Utility"]

    figures = []

    for agg_idx in range(n_aggregates):
        # Extract data for this aggregate variable across all experiments
        agg_data = {}
        for exp_name in experiment_names:
            agg_data[exp_name] = aggregates_data[exp_name][:, agg_idx]

        # Determine global min and max across all experiments for consistent bin range
        all_values = np.concatenate([agg_data[exp] for exp in experiment_names])
        global_min = np.min(all_values)
        global_max = np.max(all_values)

        # Add padding (5%) to see beyond the extremes
        padding = (global_max - global_min) * 0.05
        bin_range = (global_min - padding, global_max + padding)

        # Create bins using the global range
        bins = np.linspace(bin_range[0], bin_range[1], 31)  # 31 edges for 30 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # Plot histogram for each experiment
        for i, exp_name in enumerate(experiment_names):
            # Calculate histogram
            counts, _ = np.histogram(agg_data[exp_name], bins=bins)
            freqs = counts / len(agg_data[exp_name])

            # Plot the frequency line
            ax.plot(bin_centers, freqs, label=exp_name, color=plot_colors[i], linewidth=2, alpha=0.9)

        # Add vertical line at deterministic steady state (x=0)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Deterministic SS", alpha=0.7)

        # Styling
        ax.set_xlabel(
            f"{aggregate_labels[agg_idx]} (log deviations from deterministic SS)",
            fontweight="bold",
            fontsize=MEDIUM_SIZE,
        )
        ax.set_ylabel("Frequency", fontweight="bold", fontsize=MEDIUM_SIZE)
        ax.set_title(
            f"Ergodic Distribution: {aggregate_labels[agg_idx]}", fontweight="bold", pad=20, fontsize=LARGE_SIZE
        )

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
                filename = f"Histogram_{var_names[agg_idx]}_{analysis_name}.png"
            else:
                filename = f"Histogram_{var_names[agg_idx]}_comparative.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

        figures.append((fig, ax))

    return figures
