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


def configure_for_colab():
    """
    Configure matplotlib settings for optimal display in Google Colab.

    This function sets lower DPI for display while keeping high DPI for saved figures.
    Call this function at the beginning of your Colab notebook.
    """
    plt.rcParams["figure.dpi"] = 100  # Lower DPI for Colab display
    plt.rcParams["savefig.dpi"] = 300  # High DPI for saved figures


def plot_upstreamness(
    upstreamness_data: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
    display_dpi: int = 100,
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
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Print what's being plotted
    print("📊 Plotting: Sector Upstreamness Measures")

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
    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

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

    # Set labels
    ax.set_xlabel("Sector", fontweight="bold")
    ax.set_ylabel("Upstreamness Measure", fontweight="bold")

    # Add legend
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    # Adjust layout and save if requested
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Keep high DPI for saved files

    return fig, ax


def plot_sectoral_capital_mean(
    analysis_results: Dict[str, Any],
    sector_labels: list,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
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
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Print what's being plotted
    print("📊 Plotting: Mean Sectoral Capital Across Experiments")

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
    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

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
            sorted_capital_data[exp_name] * 100,  # Convert to percentages
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

    # Set labels using predefined font sizes
    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Average Capital (% Deviations from SS)", fontweight="bold", fontsize=MEDIUM_SIZE)

    # Add legend using predefined font size
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

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
    display_dpi: int = 100,
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
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    list of (fig, ax) tuples for each aggregate variable
    """
    # Print what's being plotted
    print("📊 Plotting: Ergodic Distribution Histograms")

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
        # Print which specific histogram is being plotted
        print(f"  📈 Creating histogram for: {aggregate_labels[agg_idx]}")

        # Extract data for this aggregate variable across all experiments and convert to percentages
        agg_data = {}
        for exp_name in experiment_names:
            agg_data[exp_name] = aggregates_data[exp_name][:, agg_idx] * 100

        # Use fixed range from -10 to 10 (percentages)
        bin_range = (-10, 10)

        # Create bins using the fixed range
        bins = np.linspace(bin_range[0], bin_range[1], 31)  # 31 edges for 30 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=display_dpi)

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
            f"{aggregate_labels[agg_idx]} (% deviations from deterministic SS)",
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
                filename = f"Histogram_{var_names[agg_idx]}_{analysis_name}.png"
            else:
                filename = f"Histogram_{var_names[agg_idx]}_comparative.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

        figures.append((fig, ax))

    return figures


def plot_gir_responses(
    gir_data: Dict[str, Any],
    aggregate_indices: list = [0, 3, 5],  # Default: Consumption, Output, Investment
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Output",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility Welfare",
    ],
    sectors_to_plot: Optional[list] = None,  # If None, plots all sectors
    figsize: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of Generalized Impulse Responses over time.
    Each sector gets its own separate plot for each aggregate variable.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from analysis
        Structure: {experiment_name: {sector_name: {"gir_aggregates": array, "sector_idx": int}}}
    aggregate_indices : list, optional
        Which aggregate variables to plot (indices 0-6). Default plots C, Y, I.
    aggregate_labels : list, optional
        Labels for the aggregate variables
    sectors_to_plot : list, optional
        Which sectors to plot. If None, plots all sectors.
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
    list of (fig, ax) tuples for each sector-aggregate combination
    """
    # Print what's being plotted
    print("📊 Plotting: Generalized Impulse Response Functions")

    # Get experiment names
    experiment_names = list(gir_data.keys())
    n_experiments = len(experiment_names)

    # Get first experiment to determine sectors and time length
    first_experiment = experiment_names[0]
    first_exp_data = gir_data[first_experiment]

    # Get sector names
    all_sector_names = list(first_exp_data.keys())
    if sectors_to_plot is None:
        # Default to all sectors
        sectors_to_plot = all_sector_names
    else:
        # Filter to requested sectors
        sectors_to_plot = [s for s in sectors_to_plot if s in all_sector_names]

    # Get time length from first sector's GIR data
    first_sector = sectors_to_plot[0]
    time_length = first_exp_data[first_sector]["gir_aggregates"].shape[0]
    time_periods = np.arange(time_length)

    # Use colors from the global palette
    plot_colors = colors

    # Variable short names for file saving
    var_names = ["C_agg", "L_agg", "K_agg", "Y_agg", "M_agg", "I_agg", "Utility_welfare"]

    figures = []

    # Loop through each sector first, then each aggregate
    for sector_name in sectors_to_plot:
        for agg_idx in aggregate_indices:
            # Print which specific GIR plot is being created
            print(f"  📈 Creating GIR plot: {aggregate_labels[agg_idx]} response to {sector_name} shock")

            # Create figure for this sector-aggregate combination
            fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

            # Plot for each experiment (if multiple)
            for j, exp_name in enumerate(experiment_names):
                gir_aggregates = gir_data[exp_name][sector_name]["gir_aggregates"]

                # Convert to percentages and extract specific aggregate
                response_pct = gir_aggregates[:, agg_idx] * 100

                # Create label
                if n_experiments > 1:
                    label = exp_name
                    color = plot_colors[j % len(plot_colors)]
                    linestyle = "-" if j == 0 else "--"
                else:
                    label = f"{sector_name} Response"
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
            ax.set_ylabel(f"{aggregate_labels[agg_idx]} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

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
                # Create filename with sector and aggregate names
                safe_sector_name = sector_name.replace(" ", "_").replace("/", "_")
                if analysis_name:
                    filename = f"GIR_{var_names[agg_idx]}_{safe_sector_name}_{analysis_name}.png"
                else:
                    filename = f"GIR_{var_names[agg_idx]}_{safe_sector_name}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

            figures.append((fig, ax))

    return figures


def plot_gir_heatmap(
    gir_data: Dict[str, Any],
    aggregate_idx: int = 0,  # Default: Consumption
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Labor",
        "Agg. Capital",
        "Agg. Output",
        "Agg. Intermediate Goods",
        "Agg. Investment",
        "Utility Welfare",
    ],
    time_slice: int = 10,  # Which time period to show (default: period 10)
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
    # Print what's being plotted
    print(f"📊 Plotting: GIR Cross-Sector Heatmap - {aggregate_labels[aggregate_idx]} at Period {time_slice}")

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
        if analysis_name:
            save_dir = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".png"
            var_names = ["C_agg", "L_agg", "K_agg", "Y_agg", "M_agg", "I_agg", "Utility_welfare"]
            new_filename = f"GIR_heatmap_{var_names[aggregate_idx]}_period{time_slice}_{analysis_name}{ext}"
            final_save_path = os.path.join(save_dir, new_filename)
        else:
            final_save_path = save_path
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax


def plot_grid_test_scaling(
    grid_test_results: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    test_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of SD vs T scaling for grid test diagnostics.

    Shows how cross-seed standard deviations scale with episode length T for
    states, policies, and aggregates. Expected slope is -0.5 for pure sampling error.

    Parameters:
    -----------
    grid_test_results : dict
        Dictionary containing grid test results from run_seed_length_grid
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    test_name : str, optional
        Name of the test to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    print("📊 Plotting: Grid Test SD vs T Scaling")

    experiment_names = list(grid_test_results.keys())
    n_experiments = len(experiment_names)

    # Use colors from the global palette
    plot_colors = colors[:n_experiments]

    # Create figure with subplots for each variable type
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=display_dpi)

    variable_types = ["State", "Policies", "Aggregates"]
    slope_keys = ["state_logsd_logT_slope", "policies_logsd_logT_slope", "aggregates_logsd_logT_slope"]

    for i, (var_type, slope_key) in enumerate(zip(variable_types, slope_keys)):
        ax = axes[i]

        for j, exp_name in enumerate(experiment_names):
            grid_data = grid_test_results[exp_name]

            # Extract length and SD data
            lengths = sorted([k for k in grid_data.keys() if isinstance(k, (int, float))])

            if lengths:
                # Use first burn-in fraction for plotting
                burnin_fracs = list(grid_data[lengths[0]].keys())
                if burnin_fracs:
                    b0 = burnin_fracs[0]

                    sd_values = []
                    for T in lengths:
                        if var_type == "State":
                            sd_val = grid_data[T][b0]["sd_state_mean"]
                        elif var_type == "Policies":
                            sd_val = grid_data[T][b0]["sd_policies_mean"]
                        else:  # Aggregates
                            sd_val = grid_data[T][b0]["sd_aggregates_mean"]
                        sd_values.append(sd_val)

                    # Plot on log-log scale
                    ax.loglog(
                        lengths,
                        sd_values,
                        "o-",
                        label=exp_name,
                        color=plot_colors[j],
                        linewidth=2,
                        markersize=6,
                        alpha=0.8,
                    )

                    # Get slope for this experiment
                    slopes = grid_data.get("sd_vs_T_slope", {})
                    slope = slopes.get(slope_key, float("nan"))

                    # Add slope annotation
                    if not np.isnan(slope):
                        ax.text(
                            0.05,
                            0.95 - j * 0.1,
                            f"{exp_name}: slope = {slope:.3f}",
                            transform=ax.transAxes,
                            fontsize=SMALL_SIZE,
                            color=plot_colors[j],
                        )

        # Add reference line with slope -0.5
        if lengths:
            x_ref = np.array([min(lengths), max(lengths)])
            # Normalize to pass through middle of typical SD range
            mid_x = np.sqrt(min(lengths) * max(lengths))
            mid_y = 1e-3  # Typical SD scale
            y_ref = mid_y * (x_ref / mid_x) ** (-0.5)
            ax.loglog(x_ref, y_ref, "--", color="black", alpha=0.5, linewidth=2, label="Slope = -0.5 (expected)")

        # Styling
        ax.set_xlabel("Episode Length T", fontweight="bold", fontsize=MEDIUM_SIZE)
        ax.set_ylabel(f"{var_type} Cross-Seed SD", fontweight="bold", fontsize=MEDIUM_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=SMALL_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        if test_name:
            filename = f"grid_test_scaling_{test_name}.png"
        else:
            filename = "grid_test_scaling.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, axes


def plot_grid_test_diagnostics(
    grid_test_results: Dict[str, Any],
    diagnostic_type: str = "iact",  # "iact", "ood", "trend"
    figsize: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    test_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of various grid test diagnostics.

    Parameters:
    -----------
    grid_test_results : dict
        Dictionary containing grid test results from run_seed_length_grid
    diagnostic_type : str
        Type of diagnostic to plot: "iact", "ood", or "trend"
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    test_name : str, optional
        Name of the test to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    print(f"📊 Plotting: Grid Test {diagnostic_type.upper()} Diagnostics")

    experiment_names = list(grid_test_results.keys())
    n_experiments = len(experiment_names)

    # Use colors from the global palette
    plot_colors = colors[:n_experiments]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    for j, exp_name in enumerate(experiment_names):
        grid_data = grid_test_results[exp_name]

        # Extract length data
        lengths = sorted([k for k in grid_data.keys() if isinstance(k, (int, float))])

        if lengths:
            # Use first burn-in fraction
            burnin_fracs = list(grid_data[lengths[0]].keys())
            if burnin_fracs:
                b0 = burnin_fracs[0]

                if diagnostic_type == "iact":
                    # Plot IACT for first few aggregates (C, L, K, Y)
                    aggregate_names = ["Consumption", "Labor", "Capital", "Output"]
                    for agg_idx in range(min(4, len(aggregate_names))):
                        iact_values = []
                        for T in lengths:
                            iact_aggs = grid_data[T][b0]["avg_iact_aggregates"]
                            if agg_idx < len(iact_aggs):
                                iact_values.append(iact_aggs[agg_idx])
                            else:
                                iact_values.append(np.nan)

                        line_style = ["-", "--", "-.", ":"][agg_idx]
                        ax.plot(
                            lengths,
                            iact_values,
                            line_style,
                            label=f"{exp_name} - {aggregate_names[agg_idx]}",
                            color=plot_colors[j],
                            linewidth=2,
                            alpha=0.8,
                        )

                    ax.set_ylabel("Integrated Autocorr. Time (IACT)", fontweight="bold", fontsize=MEDIUM_SIZE)
                    ax.set_title("IACT vs Episode Length", fontweight="bold", fontsize=LARGE_SIZE)

                elif diagnostic_type == "ood":
                    # Plot OOD fractions for different thresholds
                    thresholds = [3.0, 4.0, 5.0]
                    for thresh_idx, threshold in enumerate(thresholds):
                        ood_values = []
                        for T in lengths:
                            ood_dict = grid_data[T][b0]["avg_ood_fraction"]
                            ood_values.append(ood_dict.get(threshold, 0.0))

                        line_style = ["-", "--", "-."][thresh_idx]
                        ax.plot(
                            lengths,
                            ood_values,
                            line_style,
                            label=f"{exp_name} - Threshold {threshold}",
                            color=plot_colors[j],
                            linewidth=2,
                            alpha=0.8,
                        )

                    ax.set_ylabel("Out-of-Distribution Fraction", fontweight="bold", fontsize=MEDIUM_SIZE)
                    ax.set_title("OOD Fraction vs Episode Length", fontweight="bold", fontsize=LARGE_SIZE)

                elif diagnostic_type == "trend":
                    # Plot trend slopes for first few aggregates
                    aggregate_names = ["Consumption", "Labor", "Capital", "Output"]
                    for agg_idx in range(min(4, len(aggregate_names))):
                        trend_values = []
                        for T in lengths:
                            trend_aggs = grid_data[T][b0]["avg_trend_slope_aggregates"]
                            if agg_idx < len(trend_aggs):
                                trend_values.append(abs(trend_aggs[agg_idx]))  # Use absolute value
                            else:
                                trend_values.append(np.nan)

                        line_style = ["-", "--", "-.", ":"][agg_idx]
                        ax.semilogy(
                            lengths,
                            trend_values,
                            line_style,
                            label=f"{exp_name} - {aggregate_names[agg_idx]}",
                            color=plot_colors[j],
                            linewidth=2,
                            alpha=0.8,
                        )

                    ax.set_ylabel("Absolute Trend Slope", fontweight="bold", fontsize=MEDIUM_SIZE)
                    ax.set_title("Trend Slopes vs Episode Length", fontweight="bold", fontsize=LARGE_SIZE)

    # Common styling
    ax.set_xlabel("Episode Length T", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=SMALL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        if test_name:
            filename = f"grid_test_{diagnostic_type}_{test_name}.png"
        else:
            filename = f"grid_test_{diagnostic_type}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax
