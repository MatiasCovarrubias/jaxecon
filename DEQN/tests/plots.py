"""
Testing Visualization Module

This module provides plotting functions for visualizing grid simulation test results.
These functions are model-agnostic and work with any grid test results that follow
the standard structure from run_seed_length_grid.

Standard grid test results structure:
    {
        experiment_name: {
            length: {
                burnin_frac: {
                    'sd_state_mean': float,
                    'sd_policies_mean': float,
                    'sd_aggregates_mean': float,
                    'avg_iact_aggregates': list,
                    'avg_ood_fraction': dict,
                    'avg_trend_slope_aggregates': list,
                    ...
                }
            },
            'sd_vs_T_slope': {
                'state_logsd_logT_slope': float,
                'policies_logsd_logT_slope': float,
                'aggregates_logsd_logT_slope': float
            }
        }
    }
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
        # Create grid tests subfolder
        grid_tests_dir = os.path.join(save_dir, "grid_tests")
        os.makedirs(grid_tests_dir, exist_ok=True)

        if test_name:
            filename = f"grid_test_scaling_{test_name}.png"
        else:
            filename = "grid_test_scaling.png"
        save_path = os.path.join(grid_tests_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, axes


def plot_grid_test_diagnostics(
    grid_test_results: Dict[str, Any],
    diagnostic_type: str = "iact",
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
                                trend_values.append(abs(trend_aggs[agg_idx]))
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
        # Create grid tests subfolder
        grid_tests_dir = os.path.join(save_dir, "grid_tests")
        os.makedirs(grid_tests_dir, exist_ok=True)

        if test_name:
            filename = f"grid_test_{diagnostic_type}_{test_name}.png"
        else:
            filename = f"grid_test_{diagnostic_type}.png"
        save_path = os.path.join(grid_tests_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax
