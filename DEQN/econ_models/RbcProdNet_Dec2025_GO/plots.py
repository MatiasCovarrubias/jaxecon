"""
Model-Specific Visualization Module for RbcProdNet

This module contains plotting functions that are specific to the RbcProdNet model,
such as upstreamness measures and sectoral analysis.

These functions take raw simulation data (simul_obs, simul_policies, simul_analysis_variables)
as inputs and create model-specific visualizations.

For general plotting functions (training, analysis, testing), see:
- DEQN/training/plots.py
- DEQN/analysis/plots.py
- DEQN/tests/plots.py

## Registry Pattern

Model-specific plots should be registered using the MODEL_SPECIFIC_PLOTS list.
Each entry is a dict with:
- 'function': the plotting function
- 'model_args': list of attribute names from econ_model to pass (e.g., ['n_sectors', 'labels'])
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
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


# ============================================================================
# MODEL-SPECIFIC PLOTS REGISTRY
# ============================================================================
# Register all model-specific plots here. The analysis script will automatically
# discover and run these plots.
#
# Each plot function should have signature:
#   plot_name(simul_obs, simul_policies, simul_analysis_variables,
#             save_path, analysis_name, econ_model, experiment_label, **kwargs)
# ============================================================================

MODEL_SPECIFIC_PLOTS: List[Dict[str, Any]] = []


def configure_for_colab():
    """
    Configure matplotlib settings for optimal display in Google Colab.

    This function sets lower DPI for display while keeping high DPI for saved figures.
    Call this function at the beginning of your Colab notebook.
    """
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300


def plot_sectoral_capital(
    simul_obs: jnp.ndarray,
    simul_policies: jnp.ndarray,
    simul_analysis_variables: Dict[str, jnp.ndarray],
    save_path: str,
    analysis_name: str,
    econ_model: Any,
    experiment_label: str,
    figsize: Tuple[float, float] = (12, 8),
    display_dpi: int = 100,
):
    """
    Create a publication-quality bar graph of mean sectoral capital.

    This is a model-specific plot that takes raw simulation observations and
    extracts sectoral capital data.

    Parameters:
    -----------
    simul_obs : jnp.ndarray
        Simulation observations array of shape (n_periods, n_obs)
    simul_policies : jnp.ndarray
        Simulation policies array (not used here but part of standard signature)
    simul_analysis_variables : dict
        Analysis variables dictionary (not used here but part of standard signature)
    save_path : str
        Path where the figure should be saved
    analysis_name : str
        Name of the analysis to include in the filename
    econ_model : Any
        Economic model instance (used to get n_sectors and labels)
    experiment_label : str
        Label for the experiment
    figsize : tuple, optional
        Figure size (width, height) in inches
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels

    sectoral_capital_mean = jnp.mean(simul_obs, axis=0)[:n_sectors]

    sorted_indices = np.argsort(sectoral_capital_mean)[::-1]
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]
    sorted_capital = sectoral_capital_mean[sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    x = np.arange(n_sectors)

    ax.bar(
        x,
        sorted_capital * 100,
        0.8,
        label=experiment_label,
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Average Capital (% Deviations from SS)", fontweight="bold", fontsize=MEDIUM_SIZE)

    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if analysis_name:
        save_dir = os.path.dirname(save_path)
        base_filename = os.path.basename(save_path)
        ext = os.path.splitext(base_filename)[1] or ".png"
        new_filename = f"sectoral_capital_{experiment_label}_{analysis_name}{ext}"
        final_save_path = os.path.join(save_dir, new_filename)
    else:
        final_save_path = save_path

    plt.savefig(final_save_path, dpi=300, bbox_inches="tight", format="png")
    plt.show()

    return fig, ax


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
    plot_colors = colors[:3]
    sectors = upstreamness_data["sectors"]
    U_M = upstreamness_data["U_M"]
    U_I = upstreamness_data["U_I"]
    U_simple = upstreamness_data["U_simple"]

    sorted_indices = np.argsort(U_M)[::-1]
    sorted_sectors = [sectors[i] for i in sorted_indices]
    sorted_U_M = U_M[sorted_indices]
    sorted_U_I = U_I[sorted_indices]
    sorted_U_simple = U_simple[sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    bar_width = 0.25
    x = np.arange(len(sorted_sectors))

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

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha="right")

    ax.set_xlabel("Sector", fontweight="bold")
    ax.set_ylabel("Upstreamness Measure", fontweight="bold")

    ax.legend(frameon=True, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# ============================================================================
# REGISTER MODEL-SPECIFIC PLOTS
# ============================================================================

MODEL_SPECIFIC_PLOTS = [
    {
        "name": "sectoral_capital",
        "function": plot_sectoral_capital,
        "description": "Bar plot of mean sectoral capital across sectors",
    },
]
