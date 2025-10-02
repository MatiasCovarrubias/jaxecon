"""
Training Visualization Module

This module provides plotting functions for visualizing neural network training metrics.
These functions are model-agnostic and work with any training results dictionary that
follows the standard structure.

Standard training_results structure:
    {
        'metrics': {
            'checkpointed_steps': [...],
            'losses': [...],
            'mean_accuracy': [...],
            'min_accuracy': [...],
            'learning_rates': [...]
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


def plot_training_metrics(
    training_results: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 5),
    save_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of training metrics (losses and accuracies).

    Parameters:
    -----------
    training_results : dict
        Dictionary containing training results with 'metrics' key
        Expected structure: {'metrics': {'checkpointed_steps': [...], 'losses': [...],
                                         'mean_accuracy': [...], 'min_accuracy': [...]}}
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    experiment_name : str, optional
        Name of the experiment to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    print("📊 Plotting: Training Metrics (Losses and Accuracies)")

    metrics = training_results["metrics"]
    steps = np.array(metrics["checkpointed_steps"])
    losses = np.array(metrics["losses"])
    mean_acc = np.array(metrics["mean_accuracy"])
    min_acc = np.array(metrics["min_accuracy"])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=display_dpi)

    # Plot 1: Mean Losses
    axes[0].plot(steps, losses, color=colors[0], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[0].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0].set_ylabel("Mean Loss", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Plot 2: Mean Accuracy
    axes[1].plot(steps, mean_acc, color=colors[1], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[1].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1].set_ylabel("Mean Accuracy", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Plot 3: Min Accuracy
    axes[2].plot(steps, min_acc, color=colors[2], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[2].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[2].set_ylabel("Minimum Accuracy", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        if experiment_name:
            filename = f"training_metrics_{experiment_name}.png"
        else:
            filename = "training_metrics.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, axes


def plot_learning_rate_schedule(
    training_results: Dict[str, Any],
    figsize: Tuple[float, float] = (10, 6),
    save_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plot of learning rate schedule over training.

    Parameters:
    -----------
    training_results : dict
        Dictionary containing training results with 'metrics' key
        Expected structure: {'metrics': {'checkpointed_steps': [...], 'learning_rates': [...]}}
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    experiment_name : str, optional
        Name of the experiment to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    print("📊 Plotting: Learning Rate Schedule")

    metrics = training_results["metrics"]
    steps = np.array(metrics["checkpointed_steps"])
    learning_rates = np.array(metrics["learning_rates"])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    # Plot learning rate
    ax.plot(steps, learning_rates, color=colors[3], linewidth=2, marker="o", markersize=4, alpha=0.9)
    ax.set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Learning Rate", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Use log scale for y-axis if learning rate varies by orders of magnitude
    if len(learning_rates) > 0:
        lr_ratio = np.max(learning_rates) / np.min(learning_rates)
        if lr_ratio > 10:
            ax.set_yscale("log")

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        if experiment_name:
            filename = f"learning_rate_{experiment_name}.png"
        else:
            filename = "learning_rate.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, ax


def plot_training_summary(
    training_results: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 10),
    save_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create comprehensive summary plot of all training metrics.

    Combines loss, accuracies, and learning rate in a single figure for quick overview.

    Parameters:
    -----------
    training_results : dict
        Dictionary containing training results with 'metrics' key
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    experiment_name : str, optional
        Name of the experiment to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    print("📊 Plotting: Training Summary (All Metrics)")

    metrics = training_results["metrics"]
    steps = np.array(metrics["checkpointed_steps"])
    losses = np.array(metrics["losses"])
    mean_acc = np.array(metrics["mean_accuracy"])
    min_acc = np.array(metrics["min_accuracy"])
    learning_rates = np.array(metrics["learning_rates"])

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=display_dpi)

    # Plot 1: Mean Losses
    axes[0, 0].plot(steps, losses, color=colors[0], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[0, 0].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0, 0].set_ylabel("Mean Loss", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Plot 2: Mean Accuracy
    axes[0, 1].plot(steps, mean_acc, color=colors[1], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[0, 1].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0, 1].set_ylabel("Mean Accuracy", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Plot 3: Min Accuracy
    axes[1, 0].plot(steps, min_acc, color=colors[2], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[1, 0].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1, 0].set_ylabel("Minimum Accuracy", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Plot 4: Learning Rate
    axes[1, 1].plot(steps, learning_rates, color=colors[3], linewidth=2, marker="o", markersize=4, alpha=0.9)
    axes[1, 1].set_xlabel("Steps (NN updates)", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1, 1].set_ylabel("Learning Rate", fontweight="bold", fontsize=MEDIUM_SIZE)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Use log scale for learning rate if appropriate
    if len(learning_rates) > 0:
        lr_ratio = np.max(learning_rates) / np.min(learning_rates)
        if lr_ratio > 10:
            axes[1, 1].set_yscale("log")

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        if experiment_name:
            filename = f"training_summary_{experiment_name}.png"
        else:
            filename = "training_summary.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    return fig, axes
