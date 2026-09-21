"""
Plotting utilities for APG training results.
"""

import math
import os

import matplotlib.pyplot as plt


def _metrics(training_results):
    return training_results.get("metrics", training_results)


def plot_training_metrics(training_results, save_dir, experiment_name, display_dpi=100):
    """Plot eval-only diagnostics from experiment results."""
    metrics = _metrics(training_results)
    welfare_history = metrics.get("welfare_history", [])
    if not welfare_history:
        return None

    has_vols = "std_y" in welfare_history[0]
    has_euler = "euler_acc" in welfare_history[0]
    panels = ["ce", "k_rel"]
    if has_vols:
        panels.extend(["vols", "i_y"])
    if has_euler:
        panels.append("euler")

    ncols = 2 if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    flat = axes.flat
    epochs = [row["epoch"] for row in welfare_history]
    checkpointed_steps = metrics.get("checkpointed_steps")
    recorded_epochs = metrics.get("epochs")
    if checkpointed_steps and recorded_epochs and recorded_epochs[-1]:
        steps_per_epoch = checkpointed_steps[-1] / recorded_epochs[-1]
        x = [epoch * steps_per_epoch for epoch in epochs]
        x_label = "Adam steps"
    else:
        x = epochs
        x_label = "Epoch"

    for i, name in enumerate(panels):
        ax = flat[i]
        if name == "ce":
            ax.plot(x, [100 * row["ce_vs_fixed_saving_rate"] for row in welfare_history])
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_ylabel("CE vs $s_{ss}$ (%)")
            ax.set_title("Consumption equivalent vs $s_{ss}$")
        elif name == "k_rel":
            ax.plot(x, [row["K_rel"] for row in welfare_history])
            ax.axhline(1.0, color="0.5", linewidth=0.8)
            ax.set_ylabel(r"$K/K_{ss}$")
            ax.set_title("Mean capital")
        elif name == "vols":
            ax.plot(x, [row["std_c"] for row in welfare_history], label=r"$\sigma(\log C)$")
            ax.plot(x, [row["std_y"] for row in welfare_history], label=r"$\sigma(\log Y)$")
            ax.plot(x, [row["std_i"] for row in welfare_history], label=r"$\sigma(\log I)$")
            ax.set_ylabel("Std of log")
            ax.set_title("Aggregate volatilities")
            ax.legend()
        elif name == "i_y":
            ax.plot(
                x,
                [
                    row["std_i"] / row["std_y"] if row.get("std_y") else float("nan")
                    for row in welfare_history
                ],
            )
            ax.axhline(1.0, color="0.5", linewidth=0.8)
            ax.set_ylabel(r"$\sigma(\log I)/\sigma(\log Y)$")
            ax.set_title("Relative investment volatility")
        elif name == "euler":
            ax.plot(x, [100 * row["euler_acc"] for row in welfare_history])
            ax.set_ylabel("Euler accuracy (%)")
            ax.set_title("Euler accuracy")
        ax.set_xlabel(x_label)

    for j in range(len(panels), nrows * ncols):
        flat[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{experiment_name}_training_metrics.png"), dpi=300, bbox_inches="tight")
    if plt.isinteractive():
        plt.show()
    return fig


def plot_learning_rate_schedule(training_results, save_dir, experiment_name, display_dpi=100):
    """Plot learning rate schedule."""
    metrics = _metrics(training_results)
    learning_rates = metrics.get("learning_rates", [])
    checkpointed_steps = metrics.get("checkpointed_steps", list(range(len(learning_rates))))
    epochs = metrics.get("epochs")
    x = epochs if epochs else checkpointed_steps
    x_label = "Epoch" if epochs else "Steps"

    if not learning_rates:
        return None

    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(x, learning_rates)
    plt.xlabel(x_label)
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_learning_rate.png"), dpi=300, bbox_inches="tight")
    if plt.isinteractive():
        plt.show()
    return fig
