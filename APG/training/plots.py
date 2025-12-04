"""
Plotting utilities for APG training results.
"""

import os

import matplotlib.pyplot as plt


def plot_training_metrics(training_results, save_dir, experiment_name, display_dpi=100):
    """Plot training metrics from experiment results.

    Args:
        training_results: Dictionary containing training results with metrics
        save_dir: Directory to save plots
        experiment_name: Name of the experiment for plot titles
        display_dpi: DPI for display
    """
    metrics = training_results.get("metrics", training_results)

    mean_losses = metrics.get("losses", metrics.get("Losses_list", []))
    mean_actor_losses = metrics.get("actor_losses", metrics.get("Actor_losses_list", []))
    mean_critic_losses = metrics.get("critic_losses", metrics.get("Critic_losses_list", []))
    mean_critic_accs = metrics.get("critic_accs", metrics.get("Critic_accs_list", []))
    checkpointed_steps = metrics.get("checkpointed_steps", list(range(len(mean_losses))))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean Losses
    axes[0, 0].plot(checkpointed_steps, mean_losses)
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Mean Loss")
    axes[0, 0].set_title("Total Loss")

    # Mean Actor Losses
    if mean_actor_losses:
        axes[0, 1].plot(checkpointed_steps, mean_actor_losses)
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("Actor Loss")
        axes[0, 1].set_title("Actor Loss")

    # Mean Critic Losses
    if mean_critic_losses:
        axes[1, 0].plot(checkpointed_steps, mean_critic_losses)
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Critic Loss")
        axes[1, 0].set_title("Critic Loss")

    # Mean Accuracy
    if mean_critic_accs:
        axes[1, 1].plot(checkpointed_steps, mean_critic_accs)
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].set_ylabel("Critic Accuracy (%)")
        axes[1, 1].set_title("Critic Accuracy")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_training_metrics.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def plot_learning_rate_schedule(training_results, save_dir, experiment_name, display_dpi=100):
    """Plot learning rate schedule.

    Args:
        training_results: Dictionary containing training results
        save_dir: Directory to save plots
        experiment_name: Name of the experiment for plot titles
        display_dpi: DPI for display
    """
    metrics = training_results.get("metrics", training_results)
    learning_rates = metrics.get("learning_rates", [])
    checkpointed_steps = metrics.get("checkpointed_steps", list(range(len(learning_rates))))

    if not learning_rates:
        return None

    fig = plt.figure(figsize=(8, 6))
    plt.plot(checkpointed_steps, learning_rates)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_learning_rate.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return fig

