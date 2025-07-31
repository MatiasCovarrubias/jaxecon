import matplotlib.pyplot as plt


def plot_results(results, config):
    """Plot training results"""

    mean_losses = results["Losses_list"]
    mean_actor_losses = results["Actor_losses_list"]
    mean_critic_losses = results["Critic_losses_list"]
    mean_critic_accs = results["Critic_accs_list"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean Losses
    axes[0, 0].plot([(i) * config["steps_per_epoch"] for i in range(len(mean_losses))], mean_losses)
    axes[0, 0].set_xlabel("Episodes (NN updates)")
    axes[0, 0].set_ylabel("Mean Losses")
    axes[0, 0].set_title("Mean Losses")

    # Mean Actor Losses
    axes[0, 1].plot([(i) * config["steps_per_epoch"] for i in range(len(mean_actor_losses))], mean_actor_losses)
    axes[0, 1].set_xlabel("Episodes (NN updates)")
    axes[0, 1].set_ylabel("Mean Actor Losses")
    axes[0, 1].set_title("Mean Actor Losses")

    # Mean Critic Losses
    axes[1, 0].plot([(i) * config["steps_per_epoch"] for i in range(len(mean_critic_losses))], mean_critic_losses)
    axes[1, 0].set_xlabel("Episodes (NN updates)")
    axes[1, 0].set_ylabel("Mean Critic Losses")
    axes[1, 0].set_title("Mean Critic Losses")

    # Mean Accuracy
    axes[1, 1].plot([(i) * config["steps_per_epoch"] for i in range(len(mean_critic_accs))], mean_critic_accs)
    axes[1, 1].set_xlabel("Episodes (NN updates)")
    axes[1, 1].set_ylabel("Mean Accuracy (%)")
    axes[1, 1].set_title("Mean Critic Accuracy")

    plt.tight_layout()
    plt.savefig(config["working_dir"] + config["run_name"] + "/training_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Learning rate schedule
    plt.figure(figsize=(8, 6))
    plt.plot(
        [i * config["steps_per_epoch"] for i in range(len(mean_losses))],
        [config["learning_rate"](i * config["steps_per_epoch"]) for i in range(len(mean_losses))],
    )
    plt.xlabel("Episodes (NN updates)")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.savefig(config["working_dir"] + config["run_name"] + "/learning_rate.png", dpi=300, bbox_inches="tight")
    plt.show()
