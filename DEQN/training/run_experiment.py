"""
Experiment runner for DEQN training.

Provides high-level orchestration of training experiments including:
- Neural network initialization
- Learning rate scheduling
- Training loop execution
- Checkpointing
- Metrics collection

The runner returns training data/metrics which can be used for plotting and analysis.
"""

import itertools
import json
import os
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import orbax.checkpoint as ocp
from flax.training import checkpoints, train_state


class TrainState(train_state.TrainState):
    """Custom TrainState for compatibility."""

    pass


def run_experiment(config, econ_model, neural_net, epoch_train_fn):
    """
    Run a single training experiment and return metrics.

    This function orchestrates the entire training process but does not create plots.
    Instead, it returns all training data for later visualization and analysis.

    Args:
        config: Configuration dictionary containing all hyperparameters
        econ_model: Economic model instance
        neural_net: Pre-built neural network instance (Flax module)
        epoch_train_fn: Epoch training function (get_epoch_train_fn or get_epoch_train_fn_fast)

    Returns:
        dict: Dictionary containing:
            - train_state: Final trained state
            - metrics: Training and evaluation metrics
            - timing: Time statistics
            - config: Configuration used
    """
    n_cores = len(jax.devices())
    print(f"Running on {n_cores} device(s)")

    # CREATE RNGS, TRAIN_STATE
    rng_pol, rng_epoch, rng_eval = random.split(random.PRNGKey(config["seed"]), num=3)

    # CREATE LR SCHEDULE
    total_steps = config["n_epochs"] * config["steps_per_epoch"]
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config["learning_rate"],
        decay_steps=total_steps,
        alpha=0.0,  # decay to 0
    )

    # INITIALIZE OR RESTORE TRAIN STATE
    if not config["restore"]:
        params = neural_net.init(rng_pol, jnp.zeros_like(econ_model.state_ss))
        train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=optax.adam(lr_schedule))
    else:
        train_state_restored = checkpoints.restore_checkpoint(
            ckpt_dir=config["save_dir"] + config["restore_exper_name"], target=None
        )
        params = train_state_restored["params"]
        opt_state = train_state_restored["opt_state"]
        train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=optax.adam(lr_schedule))
        train_state_obj = train_state_obj.replace(opt_state=opt_state)

    # GET TRAIN AND EVAL FUNCTIONS (import here to avoid circular dependency)
    from DEQN.algorithm.eval import create_eval_fn
    from DEQN.algorithm.loss import create_batch_loss_fn
    from DEQN.algorithm.simulation import create_episode_simul_fn

    episode_simul_fn = create_episode_simul_fn(econ_model, config)
    batch_loss_fn = create_batch_loss_fn(econ_model, config)
    eval_fn_created = create_eval_fn(config, episode_simul_fn, batch_loss_fn)

    train_epoch_jitted = jax.jit(epoch_train_fn(econ_model, config))
    eval_fn = jax.jit(eval_fn_created)

    # COMPILE CODE
    print("Compiling functions...")
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    eval_fn(train_state_obj, rng_epoch)
    time_compilation = time() - time_start
    print(f"Time Elapsed for Compilation: {time_compilation:.2f} seconds")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    time_epoch = time() - time_start
    print(f"Time Elapsed for epoch: {time_epoch:.2f} seconds")

    time_start = time()
    eval_fn(train_state_obj, rng_epoch)
    time_eval = time() - time_start
    print(f"Time Elapsed for eval: {time_eval:.2f} seconds")

    time_experiment = (time_epoch + time_eval) * config["n_epochs"] / 60
    print(f"Estimated time for full experiment: {time_experiment:.2f} minutes")

    steps_per_second = config["steps_per_epoch"] * config["periods_per_step"] / time_epoch
    print(f"Steps per second: {steps_per_second:.2f} st/s")

    # CREATE LISTS TO STORE METRICS
    mean_losses, mean_accuracy, min_accuracy = [], [], []
    learning_rates = []
    checkpointed_steps = []

    # RUN ALL THE EPOCHS
    time_start = time()
    for i in range(1, config["n_epochs"] + 1):
        # Evaluation
        eval_metrics = eval_fn(train_state_obj, rng_eval)
        print(
            "EVALUATION:\n",
            "Iteration:",
            train_state_obj.step,
            "Mean_loss:",
            eval_metrics[0],
            ", Mean Acc:",
            eval_metrics[1],
            ", Min Acc:",
            eval_metrics[2],
            "\n",
            ", Mean Accs Foc",
            eval_metrics[3],
            "\n",
            ", Min Accs Foc:",
            eval_metrics[4],
            "\n",
        )

        # Training
        train_state_obj, rng_epoch, epoch_metrics = train_epoch_jitted(train_state_obj, rng_epoch)
        current_lr = lr_schedule(train_state_obj.step)
        print(
            "TRAINING:\n",
            "Iteration:",
            train_state_obj.step,
            ", Mean_loss:",
            jnp.mean(epoch_metrics[0]),
            ", Mean_accuracy:",
            jnp.mean(epoch_metrics[1]),
            ", Min_accuracy:",
            jnp.min(epoch_metrics[2]),
            ", Learning rate:",
            current_lr,
            "\n",
        )

        # Checkpoint and metrics storage
        if (
            train_state_obj.step >= config["checkpoint_frequency"]
            and train_state_obj.step % config["checkpoint_frequency"] == 0
        ):
            checkpoints.save_checkpoint(
                ckpt_dir=config["save_dir"] + config["exper_name"], target=train_state_obj, step=train_state_obj.step
            )
            mean_losses.append(float(eval_metrics[0]))
            mean_accuracy.append(float(eval_metrics[1]))
            min_accuracy.append(float(eval_metrics[2]))
            learning_rates.append(float(current_lr))
            checkpointed_steps.append(int(train_state_obj.step))

    # PRINT SUMMARY
    print("Minimum loss attained in evaluation:", min(mean_losses))
    print("Maximum mean accuracy attained in evaluation:", max(mean_accuracy))
    print("Maximum min accuracy attained in evaluation:", max(min_accuracy))
    time_fullexp = (time() - time_start) / 60
    print(f"Time Elapsed for Full Experiment: {time_fullexp:.2f} minutes")

    # Optional comment
    if config.get("comment_at_end", False):
        comment_result = input("Enter a comment for the researcher: ")
    else:
        comment_result = ""

    # PREPARE RESULTS
    results = {
        "exper_name": config["exper_name"],
        "comment_preexp": config["comment"],
        "comment_result": comment_result,
        "min_loss": min(mean_losses),
        "max_mean_acc": max(mean_accuracy),
        "max_min_acc": max(min_accuracy),
        "time_fullexp_minutes": time_fullexp,
        "time_epoch_seconds": time_epoch,
        "time_compilation_seconds": time_compilation,
        "steps_per_second": steps_per_second,
        "losses": mean_losses,
        "mean_accuracy": mean_accuracy,
        "min_accuracy": min_accuracy,
        "learning_rates": learning_rates,
        "checkpointed_steps": checkpointed_steps,
    }

    # SAVE RESULTS TO JSON
    if not os.path.exists(config["save_dir"] + config["exper_name"]):
        os.makedirs(config["save_dir"] + config["exper_name"])
    with open(config["save_dir"] + config["exper_name"] + "/results.json", "w") as write_file:
        # Combine config and results for saving
        save_data = {"config": config, **results}
        json.dump(save_data, write_file, indent=2)

    # Return comprehensive results
    return {
        "train_state": train_state_obj,
        "metrics": results,
        "lr_schedule": lr_schedule,
        "config": config,
    }


def run_experiment_orbax(config, econ_model, neural_net, epoch_train_fn):
    """
    Run a single training experiment with Orbax checkpointing and return metrics.

    This function is identical to run_experiment but uses Orbax for checkpointing
    instead of the deprecated Flax checkpointing API.

    Args:
        config: Configuration dictionary containing all hyperparameters
        econ_model: Economic model instance
        neural_net: Pre-built neural network instance (Flax module)
        epoch_train_fn: Epoch training function (get_epoch_train_fn or get_epoch_train_fn_fast)

    Returns:
        dict: Dictionary containing:
            - train_state: Final trained state
            - metrics: Training and evaluation metrics
            - timing: Time statistics
            - config: Configuration used
    """
    n_cores = len(jax.devices())
    print(f"Running on {n_cores} device(s)")

    # CREATE RNGS, TRAIN_STATE
    rng_pol, rng_epoch, rng_eval = random.split(random.PRNGKey(config["seed"]), num=3)

    # CREATE LR SCHEDULE
    total_steps = config["n_epochs"] * config["steps_per_epoch"]
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config["learning_rate"],
        decay_steps=total_steps,
        alpha=0.0000001,
    )

    # INITIALIZE OR RESTORE TRAIN STATE WITH ORBAX
    checkpoint_dir = Path(config["save_dir"]) / config.get("exper_name", "default_experiment")

    if not config["restore"]:
        params = neural_net.init(rng_pol, jnp.zeros_like(econ_model.state_ss))
        train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=optax.adam(lr_schedule))
    else:
        restore_dir = Path(config["save_dir"]) / config["restore_exper_name"]
        restore_checkpoint_manager = ocp.CheckpointManager(restore_dir)

        latest_step = restore_checkpoint_manager.latest_step()
        if latest_step is None:
            raise ValueError(f"No checkpoints found in {restore_dir}")

        # Create abstract target tree for environment-agnostic restoration
        dummy_params = neural_net.init(rng_pol, jnp.zeros_like(econ_model.state_ss))
        dummy_opt_state = optax.adam(lr_schedule).init(dummy_params)

        abstract_target = {
            "params": jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, dummy_params),
            "opt_state": jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, dummy_opt_state),
        }

        restored_state = restore_checkpoint_manager.restore(
            step=latest_step, args=ocp.args.StandardRestore(abstract_target)  # type: ignore
        )
        params = restored_state["params"]
        opt_state = restored_state["opt_state"]

        train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=optax.adam(lr_schedule))
        train_state_obj = train_state_obj.replace(opt_state=opt_state, step=latest_step)

    # GET TRAIN AND EVAL FUNCTIONS
    from DEQN.algorithm.eval import create_eval_fn
    from DEQN.algorithm.loss import create_batch_loss_fn
    from DEQN.algorithm.simulation import create_episode_simul_fn

    episode_simul_fn = create_episode_simul_fn(econ_model, config)
    batch_loss_fn = create_batch_loss_fn(econ_model, config)
    eval_fn_created = create_eval_fn(config, episode_simul_fn, batch_loss_fn)

    train_epoch_jitted = jax.jit(epoch_train_fn(econ_model, config))
    eval_fn = jax.jit(eval_fn_created)

    # COMPILE CODE
    print("Compiling functions...")
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    eval_fn(train_state_obj, rng_epoch)
    time_compilation = time() - time_start
    print(f"Time Elapsed for Compilation: {time_compilation:.2f} seconds")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    time_epoch = time() - time_start
    print(f"Time Elapsed for epoch: {time_epoch:.2f} seconds")

    time_start = time()
    eval_fn(train_state_obj, rng_epoch)
    time_eval = time() - time_start
    print(f"Time Elapsed for eval: {time_eval:.2f} seconds")

    time_experiment = (time_epoch + time_eval) * config["n_epochs"] / 60
    print(f"Estimated time for full experiment: {time_experiment:.2f} minutes")

    steps_per_second = config["steps_per_epoch"] * config["periods_per_step"] / time_epoch
    print(f"Steps per second: {steps_per_second:.2f} st/s")

    # CREATE LISTS TO STORE METRICS
    mean_losses, mean_accuracy, min_accuracy = [], [], []
    learning_rates = []
    checkpointed_steps = []

    # CREATE ORBAX CHECKPOINT MANAGER (keeps only the most recent checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))

    # RUN ALL THE EPOCHS
    time_start = time()
    for i in range(1, config["n_epochs"] + 1):
        # Evaluation
        eval_metrics = eval_fn(train_state_obj, rng_eval)
        print(
            "EVALUATION:\n",
            "Iteration:",
            train_state_obj.step,
            "Mean_loss:",
            eval_metrics[0],
            ", Mean Acc:",
            eval_metrics[1],
            ", Min Acc:",
            eval_metrics[2],
            "\n",
            ", Mean Accs Foc",
            eval_metrics[3],
            "\n",
            ", Min Accs Foc:",
            eval_metrics[4],
            "\n",
        )

        # Training
        train_state_obj, rng_epoch, epoch_metrics = train_epoch_jitted(train_state_obj, rng_epoch)
        current_lr = lr_schedule(train_state_obj.step)
        print(
            "TRAINING:\n",
            "Iteration:",
            train_state_obj.step,
            ", Mean_loss:",
            jnp.mean(epoch_metrics[0]),
            ", Mean_accuracy:",
            jnp.mean(epoch_metrics[1]),
            ", Min_accuracy:",
            jnp.min(epoch_metrics[2]),
            ", Learning rate:",
            current_lr,
            "\n",
        )

        # Checkpoint and metrics storage with Orbax
        if train_state_obj.step >= 100 and train_state_obj.step % 100 == 0:
            checkpoint_manager.save(
                step=train_state_obj.step, args=ocp.args.StandardSave(train_state_obj)  # type: ignore
            )
            mean_losses.append(float(eval_metrics[0]))
            mean_accuracy.append(float(eval_metrics[1]))
            min_accuracy.append(float(eval_metrics[2]))
            learning_rates.append(float(current_lr))
            checkpointed_steps.append(int(train_state_obj.step))

    # PRINT SUMMARY
    print("Minimum loss attained in evaluation:", min(mean_losses))
    print("Maximum mean accuracy attained in evaluation:", max(mean_accuracy))
    print("Maximum min accuracy attained in evaluation:", max(min_accuracy))
    time_fullexp = (time() - time_start) / 60
    print(f"Time Elapsed for Full Experiment: {time_fullexp:.2f} minutes")

    # Optional comment
    if config.get("comment_at_end", False):
        comment_result = input("Enter a comment for the researcher: ")
    else:
        comment_result = ""

    # PREPARE RESULTS
    results = {
        "exper_name": config["exper_name"],
        "comment_preexp": config["comment"],
        "comment_result": comment_result,
        "min_loss": min(mean_losses),
        "max_mean_acc": max(mean_accuracy),
        "max_min_acc": max(min_accuracy),
        "time_fullexp_minutes": time_fullexp,
        "time_epoch_seconds": time_epoch,
        "time_compilation_seconds": time_compilation,
        "steps_per_second": steps_per_second,
        "losses": mean_losses,
        "mean_accuracy": mean_accuracy,
        "min_accuracy": min_accuracy,
        "learning_rates": learning_rates,
        "checkpointed_steps": checkpointed_steps,
    }

    # SAVE RESULTS TO JSON
    with open(checkpoint_dir / "results.json", "w") as write_file:
        save_data = {"config": config, **results}
        json.dump(save_data, write_file, indent=2)

    # Return comprehensive results
    return {
        "train_state": train_state_obj,
        "metrics": results,
        "lr_schedule": lr_schedule,
        "config": config,
    }


def generate_experiment_grid(
    configs,
    econ_models,
    neural_net_builders,
    epoch_train_functions,
    experiment_names=None,
):
    """
    Generate all combinations of experiments from lists of configs, models, builders, and functions.

    This function is designed for simplicity: just provide lists of each component and it will
    create all combinations using a Cartesian product.

    Args:
        configs: List of configuration dictionaries. Each config should be a complete dict.
        econ_models: List of economic model instances.
        neural_net_builders: List of neural network builder functions.
        epoch_train_functions: List of epoch training functions.
        experiment_names: Optional list of experiment names. If provided, must match the number
                         of total experiments. If None, auto-generates names.

    Returns:
        List of experiment dictionaries, each with:
            - name: Experiment name
            - econ_model: Economic model instance
            - neural_net_builder: Neural network builder function
            - epoch_train_fn: Epoch training function
            - config: Configuration dictionary

    Examples:
        # Create 2 configs manually
        config1 = create_base_config(..., seed=7)
        config2 = create_base_config(..., seed=42)

        # Define lists
        configs = [config1, config2]
        econ_models = [model_low_ies, model_high_ies]
        neural_net_builders = [baseline_builder]
        epoch_train_functions = [baseline_train_fn]

        # This generates 4 experiments (2 configs × 2 models × 1 builder × 1 train_fn)
        experiments = generate_experiment_grid(
            configs=configs,
            econ_models=econ_models,
            neural_net_builders=neural_net_builders,
            epoch_train_functions=epoch_train_functions
        )
    """
    # Convert single items to lists
    if not isinstance(configs, list):
        configs = [configs]
    if not isinstance(econ_models, list):
        econ_models = [econ_models]
    if not isinstance(neural_net_builders, list):
        neural_net_builders = [neural_net_builders]
    if not isinstance(epoch_train_functions, list):
        epoch_train_functions = [epoch_train_functions]

    # Generate all combinations
    experiments = []
    counter = 0

    for config, econ_model, nn_builder, train_fn in itertools.product(
        configs, econ_models, neural_net_builders, epoch_train_functions
    ):
        # Use provided name or auto-generate
        if experiment_names and counter < len(experiment_names):
            exp_name = experiment_names[counter]
        else:
            exp_name = f"exp_{counter:03d}"

        # Update config with experiment name
        exp_config = config.copy()
        exp_config["exper_name"] = exp_name

        # Build experiment dict
        experiment = {
            "name": exp_name,
            "config": exp_config,
            "econ_model": econ_model,
            "neural_net_builder": nn_builder,
            "epoch_train_fn": train_fn,
        }

        experiments.append(experiment)
        counter += 1

    # Validate experiment_names length if provided
    if experiment_names and len(experiment_names) != len(experiments):
        print(
            f"Warning: {len(experiment_names)} names provided but {len(experiments)} "
            f"experiments generated. Using auto-generated names for extras."
        )

    return experiments
