"""
Utility functions for DEQN package.

This module contains common utility functions used across different DEQN components.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import optax
import orbax.checkpoint as ocp
from flax.training import checkpoints
from flax.training.train_state import TrainState

from .neural_nets.with_loglinear_baseline import NeuralNet


def load_trained_model_GPU(experiment_name: str, save_dir: str, nn_config: Dict[str, Any]) -> TrainState:
    """Load trained model from checkpoint (GPU-compatible version).

    Args:
        experiment_name: Name of the experiment/checkpoint directory
        save_dir: Base directory where experiments are saved
        nn_config: Neural network configuration containing:
            - features: List of layer sizes including output layer
            - C: Neural network parameter C
            - policies_sd: Standard deviations for policy normalization
            - params_dtype: Data type for model parameters
        econ_model: Economic model instance with dim_states and dim_policies attributes

    Returns:
        TrainState: Loaded model ready for inference
    """
    # Create neural network with nn_config parameters
    nn = NeuralNet(
        features=nn_config["features"],
        C=nn_config["C"],
        policies_sd=nn_config["policies_sd"],
        param_dtype=nn_config["params_dtype"],
    )

    # Load checkpoint with target=None to get raw data (following Google Colab approach)
    checkpoint_dir = os.path.join(save_dir, experiment_name)
    train_state_restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir, target=None, step=None, parallel=False
    )

    # Extract parameters and optimizer state from restored checkpoint
    params = train_state_restored["params"]
    opt_state = train_state_restored["opt_state"]

    # Create new TrainState with neural network apply function (learning rate irrelevant for inference)
    train_state = TrainState.create(apply_fn=nn.apply, params=params, tx=optax.adam(0.001))
    train_state = train_state.replace(opt_state=opt_state)

    return train_state


def load_experiment_data(experiments_config: Dict[str, str], save_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load experiment results and configurations.

    Args:
        experiments_config: Dictionary mapping experiment labels to experiment names
        save_dir: Base directory where experiments are saved

    Returns:
        Dictionary containing experiment data with results, config, and model_name for each experiment
    """
    experiments_data = {}

    for experiment_label, experiment_name in experiments_config.items():
        results_path = os.path.join(save_dir, experiment_name, "results.json")

        with open(results_path, "r") as f:
            results = json.load(f)

        experiments_data[experiment_label] = {
            "results": results,
            "config": results["config"],
            "model_name": results["config"]["model_dir"],
        }

    return experiments_data


def load_trained_model_orbax(
    experiment_name: str, save_dir: str, nn_config: Dict[str, Any], state_ss: Any, step: Optional[int] = None
) -> TrainState:
    """Load trained model from Orbax checkpoint.

    Args:
        experiment_name: Name of the experiment/checkpoint directory
        save_dir: Base directory where experiments are saved
        nn_config: Neural network configuration containing:
            - features: List of layer sizes including output layer
            - C: Neural network parameter C
            - policies_sd: Standard deviations for policy normalization
            - params_dtype: Data type for model parameters
        state_ss: Steady state of the economic model (used for initialization shape)
        step: Specific checkpoint step to load (None for latest)

    Returns:
        TrainState: Loaded model ready for inference
    """
    import jax
    import jax.numpy as jnp

    nn = NeuralNet(
        features=nn_config["features"],
        C=nn_config["C"],
        policies_sd=nn_config["policies_sd"],
        param_dtype=nn_config["params_dtype"],
    )

    checkpoint_dir = Path(save_dir) / experiment_name
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir)

    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Create abstract target tree for environment-agnostic restoration
    # Use same initialization approach as during training
    # IMPORTANT: Use a schedule (not a scalar) to match the optimizer structure from training
    dummy_params = nn.init(jax.random.PRNGKey(0), jnp.zeros_like(state_ss))
    dummy_schedule = optax.constant_schedule(0.001)  # Constant schedule to match training's schedule-based optimizer
    dummy_train_state = TrainState.create(apply_fn=nn.apply, params=dummy_params, tx=optax.adam(dummy_schedule))

    # Create abstract tree structure matching the saved TrainState
    abstract_target = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, dummy_train_state)

    restored_state = checkpoint_manager.restore(step=step, args=ocp.args.StandardRestore(abstract_target))  # type: ignore

    # Restored state is a TrainState-like structure
    params = restored_state.params  # type: ignore
    opt_state = restored_state.opt_state  # type: ignore
    restored_step = restored_state.step  # type: ignore

    # Use a schedule to match the training setup (learning rate value doesn't matter for inference)
    inference_schedule = optax.constant_schedule(0.001)
    train_state = TrainState.create(apply_fn=nn.apply, params=params, tx=optax.adam(inference_schedule))
    train_state = train_state.replace(opt_state=opt_state, step=restored_step)

    return train_state
