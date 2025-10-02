# Orbax Checkpoint Migration Guide

This guide explains how to migrate from Flax checkpointing to Orbax checkpointing.

## Overview

Flax's checkpointing API (`flax.training.checkpoints`) has been deprecated in favor of Orbax, which provides better forward compatibility and more features. This library now provides both the old Flax-based functions (for backward compatibility) and new Orbax-based functions.

## New Functions

### 1. Training with Orbax: `run_experiment_orbax()`

Located in `DEQN/training/runner.py`

**Usage:**

```python
from DEQN.training.runner import run_experiment_orbax
from DEQN.algorithm.epoch_train import create_epoch_train_fn

# Your config, model, and neural network setup
config = {...}
econ_model = YourModel(...)
neural_net = YourNeuralNet(...)
epoch_train_fn = create_epoch_train_fn

# Run experiment with Orbax checkpointing
results = run_experiment_orbax(config, econ_model, neural_net, epoch_train_fn)
```

**Changes from `run_experiment()`:**

-   Uses `orbax.checkpoint.StandardCheckpointer()` instead of `flax.training.checkpoints`
-   Checkpoints are saved in subdirectories named by step number (e.g., `experiment_name/1000/`, `experiment_name/2000/`)
-   Restoration uses `ocp.utils.checkpoint_steps()` to find available checkpoints
-   All other functionality remains identical

### 2. Loading Models with Orbax: `load_trained_model_orbax()`

Located in `DEQN/utils.py`

**Usage:**

```python
from DEQN.utils import load_trained_model_orbax

nn_config = {
    "features": [32, 32, n_policies],
    "C": 1.0,
    "policies_sd": policies_sd,
    "params_dtype": jnp.float32,
}

# Load latest checkpoint
train_state = load_trained_model_orbax(
    experiment_name="my_experiment",
    save_dir="./results/",
    nn_config=nn_config,
    step=None  # None loads latest checkpoint
)

# Load specific checkpoint step
train_state = load_trained_model_orbax(
    experiment_name="my_experiment",
    save_dir="./results/",
    nn_config=nn_config,
    step=5000  # Load checkpoint at step 5000
)
```

**Changes from `load_trained_model_GPU()`:**

-   Uses `orbax.checkpoint.StandardCheckpointer()` for restoration
-   Can specify a particular checkpoint step or load the latest automatically
-   Path handling uses `pathlib.Path` for better cross-platform compatibility

## Checkpoint Directory Structure

### Old Flax Format

```
results/
  experiment_name/
    checkpoint_1000
    checkpoint_2000
    checkpoint_3000
```

### New Orbax Format

```
results/
  experiment_name/
    1000/
      checkpoint
      checkpoint.metadata
    2000/
      checkpoint
      checkpoint.metadata
    3000/
      checkpoint
      checkpoint.metadata
```

## Backward Compatibility

The old functions are still available and will work with existing checkpoints:

-   `run_experiment()` - Uses Flax checkpointing
-   `load_trained_model_GPU()` - Loads Flax checkpoints

## Migration Steps

### For New Experiments

Simply use the new `_orbax` functions from the start.

### For Ongoing Experiments

1. **Option A: Continue with Flax** (for backward compatibility)

    - Keep using `run_experiment()` and `load_trained_model_GPU()`
    - Your existing checkpoints will continue to work

2. **Option B: Migrate to Orbax**
    - Start using `run_experiment_orbax()` for new training runs
    - Use `load_trained_model_GPU()` to load your old checkpoint
    - Continue training with `run_experiment_orbax()`
    - Future checkpoints will be in Orbax format

### For Analysis/Inference Only

-   Old experiments: Use `load_trained_model_GPU()`
-   New experiments: Use `load_trained_model_orbax()`

## Example: Full Migration

```python
import jax.numpy as jnp
from DEQN.training.runner import run_experiment_orbax
from DEQN.utils import load_trained_model_orbax, load_trained_model_GPU
from DEQN.algorithm.epoch_train import create_epoch_train_fn

# 1. Load an old checkpoint (if continuing training)
old_checkpoint = load_trained_model_GPU(
    experiment_name="old_experiment",
    save_dir="./results/",
    nn_config=nn_config
)

# 2. Start a new experiment with Orbax
config = {
    "seed": 42,
    "exper_name": "new_experiment_orbax",
    "save_dir": "./results/",
    "restore": False,  # Set to True if continuing from old checkpoint
    "restore_exper_name": "old_experiment",
    "learning_rate": 0.001,
    "n_epochs": 100,
    "steps_per_epoch": 100,
    # ... other config parameters
}

results = run_experiment_orbax(config, econ_model, neural_net, epoch_train_fn)

# 3. Load the new Orbax checkpoint for analysis
trained_state = load_trained_model_orbax(
    experiment_name="new_experiment_orbax",
    save_dir="./results/",
    nn_config=nn_config,
    step=None  # Latest checkpoint
)
```

## Dependencies

Make sure you have Orbax installed:

```bash
pip install orbax-checkpoint
```

The library automatically handles the import and will use the appropriate checkpointing method based on which function you call.

## Troubleshooting

### "No checkpoints found" error

-   Check that the checkpoint directory exists and contains subdirectories with checkpoint files
-   For Orbax: Look for numbered subdirectories (e.g., `1000/`, `2000/`)
-   For Flax: Look for files like `checkpoint_1000`

### Incompatible checkpoint format

-   Old Flax checkpoints cannot be loaded with `load_trained_model_orbax()`
-   New Orbax checkpoints cannot be loaded with `load_trained_model_GPU()`
-   Use the appropriate loading function for your checkpoint format

### Path issues

-   Use forward slashes in paths or let Python handle it automatically
-   The new functions use `pathlib.Path` which handles cross-platform paths correctly

## Additional Resources

-   [Orbax Documentation](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html)
-   [Flax to Orbax Migration Guide](https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html)
