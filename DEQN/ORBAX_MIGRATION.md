# Orbax Checkpoint Migration Guide

This guide explains how to migrate from Flax checkpointing to Orbax checkpointing.

## Overview

Flax's checkpointing API (`flax.training.checkpoints`) has been deprecated in favor of Orbax, which provides better forward compatibility and more features. This library now provides both the old Flax-based functions (for backward compatibility) and new Orbax-based functions.

**Note**: This library uses the **new Orbax CheckpointManager API** (post-0.5.0). The deprecated `checkpointers` parameter is not used.

## API Changes (Orbax 0.5.0+)

The library has been updated to use the refactored Orbax CheckpointManager API. Key changes:

### Before (Deprecated):

```python
# OLD - Don't use this
checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    checkpointers=ocp.StandardCheckpointer(),  # ❌ Deprecated parameter
    options=options
)
```

### After (Current):

```python
# NEW - Use this
checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    options=options  # ✅ No checkpointers parameter
)

# Saving uses args=
checkpoint_manager.save(step=step, args=ocp.args.StandardSave(train_state))

# Restoring uses args=
restored = checkpoint_manager.restore(step=step, args=ocp.args.StandardRestore(abstract_target))
```

For more details, see the [Orbax API Refactor Guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html).

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

-   Uses `ocp.CheckpointManager()` with the new API (no `checkpointers` parameter)
-   Checkpoints are saved in subdirectories named by step number (e.g., `experiment_name/1000/`, `experiment_name/2000/`)
-   Restoration uses `checkpoint_manager.latest_step()` to find the latest checkpoint
-   Uses `args=ocp.args.StandardSave()` and `args=ocp.args.StandardRestore()` for saving and restoring
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
    state_ss=econ_model.state_ss,  # Required for proper initialization
    step=None  # None loads latest checkpoint
)

# Load specific checkpoint step
train_state = load_trained_model_orbax(
    experiment_name="my_experiment",
    save_dir="./results/",
    nn_config=nn_config,
    state_ss=econ_model.state_ss,
    step=5000  # Load checkpoint at step 5000
)
```

**Changes from `load_trained_model_GPU()`:**

-   Uses `ocp.CheckpointManager()` with the new API (no `checkpointers` parameter)
-   Uses `args=ocp.args.StandardRestore()` for type-safe restoration
-   Can specify a particular checkpoint step or load the latest automatically using `checkpoint_manager.latest_step()`
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
    state_ss=econ_model.state_ss,
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

### Optimizer state structure mismatch

If you see an error like:

```
ValueError: User-provided restore item and on-disk value metadata tree structures do not match:
{'opt_state': [None, Diff(lhs=<class 'optax._src.base.EmptyState'>, rhs=<class 'dict'>)]}
```

**Cause**: The optimizer structure used during restoration doesn't match the one used during training.

**Solution**: Ensure you use a **learning rate schedule** (not a scalar) when creating the abstract target for restoration. The library now uses `optax.constant_schedule(0.001)` instead of the scalar `0.001` to match the training setup.

**Example**:

```python
# ❌ Wrong - scalar learning rate
dummy_train_state = TrainState.create(
    apply_fn=nn.apply,
    params=dummy_params,
    tx=optax.adam(0.001)  # This creates a different optimizer structure
)

# ✅ Correct - schedule learning rate
dummy_schedule = optax.constant_schedule(0.001)
dummy_train_state = TrainState.create(
    apply_fn=nn.apply,
    params=dummy_params,
    tx=optax.adam(dummy_schedule)  # This matches the training structure
)
```

### Path issues

-   Use forward slashes in paths or let Python handle it automatically
-   The new functions use `pathlib.Path` which handles cross-platform paths correctly

## Additional Resources

-   [Orbax Documentation](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html)
-   [Flax to Orbax Migration Guide](https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html)
