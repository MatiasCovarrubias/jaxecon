# Python Training Guide

This document describes the Python-side DEQN training layer for `RbcProdNet_April2026`.

It complements the MATLAB documentation:

- `MATLAB/CODEBASE_README.md` explains the MATLAB runtime pipeline that builds the saved objects.
- `MATLAB/ModelData_README.md` explains the structure of the saved MATLAB model data.
- `ANALYSIS_README.md` explains what Python does after trained checkpoints exist.

This file is about producing those trained checkpoints.

## Scope

The training layer has one main job:

1. Load MATLAB steady state, statistics, and linear solution objects.
2. Build the Python economic model used inside DEQN residuals.
3. Build the neural network policy approximation.
4. Run the DEQN optimization loop.
5. Save checkpoints, metrics, configuration metadata, and training plots.

Canonical entry point:

- `DEQN/train.py`

Generic training modules:

- `DEQN/training/run_experiment.py`
- `DEQN/training/checkpoints.py`
- `DEQN/training/plots.py`

Model-specific pieces:

- `model.py`
- any MATLAB files stored in the model folder

## Relationship with MATLAB

Training depends on the MATLAB `ModelData` object, or on a differently named object with the same schema.

The current training path uses:

1. `SteadyState`
   Provides deterministic steady state states, policies, and structural parameters.
2. `Statistics`
   Provides state and policy standard deviations for normalization.
3. `Solution.StateSpace`
   Provides the linear policy matrix `C` used by the neural network baseline.

The Python boundary is:

- MATLAB computes the deterministic steady state, ergodic statistics, and linear state-space solution.
- Python loads those objects, constructs the JAX model, and trains a nonlinear global policy approximation.

## High-level flow

The main execution flow in `DEQN/train.py` is:

1. Read the top-level `config` dictionary.
2. Import the model class from `DEQN.econ_models.<model_dir>.model`.
3. Resolve the MATLAB model data file.
4. Load the configured MATLAB object from the `.mat` file.
5. Extract steady state states, policy steady states, normalization statistics, parameters, and `C`.
6. Apply optional training-only parameter overrides.
7. Build one model for training and one model for evaluation.
8. Build the neural network with the configured layer widths.
9. Run `run_experiment(...)`.
10. Save checkpoints and metrics under the experiment folder.
11. Render training plots.
12. If configured, run an IR fine-tuning stage initialized from the baseline parameters and save it as a separate experiment.

## Rolling change log

Keep only the latest three entries here. Add newest first. Keep each entry to one short bullet focused on the behavioral change, not the implementation details.

- The default training config now matches the Colab benchmark run: `ModelData_newwds_v2.mat`, model volatility scale `1.0`, and simulation volatility scale `1.0`.
- Training can now run an optional IR fine-tuning stage that samples GIR-style shocked TFP states and saves the auxiliary network with an `IR` suffix.
- Training can now load MATLAB model data from a configurable file and object key, so objects such as `ModelData_May2026` no longer need to be renamed to `ModelData`.

## Current defaults and compatibility

Training is configured directly in `DEQN/train.py`.

Important defaults:

- `model_dir` selects the model folder under `DEQN/econ_models`.
- `exper_name` selects the experiment output folder under `<model_dir>/experiments`.
- The current default `model_data_file` is `ModelData_newwds_v2.mat`. Set it to `None` to try the default files `ModelData.mat` and then `model_data.mat`.
- `model_data_object = "ModelData"` means Python expects the MATLAB object key `ModelData` inside the selected `.mat` file.
- `SolData` remains supported as a legacy fallback if the configured `model_data_object` is not present.
- `double_precision = true` enables JAX x64 and uses `jnp.float64` for loaded arrays.
- `model_vol_scale` affects the volatility used during training.
- `config_eval["simul_vol_scale"]` and the evaluation model keep evaluation separate from the training volatility scale.
- `config_ir_finetune["enabled"] = false` keeps the training path to a single baseline experiment. When set to `true`, Python trains an auxiliary IR network after the baseline run.

To train from a renamed MATLAB object, set:

```python
config = {
    "model_dir": "RbcProdNet_April2026",
    "model_data_file": "ModelData_May2026.mat",
    "model_data_object": "ModelData_May2026",
    # ...
}
```

The object must still expose the current `ModelData` schema:

- `SteadyState.parameters`
- `SteadyState.endostates_ss`
- `SteadyState.policies_ss`
- `Statistics.states_sd`
- `Statistics.policies_sd`
- `Solution.StateSpace.C`

## Main Python files

### `DEQN/train.py`

This is the training orchestrator.

It owns:

- environment setup for local and Colab runs
- top-level training configuration
- model data file resolution
- MATLAB object selection
- model construction
- neural network construction
- training dispatch
- plot dispatch

It should stay thin. Reusable training mechanics should live in `DEQN/training`.

### `DEQN/training/run_experiment.py`

This runs one configured training experiment.

It owns:

- PRNG setup
- learning rate schedule creation
- neural network initialization
- checkpoint restore
- JIT compilation
- training and evaluation loops
- checkpoint saving
- metrics collection
- machine-readable summary output

This is the right place to inspect if checkpointing, restore behavior, learning-rate decay, or metric persistence looks wrong.

### `DEQN/training/checkpoints.py`

This contains checkpoint-loading helpers used by analysis and downstream inspection code.

Use it when analysis needs to load trained experiments without repeating Orbax restore logic.

### `DEQN/training/plots.py`

This creates training diagnostics from the results returned by `run_experiment(...)`.

Current plots include:

- training metrics
- learning-rate schedule
- training summary

### `DEQN/algorithm/ir_finetune.py`

This contains the IR fine-tuning sampler.

It owns:

- baseline simulation draws used as GIR starting points
- GIR-style positive and negative shocked TFP states
- zero-shock forward rollouts from those shocked states
- batching those rollout observations into the standard DEQN loss

This file should stay aligned with `DEQN/analysis/GIR.py` on the shock convention.

## Configuration map

### Experiment identity

- `exper_name`: name of the output folder for the training run.
- `model_dir`: model folder under `DEQN/econ_models`.
- `date`: researcher-facing date or label stored with the run.
- `comment`: free-form note stored with the run.

### MATLAB input

- `model_data_file`: MATLAB `.mat` filename inside the model folder. Use `None` for default discovery.
- `model_data_object`: MATLAB object key inside the `.mat` file. Use this for objects such as `ModelData_May2026`.

### Restore

- `restore`: when `False`, initialize the neural network from scratch.
- `restore_exper_name`: experiment folder to restore from when `restore = True`.
- `restore_step`: when `True`, continue the optimizer step count from the checkpoint. When `False`, reset the step count to zero while keeping restored parameters and optimizer state.

### Economic model

- `model_param_overrides`: dictionary of training-only parameter overrides applied after loading MATLAB parameters.
- `mc_draws`: Monte Carlo draws used for the training loss.
- `init_range`: initial-state range around the steady state.
- `model_vol_scale`: volatility scale used inside the training model.
- `simul_vol_scale`: simulation volatility scale used by the training configuration.
- `config_eval`: evaluation-specific simulation settings.

### Neural network and optimizer

- `layers`: hidden layer widths. The output layer is appended automatically from `econ_model.dim_policies`.
- `learning_rate`: initial value for the cosine decay schedule.
- `double_precision`: enables x64 and loads arrays as `float64`.

### Training length

- `periods_per_epis`: periods per episode.
- `epis_per_step`: episodes per optimization step.
- `steps_per_epoch`: optimization steps per epoch.
- `n_epochs`: number of epochs.
- `checkpoint_every_n_epochs`: checkpoint and metric persistence frequency.

### IR fine-tuning

- `config_ir_finetune["enabled"]`: run the auxiliary IR fine-tuning stage after baseline training.
- `config_ir_finetune["exper_suffix"]`: suffix appended to the baseline experiment name. The default is `_IR`.
- `config_ir_finetune["min_shock_size"]`: minimum TFP shock size in percent.
- `config_ir_finetune["max_shock_size"]`: maximum TFP shock size in percent.
- `config_ir_finetune["learning_rate"]`: optional fine-tuning learning rate override.
- `config_ir_finetune["n_epochs"]`: optional fine-tuning epoch-count override.
- `config_ir_finetune["states_to_shock"]`: optional explicit state indices to shock.
- `config_ir_finetune["ir_sectors_to_plot"]`: optional sector indices, following the same convention used by GIR analysis hooks.

Any omitted training keys are inherited from the baseline config. Supported overrides include `learning_rate`, `periods_per_epis`, `epis_per_step`, `steps_per_epoch`, `n_epochs`, `checkpoint_every_n_epochs`, `mc_draws`, `init_range`, `simul_vol_scale`, and `config_eval`.

Example:

```python
config["config_ir_finetune"] = {
    "enabled": True,
    "exper_suffix": "_IR",
    "min_shock_size": 5.0,
    "max_shock_size": 25.0,
    "learning_rate": 0.0001,
    "n_epochs": 20,
}
```

Shock sizes are percentages. A `25.0` negative shock means TFP is multiplied by `0.75`. With the default symmetric GIR convention, the corresponding positive shock has the same log magnitude and multiplies TFP by `1 / 0.75`.

Derived fields:

- `periods_per_step = periods_per_epis * epis_per_step`
- `batch_size = 16`
- `n_batches = periods_per_step // 16`

For IR fine-tuning, each sampled baseline point creates both a negative and positive zero-shock rollout. The derived count is therefore:

- `periods_per_step = periods_per_epis * epis_per_step * 2`

## Output layout

Training writes into:

```text
<model_dir>/experiments/<exper_name>/
```

The folder is managed by the Orbax checkpoint manager and the training utilities. It contains the latest retained checkpoint plus generated metadata and diagnostics.

The plot step writes figures into the same experiment folder.

When IR fine-tuning is enabled, the auxiliary network writes into:

```text
<model_dir>/experiments/<exper_name>_IR/
```

or into the same pattern with the configured `exper_suffix`.

The auxiliary experiment starts from the baseline trained parameters but initializes a fresh optimizer and learning-rate schedule.

## Restore workflow

To warm start from an existing experiment:

```python
config["restore"] = True
config["restore_exper_name"] = "previous_experiment"
config["restore_step"] = False
```

Use `restore_step = False` when the goal is to reuse parameters but restart the learning-rate schedule. Use `restore_step = True` when the goal is to continue the exact optimizer schedule from the checkpoint.

IR fine-tuning does not restore from disk during the same run. It starts from the in-memory baseline parameters returned by the baseline training stage and saves its own checkpoints under the IR experiment name.

## IR fine-tuning workflow

The IR fine-tuning stage is designed for states that are important for impulse-response analysis but unlikely under the ergodic training distribution.

Each fine-tuning step:

1. Simulates baseline stochastic episodes using the current auxiliary network.
2. Draws a baseline state from each simulated episode.
3. Selects a TFP state using the same state-selection logic as GIR analysis.
4. Draws a shock size uniformly from `[min_shock_size, max_shock_size]`.
5. Creates both a negative and a positive GIR-style shocked state.
6. Rolls each shocked state forward with zero future shocks.
7. Trains the standard DEQN loss on the resulting shocked rollout states.

The sampled states are normalized before entering the neural network, but the shock itself is applied in log-level space, matching `DEQN/analysis/GIR.py`.

## Common edit points

- To train a different model version, edit `model_dir`.
- To change the MATLAB data source, edit `model_data_file`.
- To change the MATLAB object key, edit `model_data_object`.
- To run a shorter smoke test, reduce `n_epochs`, `steps_per_epoch`, `periods_per_epis`, and `epis_per_step`.
- To change the network size, edit `layers`.
- To make training easier or harder relative to benchmark volatility, edit `model_vol_scale`.
- To evaluate under a different simulation scale, edit `config_eval`.
- To train the auxiliary IR network, set `config_ir_finetune["enabled"] = True` and choose the shock support.

## Failure modes

### Model data file not found

If `model_data_file = None`, Python tries:

1. `ModelData.mat`
2. `model_data.mat`

If a custom file is configured, it is tried first. The error message lists the attempted names.

### MATLAB object key not found

If the configured object is absent, Python falls back to legacy `SolData` if present. Otherwise the error message lists the available non-metadata keys in the `.mat` file.

For a renamed object, set:

```python
config["model_data_object"] = "ModelData_May2026"
```

### Policy dimension mismatch

Some MATLAB objects include a value-function entry in `policies_ss` that is absent from `policies_sd`. Training trims both vectors to their common policy dimension and prints the alignment.

### Checkpoint restore fails

Check that:

- `restore = True`
- `restore_exper_name` points to an existing folder under `<model_dir>/experiments`
- the current network architecture is compatible with the restored checkpoint

### IR fine-tuning config is invalid

Check that:

- `min_shock_size >= 0`
- `max_shock_size > min_shock_size`
- `max_shock_size < 100`
- the resolved `states_to_shock` list is not empty
- `periods_per_epis * epis_per_step * 2` is divisible by `batch_size`

## Local and Colab runs

Local runs should be launched from the repository root:

```bash
python -m DEQN.train
```

or:

```bash
python DEQN/train.py
```

Colab runs use the environment branch in `DEQN/train.py` to install JAX, clone the repository, mount Google Drive, and set the Drive-backed model directory.

## Relationship with analysis

Training and analysis should agree on:

- `model_dir`
- the MATLAB model data file
- the MATLAB object schema
- the experiment name consumed by analysis

Training produces:

```text
<model_dir>/experiments/<exper_name>/
```

When enabled, IR fine-tuning also produces:

```text
<model_dir>/experiments/<exper_name>_IR/
```

Analysis consumes that experiment folder through its own `experiment_to_analyze` or legacy `experiments_to_analyze` config.
