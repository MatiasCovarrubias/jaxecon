# DEQN (Deep Equilibrium Networks)

DEQN is the mature core of JaxEcon. It solves dynamic stochastic economic models
by training neural-network policy approximations against equilibrium residuals.

## Two Workflows

### Public/simple models

These models are pure Python/JAX and do not require private `.mat` files:

```bash
python -m DEQN.econ_models.RBC.train
python -m DEQN.econ_models.NK.train
```

They use `DEQN/neural_nets/neural_nets.py`, analytical or internally defined
steady states, and the shared algorithm components in `DEQN/algorithm/`.

### RbcProdNet research workflow

The production-network workflow uses the shared DEQN machinery but depends on
upstream MATLAB/Dynare artifacts:

```bash
python -m DEQN.train
python -m DEQN.analysis
```

The active reference model is `DEQN/econ_models/RbcProdNet_April2026`. Training
loads a `ModelData*.mat` object for parameters, steady states, normalization, and
the loglinear policy matrix `Solution.StateSpace.C`. Analysis can run DEQN-only
simulation objects, but benchmark overlays and common-shock comparisons depend on
MATLAB simulation and IR files.

## Entry Points

| Entry point | Purpose | Data requirements |
| --- | --- | --- |
| `DEQN/econ_models/RBC/train.py` | Pure-Python DEQN RBC example | None |
| `DEQN/econ_models/NK/train.py` | Pure-Python DEQN New Keynesian example | None |
| `DEQN/train.py` | RbcProdNet training orchestrator | `ModelData*.mat` |
| `DEQN/analysis.py` | RbcProdNet analysis and reporting | Checkpoints plus `ModelData*.mat`; optional benchmark files |
| `DEQN/train_importconfig.py` | JSON-config training wrapper | Same as selected workflow |
| `DEQN/analysis_importconfig.py` | JSON-config analysis wrapper | Same as selected workflow |
| `DEQN/test.py` | RbcProdNet diagnostics | Trained checkpoint and model data |

## Structure

```text
DEQN/
├── algorithm/             # Episode simulation, Euler loss, training, evaluation
├── analysis/              # Generic analysis, GIR, stochastic SS, tables, plots
├── configs/               # JSON overlays for reproducible RbcProdNet runs
├── econ_models/           # Model implementations and model-specific hooks
├── neural_nets/           # Plain MLP and loglinear-baseline architectures
├── tests/                 # Diagnostics and model-specification checks
├── training/              # Experiment runner, checkpoint loading, plots
├── train.py               # RbcProdNet training entry point
└── analysis.py            # RbcProdNet analysis entry point
```

## Configuration

Local and Colab workflows use editable `config` dictionaries near the top of the
script being run. For unattended or Runpod runs, use JSON overrides:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
python -m DEQN.analysis_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

JSON files are merged into the script defaults with `DEQN/import_config.py`. The
script dictionaries remain the canonical defaults; JSON files should contain only
the experiment-specific overrides needed for a reproducible run.

## Algorithm Components

| Function | Description |
| --- | --- |
| `create_episode_simul_fn()` | Simulates trajectories under a policy |
| `create_batch_loss_fn()` | Computes equilibrium residual losses |
| `create_epoch_train_fn()` | Runs one training epoch |
| `create_eval_fn()` | Evaluates policy accuracy |

## Economic Models

Economic models live under `DEQN/econ_models/`. A model exposes a `Model` class
with state and policy normalization, transition dynamics, shock sampling, and the
equilibrium residual loss. See `DEQN/econ_models/readme.md` for the full model
contract.

## Analysis Architecture

The supported analysis architecture is:

- `DEQN/analysis/` for generic analysis logic
- `DEQN/econ_models/{MODEL_DIR}/analysis_hooks.py` for model-specific context and post-processing
- optional model-local `plot_helpers.py`, `matlab_irs.py`, and `aggregation.py` when generic analysis is not enough

Older `RbcProdNet*` folders and standalone analysis scripts are transitional.
They may still be useful for research history, but `RbcProdNet_April2026` is the
current reference implementation.
