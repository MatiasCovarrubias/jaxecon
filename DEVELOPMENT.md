# Development Setup

This document describes local development and smoke validation for JaxEcon.

## Setup

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

JAX currently supports Python 3.10-3.13. The default requirements install
CPU-compatible JAX. For local GPU development, replace `jax[cpu]` with the
appropriate JAX CUDA or TPU package for the machine.

Check the environment:

```bash
python -c "from DEQN.neural_nets.neural_nets import NeuralNet; from APG.environments import RbcMultiSector; print('ok')"
```

## Public Smoke Runs

These commands should run without private data:

```bash
python -m DEQN.econ_models.RBC.train
python VFI/vfi.py
```

The New Keynesian DEQN example is also self-contained, but can take longer:

```bash
python -m DEQN.econ_models.NK.train
```

For APG, use the lightweight component smoke check:

```bash
python -m APG.smoke
```

The full `APG/train.py` runner compiles and checkpoints a heavier experiment.
Use it after the component smoke passes.

## RbcProdNet Research Runs

The canonical production-network entry points are:

```bash
python -m DEQN.train
python -m DEQN.analysis
```

They require external artifacts under the selected model folder, usually
`DEQN/econ_models/RbcProdNet_April2026/`:

- `ModelData*.mat` for training and model construction
- trained Orbax checkpoints under `experiments/` for analysis
- optional `ModelData_simulation*.mat` benchmark simulations
- optional `ModelData_IRs*.mat` benchmark impulse responses

For unattended or Runpod runs, prefer JSON configs:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
python -m DEQN.analysis_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

The layered config equivalent is:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/experiments/highsigmay.smoke.json
```

## Repository Conventions

- Run commands from the repository root.
- Use `python -m ...` for package entry points when possible.
- Keep reusable DEQN training code in `DEQN/training/`.
- Keep generic analysis in `DEQN/analysis/`.
- Put model-specific analysis integration in `DEQN/econ_models/<MODEL_DIR>/analysis_hooks.py`.
- Treat `RbcProdNet_April2026` as the current production-network reference.

## Troubleshooting

### Import errors

- Activate the virtual environment.
- Run from the repository root.
- Reinstall `requirements-dev.txt` if JAX, Flax, Optax, or Orbax imports fail.

### Missing model data

`DEQN/train.py` and `DEQN/analysis.py` are not public no-data examples. If they
fail looking for `.mat` files, either stage the RbcProdNet artifacts or run one
of the simple examples instead.

### Slow local runs

The research defaults are sized for accelerators and long experiments. For local
smoke tests, reduce epochs, steps per epoch, episodes per step, periods per
episode, and Monte Carlo draws.
