# JaxEcon

JaxEcon collects JAX-based solution algorithms for dynamic economic models. The
main research workflow is **DEQN**: Deep Equilibrium Networks for continuous-state
dynamic models. The repository also contains an educational VFI implementation
and an experimental APG implementation.

## Algorithms

| Algorithm | Status | Role |
| --- | --- | --- |
| [**DEQN**](DEQN/) | Mature core | Main framework for neural-network global solutions, training, and analysis |
| [**VFI**](VFI/) | Educational | Self-contained value-function iteration example for JAX vectorization and device parallelism |
| [**APG**](APG/) | Experimental | Analytical policy-gradient prototype for differentiable environments |
| **PI** | Planned | Policy iteration placeholder |

## Quick Start

The public examples below do not require private MATLAB or Dynare artifacts.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run a simple DEQN model:

```bash
python -m DEQN.econ_models.RBC.train
```

Run the educational VFI example:

```bash
python VFI/vfi.py
```

Run the APG component smoke check:

```bash
python -m APG.smoke
```

The default APG training settings are intentionally heavier and should not be
used as a quick local smoke test.

## Research Workflow

The production-network research workflow lives in:

```text
DEQN/train.py
DEQN/analysis.py
DEQN/econ_models/RbcProdNet_April2026/
```

That workflow is not a no-data quick start. It expects upstream MATLAB/Dynare
objects such as `ModelData*.mat`, trained checkpoints for analysis, and optional
simulation/IR benchmark files. See:

- [DEQN guide](DEQN/readme.md)
- [RbcProdNet training guide](DEQN/econ_models/RbcProdNet_April2026/TRAINING_README.md)
- [RbcProdNet analysis guide](DEQN/econ_models/RbcProdNet_April2026/ANALYSIS_README.md)

## Repository Structure

```text
jaxecon/
├── DEQN/
│   ├── algorithm/       # Shared DEQN simulation, loss, training, and evaluation functions
│   ├── analysis/        # Generic analysis utilities and reporting helpers
│   ├── configs/         # JSON overlays for reproducible research runs
│   ├── econ_models/     # Economic model implementations
│   ├── neural_nets/     # Plain MLP and loglinear-baseline network architectures
│   ├── training/        # Experiment orchestration, checkpoints, and plots
│   ├── train.py         # RbcProdNet research training entry point
│   └── analysis.py      # RbcProdNet research analysis entry point
├── VFI/                 # Self-contained value-function iteration example
├── APG/                 # Experimental analytical policy-gradient implementation
└── PI/                  # Planned policy-iteration work
```

## Configuration

Most scripts keep an editable `config` dictionary near the top for local and
Colab work. Reproducible DEQN research runs can also use JSON configs:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
python -m DEQN.analysis_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

The JSON path is mainly for Runpod or unattended runs. The script-level configs
remain the canonical defaults for interactive research.

## Requirements

- Python 3.10-3.13
- JAX 0.7+
- Flax 0.8+
- Optax 0.2+
- Orbax checkpointing for saved DEQN/APG experiments

## License

MIT
