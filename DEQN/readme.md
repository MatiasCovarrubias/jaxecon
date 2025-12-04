# DEQN (Deep Equilibrium Networks)

A JAX implementation of the [Deep Equilibrium Network](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575) algorithm for solving dynamic stochastic economic models with continuous shocks.

## Quick Start

### Training a Model

**Google Colab:**
1. Create a new Colab notebook
2. Copy the contents of `train.py` into a cell
3. Edit the `config` dictionary at the top to select your model and experiment settings
4. Run — results are saved to Google Drive

**Local:**
```bash
source .venv/bin/activate
python DEQN/train.py
```

### Analyzing Results

**Google Colab / Local:** Same workflow as training, using `analysis.py`

```bash
python DEQN/analysis.py
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Train neural network policies for economic models |
| `analysis.py` | Analyze trained models: simulations, welfare, impulse responses |

Both scripts auto-detect their environment (Colab vs local) and configure paths, dependencies, and storage accordingly.

### Configuration

Edit the `config` dictionary at the top of each script:

```python
config = {
    # Model selection
    "model_dir": "RbcProdNet_Oct2025",    # Which model to use
    "exper_name": "baseline",              # Experiment name for saving
    
    # Training parameters
    "layers": [32, 32],                    # Neural network architecture
    "learning_rate": 0.0005,
    "n_epochs": 100,
    # ...
}
```

## Structure

```
DEQN/
├── train.py               # Training script (local + Colab)
├── analysis.py            # Analysis script (local + Colab)
├── algorithm/             # Core algorithm components
│   ├── simulation.py      # Episode simulation
│   ├── loss.py            # Euler equation loss functions
│   ├── epoch_train.py     # Training loop
│   └── eval.py            # Evaluation
├── econ_models/           # Economic model implementations
│   ├── rbc_ces.py         # RBC with CES production
│   ├── rbc.py             # Basic RBC model
│   └── RbcProdNet*/       # Production network models
├── neural_nets/           # Neural network architectures
│   ├── neural_nets.py     # Standard feedforward
│   └── with_loglinear_baseline.py  # With loglinear residual
├── analysis/              # Post-training analysis tools
│   ├── GIR.py             # Generalized Impulse Responses
│   ├── plots.py           # Visualization
│   └── tables.py          # Results tables
└── training/              # Training utilities
    └── run_experiment.py  # Experiment orchestration
```

## Algorithm Components

| Function | Description |
|----------|-------------|
| `create_episode_simul_fn()` | Simulates episodes given a policy network |
| `create_batch_loss_fn()` | Computes batch loss from Euler equation residuals |
| `create_epoch_train_fn()` | Orchestrates one epoch of training |
| `create_eval_fn()` | Evaluates policy accuracy |

## Economic Models

Models are Python classes implementing:
- State and control variable dimensions
- Transition dynamics
- Euler equation residuals (FOCs)
- Steady state values

Available models:
- `rbc_ces.py` — RBC with CES production function
- `rbc.py` — Basic RBC model
- `rbc_twosectors.py` — Two-sector RBC
- `rbc_multi_sec.py` — Multi-sector RBC

## Notebooks (for Learning)

| Notebook | Description |
|----------|-------------|
| `Rbc_CES.ipynb` | Introductory example |
| `jaxDEQN.ipynb` | Detailed algorithm walkthrough |
| `analysis.ipynb` | Interactive analysis examples |

For serious experimentation, use `train.py` and `analysis.py`.
