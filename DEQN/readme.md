# DEQN (Deep Equilibrium Networks)

A JAX implementation of the [Deep Equilibrium Network](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575) algorithm for solving dynamic stochastic economic models with continuous shocks.

## Quick Start

### Training

```bash
# Local
python DEQN/train.py

# Colab: copy train.py contents into a cell and run
```

### Analysis

```bash
# Local
python DEQN/analysis.py

# Colab: copy analysis.py contents into a cell and run
```

### Testing

```bash
python DEQN/test.py
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Train neural network policies |
| `analysis.py` | Analyze trained models: simulations, welfare, IRs |
| `test.py` | Run diagnostic tests on trained models |

All scripts auto-detect their environment (Colab vs local).

## Structure

```
DEQN/
├── train.py               # Training script
├── analysis.py            # Analysis script
├── test.py                # Testing script
├── utils.py               # Shared utilities
│
├── algorithm/             # Core algorithm
│   ├── simulation.py      # Episode simulation
│   ├── loss.py            # Euler equation loss
│   ├── epoch_train.py     # Training loop
│   └── eval.py            # Evaluation
│
├── analysis/              # Analysis tools
│   ├── GIR.py             # Generalized Impulse Responses
│   ├── plots.py           # Visualization
│   ├── tables.py          # Results tables
│   ├── welfare.py         # Welfare calculations
│   └── stochastic_ss.py   # Stochastic steady state
│
├── econ_models/           # Economic models
│   ├── rbc_ces.py         # RBC with CES production
│   ├── rbc.py             # Basic RBC
│   └── RbcProdNet*/       # Production network models
│
├── neural_nets/           # Neural networks
│   ├── neural_nets.py     # Standard feedforward
│   └── with_loglinear_baseline.py
│
├── training/              # Training utilities
│   └── run_experiment.py
│
└── tests/                 # Test suite
    └── grid_simulation_analysis.py
```

## Configuration

Edit the `config` dictionary at the top of each script:

```python
config = {
    # Model selection
    "model_dir": "RbcProdNet_Oct2025",
    "exper_name": "baseline",
    
    # Training
    "layers": [32, 32],
    "learning_rate": 0.0005,
    "n_epochs": 100,
    # ...
}
```

## Algorithm Components

| Function | Description |
|----------|-------------|
| `create_episode_simul_fn()` | Simulates episodes given a policy |
| `create_batch_loss_fn()` | Computes Euler equation residuals |
| `create_epoch_train_fn()` | One epoch of training |
| `create_eval_fn()` | Evaluates policy accuracy |

## Economic Models

Models are Python classes implementing:
- State/control dimensions
- Transition dynamics  
- Euler equation residuals (FOCs)
- Steady state values

See `econ_models/readme.md` for details on implementing new models.

## Adding Model-Specific Analysis

Model-specific plots are auto-discovered. In your model's `plots.py`:

```python
def my_plot(simul_obs, simul_policies, simul_analysis_variables,
            save_path, analysis_name, econ_model, experiment_label, **kwargs):
    # Your plotting code
    plt.savefig(save_path)

MODEL_SPECIFIC_PLOTS = [
    {"name": "my_plot", "function": my_plot, "description": "..."},
]
```

The analysis script automatically discovers and runs registered plots.
