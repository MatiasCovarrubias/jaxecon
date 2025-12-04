# Analytical Policy Gradient (APG)

A policy gradient algorithm for training neural networks to solve Markov Decision Processes (MDPs) and strategic games. The key feature is that it uses differentiable step and reward functions to compute exact policy gradients via automatic differentiation.

## Quick Start

### Google Colab (Recommended for GPU/TPU)

1. Create a new Colab notebook
2. Copy the contents of `train.py` into a cell
3. Run — the script auto-detects Colab, installs dependencies, clones the repo, and mounts Drive

### Local Execution

```bash
# From repository root
python APG/train.py

# Or as module
python -m APG.train
```

## Structure

```
APG/
├── train.py                # Main training script (local + Colab)
├── algorithm/              # Core algorithm components
│   ├── __init__.py
│   ├── epoch_train.py      # Epoch training loop
│   ├── eval.py             # Evaluation functions
│   ├── loss.py             # Loss function
│   └── simulation.py       # Episode simulation
├── environments/           # Environment implementations
│   ├── __init__.py
│   └── RbcMultiSector.py   # Multi-sector RBC environment
├── neural_nets/            # Neural network architectures
│   ├── __init__.py
│   └── neural_nets.py      # Actor-Critic network
└── training/               # Training utilities
    ├── __init__.py
    ├── plots.py            # Visualization
    └── run_experiment.py   # Experiment runner
```

## Algorithm Components

| Component          | Description                                   |
| ------------------ | --------------------------------------------- |
| `simulation.py`    | Episode simulation (`create_episode_simul_fn`) |
| `loss.py`          | Policy gradient loss (`create_episode_loss_fn`) |
| `epoch_train.py`   | Training loop (`create_epoch_train_fn`)       |
| `eval.py`          | Evaluation (`create_eval_fn`)                 |

## Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    # Key configuration
    "run_name": "rbc_ms_baseline",
    "seed": 42,
    # Environment
    "n_sectors": 8,
    # Training
    "learning_rate": get_lr_schedule(),
    "n_epochs": 100,
    "steps_per_epoch": 100,
    "epis_per_step": 1024 * 8,
    "periods_per_epis": 32,
    # Neural network
    "layers_actor": [16, 8],
    "layers_critic": [8, 4],
    # ...
}
```

## How It Works

1. **Environment Setup**: Initialize multi-sector RBC environment
2. **Neural Network**: Create Actor-Critic network with configurable architecture
3. **Training**: Run APG algorithm, computing gradients through differentiable dynamics
4. **Evaluation**: Test trained policy and generate performance metrics
5. **Visualization**: Create plots showing training progress
6. **Results**: Save checkpoints and metrics to `results/` folder

## Dependencies

- JAX 0.7+
- Flax 0.8+
- Optax 0.2+
- Orbax-checkpoint
- Matplotlib

Available in Google Colab or install locally:

```bash
pip install jax flax optax orbax-checkpoint matplotlib
```
