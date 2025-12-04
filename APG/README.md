# Analytical Policy Gradient (APG)

A policy gradient algorithm for training neural networks to solve Markov Decision Processes (MDPs) and strategic games. The key feature is that it uses differentiable step and reward functions to compute exact policy gradients via automatic differentiation.

> **Note**: This algorithm is being updated to match the DEQN workflow (auto-detecting Colab/local environment). Currently, use the notebook for Colab and the script for local execution.

## Quick Start

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatiasCovarrubias/jaxecon/blob/main/APG/apg_run.ipynb)

### Local Execution

```bash
cd APG/
python apg_run.py
```

## Structure

```
APG/
├── apg_run.py              # Main script (local)
├── apg_run.ipynb           # Notebook (Colab)
├── algorithm/              # Core algorithm components
│   ├── simulation.py       # Episode simulation
│   ├── loss.py             # Loss function
│   ├── epoch_train.py      # Training loop
│   └── eval.py             # Evaluation
├── environments/           # Environment implementations
│   └── RbcMultiSector.py   # Multi-sector RBC environment
├── neural_nets/            # Neural network architectures
│   └── neural_nets.py      # Actor-Critic network
└── utilities/              # Utilities
    └── plot_results.py     # Visualization
```

## Algorithm Components

| Component | Description |
|-----------|-------------|
| `simulation.py` | Episode simulation (`create_simul_episode_fn`) |
| `loss.py` | Policy gradient loss (`create_episode_loss_fn`) |
| `epoch_train.py` | Training loop (`get_apg_train_fn`) |
| `eval.py` | Evaluation (`get_eval_fn`) |

## Configuration

Edit `get_config()` in `apg_run.py`:

```python
config_apg = {
    "learning_rate": lr_schedule,
    "n_epochs": 100,
    "steps_per_epoch": 100,
    "epis_per_step": 1024 * 8,
    "periods_per_epis": 32,
    "layers_actor": [16, 8],
    "layers_critic": [8, 4],
    # ...
}
```

## How It Works

1. **Test Setup**: Verify environment and neural network work correctly
2. **Configuration**: Set up learning rates, network architecture, training parameters
3. **Training**: Run APG algorithm, computing gradients through differentiable dynamics
4. **Evaluation**: Test trained policy and generate performance metrics
5. **Visualization**: Create plots showing training progress and policy performance
6. **Results**: Save to `results/` folder

## Dependencies

-   JAX
-   Flax
-   Optax
-   Matplotlib

Available in Google Colab or install locally:

```bash
pip install jax flax optax matplotlib
```
