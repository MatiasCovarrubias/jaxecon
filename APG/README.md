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

## Current Development Notes

The current APG implementation has been tested on the simple `RbcMultiSector`
environment with a small actor-critic network. The actor now follows the same
normalization convention used in DEQN:

```python
action_notnorm = action * env.policy_sd + env.policy_ss
Inv = jnp.exp(action_notnorm)
```

The actor output is interpreted as a normalized log-investment deviation. A zero
actor output therefore maps to deterministic steady-state investment, but this is
only an initialization and normalization convention. In a stochastic global
solution, the policy evaluated at the deterministic steady state need not choose
deterministic steady-state investment.

The critic/value head remains separate from the actor head. During rollouts the
critic output is scaled by `env.value_ss`, while the actor output is denormalized
through `policy_sd` and `policy_ss`.

## Convergence Diagnostics

`algorithm/eval.py` includes `create_convergence_eval_fn`, a many-rollout
gradient diagnostic intended for convergence checks. It reports:

- `actor_grad_norm` and `actor_grad_rms`: policy-gradient diagnostics for the
  actor head.
- `critic_grad_norm` and `critic_grad_rms`: value-head gradient diagnostics.
- `total_grad_norm` and `total_grad_rms`: combined actor and critic gradient
  diagnostics.
- `value_loss` and `value_accuracy`: critic target-fit diagnostics.
- `max_abs_grad`: largest absolute entry of the averaged gradient tree.

For APG, the actor gradient is the main first-order convergence diagnostic. The
critic diagnostics indicate whether the value head is fitting its rollout
targets; they are not by themselves proof that the policy is optimal.

The convergence runs use many short rollouts. This keeps the gradient estimate
representative while avoiding excessive horizon length. Recent successful runs
used:

```python
config = {
    "epis_per_step": 1024,
    "periods_per_epis": 16,
    "diag_n_epis": 16384,
    "diag_periods_per_epis": 16,
    "layers_actor": [32, 16],
    "layers_critic": [32, 16],
}
```

On the two-sector RBC test with `shock_sd=0.02`, this setup produced:

- actor gradient RMS around `1e-3`;
- value loss around `0.05`;
- value accuracy near `99.99%`;
- stable zero-shock rollouts with positive consumption.

These results suggest approximate actor convergence for the toy RBC problem, but
they should be compared against DEQN on the same environment before treating APG
as validated.

## Next Validation Step

The next planned comparison is to solve the same RBC environment with DEQN and
compare APG and DEQN policies, simulations, and diagnostics. The main hypotheses
to test are:

- APG convergence may be sensitive to initialization.
- APG may need many short rollouts rather than long rollouts.
- The current actor-critic architecture may be structurally suboptimal even if
  its size is adequate.
- DEQN can provide a mature benchmark for normalization, rollout configuration,
  and policy validation.

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
