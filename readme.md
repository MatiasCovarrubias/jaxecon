# JaxEcon

Solution algorithms for dynamic economic models implemented in Python using [JAX](https://github.com/google/jax) for high-performance numerical computing.

## Features

-   **Multi-backend support**: Run on CPU, GPU, or TPU with a single codebase
-   **Google Colab ready**: Scripts auto-detect environment and configure themselves
-   **Modular design**: Clean separation between algorithms, models, and analysis

## Algorithms

| Algorithm         | Status         | Description                                                                 |
| ----------------- | -------------- | --------------------------------------------------------------------------- |
| [**DEQN**](DEQN/) | ✅ Production  | Deep Equilibrium Networks for solving dynamic models with continuous shocks |
| [**VFI**](VFI/)   | ✅ Complete    | Heavily parallelized Value Function Iteration optimized for GPU/TPU         |
| [**APG**](APG/)   | 🔄 In Progress | Analytical Policy Gradient for MDPs and strategic games                     |
| **PI**            | 📋 Planned     | Policy Iteration                                                            |

## Workflow

The primary workflow uses **Python scripts** (`.py` files) that automatically detect their environment and configure themselves for either local execution or Google Colab. This approach:

-   Works seamlessly with coding agents and version control
-   Eliminates synchronization issues between notebooks and scripts
-   Provides a single source of truth for experiments

### Running Scripts

**Option 1: Google Colab (Recommended for GPU/TPU)**

1. Create a new Colab notebook
2. Copy the contents of any `.py` script (e.g., `DEQN/train.py`) into a cell
3. Run — the script auto-detects Colab, installs dependencies, clones the repo, and mounts Drive

**Option 2: Local Execution**

```bash
# Setup (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Run training
python DEQN/train.py

# Run analysis
python DEQN/analysis.py
```

### Script Structure

All main scripts follow this pattern:

```python
# 1. Environment detection
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# 2. Environment-specific setup
if IN_COLAB:
    # Install deps, clone repo, mount Drive
else:
    # Configure local paths

# 3. Configuration dictionary (edit this)
config = {
    "model_dir": "...",
    "exper_name": "...",
    # ...
}

# 4. Main logic
def main():
    ...
```

### Notebooks

Jupyter notebooks (`.ipynb`) are available for interactive exploration and learning:

-   `DEQN/Rbc_CES.ipynb` — Introductory DEQN example
-   `DEQN/jaxDEQN.ipynb` — Detailed algorithm walkthrough
-   `VFI/Value_Function_Iteration_with_TPUs.ipynb` — VFI parallelization study

For serious experimentation, use the `.py` scripts.

## Repository Structure

```
jaxecon/
├── DEQN/                    # Deep Equilibrium Networks
│   ├── train.py             # Training script (local + Colab)
│   ├── analysis.py          # Analysis script (local + Colab)
│   ├── algorithm/           # Core algorithm components
│   ├── analysis/            # Analysis utilities
│   ├── econ_models/         # Economic model implementations
│   ├── neural_nets/         # Neural network architectures
│   └── training/            # Training utilities
├── VFI/                     # Value Function Iteration
├── APG/                     # Analytical Policy Gradient
│   ├── apg_run.py           # Main script (local + Colab)
│   ├── algorithm/           # Core algorithm components
│   ├── environments/        # Environment implementations
│   └── neural_nets/         # Neural network architectures
└── PI/                      # Policy Iteration (planned)
```

## Installation

### Google Colab

No installation needed — scripts handle everything automatically.

### Local Development

```bash
git clone https://github.com/MatiasCovarrubias/jaxecon.git
cd jaxecon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed setup instructions.

## Requirements

-   Python 3.9+
-   JAX 0.7+
-   Flax 0.8+
-   Optax 0.2+

## License

MIT
