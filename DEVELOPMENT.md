# Development Setup

This document describes how to set up the development environment for local development.

## Local Development Environment

The repository uses JAX for numerical computations. While end users can run the code directly in Google Colab (which has JAX pre-installed), local development requires setting up a virtual environment.

### Setup Instructions

1. **Create and activate virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements-dev.txt
    ```

    Or install manually:

    ```bash
    pip install "jax[cpu]" flax optax jaxopt matplotlib pandas scipy numpy
    ```

3. **Configure your editor:**
    - The repository includes `.vscode/settings.json` for VS Code/Cursor workspace-specific settings
    - `pyrightconfig.json` configures Pyright to use the local virtual environment
    - These settings ensure proper import resolution and type checking

### Running Code Locally

Always activate the virtual environment before running any scripts:

```bash
source .venv/bin/activate
python -c "from DEQN.neural_nets.neural_nets import NeuralNet; print('Setup working!')"
```

#### Running Analysis Scripts

For the RBC Production Network analysis script:

```bash
# Ensure you're in the repository root directory
cd /path/to/jaxecon

# Activate virtual environment
source .venv/bin/activate

# Run as a module (recommended)
python -m DEQN.econ_models.RbcProdNetv2.RbcProdNet_Analysis_Sep23_2025

# Or run directly
python DEQN/econ_models/RbcProdNetv2/RbcProdNet_Analysis_Sep23_2025.py
```

**Prerequisites for analysis scripts:**

-   Required model data files must be in `DEQN/econ_models/RbcProdNetv2/Model_Data/`
-   Trained experiment results must be in `DEQN/econ_models/RbcProdNetv2/Experiments/`
-   See [Data Dependencies](#data-dependencies) section below

#### Running Other Algorithms

For DEQN examples:

```bash
# Run the main DEQN notebook (requires Jupyter)
jupyter notebook DEQN/Rbc_CES.ipynb

# Or run as Python script
python DEQN/run_rbc_CES.py
```

For VFI examples:

```bash
jupyter notebook Value_Function_Iteration_with_TPUs.ipynb
```

For APG examples:

```bash
jupyter notebook APG/apg_run.ipynb
# Or
python APG/apg_run.py
```

### Data Dependencies

Some analysis scripts require specific data files that are not included in the repository:

#### RBC Production Network Analysis

The `RbcProdNet_Analysis_Sep23_2025.py` script requires:

1. **Model Data Files** (in `DEQN/econ_models/RbcProdNetv2/Model_Data/`):

    - `RbcProdNet_SolData_Feb21_24_baselinev3.mat` - Economic model parameters and steady state

2. **Experiment Results** (in `DEQN/econ_models/RbcProdNetv2/Experiments/`):
    - `baseline_nostateaug/` - Trained model checkpoints and results
    - `baseline_nostateaug_seed2/` - Additional seed experiments
    - `baseline_nostateaug_seed3/`
    - `baseline_nostateaug_seed4/`

Each experiment directory should contain:

-   `checkpoint_50000/` - Trained model parameters
-   `results.json` - Training configuration and results

If these files are missing, the script will display helpful error messages indicating which files are required.

### Troubleshooting

**Import Errors:**

-   Ensure virtual environment is activated
-   Verify you're running from the repository root directory
-   Check that all `__init__.py` files are present

**Missing Data Errors:**

-   Check that required data files exist in the correct directories
-   Verify file permissions allow reading

**JAX/GPU Issues:**

-   The development setup uses CPU-only JAX for compatibility
-   For GPU support, replace `jax[cpu]` with `jax[cuda]` or `jax[tpu]` in requirements

### Notes

-   The `.venv` directory is already included in `.gitignore`
-   These settings are workspace-specific and won't affect other projects
-   End users running code in Google Colab are unaffected by these development setup requirements
-   CPU-only JAX is installed for compatibility; GPU support can be added if needed for local development

### Files Added for Development

-   `requirements-dev.txt` - Development dependencies (includes scipy, jupyter)
-   `pyrightconfig.json` - Pyright/Pylance configuration
-   `.vscode/settings.json` - VS Code workspace settings
-   `DEVELOPMENT.md` - This file (enhanced with running instructions)
-   `DEQN/__init__.py` - Package initialization files for proper imports
-   `DEQN/analysis/__init__.py`
-   `DEQN/econ_models/__init__.py`
-   `DEQN/econ_models/RbcProdNetv2/__init__.py`
-   `DEQN/neural_nets/__init__.py`
