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

### Notes

-   The `.venv` directory is already included in `.gitignore`
-   These settings are workspace-specific and won't affect other projects
-   End users running code in Google Colab are unaffected by these development setup requirements
-   CPU-only JAX is installed for compatibility; GPU support can be added if needed for local development

### Files Added for Development

-   `requirements-dev.txt` - Development dependencies
-   `pyrightconfig.json` - Pyright/Pylance configuration
-   `.vscode/settings.json` - VS Code workspace settings
-   `DEVELOPMENT.md` - This file
