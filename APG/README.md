# Analytical Policy Gradient (APG) - Modular Implementation

This directory contains a modular implementation of the Analytical Policy Gradient algorithm for training neural networks to solve nonlinear Real Business Cycle (RBC) models.

## Structure

The code has been organized into the following modules:

### Core Components

-   **`neural_nets/neural_nets.py`**: Contains the `ActorCritic` neural network implementation
-   **`environemnts/RbcMultiSector.py`**: Contains the `RbcMultiSector` environment implementation
-   **`algorithm/`**: Core algorithm components
    -   `simulation.py`: Simmulation, or the runner of the algorithm ( `create_simul_episode_fn`)
    -   `loss.py`: Loss function, or the objective function of the algorithm (`create_episode_loss_fn`)
    -   `epoch_train.py`: Training function, or workflow of the algorithm (`get_apg_train_fn`)
    -   `eval.py`: Evaluation, or the function that evaluates the performance of the algorithm (`get_eval_fn`)
-   **`utilities/plot_results.py`**: Plotting and visualization functions

### Main Script

-   **`apg_run.py`**: Complete experiment script that imports all modules and runs the training

## Usage

### Running in Google Colab (Recommended)

The easiest way to run this code is in Google Colab:

1. **First Cell - Setup and Clone Repository**: Copy and paste this into the first cell of your Colab notebook:

```python
# Clone the jaxecon repository
! git clone https://github.com/MatiasCovarrubias/jaxecon

# Add the repository to Python path
import sys
sys.path.insert(0, '/content/jaxecon')

# Basic imports and setup
import jax
from jax import config as jax_config
print("JAX devices:", jax.devices())
print("Repository cloned and ready!")
```

2. **Second Cell - Run the APG Algorithm**: Copy and paste the entire contents of `apg_run.py` into a second cell

3. **Run both cells** - The first cell will clone the repository and set up the environment, then the second cell will automatically:
    - Import all modular components from the cloned repository
    - Test the environment and neural network
    - Configure and run the training experiment
    - Generate and display plots
    - Save results and model checkpoints

The algorithm will run completely automatically once the repository is properly set up!

### What the Script Does

When you run `apg_run.py`, it will:

1. **Test Setup**: Verify that the environment and neural network work correctly
2. **Configuration**: Set up learning rates, network architecture, and training parameters
3. **Training**: Run the APG algorithm for the specified number of epochs
4. **Evaluation**: Test the trained policy and generate performance metrics
5. **Visualization**: Create plots showing training progress and policy performance
6. **Results**: Save all results, plots, and model checkpoints to a `results/` folder

## Dependencies

The script automatically handles dependencies, but requires:

-   JAX (for fast numerical computing)
-   Flax (for neural networks)
-   Optax (for optimization)
-   Matplotlib (for plotting)

These are typically pre-installed in Google Colab or will be installed automatically.
