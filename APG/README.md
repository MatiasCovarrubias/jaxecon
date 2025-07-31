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

### Execution Options

This implementation provides two ways to run the APG algorithm:

-   **`apg_run.py`**: Complete experiment script for **local execution**
-   **`apg_run.ipynb`**: Jupyter notebook for **Google Colab execution** (recommended for GPU usage)

## Usage

### Option 1: Local Execution (Recommended for Development)

If you want to run the algorithm locally on your machine:

1. **Prerequisites**: Ensure you have the required dependencies installed:

    ```bash
    pip install jax flax optax matplotlib
    ```

2. **Run the script**:
    ```bash
    cd APG/
    python apg_run.py
    ```

The script will automatically:

-   Import all modular components
-   Test the environment and neural network setup
-   Configure and run the training experiment
-   Generate and display plots
-   Save results and model checkpoints to the `results/` folder

### Option 2: Google Colab Execution (Recommended for GPU Training)

For GPU acceleration or if you don't want to set up dependencies locally:

1. **Open the notebook in Colab**: Click the "Open in Colab" badge in `apg_run.ipynb` or manually upload the notebook to Google Colab

2. **Set up GPU runtime** (optional but recommended):

    - Go to Runtime → Change runtime type
    - Select "GPU" as hardware accelerator

3. **Run all cells**: The notebook will automatically:
    - Clone the repository
    - Set up the Python path
    - Import all necessary components
    - Run the complete experiment with the same functionality as the local script

### What the Algorithm Does

When you run the APG algorithm (either locally or in Colab), it will:

1. **Test Setup**: Verify that the environment and neural network work correctly
2. **Configuration**: Set up learning rates, network architecture, and training parameters
3. **Training**: Run the APG algorithm for the specified number of epochs
4. **Evaluation**: Test the trained policy and generate performance metrics
5. **Visualization**: Create plots showing training progress and policy performance
6. **Results**: Save all results, plots, and model checkpoints to a `results/` folder

## When to Use Which Option

-   **Use `apg_run.py` (local execution)** when:

    -   You're developing or debugging the algorithm
    -   You want to modify the code and run experiments iteratively
    -   You have a local setup with sufficient computational resources
    -   You prefer working in your local development environment

-   **Use `apg_run.ipynb` (Colab execution)** when:
    -   You want to leverage free GPU resources from Google Colab
    -   You don't want to install dependencies locally
    -   You're sharing the experiment with others who need easy access
    -   You want to run longer experiments that benefit from GPU acceleration

## Dependencies

The algorithm requires:

-   JAX (for fast numerical computing)
-   Flax (for neural networks)
-   Optax (for optimization)
-   Matplotlib (for plotting)

These dependencies are automatically available in Google Colab, or can be installed locally using the pip command shown above.
