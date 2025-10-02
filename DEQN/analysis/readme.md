# Analysis Module

This directory contains **general analysis functions** that work with any economic model following the DEQN framework.

## Architecture

The analysis system follows a two-tier architecture:

### 1. General Analysis (in `DEQN/analysis/`)

General analysis functions that work with **any model** using labels from `econ_model.get_analysis_variables()`:

#### Tables (`tables.py`)

-   `create_descriptive_stats_table()` - Mean, SD, skewness, kurtosis
-   `create_comparative_stats_table()` - Compare statistics across experiments
-   `create_welfare_table()` - Welfare loss analysis
-   `create_stochastic_ss_table()` - Stochastic steady state values

#### Plots (`plots.py`)

-   `plot_ergodic_histograms()` - Distribution histograms for analysis variables
-   `plot_gir_responses()` - Generalized impulse response plots
-   `plot_gir_heatmap()` - Heatmap of GIR responses across sectors

**Key principle:** These functions take minimal preprocessing. They work directly with:

-   `analysis_variables_data: {exp_name: {var_label: array}}`
-   `gir_data: {exp_name: {state_name: {"gir_analysis_variables": {var_label: array}}}}`

### 2. Model-Specific Analysis (in `DEQN/econ_models/{MODEL_DIR}/`)

Model-specific plots and tables that require knowledge of the model structure.

#### Location

-   `DEQN/econ_models/{MODEL_DIR}/plots.py`
-   `DEQN/econ_models/{MODEL_DIR}/tables.py`

#### Registry Pattern (Automatic Discovery!)

Model-specific plots are **automatically discovered** via a registry. In your model's `plots.py`:

```python
# Define your plot function with standardized signature
def plot_sectoral_capital(
    simul_obs, simul_policies, simul_analysis_variables,
    save_path, analysis_name, econ_model, experiment_label,
    **kwargs
):
    # Extract what you need from econ_model
    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels

    # Create your plot using raw simulation data
    sectoral_capital = jnp.mean(simul_obs, axis=0)[:n_sectors]
    # ... plotting code ...
    plt.savefig(save_path, dpi=300)

# Register the plot - analysis.py will auto-discover it!
MODEL_SPECIFIC_PLOTS = [
    {
        "name": "sectoral_capital",
        "function": plot_sectoral_capital,
        "description": "Bar plot of mean sectoral capital",
    },
]
```

The analysis script will **automatically discover and run** all registered plots!

#### Standardized Signature

All model-specific plot functions must have this signature:

```python
def plot_name(
    simul_obs: jnp.ndarray,              # Required: raw observations
    simul_policies: jnp.ndarray,         # Required: raw policies
    simul_analysis_variables: dict,      # Required: analysis variables
    save_path: str,                      # Required: where to save
    analysis_name: str,                  # Required: analysis identifier
    econ_model: Any,                     # Required: access model attributes
    experiment_label: str,               # Required: experiment identifier
    **kwargs                             # Optional: extras
)
```

## Usage Pattern

### In `analysis.py`:

```python
# 1. Import general functions
from DEQN.analysis.tables import (
    create_descriptive_stats_table,
    create_welfare_table,
    create_stochastic_ss_table,
)
from DEQN.analysis.plots import (
    plot_ergodic_histograms,
    plot_gir_responses,
)

# 2. Import model-specific plots registry (automatically discovers all plots!)
plots_module = importlib.import_module(f"DEQN.econ_models.{MODEL_DIR}.plots")
MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", [])

# 3. Collect simulation data
analysis_variables_data = {}  # For general analysis
raw_simulation_data = {}      # For model-specific analysis

for experiment in experiments:
    simul_obs, simul_policies, simul_analysis_variables = simulation_analysis(...)

    # Store for general analysis (uses labels from get_analysis_variables)
    analysis_variables_data[exp_name] = simul_analysis_variables

    # Store raw data for model-specific plots
    raw_simulation_data[exp_name] = {
        "simul_obs": simul_obs,
        "simul_policies": simul_policies,
        "simul_analysis_variables": simul_analysis_variables,
    }

# 4. Generate general analysis
create_descriptive_stats_table(analysis_variables_data, ...)
plot_ergodic_histograms(analysis_variables_data, ...)

# 5. Generate model-specific plots (AUTOMATIC!)
for plot_spec in MODEL_SPECIFIC_PLOTS:
    plot_function = plot_spec["function"]

    for exp_name, data in raw_simulation_data.items():
        plot_function(
            simul_obs=data["simul_obs"],
            simul_policies=data["simul_policies"],
            simul_analysis_variables=data["simul_analysis_variables"],
            save_path=f"plots/{plot_spec['name']}_{exp_name}.png",
            analysis_name="my_analysis",
            econ_model=econ_model,
            experiment_label=exp_name,
        )
```

## Key Requirements

### For Economic Models

Your model must implement:

```python
class YourModel:
    def get_analysis_variables(self, state_logdev, policies_logdev, ...):
        """
        Returns:
            dict: {
                "Agg. Consumption": array,
                "Agg. Investment": array,
                ...
            }
        """
```

The keys in this dictionary become the variable labels used in all general analysis.

### For Model-Specific Functions

1. Use the **standardized signature** shown above
2. Register in `MODEL_SPECIFIC_PLOTS` list
3. Extract what you need from `econ_model` (e.g., `econ_model.n_sectors`, `econ_model.labels`)
4. Use raw simulation data (`simul_obs`, `simul_policies`, `simul_analysis_variables`)

## Migration Guide

### For Users

If you're using the analysis script:

-   ✅ No changes needed - script works as before
-   ✅ New structure is backward compatible
-   ✅ Just run `python -m DEQN.analysis` as usual

### For Developers Adding New Models

1. Implement `get_analysis_variables()` in your model
2. Use general analysis functions from `DEQN/analysis/`
3. Add model-specific plots with standardized signature
4. Register them in `MODEL_SPECIFIC_PLOTS` - that's it!

### For Adding New General Analysis

1. Add function to `DEQN/analysis/plots.py` or `DEQN/analysis/tables.py`
2. Use labels from `analysis_variables_data` dictionary
3. Make it work with any model - no model-specific logic

### For Adding Model-Specific Analysis

1. Add function to `DEQN/econ_models/{MODEL_DIR}/plots.py` with standardized signature
2. Register it in `MODEL_SPECIFIC_PLOTS` list
3. **That's it!** The analysis script automatically discovers and runs it
4. **No need to modify `analysis.py` at all!**

## Benefits

1. **Separation of Concerns**: General vs. model-specific logic is clearly separated
2. **Reusability**: General functions work across all models automatically
3. **Flexibility**: Easy to add model-specific analysis without modifying general code
4. **Maintainability**: Changes to general analysis don't affect model-specific code
5. **Consistency**: All models use same general analysis, ensuring comparability
6. **Automatic Discovery**: Just register a plot - no need to modify analysis.py!
7. **Standardized Interface**: All model-specific plots follow the same signature
