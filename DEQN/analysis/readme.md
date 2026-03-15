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
-   `DEQN/econ_models/{MODEL_DIR}/analysis_hooks.py`

`analysis_hooks.py` is the model-local integration point for anything that cannot be expressed
purely through labeled analysis variables. Typical responsibilities are:

-   preparing model-specific analysis context from raw states/policies
-   computing analysis variables when `econ_model.get_analysis_variables()` needs extra inputs
-   choosing which states should be shocked in GIR exercises
-   running benchmark comparisons, sectoral IR plots, or other model-specific post-processing

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

The analysis script will **automatically discover and run** all registered plots.
If your model also needs custom preprocessing or benchmark logic, put that in
`analysis_hooks.py` rather than in `DEQN/analysis.py`.

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

# 2. Import optional model-specific hooks and plots registry
analysis_hooks = load_model_analysis_hooks(MODEL_DIR)
plots_module = importlib.import_module(f"DEQN.econ_models.{MODEL_DIR}.plots")
MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", [])

# 3. Collect simulation data
analysis_variables_data = {}  # For general analysis
raw_simulation_data = {}      # For model-specific analysis

for experiment in experiments:
    simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
        ...,
        analysis_hooks=analysis_hooks,
    )

    # Store for general analysis (uses labels from get_analysis_variables)
    analysis_variables_data[exp_name] = simul_analysis_variables

    # Store raw data for model-specific plots
    raw_simulation_data[exp_name] = {
        "simul_obs": simul_obs,
        "simul_policies": simul_policies,
        "simul_analysis_variables": simul_analysis_variables,
    }

# 4. Let the model hooks run optional post-processing
model_specific_results = run_model_postprocess(
    analysis_hooks=analysis_hooks,
    ...
)

# 5. Generate general analysis
create_descriptive_stats_table(analysis_variables_data, ...)
plot_ergodic_histograms(analysis_variables_data, ...)

# 6. Generate model-specific plots (AUTOMATIC!)
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

Your model must implement either:

```python
class YourModel:
    def get_analysis_variables(self, state_logdev, policies_logdev):
        ...
```

or a model-local hook that adapts extra arguments:

```python
def compute_analysis_variables(econ_model, state_logdev, policy_logdev, analysis_context):
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

1. Put preprocessing, benchmark loading, and other model-aware logic in `analysis_hooks.py`
2. Use the **standardized signature** shown above for plot functions
3. Register plot functions in `MODEL_SPECIFIC_PLOTS`
4. Extract what you need from `econ_model` or `analysis_context`
5. Use raw simulation data (`simul_obs`, `simul_policies`, `simul_analysis_variables`)

## Migration Guide

### For Users

If you're using the analysis script:

-   The supported path is the new hook-based structure:
    `DEQN/econ_models/{MODEL_DIR}/analysis_hooks.py`
-   `python -m DEQN.analysis` remains the main entrypoint
-   Fully migrated models should work through the shared analysis pipeline without modifying `analysis.py`

Compatibility status for legacy models:

-   Older `RbcProdNet*` variants are only partially migrated
-   Some legacy models and standalone analysis scripts are in transitional support only
-   If a legacy model still depends on old price-weight assumptions or old helper contracts,
    it should be treated as not fully supported until it is migrated to `analysis_hooks.py`

### For Developers Adding New Models

1. Implement the minimal model interface needed by `DEQN/algorithm/`
2. Expose labeled analysis variables either directly on the model or through `analysis_hooks.py`
3. Use general analysis functions from `DEQN/analysis/`
4. Add model-specific plots with standardized signature
5. Register them in `MODEL_SPECIFIC_PLOTS`

### For Adding New General Analysis

1. Add function to `DEQN/analysis/plots.py` or `DEQN/analysis/tables.py`
2. Use labels from `analysis_variables_data` dictionary
3. Make it work with any model - no model-specific logic

### For Adding Model-Specific Analysis

1. Add hook functions to `DEQN/econ_models/{MODEL_DIR}/analysis_hooks.py` if you need model-aware preprocessing
2. Add plot functions to `DEQN/econ_models/{MODEL_DIR}/plots.py`
3. Register them in `MODEL_SPECIFIC_PLOTS`
4. The shared `analysis.py` should remain unchanged

## Benefits

1. **Separation of Concerns**: General vs. model-specific logic is clearly separated
2. **Reusability**: General functions work across all models automatically
3. **Flexibility**: Easy to add model-specific analysis without modifying general code
4. **Maintainability**: Changes to general analysis don't affect model-specific code
5. **Consistency**: All models use same general analysis, ensuring comparability
6. **Automatic Discovery**: Just register a plot - no need to modify analysis.py!
7. **Standardized Interface**: All model-specific plots follow the same signature
