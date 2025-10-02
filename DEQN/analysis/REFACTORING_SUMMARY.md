# Analysis Refactoring Summary

## What Changed

The analysis system has been refactored to clearly separate **general analysis** from **model-specific analysis**, with an **automatic discovery** registry pattern.

## File Structure

### New/Modified Files

#### General Analysis (Model-Agnostic)

-   ✅ **`DEQN/analysis/tables.py`** (NEW)

    -   General table functions moved here from model directories
    -   `create_descriptive_stats_table()`
    -   `create_comparative_stats_table()`
    -   `create_welfare_table()`
    -   `create_stochastic_ss_table()`

-   ✅ **`DEQN/analysis/plots.py`** (UPDATED)

    -   Removed model-specific `plot_sectoral_capital_mean()`
    -   Kept general functions:
        -   `plot_ergodic_histograms()`
        -   `plot_gir_responses()`
        -   `plot_gir_heatmap()`

-   ✅ **`DEQN/analysis/README.md`** (NEW)
    -   Documentation of the new architecture
    -   Usage patterns and examples

#### Model-Specific Analysis

-   ✅ **`DEQN/econ_models/RbcProdNet_Oct2025/plots.py`** (UPDATED)

    -   Added `plot_sectoral_capital()` with standardized signature
    -   Takes raw `simul_obs`, `simul_policies`, `simul_analysis_variables`
    -   **Registry pattern**: Added `MODEL_SPECIFIC_PLOTS` list for auto-discovery
    -   Kept `plot_upstreamness()` - model-specific

-   ✅ **`DEQN/econ_models/RbcProdNet_Oct2025/tables.py`** (UPDATED)
    -   Added deprecation notice
    -   Functions kept for backward compatibility
    -   Users should import from `DEQN/analysis/tables.py`

#### Main Analysis Script

-   ✅ **`DEQN/analysis.py`** (UPDATED)
    -   Imports general functions from `DEQN/analysis/`
    -   **Auto-discovers** model-specific plots via `MODEL_SPECIFIC_PLOTS` registry
    -   No need to manually import/call each model-specific plot!
    -   Clearly separates general and model-specific analysis sections

## Key Design Principles

### 1. General Analysis Functions

-   Work with **any model** using labels from `econ_model.get_analysis_variables()`
-   Take minimal preprocessing - work with raw simulation data
-   Input: `analysis_variables_data = {exp_name: {var_label: array}}`

### 2. Model-Specific Functions

-   Located in model directories: `DEQN/econ_models/{MODEL_DIR}/`
-   **Standardized signature** for automatic discovery
-   Take raw simulation data: `simul_obs`, `simul_policies`, `simul_analysis_variables`
-   Access model attributes via `econ_model` parameter
-   Extract/compute model-specific information

### 3. Registry Pattern (NEW!)

-   Model-specific plots registered in `MODEL_SPECIFIC_PLOTS` list
-   Analysis script **automatically discovers and runs** all registered plots
-   **No need to modify `analysis.py` when adding new plots!**

## Data Flow

```
Simulation
    ↓
simul_obs, simul_policies
    ↓
econ_model.get_analysis_variables()
    ↓
simul_analysis_variables (dict with labels as keys)
    ↓
    ├── General Analysis (uses labels from dict)
    │   ├── Histograms
    │   ├── Descriptive stats tables
    │   ├── GIR plots
    │   └── Welfare/Stochastic SS tables
    │
    └── Model-Specific Analysis (uses raw data)
        ├── Auto-discover via MODEL_SPECIFIC_PLOTS registry
        ├── Sectoral capital plot
        ├── Upstreamness analysis (if needed)
        └── Other registered plots
```

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
4. Register in `MODEL_SPECIFIC_PLOTS` - done!

### For Adding New General Analysis

1. Add function to `DEQN/analysis/plots.py` or `DEQN/analysis/tables.py`
2. Use labels from `analysis_variables_data` dictionary
3. Make it work with any model - no model-specific logic

### For Adding Model-Specific Analysis

1. Add function to `DEQN/econ_models/{MODEL_DIR}/plots.py` with standardized signature:
    ```python
    def plot_name(
        simul_obs, simul_policies, simul_analysis_variables,
        save_path, analysis_name, econ_model, experiment_label,
        **kwargs
    ):
        # Your code here
    ```
2. Register it in `MODEL_SPECIFIC_PLOTS`:
    ```python
    MODEL_SPECIFIC_PLOTS = [
        {"name": "my_plot", "function": plot_name, "description": "..."},
    ]
    ```
3. **That's it!** The analysis script automatically discovers and runs it
4. **No need to modify `analysis.py` at all!**

## Example: Sectoral Capital Plot

### Before (Mixed in general analysis)

```python
# In DEQN/analysis/plots.py - BAD
def plot_sectoral_capital_mean(analysis_results, sector_labels, ...):
    # Expects preprocessed sectoral_capital_mean
    capital_data = analysis_results[exp_name]["sectoral_capital_mean"]

# In analysis.py - BAD (manually imported and called)
from DEQN.analysis.plots import plot_sectoral_capital_mean

# Manual preprocessing
sectoral_capital_data = {
    exp: {"sectoral_capital_mean": ...} for exp in experiments
}

# Manual call
plot_sectoral_capital_mean(sectoral_capital_data, econ_model.labels, ...)
```

### After (Clean separation with auto-discovery)

```python
# In DEQN/econ_models/RbcProdNet_Oct2025/plots.py - GOOD
def plot_sectoral_capital(
    simul_obs, simul_policies, simul_analysis_variables,
    save_path, analysis_name, econ_model, experiment_label, **kwargs
):
    # Extracts what's needed from econ_model and raw data
    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels
    sectoral_capital_mean = jnp.mean(simul_obs, axis=0)[:n_sectors]
    # ... create plot ...
    plt.savefig(save_path)

# Register it - analysis.py auto-discovers it!
MODEL_SPECIFIC_PLOTS = [
    {
        "name": "sectoral_capital",
        "function": plot_sectoral_capital,
        "description": "Bar plot of sectoral capital"
    }
]

# In analysis.py - GOOD (automatic discovery and execution)
MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", [])

for plot_spec in MODEL_SPECIFIC_PLOTS:
    for exp_name, data in raw_simulation_data.items():
        plot_spec["function"](
            simul_obs=data["simul_obs"],
            simul_policies=data["simul_policies"],
            simul_analysis_variables=data["simul_analysis_variables"],
            save_path=f"plots/{plot_spec['name']}_{exp_name}.png",
            analysis_name=config["analysis_name"],
            econ_model=econ_model,
            experiment_label=exp_name,
        )
```

## Benefits

1. **Clarity**: Clear separation of general vs. model-specific logic
2. **Reusability**: General functions work across all models automatically
3. **Maintainability**: Easier to update general analysis without affecting models
4. **Extensibility**: Easy to add new model-specific analysis
5. **Consistency**: All models use same general analysis framework
6. **Automatic Discovery**: Registry pattern - just register a plot, no need to modify analysis.py!
7. **Standardized Interface**: All model-specific plots follow the same signature
8. **Zero Boilerplate**: Add a plot function, register it, done. No imports or manual calls needed.
