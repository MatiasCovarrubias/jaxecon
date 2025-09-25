# DEQN Tests

This subfolder contains all the test scripts and analysis tools for the DEQN framework.

## Grid Simulation Analysis

### Purpose

The grid simulation analysis framework provides comprehensive diagnostics to assess the quality and convergence properties of trained neural network policies across different simulation lengths and random seeds.

### Files

-   `grid_simulation_analysis.py` - Main Python module containing grid analysis functions
-   `grid_simulation_analysis.tex` - LaTeX documentation explaining the methodology and interpretation

### Key Features

-   **Comprehensive Coverage**: Analyzes all model variables (states, policies, aggregates) rather than specific measures
-   **Convergence Diagnostics**: IACT estimation, linear trend detection, out-of-distribution monitoring
-   **Cross-Seed Analysis**: Standard deviation across random seeds to distinguish sampling error from drift
-   **Scaling Analysis**: Log-log regression of standard deviation vs episode length
-   **Shock Validation**: Recovery and validation of AR(1) shock processes

### Usage

The grid analysis can be imported and used in any analysis script:

```python
from DEQN.tests.grid_simulation_analysis import run_seed_length_grid, _print_grid_summary

# Run grid analysis
results = run_seed_length_grid(
    econ_model=model,
    train_state=trained_state,
    base_config=config,
    lengths=[2000, 5000, 10000],
    n_seeds=16,
    burnin_fracs=[0.2],
    iact_num_batches=20
)

# Print formatted summary
_print_grid_summary(results)
```

### Key Diagnostics

1. **IACT (Integrated Autocorrelation Time)**: Measures effective sample size accounting for temporal correlation
2. **Linear Trends**: OLS slopes to detect non-stationarity or drift
3. **OOD Fractions**: Proportion of states outside typical ranges (|z| > 3, 4, 5)
4. **Shock Diagnostics**: Validation that recovered shocks match intended AR(1) process
5. **Cross-Seed Dispersion**: Standard deviations across seeds for all variable types
6. **Scaling Relationships**: Log-log slopes of SD vs episode length (should be ~-0.5 for pure sampling error)

### Interpretation

-   **IACT values >> 1**: Strong persistence, consider longer episodes
-   **Non-zero trend slopes**: Potential drift or misspecified aggregation
-   **High OOD fractions**: States frequently outside training distribution
-   **Poor shock recovery**: Mismatch with intended stochastic process
-   **Scaling slopes != -0.5**: Systematic bias beyond sampling error
