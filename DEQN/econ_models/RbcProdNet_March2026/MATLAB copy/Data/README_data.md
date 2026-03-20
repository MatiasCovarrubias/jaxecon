# Data Documentation for RbcProdNet Calibration

This document describes the data sources, variables, and aggregation methods used in the production network model calibration.

---

## Runtime Boundary

The `Data/` folder should be treated as **offline preprocessing and replication code**, not as part of the main runtime pipeline.

- `main.m` does **not** call the scripts in `Data/` directly.
- The runtime calibration path loads prepared artifacts such as `calibration_data.mat`, `TFP_process.mat`, and `beadat_37sec.mat`.
- If you are changing the model pipeline, prefer editing `calibration/`, `steady_state/`, `dynare/`, and `utils/`.
- Only edit the `Data/` scripts when you are intentionally regenerating the prepared input files or reproducing the empirical preprocessing steps.

This boundary matters because the `Data/` scripts are mostly legacy script-style MATLAB code with broad workspace side effects, while the main pipeline has been refactored toward explicit function contracts.

---

## 1. Data Source Files

### `beadat_37sec.mat`

**Source**: BEA National Income and Product Accounts, GDP by Industry Database, and Fixed Assets Database.

**Coverage**: 37 industries covering the entire non-farm private sector for the United States, harmonized to NAICS 1997 codes.

**Time period**: 1948-2017 (or 1948-2018, to be verified — see Known Issues below)

| Variable    | Description                                      | Units                  | Dimensions |
| ----------- | ------------------------------------------------ | ---------------------- | ---------- |
| `EMP_raw`   | Employment by industry                           | Persons or FTE         | T × 37     |
| `VA_raw`    | Real value added by industry                     | Chain-weighted dollars | T × 37     |
| `VAn`       | Nominal value added by industry                  | Current dollars        | T × 37     |
| `InvRaw`    | Real investment expenditures by industry         | Chain-weighted dollars | T × 37     |
| `Invn`      | Nominal investment expenditures by industry      | Current dollars        | T × 37     |
| `GOn`       | Nominal gross output by industry                 | Current dollars        | T × 37     |
| `TFP`       | TFP constructed using value added                | Index                  | T × 37     |
| `TFP_GO`    | TFP constructed using gross output               | Index                  | T × 37     |
| `Invprodsh` | Shares of total nominal investment production    | Shares                 | 37 × T     |
| `IOprodsh`  | Shares of total nominal intermediates production | Shares                 | 37 × T     |

### `calibration_data.mat`

**Source**: Derived from BEA data, pre-processed for model calibration.

| Variable     | Description                                         | Dimensions  |
| ------------ | --------------------------------------------------- | ----------- |
| `Cons47bea`  | Consumption expenditure by sector (production side) | T × 37      |
| `capshbea`   | Capital expenditure shares by sector                | T × 37      |
| `VA47bea`    | Value added by sector (BEA vintage)                 | T × 37      |
| `GO47bea`    | Gross output by sector (BEA vintage)                | T × 37      |
| `IOmat47bea` | Input-output matrix                                 | 37 × 37 × T |
| `invmat`     | Investment network matrix                           | 37 × 37 × T |
| `depratbea`  | Depreciation rates by sector                        | T × 37      |

### `Real GDP Components.xls`

**Source**: BEA NIPA Table 1.1.6 (Real Gross Domestic Product, Chained Dollars)

**Coverage**: Aggregate real PCE (Personal Consumption Expenditures)

| Variable | Description | Units                            |
| -------- | ----------- | -------------------------------- |
| Row 9    | Real PCE    | Billions of chained 2012 dollars |

**Note**: This is aggregate household consumption (demand side), distinct from sectoral consumption expenditure in `Cons47bea` (production side — which sector's output is consumed).

### `TFP_process.mat`

**Source**: Estimated TFP process parameters.

`TFP_process.mat` is built from the level CSVs exported by `Data/Tfp_construction.do` and estimated in `Data/TFPprep.m`.

### TFP CSV Families

The offline Stata script exports four TFP-series variants for each model family (`VA`, `GO`, `GO_noVA`):

| Series Variant | Example VA File | Construction |
| -------------- | --------------- | ------------ |
| `baseline` | `TFP_37.csv` | Raw annual TFP growth cumulated into levels |
| `smooth` | `TFP_37_sm.csv` | Uses the existing smoothed-share construction |
| `wds` | `TFP_37_wds.csv` | Winsorizes annual sectoral TFP growth within sector over time at the 5th/95th percentiles, then rebuilds levels |
| `smooth_wds` | `TFP_37_sm_wds.csv` | Smoothed-share construction plus the same winsorization rule |

The same naming scheme applies to:

- `TFP_GO_37*.csv`
- `TFP_GO_noVA_37*.csv`

### Saved Field Patterns

| Field Pattern | Dimensions | Meaning |
| ------------- | ---------- | ------- |
| `modrho*` | `37 × 1` | Sector-by-sector AR(1) persistence vector |
| `modvcv*` | `37 × 37` | Full residual covariance matrix |
| `modvcv*_diagVCV` | `37 × 37` | Diagonal covariance restriction: same variances, zero cross-sector covariances |
| `ar1resid*` | `(T-1) × 37` | Estimated AR(1) residual histories used to build `modvcv*` |

Examples:

- `modrho`, `modvcv`
- `modrho_sm`, `modvcv_sm`
- `modrho_wds`, `modvcv_wds`, `modvcv_wds_diagVCV`
- `modrho_sm_wds`, `modvcv_sm_wds`, `modvcv_sm_wds_diagVCV`
- `modrho_GO`, `modvcv_GO`, `modvcv_GO_diagVCV`
- `modrho_GO_noVA_sm_wds`, `modvcv_GO_noVA_sm_wds_diagVCV`

At runtime, `calibration/load_calibration_data.m` selects among these fields using:

- `model_type`
- `config.smooth`
- `config.wds`
- `config.diagVCV`

The runtime loader builds suffixes from those booleans:

- `smooth = true` adds `_sm`
- `wds = true` adds `_wds`
- `diagVCV = true` adds `_diagVCV` to the covariance field only

---

## 2. Variable Units and Properties

### Employment (`EMP_raw`)

- **Units**: Persons (full-time equivalent employees or headcount)
- **Property**: Additive — can be summed across sectors
- **Aggregation**: Simple sum

### Real Value Added (`VA_raw`)

- **Units**: Chain-weighted constant dollars (index-like)
- **Property**: **NOT additive** — chain-weighted series cannot be summed directly
- **Aggregation**: Törnqvist index using nominal VA shares

### Real Investment (`InvRaw`)

- **Units**: Chain-weighted constant dollars (index-like)
- **Property**: **NOT additive**
- **Aggregation**: Törnqvist index using nominal investment shares

### Nominal Variables (`VAn`, `Invn`, `GOn`)

- **Units**: Current dollars
- **Property**: Additive in levels
- **Use**: Computing shares for Törnqvist aggregation

---

## 3. Aggregation Methods

### Törnqvist Index (for VA and Investment)

The Törnqvist index is a superlative index that properly aggregates chain-weighted real series:

```
log(Y_agg(t+1)) - log(Y_agg(t)) = Σ_j [ 0.5 * (s_j(t) + s_j(t+1)) * (log(y_j(t+1)) - log(y_j(t))) ]
```

Where:

- `Y_agg` = aggregate real variable
- `y_j` = sectoral real variable
- `s_j(t)` = nominal share of sector j at time t

**For Value Added**: Use nominal VA shares (`VAsh = VAn ./ sum(VAn)`)

**For Investment**: Use nominal investment shares (`Invsh = Invn ./ sum(Invn)`)

### Simple Sum (for Employment)

Employment is in natural units (persons), so simple summation is appropriate:

```
L_agg(t) = Σ_j L_j(t)
```

### Aggregate Consumption (NIPA PCE)

Aggregate consumption uses **NIPA real PCE** directly from BEA:

- Already properly chain-weighted at the aggregate level by BEA
- No Törnqvist needed — it's a single aggregate series
- This is the standard measure in business cycle accounting

**Why not use sectoral consumption (`Cons47bea`)?**

`Cons47bea` represents consumption expenditure **by producing sector** (i.e., how much of each sector's output goes to final consumption). This is useful for calibrating consumption shares but:

- It's the production side, not household consumption
- Aggregating it would double-count (household consumption ≠ Σ sector output to consumption)

For model validation, we want household consumption (PCE), which is already aggregated.

---

## 4. Weighting Schemes for Sectoral Averages

When computing "average sectoral volatility", we weight individual sector volatilities:

```
σ_avg = Σ_j w_j * σ_j
```

### Available Weighting Schemes

| Weight             | Variable      | Formula                        | Interpretation                   |
| ------------------ | ------------- | ------------------------------ | -------------------------------- |
| VA weights         | `va_weights`  | Time-avg of `VAn / sum(VAn)`   | Economic importance (production) |
| GO weights         | `go_weights`  | Time-avg of `GOn / sum(GOn)`   | Domar-style (sales importance)   |
| Employment weights | `emp_weights` | Time-avg of `EMP / sum(EMP)`   | Labor force composition          |
| Investment weights | `inv_weights` | Time-avg of `Invn / sum(Invn)` | Capital allocation               |

### Current Implementation

| Measure                            | VA-weighted   | Own-weighted              |
| ---------------------------------- | ------------- | ------------------------- |
| Avg sectoral VA volatility         | `sigma_VA_avg` | `sigma_VA_avg`           |
| Avg sectoral labor volatility      | `sigma_L_avg` | `sigma_L_avg_empweighted` |
| Avg sectoral investment volatility | `sigma_I_avg` | `sigma_I_avg_invweighted` |

**Interpretation**:

- **VA-weighted**: "What is the average volatility contribution to aggregate value added?"
- **Employment-weighted**: "What volatility does the average worker experience?"
- **Investment-weighted**: "What volatility does the average dollar of investment experience?"

---

## 5. Business Cycle Filtering

All volatility measures use HP-filtered log levels:

```matlab
hp_lambda = 100;  % Standard for annual data
[trend, cycle] = hpfilter(log(X), hp_lambda);
sigma = std(cycle);
```

---

## 6. Computed Empirical Targets

### Aggregate Volatilities

| Data Target    | Description                        | Data Method               | Model Counterpart |
| -------------- | ---------------------------------- | ------------------------- | ----------------- |
| `sigma_VA_agg` | Aggregate GDP volatility           | HP-filter log(Tornqvist aggregate VA) | Expenditure GDP from `sum(P.*(Q-M_out))` |
| `sigma_C_agg`  | Aggregate consumption volatility   | HP-filter log(NIPA PCE)   | Expenditure C from `sum(P.*C)` |
| `sigma_L_agg`  | Aggregate employment volatility    | HP-filter log(sum(EMP))   | Exact headcount labor aggregate `log(sum(L_t)) - log(sum(L_ss))` |
| `sigma_I_agg`  | Aggregate investment volatility    | HP-filter log(Tornqvist aggregate investment) | Expenditure I from `sum(P.*I_out)` |

**Note on consumption aggregation**: The data uses NIPA chain-weighted PCE (Fisher index), while the model uses a CES aggregator. For HP-filtered log deviations, both approximate the same first-order behavior: `d log(Cagg) ≈ Σ_j s_j d log(C_j)`. The comparison is valid as long as `σ_c` is not too far from 1.

### Aggregate Correlations

The empirical-target builder also stores aggregate expenditure correlations from HP-filtered log levels:

```matlab
corr(hpfilter(log(X_agg)), hpfilter(log(Y_agg)))
```

Saved fields under `EmpiricalTargets.correlations` include:

| Data Target | Description | Data Series | Model Counterpart |
| ----------- | ----------- | ----------- | ----------------- |
| `L_C_agg` | Aggregate labor-consumption correlation | `corr(cyc_EMP_agg, cyc_Cons_agg)` | `corr_L_C_agg` |
| `I_C_agg` | Aggregate investment-consumption correlation | `corr(cyc_Inv_agg, cyc_Cons_agg)` | `corr_I_C_agg` |

On the model side, the same expenditure concepts are used:

- Simulation moments come from reconstructed aggregate log deviations:
  - `L_hc_logdev`
  - `C_logdev`
  - `I_logdev`
  - `GDP_logdev`
  - `K_logdev`
- Theoretical moments come from the implied covariance matrices of the expenditure aggregates.

### Sectoral Volatilities (Averages)

| Data Target               | Description                        | Weights            | Model Counterpart      |
| ------------------------- | ---------------------------------- | ------------------ | ---------------------- |
| `sigma_VA_avg`            | Avg sectoral value-added volatility | Nominal VA shares  | `y_j`                  |
| `sigma_L_avg`             | Avg sectoral labor volatility      | VA shares          | `l_j` (idx 112-148)    |
| `sigma_I_avg`             | Avg sectoral investment volatility | VA shares          | `i_j` (idx 297-333)    |
| `sigma_L_avg_empweighted` | Avg sectoral labor volatility      | Employment shares  | `l_j` with emp weights |
| `sigma_I_avg_invweighted` | Avg sectoral investment volatility | Nominal inv shares | `i_j` with inv weights |

For sectoral variables, compute volatility of each `VA_j`, `l_j`, and `i_j`, then take weighted average using the same weighting scheme.

### Sectoral Comovement

Sectoral comovement is stored as the average off-diagonal pairwise correlation across sectors:

```matlab
mean(corr_matrix(triu(true(size(corr_matrix)), 1)))
```

where each entry in `corr_matrix` is computed from HP-filtered log levels of the sectoral series.

Saved fields include:

| Data Target | Description | Data Series | Model Counterpart |
| ----------- | ----------- | ----------- | ----------------- |
| `avg_pairwise_corr_C` | Avg pairwise corr of sectoral consumption expenditure | `Cons47bea` | `avg_pairwise_corr_C` |
| `avg_pairwise_corr_VA` | Avg pairwise corr of sectoral value added | `VA_raw` | `avg_pairwise_corr_VA` |
| `avg_pairwise_corr_L` | Avg pairwise corr of sectoral labor | `EMP_raw` | `avg_pairwise_corr_L` |
| `avg_pairwise_corr_I` | Avg pairwise corr of sectoral investment | `InvRaw` | `avg_pairwise_corr_I` |

The full sectoral correlation matrices are also stored in the moment structs as:

- `corr_matrix_C`
- `corr_matrix_VA`
- `corr_matrix_L`
- `corr_matrix_I`

For sectoral consumption, the empirical object is `Cons47bea`, which is the production-side sectoral consumption-expenditure series from `calibration_data.mat`. This is appropriate for sectoral comovement, even though aggregate consumption validation still uses NIPA PCE.

### Labor-TFP Correlations

The runtime empirical-target builder also stores contemporaneous labor-TFP correlations computed from HP-filtered log levels:

```matlab
corr(hpfilter(log(L_j)), hpfilter(log(TFP_j)))
```

Saved fields under `EmpiricalTargets.correlations` include:

- `L_TFP_sectoral`: 1 × 37 vector of sector-by-sector labor vs VA-based TFP correlations
- `L_TFP_agg`: aggregate labor vs aggregate VA-based TFP correlation
- `L_TFP_GO_sectoral`: 1 × 37 vector of sector-by-sector labor vs GO-based TFP correlations
- `L_TFP_GO_agg`: aggregate labor vs aggregate GO-based TFP correlation

Aggregate TFP is constructed with the same chain-style aggregation logic used in the offline empirical scripts:

- VA-based TFP: Törnqvist growth aggregation with nominal VA shares
- GO-based TFP: Domar-weight growth aggregation using nominal gross output over nominal value added

**Model computation** (SS shares are the model analog to time-average shares in data):

- VA-weighted: Uses steady-state VA shares (`Y_j^ss / sum(Y_j^ss)`)
- Employment-weighted: Uses steady-state labor shares (`L_j^ss / sum(L_j^ss)`) — real units, like data
- Investment-weighted: Uses steady-state **nominal** investment shares (`(I_j^ss * P_k_j^ss) / sum(I_j^ss * P_k_j^ss)`) — matches data's use of nominal investment

For simulation-side aggregate `C`, `I`, `GDP`, `L`, and `K`, model moments are computed only on the canonical active sample `shocks_simul`. The procedure is:

1. Reconstruct the aggregate in levels each period.
2. Convert to exact log deviations from the deterministic steady state.
3. Compute moments on `shocks_simul` only.

For the expenditure aggregates this means:

- `C_nom(t) = sum_j P_j(t) * C_j(t)`
- `I_nom(t) = sum_j P_j(t) * I_j^{out}(t)`
- `GDP_nom(t) = sum_j P_j(t) * (Q_j(t) - M_j^{out}(t))`

There is no longer a separate "linearized vs nonlinear" simulation aggregate storage convention. Simulation aggregates are always exact log deviations to deterministic steady state. Theoretical moments remain separate and approximation-based.

### Files Involved If You Add More Moments

If you want to add another moment later, these are the main files in the pipeline:

| File | Role |
| ---- | ---- |
| `calibration/load_calibration_data.m` | Loads prepared datasets and passes the required raw series into the empirical-target builder |
| `calibration/compute_empirical_targets.m` | Computes and stores empirical moments from the data |
| `utils/compute_model_statistics.m` | Computes simulation-based model moments from Dynare simulated paths |
| `utils/compute_theoretical_statistics.m` | Computes theoretical model moments from the first-order state-space solution |
| `utils/print_empirical_targets.m` | Prints empirical moments at calibration load time |
| `utils/print_model_vs_empirical.m` | Prints the detailed model-vs-data comparison table |
| `utils/print_summary_table.m` | Prints the concise end-of-run summary table |

Typical workflow for a new moment:

1. Add the empirical calculation and stored field in `compute_empirical_targets.m`.
2. Add the simulation counterpart in `compute_model_statistics.m`.
3. Add the theoretical counterpart in `compute_theoretical_statistics.m` if you want it available even without simulations.
4. Expose it in `print_empirical_targets.m`, `print_model_vs_empirical.m`, and `print_summary_table.m`.

---

## 7. Known Issues and TODOs

### Dimension Mismatch

**Issue**: Different .mat files may have different time dimensions:

- `calibration_data.mat`: May have T = 71 or 72 years
- `beadat_37sec.mat`: May have T = 70, 71, or 72 years

**Current handling**: Truncate to minimum dimension:

```matlab
n_years_raw = size(VA_raw, 1);
Invn_aligned = Invn(1:n_years_raw, :);
```

**TODO**: Verify exact dimensions of each file:

```matlab
vars1 = who('-file', 'calibration_data.mat');
vars2 = who('-file', 'beadat_37sec.mat');
overlap = intersect(vars1, vars2)
```

### Variable Overwriting

**Issue**: Loading `beadat_37sec.mat` after `calibration_data.mat` may overwrite variables like `VA_raw` if both files contain them.

**TODO**: Check for overlapping variables and handle explicitly.

---

## 8. Sector List (37 Sectors)

| Index | Sector                           | Hub? |
| ----- | -------------------------------- | ---- |
| 1     | Mining                           |      |
| 2     | Utilities                        |      |
| 3     | Construction                     | ✓    |
| 4     | Wood products                    |      |
| 5     | Nonmetallic mineral products     |      |
| 6     | Primary metals                   |      |
| 7     | Fabricated metal products        |      |
| 8     | Machinery                        | ✓    |
| 9     | Computer and electronic products |      |
| 10    | Electrical equipment             |      |
| 11    | Motor vehicles                   | ✓    |
| 12-28 | Other manufacturing and services |      |
| 29    | Professional/Technical Services  | ✓    |
| 30-37 | Other services                   |      |

Hub sectors (3, 8, 11, 29) are defined based on investment network centrality (vom Lehn and Winberry, 2019).

---

## 9. References

- vom Lehn, Christian and Thomas Winberry (2019). "The Investment Network, Sectoral Comovement, and the Changing U.S. Business Cycle." _Quarterly Journal of Economics_.
- BEA GDP by Industry: https://www.bea.gov/data/gdp/gdp-industry
- BEA Fixed Assets: https://www.bea.gov/data/special-topics/fixed-assets
