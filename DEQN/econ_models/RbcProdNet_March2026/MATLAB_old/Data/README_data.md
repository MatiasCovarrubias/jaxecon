# Data Documentation for RbcProdNet Calibration

This document describes the data sources, variables, and aggregation methods used in the production network model calibration.

---

## 1. Data Source Files

### `beadat_37sec.mat`

**Source**: BEA National Income and Product Accounts, GDP by Industry Database, and Fixed Assets Database.

**Coverage**: 37 industries covering the entire non-farm private sector for the United States, harmonized to NAICS 1997 codes.

**Time period**: 1948-2017 (or 1948-2018, to be verified — see Known Issues below)

| Variable | Description | Units | Dimensions |
|----------|-------------|-------|------------|
| `EMP_raw` | Employment by industry | Persons or FTE | T × 37 |
| `VA_raw` | Real value added by industry | Chain-weighted dollars | T × 37 |
| `VAn` | Nominal value added by industry | Current dollars | T × 37 |
| `InvRaw` | Real investment expenditures by industry | Chain-weighted dollars | T × 37 |
| `Invn` | Nominal investment expenditures by industry | Current dollars | T × 37 |
| `GOn` | Nominal gross output by industry | Current dollars | T × 37 |
| `TFP` | TFP constructed using value added | Index | T × 37 |
| `TFP_GO` | TFP constructed using gross output | Index | T × 37 |
| `Invprodsh` | Shares of total nominal investment production | Shares | 37 × T |
| `IOprodsh` | Shares of total nominal intermediates production | Shares | 37 × T |

### `calibration_data.mat`

**Source**: Derived from BEA data, pre-processed for model calibration.

| Variable | Description | Dimensions |
|----------|-------------|------------|
| `Cons47bea` | Consumption expenditure by sector (production side) | T × 37 |
| `capshbea` | Capital expenditure shares by sector | T × 37 |
| `VA47bea` | Value added by sector (BEA vintage) | T × 37 |
| `GO47bea` | Gross output by sector (BEA vintage) | T × 37 |
| `IOmat47bea` | Input-output matrix | 37 × 37 × T |
| `invmat` | Investment network matrix | 37 × 37 × T |
| `depratbea` | Depreciation rates by sector | T × 37 |

### `Real GDP Components.xls`

**Source**: BEA NIPA Table 1.1.6 (Real Gross Domestic Product, Chained Dollars)

**Coverage**: Aggregate real PCE (Personal Consumption Expenditures)

| Variable | Description | Units |
|----------|-------------|-------|
| Row 9 | Real PCE | Billions of chained 2012 dollars |

**Note**: This is aggregate household consumption (demand side), distinct from sectoral consumption expenditure in `Cons47bea` (production side — which sector's output is consumed).

### `TFP_process.mat`

**Source**: Estimated TFP process parameters.

| Variable | Model Type | Description |
|----------|------------|-------------|
| `modrho` | VA | TFP persistence matrix (37×37) |
| `modvcv` | VA | TFP variance-covariance matrix (37×37) |
| `modrho_GO` | GO | TFP persistence for gross output model |
| `modvcv_GO` | GO | TFP VCV for gross output model |
| `modrho_GO_noVA` | GO_noVA | TFP persistence excluding VA |
| `modvcv_GO_noVA` | GO_noVA | TFP VCV excluding VA |

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

| Weight | Variable | Formula | Interpretation |
|--------|----------|---------|----------------|
| VA weights | `va_weights` | Time-avg of `VAn / sum(VAn)` | Economic importance (production) |
| GO weights | `go_weights` | Time-avg of `GOn / sum(GOn)` | Domar-style (sales importance) |
| Employment weights | `emp_weights` | Time-avg of `EMP / sum(EMP)` | Labor force composition |
| Investment weights | `inv_weights` | Time-avg of `Invn / sum(Invn)` | Capital allocation |

### Current Implementation

| Measure | VA-weighted | Own-weighted |
|---------|-------------|--------------|
| Avg sectoral labor volatility | `sigma_L_avg` | `sigma_L_avg_empweighted` |
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

| Data Target | Description | Data Method | Model Counterpart |
|-------------|-------------|-------------|-------------------|
| `sigma_VA_agg` | Aggregate GDP volatility | HP-filter log(sum(VA)) | `yagg` (idx 484) |
| `sigma_C_agg` | Aggregate consumption volatility | HP-filter log(NIPA PCE) | `cagg` (idx 482) |
| `sigma_L_agg` | Aggregate employment volatility | HP-filter log(sum(EMP)) | `lagg` (idx 483) |
| `sigma_I_agg` | Aggregate investment volatility | HP-filter log(sum(Inv)) | `iagg` (idx 485) |
| `sigma_M_agg` | Aggregate intermediates volatility | HP-filter log(sum(GO-VA)) | `magg` (idx 486) |

**Note on consumption aggregation**: The data uses NIPA chain-weighted PCE (Fisher index), while the model uses a CES aggregator. For HP-filtered log deviations, both approximate the same first-order behavior: `d log(Cagg) ≈ Σ_j s_j d log(C_j)`. The comparison is valid as long as `σ_c` is not too far from 1.

### Sectoral Volatilities (Averages)

| Data Target | Description | Weights | Model Counterpart |
|-------------|-------------|---------|-------------------|
| `sigma_L_avg` | Avg sectoral labor volatility | VA shares | `l_j` (idx 112-148) |
| `sigma_I_avg` | Avg sectoral investment volatility | VA shares | `i_j` (idx 297-333) |
| `sigma_L_avg_empweighted` | Avg sectoral labor volatility | Employment shares | Same |
| `sigma_I_avg_invweighted` | Avg sectoral investment volatility | Nominal inv shares | Same |

For sectoral variables, compute volatility of each `l_j` and `i_j`, then take weighted average using the same weighting scheme.

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

| Index | Sector | Hub? |
|-------|--------|------|
| 1 | Mining | |
| 2 | Utilities | |
| 3 | Construction | ✓ |
| 4 | Wood products | |
| 5 | Nonmetallic mineral products | |
| 6 | Primary metals | |
| 7 | Fabricated metal products | |
| 8 | Machinery | ✓ |
| 9 | Computer and electronic products | |
| 10 | Electrical equipment | |
| 11 | Motor vehicles | ✓ |
| 12-28 | Other manufacturing and services | |
| 29 | Professional/Technical Services | ✓ |
| 30-37 | Other services | |

Hub sectors (3, 8, 11, 29) are defined based on investment network centrality (vom Lehn and Winberry, 2019).

---

## 9. References

- vom Lehn, Christian and Thomas Winberry (2019). "The Investment Network, Sectoral Comovement, and the Changing U.S. Business Cycle." *Quarterly Journal of Economics*.
- BEA GDP by Industry: https://www.bea.gov/data/gdp/gdp-industry
- BEA Fixed Assets: https://www.bea.gov/data/special-topics/fixed-assets
