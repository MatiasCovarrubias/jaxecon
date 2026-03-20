# ModelData Structures

Saved objects:

- `ModelData.mat`
- `ModelData_simulation.mat`
- `ModelData_IRs.mat`

## `ModelData`

Core model object: metadata, calibration, steady state, solution views, summary statistics, and diagnostics.

| Field | Type | Description |
|-------|------|-------------|
| `metadata.date` | string | Experiment date label |
| `metadata.exp_label` | string | Experiment label |
| `metadata.save_label` | string | Combined experiment label |
| `metadata.model_type` | string | `'VA'`, `'GO'`, or `'GO_noVA'` |
| `metadata.smooth` | logical | `true` when the smoothed TFP series is selected |
| `metadata.wds` | logical | `true` when winsorized TFP growth is selected |
| `metadata.diagVCV` | logical | `true` when the shock covariance is forced diagonal |
| `metadata.tfp_suffix` | string | TFP data suffix implied by `smooth` and `wds` |
| `metadata.sector_indices` | vector | Analyzed sector indices |
| `metadata.sector_labels` | struct | Sector label data |
| `metadata.config` | struct | Full runtime configuration |
| `metadata.exp_paths` | struct | Experiment paths |
| `metadata.n_shocks` | int | Number of shock configurations |
| `metadata.run_flags` | struct | Final run-status flags: `{has_1storder, has_2ndorder, has_pf, has_mit}` |
| `metadata.has_irfs` | logical | Whether IRFs were packaged for this run |
| `metadata.has_diagnostics` | logical | Whether `ModelData.Diagnostics` is attached |
| `calibration` | struct | Full calibration data |
| `params` | struct | Model parameters |
| `EmpiricalTargets` | struct | Empirical target moments |
| `SteadyState.parameters` | struct | Structural parameters at SS |
| `SteadyState.policies_ss` | vector | Log steady-state policy values |
| `SteadyState.endostates_ss` | vector | Log steady-state states |
| `SteadyState.Cagg_ss` | scalar | Aggregate consumption at SS |
| `SteadyState.Lagg_ss` | scalar | Aggregate labor at SS |
| `SteadyState.Yagg_ss` | scalar | Aggregate output at SS |
| `SteadyState.Iagg_ss` | scalar | Aggregate investment at SS |
| `SteadyState.Magg_ss` | scalar | Aggregate intermediates at SS |
| `SteadyState.Kagg_ss` | scalar | Aggregate capital at steady state |
| `SteadyState.V_ss` | vector | Value function at SS |
| `Solution.StateSpace` | struct | State-space matrices (`A`, `B`, `C`, `D`) |
| `Solution.indices` | struct | Variable indices in state space |
| `Solution.steady_state` | vector | Dynare steady-state vector |
| `Statistics.TheoStats` | struct | Theoretical moments from the state-space solution |
| `Statistics.shocks_sd` | vector | Shock standard deviations |
| `Statistics.states_sd` | vector | State variable std devs |
| `Statistics.policies_sd` | vector | Policy variable std devs |
| `Statistics.FirstOrder` | struct | First-order simulation summary stats |
| `Statistics.SecondOrder` | struct | Second-order simulation summary stats |
| `Statistics.PerfectForesight` | struct | Perfect-foresight simulation summary stats |
| `Statistics.MITShocks` | struct | MIT-shocks simulation summary stats |
| `Diagnostics` | struct | Diagnostics |

## `ModelData_simulation`

Full simulation object. Contains shared solution views, shock draws, and method-specific simulation paths.

| Field | Type | Description |
|-------|------|-------------|
| `metadata.save_label` | string | Experiment label |
| `metadata.exp_paths` | struct | Output folder paths |
| `metadata.run_flags` | struct | Final run-status flags |
| `metadata.has_irfs` | logical | Whether IRFs were also packaged |
| `rng_state` | struct | RNG state for reproducibility |
| `Shared.Solution` | struct | Shared state-space solution |
| `Shared.Statistics` | struct | Shared theoretical statistics |

### Shared shocks

| Field | Type | Description |
|-------|------|-------------|
| `Shocks.data` | matrix `[simul_T × n_sectors]` | Master active shock draws from `mvnrnd` |
| `Shocks.T` | int | Number of active shock periods |
| `Shocks.rng_state` | struct | RNG state at time of generation |
| `Shocks.Sigma_A` | matrix `[n × n]` | Shock covariance matrix |
| `Shocks.usage.<Method>` | struct | `{start, end}` row range used by each method |

### Method blocks

Each of `FirstOrder`, `SecondOrder`, `PerfectForesight`, and `MITShocks` uses the same schema:

| Field | Type | Description |
|-------|------|-------------|
| `<Method>.burnin_simul` | matrix `[n_vars × burn_in]` | Stored burn-in window |
| `<Method>.shocks_simul` | matrix `[n_vars × T_active]` | Active-shock window |
| `<Method>.burnout_simul` | matrix `[n_vars × burn_out]` | Stored burn-out window |
| `<Method>.variable_indices` | struct | Variable index mapping |
| `<Method>.burn_in` | int | Burn-in periods |
| `<Method>.burn_out` | int | Burn-out periods |
| `<Method>.T_active` | int | Active-shock periods |
| `<Method>.T_total` | int | `burn_in + T_active + burn_out` |
| `<Method>.summary_stats` | struct | Simulation summary stats |

### Window conventions

- `burnin_simul` stores the zero-shock lead-in window.
- `shocks_simul` is the active-shock window.
- `burnout_simul` stores the zero-shock transition window after the active shocks.

The common timing fields are:

- `config.simul_T`
- `config.simul_burn_in`
- `config.simul_burn_out`

### MIT special case

- `MITShocks.burn_in` is always `0`
- `MITShocks.burnin_simul` is empty
- `MITShocks.T_total = T_active + burn_out`

### Aggregate moment convention

Simulation-side aggregate moments are computed from:

1. Reconstruct aggregate `C`, `I`, `GDP`, `L`, and `K` in levels period by period.
2. Convert them to `log(X_t) - log(X_ss_det)`.
3. Compute moments on `shocks_simul` only.

In `ModelData.Statistics.<Method>.ModelStats` this appears as:

- `sample_window = 'shocks_simul'`
- `aggregate_definition = 'exact_logdev_to_deterministic_ss'`
- `aggregate_moments.C`
- `aggregate_moments.I`
- `aggregate_moments.GDP`
- `aggregate_moments.L`
- `aggregate_moments.K`

Legacy comparison fields are labeled explicitly, for example:

- `sigma_L_legacy_agg`
- `sigma_M_legacy_agg`
- `sigma_Domar_avg_legacy`

Each `aggregate_moments.<X>` block stores:

- `mean`
- `std`
- `skewness`
- `kurtosis`

`PerfectForesight.CapitalStats` is attached when the PF simulation runs.

## `ModelData_IRs`

Impulse-response object. Contains one canonical artifact per shock.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `save_label` | string | Experiment label |
| `sector_indices` | vector | Analyzed sector indices |
| `ir_horizon` | int | IRF horizon (e.g., 200) |
| `shocks` | struct array `(n_shocks)` | Per-shock IR artifacts |

### Shock artifact: `shocks(i)`

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Short label, e.g. `'neg20pct'` |
| `value` | double | IRshock value (log deviation from SS) |
| `size_pct` | int | Shock size in percent (e.g., 20) |
| `sign` | int | −1 (negative) or +1 (positive) |
| `A_level` | double | TFP level: A = exp(−value) |
| `description` | string | Human-readable description |
| `sector_indices` | vector | Analyzed sectors for this shock artifact |
| `run_flags` | struct | IR methods available for this shock artifact |
| `metadata` | struct | Shock metadata |
| `entries` | struct array | Per-sector IR entries |
| `summary_stats` | struct | Per-shock summary stats |

### Per-sector entry: `shocks(i).entries(j)`

Each entry is a struct with:

| Field | Type | Description |
|-------|------|-------------|
| `sector_idx` | int | Sector number |
| `first_order` | matrix `[27 × T]` | First-order IRF |
| `second_order` | matrix `[27 × T]` | Second-order IRF |
| `perfect_foresight` | matrix `[27 × T]` | Perfect-foresight IRF |

### IRF row map

| Row | Variable | Description |
|-----|----------|-------------|
| 1 | `A_ir` | TFP level (not deviation) |
| 2 | `C_ir` | Aggregate consumption expenditure (log dev from SS) |
| 3 | `I_ir` | Aggregate investment expenditure (log dev from SS) |
| 4 | `Cj_ir` | Sectoral consumption |
| 5 | `Pj_ir` | Sectoral price |
| 6 | `Ioutj_ir` | Sectoral investment output |
| 7 | `Moutj_ir` | Sectoral intermediate output |
| 8 | `Lj_ir` | Sectoral labor |
| 9 | `Ij_ir` | Sectoral investment input |
| 10 | `Mj_ir` | Sectoral intermediate input |
| 11 | `Yj_ir` | Sectoral output |
| 12 | `Qj_ir` | Sectoral Tobin's Q |
| 13 | `A_client_ir` | Client TFP level (not deviation) |
| 14 | `Cj_client_ir` | Client consumption |
| 15 | `Pj_client_ir` | Client price |
| 16 | `Ioutj_client_ir` | Client investment output |
| 17 | `Moutj_client_ir` | Client intermediate output |
| 18 | `Lj_client_ir` | Client labor |
| 19 | `Ij_client_ir` | Client investment input |
| 20 | `Mj_client_ir` | Client intermediate input |
| 21 | `Yj_client_ir` | Client output |
| 22 | `Qj_client_ir` | Client Tobin's Q |
| 23 | `Kj_ir` | Sectoral capital (log dev from SS) |
| 24 | `GDP_ir` | Aggregate GDP expenditure (log dev from SS) |
| 25 | `Pmj_client_ir` | Client intermediate price (log dev from SS) |
| 26 | `gammaij_client_ir` | Client expenditure share deviation |
| 27 | `C_util_ir` | Utility-based aggregate consumption (log dev from SS) |

Rows 2, 3, and 24 are expenditure-based aggregates. Row 27 is utility-based aggregate consumption.
