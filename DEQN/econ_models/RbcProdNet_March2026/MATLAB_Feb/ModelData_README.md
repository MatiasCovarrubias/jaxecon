# ModelData Structures

`main.m` produces three output structures, saved separately to allow loading only what is needed.

## Experiment Folder & Steady State Caching

Each run is identified by an experiment name (`config.date + config.exp_label`), which maps to a folder:

```
experiments/
    <experiment_name>/
        steady_state.mat    ← cached ModData + params
        ModelData.mat
        ModelData_simulation.mat
        ModelData_IRs.mat
        temp/
        figures/
```

**Steady state caching**: when `main.m` runs, it checks whether `steady_state.mat` already exists in the experiment folder. If it does, the cached `ModData` and `params` are loaded directly, skipping calibration entirely. This lets you switch between experiments without recalibrating.

| Config flag | Default | Behavior |
|-------------|---------|----------|
| `config.force_recalibrate` | `false` | If `false`, loads cached SS when available. If `true`, always runs calibration and overwrites the cache. |

Typical workflow:

```matlab
config.exp_label = "_baseline";          % → runs calibration on first run, caches it
config.exp_label = "_low_complementarity"; % → switch experiment, runs calibration once
config.exp_label = "_baseline";          % → switch back, loads from cache instantly
```

To force recalibration for the current experiment (e.g., after changing parameters):

```matlab
config.force_recalibrate = true;
```

---

## ModelData (core model, lightweight)

Saved as `ModelData.mat`. Contains steady state, calibration, parameters, and summary statistics.

| Field | Type | Description |
|-------|------|-------------|
| `metadata.date` | string | Experiment date label |
| `metadata.exp_label` | string | Experiment label |
| `metadata.save_label` | string | Combined label (date + exp) |
| `metadata.model_type` | string | `'VA'`, `'GO'`, or `'GO_noVA'` |
| `metadata.sector_indices` | vector | Analyzed sector indices |
| `metadata.sector_labels` | struct | Sector label data |
| `metadata.config` | struct | Full configuration (includes `shock_values`, `shock_sizes_pct`) |
| `metadata.exp_paths` | struct | Output folder paths |
| `metadata.n_shocks` | int | Number of shock configurations |
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
| `SteadyState.Kagg_ss` | scalar | Aggregate capital at SS (Σ pk_ss × K_ss) |
| `SteadyState.V_ss` | vector | Value function at SS |
| `Solution.StateSpace` | struct | State-space matrices (`A`, `B`, `C`, `D`) |
| `Solution.indices` | struct | Variable indices in state space |
| `Solution.steady_state` | vector | Dynare steady-state vector |
| `Statistics.TheoStats` | struct | Theoretical moments from state-space |
| `Statistics.shocks_sd` | vector | Shock standard deviations |
| `Statistics.states_sd` | vector | State variable std devs |
| `Statistics.policies_sd` | vector | Policy variable std devs |
| `Statistics.FirstOrder` | struct | Summary stats from 1st-order simulation (`.ModelStats`, `.states_mean/std`, `.policies_mean/std`) |
| `Statistics.SecondOrder` | struct | Summary stats from 2nd-order simulation (same sub-fields) |
| `Statistics.PerfectForesight` | struct | Summary stats from PF simulation (same sub-fields) |
| `Statistics.MITShocks` | struct | Summary stats from MIT shocks simulation (same sub-fields) |
| `Diagnostics` | struct | Nonlinearity & preallocation diagnostics |

---

## ModelData_simulation (full simulation time series, heavy)

Saved as `ModelData_simulation.mat`. Contains full simulation matrices — load only when needed.

| Field | Type | Description |
|-------|------|-------------|
| `metadata.save_label` | string | Experiment label |
| `metadata.exp_paths` | struct | Output folder paths |
| `rng_state` | struct | RNG state for reproducibility (also in `Shocks.rng_state`) |

### Shocks (shared across all simulation methods)

All simulation methods draw from a single master shock matrix. The `Shocks` struct tracks which rows each method uses.

| Field | Type | Description |
|-------|------|-------------|
| `Shocks.data` | matrix `[T_shocks × n_sectors]` | Master shock draws from `mvnrnd` |
| `Shocks.T` | int | Number of rows in master matrix (`max` of enabled simulation lengths) |
| `Shocks.rng_state` | struct | RNG state at time of generation |
| `Shocks.Sigma_A` | matrix `[n × n]` | Covariance matrix used to generate shocks |
| `Shocks.usage.FirstOrder` | struct | `{start, end}` — row range used by 1st-order simulation |
| `Shocks.usage.SecondOrder` | struct | `{start, end}` — row range used by 2nd-order simulation |
| `Shocks.usage.PerfectForesight` | struct | `{start, end}` — row range used by PF simulation |
| `Shocks.usage.MITShocks` | struct | `{start, end}` — row range used by MIT shocks simulation |

Only the `.usage` entries for methods that actually ran will be present.

### FirstOrder (if 1st-order simulation ran)

| Field | Type | Description |
|-------|------|-------------|
| `FirstOrder.full_simul` | matrix `[n_vars × T]` | Full simulation |
| `FirstOrder.variable_indices` | struct | Variable index mapping |

### SecondOrder (if 2nd-order simulation ran)

| Field | Type | Description |
|-------|------|-------------|
| `SecondOrder.full_simul` | matrix `[n_vars × T]` | Full simulation |
| `SecondOrder.variable_indices` | struct | Variable index mapping |

### PerfectForesight (if PF simulation ran)

| Field | Type | Description |
|-------|------|-------------|
| `PerfectForesight.full_simul` | matrix `[n_vars × T_active]` | Active-period simulation |
| `PerfectForesight.variable_indices` | struct | Variable index mapping |
| `PerfectForesight.burn_in` | int | Burn-in periods |
| `PerfectForesight.burn_out` | int | Burn-out periods |
| `PerfectForesight.T_active` | int | Active simulation periods |
| `PerfectForesight.T_total` | int | Total periods including burn |
| `PerfectForesight.full_simul_with_burn` | matrix `[n_vars × T_total]` | Full simulation including burn periods |
| `PerfectForesight.CapitalStats` | struct | Capital preallocation analysis |

### MITShocks (if MIT shocks simulation ran)

MIT shocks treat every shock as a surprise — agents never anticipate future shocks in their expectations. Implemented via Dynare's `perfect_foresight_with_expectation_errors_solver` using `shocks(learnt_in=t)` blocks. MIT shocks have no burn-in (shocks start at period 1); only burn-out periods are appended.

| Field | Type | Description |
|-------|------|-------------|
| `MITShocks.full_simul` | matrix `[n_vars × T_active]` | Active-period simulation |
| `MITShocks.variable_indices` | struct | Variable index mapping |
| `MITShocks.burn_in` | int | Always 0 (no burn-in for MIT shocks) |
| `MITShocks.burn_out` | int | Burn-out periods |
| `MITShocks.T_active` | int | Active simulation periods |
| `MITShocks.T_total` | int | Total periods (T_active + burn_out) |
| `MITShocks.full_simul_with_burn` | matrix `[n_vars × (T_total+2)]` | Full Dynare output including initial/terminal conditions |

---

## ModelData_IRs (impulse response data)

Saved as `ModelData_IRs.mat`. Contains IRF matrices and statistics in a flat structure.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `save_label` | string | Experiment label |
| `sector_indices` | vector | Analyzed sector indices |
| `ir_horizon` | int | IRF horizon (e.g., 200) |
| `shocks` | struct array `(n_shocks)` | Shock configurations (see below) |
| `irfs` | cell `{n_shocks}` | IRF data per shock (see below) |
| `peaks.first_order` | matrix `[n_shocks × n_sectors]` | Peak |consumption deviation| (1st order) |
| `peaks.second_order` | matrix `[n_shocks × n_sectors]` | Peak (2nd order) |
| `peaks.perfect_foresight` | matrix `[n_shocks × n_sectors]` | Peak (perfect foresight) |
| `half_lives.first_order` | matrix `[n_shocks × n_sectors]` | Half-life in periods (1st order) |
| `half_lives.second_order` | matrix `[n_shocks × n_sectors]` | Half-life (2nd order) |
| `half_lives.perfect_foresight` | matrix `[n_shocks × n_sectors]` | Half-life (PF) |
| `amplifications.abs` | matrix `[n_shocks × n_sectors]` | |Peak(PF)| − |Peak(1st)| |
| `amplifications.rel` | matrix `[n_shocks × n_sectors]` | (PF/1st − 1) × 100 (%) |

### Shock configuration: `shocks(i)`

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Short label, e.g. `'neg20pct'` |
| `value` | double | IRshock value (log deviation from SS) |
| `size_pct` | int | Shock size in percent (e.g., 20) |
| `sign` | int | −1 (negative) or +1 (positive) |
| `A_level` | double | TFP level: A = exp(−value) |
| `description` | string | Human-readable description |

### IRF data: `irfs{shock_idx}(sector_pos)`

Each entry is a struct with:

| Field | Type | Description |
|-------|------|-------------|
| `sector_idx` | int | Sector number (1–37) |
| `first_order` | matrix `[27 × T]` | IRF from 1st-order approximation |
| `second_order` | matrix `[27 × T]` | IRF from 2nd-order approximation |
| `perfect_foresight` | matrix `[27 × T]` | IRF from perfect foresight solver |

IRF matrix rows (see `process_ir_data.m` for details):

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

Rows 2–3 and 24 are expenditure-based aggregates (nominal sums across sectors). Row 27 is the utility-based aggregate from the household problem. Rows 4–12 and 14–22 are log deviations from steady state.

### Access examples

```matlab
load('ModelData_IRs.mat', 'ModelData_IRs');

% Get first-order IRF for the first shock, first sector
irf = ModelData_IRs.irfs{1}(1).first_order;

% Aggregate consumption response (row 2)
C_response = irf(2, :);

% Peak consumption deviations for all shocks (first sector)
peaks = ModelData_IRs.peaks.perfect_foresight(:, 1);

% Amplification for a specific shock
shock_labels = {ModelData_IRs.shocks.label};
idx = find(strcmp(shock_labels, 'neg20pct'));
amp = ModelData_IRs.amplifications.rel(idx, 1);

% Filter by shock sign
neg_mask = [ModelData_IRs.shocks.sign] == -1;
pos_mask = [ModelData_IRs.shocks.sign] == +1;
neg_peaks = ModelData_IRs.peaks.perfect_foresight(neg_mask, :);
pos_peaks = ModelData_IRs.peaks.perfect_foresight(pos_mask, :);

% Filter by shock size
sizes = [ModelData_IRs.shocks.size_pct];
large_mask = sizes == 20;
large_peaks = ModelData_IRs.peaks.perfect_foresight(large_mask, :);
```
