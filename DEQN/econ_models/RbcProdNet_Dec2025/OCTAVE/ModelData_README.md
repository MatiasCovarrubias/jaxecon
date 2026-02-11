# ModelData Structures Reference

Multiple `.mat` files are produced by `main.m`:

| File                                        | Contents                             |
| ------------------------------------------- | ------------------------------------ |
| `ModelData.mat`                             | Metadata, calibration, SS, stats     |
| `ModelData_simulation_FirstOrder.mat`       | First-order simulation (modorder=1)  |
| `ModelData_simulation_SecondOrder.mat`      | Second-order simulation (modorder=2) |
| `ModelData_simulation_PerfectForesight.mat` | Perfect foresight simulation         |
| `ModelData_IRs.mat`                         | Impulse response functions           |

**Note**: Simulation files are saved separately (one per method) to reduce file size for email transfer.

---

### Backward Compatibility Notice

**Version Feb 2026+**: `ModelStats` is now computed for **all three simulation methods** (FirstOrder, SecondOrder, PerfectForesight). This enables the summary table to display business cycle moments from each approximation method side by side.

**Existing ModelData files** (created before Feb 2026) may be missing:

| Field                                     | Status in Old Files |
| ----------------------------------------- | ------------------- |
| `Statistics.FirstOrder.ModelStats`        | ✓ Available         |
| `Statistics.SecondOrder.ModelStats`       | ✓ Available         |
| `Statistics.PerfectForesight.ModelStats`  | ✗ Missing           |
| `BaseResults.ModelStatsPF` (in workspace) | ✗ Missing           |

**Impact**: When loading old `ModelData.mat` files, the summary table and recovery code will show `N/A` for PerfectForesight columns. To populate these fields, re-run `main.m` with `config.run_pf_simul = true`.

---

**Shock Sharing**: All simulation methods use the SAME shock sequence for direct comparability:

- First-order and second-order use the full `T_loglin` periods
- Perfect foresight uses a subset of periods with BURN-IN/BURN-OUT:
    - Total periods = burn_in + T_active + burn_out (default: 100 + 500 + 100 = 700)
    - Burn periods have ZERO shocks to help convergence (smooth transition from/to SS)
    - Only the active periods (T_active) are stored in the main output
    - This approach helps convergence for longer perfect foresight simulations

---

## 1. ModelData

### metadata

```matlab
metadata.date              % string: experiment date
metadata.exp_label         % string: experiment name
metadata.save_label        % string: combined label
metadata.sector_indices    % 1×S: sectors analyzed for IRFs
metadata.sector_labels     % S×1 cell: sector names
metadata.config            % struct: full run configuration
metadata.exp_paths         % struct: output folder paths
metadata.shock_configs     % n×1 struct: shock configurations
metadata.n_shocks          % double: number of shocks
metadata.shock_labels      % n×1 cell: shock labels
metadata.shock_values      % n×1: IRshock values (log scale)
```

### calibration

```matlab
calibration.conssh_data       % 37×1: consumption shares
calibration.capsh_data        % 37×1: capital shares
calibration.vash_data         % 37×1: value-added shares
calibration.ionet_data        % 37×37: IO matrix
calibration.invnet_data       % 37×37: investment network
calibration.labels            % struct: sector labels
calibration.empirical_targets % struct: business cycle moments (see below)
```

### calibration.empirical_targets

```matlab
empirical_targets.hp_lambda              % 100: HP filter parameter for annual data
empirical_targets.aggregation_method     % 'tornqvist': aggregation method used
empirical_targets.va_weights             % 1×37: time-average VA shares (from nominal)
empirical_targets.go_weights             % 1×37: time-average GO shares
empirical_targets.emp_weights            % 1×37: time-average employment shares
empirical_targets.inv_weights            % 1×37: time-average nominal investment shares

% Aggregate volatilities (std of HP-filtered log aggregate)
empirical_targets.sigma_VA_agg           % scalar: aggregate VA volatility (Törnqvist)
empirical_targets.sigma_C_agg            % scalar: aggregate consumption volatility (NIPA PCE)
empirical_targets.sigma_L_agg            % scalar: aggregate employment volatility (simple sum)
empirical_targets.sigma_I_agg            % scalar: aggregate investment volatility (Törnqvist)

% Domar weight volatility
empirical_targets.sigma_Domar_avg        % scalar: GO-weighted avg of sectoral Domar volatilities
empirical_targets.sigma_Domar_sectoral   % 1×37: sectoral Domar weight volatilities

% Sectoral volatilities (VA-weighted average of sectoral volatilities)
empirical_targets.sigma_L_avg            % scalar: VA-weighted avg sectoral labor vol
empirical_targets.sigma_I_avg            % scalar: VA-weighted avg sectoral investment vol

% Sectoral volatilities (own-variable weighted)
empirical_targets.sigma_L_avg_empweighted % scalar: employment-weighted avg sectoral labor vol
empirical_targets.sigma_I_avg_invweighted % scalar: investment-weighted avg sectoral investment vol

% Full sectoral distributions
empirical_targets.sigma_L_sectoral       % 1×37: sectoral labor volatilities
empirical_targets.sigma_I_sectoral       % 1×37: sectoral investment volatilities
```

### params

```matlab
params.beta, params.eps_l, params.eps_c, params.theta, params.phi
params.sigma_c, params.sigma_m, params.sigma_q, params.sigma_y, params.sigma_I, params.sigma_l
params.delta       % 37×1: depreciation rates
params.rho         % 37×37: TFP persistence
params.Sigma_A     % 37×37: shock covariance
params.n_sectors   % 37
params.IRshock     % last used shock value
```

### SteadyState

```matlab
SteadyState.parameters    % struct: CES parameters
SteadyState.policies_ss   % 412×1: SS policies (log)
SteadyState.endostates_ss % 37×1: SS capital (log)
SteadyState.Cagg_ss       % scalar: aggregate C
SteadyState.Lagg_ss       % scalar: aggregate L
SteadyState.Yagg_ss       % scalar: aggregate Y
SteadyState.Iagg_ss       % scalar: aggregate I
SteadyState.Magg_ss       % scalar: aggregate M
```

**policies_ss indexing (n=37):**

- `1:n` c_j, `n+1:2n` l_j, `2n+1:3n` pk_j, `3n+1:4n` pm_j
- `4n+1:5n` m_j, `5n+1:6n` mout_j, `6n+1:7n` i_j, `7n+1:8n` iout_j
- `8n+1:9n` p_j, `9n+1:10n` q_j, `10n+1:11n` y_j
- `11n+1` cagg, `11n+2` lagg, `11n+3` yagg, `11n+4` iagg, `11n+5` magg

### Solution

```matlab
Solution.StateSpace.A     % 74×74: state transition
Solution.StateSpace.B     % 74×37: shock impact
Solution.StateSpace.C     % 412×74: policy function
Solution.StateSpace.D     % 412×37: shock-to-policy
Solution.indices          % struct: variable indices
Solution.steady_state     % 486×1: Dynare SS vector
```

**State vector (74 = 2×37):** `S(1:37)` = k_j, `S(38:74)` = a_j

### Statistics

```matlab
Statistics.TheoStats           % struct: theoretical moments
Statistics.shocks_sd           % 37×1: shock std
Statistics.states_sd           % 74×1: state std
Statistics.policies_sd         % 412×1: policy std
Statistics.FirstOrder.*        % first-order stats
Statistics.SecondOrder.*       % second-order stats
Statistics.PerfectForesight.*  % perfect foresight stats
```

**Per-method (FirstOrder, SecondOrder, PerfectForesight):**

```matlab
*.states_mean    % 74×1: mean states
*.states_std     % 74×1: std states
*.policies_mean  % 412×1: mean policies
*.policies_std   % 412×1: std policies
*.ModelStats     % struct: business cycle stats (see below)
```

**ModelStats structure** (computed from simulation, directly comparable to empirical_targets):

```matlab
ModelStats.sigma_VA_agg            % scalar: aggregate GDP volatility
ModelStats.sigma_L_agg             % scalar: aggregate labor volatility (CES aggregator)
ModelStats.sigma_L_hc_agg          % scalar: aggregate labor volatility (headcount, comparable to data)
ModelStats.sigma_I_agg             % scalar: aggregate investment volatility
ModelStats.sigma_M_agg             % scalar: aggregate intermediates volatility
ModelStats.rho_VA_agg              % scalar: aggregate GDP autocorrelation
ModelStats.avg_pairwise_corr_VA    % scalar: average pairwise correlation of sectoral VA

% Sectoral volatilities (VA-weighted averages, comparable to empirical_targets.sigma_L_avg)
ModelStats.sigma_L_avg             % scalar: VA-weighted avg sectoral labor volatility
ModelStats.sigma_I_avg             % scalar: VA-weighted avg sectoral investment volatility

% Sectoral volatilities (own-variable weighted, comparable to empirical_targets)
ModelStats.sigma_L_avg_empweighted % scalar: employment-weighted avg sectoral labor volatility
ModelStats.sigma_I_avg_invweighted % scalar: investment-weighted avg sectoral investment volatility

% Domar weight volatility
ModelStats.sigma_Domar_avg         % scalar: GO-weighted avg of sectoral Domar volatilities
ModelStats.sigma_Domar_sectoral    % 1×37: sectoral Domar weight volatilities

% Full sectoral distributions (for diagnostics)
ModelStats.sigma_L_sectoral        % 1×37: sectoral labor volatilities
ModelStats.sigma_I_sectoral        % 1×37: sectoral investment volatilities
ModelStats.corr_matrix_VA          % 37×37: pairwise correlations of sectoral VA

% Weights used (from steady state — SS shares are model analog to time-avg shares in data)
ModelStats.va_weights              % 1×37: steady-state VA shares (Y_j^ss / sum Y_j^ss)
ModelStats.go_weights              % 1×37: steady-state GO shares (Q_j^ss / sum Q_j^ss)
ModelStats.emp_weights             % 1×37: steady-state employment shares (L_j^ss / sum L_j^ss)
ModelStats.inv_weights             % 1×37: steady-state NOMINAL investment shares ((I_j*Pk_j)^ss / sum)
```

### Diagnostics

```matlab
Diagnostics.has_firstorder, has_secondorder, has_pf, has_irfs  % logical
Diagnostics.prealloc_mean_abs_k, prealloc_k_mining             % double (%)
Diagnostics.prealloc_k_max, prealloc_k_min                     % double (%)
Diagnostics.prealloc_k_max_sector, prealloc_k_min_sector       % string
Diagnostics.prealloc_k_all                                     % 37×1 (%)
Diagnostics.firstorder_C_mean_logdev, firstorder_C_std         % double (%)
Diagnostics.secondorder_C_mean_logdev, secondorder_C_std       % double (%)
Diagnostics.pf_C_mean_logdev, pf_C_std                         % double (%)
Diagnostics.precautionary_C                                    % double (%)
Diagnostics.irf_peak_firstorder, irf_peak_pf                   % n×1 (%)
Diagnostics.irf_amplification_rel                              % n×1 (%)
Diagnostics.irf_shock_labels                                   % n×1 cell
```

---

## 2. Method-Specific Simulation Files

Each simulation method is saved in its own file to reduce size for email transfer.

### 2.1 ModelData_simulation_FirstOrder.mat

First-order (log-linear) approximation simulation. Created when `config.modorder = 1`.

```matlab
metadata.save_label           % string
metadata.exp_paths            % struct
metadata.n_sectors            % 37
metadata.method               % 'FirstOrder'
metadata.approximation_order  % 1

rng_state.Type       % string
rng_state.Seed       % double
rng_state.State      % 625×1 uint32

full_simul           % 486×T: all variables (log dev from SS)
shocks               % T×37: shock realizations used in simulation
variable_indices     % struct: index mappings (see below)

% Aggregate series (in levels)
Cagg                 % 1×T: aggregate consumption
Lagg                 % 1×T: aggregate labor
Yagg                 % 1×T: aggregate value added
Iagg                 % 1×T: aggregate investment
Magg                 % 1×T: aggregate intermediates

Cagg_volatility      % scalar: std(Cagg)/Cagg_ss
Lagg_volatility      % scalar: std(Lagg)/Lagg_ss

ModelStats           % struct: business cycle statistics
```

### 2.2 ModelData_simulation_SecondOrder.mat

Second-order approximation simulation. Created when `config.modorder = 2`.
Structure identical to FirstOrder, with `metadata.approximation_order = 2`.

### 2.3 ModelData_simulation_PerfectForesight.mat

Perfect foresight (deterministic) simulation with burn-in/burn-out periods.

```matlab
metadata.save_label           % string
metadata.exp_paths            % struct
metadata.n_sectors            % 37
metadata.method               % 'PerfectForesight'
metadata.approximation_order  % 'exact'

rng_state.Type       % string (same RNG state as linear simulations)
rng_state.Seed       % double
rng_state.State      % 625×1 uint32

full_simul           % 486×T_active: active periods only (log dev from SS)
shocks               % T_active×37: shock realizations for active periods
variable_indices     % struct: index mappings (see below)

% Burn-in/burn-out metadata
burn_in              % scalar: number of burn-in periods (default: 100)
burn_out             % scalar: number of burn-out periods (default: 100)
T_active             % scalar: number of active periods with shocks (default: 500)
T_total              % scalar: total simulation periods (burn_in + T_active + burn_out)

% Full simulation including burn periods (optional, for diagnostics)
full_simul_with_burn % 486×T_total: all periods including burn-in/burn-out
shocks_with_burn     % T_total×37: full shock matrix (zeros in burn periods)

% Aggregate series (in levels)
Cagg                 % 1×T: aggregate consumption
Lagg                 % 1×T: aggregate labor

Cagg_volatility      % scalar: std(Cagg)/Cagg_ss
Lagg_volatility      % scalar: std(Lagg)/Lagg_ss
```

**Note on burn-in/burn-out**: Burn periods have zero shocks and help convergence by allowing the model to smoothly transition from/to the deterministic steady state. The main output (`full_simul`, `shocks`) contains only the active periods.

### variable_indices (common to all methods)

```matlab
idx.k = [1, 37]        % capital
idx.a = [38, 74]       % TFP
idx.c = [75, 111]      % consumption
idx.l = [112, 148]     % labor
idx.pk = [149, 185]    % capital price
idx.pm = [186, 222]    % intermediate price
idx.m = [223, 259]     % intermediate input
idx.mout = [260, 296]  % intermediate output
idx.i = [297, 333]     % investment
idx.iout = [334, 370]  % investment output
idx.p = [371, 407]     % output price
idx.q = [408, 444]     % gross output
idx.y = [445, 481]     % value added
idx.cagg = 482, idx.lagg = 483, idx.yagg = 484, idx.iagg = 485, idx.magg = 486
```

---

## 3. ModelData_IRs

### Top-level

```matlab
metadata.save_label  % string
metadata.exp_paths   % struct
shock_configs        % n×1 struct
n_shocks             % double
sector_indices       % 1×S
ir_horizon           % double
```

### Access patterns

```matlab
by_shock{i}           % cell array access (i = 1..n_shocks)
by_label.neg20pct     % struct field access
by_label.pos20pct
by_label.neg5pct
by_label.pos5pct
```

### IRF result structure

Each `by_shock{i}` or `by_label.*` contains:

```matlab
*.IRFs              % S×1 cell: IRF data per sector
*.Statistics        % struct: peak values, half-lives
*.labels            % struct: sector labels
*.shock_description % string
*.shock_config      % struct
```

**IRFs{i}:**

```matlab
IRFs{i}.sector_idx          % which sector shocked
IRFs{i}.client_idx          % largest client
IRFs{i}.IRSFirstOrder       % 24×T: first-order IRF (always present)
IRFs{i}.IRSSecondOrder      % 24×T: second-order IRF (optional, if run_secondorder_irs=true)
IRFs{i}.IRSPF               % 24×T: perfect foresight IRF (if run_pf_irs=true)
```

**IRF rows:** 1=A_ir, 2=Cagg, 3=Lagg, 4-7=shocked outputs, 8-12=shocked inputs, 13=client TFP, 14-17=client outputs, 18-22=client inputs, 23-24=Yagg/Iagg

**Statistics:**

```matlab
Statistics.sector_indices          % 1×S
Statistics.peak_values_firstorder  % S×1
Statistics.peak_values_secondorder % S×1
Statistics.peak_values_pf          % S×1
Statistics.peak_periods_firstorder % S×1
Statistics.peak_periods_secondorder% S×1
Statistics.peak_periods_pf         % S×1
Statistics.half_lives_firstorder   % S×1
Statistics.half_lives_secondorder  % S×1
Statistics.half_lives_pf           % S×1
Statistics.amplifications          % S×1: PF - 1st
Statistics.amplifications_2nd      % S×1: 2nd - 1st
Statistics.amplifications_rel      % S×1: relative (%)
```

---

## 4. Recovering the Model vs Data Summary Table

After loading `ModelData.mat`, you can reconstruct the model vs data comparison table printed by `main.m`. The summary table now supports **multiple model columns** (1st-Order, 2nd-Order, Perfect Foresight) when available.

### Quick Example (All Three Methods)

```matlab
load('ModelData.mat');

% Access empirical targets
emp = ModelData.EmpiricalTargets;

% Access ModelStats for each simulation method (check availability first)
has_1st = isfield(ModelData.Statistics, 'FirstOrder') && ...
          isfield(ModelData.Statistics.FirstOrder, 'ModelStats');
has_2nd = isfield(ModelData.Statistics, 'SecondOrder') && ...
          isfield(ModelData.Statistics.SecondOrder, 'ModelStats');
has_PF  = isfield(ModelData.Statistics, 'PerfectForesight') && ...
          isfield(ModelData.Statistics.PerfectForesight, 'ModelStats');

if has_1st, ms1 = ModelData.Statistics.FirstOrder.ModelStats; end
if has_2nd, ms2 = ModelData.Statistics.SecondOrder.ModelStats; end
if has_PF,  msPF = ModelData.Statistics.PerfectForesight.ModelStats; end

% Print header
fprintf('                      1st        2nd         PF       Data\n');

% Aggregate GDP volatility
fprintf('σ(Y_agg)         ');
if has_1st, fprintf('%8.4f ', ms1.sigma_VA_agg); else, fprintf('     N/A '); end
if has_2nd, fprintf('%8.4f ', ms2.sigma_VA_agg); else, fprintf('     N/A '); end
if has_PF,  fprintf('%8.4f ', msPF.sigma_VA_agg); else, fprintf('     N/A '); end
fprintf('%8.4f\n', emp.sigma_VA_agg);

% Labor aggregate (headcount)
fprintf('σ(L_hc_agg)      ');
if has_1st, fprintf('%8.4f ', ms1.sigma_L_hc_agg); else, fprintf('     N/A '); end
if has_2nd, fprintf('%8.4f ', ms2.sigma_L_hc_agg); else, fprintf('     N/A '); end
if has_PF,  fprintf('%8.4f ', msPF.sigma_L_hc_agg); else, fprintf('     N/A '); end
fprintf('%8.4f\n', emp.sigma_L_agg);

% Sectoral labor volatility (employment-weighted)
fprintf('σ(L) emp-wgt     ');
if has_1st, fprintf('%8.4f ', ms1.sigma_L_avg_empweighted); else, fprintf('     N/A '); end
if has_2nd, fprintf('%8.4f ', ms2.sigma_L_avg_empweighted); else, fprintf('     N/A '); end
if has_PF,  fprintf('%8.4f ', msPF.sigma_L_avg_empweighted); else, fprintf('     N/A '); end
fprintf('%8.4f\n', emp.sigma_L_avg_empweighted);

% Sectoral investment volatility (investment-weighted)
fprintf('σ(I) inv-wgt     ');
if has_1st, fprintf('%8.4f ', ms1.sigma_I_avg_invweighted); else, fprintf('     N/A '); end
if has_2nd, fprintf('%8.4f ', ms2.sigma_I_avg_invweighted); else, fprintf('     N/A '); end
if has_PF,  fprintf('%8.4f ', msPF.sigma_I_avg_invweighted); else, fprintf('     N/A '); end
fprintf('%8.4f\n', emp.sigma_I_avg_invweighted);
```

### Field Mapping: Summary Table → ModelData

| Summary Table Row | Model Value Source                                | Data Value Source                          |
| ----------------- | ------------------------------------------------- | ------------------------------------------ |
| σ(Y_agg)          | `Statistics.*.ModelStats.sigma_VA_agg`            | `EmpiricalTargets.sigma_VA_agg`            |
| σ(I_agg)          | `Statistics.*.ModelStats.sigma_I_agg`             | `EmpiricalTargets.sigma_I_agg`             |
| σ(L_hc_agg)       | `Statistics.*.ModelStats.sigma_L_hc_agg`          | `EmpiricalTargets.sigma_L_agg`             |
| σ(Domar) avg      | `Statistics.*.ModelStats.sigma_Domar_avg`         | `EmpiricalTargets.sigma_Domar_avg`         |
| σ(L) avg (VA-wgt) | `Statistics.*.ModelStats.sigma_L_avg`             | `EmpiricalTargets.sigma_L_avg`             |
| σ(I) avg (VA-wgt) | `Statistics.*.ModelStats.sigma_I_avg`             | `EmpiricalTargets.sigma_I_avg`             |
| σ(L) emp-wgt      | `Statistics.*.ModelStats.sigma_L_avg_empweighted` | `EmpiricalTargets.sigma_L_avg_empweighted` |
| σ(I) inv-wgt      | `Statistics.*.ModelStats.sigma_I_avg_invweighted` | `EmpiricalTargets.sigma_I_avg_invweighted` |

**Note**: Replace `*` with `FirstOrder`, `SecondOrder`, or `PerfectForesight` depending on which simulation method you want. All three methods now compute and store `ModelStats` (for new runs after Feb 2026). Old ModelData files may lack `PerfectForesight.ModelStats` — see Backward Compatibility Notice above.
