# Impulse Response Pipeline

This document describes the full IR (Impulse Response) pipeline, from MATLAB/Dynare generation through Python plotting. The pipeline has two independent production sides — MATLAB (linear/PF benchmarks) and DEQN/Python (nonlinear) — that meet at the plotting stage.

---

## Architecture Overview

```
MATLAB side                          Python side (DEQN)
──────────                           ──────────────────

main.m                               analysis.py
  │                                    │
  ▼                                    ├─ create_GIR_fn() ──► gir_data
run_dynare_analysis.m                  │   (JAX, nonlinear)
  │                                    │
  ├─ simult_() ──► IRSFirstOrder_raw   ├─ analysis_hooks.compute_analysis_variables()
  ├─ 2nd order ──► IRSSecondOrder_raw  │   (`get_aggregates()` by default, fixed-price reaggregation when configured)
  ├─ PF solver ──► IRSPerfectForesight_raw│
  │                                    │
  ▼                                    │
process_sector_irs.m                   │
  │                                    │
  ▼                                    │
process_ir_data.m                      │
  │                                    │
  ├─ 29-row irs matrix                 │
  ├─ sectoral struct (37×T each)       │
  │                                    │
  ▼                                    │
build_ModelData_IRs.m                  │
  │                                    │
  ├─ flat irfs struct per shock        │
  │                                    │
  ▼                                    │
.mat file  ─────────────────────►  load_matlab_irs()
                                       │
                                       ▼
                                   matlab_irs.py
                                       │
                                       ├─ get_matlab_ir_fixedprice()
                                       │   (re-aggregates with P_ergodic)
                                       │
                                       ├─ get_matlab_ir_for_analysis_variable()
                                       │   (row-based lookup, sectoral vars)
                                       │
                                       ▼
                                   plots.py
                                       │
                                       ▼
                                   plot_sector_ir_by_shock_size()
                                       (overlays DEQN + MATLAB on same axes)
```

---

## File Reference

### MATLAB Side

| File | Role |
|------|------|
| `MATLAB/main.m` | Entry point. Configures shock sizes, sectors, horizon. Calls calibration, steady-state solver, then `run_irf_loop`. |
| `MATLAB/utils/run_irf_loop.m` | Loops over shock configurations, calls `run_dynare_analysis` for each, then `process_sector_irs` to post-process. |
| `MATLAB/dynare/run_dynare_analysis.m` | Dynare workhorse. Solves the model, computes first-order IRFs via `simult_()`, second-order IRFs, and perfect-foresight IRFs via Dynare's PF solver. Stores raw `dynare_simul` matrices (`IRSFirstOrder_raw`, `IRSSecondOrder_raw`, `IRSPerfectForesight_raw`). |
| `MATLAB/dynare/process_sector_irs.m` | Post-processing orchestrator. Loops over shocked sectors, calls `process_ir_data` on each raw simulation (capturing the sectoral output), collects results into `IRFs` cell array, computes statistics. |
| `MATLAB/utils/process_ir_data.m` | Row-extraction function. Takes a raw `dynare_simul` matrix (486×T for 37 sectors) and produces two outputs: a compact **29-row `irs` matrix** and a **`sectoral` struct** with full 37-sector vectors. |
| `MATLAB/utils/build_ModelData_IRs.m` | Repackages the `IRFs` cell array into a flat struct suitable for saving. Copies `sectoral_loglin` and `sectoral_determ` through to the output. |
| `MATLAB/utils/get_variable_indices.m` | Single source of truth for variable index ranges within Dynare's simulation output and `policies_ss`. All other MATLAB files call this. |

### The `.mat` File (Handoff Point)

Saved as `ModelData_IRs.mat` (or a custom name like `ModelData_IRs_benchMar.mat`). Top-level structure:

| Field | Description |
|-------|-------------|
| `shocks` | Struct array of shock configurations (value, label, description) |
| `irfs` | Cell array {n_shocks}, each containing a struct array over sectors |
| `peaks.first_order`, `peaks.second_order`, `peaks.perfect_foresight` | Peak value matrices (n_shocks × n_sectors) |
|| `half_lives.first_order`, `half_lives.second_order`, `half_lives.perfect_foresight` | Half-life matrices (n_shocks × n_sectors) |
|| `amplifications.abs`, `amplifications.rel` | Absolute and relative (%) amplification matrices (n_shocks × n_sectors) |
| `sector_indices`, `ir_horizon` | Metadata |

Each element of `irfs{i}` is a struct with:

| Field | Shape | Description |
|-------|-------|-------------|
| `first_order` | 27 × T | Compact IR matrix (first-order / linear) |
| `second_order` | 27 × T | Compact IR matrix (second-order) |
| `perfect_foresight` | 27 × T | Compact IR matrix (perfect foresight / nonlinear) |
| `sectoral_loglin` | struct | Full sectoral vectors for Python-side re-aggregation (first-order) |
| `sectoral_determ` | struct | Full sectoral vectors for Python-side re-aggregation (PF) |
| `sector_idx` | scalar | Which sector was shocked (1-indexed) |

The `sectoral` structs contain four fields, each 37×T in log deviations from deterministic SS:

| Field | Variable | `policies_ss` offset (0-indexed) |
|-------|----------|----------------------------------|
| `C_all` | Consumption | `[0, n)` |
| `Iout_all` | Investment output | `[7n, 8n)` |
| `Q_all` | Gross output | `[9n, 10n)` |
| `Mout_all` | Intermediate output | `[5n, 6n)` |

### Python Side

| File | Role |
|------|------|
| `DEQN/analysis.py` | Main orchestrator. Loads MATLAB data, runs the DEQN IR selected by `config["use_gir"]` (`True` = `GIR`, `False` = `IR_stoch_ss`), and delegates IR rendering to the active model hooks. |
| `DEQN/econ_models/RbcProdNet_April2026/matlab_irs.py` | Loads `ModelData_IRs.mat`, parses MATLAB structs, discovers available shock sizes, provides row-based lookup for sectoral variables, and provides fixed-price re-aggregation for aggregate benchmark IRs. |
| `DEQN/econ_models/RbcProdNet_April2026/plot_helpers.py` | Plotting layer. `plot_sector_ir_by_shock_size()` overlays DEQN nonlinear IRs with one or more MATLAB benchmark series. |
| `DEQN/econ_models/RbcProdNet_April2026/model.py` | Model definition. `get_aggregates()` exposes the aggregate policy tail directly. |
| `DEQN/econ_models/RbcProdNet_April2026/analysis_hooks.py` | Builds the IR rendering context, switches between direct aggregate extraction and optional fixed-price ergodic reaggregation, and requests the full aggregate IR panel. |
| `DEQN/econ_models/RbcProdNet_April2026/aggregation.py` | Shared fixed-price aggregation logic used for moments and for consistent aggregate definitions across nonlinear and benchmark objects. |

---

## The 29-Row IR Matrix

Rows in the compact `irs` matrix produced by `process_ir_data.m` (0-indexed):

| Row | Variable | Description |
|-----|----------|-------------|
| 0 | `A_ir` | TFP of shocked sector (in levels, not log dev) |
| 1 | `C_ir` | Aggregate consumption expenditure (current-price, log dev) |
| 2 | `I_ir` | Aggregate investment expenditure (current-price, log dev) |
| 3 | `Cj_ir` | Sectoral consumption, shocked sector |
| 4 | `Pj_ir` | Sectoral price, shocked sector |
| 5 | `Ioutj_ir` | Sectoral investment output, shocked sector |
| 6 | `Moutj_ir` | Sectoral intermediate output, shocked sector |
| 7 | `Lj_ir` | Sectoral labor, shocked sector |
| 8 | `Ij_ir` | Sectoral investment input, shocked sector |
| 9 | `Mj_ir` | Sectoral intermediate input, shocked sector |
| 10 | `Yj_ir` | Sectoral value added, shocked sector |
| 11 | `Qj_ir` | Sectoral gross output, shocked sector |
| 12 | `A_client_ir` | TFP, client sector (levels) |
| 13–21 | `*_client_ir` | Same as rows 3–11 but for the client sector |
| 22 | `Kj_ir` | Sectoral capital, shocked sector |
| 23 | `GDP_ir` | Aggregate GDP expenditure (current-price, log dev) |
| 24 | `Pmj_client_ir` | Client intermediate price |
| 25 | `gammaij_client_ir` | Client expenditure share deviation |
| 26 | `L_ir` | Aggregate labor headcount (log dev) |
| 27 | `K_ir` | Aggregate capital (log dev) |
| 28 | `utility_intratemp_ir` | Intratemporal utility (level deviation) |

**Important**: Rows 1, 2, and 23 are the stored MATLAB aggregate expenditure IR rows, while rows 26, 27, and 28 are Dynare's built-in aggregate labor, capital, and intratemporal-utility rows. In the current April contract, DEQN aggregate IRs come directly from the aggregate policy tail by default. When `ergodic_price_aggregation=true`, the Python side instead re-aggregates the relevant MATLAB benchmark IRs from full sectoral vectors using fixed ergodic prices.

---

## `policies_ss` Layout

The April model's `policies_ss` vector contains the sectoral policy blocks followed by an 8-entry aggregate tail. The first `11n + 7` entries are stored in logs; the final `utility_intratemp` entry is stored in levels. This ordering is the canonical reference for the Python-side hardcoded offsets.

With `n` = number of sectors (typically 37):

| Block | Range (0-indexed) | Variable |
|-------|-------------------|----------|
| 0 | `[0, n)` | Consumption `c` |
| 1 | `[n, 2n)` | Labor `l` |
| 2 | `[2n, 3n)` | Capital price `pk` |
| 3 | `[3n, 4n)` | Intermediate price `pm` |
| 4 | `[4n, 5n)` | Intermediate input `m` |
| 5 | `[5n, 6n)` | Intermediate output `mout` |
| 6 | `[6n, 7n)` | Investment input `i` |
| 7 | `[7n, 8n)` | Investment output `iout` |
| 8 | `[8n, 9n)` | Output price `p` |
| 9 | `[9n, 10n)` | Gross output `q` |
| 10 | `[10n, 11n)` | Value added `y` |
| 11 | `11n` | Utility consumption aggregate `c_util` |
| 12 | `11n+1` | Utility labor aggregate `l_util` |
| 13 | `11n+2` | Aggregate consumption `cagg` |
| 14 | `11n+3` | Aggregate labor `lagg` |
| 15 | `11n+4` | Aggregate GDP `gdpagg` |
| 16 | `11n+5` | Aggregate investment `iagg` |
| 17 | `11n+6` | Aggregate capital `kagg` |
| 18 | `11n+7` | Intratemporal utility `utility_intratemp` (level) |

Total length: `11n + 8` (= 415 for n=37).

In Dynare's full simulation output, states (capital `k` and TFP `a`, 2n variables) precede these policies. The offset is `ss_offset = 2n`. So Dynare index for consumption of sector j is `2n + j`, while `policies_ss` index is `j`.

---

## Aggregation: The Core Design Decision

### The current contract

The April model now carries the main aggregate objects directly in the nonlinear policy tail:

```
[c_util, l_util, C, L, GDP, I, K, utility_intratemp]
```

This means the default DEQN IR workflow does **not** reconstruct nonlinear aggregates from sectoral objects.

The remaining aggregation choice is only about the comparison convention:

- Default mode: `ergodic_price_aggregation = false`
  DEQN uses the direct aggregate policy tail. MATLAB benchmark aggregates are read from their direct rows.
- Optional mode: `ergodic_price_aggregation = true`
  Python re-aggregates the relevant MATLAB and simulation objects using fixed ergodic prices so the comparison matches the fixed-price convention.

### How it works in code

For **aggregate** variables:

- Default mode:
  `analysis_hooks.compute_analysis_variables()` uses `model.get_aggregates()` for DEQN outputs, and MATLAB IR lookups use the direct aggregate rows.
- Optional fixed-price mode:
  `get_matlab_ir_fixedprice()` is used when the comparison should be re-aggregated under fixed ergodic prices.
  It reads full sectoral vectors from `sectoral_loglin` / `sectoral_determ`, converts log deviations to levels, weights by `P_ergodic` (and `Pk_ergodic` for capital), and takes log deviations relative to the corresponding fixed-price steady-state aggregate.

For **sectoral** variables ("Cj", "Pj", "Yj", etc.):
- `get_matlab_ir_for_analysis_variable()` is called (row-based lookup)
- No re-aggregation needed — these are single-sector quantities

### Reference point difference (intentional)

| | DEQN nonlinear | MATLAB benchmark |
|---|---|---|
| Starting point | Stochastic SS (or ergodic draws for GIR) | Deterministic SS |
| Aggregation weights | Direct policy tail by default; fixed-price only when configured | Direct aggregate rows by default; fixed-price only when configured |

The stochastic vs. deterministic SS difference is intentional — it reveals the Jensen's inequality / precautionary savings gap.

---

## Current IR plotting defaults

The active April 2026 IR contract is intentionally comprehensive by default.

- All six reported aggregates are rendered in the aggregate IR section: `Agg. Consumption`, `Agg. Investment`, `Agg. GDP`, `Agg. Capital`, `Agg. Labor`, and `Intratemporal Utility`.
- Each aggregate figure is a full panel: one row per discovered shock size, with negative shocks in the left column and positive shocks in the right column.
- Python also exports a simplified aggregate IR figure for each reported aggregate variable using only the largest discovered negative shock.
- In the combined LaTeX wrapper, those simplified aggregate IR PNGs are recombined into a paper figure with consumption and GDP side by side, plus an appendix figure with investment, capital, and labor.
- Shock sizes are discovered from `ModelData_IRs.mat`, not hardcoded in the main analysis config.
- The nonlinear DEQN line is always solid. `config["use_gir"] = True` switches that solid line to the generalized impulse response; `False` switches it to the stochastic-steady-state IR.
- The default benchmark overlays are `["PerfectForesight", "FirstOrder"]`, though `ir_benchmark_methods` can still override the set or ordering.
- Sectoral IR figures still focus on the largest discovered shock and the negative-shock panel only.
- Saved IR PNGs print `Saved: <filename>` to the Python console when written.

---

## Key Dictionaries in `matlab_irs.py`

### `NEW_FORMAT_VARIABLE_INDICES`

Maps internal variable names to row indices in the 29-row `irs` matrix.

```python
"A": 0, "Cexp": 1, "Iexp": 2, "Cj": 3, "Pj": 4, ...
"GDPexp": 23, ..., "C_util": 26
```

### `ANALYSIS_TO_MATLAB_MAPPING`

Maps human-readable analysis names to internal names:

```python
"Agg. Consumption": "Cexp",   # stored aggregate benchmark row
"Agg. Investment": "Iexp",    # stored aggregate benchmark row
"Agg. Labor": "Lagg",         # direct aggregate row
"Agg. Capital": "Kagg",       # direct aggregate row
"Agg. Output": "GDPexp",      # stored aggregate benchmark row
"Agg. GDP": "GDPexp",         # stored aggregate benchmark row
"Intratemporal Utility": "utility_intratemp",  # direct aggregate row
"Cj": "Cj",                   # direct mapping
"Pj": "Pj",                   # direct mapping
...
```

### `AGGREGATE_VARIABLE_SECTORAL_MAP`

Defines which aggregates can be re-computed from sectoral data:

```python
"Agg. Consumption": "C_all",
"Agg. Investment": "Iout_all",
```

### `AGGREGATE_VARIABLE_GDP_COMPONENTS`

GDP = gross output minus intermediates:

```python
"Agg. Output": ("Q_all", "Mout_all"),
"Agg. GDP": ("Q_all", "Mout_all"),
```

---

## Period Alignment

MATLAB simulations include period 0 (= steady state before the shock hits). DEQN GIRs start at the first response period. The `skip_initial` flag (default `True`) in the Python loading functions drops MATLAB's period 0 so both series align at "first period after shock."

---

## Common Pitfalls

1. **Hardcoded offsets**: Python uses `ps_levels[8*n:9*n]` for prices, `ps_levels[:n]` for consumption, etc. These must match the `policies_ss` layout defined in `get_variable_indices.m`. Any reordering of blocks in MATLAB silently breaks Python.

2. **Row index drift**: The 29-row matrix ordering is defined by the concatenation order in `process_ir_data.m`. The Python variable-index mapping must mirror this exactly. There is no automated check.

3. **1-indexed vs 0-indexed**: MATLAB uses 1-indexed sector indices everywhere. Python uses 0-indexed. The `.mat` file stores MATLAB's 1-indexed values; `matlab_irs.py` converts to 0-indexed during loading.

4. **Log vs levels**: Most variables in `policies_ss` and the IR matrices are in **log** (log deviations from log SS). The exception is `A_ir` (row 0) and `A_client_ir` (row 12), which are in **levels** (`exp(dynare_simul(...))`).

5. **Format detection**: `matlab_irs.py` auto-detects flat vs nested `.mat` format. The current path standardizes the canonical shock artifact into keys like `pos_12.5` and `neg_50`; legacy files can still be inferred from filenames if needed. If the file was generated without the sectoral data changes, the sectoral blocks needed for fixed-price re-aggregation may be missing and aggregate IRs will fall back to the stored aggregate rows.

6. **PF-only runs**: When MATLAB runs with `run_firstorder_irs=false`, `IRSFirstOrder = []` for every sector. `get_sector_irs` is guarded to skip a shock key only when **both** `IRSFirstOrder` and `IRSPerfectForesight` are absent, so sectoral row-based lookup still returns PF data even with no first-order IRFs. (Older code gated on `first_order` alone, causing sectoral variables to silently return no data in PF-only runs.)

6. **Two code copies**: The MATLAB code may exist at both `/Users/.../jaxecon/` and `/Users/.../Documents/Repositories/jaxecon/`. MATLAB's path determines which version runs. Changes to one copy do not propagate to the other.

7. **`build_ModelData_IRs.m` must pass through new fields**: Any new fields added to the `IRFs{idx}` struct in `process_sector_irs.m` must also be explicitly copied in `build_ModelData_IRs.m`, or they will be silently dropped when building the flat output struct.
