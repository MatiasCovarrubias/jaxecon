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
  ├─ simult_() ──► IRSLoglin_raw       ├─ model.get_analysis_variables()
  ├─ PF solver ──► IRSDeterm_raw       │   (aggregation with P_ergodic)
  │                                    │
  ▼                                    │
process_sector_irs.m                   │
  │                                    │
  ▼                                    │
process_ir_data.m                      │
  │                                    │
  ├─ 26-row irs matrix                 │
  ├─ sectoral struct (37×T each)       │
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
| `MATLAB/main.m` | Entry point. Configures shock sizes, sectors, horizon. Calls calibration, steady-state solver, then `run_dynare_analysis`. |
| `MATLAB/dynare/run_dynare_analysis.m` | Dynare workhorse. Solves the model, computes log-linear IRFs via `simult_()` and perfect-foresight IRFs via Dynare's PF solver. Stores raw `dynare_simul` matrices (`IRSLoglin_raw`, `IRSDeterm_raw`). |
| `MATLAB/dynare/process_sector_irs.m` | Post-processing orchestrator. Loops over shocked sectors, calls `process_ir_data` on each raw simulation, collects results into `IRFs` cell array, computes statistics, saves `.mat` file. |
| `MATLAB/utils/process_ir_data.m` | Row-extraction function. Takes a raw `dynare_simul` matrix (488×T for 37 sectors) and produces two outputs: a compact **26-row `irs` matrix** and a **`sectoral` struct** with full 37-sector vectors. |
| `MATLAB/utils/get_variable_indices.m` | Single source of truth for variable index ranges within Dynare's simulation output and `policies_ss`. All other MATLAB files call this. |

### The `.mat` File (Handoff Point)

Saved by `process_sector_irs.m`. For each shock-size / sector combination it contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `IRSLoglin` | 26 × T | Compact IR matrix (log-linear / first-order) |
| `IRSDeterm` | 26 × T | Compact IR matrix (perfect foresight) |
| `sectoral_loglin` | struct | Full sectoral vectors for Python-side re-aggregation (log-linear) |
| `sectoral_determ` | struct | Full sectoral vectors for Python-side re-aggregation (PF) |
| `sector_idx` | scalar | Which sector was shocked (1-indexed) |
| `client_idx` | scalar | Main customer sector (1-indexed) |

The `sectoral` struct contains four fields, each 37×T in log deviations from deterministic SS:

| Field | Variable | `policies_ss` offset (0-indexed) |
|-------|----------|----------------------------------|
| `C_all` | Consumption | `[0, n)` |
| `Iout_all` | Investment output | `[7n, 8n)` |
| `Q_all` | Gross output | `[9n, 10n)` |
| `Mout_all` | Intermediate output | `[5n, 6n)` |

### Python Side

| File | Role |
|------|------|
| `DEQN/analysis.py` | Main orchestrator. Loads MATLAB data, runs DEQN GIR/IR_stoch_ss, calls plotting. Converts `policies_ss` and `P_ergodic` from JAX to NumPy and passes them to the plotting layer. |
| `DEQN/analysis/matlab_irs.py` | Bridge module (~1200 lines). Loads `.mat` files, parses MATLAB structs, provides row-based lookup for sectoral variables and fixed-price re-aggregation for aggregate variables. |
| `DEQN/analysis/plots.py` | Plotting. `plot_sector_ir_by_shock_size()` overlays DEQN nonlinear IRs with MATLAB first-order and PF benchmarks. |
| `econ_models/.../model.py` | Model definition. `get_analysis_variables()` computes aggregate quantities using ergodic-mean prices (`P_ss * exp(P_weights)`). This is what the DEQN GIR uses internally. |
| `DEQN/analysis/aggregation_correction.py` | Consistent aggregation for ergodic simulation moments (separate from IRs). |

---

## The 26-Row IR Matrix

Rows in the compact `irs` matrix produced by `process_ir_data.m` (0-indexed):

| Row | Variable | Description |
|-----|----------|-------------|
| 0 | `A_ir` | TFP of shocked sector (in levels, not log dev) |
| 1 | `C_ir` | Dynare CES aggregate consumption (current-price, log dev) |
| 2 | `L_ir` | Dynare aggregate labor (log dev) |
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
| 23 | `Y_ir` | Dynare aggregate output (current-price, log dev) |
| 24 | `Pmj_client_ir` | Client intermediate price |
| 25 | `gammaij_client_ir` | Client expenditure share deviation |

**Important**: Rows 1, 2, 23 are Dynare's built-in aggregates computed with current (time-varying) prices. For aggregate IR comparisons with DEQN, the Python side re-aggregates from the full sectoral vectors using fixed ergodic prices instead.

---

## `policies_ss` Layout

The `policies_ss` vector contains steady-state values of all policy variables in log. Its ordering (defined in `get_variable_indices.m`) is the canonical reference for both MATLAB and Python hardcoded offsets.

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
| 11 | `11n` | Aggregate consumption `cagg` |
| 12 | `11n+1` | Aggregate labor `lagg` |
| 13 | `11n+2` | Aggregate output `yagg` |
| 14 | `11n+3` | Aggregate investment `iagg` |
| 15 | `11n+4` | Aggregate intermediates `magg` |

Total length: `11n + 5` (= 412 for n=37).

In Dynare's full simulation output, states (capital `k` and TFP `a`, 2n variables) precede these policies. The offset is `ss_offset = 2n`. So Dynare index for consumption of sector j is `2n + j`, while `policies_ss` index is `j`.

---

## Aggregation: The Core Design Decision

### The inconsistency that was fixed

DEQN computes aggregate IRs using **fixed ergodic prices** (`P_ergodic`):

```
Agg_C_DEQN(t) = log(P_erg' · C(t)) − log(P_erg' · C_baseline(t))
```

MATLAB's pre-computed aggregate rows (1, 23) use **current prices** from Dynare's CES aggregator:

```
Agg_C_MATLAB_old(t) = log(Σ_j P_j(t) · C_j(t)) − log(Σ_j P_j_ss · C_j_ss)
```

These are not comparable. The fix: Python-side re-aggregation of MATLAB IRs using `P_ergodic`:

```
Agg_C_MATLAB_new(t) = log(P_erg' · C_j(t)) − log(P_erg' · C_j_ss)
```

### How it works in code

For **aggregate** variables ("Agg. Consumption", "Agg. Investment", "Agg. GDP"):
- `get_matlab_ir_fixedprice()` is called
- It reads full sectoral vectors from `sectoral_loglin` / `sectoral_determ`
- `_reaggregate_sectoral_ir()` converts log deviations → levels, weights by `P_ergodic`, sums, takes log deviation from SS
- Both numerator and denominator use `P_ergodic` so the IR starts at exactly 0

For **sectoral** variables ("Cj", "Pj", "Yj", etc.):
- `get_matlab_ir_for_analysis_variable()` is called (row-based lookup)
- No re-aggregation needed — these are single-sector quantities

### Reference point difference (intentional)

| | DEQN nonlinear | MATLAB benchmark |
|---|---|---|
| Starting point | Stochastic SS (or ergodic draws for GIR) | Deterministic SS |
| Aggregation weights | P_ergodic (fixed) | P_ergodic (fixed, after the fix) |

The stochastic vs. deterministic SS difference is intentional — it reveals the Jensen's inequality / precautionary savings gap.

---

## Key Dictionaries in `matlab_irs.py`

### `NEW_FORMAT_VARIABLE_INDICES`

Maps internal variable names to row indices in the 26-row `irs` matrix.

```python
"A": 0, "Cexp": 1, "Lexp": 2, "Cj": 3, "Pj": 4, ...
```

### `ANALYSIS_TO_MATLAB_MAPPING`

Maps human-readable analysis names to internal names:

```python
"Agg. Consumption": "Cexp",   # fallback row (current-price)
"Agg. Output": "GDPexp",      # fallback row
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

2. **Row index drift**: The 26-row matrix ordering is defined by the concatenation order in `process_ir_data.m`. The `NEW_FORMAT_VARIABLE_INDICES` dict in Python must mirror this exactly. There is no automated check.

3. **1-indexed vs 0-indexed**: MATLAB uses 1-indexed sector indices everywhere. Python uses 0-indexed. The `.mat` file stores MATLAB's 1-indexed values; `matlab_irs.py` converts to 0-indexed during loading.

4. **Log vs levels**: Most variables in `policies_ss` and the IR matrices are in **log** (log deviations from log SS). The exception is `A_ir` (row 0) and `A_client_ir` (row 12), which are in **levels** (`exp(dynare_simul(...))`).

5. **Format detection**: `matlab_irs.py` auto-detects legacy vs new `.mat` format. If the file was generated by an old version of `process_sector_irs.m`, the `sectoral_loglin`/`sectoral_determ` fields won't exist, and aggregate IRs will fall back to the current-price rows.
