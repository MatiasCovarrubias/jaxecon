import os
import json

import jax.numpy as jnp


def _write_analysis_config(config_dict, analysis_dir):
    config_path = os.path.join(analysis_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def _resolve_data_file(model_dir, configured_name, fallback_names, *, label, required):
    candidate_names = []
    for name in [configured_name, *fallback_names]:
        if name is None or name in candidate_names:
            continue
        candidate_names.append(name)

    for filename in candidate_names:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            if configured_name is not None and filename != configured_name:
                print(f"  Using fallback {label} file: {filename} (configured '{configured_name}' not found)")
            return filename, path

    if required:
        raise FileNotFoundError(f"{label} file not found in {model_dir}. Tried: {candidate_names}")

    if configured_name is not None:
        print(f"  ⚠ {label} file not found. Tried: {candidate_names} (skipping)")
    return None, None


def _normalize_dynare_simulation_orientation(simul_matrix, expected_n_vars, precision):
    arr = jnp.array(simul_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((expected_n_vars, 0), dtype=precision)
        if arr.size == expected_n_vars:
            return arr.reshape(expected_n_vars, 1)
        raise ValueError(
            "Unexpected 1D Dynare simulation vector with "
            f"length={arr.size}; expected length {expected_n_vars} for a single-period slice."
        )
    if arr.ndim != 2:
        raise ValueError(f"Unexpected Dynare simulation ndim={arr.ndim}; expected 1 or 2.")
    if arr.shape[0] == expected_n_vars:
        return arr
    if arr.shape[1] == expected_n_vars:
        return arr.T
    raise ValueError(f"Unexpected Dynare simulation shape {arr.shape}; expected one axis = {expected_n_vars}.")


def _extract_dynare_simulation_artifact(simul, method_names, expected_n_vars, precision):
    """Load active/full simulation paths for one Dynare method."""
    for method_name in method_names:
        method_block = simul.get(method_name)
        if not isinstance(method_block, dict):
            continue

        full_simul = None
        active_simul = None

        full_simul_raw = method_block.get("full_simul")
        if full_simul_raw is not None:
            full_simul = _normalize_dynare_simulation_orientation(full_simul_raw, expected_n_vars, precision)

        burnin_simul = method_block.get("burnin_simul")
        shocks_simul = method_block.get("shocks_simul")
        burnout_simul = method_block.get("burnout_simul")

        if shocks_simul is not None:
            active_simul = _normalize_dynare_simulation_orientation(shocks_simul, expected_n_vars, precision)

        if full_simul is None:
            windows = []
            for window in (burnin_simul, shocks_simul, burnout_simul):
                if window is None:
                    continue
                window_arr = _normalize_dynare_simulation_orientation(window, expected_n_vars, precision)
                if window_arr.shape[1] > 0:
                    windows.append(window_arr)
            if windows:
                full_simul = jnp.concatenate(windows, axis=1)

        if active_simul is None and full_simul is not None:
            burn_in = int(method_block.get("burn_in", 0))
            t_active = method_block.get("T_active")
            if t_active is not None:
                t_active = int(t_active)
                active_simul = full_simul[:, burn_in : burn_in + t_active]
            else:
                active_simul = full_simul

        if active_simul is not None or full_simul is not None:
            return {
                "active_simul": active_simul if active_simul is not None else full_simul,
                "full_simul": full_simul if full_simul is not None else active_simul,
            }

    return {"active_simul": None, "full_simul": None}


def _normalize_shock_matrix(shocks_matrix, shock_dimension, precision):
    arr = jnp.array(shocks_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((0, shock_dimension), dtype=precision)
        if arr.size == shock_dimension:
            return arr.reshape(1, shock_dimension)
        raise ValueError(
            "Unexpected 1D shock vector with "
            f"length={arr.size}; expected length {shock_dimension} for a single-period shock path."
        )
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D shock matrix, got shape {arr.shape}")
    if arr.shape[1] == shock_dimension:
        return arr
    if arr.shape[0] == shock_dimension:
        return arr.T
    raise ValueError(f"Unexpected shock matrix shape {arr.shape} for shock_dimension={shock_dimension}")


def _extract_matlab_common_shock_schedule(simul, shock_dimension, precision):
    shocks_block = simul.get("Shocks")
    if not isinstance(shocks_block, dict) or "data" not in shocks_block:
        return None

    active_shocks_full = _normalize_shock_matrix(shocks_block["data"], shock_dimension, precision)
    usage = shocks_block.get("usage", {})

    candidate_methods = [
        ("FirstOrder", ["FirstOrder", "Loglin", "LogLinear"]),
        ("SecondOrder", ["SecondOrder"]),
        ("PerfectForesight", ["PerfectForesight", "Determ"]),
        ("MITShocks", ["MITShocks", "MITShock"]),
    ]

    selected_method = None
    method_block = None
    usage_block = None
    for canonical_name, aliases in candidate_methods:
        for alias in aliases:
            block = simul.get(alias)
            if not isinstance(block, dict):
                continue
            usage_candidate = usage.get(alias) or usage.get(canonical_name)
            if usage_candidate is not None or "T_active" in block or "burn_in" in block:
                selected_method = canonical_name
                method_block = block
                usage_block = usage_candidate
                break
        if selected_method is not None:
            break

    if method_block is None:
        return None

    active_shocks = active_shocks_full
    if isinstance(usage_block, dict) and "start" in usage_block and "end" in usage_block:
        start_idx = max(int(usage_block["start"]) - 1, 0)
        end_idx = min(int(usage_block["end"]), active_shocks_full.shape[0])
        active_shocks = active_shocks_full[start_idx:end_idx]

    burn_in = int(method_block.get("burn_in", 0))
    burn_out = int(method_block.get("burn_out", 0))
    zero_burnin = jnp.zeros((burn_in, shock_dimension), dtype=precision)
    zero_burnout = jnp.zeros((burn_out, shock_dimension), dtype=precision)
    full_shocks = jnp.concatenate([zero_burnin, active_shocks, zero_burnout], axis=0)

    return {
        "reference_method": selected_method,
        "active_shocks": active_shocks,
        "full_shocks": full_shocks,
        "burn_in": burn_in,
        "burn_out": burn_out,
        "active_start": burn_in,
        "active_end": burn_in + active_shocks.shape[0],
    }


def _normalize_dynare_full_simul(simul_data, state_ss_vec, policies_ss_vec):
    """Return simulation in log deviations with shape (n_vars, T)."""
    expected_n_vars = state_ss_vec.shape[0] + policies_ss_vec.shape[0]
    if simul_data.shape[0] == expected_n_vars:
        simul_matrix = simul_data
    elif simul_data.shape[1] == expected_n_vars:
        simul_matrix = simul_data.T
    else:
        raise ValueError(
            f"Unexpected Dynare simulation shape {simul_data.shape}; expected one axis = {expected_n_vars}."
        )

    ss_full = jnp.concatenate([state_ss_vec, policies_ss_vec])
    dist_to_zero = jnp.mean(jnp.abs(simul_matrix[:, 0]))
    dist_to_ss = jnp.mean(jnp.abs(simul_matrix[:, 0] - ss_full))
    if dist_to_ss < dist_to_zero:
        simul_matrix = simul_matrix - ss_full[:, None]
    return simul_matrix
