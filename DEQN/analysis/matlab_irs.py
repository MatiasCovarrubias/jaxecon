"""
MATLAB Impulse Response Loader Module

This module provides functions to load and process impulse response data from MATLAB .mat files.
The MATLAB files contain perfect foresight and loglinear impulse responses computed from Dynare.

Supports two formats:
1. New format (Dec 2025+): Single ModelData_IRs.mat file with all shocks
2. Legacy format: Separate files per shock size/sign (AllSectors_IRS_*.mat)
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io as sio


def _coerce_mat_struct(obj: Any) -> Any:
    """
    Convert MATLAB struct-like objects from scipy.io.loadmat into plain Python types.

    Depending on scipy version and the exact .mat encoding, struct entries may arrive as
    dicts, numpy structured scalars (np.void), object arrays, or mat_struct-like objects.
    """
    if isinstance(obj, dict):
        return {k: _coerce_mat_struct(v) for k, v in obj.items()}

    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: _coerce_mat_struct(obj[name]) for name in obj.dtype.names}

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return [_coerce_mat_struct(v) for v in obj.ravel()]
        if obj.dtype == object:
            return [_coerce_mat_struct(v) for v in obj.ravel()]
        return obj

    # scipy mat_struct fallback for some scipy/matlab combinations.
    field_names = getattr(obj, "_fieldnames", None)
    if field_names:
        return {name: _coerce_mat_struct(getattr(obj, name)) for name in field_names}

    return obj


def _as_struct_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize a MATLAB struct/struct-array into a list of dictionaries.
    """
    normalized = _coerce_mat_struct(obj)
    if normalized is None:
        return []
    if isinstance(normalized, dict):
        return [normalized]
    if isinstance(normalized, list):
        return [item for item in normalized if isinstance(item, dict)]
    return []


def _to_python_sector_idx(raw_sector_idx: Any) -> int:
    """
    Convert sector index to Python 0-based convention.

    MATLAB objects typically store 1-based indices. Some preprocessed files may
    already be 0-based. This helper handles both safely.
    """
    if isinstance(raw_sector_idx, np.ndarray):
        raw_sector_idx = raw_sector_idx.item()
    sector_idx = int(raw_sector_idx)
    if sector_idx >= 1:
        return sector_idx - 1
    return sector_idx


def load_matlab_irs(
    matlab_ir_dir: str,
    shock_sizes: List[int] = [5, 10, 20],
    file_pattern: str = "AllSectors_IRS__Oct_25nonlinear_{sign}_{size}.mat",
    irs_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load MATLAB impulse response data.

    If irs_file_path is provided, loads directly from that file.
    Otherwise, attempts to find ModelData_IRs.mat in standard locations.
    Falls back to legacy separate files if not found.

    Args:
        matlab_ir_dir: Directory containing the MATLAB IR files
        shock_sizes: List of shock sizes in percent (used for legacy format)
        file_pattern: File name pattern for legacy format
        irs_file_path: Optional explicit path to ModelData_IRs.mat file

    Returns:
        Dictionary with structure:
        {
            "pos_20": {"sectors": {...}, "peak_values_loglin": array, ...},
            "neg_20": {"sectors": {...}, "peak_values_loglin": array, ...},
            ...
        }
    """
    # If explicit path provided, use it directly
    if irs_file_path is not None and os.path.exists(irs_file_path):
        print(f"  Loading IRs from: {irs_file_path}")
        return _load_new_format(irs_file_path)

    # First, try to find ModelData_IRs.mat in the model directory (parent of MATLAB/IRs)
    model_dir = os.path.dirname(os.path.dirname(matlab_ir_dir))
    modeldata_irs_path = os.path.join(model_dir, "ModelData_IRs.mat")

    if os.path.exists(modeldata_irs_path):
        print(f"  Found ModelData_IRs.mat: {modeldata_irs_path}")
        return _load_new_format(modeldata_irs_path)

    # Also check in the experiment folder (for saved results)
    experiments_dir = os.path.join(model_dir, "experiments")
    if os.path.exists(experiments_dir):
        for exp_folder in os.listdir(experiments_dir):
            exp_modeldata_path = os.path.join(experiments_dir, exp_folder, "ModelData_IRs.mat")
            if os.path.exists(exp_modeldata_path):
                print(f"  Found ModelData_IRs.mat in experiments: {exp_modeldata_path}")
                return _load_new_format(exp_modeldata_path)

    # Fall back to legacy format
    print("  No ModelData_IRs.mat found, trying legacy format...")
    return _load_legacy_format(matlab_ir_dir, shock_sizes, file_pattern)


def _parse_shock_label_to_key(label: str, shock_value: float) -> str:
    """
    Parse MATLAB shock label to standardized key format.

    Handles labels like:
    - "neg20pct" → "neg_20"
    - "pos5pct" → "pos_5"
    - "neg_5pct" → "neg_5"

    Falls back to computing from shock_value if label format is unexpected.

    Args:
        label: Shock label from MATLAB (e.g., "neg20pct")
        shock_value: Numeric shock value as fallback

    Returns:
        Standardized key like "neg_20" or "pos_5"
    """
    import re

    label_str = str(label).strip()
    match = re.search(r"(neg|pos)_?(\d+)pct", label_str, re.IGNORECASE)
    if match:
        sign = match.group(1).lower()
        pct = int(match.group(2))
        return f"{sign}_{pct}"

    # Fallback: use shock value and description heuristics
    # In MATLAB: -log(0.8) ≈ 0.223 means A drops to 0.8 (negative shock)
    # In MATLAB: log(1.2) ≈ 0.182 means A rises to 1.2 (positive shock)
    # The sign depends on how the label describes it, not just the numeric value
    if "neg" in label_str.lower():
        sign = "neg"
    elif "pos" in label_str.lower():
        sign = "pos"
    else:
        # Last resort: compute from value (may be inaccurate)
        sign = "neg" if shock_value > 0 else "pos"

    # Extract percentage from shock value
    pct = int(round(abs(1 - np.exp(-abs(shock_value))) * 100))

    return f"{sign}_{pct}"


def _load_new_format(filepath: str) -> Dict[str, Any]:
    """
    Load IR data from ModelData_IRs.mat produced by main.m.

    Supports both "new format" variants used by MATLAB:
    1) Flat structure (current main.m):
       - ModelData_IRs.shocks: struct array with .value/.label/.size_pct/.sign/...
       - ModelData_IRs.irfs: cell {n_shocks}, each element is struct array over sectors
         with fields .sector_idx, .first_order, .second_order, .perfect_foresight
       - ModelData_IRs.peaks / half_lives / amplifications: [n_shocks x n_sectors] matrices
    2) Nested structure (older transitional format):
       - ModelData_IRs.shock_configs + ModelData_IRs.by_shock

    Args:
        filepath: Path to ModelData_IRs.mat file

    Returns:
        Standardized dict: { "neg_20": { "sectors": { 0: {"IRSLoglin": ..., "IRSDeterm": ...}, ... }, ... }, ... }
    """
    try:
        mat_data = sio.loadmat(filepath, simplify_cells=True)
    except Exception as e:
        print(f"    ✗ Error loading {filepath}: {e}")
        return {}

    if "ModelData_IRs" not in mat_data:
        print(f"    ✗ 'ModelData_IRs' key not found in {filepath}")
        return {}

    md_irs = mat_data["ModelData_IRs"]
    ir_data = {}

    # Variant A: flat structure from current main.m
    if "irfs" in md_irs:
        shocks = _coerce_mat_struct(md_irs.get("shocks", []))
        irfs_by_shock = _coerce_mat_struct(md_irs.get("irfs", []))

        if not isinstance(shocks, (list, np.ndarray)):
            shocks = [shocks]
        if not isinstance(irfs_by_shock, (list, np.ndarray)):
            irfs_by_shock = [irfs_by_shock]

        n_shocks = min(len(shocks), len(irfs_by_shock))
        print(f"    Found {n_shocks} shock configurations (flat format)")

        for i in range(n_shocks):
            shock_cfg = _coerce_mat_struct(shocks[i])
            if not isinstance(shock_cfg, dict):
                shock_cfg = {}

            shock_value = shock_cfg.get("value", shock_cfg.get("Value", 0))
            shock_label = shock_cfg.get("label", shock_cfg.get("Label", f"shock_{i}"))
            shock_desc = shock_cfg.get("description", shock_cfg.get("Description", ""))

            key = _parse_shock_label_to_key(shock_label, shock_value)
            print(f"    Processing: {shock_label} ({shock_desc}) → key={key}")

            processed = _process_flat_format_shock(md_irs, i)
            if processed:
                ir_data[key] = processed
                print(f"      ✓ Loaded {len(processed.get('sectors', {}))} sectors")

        return ir_data

    # Variant B: nested structure
    shock_configs = md_irs.get("shock_configs", [])
    by_shock = md_irs.get("by_shock", [])

    if not isinstance(shock_configs, (list, np.ndarray)):
        shock_configs = [shock_configs]
    if not isinstance(by_shock, (list, np.ndarray)):
        by_shock = [by_shock]

    n_shocks = min(len(shock_configs), len(by_shock))
    print(f"    Found {n_shocks} shock configurations (nested format)")

    for i in range(n_shocks):
        shock_cfg = _coerce_mat_struct(shock_configs[i])
        if not isinstance(shock_cfg, dict):
            shock_cfg = {}
        shock_result = _coerce_mat_struct(by_shock[i])

        shock_value = shock_cfg.get("value", shock_cfg.get("Value", 0))
        shock_label = shock_cfg.get("label", shock_cfg.get("Label", f"shock_{i}"))
        shock_desc = shock_cfg.get("description", shock_cfg.get("Description", ""))

        key = _parse_shock_label_to_key(shock_label, shock_value)
        print(f"    Processing: {shock_label} ({shock_desc}) → key={key}")

        if not isinstance(shock_result, dict):
            shock_result = {}
        processed = _process_new_format_shock(shock_result)
        if processed:
            ir_data[key] = processed
            print(f"      ✓ Loaded {len(processed.get('sectors', {}))} sectors")

    return ir_data


def _process_flat_format_shock(md_irs: Dict, shock_idx: int) -> Dict[str, Any]:
    """
    Process one shock from the flat ModelData_IRs format.
    """
    processed = {
        "sectors": {},
        "peak_values_loglin": None,
        "peak_values_determ": None,
        "amplifications": None,
        "half_lives_loglin": None,
        "half_lives_determ": None,
    }

    peaks = md_irs.get("peaks", {}) or {}
    half_lives = md_irs.get("half_lives", {}) or {}
    amplifications = md_irs.get("amplifications", {}) or {}

    peak_fo = peaks.get("first_order")
    peak_pf = peaks.get("perfect_foresight")
    hl_fo = half_lives.get("first_order")
    hl_pf = half_lives.get("perfect_foresight")
    amp_abs = amplifications.get("abs")

    def _extract_shock_row(data: Any, idx: int) -> Optional[np.ndarray]:
        """Extract one shock slice robustly under simplify_cells singleton squeezing."""
        if data is None:
            return None
        arr = np.asarray(data)
        if arr.size == 0:
            return None

        # Scalar: single shock/single sector case.
        if arr.ndim == 0:
            return np.atleast_1d(arr.item())

        # Vector: either [n_shocks] (single sector) or [n_sectors] (single shock).
        if arr.ndim == 1:
            if arr.size > idx and arr.size > 1:
                return np.atleast_1d(arr[idx])
            return np.atleast_1d(arr)

        # Matrix/tensor: try shock dimension first (rows), then fallback to columns.
        if arr.shape[0] > idx:
            return np.atleast_1d(arr[idx, ...]).ravel()
        if arr.shape[1] > idx:
            return np.atleast_1d(arr[:, idx, ...]).ravel()

        return np.atleast_1d(arr).ravel()

    processed["peak_values_loglin"] = _extract_shock_row(peak_fo, shock_idx)
    processed["peak_values_determ"] = _extract_shock_row(peak_pf, shock_idx)
    processed["half_lives_loglin"] = _extract_shock_row(hl_fo, shock_idx)
    processed["half_lives_determ"] = _extract_shock_row(hl_pf, shock_idx)
    processed["amplifications"] = _extract_shock_row(amp_abs, shock_idx)

    irfs_by_shock = md_irs.get("irfs", [])
    if not isinstance(irfs_by_shock, (list, np.ndarray)) or shock_idx >= len(irfs_by_shock):
        return processed

    irfs = _as_struct_list(irfs_by_shock[shock_idx])

    for irf_data in irfs:
        if irf_data is None:
            continue
        if not isinstance(irf_data, dict):
            continue

        sector_idx = _to_python_sector_idx(irf_data.get("sector_idx", 0))
        if sector_idx < 0:
            continue

        irs_first_order = irf_data.get("first_order", irf_data.get("IRSFirstOrder"))
        irs_second_order = irf_data.get("second_order", irf_data.get("IRSSecondOrder"))
        irs_pf = irf_data.get("perfect_foresight", irf_data.get("IRSPerfectForesight"))
        if irs_pf is None:
            irs_pf = irf_data.get("IRSPF", irf_data.get("IRSDeterm"))

        if irs_first_order is None:
            continue

        sector_entry = {
            "IRSFirstOrder": np.array(irs_first_order),
            "IRSSecondOrder": np.array(irs_second_order) if irs_second_order is not None else None,
            "IRSPerfectForesight": np.array(irs_pf) if irs_pf is not None else None,
            # Backward-compatible keys.
            "IRSLoglin": np.array(irs_first_order),
            "IRSDeterm": np.array(irs_pf) if irs_pf is not None else None,
        }

        # Extract full sectoral vectors if available
        for method_key, struct_key in [
            ("sectoral_first_order", "sectoral_loglin"),
            ("sectoral_perfect_foresight", "sectoral_determ"),
        ]:
            sectoral_data = irf_data.get(struct_key)
            if sectoral_data is not None:
                sectoral_dict = _coerce_mat_struct(sectoral_data)
                if isinstance(sectoral_dict, dict):
                    parsed = {}
                    for field in ("C_all", "Iout_all", "Q_all", "Mout_all"):
                        arr = sectoral_dict.get(field)
                        if arr is not None:
                            parsed[field] = np.array(arr)
                    if parsed:
                        sector_entry[method_key] = parsed

        processed["sectors"][sector_idx] = sector_entry

    return processed


def _process_new_format_shock(shock_result: Dict) -> Dict[str, Any]:
    """
    Process a single shock result from the new format.

    Args:
        shock_result: Dictionary with IRFs and Statistics

    Returns:
        Processed dictionary with sectors and statistics
    """
    if shock_result is None:
        return {}

    processed = {
        "sectors": {},
        "peak_values_loglin": None,
        "peak_values_determ": None,
        "amplifications": None,
        "half_lives_loglin": None,
        "half_lives_determ": None,
    }

    # Extract statistics (MATLAB uses peak_values_firstorder/peak_values_pf; legacy uses _loglin/_determ)
    stats = shock_result.get("Statistics")
    if stats is None:
        stats = {}
    if stats:
        v = stats.get("peak_values_loglin")
        processed["peak_values_loglin"] = v if v is not None else stats.get("peak_values_firstorder")
        v = stats.get("peak_values_determ")
        processed["peak_values_determ"] = v if v is not None else stats.get("peak_values_pf")
        processed["amplifications"] = stats.get("amplifications")
        v = stats.get("half_lives_loglin")
        processed["half_lives_loglin"] = v if v is not None else stats.get("half_lives_firstorder")
        v = stats.get("half_lives_determ")
        processed["half_lives_determ"] = v if v is not None else stats.get("half_lives_pf")

    # IRF field names from process_sector_irs.m: IRSFirstOrder, IRSPerfectForesight (and optional IRSSecondOrder)
    irfs = _as_struct_list(shock_result.get("IRFs", []))

    for irf_data in irfs:
        if irf_data is None:
            continue
        if not isinstance(irf_data, dict):
            continue

        sector_idx = _to_python_sector_idx(irf_data.get("sector_idx", 0))
        if sector_idx < 0:
            continue

        irs_first_order = irf_data.get("IRSFirstOrder")
        if irs_first_order is None:
            irs_first_order = irf_data.get("IRSLoglin")
        irs_second_order = irf_data.get("IRSSecondOrder")
        irs_pf = irf_data.get("IRSPerfectForesight")
        if irs_pf is None:
            irs_pf = irf_data.get("IRSPF")
        if irs_pf is None:
            irs_pf = irf_data.get("IRSDeterm")

        if irs_first_order is not None:
            arr_first = np.array(irs_first_order)
            arr_second = np.array(irs_second_order) if irs_second_order is not None else None
            arr_pf = np.array(irs_pf) if irs_pf is not None else None
            sector_entry = {
                "IRSFirstOrder": arr_first,
                "IRSSecondOrder": arr_second,
                "IRSPerfectForesight": arr_pf,
                # Backward-compatible keys.
                "IRSLoglin": arr_first,
                "IRSDeterm": arr_pf,
            }

            for method_key, struct_key in [
                ("sectoral_first_order", "sectoral_loglin"),
                ("sectoral_perfect_foresight", "sectoral_determ"),
            ]:
                sectoral_data = irf_data.get(struct_key)
                if sectoral_data is not None:
                    sectoral_dict = _coerce_mat_struct(sectoral_data)
                    if isinstance(sectoral_dict, dict):
                        parsed = {}
                        for field in ("C_all", "Iout_all", "Q_all", "Mout_all"):
                            arr = sectoral_dict.get(field)
                            if arr is not None:
                                parsed[field] = np.array(arr)
                        if parsed:
                            sector_entry[method_key] = parsed

            processed["sectors"][sector_idx] = sector_entry

    return processed


def _load_legacy_format(
    matlab_ir_dir: str,
    shock_sizes: List[int],
    file_pattern: str,
) -> Dict[str, Any]:
    """
    Load MATLAB impulse response files using the legacy format (separate files).

    Args:
        matlab_ir_dir: Directory containing the MATLAB IR files
        shock_sizes: List of shock sizes in percent (e.g., [5, 10, 20])
        file_pattern: File name pattern with {sign} and {size} placeholders

    Returns:
        Dictionary with structure compatible with new format
    """
    ir_data = {}
    files_found = 0
    files_missing = 0
    files_error = 0

    print(f"  Looking for legacy MATLAB IR files in: {matlab_ir_dir}")

    for size in shock_sizes:
        for sign in ["pos", "neg"]:
            filename = file_pattern.format(sign=sign, size=size)
            filepath = os.path.join(matlab_ir_dir, filename)

            if not os.path.exists(filepath):
                print(f"    ✗ NOT FOUND: {filename}")
                files_missing += 1
                continue

            try:
                mat_data = sio.loadmat(filepath, simplify_cells=True)["AllIRS"]
                key = f"{sign}_{size}"
                ir_data[key] = _process_legacy_ir_data(mat_data)
                print(f"    ✓ Loaded: {filename}")
                files_found += 1
            except Exception as e:
                print(f"    ✗ ERROR loading {filename}: {e}")
                files_error += 1

    print(f"  Summary: {files_found} loaded, {files_missing} missing, {files_error} errors")

    return ir_data


def _process_legacy_ir_data(mat_data: Dict) -> Dict[str, Any]:
    """
    Process raw MATLAB IR data from legacy format into standardized structure.

    Args:
        mat_data: Raw dictionary from scipy.io.loadmat

    Returns:
        Processed dictionary with sectors and global statistics
    """
    processed = {
        "sectors": {},
        "peak_values_shock": mat_data.get("peak_values_shock"),
        "peak_values_loglin": mat_data.get("peak_values_loglin"),
        "peak_values_determ": mat_data.get("peak_values_determ"),
        "peak_periods_shock": mat_data.get("peak_periods_shock"),
        "peak_periods_loglin": mat_data.get("peak_periods_loglin"),
        "peak_periods_determ": mat_data.get("peak_periods_determ"),
        "half_lives_shock": mat_data.get("half_lives_shock"),
        "half_lives_loglin": mat_data.get("half_lives_loglin"),
        "half_lives_determ": mat_data.get("half_lives_determ"),
        "amplifications": mat_data.get("amplifications"),
    }

    sector_keys = [k for k in mat_data.keys() if k.startswith("Sector_")]
    for sector_key in sorted(sector_keys, key=lambda x: int(x.split("_")[1])):
        sector_data = mat_data[sector_key]
        sector_idx = int(sector_key.split("_")[1]) - 1

        processed["sectors"][sector_idx] = {
            "IRSFirstOrder": np.array(sector_data["IRSLoglin"]),
            "IRSSecondOrder": None,
            "IRSPerfectForesight": np.array(sector_data["IRSDeterm"]),
            "IRSLoglin": np.array(sector_data["IRSLoglin"]),
            "IRSDeterm": np.array(sector_data["IRSDeterm"]),
        }

    return processed


def get_sector_irs(
    ir_data: Dict[str, Any],
    sector_idx: int,
    variable_idx: int = 0,
    max_periods: int = 100,
    skip_initial: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract IRs for a specific sector and variable across all shock sizes/signs.

    Args:
        ir_data: Dictionary from load_matlab_irs
        sector_idx: Sector index (0-based)
        variable_idx: Variable index within the IR array
        max_periods: Maximum number of periods to return
        skip_initial: If True, skip period 0 (default True for correct alignment).

                      Dynare timing convention:
                      - MATLAB IR index 0: Initial condition with shocked TFP, but policies
                        are still at steady state (no response yet)
                      - MATLAB IR index 1+: Policies respond to the shocked TFP

                      Python GIR:
                      - GIR index 0: Policy response to shocked state (first response)

                      With skip_initial=True: MATLAB[1:] aligns with Python GIR[0:]

    Returns:
        Dictionary with structure:
        {
            "pos_5": {"loglin": array, "determ": array},
            "neg_5": {"loglin": array, "determ": array},
            ...
        }
    """
    result = {}

    def _extract_series(ir_array: Any) -> Optional[np.ndarray]:
        """
        Extract one variable IR series robustly from MATLAB-loaded arrays.

        Expected canonical shape is (n_variables, n_periods), but scipy's
        simplify_cells can squeeze singleton dimensions into 1D vectors.
        """
        if ir_array is None:
            return None

        arr = np.asarray(ir_array)
        if arr.size == 0:
            return None

        # Scalar edge case.
        if arr.ndim == 0:
            return np.atleast_1d(arr.item())

        # Squeezed single-variable IR: treat as time series.
        if arr.ndim == 1:
            if variable_idx != 0:
                return None
            return arr

        # Canonical orientation: [variables, time].
        if arr.shape[0] > variable_idx:
            return np.asarray(arr[variable_idx, ...]).ravel()

        # Transposed orientation: [time, variables].
        if arr.shape[1] > variable_idx:
            return np.asarray(arr[:, variable_idx]).ravel()

        return None

    for key, data in ir_data.items():
        if "sectors" not in data or sector_idx not in data["sectors"]:
            continue

        sector_data = data["sectors"][sector_idx]

        first_raw = _extract_series(sector_data.get("IRSFirstOrder"))
        if first_raw is None:
            continue
        if skip_initial:
            first_order = first_raw[1 : max_periods + 1]
        else:
            first_order = first_raw[:max_periods]

        irs_second = sector_data.get("IRSSecondOrder")
        second_raw = _extract_series(irs_second)
        if second_raw is not None:
            if skip_initial:
                second_order = second_raw[1 : max_periods + 1]
            else:
                second_order = second_raw[:max_periods]
        else:
            second_order = None

        irs_pf = sector_data.get("IRSPerfectForesight")
        pf_raw = _extract_series(irs_pf)
        if pf_raw is not None:
            if skip_initial:
                perfect_foresight = pf_raw[1 : max_periods + 1]
            else:
                perfect_foresight = pf_raw[:max_periods]
        else:
            perfect_foresight = None

        result[key] = {
            "first_order": first_order,
            "second_order": second_order,
            "perfect_foresight": perfect_foresight,
            # Backward-compatible aliases.
            "loglin": first_order,
            "determ": perfect_foresight,
        }

    return result


# New format (Dec 2025+), actual rows from process_ir_data.m:
#   Row  1: A_ir        (TFP level of shocked sector)
#   Row  2: C_ir        (Dynare CES aggregate consumption, current prices)
#   Row  3: L_ir        (Dynare CES aggregate labor)
#   Row  4: Cj_ir       (sectoral consumption, shocked sector)
#   Row  5: Pj_ir       (sectoral price, shocked sector)
#   Row  6: Ioutj_ir    (sectoral investment output, shocked sector)
#   Row  7: Moutj_ir    (sectoral intermediate output, shocked sector)
#   Row  8: Lj_ir       (sectoral labor, shocked sector)
#   Row  9: Ij_ir       (sectoral investment input, shocked sector)
#   Row 10: Mj_ir       (sectoral intermediate input, shocked sector)
#   Row 11: Yj_ir       (sectoral value added, shocked sector)
#   Row 12: Qj_ir       (sectoral gross output, shocked sector)
#   Row 13: A_client_ir (TFP level, client sector)
#   Row 14: Cj_client_ir
#   Row 15: Pj_client_ir
#   Row 16: Ioutj_client_ir
#   Row 17: Moutj_client_ir
#   Row 18: Lj_client_ir
#   Row 19: Ij_client_ir
#   Row 20: Mj_client_ir
#   Row 21: Yj_client_ir
#   Row 22: Qj_client_ir
#   Row 23: Kj_ir       (sectoral capital, shocked sector)
#   Row 24: Y_ir        (Dynare aggregate output, current prices)
#   Row 25: Pmj_client_ir
#   Row 26: gammaij_client_ir
#
# NOTE: Rows 2, 3, 24 are Dynare's built-in aggregates (current-price weighted).
# For aggregate IRs (Agg. Consumption, Agg. Investment, Agg. GDP), the Python side
# re-aggregates from full sectoral vectors stored in sectoral_loglin/sectoral_determ
# using fixed ergodic prices (P_ergodic) via get_matlab_ir_fixedprice().
# This ensures consistency with the DEQN nonlinear IRs.
NEW_FORMAT_VARIABLE_INDICES = {
    "A": 0,  # TFP level (shocked sector)
    "Cexp": 1,  # Dynare aggregate consumption (CES / current-price)
    "Lexp": 2,  # Dynare aggregate labor
    "Cj": 3,  # Sectoral consumption
    "Pj": 4,  # Sectoral price
    "Ioutj": 5,  # Sectoral investment output
    "Moutj": 6,  # Sectoral intermediate output
    "Lj": 7,  # Sectoral labor
    "Ij": 8,  # Sectoral investment input
    "Mj": 9,  # Sectoral intermediate input
    "Yj": 10,  # Sectoral output
    "Qj": 11,  # Sectoral Tobin's Q
    "A_client": 12,  # Client TFP level
    "Cj_client": 13,  # Client consumption
    "Pj_client": 14,  # Client price
    "Ioutj_client": 15,  # Client investment output
    "Moutj_client": 16,  # Client intermediate output
    "Lj_client": 17,  # Client labor
    "Ij_client": 18,  # Client investment input
    "Mj_client": 19,  # Client intermediate input
    "Yj_client": 20,  # Client output
    "Qj_client": 21,  # Client Tobin's Q
    "Kj": 22,  # Sectoral capital
    "GDPexp": 23,  # Dynare aggregate output (current-price)
    "Pmj_client": 24,  # Client intermediate price
    "gammaij_client": 25,  # Client expenditure share deviation
}

# Legacy format variable indices (for backwards compatibility)
LEGACY_VARIABLE_INDICES = {
    "K": 0,
    "C": 2,
    "L": 3,
    "Pk": 4,
    "Pm": 5,
    "M": 6,
    "Mout": 7,
    "Inv": 8,
    "P": 10,
    "Y": 11,
    "Q": 12,
    "Cagg": 14,
    "Kagg": 15,
    "Yagg": 16,
    "Magg": 17,
    "Iagg": 18,
    "Wage": 19,
    "Lagg": 20,
    "R": 21,
    "GDP": 22,
    "Utility": 23,
    "Lambda": 24,
    "Mu_k": 25,
    "Mu_c": 26,
}

# Active variable indices (auto-detected based on format)
MATLAB_IR_VARIABLE_INDICES = NEW_FORMAT_VARIABLE_INDICES.copy()

# Mapping from analysis variable names to MATLAB variable names
ANALYSIS_TO_MATLAB_MAPPING = {
    "Agg. Consumption": "Cexp",
    "Agg. Output": "GDPexp",
    "Agg. GDP": "GDPexp",
    "Agg. Labor": "Lexp",
    # Own-sector variables
    "Cj": "Cj",
    "Pj": "Pj",
    "Ioutj": "Ioutj",
    "Moutj": "Moutj",
    "Lj": "Lj",
    "Ij": "Ij",
    "Mj": "Mj",
    "Yj": "Yj",
    "Qj": "Qj",
    "Kj": "Kj",
    # Client (main customer) variables
    "Cj_client": "Cj_client",
    "Pj_client": "Pj_client",
    "Ioutj_client": "Ioutj_client",
    "Moutj_client": "Moutj_client",
    "Lj_client": "Lj_client",
    "Ij_client": "Ij_client",
    "Mj_client": "Mj_client",
    "Yj_client": "Yj_client",
    "Qj_client": "Qj_client",
    "Pmj_client": "Pmj_client",
    "gammaij_client": "gammaij_client",
}


def set_matlab_variable_mapping(mapping: Dict[str, int]) -> None:
    """
    Update the MATLAB variable index mapping.

    Args:
        mapping: Dictionary mapping variable names to row indices in MATLAB IR arrays
    """
    global MATLAB_IR_VARIABLE_INDICES
    MATLAB_IR_VARIABLE_INDICES.update(mapping)


def set_analysis_to_matlab_mapping(mapping: Dict[str, str]) -> None:
    """
    Update the mapping from analysis variable names to MATLAB variable names.

    Args:
        mapping: Dictionary mapping analysis variable names to MATLAB variable names
    """
    global ANALYSIS_TO_MATLAB_MAPPING
    ANALYSIS_TO_MATLAB_MAPPING.update(mapping)


def get_matlab_ir_for_analysis_variable(
    ir_data: Dict[str, Any],
    sector_idx: int,
    analysis_var_name: str,
    max_periods: int = 100,
    skip_initial: bool = True,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Get MATLAB IRs for a given analysis variable name.

    Args:
        ir_data: Dictionary from load_matlab_irs
        sector_idx: Sector index (0-based)
        analysis_var_name: Name of analysis variable (e.g., "Agg. Consumption")
        max_periods: Maximum number of periods to return
        skip_initial: If True, skip period 0 (default True for correct alignment).
                      MATLAB IR[0] is initial condition; GIR[0] is first response.

    Returns:
        Dictionary with IRs for each shock size/sign, or None if variable not found
    """
    if analysis_var_name not in ANALYSIS_TO_MATLAB_MAPPING:
        return None

    matlab_var_name = ANALYSIS_TO_MATLAB_MAPPING[analysis_var_name]

    if matlab_var_name not in MATLAB_IR_VARIABLE_INDICES:
        return None

    var_idx = MATLAB_IR_VARIABLE_INDICES[matlab_var_name]

    return get_sector_irs(ir_data, sector_idx, var_idx, max_periods, skip_initial)


AGGREGATE_VARIABLE_SECTORAL_MAP = {
    "Agg. Consumption": "C_all",
    "Agg. Investment": "Iout_all",
}

AGGREGATE_VARIABLE_GDP_COMPONENTS = {
    "Agg. Output": ("Q_all", "Mout_all"),
    "Agg. GDP": ("Q_all", "Mout_all"),
}


def _reaggregate_sectoral_ir(
    sectoral_data: Dict[str, np.ndarray],
    policies_ss: np.ndarray,
    P_ergodic: np.ndarray,
    analysis_var_name: str,
    n_sectors: int,
    skip_initial: bool = True,
    max_periods: int = 100,
) -> Optional[np.ndarray]:
    """
    Re-aggregate a MATLAB IR using fixed ergodic prices from sectoral vectors.

    The sectoral data contains log deviations from SS for all n_sectors.
    We convert to levels, weight by P_ergodic, sum, and take log deviations
    from the deterministic SS aggregate.

    Args:
        sectoral_data: Dict with C_all, Iout_all, Q_all, Mout_all (each n_sectors x T)
        policies_ss: MATLAB steady-state policies in log (412-vector)
        P_ergodic: Fixed ergodic output prices in levels (n_sectors-vector)
        analysis_var_name: Which aggregate to compute
        n_sectors: Number of sectors
        skip_initial: If True, skip period 0 (align MATLAB[1:] with DEQN GIR[0:])
        max_periods: Maximum periods to return

    Returns:
        IR as log deviations from deterministic SS, or None if data not available
    """
    n = n_sectors
    ps_levels = np.exp(policies_ss)

    if analysis_var_name in AGGREGATE_VARIABLE_SECTORAL_MAP:
        field = AGGREGATE_VARIABLE_SECTORAL_MAP[analysis_var_name]
        logdev_all = sectoral_data.get(field)
        if logdev_all is None:
            return None

        if field == "C_all":
            X_ss = ps_levels[:n]
        elif field == "Iout_all":
            X_ss = ps_levels[7 * n : 8 * n]
        else:
            return None

        X_levels = X_ss[:, None] * np.exp(logdev_all)

        agg_t = P_ergodic @ X_levels
        agg_ss = P_ergodic @ X_ss

    elif analysis_var_name in AGGREGATE_VARIABLE_GDP_COMPONENTS:
        q_field, mout_field = AGGREGATE_VARIABLE_GDP_COMPONENTS[analysis_var_name]
        logdev_q = sectoral_data.get(q_field)
        logdev_mout = sectoral_data.get(mout_field)
        if logdev_q is None or logdev_mout is None:
            return None

        Q_ss = ps_levels[9 * n : 10 * n]
        Mout_ss = ps_levels[5 * n : 6 * n]

        Q_levels = Q_ss[:, None] * np.exp(logdev_q)
        Mout_levels = Mout_ss[:, None] * np.exp(logdev_mout)

        agg_t = P_ergodic @ (Q_levels - Mout_levels)
        agg_ss = P_ergodic @ (Q_ss - Mout_ss)
    else:
        return None

    eps = 1e-12
    ir = np.log(np.maximum(agg_t, eps)) - np.log(np.maximum(agg_ss, eps))

    if skip_initial:
        ir = ir[1 : max_periods + 1]
    else:
        ir = ir[:max_periods]

    return ir


def get_matlab_ir_fixedprice(
    ir_data: Dict[str, Any],
    sector_idx: int,
    analysis_var_name: str,
    policies_ss: np.ndarray,
    P_ergodic: np.ndarray,
    n_sectors: int,
    max_periods: int = 100,
    skip_initial: bool = True,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Get MATLAB IRs re-aggregated with fixed ergodic prices.

    Falls back to the pre-computed row if sectoral data is not available.

    Args:
        ir_data: Dictionary from load_matlab_irs
        sector_idx: Sector index (0-based)
        analysis_var_name: Name of analysis variable (e.g., "Agg. Consumption")
        policies_ss: MATLAB steady-state policies in log
        P_ergodic: Ergodic output prices in levels
        n_sectors: Number of sectors
        max_periods: Maximum number of periods to return
        skip_initial: If True, skip period 0

    Returns:
        Dictionary with IRs for each shock size/sign, or None if variable not found
    """
    is_aggregate = (
        analysis_var_name in AGGREGATE_VARIABLE_SECTORAL_MAP
        or analysis_var_name in AGGREGATE_VARIABLE_GDP_COMPONENTS
    )

    if not is_aggregate:
        return get_matlab_ir_for_analysis_variable(
            ir_data, sector_idx, analysis_var_name, max_periods, skip_initial
        )

    result = {}

    for key, data in ir_data.items():
        if "sectors" not in data or sector_idx not in data["sectors"]:
            continue

        sector_data = data["sectors"][sector_idx]

        entry = {}
        for method_label, sectoral_key in [
            ("first_order", "sectoral_first_order"),
            ("perfect_foresight", "sectoral_perfect_foresight"),
        ]:
            sectoral = sector_data.get(sectoral_key)
            if sectoral is not None:
                reagg = _reaggregate_sectoral_ir(
                    sectoral, policies_ss, P_ergodic, analysis_var_name,
                    n_sectors, skip_initial, max_periods,
                )
                if reagg is not None:
                    entry[method_label] = reagg

        if entry:
            entry["loglin"] = entry.get("first_order")
            entry["determ"] = entry.get("perfect_foresight")
            result[key] = entry

    if result:
        return result

    return get_matlab_ir_for_analysis_variable(
        ir_data, sector_idx, analysis_var_name, max_periods, skip_initial
    )


def get_amplification_stats(ir_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract amplification statistics across all shock sizes.

    Args:
        ir_data: Dictionary from load_matlab_irs

    Returns:
        Dictionary with amplification data for each shock configuration
    """
    stats = {}

    for key, data in ir_data.items():
        stats[key] = {
            "peak_values_loglin": data.get("peak_values_loglin"),
            "peak_values_determ": data.get("peak_values_determ"),
            "amplifications": data.get("amplifications"),
            "half_lives_loglin": data.get("half_lives_loglin"),
            "half_lives_determ": data.get("half_lives_determ"),
        }

    return stats


def inspect_matlab_ir_structure(
    filepath: str,
    sector_idx: int = 0,
    shock_idx: int = 0,
) -> None:
    """
    Utility function to inspect MATLAB IR data structure and help identify variable mappings.

    This function prints summary statistics for each row in the IR arrays to help
    identify which row corresponds to which variable.

    Args:
        filepath: Path to ModelData_IRs.mat or legacy IR file
        sector_idx: Sector to inspect (0-based, relative to available sectors)
        shock_idx: Shock configuration to inspect (0-based)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    mat_data = sio.loadmat(filepath, simplify_cells=True)

    # Check if new format
    if "ModelData_IRs" in mat_data:
        md_irs = mat_data["ModelData_IRs"]

        print("\n=== New Format IR Structure ===")

        # Flat structure path
        if "irfs" in md_irs:
            shocks = md_irs.get("shocks", [])
            irfs_by_shock = md_irs.get("irfs", [])

            if not isinstance(shocks, (list, np.ndarray)):
                shocks = [shocks]
            if not isinstance(irfs_by_shock, (list, np.ndarray)):
                irfs_by_shock = [irfs_by_shock]

            if shock_idx >= len(irfs_by_shock):
                print(f"Shock index {shock_idx} out of range (max: {len(irfs_by_shock) - 1})")
                return

            shock_cfg = shocks[shock_idx] if shock_idx < len(shocks) else {}
            if not isinstance(shock_cfg, dict):
                shock_cfg = {}
            print(f"Shock: {shock_cfg.get('label', f'shock_{shock_idx}')} ({shock_cfg.get('description', '')})")

            irfs = irfs_by_shock[shock_idx]
            if not isinstance(irfs, (list, np.ndarray)):
                irfs = [irfs]
            if sector_idx >= len(irfs):
                print(f"Sector index {sector_idx} out of range (max: {len(irfs) - 1})")
                return

            irf_data = irfs[sector_idx]
            loglin = irf_data.get("first_order", irf_data.get("IRSFirstOrder"))
            determ = irf_data.get("perfect_foresight", irf_data.get("IRSPerfectForesight"))
            if determ is None:
                determ = irf_data.get("IRSDeterm")

            if loglin is None:
                print("No first_order/IRSFirstOrder data found")
                return

            _print_ir_analysis(
                np.array(loglin),
                np.array(determ) if determ is not None else None,
                irf_data.get("sector_idx", sector_idx),
            )
            return

        # Nested structure path (backward compatibility)
        by_shock = md_irs.get("by_shock", [])
        if shock_idx >= len(by_shock):
            print(f"Shock index {shock_idx} out of range (max: {len(by_shock) - 1})")
            return

        shock_result = by_shock[shock_idx]
        shock_cfg = md_irs.get("shock_configs", [])[shock_idx]
        print(f"Shock: {shock_cfg.get('label', 'unknown')} ({shock_cfg.get('description', '')})")

        irfs = shock_result.get("IRFs", [])
        if sector_idx >= len(irfs):
            print(f"Sector index {sector_idx} out of range (max: {len(irfs) - 1})")
            return

        irf_data = irfs[sector_idx]
        loglin = irf_data.get("IRSLoglin")
        determ = irf_data.get("IRSDeterm")

        if loglin is None:
            print("No IRSLoglin data found")
            return

        _print_ir_analysis(loglin, determ, irf_data.get("sector_idx", sector_idx))
    else:
        # Legacy format
        if "AllIRS" not in mat_data:
            print("Unknown file format")
            return

        mat_data = mat_data["AllIRS"]
        sector_key = f"Sector_{sector_idx + 1}"

        if sector_key not in mat_data:
            print(f"Sector {sector_idx + 1} not found in data")
            return

        sector_data = mat_data[sector_key]
        loglin = sector_data["IRSLoglin"]
        determ = sector_data["IRSDeterm"]

        print(f"\n=== Legacy Format IR Structure for {sector_key} ===")
        _print_ir_analysis(loglin, determ, sector_idx + 1)


def _print_ir_analysis(loglin: np.ndarray, determ: Optional[np.ndarray], sector_id: int) -> None:
    """Helper to print IR array analysis."""
    print(f"Sector ID: {sector_id}")
    print(f"IRSLoglin shape: {loglin.shape} (rows=variables, cols=time periods)")
    if determ is not None:
        print(f"IRSDeterm shape: {determ.shape}")
    else:
        print("IRSDeterm shape: None")

    print(f"\n{'Row':<5} {'Initial':<12} {'Peak':<12} {'Final':<12} {'Monotonic?':<10} Notes")
    print("-" * 70)

    for i in range(loglin.shape[0]):
        row = loglin[i, :]
        initial = row[0] if len(row) > 0 else 0
        peak = row[np.argmax(np.abs(row))] if len(row) > 0 else 0
        final = row[-1] if len(row) > 0 else 0

        diffs = np.diff(row)
        monotonic = "Yes" if np.all(diffs >= 0) or np.all(diffs <= 0) else "No"

        notes = ""
        if abs(initial) > 0.1:
            notes += f"Starts at {initial:.2f} "
        if np.all(row == row[0]):
            notes += "Constant "
        if abs(peak) > 0.5:
            notes += "Large swing "

        print(f"{i:<5} {initial:<12.4f} {peak:<12.4f} {final:<12.4f} {monotonic:<10} {notes}")

    print("\nNew format variable mapping (from process_ir_data.m):")
    for name, idx in NEW_FORMAT_VARIABLE_INDICES.items():
        print(f"  Row {idx:2d}: {name}")
