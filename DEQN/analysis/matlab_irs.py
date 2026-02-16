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
        shocks = md_irs.get("shocks", [])
        irfs_by_shock = md_irs.get("irfs", [])

        if not isinstance(shocks, (list, np.ndarray)):
            shocks = [shocks]
        if not isinstance(irfs_by_shock, (list, np.ndarray)):
            irfs_by_shock = [irfs_by_shock]

        n_shocks = min(len(shocks), len(irfs_by_shock))
        print(f"    Found {n_shocks} shock configurations (flat format)")

        for i in range(n_shocks):
            shock_cfg = shocks[i]
            if isinstance(shock_cfg, np.ndarray) and shock_cfg.size > 0:
                shock_cfg = shock_cfg.ravel()[0]
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
        shock_cfg = shock_configs[i]
        if isinstance(shock_cfg, np.ndarray) and shock_cfg.size > 0:
            shock_cfg = shock_cfg.ravel()[0]
        if not isinstance(shock_cfg, dict):
            shock_cfg = {}
        shock_result = by_shock[i]

        shock_value = shock_cfg.get("value", shock_cfg.get("Value", 0))
        shock_label = shock_cfg.get("label", shock_cfg.get("Label", f"shock_{i}"))
        shock_desc = shock_cfg.get("description", shock_cfg.get("Description", ""))

        key = _parse_shock_label_to_key(shock_label, shock_value)
        print(f"    Processing: {shock_label} ({shock_desc}) → key={key}")

        if isinstance(shock_result, np.ndarray) and shock_result.size > 0:
            shock_result = shock_result.ravel()[0]
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

    if peak_fo is not None:
        processed["peak_values_loglin"] = np.array(peak_fo)[shock_idx, :]
    if peak_pf is not None:
        processed["peak_values_determ"] = np.array(peak_pf)[shock_idx, :]
    if hl_fo is not None:
        processed["half_lives_loglin"] = np.array(hl_fo)[shock_idx, :]
    if hl_pf is not None:
        processed["half_lives_determ"] = np.array(hl_pf)[shock_idx, :]
    if amp_abs is not None:
        processed["amplifications"] = np.array(amp_abs)[shock_idx, :]

    irfs_by_shock = md_irs.get("irfs", [])
    if not isinstance(irfs_by_shock, (list, np.ndarray)) or shock_idx >= len(irfs_by_shock):
        return processed

    irfs = irfs_by_shock[shock_idx]
    if not isinstance(irfs, (list, np.ndarray)):
        irfs = [irfs]

    for irf_data in irfs:
        if irf_data is None:
            continue
        if isinstance(irf_data, np.ndarray) and irf_data.size > 0:
            irf_data = irf_data.ravel()[0]
        if not isinstance(irf_data, dict):
            continue

        sector_idx = irf_data.get("sector_idx", 0)
        if isinstance(sector_idx, np.ndarray):
            sector_idx = int(sector_idx.item())
        sector_idx = int(sector_idx) - 1

        irs_loglin = irf_data.get("first_order", irf_data.get("IRSFirstOrder"))
        irs_determ = irf_data.get("perfect_foresight", irf_data.get("IRSPerfectForesight"))
        if irs_determ is None:
            irs_determ = irf_data.get("IRSPF", irf_data.get("IRSDeterm"))

        if irs_loglin is None:
            continue

        processed["sectors"][sector_idx] = {
            "IRSLoglin": np.array(irs_loglin),
            "IRSDeterm": np.array(irs_determ) if irs_determ is not None else None,
        }

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
    irfs = shock_result.get("IRFs", [])
    if not isinstance(irfs, (list, np.ndarray)):
        irfs = [irfs]

    for irf_data in irfs:
        if irf_data is None:
            continue
        if isinstance(irf_data, np.ndarray) and irf_data.size > 0:
            irf_data = irf_data.ravel()[0]
        if not isinstance(irf_data, dict):
            continue

        sector_idx = irf_data.get("sector_idx", 0)
        if isinstance(sector_idx, np.ndarray):
            sector_idx = int(sector_idx.item())
        sector_idx = int(sector_idx) - 1  # MATLAB 1-based -> Python 0-based

        irs_loglin = irf_data.get("IRSFirstOrder")
        if irs_loglin is None:
            irs_loglin = irf_data.get("IRSLoglin")
        irs_determ = irf_data.get("IRSPerfectForesight")
        if irs_determ is None:
            irs_determ = irf_data.get("IRSPF")
        if irs_determ is None:
            irs_determ = irf_data.get("IRSDeterm")

        if irs_loglin is not None:
            arr_loglin = np.array(irs_loglin)
            arr_determ = np.array(irs_determ) if irs_determ is not None else None
            processed["sectors"][sector_idx] = {
                "IRSLoglin": arr_loglin,
                "IRSDeterm": arr_determ,
            }

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

    for key, data in ir_data.items():
        if "sectors" not in data or sector_idx not in data["sectors"]:
            continue

        sector_data = data["sectors"][sector_idx]

        loglin_raw = sector_data["IRSLoglin"][variable_idx, :]
        if skip_initial:
            loglin = loglin_raw[1 : max_periods + 1]
        else:
            loglin = loglin_raw[:max_periods]

        irs_determ = sector_data.get("IRSDeterm")
        if irs_determ is not None:
            determ_raw = irs_determ[variable_idx, :]
            if skip_initial:
                determ = determ_raw[1 : max_periods + 1]
            else:
                determ = determ_raw[:max_periods]
        else:
            determ = None

        result[key] = {
            "loglin": loglin,
            "determ": determ,
        }

    return result


# New format (Dec 2025+), aligned with process_ir_data.m:
#   Row  1: A_ir
#   Row  2: C_exp_ir
#   Row  3: I_exp_ir
#   Row  4: Cj_ir
#   Row  5: Pj_ir
#   Row  6: Ioutj_ir
#   Row  7: Moutj_ir
#   Row  8: Lj_ir
#   Row  9: Ij_ir
#   Row 10: Mj_ir
#   Row 11: Yj_ir
#   Row 12: Qj_ir
#   Row 13: A_client_ir
#   Row 14: Cj_client_ir
#   Row 15: Pj_client_ir
#   Row 16: Ioutj_client_ir
#   Row 17: Moutj_client_ir
#   Row 18: Lj_client_ir
#   Row 19: Ij_client_ir
#   Row 20: Mj_client_ir
#   Row 21: Yj_client_ir
#   Row 22: Qj_client_ir
#   Row 23: Kj_ir
#   Row 24: GDP_exp_ir
#   Row 25: Pmj_client_ir
#   Row 26: gammaij_client_ir
#   Row 27: C_utility_ir
NEW_FORMAT_VARIABLE_INDICES = {
    "A": 0,  # TFP level (shocked sector)
    "Cexp": 1,  # Aggregate consumption expenditure
    "Iexp": 2,  # Aggregate investment expenditure
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
    "GDPexp": 23,  # Aggregate GDP expenditure
    "Pmj_client": 24,  # Client intermediate price
    "gammaij_client": 25,  # Client expenditure share deviation
    "Cutil": 26,  # Utility aggregate consumption
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
    "Agg. Consumption (Utility)": "Cutil",
    "Agg. Output": "GDPexp",
    "Agg. GDP": "GDPexp",
    "Agg. Investment": "Iexp",
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
