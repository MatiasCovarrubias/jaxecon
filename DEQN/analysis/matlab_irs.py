"""
MATLAB Impulse Response Loader Module

This module provides functions to load and process impulse response data from MATLAB .mat files.
The MATLAB files contain perfect foresight and loglinear impulse responses computed from Dynare.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio


def load_matlab_irs(
    matlab_ir_dir: str,
    shock_sizes: List[int] = [5, 10, 20],
    file_pattern: str = "AllSectors_IRS__Oct_25nonlinear_{sign}_{size}.mat",
) -> Dict[str, Any]:
    """
    Load MATLAB impulse response files for multiple shock sizes and signs.

    Args:
        matlab_ir_dir: Directory containing the MATLAB IR files
        shock_sizes: List of shock sizes in percent (e.g., [5, 10, 20])
        file_pattern: File name pattern with {sign} and {size} placeholders

    Returns:
        Dictionary with structure:
        {
            "pos_5": {"sectors": {...}, "peak_values_loglin": array, ...},
            "neg_5": {"sectors": {...}, "peak_values_loglin": array, ...},
            ...
        }
    """
    ir_data = {}

    for size in shock_sizes:
        for sign in ["pos", "neg"]:
            filename = file_pattern.format(sign=sign, size=size)
            filepath = os.path.join(matlab_ir_dir, filename)

            if not os.path.exists(filepath):
                print(f"Warning: IR file not found: {filepath}")
                continue

            try:
                mat_data = sio.loadmat(filepath, simplify_cells=True)["AllIRS"]
                key = f"{sign}_{size}"
                ir_data[key] = _process_matlab_ir_data(mat_data)
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return ir_data


def _process_matlab_ir_data(mat_data: Dict) -> Dict[str, Any]:
    """
    Process raw MATLAB IR data into a structured format.

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
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract IRs for a specific sector and variable across all shock sizes/signs.

    Args:
        ir_data: Dictionary from load_matlab_irs
        sector_idx: Sector index (0-based)
        variable_idx: Variable index within the IR array (0-26 typically)
        max_periods: Maximum number of periods to return

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

        loglin = sector_data["IRSLoglin"][variable_idx, :max_periods]
        determ = sector_data["IRSDeterm"][variable_idx, :max_periods]

        result[key] = {
            "loglin": loglin,
            "determ": determ,
        }

    return result


# MATLAB IR variable indices mapping
# These indices correspond to the rows in IRSLoglin and IRSDeterm
# The exact mapping depends on how the Dynare model exports variables.
#
# To determine the correct mapping for your model:
# 1. Check the Dynare .mod file for variable ordering in the stoch_simul command
# 2. Or use inspect_matlab_ir_structure() to examine the data patterns
# 3. Variables that start at the shock value (e.g., 0.8 for -20%) are the shocked variable
#
# Default mapping based on RbcProdNet_nonlinear model inspection:
# (Verified using inspect_matlab_ir_structure with neg_20 shock)
MATLAB_IR_VARIABLE_INDICES = {
    # Sectoral variables (for the shocked sector)
    "K": 0,  # Capital - starts at shocked value (0.8 for -20% shock)
    "C": 2,  # Sectoral consumption
    "L": 3,  # Sectoral labor
    "Pk": 4,  # Capital price
    "Pm": 5,  # Intermediate price
    "M": 6,  # Intermediate inputs
    "Mout": 7,  # Intermediate outputs
    "Inv": 8,  # Investment
    "P": 10,  # Output price
    "Y": 11,  # Sectoral output
    "Q": 12,  # Quantity
    # Row 13 is constant (normalization)
    # Aggregate variables
    "Cagg": 14,  # Aggregate consumption (peak ~-0.11 for -20% shock)
    "Kagg": 15,  # Aggregate capital (peak ~+0.21 for -20% shock)
    "Yagg": 16,  # Aggregate output (peak ~-0.12 for -20% shock)
    "Magg": 17,  # Aggregate intermediates
    "Iagg": 18,  # Aggregate investment (small, ~0.004)
    "Wage": 19,  # Wage
    "Lagg": 20,  # Aggregate labor (peak ~-0.08 for -20% shock)
    "R": 21,  # Interest rate
    "GDP": 22,  # GDP proxy
    "Utility": 23,  # Utility welfare measure
    "Lambda": 24,  # Lagrange multiplier
    "Mu_k": 25,  # Capital-related multiplier (large response)
    "Mu_c": 26,  # Consumption-related multiplier (large response)
}

# Mapping from analysis variable names (used in JAX) to MATLAB variable names
# Adjust these mappings based on your specific Dynare model output
ANALYSIS_TO_MATLAB_MAPPING = {
    "Agg. Consumption": "Cagg",
    "Agg. Labor": "Lagg",
    "Agg. Capital": "Kagg",
    "Agg. Output": "Yagg",
    "Agg. Intermediates": "Magg",
    "Agg. Investment": "Iagg",
    "Utility": "Utility",
}


def set_matlab_variable_mapping(mapping: Dict[str, int]) -> None:
    """
    Update the MATLAB variable index mapping.

    Use this to customize the mapping based on your specific Dynare model output.

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
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Get MATLAB IRs for a given analysis variable name.

    Args:
        ir_data: Dictionary from load_matlab_irs
        sector_idx: Sector index (0-based)
        analysis_var_name: Name of analysis variable (e.g., "Agg. Consumption")
        max_periods: Maximum number of periods to return

    Returns:
        Dictionary with IRs for each shock size/sign, or None if variable not found
    """
    if analysis_var_name not in ANALYSIS_TO_MATLAB_MAPPING:
        return None

    matlab_var_name = ANALYSIS_TO_MATLAB_MAPPING[analysis_var_name]

    if matlab_var_name not in MATLAB_IR_VARIABLE_INDICES:
        return None

    var_idx = MATLAB_IR_VARIABLE_INDICES[matlab_var_name]

    return get_sector_irs(ir_data, sector_idx, var_idx, max_periods)


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
    matlab_ir_dir: str,
    sector_idx: int = 0,
    shock_config: str = "neg_20",
    file_pattern: str = "AllSectors_IRS__Oct_25nonlinear_{sign}_{size}.mat",
) -> None:
    """
    Utility function to inspect MATLAB IR data structure and help identify variable mappings.

    This function prints summary statistics for each row in the IR arrays to help
    identify which row corresponds to which variable.

    Args:
        matlab_ir_dir: Directory containing MATLAB IR files
        sector_idx: Sector to inspect (0-based)
        shock_config: Shock configuration to inspect (e.g., "neg_20", "pos_10")
        file_pattern: File name pattern
    """
    sign, size = shock_config.split("_")
    filename = file_pattern.format(sign=sign, size=size)
    filepath = os.path.join(matlab_ir_dir, filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    mat_data = sio.loadmat(filepath, simplify_cells=True)["AllIRS"]
    sector_key = f"Sector_{sector_idx + 1}"

    if sector_key not in mat_data:
        print(f"Sector {sector_idx + 1} not found in data")
        return

    sector_data = mat_data[sector_key]
    loglin = sector_data["IRSLoglin"]
    determ = sector_data["IRSDeterm"]

    print(f"\n=== MATLAB IR Structure for {sector_key} ({shock_config}) ===")
    print(f"IRSLoglin shape: {loglin.shape} (rows=variables, cols=time periods)")
    print(f"IRSDeterm shape: {determ.shape}")

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
            notes += f"Large swing "

        print(f"{i:<5} {initial:<12.4f} {peak:<12.4f} {final:<12.4f} {monotonic:<10} {notes}")

    print("\nSuggested variable identification based on patterns:")
    print("- Row with initial value ≈ shock size: likely the shocked variable (K or A)")
    print("- Row with all 1s: likely a constant or normalization")
    print("- Large amplitude rows: typically prices or quantities")
    print("- Small amplitude rows: typically aggregate welfare measures")

