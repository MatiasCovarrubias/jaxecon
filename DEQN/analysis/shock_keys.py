from typing import Any, Optional

import numpy as np


def format_shock_size_token(value: Any) -> str:
    scalar = float(np.asarray(value).item())
    rounded = round(scalar, 8)
    if float(rounded).is_integer():
        return str(int(round(rounded)))
    return f"{rounded:.8f}".rstrip("0").rstrip(".").replace(".", "_")


def parse_shock_size_token(token: str) -> Optional[float]:
    try:
        return float(str(token).replace("_", "."))
    except ValueError:
        return None


def build_shock_key(sign_prefix: str, shock_size: Any, suffix: str = "") -> str:
    return f"{sign_prefix}_{format_shock_size_token(shock_size)}{suffix}"

