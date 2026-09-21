"""In-house log-linear policy for APG (LQ approximation + Riccati)."""

from .baseline import build_actor, prepare_loglinear
from .design import (
    LQDesign,
    apply_lq_design,
    inject_lq_design,
    knobs_from_config,
    lq_design,
    needs_lq_objects,
)
from .solve import (
    LogLinearSolution,
    linearize_environment,
    print_loglinear_solution,
    solve_loglinear,
    solve_riccati,
    stationary_sd,
)

__all__ = [
    "LQDesign",
    "LogLinearSolution",
    "apply_lq_design",
    "build_actor",
    "inject_lq_design",
    "knobs_from_config",
    "lq_design",
    "needs_lq_objects",
    "linearize_environment",
    "prepare_loglinear",
    "print_loglinear_solution",
    "solve_loglinear",
    "solve_riccati",
    "stationary_sd",
]
