"""Time iteration with a semi-smooth Newton local solver."""

import jax

jax.config.update("jax_enable_x64", True)

from TimeIteration.algorithm.implicit_diff import implicit_policy_fn
from TimeIteration.algorithm.solve import Solution, SolveInfo, build_grids, solve
from TimeIteration.models.protocol import GridSpec, Grids
from TimeIteration.models.rbc import Params, RbcModel, closed_form_params, default_params

__all__ = [
    "GridSpec",
    "Grids",
    "Params",
    "RbcModel",
    "Solution",
    "SolveInfo",
    "build_grids",
    "closed_form_params",
    "default_params",
    "implicit_policy_fn",
    "solve",
]
