"""Diagnostics for validating analytical policy-gradient identities."""

from .theory import (
    StatewisePolicyGradient,
    TheoryDiagnostic,
    central_finite_difference_gradient,
    explicit_forward_gradient,
    explicit_reverse_gradient,
    finite_horizon_return,
    policy_gradient_diagnostic,
    statewise_action_value_gradient,
)

__all__ = [
    "StatewisePolicyGradient",
    "TheoryDiagnostic",
    "central_finite_difference_gradient",
    "explicit_forward_gradient",
    "explicit_reverse_gradient",
    "finite_horizon_return",
    "policy_gradient_diagnostic",
    "statewise_action_value_gradient",
]
