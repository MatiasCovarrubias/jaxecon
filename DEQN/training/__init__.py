"""
Training utilities for DEQN models.

This module provides high-level experiment management and training orchestration.
"""

from .runner import generate_experiment_grid, run_experiment

__all__ = ["run_experiment", "generate_experiment_grid"]
