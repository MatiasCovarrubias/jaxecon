"""
Economic models module for DEQN.

This module contains implementations of various economic models that can be solved
using the DEQN algorithm.
"""

from __future__ import annotations

import importlib


def load_model_class(model_dir: str, exact_cobb_douglas: bool = False):
    model_module = "model_CD" if exact_cobb_douglas else "model"
    module = importlib.import_module(f"DEQN.econ_models.{model_dir}.{model_module}")
    return module.Model
