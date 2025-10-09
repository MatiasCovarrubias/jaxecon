"""
Neural networks module for DEQN.

This module contains neural network implementations used by the DEQN algorithm.
"""

from .neural_nets import (
    NeuralNet,
    NeuralNet_dropout,
    NeuralNet_freezed_layers,
    create_neural_net_builder,
)
from .with_loglinear_baseline import (
    NeuralNet as NeuralNet_loglinear,
)

# from .with_loglinear_baseline import (
#     create_neural_net_loglinear_builder,
# )

__all__ = [
    "NeuralNet",
    "NeuralNet_freezed_layers",
    "NeuralNet_dropout",
    "create_neural_net_builder",
    "NeuralNet_loglinear",
    "create_neural_net_loglinear_builder",
]
