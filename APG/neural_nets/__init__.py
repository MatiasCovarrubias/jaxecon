"""
Neural networks module for APG.
"""

from DEQN.neural_nets.neural_nets import PolicyNet

from .neural_nets import ActorCritic, PolicyNetLoglinear

__all__ = [
    "PolicyNet",
    "PolicyNetLoglinear",
    "ActorCritic",
]
