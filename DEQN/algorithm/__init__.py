"""
Backward compatibility layer for DEQN algorithm components.

This module re-exports the algorithm functions to maintain compatibility
with existing code while users migrate to the new core interfaces.
"""

# Re-export the core algorithm functions for backward compatibility
from .epoch_train import create_epoch_train_fn, create_fast_epoch_train_fn
from .eval import create_eval_fn
from .loss import (
    create_batch_loss_fn,
)
from .simulation import (
    create_episode_simul_fn,
)

# from .steady_state_solver import (
#     solve_steady_state,
#     SteadyStateSolution,
# )

# Re-export everything that might be imported by existing code
__all__ = [
    # Training
    "create_epoch_train_fn",
    "create_fast_epoch_train_fn",
    # Evaluation
    "create_eval_fn",
    # Loss functions
    "create_batch_loss_fn",
    # Simulation
    "create_episode_simul_fn",
    # Steady state
    # "solve_steady_state",
    # "SteadyStateSolution",
]
