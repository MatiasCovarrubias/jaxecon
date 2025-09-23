"""
Backward compatibility layer for DEQN algorithm components.

This module re-exports the algorithm functions to maintain compatibility
with existing code while users migrate to the new core interfaces.
"""

# Re-export the core algorithm functions for backward compatibility
from .epoch_train import create_epoch_train_fn
from .eval import create_eval_fn
from .loss import (
    create_batch_loss_fn,
    create_batch_loss_fn_flexibleproxy,
    create_batch_loss_fn_pretrain,
    create_batch_loss_fn_proxied,
    create_batch_loss_fn_simple_pretrain,
)
from .simulation import (
    create_episode_simul_fn,
    create_episode_simul_fn_proxied,
)

# Re-export everything that might be imported by existing code
__all__ = [
    # Training
    "create_epoch_train_fn",
    # Evaluation
    "create_eval_fn",
    # Loss functions
    "create_batch_loss_fn",
    "create_batch_loss_fn_pretrain",
    "create_batch_loss_fn_proxied",
    "create_batch_loss_fn_flexibleproxy",
    "create_batch_loss_fn_simple_pretrain",
    # Simulation
    "create_episode_simul_fn",
    "create_episode_simul_fn_proxied",
]
