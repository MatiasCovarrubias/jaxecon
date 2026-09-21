"""Explicit constrained APG algorithm variants."""

from .exact_kink import (
    ExactKinkMetrics,
    create_exact_kink_epoch_train_fn,
    create_exact_kink_eval_fn,
    create_exact_kink_objectives,
    log_linear_beta,
    residual_by_slack_bin,
)
from .naive_projected import (
    NaiveProjectedMetrics,
    create_naive_projected_eval_fn,
    gate_decision,
    naive_projected_metrics_to_dict,
)
from .networks import GridMultiplierNet, MultiplierNet
from .primal_dual import (
    ConstrainedObjectiveMetrics,
    ConstrainedTrainMetrics,
    ConstrainedTrainState,
    create_constrained_epoch_train_fn,
    create_constrained_eval_fn,
    create_constrained_objectives,
    create_multiplier_warmup_fn,
)
from .run_experiment import run_constrained_experiment

__all__ = [
    "ConstrainedObjectiveMetrics",
    "ConstrainedTrainMetrics",
    "ConstrainedTrainState",
    "ExactKinkMetrics",
    "GridMultiplierNet",
    "MultiplierNet",
    "NaiveProjectedMetrics",
    "create_constrained_epoch_train_fn",
    "create_constrained_eval_fn",
    "create_constrained_objectives",
    "create_exact_kink_epoch_train_fn",
    "create_exact_kink_eval_fn",
    "create_exact_kink_objectives",
    "create_multiplier_warmup_fn",
    "create_naive_projected_eval_fn",
    "gate_decision",
    "log_linear_beta",
    "naive_projected_metrics_to_dict",
    "residual_by_slack_bin",
    "run_constrained_experiment",
]
