"""Trainers for a shared economic model."""

from trainers.evaluate import evaluate, format_metrics
from trainers.figures import report
from trainers.train import train

__all__ = ["evaluate", "format_metrics", "report", "train"]
