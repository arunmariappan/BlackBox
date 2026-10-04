"""Deterministic run metrics."""

from blackbox.metrics import generic, opsdesk, paperpilot  # noqa: F401  (registers the modules)
from blackbox.metrics.framework import MODULES, MetricInput, MetricValue, compute, compute_and_store

__all__ = ["MODULES", "MetricInput", "MetricValue", "compute", "compute_and_store"]
