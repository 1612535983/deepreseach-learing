"""Lifecycle hooks that add research behavior around the core agent loop."""

from deepresearch.middlewares.evidence import EvidenceMiddleware
from deepresearch.middlewares.plan_context import PlanContextMiddleware
from deepresearch.middlewares.reflection import ReflectionMiddleware
from deepresearch.middlewares.sequential_tools import SequentialToolCallMiddleware

__all__ = [
    "EvidenceMiddleware",
    "PlanContextMiddleware",
    "ReflectionMiddleware",
    "SequentialToolCallMiddleware",
]
