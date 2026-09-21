"""Lifecycle hooks that add research behavior around the core agent loop."""

from deepresearch.middlewares.evidence import EvidenceMiddleware
from deepresearch.middlewares.plan_context import PlanContextMiddleware

__all__ = ["EvidenceMiddleware", "PlanContextMiddleware"]
