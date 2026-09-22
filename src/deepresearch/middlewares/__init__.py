"""Lifecycle hooks that add research behavior around the core agent loop."""

from deepresearch.middlewares.context_governance import ContextGovernanceMiddleware
from deepresearch.middlewares.context_externalization import (
    ContextExternalizationMiddleware,
)
from deepresearch.middlewares.context_compaction import ContextCompactionMiddleware
from deepresearch.middlewares.evidence import EvidenceMiddleware
from deepresearch.middlewares.plan_context import PlanContextMiddleware
from deepresearch.middlewares.reflection import ReflectionMiddleware
from deepresearch.middlewares.sequential_tools import SequentialToolCallMiddleware
from deepresearch.middlewares.tagged_context import TaggedContextMiddleware

__all__ = [
    "ContextGovernanceMiddleware",
    "ContextExternalizationMiddleware",
    "ContextCompactionMiddleware",
    "EvidenceMiddleware",
    "PlanContextMiddleware",
    "ReflectionMiddleware",
    "SequentialToolCallMiddleware",
    "TaggedContextMiddleware",
]
