"""Lifecycle hooks that add research behavior around the core agent loop."""

from deepresearch.middlewares.context_compaction import ContextCompactionMiddleware
from deepresearch.middlewares.context_externalization import (
    ContextExternalizationMiddleware,
)
from deepresearch.middlewares.context_finalization import ContextFinalizationMiddleware
from deepresearch.middlewares.context_governance import ContextGovernanceMiddleware
from deepresearch.middlewares.evidence import EvidenceMiddleware
from deepresearch.middlewares.memory_recall import MemoryRecallMiddleware
from deepresearch.middlewares.memory_consolidation import MemoryConsolidationMiddleware
from deepresearch.middlewares.plan_context import PlanContextMiddleware
from deepresearch.middlewares.reflection import ReflectionMiddleware
from deepresearch.middlewares.report_evaluation import ReportEvaluationMiddleware
from deepresearch.middlewares.sequential_tools import SequentialToolCallMiddleware
from deepresearch.middlewares.skill_selection import SkillSelectionMiddleware
from deepresearch.middlewares.skill_injection import SkillInjectionMiddleware
from deepresearch.middlewares.skill_evaluation import SkillEvaluationMiddleware
from deepresearch.middlewares.skill_metrics import SkillMetricsMiddleware
from deepresearch.middlewares.tagged_context import TaggedContextMiddleware

__all__ = [
    "ContextCompactionMiddleware",
    "ContextExternalizationMiddleware",
    "ContextFinalizationMiddleware",
    "ContextGovernanceMiddleware",
    "EvidenceMiddleware",
    "MemoryRecallMiddleware",
    "MemoryConsolidationMiddleware",
    "PlanContextMiddleware",
    "ReflectionMiddleware",
    "ReportEvaluationMiddleware",
    "SequentialToolCallMiddleware",
    "SkillSelectionMiddleware",
    "SkillInjectionMiddleware",
    "SkillEvaluationMiddleware",
    "SkillMetricsMiddleware",
    "TaggedContextMiddleware",
]
