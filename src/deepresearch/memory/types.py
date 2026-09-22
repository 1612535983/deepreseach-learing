"""Queries, results, and checkpoint-safe runtime memory state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from deepresearch.memory.schema import MemoryTrace, MemoryType


@dataclass(frozen=True)
class MemoryQuery:
    """A namespace-scoped request for relevant memory traces."""

    text: str
    namespace: str = "default"
    top_k: int = 5
    type_filter: MemoryType | None = None
    min_strength: float = 0.0


@dataclass(frozen=True)
class MemoryFilter:
    """Coarse fields a store can use without ranking full-text relevance."""

    namespace: str | None = None
    type: MemoryType | None = None
    max_age_hours: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    include_forgotten: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked memory result and its score components."""

    trace: MemoryTrace
    similarity: float
    strength: float
    score: float

    @classmethod
    def compute_score(
        cls,
        trace: MemoryTrace,
        similarity: float,
        strength: float,
        *,
        similarity_weight: float = 0.7,
    ) -> "RetrievalResult":
        bounded_similarity = max(0.0, min(1.0, similarity))
        bounded_strength = max(0.0, min(1.0, strength))
        weight = max(0.0, min(1.0, similarity_weight))
        score = bounded_similarity * weight + bounded_strength * (1.0 - weight)
        return cls(
            trace=trace,
            similarity=bounded_similarity,
            strength=bounded_strength,
            score=score,
        )


class MemoryRecallRef(TypedDict):
    """Small State reference to a trace recalled for the current query."""

    id: str
    score: float
    strength: float


class MemoryRuntimeState(TypedDict):
    """Checkpointed runtime metrics; complete memory bodies remain in the store."""

    namespace: str
    last_query_hash: str | None
    recalled: list[MemoryRecallRef]
    recall_count: int
    injected_tokens: int
    pending_write_count: int
    last_error: str | None
    last_processed_run: NotRequired[str | None]
