"""Immutable domain objects used by the long-term memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """The cognitive role of one memory trace."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class Association:
    """A directed relationship to another memory trace."""

    target_id: str
    strength: float = 0.5
    type: str = "related"


@dataclass(frozen=True)
class OperationLog:
    """One auditable mutation performed on a memory trace."""

    timestamp: float
    operation: str
    actor: str | None = None
    diff: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryTrace:
    """The immutable persistence unit for one long-term memory."""

    id: str
    namespace: str
    content: str
    type: MemoryType
    strength: float = 0.0
    base_strength: float = 0.7
    decay_rate: float = 0.1
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 0.5
    associations: tuple[Association, ...] = field(default_factory=tuple)
    embedding: tuple[float, ...] | None = None
    source: str | None = None
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    operation_log: tuple[OperationLog, ...] = field(default_factory=tuple)

    def with_strength(self, new_strength: float, accessed_at: float) -> "MemoryTrace":
        """Return a strengthened access copy without mutating this trace."""

        return replace(
            self,
            strength=max(0.0, min(1.0, new_strength)),
            access_count=self.access_count + 1,
            last_accessed=accessed_at,
        )

    def with_operation(
        self,
        log: OperationLog,
        *,
        max_log: int = 20,
    ) -> "MemoryTrace":
        """Append an operation, retaining only the newest bounded audit entries."""

        if max_log < 1:
            raise ValueError("max_log 必须大于 0。")
        logs = (self.operation_log + (log,))[-max_log:]
        return replace(self, operation_log=logs)
