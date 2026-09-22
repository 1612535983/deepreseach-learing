"""Persistence contract for long-term memory traces."""

from __future__ import annotations

from typing import Protocol

from deepresearch.memory.schema import MemoryTrace, MemoryType
from deepresearch.memory.types import MemoryFilter


class MemoryStore(Protocol):
    """Storage operations required by memory policies and retrieval."""

    def add(self, trace: MemoryTrace) -> None: ...

    def get(self, trace_id: str) -> MemoryTrace | None: ...

    def update(self, trace: MemoryTrace) -> None: ...

    def batch_update(self, traces: list[MemoryTrace]) -> None: ...

    def remove(self, trace_id: str) -> None: ...

    def list_all(self) -> list[MemoryTrace]: ...

    def list_by_type(
        self,
        type: MemoryType,
        *,
        namespace: str | None = None,
    ) -> list[MemoryTrace]: ...

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]: ...
