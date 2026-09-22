"""Business-operation contract for long-term memory mutation."""

from __future__ import annotations

from typing import Any, Protocol

from deepresearch.memory.schema import MemoryTrace, MemoryType


class MemoryManager(Protocol):
    def encode(
        self,
        content: str,
        type: MemoryType,
        *,
        namespace: str = "default",
        importance: float = 0.5,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryTrace: ...

    def associate(
        self,
        trace_id_a: str,
        trace_id_b: str,
        *,
        strength: float = 0.5,
        type: str = "related",
    ) -> None: ...

    def consolidate(self, trace_ids: list[str], merged_content: str) -> MemoryTrace: ...

    def reconsolidate(self, trace_id: str, new_content: str) -> MemoryTrace: ...

    def forget(self, trace_id: str, *, reason: str) -> MemoryTrace: ...
