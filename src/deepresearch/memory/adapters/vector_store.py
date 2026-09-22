"""Optional vector index contract; Markdown remains the truth source."""

from __future__ import annotations

from typing import Protocol

from deepresearch.memory.schema import MemoryTrace


class VectorStore(Protocol):
    def upsert(self, trace: MemoryTrace) -> None: ...

    def remove(self, trace_id: str) -> None: ...

    def search(self, text: str, *, namespace: str, top_k: int) -> list[tuple[str, float]]: ...
