"""Retrieval contract for ranking relevant long-term memories."""

from __future__ import annotations

from typing import Protocol

from deepresearch.memory.types import MemoryQuery, RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]: ...
