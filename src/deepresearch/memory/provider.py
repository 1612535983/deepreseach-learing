"""Composition contract for the memory subsystem."""

from __future__ import annotations

from typing import Protocol

from deepresearch.memory.manager import MemoryManager
from deepresearch.memory.retriever import Retriever
from deepresearch.memory.store import MemoryStore


class MemoryProvider(Protocol):
    def store(self) -> MemoryStore: ...

    def retriever(self) -> Retriever: ...

    def manager(self) -> MemoryManager: ...
