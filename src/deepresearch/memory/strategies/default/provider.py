"""Default composition of Markdown storage, BM25 retrieval, and management."""

from __future__ import annotations

from typing import Any

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryTrace, MemoryType
from deepresearch.memory.store import MemoryStore
from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy
from deepresearch.memory.strategies.default.manager import DefaultMemoryManager
from deepresearch.memory.strategies.default.retriever import HybridRetriever
from deepresearch.memory.strategies.default.store import MarkdownFileStore
from deepresearch.memory.types import MemoryFilter


class _IndexingMemoryStore:
    """Notify the derived BM25 index after truth-store mutations."""

    def __init__(self, store: MemoryStore, retriever: HybridRetriever) -> None:
        self._store = store
        self._retriever = retriever

    def add(self, trace: MemoryTrace) -> None:
        self._store.add(trace)
        self._retriever.on_trace_added(trace)

    def get(self, trace_id: str) -> MemoryTrace | None:
        return self._store.get(trace_id)

    def update(self, trace: MemoryTrace) -> None:
        self._store.update(trace)
        self._retriever.on_trace_updated(trace)

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        self._store.batch_update(traces)
        for trace in traces:
            self._retriever.on_trace_updated(trace)

    def remove(self, trace_id: str) -> None:
        self._store.remove(trace_id)
        self._retriever.on_trace_removed(trace_id)

    def list_all(self) -> list[MemoryTrace]:
        return self._store.list_all()

    def list_by_type(
        self,
        type: MemoryType,
        *,
        namespace: str | None = None,
    ) -> list[MemoryTrace]:
        return self._store.list_by_type(type, namespace=namespace)

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]:
        return self._store.list_by_filter(filter)


class DefaultMemoryProvider:
    """Expose one coherent store/retriever/manager component graph."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: HybridRetriever,
        manager: DefaultMemoryManager,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._manager = manager

    def store(self) -> MemoryStore:
        return self._store

    def retriever(self) -> HybridRetriever:
        return self._retriever

    def manager(self) -> DefaultMemoryManager:
        return self._manager


def build_default_provider(
    config: MemoryConfig,
    *,
    now_provider: Any = None,
) -> DefaultMemoryProvider:
    """Build defaults from one configuration without hidden app imports."""

    kwargs = {"now_provider": now_provider} if now_provider is not None else {}
    truth_store = MarkdownFileStore(config.storage_path, **kwargs)
    decay = EbbinghausDecayPolicy(**kwargs)
    forget = CompositeForgetPolicy(
        decay,
        strength_threshold=config.forget_strength_threshold,
        ttl_hours=config.forget_ttl_hours,
        **kwargs,
    )
    retriever = HybridRetriever(truth_store, decay, forget, **kwargs)
    indexing_store = _IndexingMemoryStore(truth_store, retriever)
    manager = DefaultMemoryManager(indexing_store, **kwargs)
    return DefaultMemoryProvider(indexing_store, retriever, manager)
