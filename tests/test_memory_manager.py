from __future__ import annotations

import pytest

from deepresearch.memory.exceptions import MemoryNotFoundError
from deepresearch.memory.schema import MemoryTrace, MemoryType
from deepresearch.memory.strategies.default.manager import (
    DefaultMemoryManager,
    set_memory_actor,
)
from deepresearch.memory.types import MemoryFilter


class MemoryStoreStub:
    def __init__(self) -> None:
        self.items: dict[str, MemoryTrace] = {}

    def add(self, trace: MemoryTrace) -> None:
        assert trace.id not in self.items
        self.items[trace.id] = trace

    def get(self, trace_id: str) -> MemoryTrace | None:
        return self.items.get(trace_id)

    def update(self, trace: MemoryTrace) -> None:
        if trace.id not in self.items:
            raise MemoryNotFoundError(trace.id)
        self.items[trace.id] = trace

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        for trace in traces:
            self.update(trace)

    def remove(self, trace_id: str) -> None:
        self.items.pop(trace_id, None)

    def list_all(self) -> list[MemoryTrace]:
        return list(self.items.values())

    def list_by_type(
        self,
        type: MemoryType,
        *,
        namespace: str | None = None,
    ) -> list[MemoryTrace]:
        return [
            trace
            for trace in self.items.values()
            if trace.type == type and (namespace is None or trace.namespace == namespace)
        ]

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]:
        return self.list_all()


def _manager() -> tuple[DefaultMemoryManager, MemoryStoreStub]:
    store = MemoryStoreStub()
    return DefaultMemoryManager(store, now_provider=lambda: 1_000.0), store


def test_encode_is_idempotent_and_namespace_scoped() -> None:
    manager, store = _manager()

    first = manager.encode("用户喜欢结构化讲解", MemoryType.SEMANTIC, namespace="a")
    duplicate = manager.encode("用户喜欢结构化讲解", MemoryType.SEMANTIC, namespace="a")
    other_namespace = manager.encode(
        "用户喜欢结构化讲解", MemoryType.SEMANTIC, namespace="b"
    )

    assert first is duplicate
    assert first.id != other_namespace.id
    assert len(store.items) == 2
    assert first.strength == first.base_strength == 0.8


def test_associate_updates_both_traces_and_records_actor() -> None:
    manager, store = _manager()
    first = manager.encode("first", MemoryType.EPISODIC)
    second = manager.encode("second", MemoryType.EPISODIC)

    set_memory_actor("thread-1:turn-1")
    try:
        manager.associate(first.id, second.id, strength=0.7, type="causal")
    finally:
        set_memory_actor(None)

    updated_first = store.get(first.id)
    updated_second = store.get(second.id)
    assert updated_first is not None and updated_second is not None
    assert updated_first.associations[0].target_id == second.id
    assert updated_second.associations[0].target_id == first.id
    assert updated_first.operation_log[-1].actor == "thread-1:turn-1"


def test_consolidate_creates_semantic_and_soft_deletes_sources() -> None:
    manager, store = _manager()
    first = manager.encode("event one", MemoryType.EPISODIC, importance=0.4)
    second = manager.encode("event two", MemoryType.EPISODIC, importance=0.7)

    semantic = manager.consolidate([first.id, second.id], "stable knowledge")

    assert semantic.type == MemoryType.SEMANTIC
    assert semantic.importance == pytest.approx(0.8)
    assert semantic.metadata["consolidated_from"] == [first.id, second.id]
    assert store.get(first.id).metadata["forgotten"] is True  # type: ignore[union-attr]
    assert store.get(second.id).metadata["consolidated_into"] == semantic.id  # type: ignore[union-attr]


def test_reconsolidate_preserves_id_and_records_content_diff() -> None:
    manager, store = _manager()
    trace = manager.encode("old", MemoryType.SEMANTIC)

    updated = manager.reconsolidate(trace.id, "new")

    assert updated.id == trace.id
    assert updated.content == "new"
    assert updated.operation_log[-1].operation == "reconsolidate"
    assert store.get(trace.id) == updated


def test_forget_is_auditable_soft_delete() -> None:
    manager, store = _manager()
    trace = manager.encode("temporary", MemoryType.EPISODIC)

    forgotten = manager.forget(trace.id, reason="expired")

    assert forgotten.metadata["forgotten"] is True
    assert forgotten.metadata["forget_reason"] == "expired"
    assert store.get(trace.id) is not None
