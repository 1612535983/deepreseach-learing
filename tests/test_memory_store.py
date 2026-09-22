from dataclasses import replace

import pytest

from deepresearch.memory.exceptions import MemoryConflictError, MemoryNotFoundError
from deepresearch.memory.schema import Association, MemoryTrace, MemoryType, OperationLog
from deepresearch.memory.strategies.default.store import MarkdownFileStore
from deepresearch.memory.types import MemoryFilter


def _trace(
    trace_id: str = "abc123",
    *,
    namespace: str = "default",
    type: MemoryType = MemoryType.SEMANTIC,
    created_at: float = 1_000.0,
) -> MemoryTrace:
    return MemoryTrace(
        id=trace_id,
        namespace=namespace,
        content="用户偏好先看完整设计。",
        type=type,
        strength=0.8,
        base_strength=0.8,
        decay_rate=0.02,
        last_accessed=created_at,
        importance=0.9,
        associations=(Association("def456", 0.6, "related"),),
        embedding=(0.1, 0.2),
        source="thread-1",
        created_at=created_at,
        metadata={"project": "demo"},
        operation_log=(OperationLog(created_at, "encode", "run-1", {"x": (1, 2)}),),
    )


def test_store_round_trips_complete_trace(tmp_path) -> None:
    store = MarkdownFileStore(tmp_path)
    trace = _trace()
    store.add(trace)

    reopened = MarkdownFileStore(tmp_path)

    assert reopened.get(trace.id) == trace
    assert "用户偏好" in reopened.path.read_text(encoding="utf-8")


def test_store_rejects_conflicts_and_missing_updates(tmp_path) -> None:
    store = MarkdownFileStore(tmp_path)
    trace = _trace()
    store.add(trace)

    with pytest.raises(MemoryConflictError):
        store.add(trace)
    with pytest.raises(MemoryNotFoundError):
        store.update(_trace("missing"))


def test_store_update_batch_update_and_remove_are_persistent(tmp_path) -> None:
    store = MarkdownFileStore(tmp_path)
    first = _trace("aaa111")
    second = _trace("bbb222", type=MemoryType.EPISODIC)
    store.add(first)
    store.add(second)

    store.batch_update(
        [
            replace(first, content="updated one"),
            replace(second, content="updated two"),
        ]
    )
    store.remove(first.id)
    store.remove("already-missing")

    reopened = MarkdownFileStore(tmp_path)
    assert reopened.get(first.id) is None
    assert reopened.get(second.id).content == "updated two"  # type: ignore[union-attr]


def test_store_filters_namespace_type_age_metadata_and_forgotten(tmp_path) -> None:
    store = MarkdownFileStore(tmp_path, now_provider=lambda: 10_000.0)
    store.add(_trace("aaa111", namespace="a", created_at=9_000.0))
    store.add(_trace("bbb222", namespace="b", created_at=1_000.0))
    forgotten = replace(
        _trace("ccc333", namespace="a", type=MemoryType.EPISODIC),
        metadata={"project": "demo", "forgotten": True},
    )
    store.add(forgotten)

    assert [item.id for item in store.list_by_filter(MemoryFilter(namespace="a"))] == [
        "aaa111"
    ]
    assert store.list_by_filter(MemoryFilter(namespace="b", max_age_hours=1)) == []
    assert store.list_by_filter(
        MemoryFilter(namespace="a", type=MemoryType.EPISODIC)
    ) == []
    assert store.list_by_filter(
        MemoryFilter(namespace="a", include_forgotten=True, metadata={"project": "demo"})
    ) == [_trace("aaa111", namespace="a", created_at=9_000.0), forgotten]


def test_store_skips_one_corrupt_trace_without_losing_valid_traces(tmp_path, caplog) -> None:
    store = MarkdownFileStore(tmp_path)
    valid = _trace("abc123")
    store.add(valid)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("<!-- trace: deadbeef -->\n---\nid: wrong\ntype: semantic\n---\nbroken\n")

    reopened = MarkdownFileStore(tmp_path)

    assert reopened.get(valid.id) == valid
    assert reopened.get("deadbeef") is None
    assert "Skipping corrupt memory trace" in caplog.text
