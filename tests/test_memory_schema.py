from dataclasses import FrozenInstanceError

import pytest

from deepresearch.memory.schema import MemoryTrace, MemoryType, OperationLog
from deepresearch.memory.types import RetrievalResult


def _trace() -> MemoryTrace:
    return MemoryTrace(
        id="trace-1",
        namespace="default",
        content="用户偏好先阅读设计。",
        type=MemoryType.SEMANTIC,
        strength=0.8,
    )


def test_memory_trace_is_immutable_and_strengthening_returns_copy() -> None:
    trace = _trace()

    with pytest.raises(FrozenInstanceError):
        trace.strength = 0.2  # type: ignore[misc]

    strengthened = trace.with_strength(0.9, 123.0)
    assert trace.access_count == 0
    assert strengthened.strength == 0.9
    assert strengthened.access_count == 1
    assert strengthened.last_accessed == 123.0


def test_operation_log_retains_only_latest_twenty_entries() -> None:
    trace = _trace()
    for index in range(25):
        trace = trace.with_operation(OperationLog(float(index), f"op-{index}"))

    assert len(trace.operation_log) == 20
    assert trace.operation_log[0].operation == "op-5"
    assert trace.operation_log[-1].operation == "op-24"


def test_retrieval_result_combines_similarity_and_strength_once() -> None:
    result = RetrievalResult.compute_score(_trace(), 0.8, 0.5)

    assert result.similarity == 0.8
    assert result.strength == 0.5
    assert result.score == pytest.approx(0.71)
