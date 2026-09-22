from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.memory.config import MemoryConfig
from deepresearch.middlewares.memory_consolidation import MemoryConsolidationMiddleware
from deepresearch.state import create_initial_state, merge_memory_runtime


class WorkerStub:
    def __init__(self) -> None:
        self.tasks = []
        self.pending_count = 0
        self.last_error = None

    def submit(self, task):  # noqa: ANN001, ANN201
        self.tasks.append(task)
        self.pending_count += 1
        return True


def test_after_agent_submits_one_bounded_completed_run() -> None:
    worker = WorkerStub()
    config = MemoryConfig(enable_extract=True, namespace="project-a")
    middleware = MemoryConsolidationMiddleware(worker, config, max_messages=2)  # type: ignore[arg-type]
    state = create_initial_state("研究问题", memory_namespace="project-a")
    state["messages"].extend(
        [AIMessage(content="中间"), AIMessage(content="最终")]
    )
    state["final_report"] = "报告"

    update = middleware.after_agent(state, Runtime())

    assert update is not None
    assert len(worker.tasks) == 1
    assert len(worker.tasks[0].messages) == 2
    assert worker.tasks[0].namespace == "project-a"
    assert worker.tasks[0].final_report == "报告"
    assert update["memory"]["pending_write_count"] == 1


def test_after_agent_does_not_resubmit_processed_run() -> None:
    worker = WorkerStub()
    middleware = MemoryConsolidationMiddleware(
        worker,  # type: ignore[arg-type]
        MemoryConfig(enable_extract=True),
    )
    state = create_initial_state("研究问题")
    state["messages"].append(AIMessage(content="最终"))
    first = middleware.after_agent(state, Runtime())
    assert first is not None
    state["memory"] = merge_memory_runtime(state["memory"], first["memory"])

    assert middleware.after_agent(state, Runtime()) is None
    assert len(worker.tasks) == 1


def test_consolidation_is_disabled_by_default_and_async_matches_sync() -> None:
    worker = WorkerStub()
    state = create_initial_state("研究问题")
    disabled = MemoryConsolidationMiddleware(worker, MemoryConfig())  # type: ignore[arg-type]
    assert disabled.after_agent(state, Runtime()) is None

    enabled = MemoryConsolidationMiddleware(  # type: ignore[arg-type]
        worker,
        MemoryConfig(enable_extract=True),
    )
    update = asyncio.run(enabled.aafter_agent(state, Runtime()))
    assert update is not None
    assert worker.tasks
