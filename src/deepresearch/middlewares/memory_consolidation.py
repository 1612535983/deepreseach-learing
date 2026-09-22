"""Submit one memory extraction task after a complete Agent run."""

from __future__ import annotations

import hashlib
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.worker import MemoryTask, MemoryWorker
from deepresearch.state import ResearchState


class MemoryConsolidationMiddleware(AgentMiddleware):
    """Bridge completed canonical State to the asynchronous memory worker."""

    state_schema = ResearchState

    def __init__(
        self,
        worker: MemoryWorker,
        config: MemoryConfig,
        *,
        max_messages: int = 20,
    ) -> None:
        self._worker = worker
        self._config = config
        self._max_messages = max(1, max_messages)

    @override
    def after_agent(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        if not self._config.enable_extract:
            return None
        messages = tuple(state.get("messages", [])[-self._max_messages :])
        if not messages:
            return None
        memory_state = state.get("memory") or {}
        namespace = str(memory_state.get("namespace") or self._config.namespace or "default")
        thread_id = self._thread_id(state, runtime)
        final_report = state.get("final_report")
        fingerprint = "\n".join(
            [
                namespace,
                thread_id,
                str(final_report or ""),
                *(f"{message.type}:{message.id}:{message.content}" for message in messages),
            ]
        )
        task_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        if memory_state.get("last_processed_run") == task_id:
            return None
        accepted = self._worker.submit(
            MemoryTask(
                task_id=task_id,
                namespace=namespace,
                thread_id=thread_id,
                messages=messages,
                final_report=final_report,
            )
        )
        return {
            "memory": {
                "last_processed_run": task_id,
                "pending_write_count": self._worker.pending_count,
                "last_error": None if accepted else self._worker.last_error,
            }
        }

    @override
    async def aafter_agent(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    @staticmethod
    def _thread_id(state: ResearchState, runtime: Runtime) -> str:
        try:
            config = getattr(runtime, "config", None) or {}
            configurable = config.get("configurable", {})
            value = configurable.get("thread_id")
            if value:
                return str(value)
        except Exception:
            pass
        question = state.get("research_question") or "uncheckpointed"
        return hashlib.sha256(str(question).encode("utf-8")).hexdigest()[:12]
