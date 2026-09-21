"""Helpers for thread-scoped LangGraph checkpoint configuration."""

from __future__ import annotations

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


_DEFAULT_IN_MEMORY_CHECKPOINTER = InMemorySaver()


def create_in_memory_checkpointer() -> InMemorySaver:
    """Create an isolated saver whose data lasts for the current process only."""

    return InMemorySaver()


def get_default_in_memory_checkpointer() -> InMemorySaver:
    """Return the process-scoped saver used by the default real-agent entry points."""

    return _DEFAULT_IN_MEMORY_CHECKPOINTER


def generate_thread_id() -> str:
    """Generate a readable identifier for one research execution thread."""

    return f"research-{uuid4().hex[:12]}"


def normalize_thread_id(thread_id: str) -> str:
    """Validate a caller-supplied thread identifier."""

    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id 不能为空。")
    if len(normalized) > 200:
        raise ValueError("thread_id 不能超过 200 个字符。")
    if any(character in normalized for character in ("\x00", "\n", "\r")):
        raise ValueError("thread_id 不能包含换行符或空字符。")
    return normalized


def build_thread_config(thread_id: str) -> RunnableConfig:
    """Build the runtime config LangGraph passes to the Checkpointer."""

    return {
        "configurable": {
            "thread_id": normalize_thread_id(thread_id),
        }
    }


def resolve_checkpoint_run(
    checkpointer: BaseCheckpointSaver | None,
    thread_id: str | None,
) -> tuple[str | None, RunnableConfig | None]:
    """Resolve a matching thread ID and config for one graph execution."""

    if checkpointer is None:
        if thread_id is not None:
            raise ValueError("提供 thread_id 时必须同时提供 checkpointer。")
        return None, None

    resolved_thread_id = (
        normalize_thread_id(thread_id) if thread_id is not None else generate_thread_id()
    )
    return resolved_thread_id, build_thread_config(resolved_thread_id)
