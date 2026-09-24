"""Create persistent checkpoint savers and address research threads."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


DEFAULT_CHECKPOINT_DB = Path(".deepresearch/checkpoints.sqlite")


def create_in_memory_checkpointer() -> InMemorySaver:
    """Create an isolated saver for unit tests and short-lived callers."""

    return InMemorySaver()


def _prepare_database_path(path: str | Path) -> Path:
    """Create the database parent directory and reject directory targets."""

    database_path = Path(path).expanduser()
    if database_path.exists() and database_path.is_dir():
        raise ValueError(f"Checkpoint 数据库路径不能是目录：{database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return database_path


@contextmanager
def open_sqlite_checkpointer(
    path: str | Path = DEFAULT_CHECKPOINT_DB,
) -> Iterator[SqliteSaver]:
    """Open a disk-backed saver for the duration of a synchronous graph run."""

    database_path = _prepare_database_path(path)
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        yield checkpointer


@asynccontextmanager
async def open_async_sqlite_checkpointer(
    path: str | Path = DEFAULT_CHECKPOINT_DB,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a disk-backed saver for the duration of an asynchronous graph run."""

    database_path = _prepare_database_path(path)
    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        yield checkpointer


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


def get_checkpoint_tuple(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> CheckpointTuple:
    """Return the latest checkpoint or raise a user-facing missing-thread error."""

    normalized_thread_id = normalize_thread_id(thread_id)
    checkpoint = checkpointer.get_tuple(build_thread_config(normalized_thread_id))
    if checkpoint is None:
        raise ValueError(f"找不到任务：{normalized_thread_id}")
    return checkpoint


async def aget_checkpoint_tuple(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> CheckpointTuple:
    """Asynchronously return the latest checkpoint for one thread."""

    normalized_thread_id = normalize_thread_id(thread_id)
    checkpoint = await checkpointer.aget_tuple(
        build_thread_config(normalized_thread_id)
    )
    if checkpoint is None:
        raise ValueError(f"找不到任务：{normalized_thread_id}")
    return checkpoint


def get_checkpoint_state(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> dict[str, Any]:
    """Read the latest State values saved for one thread."""

    checkpoint = get_checkpoint_tuple(checkpointer, thread_id)
    values = checkpoint.checkpoint.get("channel_values", {})
    if not isinstance(values, Mapping):
        raise RuntimeError(f"任务 {thread_id} 的 Checkpoint State 格式无效。")
    return dict(values)


async def aget_checkpoint_state(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> dict[str, Any]:
    """Asynchronously read the latest State values for one thread."""

    checkpoint = await aget_checkpoint_tuple(checkpointer, thread_id)
    values = checkpoint.checkpoint.get("channel_values", {})
    if not isinstance(values, Mapping):
        raise RuntimeError(f"任务 {thread_id} 的 Checkpoint State 格式无效。")
    return dict(values)


def ensure_new_thread(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> None:
    """Prevent a new run from accidentally merging into an existing thread."""

    if checkpointer.get_tuple(build_thread_config(thread_id)) is not None:
        raise ValueError(f"任务 {thread_id} 已存在，请使用 resume 命令继续。")


async def ensure_new_thread_async(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> None:
    """Asynchronously prevent reuse of an existing thread ID."""

    if await checkpointer.aget_tuple(build_thread_config(thread_id)) is not None:
        raise ValueError(f"任务 {thread_id} 已存在，请使用 resume 命令继续。")


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
