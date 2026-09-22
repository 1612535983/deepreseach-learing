"""Bounded background extraction and consolidation for long-term memory."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any

from langchain_core.messages import BaseMessage

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryType
from deepresearch.memory.strategies.default.manager import (
    DefaultMemoryManager,
    set_memory_actor,
)


logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Analyze the completed research conversation and extract durable memories.
Return a JSON array only. Each item must contain:
{{"content": "...", "type": "episodic|semantic|procedural", "importance": 0.0-1.0}}
Keep noteworthy user preferences, project facts, decisions, and reusable working patterns.
Do not store transient tool output, unsupported claims, secrets, or trivial conversation.

Conversation:
{conversation}

Final report:
{final_report}
"""

_CONSOLIDATE_PROMPT = """Merge these episodic memories into one stable semantic memory.
Preserve only consistent durable knowledge. Return only the merged content.

Memories:
{memories}
"""


@dataclass(frozen=True)
class MemoryTask:
    """A completed Agent run waiting for memory extraction."""

    task_id: str
    namespace: str
    thread_id: str
    messages: tuple[BaseMessage, ...]
    final_report: str | None = None


class MemoryWorker:
    """Process memory writes off the Agent latency path with bounded resources."""

    def __init__(
        self,
        manager: DefaultMemoryManager,
        llm: Any,
        config: MemoryConfig,
    ) -> None:
        self._manager = manager
        self._llm = llm
        self._config = config
        self._queue: Queue[MemoryTask] = Queue(maxsize=max(1, config.worker_queue_size))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seen_lock = threading.Lock()
        self._seen_task_ids: set[str] = set()
        self._last_error: str | None = None

    @property
    def pending_count(self) -> int:
        return int(self._queue.unfinished_tasks)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deepresearch-memory-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, task: MemoryTask) -> bool:
        """Enqueue once without blocking; return false for duplicates or saturation."""

        with self._seen_lock:
            if task.task_id in self._seen_task_ids:
                return False
            self._seen_task_ids.add(task.task_id)
        try:
            self._queue.put_nowait(task)
            return True
        except Full:
            with self._seen_lock:
                self._seen_task_ids.discard(task.task_id)
            self._last_error = "memory worker queue is full"
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until submitted tasks finish, bounded by timeout."""

        deadline = time.monotonic() + max(0.0, timeout)
        while self.pending_count:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain accepted tasks, stop the loop, and join the daemon thread."""

        self.flush(timeout)
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def process_now(self, task: MemoryTask) -> None:
        """Synchronous entry point used by deterministic tests and maintenance."""

        self._process(task)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._process(task)
            except Exception as exc:
                self._record_error(exc)
            finally:
                self._queue.task_done()

    def _process(self, task: MemoryTask) -> None:
        set_memory_actor(f"worker:{task.thread_id}:{task.task_id}")
        try:
            self._extract_and_encode(task)
            self._maybe_consolidate(task.namespace)
        finally:
            set_memory_actor(None)

    def _extract_and_encode(self, task: MemoryTask) -> list[str]:
        prompt = _EXTRACT_PROMPT.format(
            conversation=self._format_messages(task.messages),
            final_report=task.final_report or "",
        )
        try:
            response = self._llm.invoke(prompt)
            items = json.loads(self._response_text(response).strip().removeprefix("```json").removesuffix("```").strip())
            if not isinstance(items, list):
                raise ValueError("memory extraction response is not a JSON array")
        except Exception as exc:
            self._record_error(exc)
            return []

        encoded_ids: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            try:
                memory_type = MemoryType(str(item.get("type") or "episodic"))
                importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
                trace = self._manager.encode(
                    content,
                    memory_type,
                    namespace=task.namespace,
                    importance=importance,
                    source=f"worker:{task.thread_id}",
                    metadata={"task_id": task.task_id},
                )
                encoded_ids.append(trace.id)
            except Exception as exc:
                self._record_error(exc)
        return encoded_ids

    def _maybe_consolidate(self, namespace: str) -> None:
        active = [
            trace
            for trace in self._manager._store.list_by_type(  # noqa: SLF001
                MemoryType.EPISODIC,
                namespace=namespace,
            )
            if not trace.metadata.get("forgotten")
        ]
        threshold = max(2, self._config.consolidate_threshold)
        if len(active) < threshold:
            return
        active.sort(key=lambda trace: trace.created_at)
        selected = active[:10]
        prompt = _CONSOLIDATE_PROMPT.format(
            memories="\n".join(f"- {trace.content}" for trace in selected)
        )
        try:
            response = self._llm.invoke(prompt)
            merged_content = self._response_text(response).strip()
            if not merged_content:
                raise ValueError("memory consolidation response is empty")
            self._manager.consolidate(
                [trace.id for trace in selected],
                merged_content,
            )
        except Exception as exc:
            self._record_error(exc)

    def _record_error(self, exc: Exception) -> None:
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Memory worker task failed: %s", self._last_error)

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    @staticmethod
    def _format_messages(messages: tuple[BaseMessage, ...]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.content
            if isinstance(content, list):
                text = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            else:
                text = str(content)
            lines.append(f"{message.type}: {text}")
        return "\n".join(lines)
