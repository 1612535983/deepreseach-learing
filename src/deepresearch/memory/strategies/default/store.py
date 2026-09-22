"""Markdown truth store with an in-memory index and atomic rewrites."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from deepresearch.memory.exceptions import MemoryConflictError, MemoryNotFoundError
from deepresearch.memory.schema import Association, MemoryTrace, MemoryType, OperationLog
from deepresearch.memory.types import MemoryFilter


logger = logging.getLogger(__name__)

_TRACE_SEPARATOR = re.compile(
    r"^<!-- trace: ([a-f0-9]+) -->\n(.*?)(?=^<!-- trace:|\Z)",
    re.MULTILINE | re.DOTALL,
)


class MarkdownFileStore:
    """Persist all traces in one inspectable Markdown file."""

    def __init__(
        self,
        storage_path: str | Path,
        *,
        now_provider: Any = time.time,
    ) -> None:
        self._root = Path(storage_path).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._traces_file = self._root / "traces.md"
        self._lock = threading.RLock()
        self._now_provider = now_provider
        self._traces: dict[str, MemoryTrace] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._traces_file

    def _load(self) -> None:
        if not self._traces_file.exists():
            self._atomic_write("# Memory Traces\n\n")
            return
        text = self._traces_file.read_text(encoding="utf-8")
        for match in _TRACE_SEPARATOR.finditer(text):
            separator_id = match.group(1)
            trace = self._parse_trace(separator_id, match.group(2).strip())
            if trace is not None:
                self._traces[trace.id] = trace

    def add(self, trace: MemoryTrace) -> None:
        with self._lock:
            if trace.id in self._traces:
                raise MemoryConflictError(trace.id)
            self._traces[trace.id] = trace
            try:
                self._rewrite_file()
            except Exception:
                self._traces.pop(trace.id, None)
                raise

    def get(self, trace_id: str) -> MemoryTrace | None:
        with self._lock:
            return self._traces.get(trace_id)

    def update(self, trace: MemoryTrace) -> None:
        with self._lock:
            if trace.id not in self._traces:
                raise MemoryNotFoundError(trace.id)
            previous = self._traces[trace.id]
            self._traces[trace.id] = trace
            try:
                self._rewrite_file()
            except Exception:
                self._traces[trace.id] = previous
                raise

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        with self._lock:
            for trace in traces:
                if trace.id not in self._traces:
                    raise MemoryNotFoundError(trace.id)
            previous = {trace.id: self._traces[trace.id] for trace in traces}
            self._traces.update({trace.id: trace for trace in traces})
            try:
                self._rewrite_file()
            except Exception:
                self._traces.update(previous)
                raise

    def remove(self, trace_id: str) -> None:
        with self._lock:
            previous = self._traces.pop(trace_id, None)
            if previous is None:
                return
            try:
                self._rewrite_file()
            except Exception:
                self._traces[trace_id] = previous
                raise

    def list_all(self) -> list[MemoryTrace]:
        with self._lock:
            return list(self._traces.values())

    def list_by_type(
        self,
        type: MemoryType,
        *,
        namespace: str | None = None,
    ) -> list[MemoryTrace]:
        with self._lock:
            return [
                trace
                for trace in self._traces.values()
                if trace.type == type
                and (namespace is None or trace.namespace == namespace)
            ]

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]:
        now = float(self._now_provider())
        matches: list[MemoryTrace] = []
        with self._lock:
            for trace in self._traces.values():
                if filter.namespace is not None and trace.namespace != filter.namespace:
                    continue
                if filter.type is not None and trace.type != filter.type:
                    continue
                if not filter.include_forgotten and trace.metadata.get("forgotten") is True:
                    continue
                if filter.max_age_hours is not None and trace.created_at > 0:
                    age_hours = max(0.0, now - trace.created_at) / 3600.0
                    if age_hours > filter.max_age_hours:
                        continue
                if any(trace.metadata.get(key) != value for key, value in filter.metadata.items()):
                    continue
                matches.append(trace)
        return matches

    def _rewrite_file(self) -> None:
        blocks = ["# Memory Traces\n\n"]
        for trace in self._traces.values():
            blocks.append(
                f"<!-- trace: {trace.id} -->\n{self._serialize_trace(trace)}\n\n"
            )
        self._atomic_write("".join(blocks))

    def _atomic_write(self, content: str) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".traces-",
            suffix=".tmp",
            dir=self._root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._traces_file)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _serialize_trace(self, trace: MemoryTrace) -> str:
        data = {
            "id": trace.id,
            "namespace": trace.namespace,
            "type": trace.type.value,
            "strength": trace.strength,
            "base_strength": trace.base_strength,
            "decay_rate": trace.decay_rate,
            "access_count": trace.access_count,
            "last_accessed": trace.last_accessed,
            "importance": trace.importance,
            "associations": [
                {"target_id": item.target_id, "strength": item.strength, "type": item.type}
                for item in trace.associations
            ],
            "embedding": list(trace.embedding) if trace.embedding is not None else None,
            "source": trace.source,
            "created_at": trace.created_at,
            "metadata": trace.metadata,
            "operation_log": [
                {
                    "timestamp": entry.timestamp,
                    "operation": entry.operation,
                    "actor": entry.actor,
                    "diff": self._to_yaml_value(entry.diff),
                }
                for entry in trace.operation_log
            ],
        }
        frontmatter = yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        return f"---\n{frontmatter}\n---\n{trace.content}"

    @classmethod
    def _to_yaml_value(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return [cls._to_yaml_value(item) for item in value]
        if isinstance(value, list):
            return [cls._to_yaml_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._to_yaml_value(item) for key, item in value.items()}
        return value

    def _parse_trace(self, separator_id: str, body: str) -> MemoryTrace | None:
        try:
            if not body.startswith("---\n"):
                raise ValueError("missing frontmatter opener")
            parts = body[4:].split("\n---\n", 1)
            if len(parts) != 2:
                raise ValueError("missing frontmatter closer")
            frontmatter, content = parts
            data = yaml.safe_load(frontmatter)
            if not isinstance(data, dict):
                raise ValueError("frontmatter is not a mapping")
            if data.get("id") != separator_id:
                raise ValueError("separator id does not match frontmatter id")

            associations = tuple(
                Association(**item) for item in data.pop("associations", [])
            )
            operation_log = tuple(
                OperationLog(
                    timestamp=item["timestamp"],
                    operation=item["operation"],
                    actor=item.get("actor"),
                    diff=self._restore_operation_diff(item.get("diff")),
                )
                for item in data.pop("operation_log", [])
            )
            embedding_data = data.pop("embedding", None)
            embedding = tuple(embedding_data) if embedding_data is not None else None
            memory_type = MemoryType(data.pop("type"))
            return MemoryTrace(
                content=content,
                type=memory_type,
                associations=associations,
                embedding=embedding,
                operation_log=operation_log,
                **data,
            )
        except Exception as exc:
            logger.warning("Skipping corrupt memory trace %s: %s", separator_id, exc)
            return None

    @classmethod
    def _restore_operation_diff(cls, value: Any) -> Any:
        """Restore immutable tuple-shaped audit values serialized as YAML lists."""

        if isinstance(value, list):
            return tuple(cls._restore_operation_diff(item) for item in value)
        if isinstance(value, dict):
            return {key: cls._restore_operation_diff(item) for key, item in value.items()}
        return value
