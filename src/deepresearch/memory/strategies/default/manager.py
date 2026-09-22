"""Poirot-style memory operations with no model dependency."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from deepresearch.memory.exceptions import MemoryNotFoundError
from deepresearch.memory.schema import Association, MemoryTrace, MemoryType, OperationLog
from deepresearch.memory.store import MemoryStore
from deepresearch.memory.strategies.default.constants import (
    CONSOLIDATED_IMPORTANCE_BOOST,
    DECAY_PARAMS,
    MAX_ASSOCIATIONS_PER_TRACE,
    MAX_TRACES_TO_CONSOLIDATE,
    MIN_TRACES_TO_CONSOLIDATE,
)


_memory_actor: ContextVar[str | None] = ContextVar("deepresearch_memory_actor", default=None)


def set_memory_actor(actor: str | None) -> None:
    """Associate subsequent manager operations with a run or worker task."""

    _memory_actor.set(actor)


class DefaultMemoryManager:
    """Encode and evolve immutable traces through an injected store."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        now_provider: Callable[[], float] = time.time,
        journal: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._store = store
        self._now_provider = now_provider
        self._journal = journal

    @staticmethod
    def compute_trace_id(namespace: str, content: str, type: MemoryType) -> str:
        raw = f"{namespace}\x00{content}\x00{type.value}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def encode(
        self,
        content: str,
        type: MemoryType,
        *,
        namespace: str = "default",
        importance: float = 0.5,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryTrace:
        normalized_content = content.strip()
        normalized_namespace = namespace.strip()
        if not normalized_content:
            raise ValueError("记忆内容不能为空。")
        if not normalized_namespace:
            raise ValueError("memory namespace 不能为空。")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance 必须在 0 到 1 之间。")

        trace_id = self.compute_trace_id(normalized_namespace, normalized_content, type)
        existing = self._store.get(trace_id)
        if existing is not None:
            self._emit("memory.encode.duplicate", {"trace_id": trace_id})
            return existing

        now = self._now_provider()
        params = DECAY_PARAMS[type]
        trace = MemoryTrace(
            id=trace_id,
            namespace=normalized_namespace,
            content=normalized_content,
            type=type,
            strength=params["base_strength"],
            base_strength=params["base_strength"],
            decay_rate=params["decay_rate"],
            last_accessed=now,
            importance=importance,
            source=source,
            created_at=now,
            metadata=dict(metadata or {}),
            operation_log=(
                OperationLog(
                    timestamp=now,
                    operation="encode",
                    actor=_memory_actor.get(),
                    diff={"content": (None, normalized_content[:200]), "type": (None, type.value)},
                ),
            ),
        )
        self._store.add(trace)
        self._emit("memory.encode", {"trace_id": trace.id, "type": type.value})
        return trace

    def associate(
        self,
        trace_id_a: str,
        trace_id_b: str,
        *,
        strength: float = 0.5,
        type: str = "related",
    ) -> None:
        if trace_id_a == trace_id_b:
            raise ValueError("不能把一条记忆关联到自身。")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("关联强度必须在 0 到 1 之间。")
        trace_a = self._require(trace_id_a)
        trace_b = self._require(trace_id_b)
        if trace_a.namespace != trace_b.namespace:
            raise ValueError("不能跨 namespace 建立记忆关联。")

        now = self._now_provider()
        updated_a = self._with_association(
            trace_a,
            Association(target_id=trace_b.id, strength=strength, type=type),
        ).with_operation(
            OperationLog(now, "associate", _memory_actor.get(), {"target": trace_b.id})
        )
        updated_b = self._with_association(
            trace_b,
            Association(target_id=trace_a.id, strength=strength, type=type),
        ).with_operation(
            OperationLog(now, "associate", _memory_actor.get(), {"target": trace_a.id})
        )
        self._store.batch_update([updated_a, updated_b])
        self._emit("memory.associate", {"trace_id_a": trace_id_a, "trace_id_b": trace_id_b})

    def consolidate(self, trace_ids: list[str], merged_content: str) -> MemoryTrace:
        unique_ids = list(dict.fromkeys(trace_ids))
        if not MIN_TRACES_TO_CONSOLIDATE <= len(unique_ids) <= MAX_TRACES_TO_CONSOLIDATE:
            raise ValueError(
                "consolidate 的记忆数量必须在 "
                f"{MIN_TRACES_TO_CONSOLIDATE} 到 {MAX_TRACES_TO_CONSOLIDATE} 之间。"
            )
        content = merged_content.strip()
        if not content:
            raise ValueError("合并后的记忆内容不能为空。")

        old_traces = [self._require(trace_id) for trace_id in unique_ids]
        namespace = old_traces[0].namespace
        if any(trace.namespace != namespace for trace in old_traces):
            raise ValueError("不能跨 namespace 巩固记忆。")
        new_id = self.compute_trace_id(namespace, content, MemoryType.SEMANTIC)
        existing = self._store.get(new_id)
        if existing is not None:
            self._emit("memory.consolidate.duplicate", {"trace_id": new_id})
            return existing

        internal_ids = set(unique_ids)
        external_associations: dict[str, Association] = {}
        for trace in old_traces:
            for association in trace.associations:
                if association.target_id not in internal_ids:
                    current = external_associations.get(association.target_id)
                    if current is None or association.strength > current.strength:
                        external_associations[association.target_id] = association

        now = self._now_provider()
        params = DECAY_PARAMS[MemoryType.SEMANTIC]
        importance = min(
            1.0,
            max(trace.importance for trace in old_traces) + CONSOLIDATED_IMPORTANCE_BOOST,
        )
        new_trace = MemoryTrace(
            id=new_id,
            namespace=namespace,
            content=content,
            type=MemoryType.SEMANTIC,
            strength=params["base_strength"],
            base_strength=params["base_strength"],
            decay_rate=params["decay_rate"],
            last_accessed=now,
            importance=importance,
            associations=tuple(external_associations.values()),
            source=f"consolidate:{','.join(unique_ids)}",
            created_at=now,
            metadata={"consolidated_from": unique_ids},
            operation_log=(
                OperationLog(
                    now,
                    "consolidate",
                    _memory_actor.get(),
                    {"source_trace_ids": tuple(unique_ids)},
                ),
            ),
        )
        self._store.add(new_trace)

        forgotten: list[MemoryTrace] = []
        for trace in old_traces:
            metadata = dict(trace.metadata)
            metadata.update({"forgotten": True, "consolidated_into": new_id})
            forgotten.append(
                replace(trace, metadata=metadata).with_operation(
                    OperationLog(
                        now,
                        "forget",
                        _memory_actor.get(),
                        {"consolidated_into": new_id},
                    )
                )
            )
        self._store.batch_update(forgotten)
        self._emit("memory.consolidate", {"trace_id": new_id, "old_trace_ids": unique_ids})
        return new_trace

    def reconsolidate(self, trace_id: str, new_content: str) -> MemoryTrace:
        trace = self._require(trace_id)
        content = new_content.strip()
        if not content:
            raise ValueError("修正后的记忆内容不能为空。")
        now = self._now_provider()
        updated = replace(trace, content=content, last_accessed=now).with_operation(
            OperationLog(
                now,
                "reconsolidate",
                _memory_actor.get(),
                {"content": (trace.content[:200], content[:200])},
            )
        )
        self._store.update(updated)
        self._emit("memory.reconsolidate", {"trace_id": trace_id})
        return updated

    def forget(self, trace_id: str, *, reason: str) -> MemoryTrace:
        trace = self._require(trace_id)
        now = self._now_provider()
        metadata = dict(trace.metadata)
        metadata.update({"forgotten": True, "forget_reason": reason})
        updated = replace(trace, metadata=metadata).with_operation(
            OperationLog(now, "forget", _memory_actor.get(), {"reason": reason})
        )
        self._store.update(updated)
        self._emit("memory.forget", {"trace_id": trace_id, "reason": reason})
        return updated

    def _require(self, trace_id: str) -> MemoryTrace:
        trace = self._store.get(trace_id)
        if trace is None:
            raise MemoryNotFoundError(trace_id)
        return trace

    @staticmethod
    def _with_association(trace: MemoryTrace, association: Association) -> MemoryTrace:
        remaining = [
            item for item in trace.associations if item.target_id != association.target_id
        ]
        remaining.append(association)
        if len(remaining) > MAX_ASSOCIATIONS_PER_TRACE:
            remaining = sorted(remaining, key=lambda item: item.strength, reverse=True)[
                :MAX_ASSOCIATIONS_PER_TRACE
            ]
        return replace(trace, associations=tuple(remaining))

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._journal is not None:
            self._journal(event, payload)
