"""Create and verify recoverable State snapshots before P4 compaction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


DEFAULT_SNAPSHOT_DIR = Path(".deepresearch/snapshots")
SNAPSHOT_SCHEMA_VERSION = 1
_CST = timezone(timedelta(hours=8))


def _now() -> datetime:
    return datetime.now(_CST)


def _safe_namespace(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if normalized:
        return normalized[:80]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"namespace-{digest}"


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _checksum(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextSnapshot:
    """Verified snapshot content loaded from disk."""

    path: Path
    snapshot_id: str
    created_at: str
    namespace: str
    messages: list[BaseMessage]
    state: dict[str, Any]


@dataclass(frozen=True)
class SnapshotResult:
    """Metadata returned after a snapshot is durably persisted."""

    path: Path
    snapshot_id: str
    message_count: int
    bytes_written: int


class ContextSnapshotter:
    """Persist pre-compaction messages and non-message State with a checksum."""

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        *,
        now_provider: Callable[[], datetime] = _now,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._now_provider = now_provider

    def create(
        self,
        state: Mapping[str, Any],
        messages: Sequence[BaseMessage],
        *,
        namespace: str,
    ) -> SnapshotResult:
        """Write one immutable snapshot and return its durable identity."""

        created_at = self._now_provider()
        snapshot_state = {
            key: _json_safe(value)
            for key, value in state.items()
            if key != "messages"
        }
        body: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": created_at.isoformat(),
            "namespace": namespace,
            "messages": [message_to_dict(message) for message in messages],
            "state": snapshot_state,
        }
        snapshot_id = _checksum(body)
        payload = {**body, "sha256": snapshot_id}
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
        directory = self._root_dir / _safe_namespace(namespace)
        path = directory / f"snapshot-{timestamp}-{snapshot_id[:12]}.json"
        self._write_once(path, serialized)
        return SnapshotResult(
            path=path,
            snapshot_id=snapshot_id,
            message_count=len(messages),
            bytes_written=len((serialized + "\n").encode("utf-8")),
        )

    @staticmethod
    def load(path: str | Path) -> ContextSnapshot:
        """Load a snapshot, validate its schema and checksum, and restore messages."""

        snapshot_path = Path(path)
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("不支持的上下文快照版本。")
        stored_checksum = payload.pop("sha256", None)
        calculated_checksum = _checksum(payload)
        if stored_checksum != calculated_checksum:
            raise ValueError("上下文快照校验失败，文件可能已损坏或被修改。")
        message_payload = payload.get("messages")
        state_payload = payload.get("state")
        if not isinstance(message_payload, list) or not isinstance(
            state_payload,
            dict,
        ):
            raise ValueError("上下文快照缺少 messages 或 state。")
        return ContextSnapshot(
            path=snapshot_path,
            snapshot_id=calculated_checksum,
            created_at=str(payload.get("created_at") or ""),
            namespace=str(payload.get("namespace") or ""),
            messages=messages_from_dict(message_payload),
            state=state_payload,
        )

    @staticmethod
    def _write_once(path: Path, serialized: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as output_file:
                output_file.write(serialized)
                output_file.write("\n")
        except FileExistsError:
            existing = path.read_text(encoding="utf-8").rstrip("\n")
            if existing != serialized:
                raise RuntimeError(f"快照文件已存在但内容不一致：{path}") from None
