"""SQLite skill registry with immutable content objects and version lineage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from deepresearch.skill.exceptions import SkillNotFoundError, SkillStoreError
from deepresearch.skill.types import (
    ParsedSkill,
    SkillLineage,
    SkillMetrics,
    SkillRecord,
)


SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_records (
    skill_id                 TEXT PRIMARY KEY,
    name                     TEXT NOT NULL,
    description              TEXT NOT NULL,
    source_path              TEXT NOT NULL,
    object_path              TEXT NOT NULL,
    content_hash             TEXT NOT NULL,
    version                  INTEGER NOT NULL,
    tags_json                TEXT NOT NULL DEFAULT '[]',
    allowed_tools_json       TEXT NOT NULL DEFAULT '[]',
    enabled                  INTEGER NOT NULL DEFAULT 1,
    is_active                INTEGER NOT NULL DEFAULT 1,
    generation               INTEGER NOT NULL DEFAULT 0,
    origin                   TEXT NOT NULL,
    created_by               TEXT,
    created_at               TEXT NOT NULL,
    last_updated             TEXT NOT NULL,
    total_selections         INTEGER NOT NULL DEFAULT 0,
    total_injections         INTEGER NOT NULL DEFAULT 0,
    total_aligned_tool_calls INTEGER NOT NULL DEFAULT 0,
    total_completed_runs     INTEGER NOT NULL DEFAULT 0,
    total_failed_runs        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_records(name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_active_name
    ON skill_records(name) WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS skill_lineage_parents (
    skill_id        TEXT NOT NULL,
    parent_skill_id TEXT NOT NULL,
    PRIMARY KEY (skill_id, parent_skill_id),
    FOREIGN KEY (skill_id) REFERENCES skill_records(skill_id)
);

CREATE TABLE IF NOT EXISTS skill_events (
    event_key  TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    skill_id   TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES skill_records(skill_id)
);
CREATE INDEX IF NOT EXISTS idx_skill_events_run ON skill_events(run_id);

CREATE TABLE IF NOT EXISTS skill_outcomes (
    run_id         TEXT NOT NULL,
    skill_id       TEXT NOT NULL,
    completed      INTEGER NOT NULL,
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, skill_id),
    FOREIGN KEY (skill_id) REFERENCES skill_records(skill_id)
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillStore(Protocol):
    """Storage boundary consumed by selectors, renderers, and middleware."""

    def install(self, parsed: ParsedSkill, *, activate: bool = True) -> SkillRecord: ...

    def get(self, skill_id: str) -> SkillRecord | None: ...

    def get_active(self, name: str) -> SkillRecord | None: ...

    def list_active(self) -> list[SkillRecord]: ...

    def get_versions(self, name: str) -> list[SkillRecord]: ...

    def set_enabled(self, skill_id: str, enabled: bool) -> bool: ...

    def rollback(self, skill_id: str) -> None: ...

    def read_body(self, skill_id: str) -> str: ...

    def record_selection(self, skill_id: str, run_id: str) -> bool: ...

    def record_injection(self, skill_id: str, run_id: str) -> bool: ...

    def record_tool_alignment(
        self, skill_id: str, run_id: str, tool_call_id: str, tool_name: str
    ) -> bool: ...

    def record_outcome(
        self, skill_id: str, run_id: str, *, completed: bool, note: str = ""
    ) -> bool: ...

    def get_metrics(self, skill_id: str) -> SkillMetrics | None: ...


class SQLiteSkillStore:
    """Thread-safe SQLite index backed by content-addressed Markdown objects."""

    def __init__(self, db_path: str | Path, storage_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._storage_path = Path(storage_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._mu = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._mu:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise SkillStoreError(
                    f"Skill DB schema {current} 高于程序支持版本 {SCHEMA_VERSION}。"
                )
            self._conn.executescript(_SCHEMA_SQL)
            if current < SCHEMA_VERSION:
                self._migrate(current, SCHEMA_VERSION)
                self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    def _migrate(self, from_version: int, to_version: int) -> None:
        """Run the explicit migration chain as future schema versions are added."""

        version = from_version
        migrations: dict[tuple[int, int], object] = {}
        while version < to_version:
            migration = migrations.get((version, version + 1))
            if migration is not None:
                migration(self._conn)  # type: ignore[operator]
            version += 1

    def _object_path(self, content_hash: str) -> Path:
        return self._storage_path / f"{content_hash}.md"

    def _write_object(self, parsed: ParsedSkill) -> Path:
        calculated = hashlib.sha256(parsed.raw_content.encode("utf-8")).hexdigest()
        if calculated != parsed.record.content_hash:
            raise SkillStoreError("Skill 内容在解析后发生变化，hash 校验失败。")
        target = self._object_path(calculated)
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            existing_hash = hashlib.sha256(existing.encode("utf-8")).hexdigest()
            if existing_hash != calculated:
                raise SkillStoreError(f"Skill 内容对象损坏：{target}")
            return target
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(parsed.raw_content, encoding="utf-8")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def install(self, parsed: ParsedSkill, *, activate: bool = True) -> SkillRecord:
        """Install idempotently; new content becomes a child of the active version."""

        object_path = self._write_object(parsed)
        with self._mu:
            existing = self._conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?",
                (parsed.record.skill_id,),
            ).fetchone()
            if existing is not None:
                self._conn.execute(
                    """UPDATE skill_records
                       SET source_path=?, object_path=?, description=?, version=?,
                           tags_json=?, allowed_tools_json=?, last_updated=?
                       WHERE skill_id=?""",
                    (
                        parsed.record.source_path,
                        str(object_path.resolve()),
                        parsed.record.description,
                        parsed.record.version,
                        json.dumps(list(parsed.record.tags), ensure_ascii=False),
                        json.dumps(list(parsed.record.allowed_tools), ensure_ascii=False),
                        _utc_now(),
                        parsed.record.skill_id,
                    ),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM skill_records WHERE skill_id=?",
                    (parsed.record.skill_id,),
                ).fetchone()
                return self._row_to_record(row)

            active_row = self._conn.execute(
                "SELECT * FROM skill_records WHERE name=? AND is_active=1",
                (parsed.record.name,),
            ).fetchone()
            parents = parsed.record.lineage.parent_skill_ids
            generation = parsed.record.lineage.generation
            if active_row is not None and not parents:
                parents = (str(active_row["skill_id"]),)
                generation = int(active_row["generation"]) + 1
            if activate:
                self._conn.execute(
                    "UPDATE skill_records SET is_active=0 WHERE name=?",
                    (parsed.record.name,),
                )
            timestamp = _utc_now()
            self._conn.execute(
                """INSERT INTO skill_records
                   (skill_id, name, description, source_path, object_path,
                    content_hash, version, tags_json, allowed_tools_json,
                    enabled, is_active, generation, origin, created_by,
                    created_at, last_updated)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    parsed.record.skill_id,
                    parsed.record.name,
                    parsed.record.description,
                    parsed.record.source_path,
                    str(object_path.resolve()),
                    parsed.record.content_hash,
                    parsed.record.version,
                    json.dumps(list(parsed.record.tags), ensure_ascii=False),
                    json.dumps(list(parsed.record.allowed_tools), ensure_ascii=False),
                    1 if parsed.record.enabled else 0,
                    1 if activate else 0,
                    generation,
                    parsed.record.lineage.origin,
                    parsed.record.lineage.created_by,
                    timestamp,
                    timestamp,
                ),
            )
            for parent_id in parents:
                self._conn.execute(
                    "INSERT OR IGNORE INTO skill_lineage_parents VALUES (?,?)",
                    (parsed.record.skill_id, parent_id),
                )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?",
                (parsed.record.skill_id,),
            ).fetchone()
            return self._row_to_record(row)

    def get(self, skill_id: str) -> SkillRecord | None:
        with self._mu:
            row = self._conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_active(self, name: str) -> SkillRecord | None:
        with self._mu:
            row = self._conn.execute(
                "SELECT * FROM skill_records WHERE name=? AND is_active=1", (name,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_active(self) -> list[SkillRecord]:
        with self._mu:
            rows = self._conn.execute(
                "SELECT * FROM skill_records WHERE is_active=1 ORDER BY name"
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_versions(self, name: str) -> list[SkillRecord]:
        with self._mu:
            rows = self._conn.execute(
                """SELECT * FROM skill_records WHERE name=?
                   ORDER BY generation, created_at""",
                (name,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def set_enabled(self, skill_id: str, enabled: bool) -> bool:
        with self._mu:
            cursor = self._conn.execute(
                """UPDATE skill_records SET enabled=?, last_updated=?
                   WHERE skill_id=?""",
                (1 if enabled else 0, _utc_now(), skill_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def rollback(self, skill_id: str) -> None:
        with self._mu:
            row = self._conn.execute(
                "SELECT name FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            if row is None:
                raise SkillNotFoundError(f"Skill 版本不存在：{skill_id}")
            self._conn.execute(
                "UPDATE skill_records SET is_active=0 WHERE name=?", (row["name"],)
            )
            self._conn.execute(
                """UPDATE skill_records SET is_active=1, last_updated=?
                   WHERE skill_id=?""",
                (_utc_now(), skill_id),
            )
            self._conn.commit()

    def read_body(self, skill_id: str) -> str:
        record = self.get(skill_id)
        if record is None:
            raise SkillNotFoundError(f"Skill 版本不存在：{skill_id}")
        path = Path(record.object_path)
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SkillStoreError(f"无法读取 Skill 内容对象 {path}：{exc}") from exc
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != record.content_hash:
            raise SkillStoreError(f"Skill 内容对象 hash 不匹配：{path}")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3:
                return parts[2].lstrip("\r\n").strip()
        return content.strip()

    def _record_event(
        self,
        skill_id: str,
        run_id: str,
        event_type: str,
        event_suffix: str,
        counter: str,
        detail: str = "",
    ) -> bool:
        event_key = hashlib.sha256(
            f"{run_id}\x00{skill_id}\x00{event_type}\x00{event_suffix}".encode()
        ).hexdigest()
        with self._mu:
            exists = self._conn.execute(
                "SELECT 1 FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            if exists is None:
                return False
            cursor = self._conn.execute(
                """INSERT OR IGNORE INTO skill_events
                   (event_key, run_id, skill_id, event_type, detail, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (event_key, run_id, skill_id, event_type, detail, _utc_now()),
            )
            if cursor.rowcount == 0:
                self._conn.commit()
                return False
            allowed_counters = {
                "total_selections",
                "total_injections",
                "total_aligned_tool_calls",
            }
            if counter not in allowed_counters:
                self._conn.rollback()
                raise SkillStoreError(f"不支持的 Skill counter：{counter}")
            self._conn.execute(
                f"UPDATE skill_records SET {counter}={counter}+1, last_updated=? "
                "WHERE skill_id=?",
                (_utc_now(), skill_id),
            )
            self._conn.commit()
            return True

    def record_selection(self, skill_id: str, run_id: str) -> bool:
        return self._record_event(
            skill_id, run_id, "selection", "once", "total_selections"
        )

    def record_injection(self, skill_id: str, run_id: str) -> bool:
        return self._record_event(
            skill_id, run_id, "injection", "once", "total_injections"
        )

    def record_tool_alignment(
        self,
        skill_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
    ) -> bool:
        return self._record_event(
            skill_id,
            run_id,
            "tool_alignment",
            tool_call_id,
            "total_aligned_tool_calls",
            detail=tool_name,
        )

    def record_outcome(
        self,
        skill_id: str,
        run_id: str,
        *,
        completed: bool,
        note: str = "",
    ) -> bool:
        with self._mu:
            exists = self._conn.execute(
                "SELECT 1 FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            if exists is None:
                return False
            cursor = self._conn.execute(
                """INSERT OR IGNORE INTO skill_outcomes
                   (run_id, skill_id, completed, note, created_at)
                   VALUES (?,?,?,?,?)""",
                (run_id, skill_id, 1 if completed else 0, note, _utc_now()),
            )
            if cursor.rowcount == 0:
                self._conn.commit()
                return False
            counter = "total_completed_runs" if completed else "total_failed_runs"
            self._conn.execute(
                f"UPDATE skill_records SET {counter}={counter}+1, last_updated=? "
                "WHERE skill_id=?",
                (_utc_now(), skill_id),
            )
            self._conn.commit()
            return True

    def get_metrics(self, skill_id: str) -> SkillMetrics | None:
        record = self.get(skill_id)
        if record is None:
            return None
        return SkillMetrics(
            skill_id=record.skill_id,
            selections=record.total_selections,
            injections=record.total_injections,
            aligned_tool_calls=record.total_aligned_tool_calls,
            completed_runs=record.total_completed_runs,
            failed_runs=record.total_failed_runs,
            injection_rate=record.injection_rate,
            completion_rate=record.completion_rate,
        )

    def _row_to_record(self, row: sqlite3.Row) -> SkillRecord:
        parent_rows = self._conn.execute(
            """SELECT parent_skill_id FROM skill_lineage_parents
               WHERE skill_id=? ORDER BY parent_skill_id""",
            (row["skill_id"],),
        ).fetchall()
        return SkillRecord(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"],
            source_path=row["source_path"],
            object_path=row["object_path"],
            content_hash=row["content_hash"],
            version=row["version"],
            tags=tuple(json.loads(row["tags_json"])),
            allowed_tools=tuple(json.loads(row["allowed_tools_json"])),
            enabled=bool(row["enabled"]),
            is_active=bool(row["is_active"]),
            lineage=SkillLineage(
                parent_skill_ids=tuple(item["parent_skill_id"] for item in parent_rows),
                generation=row["generation"],
                origin=row["origin"],
                created_by=row["created_by"],
            ),
            created_at=row["created_at"],
            last_updated=row["last_updated"],
            total_selections=row["total_selections"],
            total_injections=row["total_injections"],
            total_aligned_tool_calls=row["total_aligned_tool_calls"],
            total_completed_runs=row["total_completed_runs"],
            total_failed_runs=row["total_failed_runs"],
        )

    def close(self) -> None:
        with self._mu:
            self._conn.close()

    def __enter__(self) -> "SQLiteSkillStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
