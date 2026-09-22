from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from deepresearch.skill.exceptions import SkillNotFoundError, SkillStoreError
from deepresearch.skill.parser import parse_skill_file
from deepresearch.skill.store import SCHEMA_VERSION, SQLiteSkillStore


def _skill(root: Path, body: str, *, version: int = 1):
    directory = root / "verify"
    directory.mkdir(exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        "\n".join(
            [
                "---",
                "name: verify",
                "description: Verify sources",
                f"version: {version}",
                "tags: [research]",
                "tools: [read_page]",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return parse_skill_file(path)


def _store(tmp_path: Path) -> SQLiteSkillStore:
    return SQLiteSkillStore(tmp_path / "skills.db", tmp_path / "objects")


def test_store_initializes_wal_and_schema_version(tmp_path: Path) -> None:
    store = _store(tmp_path)

    journal_mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]  # noqa: SLF001
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]  # noqa: SLF001

    assert journal_mode == "wal"
    assert version == SCHEMA_VERSION
    store.close()


def test_install_writes_immutable_object_and_is_idempotent(tmp_path: Path) -> None:
    parsed = _skill(tmp_path, "# Verify\n\nRead the source.")
    store = _store(tmp_path)

    first = store.install(parsed)
    second = store.install(parsed)

    assert first.skill_id == second.skill_id
    assert Path(first.object_path).read_text(encoding="utf-8") == parsed.raw_content
    assert store.read_body(first.skill_id) == parsed.body
    assert len(store.get_versions("verify")) == 1
    store.close()


def test_new_content_creates_child_and_moves_active_pointer(tmp_path: Path) -> None:
    first_parsed = _skill(tmp_path, "first", version=1)
    store = _store(tmp_path)
    first = store.install(first_parsed)
    second_parsed = _skill(tmp_path, "second", version=2)

    second = store.install(second_parsed)
    versions = store.get_versions("verify")

    assert [item.skill_id for item in versions] == [first.skill_id, second.skill_id]
    assert versions[0].is_active is False
    assert versions[1].is_active is True
    assert versions[1].lineage.parent_skill_ids == (first.skill_id,)
    assert versions[1].lineage.generation == 1
    assert store.get_active("verify").skill_id == second.skill_id  # type: ignore[union-attr]
    store.close()


def test_rollback_and_enable_state_survive_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.install(_skill(tmp_path, "first"))
    second = store.install(_skill(tmp_path, "second", version=2))

    assert store.set_enabled(first.skill_id, False) is True
    store.rollback(first.skill_id)
    store.close()

    reopened = _store(tmp_path)
    active = reopened.get_active("verify")
    assert active is not None
    assert active.skill_id == first.skill_id
    assert active.enabled is False
    assert reopened.get(second.skill_id).is_active is False  # type: ignore[union-attr]
    reopened.close()


def test_runtime_metrics_are_idempotent_per_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.install(_skill(tmp_path, "body"))

    assert store.record_selection(record.skill_id, "run-1") is True
    assert store.record_selection(record.skill_id, "run-1") is False
    assert store.record_injection(record.skill_id, "run-1") is True
    assert store.record_injection(record.skill_id, "run-1") is False
    assert store.record_tool_alignment(
        record.skill_id, "run-1", "call-1", "read_page"
    ) is True
    assert store.record_tool_alignment(
        record.skill_id, "run-1", "call-1", "read_page"
    ) is False
    assert store.record_outcome(record.skill_id, "run-1", completed=True) is True
    assert store.record_outcome(record.skill_id, "run-1", completed=False) is False

    metrics = store.get_metrics(record.skill_id)
    assert metrics is not None
    assert metrics.selections == 1
    assert metrics.injections == 1
    assert metrics.aligned_tool_calls == 1
    assert metrics.completed_runs == 1
    assert metrics.failed_runs == 0
    assert metrics.injection_rate == 1.0
    assert metrics.completion_rate == 1.0
    store.close()


def test_read_body_detects_tampered_object(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.install(_skill(tmp_path, "body"))
    Path(record.object_path).write_text("tampered", encoding="utf-8")

    with pytest.raises(SkillStoreError, match="hash 不匹配"):
        store.read_body(record.skill_id)
    store.close()


def test_rollback_rejects_unknown_version(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(SkillNotFoundError, match="不存在"):
        store.rollback("missing")
    store.close()


def test_store_rejects_database_from_newer_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(SkillStoreError, match="高于程序支持版本"):
        SQLiteSkillStore(db_path, tmp_path / "objects")
