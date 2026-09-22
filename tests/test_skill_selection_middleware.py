from __future__ import annotations

import asyncio
from pathlib import Path

from langgraph.runtime import Runtime

from deepresearch.middlewares.skill_selection import SkillSelectionMiddleware
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import SkillManager
from deepresearch.state import create_initial_state, merge_skill_runtime


def _manager(tmp_path: Path) -> SkillManager:
    skill_dir = tmp_path / "user-skills" / "verify"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: source-verification
description: 核验事实声明和来源
tags: [来源, 核验]
tools: [web_search, read_page]
---
# Verify

Read primary sources and cross-check claims.
""",
        encoding="utf-8",
    )
    manager = SkillManager(
        SkillConfig(
            use="default",
            db_path=tmp_path / "skills.db",
            storage_path=tmp_path / "objects",
            skill_dirs=(tmp_path / "user-skills",),
            include_builtin=False,
            min_relevance=0.01,
        )
    )
    manager.load_startup()
    return manager


def test_selection_records_compact_refs_once_per_question(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    middleware = SkillSelectionMiddleware(
        manager,
        available_tools=("web_search", "read_page"),
    )
    state = create_initial_state("请核验这些来源")

    first = middleware.before_model(state, Runtime())
    assert first is not None
    state["skills"] = merge_skill_runtime(state["skills"], first["skills"])
    selected = state["skills"]["selected"]

    assert len(selected) == 1
    assert selected[0]["name"] == "source-verification"
    assert "body" not in selected[0]
    assert state["skills"]["selection_count"] == 1
    assert middleware.before_model(state, Runtime()) is None
    metrics = manager.store.get_metrics(selected[0]["skill_id"])
    assert metrics is not None
    assert metrics.selections == 1
    manager.close()


def test_selection_supports_override_for_unrelated_question(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    middleware = SkillSelectionMiddleware(
        manager,
        available_tools=(),
        overrides=("source-verification",),
    )
    state = create_initial_state("你好")

    update = middleware.before_model(state, Runtime())

    assert update is not None
    assert update["skills"]["selected"][0]["forced"] is True
    manager.close()


def test_async_selection_matches_sync_behavior(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    middleware = SkillSelectionMiddleware(
        manager,
        available_tools=("read_page",),
    )
    state = create_initial_state("核验来源")

    update = asyncio.run(middleware.abefore_model(state, Runtime()))

    assert update is not None
    assert update["skills"]["selected"]
    manager.close()


def test_selection_failure_is_checkpoint_visible_and_fail_open(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    middleware = SkillSelectionMiddleware(
        manager,
        available_tools=("read_page",),
    )
    state = create_initial_state("核验来源")
    manager.close()

    update = middleware.before_model(state, Runtime())

    assert update is not None
    assert update["skills"]["selected"] == []
    assert "ProgrammingError" in update["skills"]["last_error"]
