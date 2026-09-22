from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.middlewares.skill_metrics import SkillMetricsMiddleware
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import SkillManager
from deepresearch.state import create_initial_state, merge_skill_runtime


def _manager(tmp_path: Path) -> SkillManager:
    skill_dir = tmp_path / "skills" / "verification"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: source-verification
description: 核验事实与来源
tools: [web_search, read_page]
---
交叉核验关键结论。
""",
        encoding="utf-8",
    )
    manager = SkillManager(
        SkillConfig(
            use="default",
            db_path=tmp_path / "skills.db",
            storage_path=tmp_path / "objects",
            skill_dirs=(tmp_path / "skills",),
            include_builtin=False,
        )
    )
    manager.load_startup()
    return manager


def _completed_state(manager: SkillManager):  # noqa: ANN202
    record = manager.list_skills()[0]
    state = create_initial_state("核验来源")
    state["skills"] = merge_skill_runtime(
        state["skills"],
        {
            "selected": [
                {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "content_hash": record.content_hash,
                    "score": 1.0,
                    "reason": "test",
                    "forced": False,
                    "allowed_tools": list(record.allowed_tools),
                }
            ],
            "injection_count": 1,
        },
    )
    state["messages"].extend(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {}, "id": "search-1"},
                    {"name": "write_final_report", "args": {}, "id": "report-1"},
                ],
            ),
            AIMessage(content="最终答案"),
        ]
    )
    state["final_report"] = "# 报告"
    return state, record


def test_metrics_record_aligned_tools_and_outcome_once(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    state, record = _completed_state(manager)
    middleware = SkillMetricsMiddleware(manager)

    update = middleware.after_agent(state, Runtime())

    assert update is not None
    assert update["skills"]["aligned_tool_calls"] == 1
    assert update["skills"]["completed_recorded"] is True
    state["skills"] = merge_skill_runtime(state["skills"], update["skills"])
    assert middleware.after_agent(state, Runtime()) is None
    metrics = manager.store.get_metrics(record.skill_id)
    assert metrics is not None
    assert metrics.aligned_tool_calls == 1
    assert metrics.completed_runs == 1
    manager.close()


def test_metrics_skip_dropped_skills_and_support_async(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    state, record = _completed_state(manager)
    state["skills"]["dropped"] = [
        {**state["skills"]["selected"][0], "drop_reason": "token_budget"}
    ]
    middleware = SkillMetricsMiddleware(manager)

    assert asyncio.run(middleware.aafter_agent(state, Runtime())) is None
    metrics = manager.store.get_metrics(record.skill_id)
    assert metrics is not None
    assert metrics.aligned_tool_calls == 0
    assert metrics.completed_runs == 0
    manager.close()


def test_uncheckpointed_runs_with_same_question_have_distinct_metrics_ids(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    middleware = SkillMetricsMiddleware(manager)
    first, record = _completed_state(manager)
    second, _ = _completed_state(manager)

    assert first["skills"]["run_id"] != second["skills"]["run_id"]
    assert middleware.after_agent(first, Runtime()) is not None
    assert middleware.after_agent(second, Runtime()) is not None
    metrics = manager.store.get_metrics(record.skill_id)
    assert metrics is not None
    assert metrics.completed_runs == 2
    manager.close()
