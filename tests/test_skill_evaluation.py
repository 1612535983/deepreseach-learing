from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from deepresearch.agent import build_agent
from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.skill import (
    SkillRunEvaluator,
    build_skill_evaluation_payload,
)
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
)
from deepresearch.middlewares.skill_evaluation import SkillEvaluationMiddleware
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import SkillManager
from deepresearch.state import create_initial_state, merge_skill_runtime


class ProviderStub:
    name = "stub"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def _response(self, questions):  # noqa: ANN001, ANN202
        answers = {}
        for name, question in questions.items():
            if name.endswith("failure_cause"):
                answers[name] = DecisionAnswer(
                    "choice",
                    "task_succeeded",
                    probabilities={"task_succeeded": 0.9, "unclear": 0.1},
                    confidence=0.9,
                )
            else:
                value = 0.1 if name.endswith("instruction_defect") else 0.9
                answers[name] = DecisionAnswer("noul", value)
        return DecisionResponse(
            provider="stub",
            model="stub-v1",
            answers=answers,
            usage=DecisionUsage(input_tokens=120, output_tokens=20, cost_usd=0.003),
            latency_ms=17,
        )

    def evaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.error:
            raise self.error
        return self._response(questions)

    async def aevaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.error:
            raise self.error
        return self._response(questions)


def _manager(tmp_path: Path) -> SkillManager:
    root = tmp_path / "skills" / "verify"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        """---
name: verify
description: 核验关键来源
tags: [verification]
tools: [read_page]
---
读取一手来源，并为关键结论提供引用。
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
            enable_evaluation=True,
        )
    )
    manager.load_startup()
    return manager


def _state(manager: SkillManager):  # noqa: ANN202
    record = manager.list_skills()[0]
    state = create_initial_state("核验这个技术结论")
    state["messages"].append(
        AIMessage(
            content="研究完成",
            tool_calls=[],
        )
    )
    state["final_report"] = "# 报告\n\n结论有来源支持。"
    state["skills"]["selected"] = [
        {
            "skill_id": record.skill_id,
            "name": record.name,
            "content_hash": record.content_hash,
            "score": 0.8,
            "reason": "test",
            "allowed_tools": ["read_page"],
        }
    ]
    state["skills"]["injection_count"] = 1
    state["evaluation"]["report"].update(
        {"status": "completed", "composite_score": 0.82}
    )
    state["memory"]["last_error"] = "secret-memory-value"
    return state, record


def _evaluator(manager: SkillManager, provider: ProviderStub) -> SkillRunEvaluator:
    return SkillRunEvaluator(
        provider,
        EvaluationConfig(use="jev", api_key="test", max_payload_chars=12_000),
        manager.config,
        manager.store,
    )


def test_skill_payload_is_bounded_and_excludes_memory_and_messages(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        state, record = _state(manager)
        payload = build_skill_evaluation_payload(
            state,
            manager.store,
            EvaluationConfig(use="jev", api_key="test", max_payload_chars=12_000),
            manager.config,
        )
        serialized = json.dumps(payload.state, ensure_ascii=False)

        assert payload.input_chars <= 12_000
        assert "secret-memory-value" not in serialized
        assert "messages" not in payload.state
        assert payload.skill_keys == {"skill_1": record.skill_id}
        assert set(payload.questions) == {
            "skill_1_applicable",
            "skill_1_followed",
            "skill_1_helpful",
            "skill_1_instruction_defect",
            "skill_1_failure_cause",
        }
    finally:
        manager.close()


def test_skill_evaluator_batches_and_persists_shadow_result(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        state, record = _state(manager)
        provider = ProviderStub()
        middleware = SkillEvaluationMiddleware(_evaluator(manager, provider))

        update = middleware.after_agent(state, None)  # type: ignore[arg-type]

        assert update is not None
        assert update["skills"]["evaluation_status"] == "completed"
        assert update["skills"]["evaluation_count"] == 1
        evaluations = manager.store.list_evaluations(record.skill_id)
        assert len(evaluations) == 1
        assert evaluations[0].applicable == 0.9
        assert evaluations[0].helpful == 0.9
        assert evaluations[0].instruction_defect == 0.1
        assert evaluations[0].failure_cause == "task_succeeded"
        assert evaluations[0].report_score == 0.82
        assert provider.calls == 1

        state["skills"] = merge_skill_runtime(state["skills"], update["skills"])
        assert middleware.after_agent(state, None) is None  # type: ignore[arg-type]
        assert provider.calls == 1
    finally:
        manager.close()


def test_skill_evaluation_fails_open_and_supports_async(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        state, record = _state(manager)
        failing = SkillEvaluationMiddleware(
            _evaluator(manager, ProviderStub(RuntimeError("secret")))
        )
        update = failing.after_agent(state, None)  # type: ignore[arg-type]
        assert update is not None
        assert update["skills"]["evaluation_status"] == "error"
        assert update["skills"]["evaluation_error"] == (
            "RuntimeError: skill evaluation failed"
        )
        assert "secret" not in str(update)
        failed_rows = manager.store.list_evaluations(record.skill_id)
        assert failed_rows[0].status == "error"
        assert failed_rows[0].error == "RuntimeError: skill evaluation failed"

        provider = ProviderStub()
        middleware = SkillEvaluationMiddleware(_evaluator(manager, provider))
        async_update = asyncio.run(
            middleware.aafter_agent(state, None)  # type: ignore[arg-type]
        )
        assert async_update is not None
        assert async_update["skills"]["evaluation_status"] == "completed"
        assert manager.store.list_evaluations(record.skill_id)
    finally:
        manager.close()


def test_agent_assembles_skill_shadow_evaluation_after_run(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        provider = ProviderStub()
        graph = build_agent(
            FakeMessagesListChatModel(responses=[AIMessage(content="完成")]),
            skill_manager=manager,
            skill_config=manager.config,
            skill_overrides=("verify",),
            evaluation_provider=provider,
            evaluation_config=EvaluationConfig(use="jev", api_key="test"),
        )

        state = graph.invoke(create_initial_state("核验这个结论"))

        skill_id = state["skills"]["selected"][0]["skill_id"]
        assert state["skills"]["evaluation_status"] == "completed"
        assert state["skills"]["evaluation_count"] == 1
        assert manager.store.list_evaluations(skill_id)[0].helpful == 0.9
        assert provider.calls == 1
    finally:
        manager.close()


def test_agent_requires_provider_when_skill_evaluation_is_enabled(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        with pytest.raises(ValueError, match="评估 Provider"):
            build_agent(
                FakeMessagesListChatModel(responses=[AIMessage(content="完成")]),
                skill_manager=manager,
                skill_config=manager.config,
            )
    finally:
        manager.close()


def test_evaluation_skips_skills_dropped_by_context_budget(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        state, record = _state(manager)
        state["skills"]["dropped"] = [
            {**state["skills"]["selected"][0], "drop_reason": "token_budget"}
        ]
        provider = ProviderStub()
        middleware = SkillEvaluationMiddleware(_evaluator(manager, provider))

        assert middleware.after_agent(state, None) is None  # type: ignore[arg-type]
        assert provider.calls == 0
        assert manager.store.list_evaluations(record.skill_id) == []
    finally:
        manager.close()
