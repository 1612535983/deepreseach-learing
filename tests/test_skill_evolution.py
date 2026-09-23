from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
)
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.evolution import (
    CANDIDATE_REVIEW_QUESTIONS,
    SkillEvolutionService,
)
from deepresearch.skill.manager import SkillManager


class ReviewProviderStub:
    name = "stub"

    def __init__(self, *, regression: float = 0.1) -> None:
        self.regression = regression
        self.calls = 0

    def evaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        assert set(questions) == set(CANDIDATE_REVIEW_QUESTIONS)
        return DecisionResponse(
            provider="stub",
            model="judge-v1",
            answers={
                "addresses_reason": DecisionAnswer("noul", 0.92),
                "preserves_usefulness": DecisionAnswer("noul", 0.9),
                "instruction_clarity": DecisionAnswer(
                    "score",
                    2.8,
                    probabilities={"0": 0.0, "1": 0.05, "2": 0.2, "3": 0.75},
                    confidence=0.9,
                ),
                "regression_risk": DecisionAnswer("noul", self.regression),
            },
            usage=DecisionUsage(input_tokens=80, output_tokens=12),
            latency_ms=20,
        )

    async def aevaluate(self, state, questions):  # noqa: ANN001, ANN201
        return self.evaluate(state, questions)


def _manager(tmp_path: Path, *, max_changed_lines: int = 20) -> SkillManager:
    root = tmp_path / "skills" / "verify"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        """---
name: verify
description: 核验关键来源
version: 1
tags: [verification]
tools: [read_page]
---
## 步骤

1. 读取来源。
2. 给出结论。
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
            evolution_max_changed_lines=max_changed_lines,
        )
    )
    manager.load_startup()
    return manager


def _candidate_service(manager: SkillManager, body: str) -> SkillEvolutionService:
    model = FakeMessagesListChatModel(responses=[AIMessage(content=body)])
    return SkillEvolutionService(manager.store, manager.config, model=model)


def test_candidate_is_inactive_versioned_and_auditable(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        baseline = manager.list_skills()[0]
        service = _candidate_service(
            manager,
            """## 步骤

1. 优先读取一手来源。
2. 为每个关键结论附上来源 URL。
""",
        )
        experiment = service.create_candidate(
            "verify", reason="原流程没有要求一手来源和逐项引用"
        )
        candidate = manager.store.get(experiment.candidate_skill_id)

        assert candidate is not None
        assert candidate.is_active is False
        assert candidate.version == 2
        assert candidate.lineage.parent_skill_ids == (baseline.skill_id,)
        assert candidate.lineage.origin == "DERIVED"
        assert manager.store.get_active("verify").skill_id == baseline.skill_id  # type: ignore[union-attr]
        assert experiment.status == "candidate"
        assert experiment.changed_lines > 0
        assert "baseline/SKILL.md" in experiment.mutation_diff
    finally:
        manager.close()


def test_review_then_explicit_promote_and_rollback(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        baseline = manager.list_skills()[0]
        experiment = _candidate_service(
            manager,
            """## 步骤

1. 优先读取一手来源。
2. 为每个关键结论附上来源 URL。
""",
        ).create_candidate("verify", reason="提升来源核验质量")
        provider = ReviewProviderStub()
        service = SkillEvolutionService(
            manager.store,
            manager.config,
            provider=provider,
            evaluation_config=EvaluationConfig(use="jev", api_key="test"),
        )

        reviewed = service.review_candidate(experiment.candidate_skill_id)

        assert reviewed.status == "reviewed"
        assert reviewed.recommendation == "approve"
        assert reviewed.rule_passed is True
        assert reviewed.score is not None and reviewed.score >= 0.7
        assert reviewed.input_chars > 0
        assert reviewed.input_tokens == 80
        assert reviewed.output_tokens == 12
        assert manager.store.get_active("verify").skill_id == baseline.skill_id  # type: ignore[union-attr]

        promoted = service.promote(experiment.candidate_skill_id)
        assert promoted.status == "promoted"
        assert manager.store.get_active("verify").skill_id == experiment.candidate_skill_id  # type: ignore[union-attr]

        manager.store.rollback(baseline.skill_id)
        assert manager.store.get_active("verify").skill_id == baseline.skill_id  # type: ignore[union-attr]
    finally:
        manager.close()


def test_candidate_change_budget_and_promotion_gate_fail_closed(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path, max_changed_lines=2)
    try:
        service = _candidate_service(
            manager,
            """完全不同的第一行
完全不同的第二行
完全不同的第三行
""",
        )
        with pytest.raises(ValueError, match="修改超过预算"):
            service.create_candidate("verify", reason="大幅重写")
    finally:
        manager.close()

    manager = _manager(tmp_path / "second")
    try:
        experiment = _candidate_service(
            manager,
            """## 步骤

1. 优先读取一手来源。
2. 为每个关键结论附上来源 URL。
""",
        ).create_candidate("verify", reason="提升核验质量")
        with pytest.raises(ValueError, match="概率评估 Provider"):
            SkillEvolutionService(
                manager.store, manager.config
            ).review_candidate(experiment.candidate_skill_id)
        assert manager.store.get_experiment(experiment.experiment_id).status == "candidate"  # type: ignore[union-attr]
        with pytest.raises(Exception, match="尚未通过评审"):
            manager.store.promote_candidate(experiment.candidate_skill_id)
    finally:
        manager.close()


def test_high_regression_risk_rejects_candidate(tmp_path) -> None:  # noqa: ANN001
    manager = _manager(tmp_path)
    try:
        experiment = _candidate_service(
            manager,
            """## 步骤

1. 优先读取一手来源。
2. 为每个关键结论附上来源 URL。
""",
        ).create_candidate("verify", reason="提升核验质量")
        service = SkillEvolutionService(
            manager.store,
            manager.config,
            provider=ReviewProviderStub(regression=0.9),
            evaluation_config=EvaluationConfig(use="jev", api_key="test"),
        )

        reviewed = service.review_candidate(experiment.candidate_skill_id)

        assert reviewed.status == "rejected"
        assert reviewed.recommendation == "reject"
        assert "regression_risk_high" in reviewed.notes
        with pytest.raises(Exception, match="尚未通过评审"):
            service.promote(experiment.candidate_skill_id)
    finally:
        manager.close()
