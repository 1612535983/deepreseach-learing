from __future__ import annotations

import pytest

from deepresearch.skill.config import SkillConfig
from deepresearch.skill.types import SkillRecord


def test_skill_config_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPRESEARCH_SKILL_USE", raising=False)

    config = SkillConfig.from_env()

    assert config.enabled is False
    assert config.max_skills == 3
    assert config.token_budget == 2_500
    assert config.include_builtin is True
    assert config.enable_evaluation is False
    assert config.evolution_max_changed_lines == 20


def test_skill_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPRESEARCH_SKILL_USE", "default")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_DIRS", "one,two")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_MAX_SKILLS", "2")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_TOKEN_BUDGET", "900")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_MIN_RELEVANCE", "0.2")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_ENABLE_METRICS", "false")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_ENABLE_EVALUATION", "true")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_EVOLUTION_MAX_CHANGED_LINES", "12")
    monkeypatch.setenv("DEEPRESEARCH_SKILL_PROMOTION_THRESHOLD", "0.8")

    config = SkillConfig.from_env()

    assert config.enabled is True
    assert config.skill_dirs == ("one", "two")
    assert config.max_skills == 2
    assert config.token_budget == 900
    assert config.min_relevance == 0.2
    assert config.enable_metrics is False
    assert config.enable_evaluation is True
    assert config.evolution_max_changed_lines == 12
    assert config.promotion_threshold == 0.8


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("DEEPRESEARCH_SKILL_MAX_SKILLS", "zero", "必须是整数"),
        ("DEEPRESEARCH_SKILL_MIN_RELEVANCE", "high", "必须是数字"),
        ("DEEPRESEARCH_SKILL_INCLUDE_BUILTIN", "sometimes", "true 或 false"),
    ],
)
def test_skill_config_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        SkillConfig.from_env()


def test_skill_record_rates_protect_zero_division() -> None:
    record = SkillRecord(
        skill_id="verify__abc",
        name="verify",
        description="Verify sources",
        source_path="verify/SKILL.md",
        object_path="objects/abc.md",
        content_hash="abc",
    )

    assert record.injection_rate == 0.0
    assert record.completion_rate == 0.0
