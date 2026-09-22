from __future__ import annotations

import pytest

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.state import create_initial_state, merge_evaluation_state


def test_evaluation_is_disabled_and_shadowed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPRESEARCH_EVALUATION_USE", raising=False)
    monkeypatch.delenv("DEEPRESEARCH_JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    config = EvaluationConfig.from_env()

    assert config.enabled is False
    assert config.mode == "shadow"
    assert config.model == "jev-latest"


def test_evaluation_config_reads_jev_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPRESEARCH_EVALUATION_USE", "jev")
    monkeypatch.setenv("DEEPRESEARCH_EVALUATION_MODE", "gate")
    monkeypatch.setenv("DEEPRESEARCH_JEV_API_KEY", "secret")
    monkeypatch.setenv("DEEPRESEARCH_EVALUATION_MAX_EVIDENCE_ITEMS", "5")
    monkeypatch.setenv(
        "DEEPRESEARCH_EVALUATION_EVIDENCE_SUFFICIENCY_THRESHOLD", "0.82"
    )

    config = EvaluationConfig.from_env()

    assert config.enabled is True
    assert config.mode == "gate"
    assert config.api_key == "secret"
    assert config.max_evidence_items == 5
    assert config.evidence_sufficiency_threshold == 0.82
    assert "secret" not in repr(config)


def test_enabled_evaluation_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPRESEARCH_EVALUATION_USE", "jev")
    monkeypatch.delenv("DEEPRESEARCH_JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API_KEY"):
        EvaluationConfig.from_env()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use": "unknown"},
        {"mode": "active"},
        {"timeout_seconds": 0},
        {"max_retries": -1},
        {"max_payload_chars": 0},
        {"evidence_support_threshold": 1.1},
    ],
)
def test_evaluation_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


def test_evaluation_state_is_complete_and_deep_merges() -> None:
    state = create_initial_state("问题")

    merged = merge_evaluation_state(
        state["evaluation"],
        {"report": {"status": "completed", "evaluation_count": 1}},
    )

    assert merged["report"]["status"] == "completed"
    assert merged["report"]["evaluation_count"] == 1
    assert merged["report"]["answers"] == {}
