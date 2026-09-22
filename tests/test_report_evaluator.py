from __future__ import annotations

import asyncio
import json

import pytest

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.report import (
    REPORT_QUESTIONS,
    ReportEvaluator,
    build_report_evaluation_payload,
)
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
)
from deepresearch.state import create_initial_state


URL_A = "https://a.example/policy"
URL_B = "https://b.example/paper"


def _state():  # noqa: ANN202
    state = create_initial_state("政策支持有哪些？")
    state["final_report"] = (
        f"# 报告\n\n政策提供研发支持。[{URL_A}]({URL_A})\n\n"
        f"另有产业支持。[{URL_B}]({URL_B})"
    )
    state["final_report_source_urls"] = [URL_A, URL_B]
    state["sources"] = [
        {"title": "政策原文", "url": URL_A, "snippet": "", "query": "政策"},
        {"title": "研究论文", "url": URL_B, "snippet": "", "query": "研究"},
        {
            "title": "未引用页面",
            "url": "https://unused.example",
            "snippet": "",
            "query": "其他",
        },
    ]
    state["observations"] = [
        {"content": "A" * 2_000, "source_url": URL_A, "query": "政策"},
        {"content": "证据 B", "source_url": URL_B, "query": "研究"},
        {
            "content": "不应发送",
            "source_url": "https://unused.example",
            "query": "其他",
        },
    ]
    state["memory"]["recalled"] = [
        {"id": "secret-memory", "score": 1.0, "strength": 1.0}
    ]
    return state


def _answer_values(**updates: float) -> dict[str, float]:
    values = {
        "answer_relevance": 0.9,
        "evidence_support": 0.85,
        "citation_coverage": 0.8,
        "evidence_sufficient": 0.9,
        "continue_research": 0.1,
    }
    values.update(updates)
    return values


class ProviderStub:
    name = "stub"

    def __init__(
        self,
        values: dict[str, float] | None = None,
        *,
        quality: float = 2.5,
        quality_confidence: float = 0.9,
    ) -> None:
        self.values = values or _answer_values()
        self.quality = quality
        self.quality_confidence = quality_confidence
        self.calls = []

    def _response(self) -> DecisionResponse:
        answers = {
            name: DecisionAnswer("noul", value)
            for name, value in self.values.items()
        }
        answers["source_quality"] = DecisionAnswer(
            "score",
            self.quality,
            probabilities={"0": 0.0, "1": 0.1, "2": 0.3, "3": 0.6},
            confidence=self.quality_confidence,
        )
        return DecisionResponse(
            provider="stub",
            model="stub-v1",
            answers=answers,
            usage=DecisionUsage(input_tokens=100, output_tokens=10),
            latency_ms=25,
        )

    def evaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls.append((state, questions))
        return self._response()

    async def aevaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls.append((state, questions))
        return self._response()


def _config(**updates) -> EvaluationConfig:  # noqa: ANN003
    values = {
        "use": "jev",
        "api_key": "test",
        "max_payload_chars": 8_000,
        "max_report_chars": 3_000,
        "max_evidence_chars": 200,
    }
    values.update(updates)
    return EvaluationConfig(**values)


def test_payload_is_bounded_and_excludes_unrelated_state() -> None:
    payload = build_report_evaluation_payload(_state(), _config())
    serialized = json.dumps(
        payload.state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert len(serialized) == payload.input_chars
    assert payload.input_chars <= 8_000
    assert len(payload.state["evidence"][0]["content"]) <= 200
    assert {item["url"] for item in payload.state["cited_sources"]} == {
        URL_A,
        URL_B,
    }
    assert "unused.example" not in serialized
    assert "secret-memory" not in serialized
    assert "messages" not in payload.state
    assert "memory" not in payload.state
    assert "evidence_truncated" in payload.notes


def test_payload_recovers_provenance_for_old_checkpoint() -> None:
    state = _state()
    state["final_report_source_urls"] = []

    payload = build_report_evaluation_payload(state, _config())

    assert [item["url"] for item in payload.state["cited_sources"]] == [URL_A, URL_B]


def test_payload_signature_changes_with_report_or_evidence() -> None:
    state = _state()
    first = build_report_evaluation_payload(state, _config())
    state["final_report"] += "\n新结论"
    report_changed = build_report_evaluation_payload(state, _config())
    state["observations"][0]["content"] += "新证据"
    evidence_changed = build_report_evaluation_payload(state, _config())

    assert first.report_hash != report_changed.report_hash
    assert first.evidence_hash == report_changed.evidence_hash
    assert report_changed.evidence_hash != evidence_changed.evidence_hash
    assert len({first.signature, report_changed.signature, evidence_changed.signature}) == 3


def test_evaluator_batches_questions_and_recommends_pass() -> None:
    provider = ProviderStub()
    result = ReportEvaluator(provider, _config()).evaluate(_state())

    assert len(provider.calls) == 1
    assert set(provider.calls[0][1]) == set(REPORT_QUESTIONS)
    assert result.recommended_action == "pass"
    assert result.composite_score == pytest.approx(0.858333, abs=1e-6)
    assert result.answers["evidence_support"]["value"] == 0.85
    assert result.usage.input_tokens == 100
    assert result.latency_ms == 25


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            _answer_values(evidence_sufficient=0.4, continue_research=0.9),
            "continue_research",
        ),
        (_answer_values(evidence_support=0.4), "revise_report"),
        (
            _answer_values(evidence_sufficient=0.4, continue_research=0.3),
            "review",
        ),
    ],
)
def test_evaluator_composes_recommendations(values, expected) -> None:  # noqa: ANN001
    result = ReportEvaluator(ProviderStub(values), _config()).evaluate(_state())

    assert result.recommended_action == expected


def test_low_score_confidence_routes_to_review() -> None:
    provider = ProviderStub(quality_confidence=0.2)

    result = ReportEvaluator(provider, _config()).evaluate(_state())

    assert result.recommended_action == "review"
    assert "source_quality_low_confidence" in result.notes


def test_async_evaluator_matches_sync() -> None:
    provider = ProviderStub()

    result = asyncio.run(ReportEvaluator(provider, _config()).aevaluate(_state()))

    assert result.recommended_action == "pass"
    assert len(provider.calls) == 1


def test_payload_requires_report() -> None:
    with pytest.raises(ValueError, match="final_report"):
        build_report_evaluation_payload(create_initial_state("问题"), _config())
