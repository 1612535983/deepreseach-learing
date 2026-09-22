"""Typed, checkpoint-safe contracts for probabilistic decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict


QuestionType = Literal["noul", "choice", "score"]
RecommendedAction = Literal[
    "pass",
    "revise_report",
    "continue_research",
    "review",
]
EvaluationStatus = Literal["not_run", "completed", "error"]


@dataclass(frozen=True)
class DecisionQuestion:
    """One atomic question evaluated against a shared provider state."""

    question_type: QuestionType
    instructions: str
    criteria: dict[str, str] | tuple[str, ...] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.question_type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = (
                dict(self.criteria)
                if isinstance(self.criteria, dict)
                else list(self.criteria)
            )
        return payload


@dataclass(frozen=True)
class DecisionAnswer:
    """Normalized provider answer without leaking provider-specific classes."""

    answer_type: QuestionType
    value: str | float
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    legend: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


@dataclass(frozen=True)
class DecisionResponse:
    """One validated batch of typed decisions from a provider."""

    provider: str
    model: str
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage = field(default_factory=DecisionUsage)
    latency_ms: int = 0


class ReportEvaluationState(TypedDict, total=False):
    """Serializable evaluation record for the latest final-report version."""

    status: EvaluationStatus
    mode: Literal["shadow", "gate"]
    signature: str | None
    report_hash: str | None
    evidence_hash: str | None
    provider: str | None
    model: str | None
    answers: dict[str, dict[str, Any]]
    composite_score: float | None
    recommended_action: RecommendedAction | None
    runtime_action: str | None
    evaluation_count: int
    gate_attempts: int
    input_chars: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    evaluated_at: str | None
    last_error: str | None
    notes: NotRequired[list[str]]


class EvaluationState(TypedDict, total=False):
    """Top-level namespace reserved for current and future evaluators."""

    report: ReportEvaluationState
