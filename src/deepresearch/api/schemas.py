"""Stable, bounded JSON contracts exposed by the Web API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunStatus(StrEnum):
    """Lifecycle states understood by the browser."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CreateRunRequest(BaseModel):
    """User input required to start one research task."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=10_000)
    skill_overrides: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("研究问题不能为空。")
        return normalized

    @field_validator("skill_overrides")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            name = value.strip()
            if name and name not in normalized:
                normalized.append(name)
        return normalized


class RunAcceptedResponse(BaseModel):
    thread_id: str
    status: RunStatus


class PlanStepResponse(BaseModel):
    step_id: str
    title: str
    status: str


class PlanResponse(BaseModel):
    goal: str
    steps: list[PlanStepResponse] = Field(default_factory=list)


class SourceResponse(BaseModel):
    title: str
    url: str
    snippet: str = ""
    query: str = ""


class SkillSelectionResponse(BaseModel):
    skill_id: str
    name: str
    score: float = 0.0
    reason: str = ""
    forced: bool = False


class EvaluationResponse(BaseModel):
    status: str = "not_run"
    mode: str = "shadow"
    composite_score: float | None = None
    recommended_action: str | None = None
    runtime_action: str | None = None
    gate_attempts: int = 0
    provider: str | None = None
    model: str | None = None
    latency_ms: int = 0
    cost_usd: float | None = None
    last_error: str | None = None
    notes: list[str] = Field(default_factory=list)
    answers: dict[str, dict[str, Any]] = Field(default_factory=dict)


class GovernanceResponse(BaseModel):
    current_tokens: int = 0
    window_tokens: int = 0
    utilization_ratio: float = 0.0
    model_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    finalization_active: bool = False
    finalization_reason: str | None = None


class RunProgressResponse(BaseModel):
    search_count: int = 0
    page_read_count: int = 0
    source_count: int = 0
    evidence_count: int = 0
    reflection_attempts: int = 0


class ResearchEventResponse(BaseModel):
    id: int
    event_type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    node: str | None = None
    created_at: datetime


class RunDetailResponse(BaseModel):
    """Public task view; intentionally excludes messages and evidence bodies."""

    thread_id: str
    question: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    current_step_id: str | None = None
    research_gaps: list[str] = Field(default_factory=list)
    plan: PlanResponse | None = None
    progress: RunProgressResponse = Field(default_factory=RunProgressResponse)
    sources: list[SourceResponse] = Field(default_factory=list)
    selected_skills: list[SkillSelectionResponse] = Field(default_factory=list)
    evaluation: EvaluationResponse = Field(default_factory=EvaluationResponse)
    governance: GovernanceResponse = Field(default_factory=GovernanceResponse)
    final_report: str | None = None
    final_report_source_urls: list[str] = Field(default_factory=list)


class SkillResponse(BaseModel):
    skill_id: str
    name: str
    description: str
    version: int
    origin: str
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    total_selections: int = 0
    completion_rate: float = 0.0


def run_detail_from_state(
    *,
    thread_id: str,
    question: str,
    status: RunStatus,
    state: Mapping[str, Any] | None,
    created_at: datetime,
    updated_at: datetime,
    error: str | None = None,
) -> RunDetailResponse:
    """Project internal state into a small, JSON-safe browser snapshot."""

    values = state or {}
    raw_plan = _mapping(values.get("plan"))
    plan = None
    if raw_plan:
        plan = PlanResponse(
            goal=str(raw_plan.get("goal") or ""),
            steps=[
                PlanStepResponse(
                    step_id=str(step.get("step_id") or ""),
                    title=str(step.get("title") or ""),
                    status=str(step.get("status") or "pending"),
                )
                for raw_step in _list(raw_plan.get("steps"))
                if (step := _mapping(raw_step))
            ],
        )

    sources = [
        SourceResponse(
            title=str(source.get("title") or ""),
            url=str(source.get("url") or ""),
            snippet=str(source.get("snippet") or ""),
            query=str(source.get("query") or ""),
        )
        for raw_source in _list(values.get("sources"))
        if (source := _mapping(raw_source)) and source.get("url")
    ]

    skill_state = _mapping(values.get("skills"))
    selected_skills = [
        SkillSelectionResponse(
            skill_id=str(skill.get("skill_id") or ""),
            name=str(skill.get("name") or ""),
            score=_float(skill.get("score")),
            reason=str(skill.get("reason") or ""),
            forced=skill.get("forced") is True,
        )
        for raw_skill in _list(skill_state.get("selected"))
        if (skill := _mapping(raw_skill)) and skill.get("name")
    ]

    report_evaluation = _mapping(_mapping(values.get("evaluation")).get("report"))
    evaluation = EvaluationResponse(
        status=str(report_evaluation.get("status") or "not_run"),
        mode=str(report_evaluation.get("mode") or "shadow"),
        composite_score=_optional_float(report_evaluation.get("composite_score")),
        recommended_action=_optional_str(
            report_evaluation.get("recommended_action")
        ),
        runtime_action=_optional_str(report_evaluation.get("runtime_action")),
        gate_attempts=_int(report_evaluation.get("gate_attempts")),
        provider=_optional_str(report_evaluation.get("provider")),
        model=_optional_str(report_evaluation.get("model")),
        latency_ms=_int(report_evaluation.get("latency_ms")),
        cost_usd=_optional_float(report_evaluation.get("cost_usd")),
        last_error=_optional_str(report_evaluation.get("last_error")),
        notes=[str(note) for note in _list(report_evaluation.get("notes"))],
        answers={
            str(key): dict(answer)
            for key, raw_answer in _mapping(
                report_evaluation.get("answers")
            ).items()
            if isinstance(raw_answer, Mapping)
            for answer in [raw_answer]
        },
    )

    context = _mapping(_mapping(values.get("governance")).get("context"))
    budget = _mapping(context.get("budget"))
    usage = _mapping(context.get("cumulative_usage"))
    finalization = _mapping(context.get("finalization"))
    governance = GovernanceResponse(
        current_tokens=_int(budget.get("current_tokens")),
        window_tokens=_int(budget.get("window_tokens")),
        utilization_ratio=_float(budget.get("utilization_ratio")),
        model_call_count=_int(context.get("model_call_count")),
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        finalization_active=finalization.get("active") is True,
        finalization_reason=_optional_str(
            finalization.get("last_reason") or finalization.get("trigger_reason")
        ),
    )

    return RunDetailResponse(
        thread_id=thread_id,
        question=question,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        error=error,
        current_step_id=_optional_str(values.get("current_step_id")),
        research_gaps=[str(gap) for gap in _list(values.get("research_gaps"))],
        plan=plan,
        progress=RunProgressResponse(
            search_count=len(_list(values.get("search_records"))),
            page_read_count=len(_list(values.get("page_records"))),
            source_count=len(sources),
            evidence_count=len(_list(values.get("observations"))),
            reflection_attempts=_int(values.get("reflection_attempts")),
        ),
        sources=sources,
        selected_skills=selected_skills,
        evaluation=evaluation,
        governance=governance,
        final_report=_optional_str(values.get("final_report")),
        final_report_source_urls=[
            str(url) for url in _list(values.get("final_report_source_urls"))
        ],
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    return None if value is None else _float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
