"""Bounded report projection and deterministic composition of Jev answers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.provider import DecisionProvider
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    RecommendedAction,
    ReportEvaluationPayload,
    ReportEvaluationResult,
)
from deepresearch.state import ResearchState


REPORT_EVALUATION_SCHEMA = "report-evaluation-v1"
REPORT_QUESTIONS: dict[str, DecisionQuestion] = {
    "answer_relevance": DecisionQuestion(
        "noul",
        "Does `report` directly answer the user's `research_question`?",
    ),
    "evidence_support": DecisionQuestion(
        "noul",
        (
            "Does `evidence` directly support the main factual conclusions in "
            "`report`, without relying on unsupported inference?"
        ),
    ),
    "citation_coverage": DecisionQuestion(
        "noul",
        (
            "Do the citations in `report` adequately cover its important factual "
            "claims, using URLs present in `cited_sources`?"
        ),
    ),
    "evidence_sufficient": DecisionQuestion(
        "noul",
        (
            "Is the currently supplied `evidence` sufficient to answer "
            "`research_question` responsibly?"
        ),
    ),
    "continue_research": DecisionQuestion(
        "noul",
        (
            "Would additional web research likely be necessary to answer "
            "`research_question` responsibly?"
        ),
    ),
    "source_quality": DecisionQuestion(
        "score",
        "Rate the overall quality and directness of `cited_sources` for this report.",
        (
            "Mostly unreliable, irrelevant, or indirect sources.",
            "Mixed quality with substantial indirect or weak sourcing.",
            "Mostly reliable sources with reasonable direct support.",
            "Mostly authoritative primary sources or original evidence.",
        ),
    ),
}


def _clip(text: object, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 0:
        return ""
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 2:
        return normalized[:limit]
    remaining = limit - len(marker)
    head = (remaining * 2) // 3
    tail = remaining - head
    return f"{normalized[:head]}{marker}{normalized[-tail:]}"


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_report_evaluation_payload(
    state: ResearchState,
    config: EvaluationConfig,
) -> ReportEvaluationPayload:
    """Project only bounded report provenance; never include messages or memory."""

    report = str(state.get("final_report") or "").strip()
    if not report:
        raise ValueError("没有 final_report，无法构造报告评估输入。")
    question = _clip(state.get("research_question"), 2_000)
    known_sources = {
        str(source.get("url") or ""): source
        for source in state.get("sources", [])
        if isinstance(source, Mapping) and source.get("url")
    }
    explicit_urls = [
        str(url).strip()
        for url in state.get("final_report_source_urls", [])
        if str(url).strip()
    ]
    # Old checkpoints predate explicit report provenance. Recover only URLs that
    # were collected by this run and are literally present in the saved report.
    cited_urls = list(
        dict.fromkeys(
            explicit_urls
            or [url for url in known_sources if url and url in report]
        )
    )
    cited_sources = [
        {
            "title": _clip(known_sources.get(url, {}).get("title"), 300),
            "url": url,
        }
        for url in cited_urls[: config.max_evidence_items]
    ]
    relevant_observations = [
        observation
        for observation in state.get("observations", [])
        if isinstance(observation, Mapping)
        and (
            not cited_urls
            or str(observation.get("source_url") or "") in cited_urls
        )
    ]
    evidence_fingerprint = [
        {
            "source_url": str(item.get("source_url") or ""),
            "content": str(item.get("content") or ""),
        }
        for item in relevant_observations
    ]
    report_hash = _hash_text(report)
    evidence_hash = _hash_text(_stable_json(evidence_fingerprint))
    signature = _hash_text(
        f"{REPORT_EVALUATION_SCHEMA}\x00{report_hash}\x00{evidence_hash}"
    )
    plan = state.get("plan")
    steps = plan.get("steps", []) if isinstance(plan, Mapping) else []
    plan_summary = [
        {
            "step_id": _clip(step.get("step_id"), 100),
            "title": _clip(step.get("title"), 200),
            "status": _clip(step.get("status"), 40),
        }
        for step in steps
        if isinstance(step, Mapping)
    ]
    base = {
        "research_question": question,
        "report": "",
        "cited_sources": cited_sources,
        "evidence": [],
        "plan": plan_summary,
    }
    overhead = len(_stable_json(base))
    report_budget = max(
        0,
        min(config.max_report_chars, config.max_payload_chars - overhead),
    )
    bounded_report = _clip(report, report_budget)
    base["report"] = bounded_report
    notes: list[str] = []
    if bounded_report != report:
        notes.append("report_truncated")

    for item in relevant_observations[: config.max_evidence_items]:
        raw_content = str(item.get("content") or "").strip()
        bounded_content = _clip(raw_content, config.max_evidence_chars)
        if bounded_content != raw_content:
            notes.append("evidence_truncated")
        candidate = {
            "source_url": _clip(item.get("source_url"), 2_000),
            "query": _clip(item.get("query"), 300),
            "content": bounded_content,
        }
        candidate_payload = {**base, "evidence": [*base["evidence"], candidate]}
        if len(_stable_json(candidate_payload)) > config.max_payload_chars:
            notes.append("evidence_budget_exhausted")
            break
        base["evidence"].append(candidate)
    if len(base["evidence"]) < len(relevant_observations):
        notes.append("evidence_truncated")
    serialized = _stable_json(base)
    if len(serialized) > config.max_payload_chars:
        raise ValueError("报告评估基础元数据超过 max_payload_chars。")
    return ReportEvaluationPayload(
        state=base,
        report_hash=report_hash,
        evidence_hash=evidence_hash,
        signature=signature,
        input_chars=len(serialized),
        notes=tuple(dict.fromkeys(notes)),
    )


@dataclass(frozen=True)
class ReportEvaluator:
    """Ask all report questions in one provider call and compose the result."""

    provider: DecisionProvider
    config: EvaluationConfig

    def evaluate(self, state: ResearchState) -> ReportEvaluationResult:
        payload = build_report_evaluation_payload(state, self.config)
        response = self.provider.evaluate(payload.state, REPORT_QUESTIONS)
        return self._compose(payload, response)

    async def aevaluate(self, state: ResearchState) -> ReportEvaluationResult:
        payload = build_report_evaluation_payload(state, self.config)
        response = await self.provider.aevaluate(payload.state, REPORT_QUESTIONS)
        return self._compose(payload, response)

    def _compose(
        self,
        payload: ReportEvaluationPayload,
        response: DecisionResponse,
    ) -> ReportEvaluationResult:
        values = {
            name: self._noul(response.answers[name], name)
            for name in (
                "answer_relevance",
                "evidence_support",
                "citation_coverage",
                "evidence_sufficient",
                "continue_research",
            )
        }
        quality_answer = response.answers["source_quality"]
        quality = min(3.0, max(0.0, float(quality_answer.value))) / 3.0
        composite = (
            values["answer_relevance"] * 0.20
            + values["evidence_support"] * 0.30
            + values["citation_coverage"] * 0.20
            + values["evidence_sufficient"] * 0.20
            + quality * 0.10
        )
        action, action_notes = self._recommend(values, quality_answer)
        answers = {
            name: self._serialize_answer(answer)
            for name, answer in response.answers.items()
        }
        return ReportEvaluationResult(
            report_hash=payload.report_hash,
            evidence_hash=payload.evidence_hash,
            signature=payload.signature,
            provider=response.provider,
            model=response.model,
            answers=answers,
            composite_score=round(composite, 6),
            recommended_action=action,
            input_chars=payload.input_chars,
            latency_ms=response.latency_ms,
            usage=response.usage,
            notes=tuple([*payload.notes, *action_notes]),
        )

    def _recommend(
        self,
        values: Mapping[str, float],
        quality_answer: DecisionAnswer,
    ) -> tuple[RecommendedAction, list[str]]:
        notes: list[str] = []
        if (
            values["evidence_sufficient"]
            < self.config.evidence_sufficiency_threshold
        ):
            notes.append("evidence_below_threshold")
            if (
                values["continue_research"]
                >= self.config.continue_research_threshold
            ):
                return "continue_research", notes
            return "review", notes
        failed_report_dimensions = [
            name
            for name, threshold in (
                ("answer_relevance", self.config.answer_relevance_threshold),
                ("evidence_support", self.config.evidence_support_threshold),
                ("citation_coverage", self.config.citation_coverage_threshold),
            )
            if values[name] < threshold
        ]
        if failed_report_dimensions:
            notes.extend(f"{name}_below_threshold" for name in failed_report_dimensions)
            return "revise_report", notes
        if (
            quality_answer.confidence is not None
            and quality_answer.confidence < self.config.confidence_floor
        ):
            notes.append("source_quality_low_confidence")
            return "review", notes
        return "pass", notes

    @staticmethod
    def _noul(answer: DecisionAnswer, name: str) -> float:
        if answer.answer_type != "noul" or not isinstance(answer.value, float):
            raise ValueError(f"{name} 必须是 Jev noul answer。")
        return answer.value

    @staticmethod
    def _serialize_answer(answer: DecisionAnswer) -> dict[str, Any]:
        return {
            "type": answer.answer_type,
            "value": answer.value,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
            "legend": dict(answer.legend),
        }
