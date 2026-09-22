"""Evaluate completed reports without exposing unbounded Agent state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.report import (
    ReportEvaluator,
    build_report_evaluation_payload,
)
from deepresearch.evaluation.types import ReportEvaluationResult
from deepresearch.quality import assess_research
from deepresearch.state import ResearchState


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _last_ai_message(state: ResearchState) -> AIMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


class ReportEvaluationMiddleware(AgentMiddleware):
    """Run one idempotent report evaluation after deterministic checks pass."""

    state_schema = ResearchState

    def __init__(
        self,
        evaluator: ReportEvaluator,
        config: EvaluationConfig,
    ) -> None:
        self._evaluator = evaluator
        self._config = config

    def _candidate_signature(self, state: ResearchState) -> str | None:
        last_message = _last_ai_message(state)
        if last_message is None or last_message.tool_calls:
            return None
        if assess_research(state):
            return None
        return build_report_evaluation_payload(state, self._config).signature

    def _previous_report_state(self, state: ResearchState) -> Mapping[str, Any]:
        return _mapping(_mapping(state.get("evaluation")).get("report"))

    def _should_evaluate(self, state: ResearchState) -> str | None:
        signature = self._candidate_signature(state)
        if signature is None:
            return None
        return signature

    @staticmethod
    def _p5_active(state: ResearchState) -> bool:
        context = _mapping(_mapping(state.get("governance")).get("context"))
        pending = context.get("pending_stages")
        finalization = _mapping(context.get("finalization"))
        return (
            isinstance(pending, (list, tuple))
            and "P5" in pending
        ) or finalization.get("active") is True

    @staticmethod
    def _probability(result: ReportEvaluationResult, name: str) -> float:
        answer = _mapping(result.answers.get(name))
        value = answer.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return 0.0

    def _runtime_policy(
        self,
        state: ResearchState,
        result: ReportEvaluationResult,
    ) -> tuple[str, int, list[str], list[str], bool]:
        """Return action, attempts, gaps, notes, and whether to jump."""

        previous = self._previous_report_state(state)
        attempts = int(previous.get("gate_attempts") or 0)
        notes = list(result.notes)
        if self._config.mode == "shadow":
            return "observed", attempts, [], notes, False
        if result.recommended_action == "pass":
            return "pass", attempts, [], notes, False
        if result.recommended_action == "review":
            notes.append("human_review_recommended")
            return "review_required", attempts, [], notes, False
        if self._p5_active(state):
            notes.append("p5_forced_finalization")
            return "p5_bypass", attempts, [], notes, False
        if attempts >= self._config.max_gate_attempts:
            notes.append("gate_attempt_limit_reached")
            return "gate_exhausted", attempts, [], notes, False

        next_attempts = attempts + 1
        if result.recommended_action == "continue_research":
            enough = self._probability(result, "evidence_sufficient")
            continuation = self._probability(result, "continue_research")
            gap = (
                "Jev 语义检查认为证据仍不足："
                f"证据足够概率 {enough:.1%}，继续研究概率 {continuation:.1%}。"
                "请继续搜索并读取能直接支持结论的来源，然后重新生成报告。"
            )
            return "continue_research", next_attempts, [gap], notes, True

        relevance = self._probability(result, "answer_relevance")
        support = self._probability(result, "evidence_support")
        citations = self._probability(result, "citation_coverage")
        gap = (
            "Jev 语义检查要求修改报告："
            f"相关概率 {relevance:.1%}，证据支持概率 {support:.1%}，"
            f"引用充分概率 {citations:.1%}。"
            "请优先使用现有证据修订内容和引用，并再次调用 write_final_report。"
        )
        return "revise_report", next_attempts, [gap], notes, True

    def _completed_patch(
        self,
        state: ResearchState,
        result: ReportEvaluationResult,
    ) -> dict[str, Any]:
        previous = self._previous_report_state(state)
        runtime_action, gate_attempts, gaps, notes, should_jump = (
            self._runtime_policy(state, result)
        )
        patch: dict[str, Any] = {
            "evaluation": {
                "report": {
                    "status": "completed",
                    "mode": self._config.mode,
                    "signature": result.signature,
                    "report_hash": result.report_hash,
                    "evidence_hash": result.evidence_hash,
                    "provider": result.provider,
                    "model": result.model,
                    "answers": result.answers,
                    "composite_score": result.composite_score,
                    "recommended_action": result.recommended_action,
                    "runtime_action": runtime_action,
                    "evaluation_count": int(previous.get("evaluation_count") or 0)
                    + 1,
                    "gate_attempts": gate_attempts,
                    "input_chars": result.input_chars,
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cost_usd": result.usage.cost_usd,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": None,
                    "notes": notes,
                }
            }
        }
        if gaps:
            patch["research_gaps"] = gaps
        if should_jump:
            patch["jump_to"] = "model"
        return patch

    def _duplicate_patch(self, state: ResearchState) -> dict[str, Any] | None:
        """Stop a gate loop when the model leaves report and evidence unchanged."""

        previous = self._previous_report_state(state)
        if self._config.mode != "gate" or previous.get("runtime_action") not in {
            "continue_research",
            "revise_report",
        }:
            return None
        previous_notes = previous.get("notes")
        notes = list(previous_notes) if isinstance(previous_notes, list) else []
        notes.extend(["unchanged_evaluation_signature", "gate_attempt_limit_reached"])
        return {
            "evaluation": {
                "report": {
                    "runtime_action": "gate_exhausted",
                    "notes": list(dict.fromkeys(notes)),
                }
            }
        }

    def _error_patch(
        self,
        state: ResearchState,
        signature: str,
        exc: Exception,
    ) -> dict[str, Any]:
        previous = self._previous_report_state(state)
        return {
            "evaluation": {
                "report": {
                    "status": "error",
                    "mode": self._config.mode,
                    "signature": signature,
                    "runtime_action": "fail_open",
                    "evaluation_count": int(previous.get("evaluation_count") or 0)
                    + 1,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": f"{type(exc).__name__}: evaluation failed",
                }
            }
        }

    @override
    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Evaluate synchronously; provider failures never fail the research run."""

        try:
            signature = self._should_evaluate(state)
        except Exception as exc:
            return self._error_patch(state, "payload-error", exc)
        if signature is None:
            return None
        if self._previous_report_state(state).get("signature") == signature:
            return self._duplicate_patch(state)
        try:
            result = self._evaluator.evaluate(state)
        except Exception as exc:
            return self._error_patch(state, signature, exc)
        return self._completed_patch(state, result)

    @override
    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Evaluate asynchronously without blocking the event loop."""

        try:
            signature = self._should_evaluate(state)
        except Exception as exc:
            return self._error_patch(state, "payload-error", exc)
        if signature is None:
            return None
        if self._previous_report_state(state).get("signature") == signature:
            return self._duplicate_patch(state)
        try:
            result = await self._evaluator.aevaluate(state)
        except Exception as exc:
            return self._error_patch(state, signature, exc)
        return self._completed_patch(state, result)
