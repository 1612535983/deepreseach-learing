"""Evaluate completed reports without exposing unbounded Agent state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
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
        if self._previous_report_state(state).get("signature") == signature:
            return None
        return signature

    def _completed_patch(
        self,
        state: ResearchState,
        result: ReportEvaluationResult,
    ) -> dict[str, Any]:
        previous = self._previous_report_state(state)
        return {
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
                    "runtime_action": "observed",
                    "evaluation_count": int(previous.get("evaluation_count") or 0)
                    + 1,
                    "input_chars": result.input_chars,
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cost_usd": result.usage.cost_usd,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": None,
                    "notes": list(result.notes),
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
        try:
            result = self._evaluator.evaluate(state)
        except Exception as exc:
            return self._error_patch(state, signature, exc)
        return self._completed_patch(state, result)

    @override
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
        try:
            result = await self._evaluator.aevaluate(state)
        except Exception as exc:
            return self._error_patch(state, signature, exc)
        return self._completed_patch(state, result)
