"""Measure context pressure around each model call and record it in State."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

from deepresearch.context.policy import (
    DEFAULT_CONTEXT_THRESHOLDS,
    ContextThresholds,
    evaluate_context_pressure,
)
from deepresearch.context.tokens import collect_usage_delta, estimate_context_tokens
from deepresearch.context.types import CumulativeTokenUsage
from deepresearch.context.windows import resolve_context_window
from deepresearch.state import ResearchState


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class ContextGovernanceMiddleware(AgentMiddleware):
    """Classify pre-call pressure and record post-call usage without routing changes."""

    state_schema = ResearchState

    def __init__(
        self,
        model: Any,
        *,
        context_window: int | None = None,
        thresholds: ContextThresholds = DEFAULT_CONTEXT_THRESHOLDS,
        message_projector: Callable[[ResearchState], Sequence[BaseMessage]] | None = None,
    ) -> None:
        self._model = model
        self._context_window = context_window
        self._thresholds = thresholds
        self._message_projector = message_projector

    def _messages(self, state: ResearchState) -> list[BaseMessage]:
        if self._message_projector is not None:
            return list(self._message_projector(state))
        return list(state.get("messages", []))

    def _budget_update(self, state: ResearchState) -> dict[str, Any]:
        estimate = estimate_context_tokens(self._messages(state), self._model)
        window = resolve_context_window(self._model, self._context_window)
        decision = evaluate_context_pressure(
            estimate.token_count,
            window.window_tokens,
            self._thresholds,
        )
        return {
            "model_name": window.model_name,
            "budget": {
                "current_tokens": decision.current_tokens,
                "window_tokens": decision.window_tokens,
                "utilization_ratio": decision.utilization_ratio,
                "token_count_method": estimate.method,
                "window_source": window.source,
            },
            "pending_stages": list(decision.pending_stages),
            "hard_limit_reached": decision.hard_limit_reached,
        }

    def _governance_update(self, state: ResearchState) -> dict[str, Any]:
        messages = self._messages(state)
        governance = _mapping(state.get("governance"))
        previous_context = _mapping(governance.get("context"))
        previous_usage = _mapping(previous_context.get("cumulative_usage"))
        seen_usage = _mapping(previous_context.get("seen_message_usage"))

        usage_update = collect_usage_delta(messages, seen_usage)
        cumulative_usage: CumulativeTokenUsage = {
            field: _non_negative_int(previous_usage.get(field))
            + usage_update.delta[field]
            for field in ("input_tokens", "output_tokens", "total_tokens")
        }

        return {
            "governance": {
                "context": {
                    **self._budget_update(state),
                    "cumulative_usage": cumulative_usage,
                    "model_call_count": _non_negative_int(
                        previous_context.get("model_call_count")
                    )
                    + 1,
                    "seen_message_usage": usage_update.seen_message_usage,
                }
            }
        }

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        """Classify pressure before Tool results enter the next model request."""

        return {"governance": {"context": self._budget_update(state)}}

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        """Apply the same pre-call classification in asynchronous runs."""

        return {"governance": {"context": self._budget_update(state)}}

    @override
    def after_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        """Record one synchronous model call without changing graph routing."""

        return self._governance_update(state)

    @override
    async def aafter_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        """Record one asynchronous model call using the same pure calculation."""

        return self._governance_update(state)
