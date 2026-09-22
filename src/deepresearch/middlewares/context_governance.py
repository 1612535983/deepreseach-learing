"""Measure context pressure after each model call and record it in State."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
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
    """Record token usage and pressure without changing messages or graph routing."""

    state_schema = ResearchState

    def __init__(
        self,
        model: Any,
        *,
        context_window: int | None = None,
        thresholds: ContextThresholds = DEFAULT_CONTEXT_THRESHOLDS,
    ) -> None:
        self._model = model
        self._context_window = context_window
        self._thresholds = thresholds

    def _governance_update(self, state: ResearchState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        governance = _mapping(state.get("governance"))
        previous_context = _mapping(governance.get("context"))
        previous_usage = _mapping(previous_context.get("cumulative_usage"))
        seen_usage = _mapping(previous_context.get("seen_message_usage"))

        estimate = estimate_context_tokens(messages, self._model)
        window = resolve_context_window(self._model, self._context_window)
        decision = evaluate_context_pressure(
            estimate.token_count,
            window.window_tokens,
            self._thresholds,
        )
        usage_update = collect_usage_delta(messages, seen_usage)
        cumulative_usage: CumulativeTokenUsage = {
            field: _non_negative_int(previous_usage.get(field))
            + usage_update.delta[field]
            for field in ("input_tokens", "output_tokens", "total_tokens")
        }

        return {
            "governance": {
                "context": {
                    "model_name": window.model_name,
                    "budget": {
                        "current_tokens": decision.current_tokens,
                        "window_tokens": decision.window_tokens,
                        "utilization_ratio": decision.utilization_ratio,
                        "token_count_method": estimate.method,
                        "window_source": window.source,
                    },
                    "cumulative_usage": cumulative_usage,
                    "model_call_count": _non_negative_int(
                        previous_context.get("model_call_count")
                    )
                    + 1,
                    "pending_stages": list(decision.pending_stages),
                    "hard_limit_reached": decision.hard_limit_reached,
                    "seen_message_usage": usage_update.seen_message_usage,
                }
            }
        }

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
