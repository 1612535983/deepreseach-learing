"""Bounded, deterministic quality checks before a research run may finish."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.quality import assess_research
from deepresearch.state import ResearchState


MAX_REFLECTION_ATTEMPTS = 2


def _last_ai_message(state: ResearchState) -> AIMessage | None:
    """Find the latest model response in the message history."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class ReflectionMiddleware(AgentMiddleware):
    """Send an incomplete run back to the model, with a strict retry cap."""

    state_schema = ResearchState

    def _inspect_after_model(self, state: ResearchState) -> dict[str, Any] | None:
        governance = _mapping(state.get("governance"))
        context = _mapping(governance.get("context"))
        pending = context.get("pending_stages", [])
        finalization = _mapping(context.get("finalization"))
        if "P5" in pending or finalization.get("active") is True:
            # P5 owns termination once the context budget is critical. Reflection
            # must not create extra model loops that work against forced finalization.
            return None

        last_message = _last_ai_message(state)
        if last_message is None or last_message.tool_calls:
            # A tool call means the agent is still working; let the normal
            # model -> tools edge continue without judging an intermediate state.
            return None

        gaps = assess_research(state)
        if not gaps:
            return {"research_gaps": []}

        attempts = state.get("reflection_attempts", 0)
        if attempts >= MAX_REFLECTION_ATTEMPTS:
            # Preserve the remaining gaps for observability, but stop looping.
            return {"research_gaps": gaps}

        return {
            "reflection_attempts": attempts + 1,
            "research_gaps": gaps,
            "jump_to": "model",
        }

    @override
    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect synchronous model output before the graph decides to end."""

        return self._inspect_after_model(state)

    @override
    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inspect asynchronous model output with the same deterministic rules."""

        return self._inspect_after_model(state)
