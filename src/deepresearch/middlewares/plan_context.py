"""Expose a small, request-scoped projection of ResearchState to the model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepresearch.context.tagged import format_research_context
from deepresearch.state import ResearchState


def format_plan_context(state: ResearchState) -> str:
    """Render only plan progress and aggregate counts, never raw evidence."""

    return format_research_context(state)


def _system_text(message: SystemMessage | None) -> str:
    if message is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in message.content
        )
    return str(message.content)


def _with_plan_context(request: ModelRequest) -> ModelRequest:
    base_prompt = _system_text(request.system_message)
    plan_context = format_plan_context(request.state)
    combined = f"{base_prompt}\n\n{plan_context}" if base_prompt else plan_context
    return request.override(system_message=SystemMessage(content=combined))


class PlanContextMiddleware(AgentMiddleware):
    """Inject plan progress into each model request without persisting messages."""

    state_schema = ResearchState

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_with_plan_context(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(_with_plan_context(request))
