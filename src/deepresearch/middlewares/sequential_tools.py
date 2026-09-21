"""Force state-mutating research tools to run one model turn at a time."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from deepresearch.state import ResearchState


def _disable_parallel_tool_calls(request: ModelRequest) -> ModelRequest:
    """Pass the provider setting through LangChain's model binding step."""

    model_settings = {
        **request.model_settings,
        "parallel_tool_calls": False,
    }
    return request.override(model_settings=model_settings)


class SequentialToolCallMiddleware(AgentMiddleware):
    """Prevent concurrent Commands from writing the same State fields."""

    state_schema = ResearchState

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_disable_parallel_tool_calls(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(_disable_parallel_tool_calls(request))
