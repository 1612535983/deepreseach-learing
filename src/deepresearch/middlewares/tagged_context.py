"""Project canonical State into the tagged context used for each model request."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from deepresearch.context.tagged import AssembledContext, ContextAssembler
from deepresearch.state import ResearchState


_CST = timezone(timedelta(hours=8))


def _created_at() -> str:
    return datetime.now(_CST).isoformat()


class TaggedContextMiddleware(AgentMiddleware):
    """Tag model-visible context without overwriting canonical State messages."""

    state_schema = ResearchState

    def __init__(
        self,
        system_prompt: str,
        *,
        include_research_context: bool,
        assembler: ContextAssembler | None = None,
        created_at_provider: Callable[[], str] = _created_at,
    ) -> None:
        self._system_message = SystemMessage(content=system_prompt)
        self._include_research_context = include_research_context
        self._assembler = assembler or ContextAssembler()
        self._created_at_provider = created_at_provider

    def _assemble_state(self, state: ResearchState) -> AssembledContext:
        return self._assembler.assemble(
            state,
            list(state.get("messages", [])),
            self._system_message,
            include_research_context=self._include_research_context,
        )

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, object]:
        """Persist the exact pre-call projection for Checkpoint and trace audits."""

        assembled = self._assemble_state(state)
        return {
            "tagged_context": {
                "rendered": assembled.rendered,
                "message_count": assembled.message_count,
                "created_at": self._created_at_provider(),
            }
        }

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, object]:
        """Persist the same projection for asynchronous graph execution."""

        return self.before_model(state, runtime)

    def _assemble_request(self, request: ModelRequest) -> ModelRequest:
        assembled = self._assembler.assemble(
            request.state,
            request.messages,
            request.system_message,
            include_research_context=self._include_research_context,
        )
        return request.override(
            system_message=assembled.system_message,
            messages=assembled.messages,
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Send the tagged projection to the synchronous model handler."""

        return handler(self._assemble_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Send the tagged projection to the asynchronous model handler."""

        return await handler(self._assemble_request(request))
