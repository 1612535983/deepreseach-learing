"""Bounded, deterministic quality checks before a research run may finish."""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.state import ResearchState


MAX_REFLECTION_ATTEMPTS = 2
MIN_UNIQUE_SOURCES = 2
MIN_SUCCESSFUL_PAGE_READS = 1
MIN_OBSERVATIONS = 2


def assess_research(state: ResearchState) -> list[str]:
    """Return concrete gaps found in the structured research state.

    This is deliberately a programmatic check rather than another LLM call, so
    the result is predictable, testable, and inexpensive.
    """

    gaps: list[str] = []
    plan = state.get("plan")
    if not plan:
        gaps.append("尚未创建研究计划。")
    else:
        unfinished_steps = [
            step
            for step in plan["steps"]
            if step["status"] in {"pending", "in_progress"}
        ]
        for step in unfinished_steps:
            gaps.append(
                f'{step["step_id"]} 尚未完成：{step["title"]}'
            )

    source_count = len(state.get("sources", []))
    if source_count < MIN_UNIQUE_SOURCES:
        gaps.append(
            f"去重来源不足：当前 {source_count} 个，至少需要 {MIN_UNIQUE_SOURCES} 个。"
        )

    successful_page_reads = sum(
        record["success"] for record in state.get("page_records", [])
    )
    if successful_page_reads < MIN_SUCCESSFUL_PAGE_READS:
        gaps.append(
            "网页正文读取不足："
            f"当前成功 {successful_page_reads} 次，"
            f"至少需要 {MIN_SUCCESSFUL_PAGE_READS} 次。"
        )

    observation_count = len(state.get("observations", []))
    if observation_count < MIN_OBSERVATIONS:
        gaps.append(
            f"证据观察不足：当前 {observation_count} 条，"
            f"至少需要 {MIN_OBSERVATIONS} 条。"
        )

    return gaps


def _last_ai_message(state: ResearchState) -> AIMessage | None:
    """Find the latest model response in the message history."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


class ReflectionMiddleware(AgentMiddleware):
    """Send an incomplete run back to the model, with a strict retry cap."""

    state_schema = ResearchState

    def _inspect_after_model(self, state: ResearchState) -> dict[str, Any] | None:
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
