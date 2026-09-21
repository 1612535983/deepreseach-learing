"""Expose a small, request-scoped projection of ResearchState to the model."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepresearch.state import ResearchState


def format_plan_context(state: ResearchState) -> str:
    """Render only plan progress and aggregate counts, never raw evidence."""

    plan = state.get("plan")
    if not plan:
        lines = [
            "<research_plan>",
            "No research plan exists yet. Before calling web_search or read_page, call",
            "write_research_plan with a concise goal and 2 to 5 ordered steps.",
            "</research_plan>",
        ]
        return "\n".join(_append_reflection_context(lines, state))

    status_labels = {
        "pending": "待处理",
        "in_progress": "进行中",
        "completed": "已完成",
        "failed": "失败",
    }
    lines = [
        "<research_plan>",
        f'研究目标：{plan["goal"]}',
        f'当前步骤：{state.get("current_step_id") or "无"}',
        "步骤：",
    ]
    for step in plan["steps"]:
        lines.append(
            f'- [{status_labels[step["status"]]}] {step["step_id"]}: {step["title"]}'
        )
    lines.extend(
        [
            "进度统计：",
            f'- 搜索次数：{len(state.get("search_records", []))}',
            f'- 网页读取次数：{len(state.get("page_records", []))}',
            f'- 去重来源数：{len(state.get("sources", []))}',
            f'- Observation 数：{len(state.get("observations", []))}',
            "完成当前步骤后调用 update_plan_step，再继续下一步。",
            "</research_plan>",
        ]
    )
    return "\n".join(_append_reflection_context(lines, state))


def _append_reflection_context(
    lines: list[str],
    state: ResearchState,
) -> list[str]:
    """Append only actionable gap text produced by ReflectionMiddleware."""

    gaps = state.get("research_gaps", [])
    if not gaps:
        return lines

    lines.extend(
        [
            "<research_gaps>",
            "上一次回答前的程序检查发现以下缺口，请继续使用工具补充：",
            *(f"- {gap}" for gap in gaps),
            f'当前反思次数：{state.get("reflection_attempts", 0)}',
            "</research_gaps>",
        ]
    )
    return lines


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
