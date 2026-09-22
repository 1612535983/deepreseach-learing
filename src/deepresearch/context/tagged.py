"""Build the request-scoped, semantically tagged view shown to the model."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import escape
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


DEEPRESEARCH_THINKING = "deepresearch.thinking"
DEEPRESEARCH_SUMMARY = "deepresearch.summary"
DEEPRESEARCH_EXTERNALIZED = "deepresearch.externalized"
DEEPRESEARCH_EXTERNALIZED_PATH = "deepresearch.externalized_path"
DEEPRESEARCH_EXTERNALIZED_META = "deepresearch.externalized_meta"
DEEPRESEARCH_COMPACTION_STAGE = "deepresearch.compaction_stage"
DEEPRESEARCH_SNAPSHOT_PATH = "deepresearch.snapshot_path"

_CST = timezone(timedelta(hours=8))


def _today() -> date:
    return datetime.now(_CST).date()


def _message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(dict(item), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content is not None else ""


def _attribute(value: object) -> str:
    return escape(str(value), quote=True)


def format_research_context(state: Mapping[str, Any]) -> str:
    """Render plan progress and reflection gaps without raw evidence bodies."""

    plan = state.get("plan")
    if not isinstance(plan, Mapping):
        lines = [
            '<research_plan status="missing">',
            "No research plan exists yet. Before calling web_search or read_page, call",
            "write_research_plan with a concise goal and 2 to 5 ordered steps.",
            "</research_plan>",
        ]
    else:
        status_labels = {
            "pending": "待处理",
            "in_progress": "进行中",
            "completed": "已完成",
            "failed": "失败",
        }
        lines = [
            "<research_plan>",
            f'研究目标：{escape(str(plan.get("goal") or ""))}',
            f'当前步骤：{escape(str(state.get("current_step_id") or "无"))}',
        ]
        steps = plan.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                status = status_labels.get(str(step.get("status")), "未知")
                step_id = escape(str(step.get("step_id") or "unknown"))
                title = escape(str(step.get("title") or ""))
                lines.append(f"- [{status}] {step_id}: {title}")
        lines.extend(
            [
                "完成当前步骤后调用 update_plan_step，再继续下一步。",
                "</research_plan>",
            ]
        )

    lines.extend(
        [
            "<research_progress>",
            f'- 搜索次数：{len(state.get("search_records", []))}',
            f'- 网页读取次数：{len(state.get("page_records", []))}',
            f'- 去重来源数：{len(state.get("sources", []))}',
            f'- Observation 数：{len(state.get("observations", []))}',
            f'- 正式报告：{"已生成" if state.get("final_report") else "未生成"}',
            "</research_progress>",
        ]
    )
    gaps = state.get("research_gaps")
    if isinstance(gaps, list) and gaps:
        lines.extend(
            [
                "<research_gaps>",
                "上一次回答前的程序检查发现以下缺口，请继续使用工具补充：",
                *(f"- {escape(str(gap))}" for gap in gaps),
                f'当前反思次数：{state.get("reflection_attempts", 0)}',
                "</research_gaps>",
            ]
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class AssembledContext:
    """The final request projection plus a human-readable audit rendering."""

    system_message: SystemMessage
    messages: list[BaseMessage]
    rendered: str
    message_count: int


class ContextAssembler:
    """Turn State and raw messages into a stable, tagged model-context view."""

    def __init__(
        self,
        today_provider: Callable[[], date] = _today,
        *,
        memory_context_provider: Callable[[Mapping[str, Any]], str] | None = None,
        skill_context_provider: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> None:
        self._today_provider = today_provider
        self._memory_context_provider = memory_context_provider
        self._skill_context_provider = skill_context_provider

    def assemble(
        self,
        state: Mapping[str, Any],
        messages: Sequence[BaseMessage],
        system_message: SystemMessage | None,
        *,
        include_research_context: bool,
    ) -> AssembledContext:
        """Build model messages without mutating the canonical State messages."""

        context_block = self.render_context_block(
            state,
            system_message,
            include_research_context=include_research_context,
        )
        rewritten = self.rewrite_messages_for_model(messages)
        rendered_messages = self.render_messages(messages)
        rendered = (
            f"{context_block}\n\n{rendered_messages}"
            if rendered_messages
            else context_block
        )
        tagged_system = (
            system_message.model_copy(update={"content": context_block})
            if system_message is not None
            else SystemMessage(content=context_block)
        )
        return AssembledContext(
            system_message=tagged_system,
            messages=rewritten,
            rendered=rendered,
            message_count=len(rewritten) + 1,
        )

    def render_context_block(
        self,
        state: Mapping[str, Any],
        system_message: SystemMessage | None,
        *,
        include_research_context: bool,
    ) -> str:
        """Render system instructions and selected State fields as tagged text."""

        lines = [f"<system>{escape(_message_text(system_message))}</system>"]
        question = state.get("research_question")
        if isinstance(question, str) and question:
            lines.append(f"<goal>{escape(question)}</goal>")

        governance = state.get("governance")
        if isinstance(governance, Mapping):
            context = governance.get("context")
            summary = context.get("summary") if isinstance(context, Mapping) else None
            if isinstance(summary, str) and summary.strip():
                lines.extend(
                    [
                        "<summary>",
                        escape(summary),
                        "</summary>",
                    ]
                )

        if include_research_context:
            lines.extend(format_research_context(state).splitlines())

        if self._skill_context_provider is not None:
            skill_context = self._skill_context_provider(state)
            if skill_context.strip():
                lines.extend(skill_context.splitlines())

        if self._memory_context_provider is not None:
            memory_context = self._memory_context_provider(state)
            if memory_context.strip():
                lines.extend(memory_context.splitlines())

        lines.append(f"<date>{self._today_provider().isoformat()}</date>")
        return "\n".join(lines)

    def rewrite_messages_for_model(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        """Tag AI semantics while preserving native roles and Tool pairing."""

        rewritten: list[BaseMessage] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if (
                isinstance(message, HumanMessage)
                and message.additional_kwargs.get(DEEPRESEARCH_SUMMARY)
            ):
                continue
            if isinstance(message, AIMessage):
                rewritten.append(self._rewrite_ai_message(message))
            else:
                rewritten.append(message)
        return rewritten

    @staticmethod
    def _rewrite_ai_message(message: AIMessage) -> AIMessage:
        parts: list[str] = []
        if message.additional_kwargs.get(DEEPRESEARCH_THINKING):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning:
                parts.append(f"<thinking>{escape(str(reasoning))}</thinking>")
        content = _message_text(message)
        if content:
            parts.append(f"<answer>{escape(content)}</answer>")
        if not parts:
            return message
        return message.model_copy(update={"content": "".join(parts)})

    def render_messages(self, messages: Sequence[BaseMessage]) -> str:
        """Render an audit trace grouped into chronological user turns."""

        lines: list[str] = []
        turn_id = 0
        turn_open = False
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if (
                isinstance(message, HumanMessage)
                and message.additional_kwargs.get(DEEPRESEARCH_SUMMARY)
            ):
                continue
            if isinstance(message, HumanMessage):
                if turn_open:
                    lines.append("</turn>")
                turn_id += 1
                turn_open = True
                lines.extend(
                    [
                        f'<turn id="{turn_id}">',
                        f'<message role="user">{escape(_message_text(message))}</message>',
                    ]
                )
                continue
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    name = _attribute(tool_call.get("name", ""))
                    arguments = _attribute(
                        json.dumps(
                            tool_call.get("args", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    lines.append(f'<toolcall name="{name}" args="{arguments}"/>')
                content = _message_text(message)
                if content:
                    lines.append(f"<answer>{escape(content)}</answer>")
                continue
            if isinstance(message, ToolMessage):
                name = _attribute(message.name or "unknown")
                content = escape(_message_text(message))
                if message.additional_kwargs.get(DEEPRESEARCH_EXTERNALIZED):
                    path = _attribute(
                        message.additional_kwargs.get(
                            DEEPRESEARCH_EXTERNALIZED_PATH,
                            "",
                        )
                    )
                    metadata = message.additional_kwargs.get(
                        DEEPRESEARCH_EXTERNALIZED_META,
                        {},
                    )
                    tokens_saved = (
                        metadata.get("estimated_tokens_saved", "")
                        if isinstance(metadata, Mapping)
                        else ""
                    )
                    lines.append(
                        f'<toolresult name="{name}" path="{path}" '
                        f'tokens_saved="{_attribute(tokens_saved)}">'
                        f"{content}</toolresult>"
                    )
                else:
                    lines.append(
                        f'<toolresult name="{name}">{content}</toolresult>'
                    )
        if turn_open:
            lines.append("</turn>")
        return "\n".join(lines)
