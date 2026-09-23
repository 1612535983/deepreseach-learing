"""Execute bounded P5 finalization before expansive Tool calls can run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from deepresearch.state import ResearchState


DEFAULT_TERMINAL_TOOLS = frozenset(
    {
        "update_plan_step",
        "write_final_report",
    }
)
CONTEXT_BUDGET_STOP_NAME = "context_budget_stop"
DEEPRESEARCH_CONTEXT_BUDGET_STOP = "deepresearch.context_budget_stop"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _ratio(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    return 0.0


def _last_ai_message(state: ResearchState) -> AIMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


class ContextFinalizationMiddleware(AgentMiddleware):
    """Turn P5 or unproductive Tool loops into bounded terminal mode."""

    state_schema = ResearchState

    def __init__(
        self,
        *,
        terminal_tools: Iterable[str] = DEFAULT_TERMINAL_TOOLS,
        max_redirects: int = 1,
        max_terminal_tool_calls: int = 3,
        max_search_attempts: int = 12,
        max_consecutive_unproductive_searches: int = 4,
        max_repeated_search_query: int = 2,
        max_page_reads: int = 12,
        max_consecutive_page_failures: int = 3,
    ) -> None:
        if max_redirects <= 0:
            raise ValueError("max_redirects 必须大于 0。")
        if max_terminal_tool_calls <= 0:
            raise ValueError("max_terminal_tool_calls 必须大于 0。")
        for name, value in (
            ("max_search_attempts", max_search_attempts),
            (
                "max_consecutive_unproductive_searches",
                max_consecutive_unproductive_searches,
            ),
            ("max_repeated_search_query", max_repeated_search_query),
            ("max_page_reads", max_page_reads),
            ("max_consecutive_page_failures", max_consecutive_page_failures),
        ):
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0。")
        self._terminal_tools = frozenset(
            name.strip() for name in terminal_tools if name.strip()
        )
        self._max_redirects = max_redirects
        self._max_terminal_tool_calls = max_terminal_tool_calls
        self._max_search_attempts = max_search_attempts
        self._max_consecutive_unproductive_searches = (
            max_consecutive_unproductive_searches
        )
        self._max_repeated_search_query = max_repeated_search_query
        self._max_page_reads = max_page_reads
        self._max_consecutive_page_failures = max_consecutive_page_failures

    @staticmethod
    def _context(state: ResearchState) -> Mapping[str, Any]:
        governance = _mapping(state.get("governance"))
        return _mapping(governance.get("context"))

    def _update(self, state: ResearchState) -> dict[str, Any] | None:
        context = self._context(state)
        previous = _mapping(context.get("finalization"))
        pending = context.get("pending_stages")
        p5_pending = isinstance(pending, list) and "P5" in pending
        already_active = previous.get("active") is True
        last_ai = _last_ai_message(state)
        research_budget_reason = self._research_budget_reason(state, last_ai)
        if not p5_pending and not already_active and research_budget_reason is None:
            return None

        budget = _mapping(context.get("budget"))
        utilization_ratio = _ratio(budget.get("utilization_ratio"))
        hard_limit = context.get("hard_limit_reached") is True
        trigger_reason = previous.get("trigger_reason")
        if not isinstance(trigger_reason, str) or not trigger_reason:
            if hard_limit:
                trigger_reason = "hard_limit"
            elif p5_pending:
                trigger_reason = "p5_threshold"
            else:
                trigger_reason = research_budget_reason
        if last_ai is None or not last_ai.tool_calls:
            return self._state_update(
                previous,
                utilization_ratio=utilization_ratio,
                reason="model_stopped_tools",
                trigger_reason=trigger_reason,
            )

        tool_names = [
            str(tool_call.get("name") or "unknown")
            for tool_call in last_ai.tool_calls
        ]
        if tool_names and all(name in self._terminal_tools for name in tool_names):
            terminal_count = _non_negative_int(
                previous.get("terminal_tool_call_count")
            )
            if terminal_count + len(tool_names) > self._max_terminal_tool_calls:
                stopped = self._strip_tool_calls(
                    last_ai,
                    utilization_ratio,
                    fallback_content=self._forced_stop_message(tool_names),
                )
                return {
                    "messages": self._replacement_patch(state, last_ai, stopped),
                    "governance": {
                        "context": {
                            "finalization": self._metrics(
                                previous,
                                utilization_ratio=utilization_ratio,
                                reason="terminal_tool_limit",
                                blocked_tool_names=tool_names,
                                forced_stop_increment=1,
                                trigger_reason=trigger_reason,
                            )
                        }
                    },
                    "jump_to": "end",
                }
            return self._state_update(
                previous,
                utilization_ratio=utilization_ratio,
                reason="terminal_tool_allowed",
                terminal_tool_call_increment=len(tool_names),
                trigger_reason=trigger_reason,
            )

        redirect_count = _non_negative_int(previous.get("redirect_count"))
        if redirect_count >= self._max_redirects:
            stopped = self._strip_tool_calls(
                last_ai,
                utilization_ratio,
                fallback_content=self._forced_stop_message(tool_names),
            )
            return {
                "messages": self._replacement_patch(state, last_ai, stopped),
                "governance": {
                    "context": {
                        "finalization": self._metrics(
                            previous,
                            utilization_ratio=utilization_ratio,
                            reason="redirect_limit",
                            blocked_tool_names=tool_names,
                            forced_stop_increment=1,
                            trigger_reason=trigger_reason,
                        )
                    }
                },
                "jump_to": "end",
            }

        reminder_id = self._reminder_id(last_ai, redirect_count + 1)
        cleared = self._strip_tool_calls(last_ai, utilization_ratio)
        reminder = HumanMessage(
            id=reminder_id,
            name=CONTEXT_BUDGET_STOP_NAME,
            content=self._stop_message(
                utilization_ratio,
                hard_limit=hard_limit,
                trigger_reason=str(trigger_reason or "research_budget"),
            ),
            additional_kwargs={
                "hide_from_ui": True,
                DEEPRESEARCH_CONTEXT_BUDGET_STOP: True,
            },
        )
        return {
            "messages": [
                *self._replacement_patch(state, last_ai, cleared),
                reminder,
            ],
            "governance": {
                "context": {
                    "finalization": self._metrics(
                        previous,
                        utilization_ratio=utilization_ratio,
                        reason=(
                            "hard_limit"
                            if hard_limit
                            else "p5_threshold"
                            if p5_pending
                            else str(trigger_reason or "research_budget")
                        ),
                        blocked_tool_names=tool_names,
                        redirect_increment=1,
                        reminder_id=reminder_id,
                        trigger_reason=trigger_reason,
                    )
                }
            },
            "jump_to": "model",
        }

    def _state_update(
        self,
        previous: Mapping[str, Any],
        *,
        utilization_ratio: float,
        reason: str,
        terminal_tool_call_increment: int = 0,
        trigger_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "governance": {
                "context": {
                    "finalization": self._metrics(
                        previous,
                        utilization_ratio=utilization_ratio,
                        reason=reason,
                        terminal_tool_call_increment=(
                            terminal_tool_call_increment
                        ),
                        trigger_reason=trigger_reason,
                    )
                }
            }
        }

    @staticmethod
    def _metrics(
        previous: Mapping[str, Any],
        *,
        utilization_ratio: float,
        reason: str,
        blocked_tool_names: list[str] | None = None,
        redirect_increment: int = 0,
        terminal_tool_call_increment: int = 0,
        forced_stop_increment: int = 0,
        reminder_id: str | None = None,
        trigger_reason: str | None = None,
    ) -> dict[str, Any]:
        previous_names = previous.get("last_blocked_tool_names")
        if not isinstance(previous_names, list):
            previous_names = []
        return {
            "active": True,
            "redirect_count": _non_negative_int(previous.get("redirect_count"))
            + redirect_increment,
            "blocked_tool_call_count": _non_negative_int(
                previous.get("blocked_tool_call_count")
            )
            + len(blocked_tool_names or []),
            "terminal_tool_call_count": _non_negative_int(
                previous.get("terminal_tool_call_count")
            )
            + terminal_tool_call_increment,
            "forced_stop_count": _non_negative_int(
                previous.get("forced_stop_count")
            )
            + forced_stop_increment,
            "last_blocked_tool_names": (
                list(blocked_tool_names)
                if blocked_tool_names is not None
                else list(previous_names)
            ),
            "last_utilization_ratio": utilization_ratio,
            "last_reason": reason,
            "trigger_reason": (
                trigger_reason
                if trigger_reason is not None
                else previous.get("trigger_reason")
            ),
            "last_reminder_id": (
                reminder_id
                if reminder_id is not None
                else previous.get("last_reminder_id")
            ),
        }

    @staticmethod
    def _strip_tool_calls(
        message: AIMessage,
        utilization_ratio: float,
        *,
        fallback_content: str | None = None,
    ) -> AIMessage:
        kwargs = {
            key: value
            for key, value in message.additional_kwargs.items()
            if key not in {"tool_calls", "function_call"}
        }
        kwargs[DEEPRESEARCH_CONTEXT_BUDGET_STOP] = round(
            utilization_ratio,
            6,
        )
        return message.model_copy(
            update={
                "content": (
                    fallback_content
                    if fallback_content is not None
                    else message.content or ""
                ),
                "tool_calls": [],
                "invalid_tool_calls": [],
                "additional_kwargs": kwargs,
            }
        )

    @staticmethod
    def _replacement_patch(
        state: ResearchState,
        original: AIMessage,
        replacement: AIMessage,
    ) -> list[BaseMessage]:
        if original.id:
            return [replacement]
        identity = json.dumps(
            original.tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        replacement = replacement.model_copy(
            update={
                "id": "p5-ai-"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            }
        )
        messages = list(state.get("messages", []))
        for index in range(len(messages) - 1, -1, -1):
            if messages[index] is original:
                messages[index] = replacement
                break
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]

    @staticmethod
    def _reminder_id(message: AIMessage, redirect_number: int) -> str:
        identity = json.dumps(
            {
                "message_id": message.id,
                "tool_calls": message.tool_calls,
                "redirect_number": redirect_number,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return "p5-reminder-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _stop_message(
        utilization_ratio: float,
        *,
        hard_limit: bool,
        trigger_reason: str,
    ) -> str:
        if trigger_reason == "search_attempt_limit":
            pressure = "网页搜索次数已经达到运行上限"
        elif trigger_reason == "consecutive_unproductive_searches":
            pressure = "网页搜索已经连续多次失败或没有结果"
        elif trigger_reason == "repeated_search_query":
            pressure = "同一搜索词已经重复尝试多次"
        elif trigger_reason == "page_read_limit":
            pressure = "网页读取次数已经达到运行上限"
        elif trigger_reason == "consecutive_page_failures":
            pressure = "网页读取已经连续多次失败"
        elif hard_limit:
            pressure = "上下文窗口已达到 99% 硬限制"
        else:
            pressure = f"上下文窗口占用已达到 {utilization_ratio:.0%}"
        return (
            "<system_reminder>\n"
            f"{pressure}，现在必须收尾。不得继续搜索、读取网页、创建新计划或调用其他扩张型工具。"
            "只允许用 update_plan_step 整理已有计划，或用 write_final_report 保存已有证据能够支持的报告。"
            "如果现有证据不足以通过报告校验，请直接给出基于现有信息的最佳答案，并明确说明缺口；"
            "不要再调用工具。\n"
            "</system_reminder>"
        )

    def _research_budget_reason(
        self,
        state: ResearchState,
        last_ai: AIMessage | None,
    ) -> str | None:
        """Detect unproductive research loops before their next Tool executes."""

        search_records = list(state.get("search_records", []))
        if len(search_records) >= self._max_search_attempts:
            return "search_attempt_limit"
        consecutive_searches = 0
        for record in reversed(search_records):
            if (
                record.get("success") is True
                and int(record.get("result_count") or 0) > 0
            ):
                break
            consecutive_searches += 1
        if consecutive_searches >= self._max_consecutive_unproductive_searches:
            return "consecutive_unproductive_searches"

        if last_ai is not None:
            for tool_call in last_ai.tool_calls:
                if tool_call.get("name") != "web_search":
                    continue
                args = tool_call.get("args")
                raw_query = args.get("query") if isinstance(args, Mapping) else ""
                query = " ".join(str(raw_query or "").lower().split())
                repeated = sum(
                    " ".join(str(record.get("query") or "").lower().split())
                    == query
                    for record in search_records
                    if query
                )
                if repeated >= self._max_repeated_search_query:
                    return "repeated_search_query"

        page_records = list(state.get("page_records", []))
        if len(page_records) >= self._max_page_reads:
            return "page_read_limit"
        consecutive_page_failures = 0
        for record in reversed(page_records):
            if record.get("success") is True:
                break
            consecutive_page_failures += 1
        if consecutive_page_failures >= self._max_consecutive_page_failures:
            return "consecutive_page_failures"
        return None

    @staticmethod
    def _forced_stop_message(tool_names: list[str]) -> str:
        names = ", ".join(tool_names) or "unknown"
        return (
            "上下文预算已经进入强制停止状态。为避免超过模型窗口，未执行以下 Tool Call："
            f"{names}。请从 Checkpoint、P4 Snapshot 或外化文件恢复后再继续。"
        )

    @override
    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Intercept synchronous model output before the Tool node runs."""

        return self._update(state)

    @override
    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Apply the same bounded P5 policy to asynchronous runs."""

        return self._update(state)
