"""Stable application events derived from LangGraph's raw stream updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage


ResearchEventType = Literal[
    "run_started",
    "tool_requested",
    "tool_completed",
    "plan_updated",
    "search_completed",
    "page_read_completed",
    "reflection",
    "report_created",
    "run_completed",
    "run_failed",
]


@dataclass(frozen=True)
class ResearchEvent:
    """One user-facing progress event independent of LangGraph's chunk format."""

    event_type: ResearchEventType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    node: str | None = None


def _tool_request_event(tool_call: dict[str, Any], node: str) -> ResearchEvent:
    name = str(tool_call.get("name") or "unknown_tool")
    raw_args = tool_call.get("args")
    args = raw_args if isinstance(raw_args, dict) else {}

    if name == "web_search":
        query = str(args.get("query") or "")
        message = f"请求搜索：{query}" if query else "请求执行网页搜索"
        data = {"tool_name": name, "query": query}
    elif name == "read_page":
        url = str(args.get("url") or "")
        message = f"请求读取网页：{url}" if url else "请求读取网页"
        data = {"tool_name": name, "url": url}
    elif name == "write_research_plan":
        goal = str(args.get("goal") or "")
        steps = args.get("steps")
        step_count = len(steps) if isinstance(steps, list) else 0
        message = f"请求创建研究计划，共 {step_count} 个步骤"
        data = {"tool_name": name, "goal": goal, "step_count": step_count}
    elif name == "update_plan_step":
        step_id = str(args.get("step_id") or "")
        status = str(args.get("status") or "")
        message = f"请求更新计划：{step_id} → {status}"
        data = {"tool_name": name, "step_id": step_id, "status": status}
    elif name == "write_final_report":
        title = str(args.get("title") or "")
        urls = args.get("used_source_urls")
        source_count = len(urls) if isinstance(urls, list) else 0
        message = f"请求写入最终报告：{title}" if title else "请求写入最终报告"
        # Deliberately omit the potentially very large report argument.
        data = {
            "tool_name": name,
            "title": title,
            "source_count": source_count,
        }
    else:
        message = f"请求执行工具：{name}"
        data = {"tool_name": name}

    return ResearchEvent("tool_requested", message, data, node)


def _message_events(
    messages: object,
    node: str,
    *,
    has_domain_update: bool,
) -> list[ResearchEvent]:
    if not isinstance(messages, list):
        return []

    events: list[ResearchEvent] = []
    for message in messages:
        if isinstance(message, AIMessage):
            events.extend(
                _tool_request_event(tool_call, node)
                for tool_call in message.tool_calls
            )
        elif isinstance(message, ToolMessage) and not has_domain_update:
            tool_name = message.name or "unknown_tool"
            events.append(
                ResearchEvent(
                    "tool_completed",
                    f"工具已返回：{tool_name}",
                    {"tool_name": tool_name},
                    node,
                )
            )
    return events


def events_from_update(update: object) -> list[ResearchEvent]:
    """Convert one ``stream_mode='updates'`` chunk into safe progress events."""

    if not isinstance(update, dict):
        return []

    events: list[ResearchEvent] = []
    for raw_node, raw_patch in update.items():
        node = str(raw_node)
        if not isinstance(raw_patch, dict):
            continue

        domain_keys = {
            "plan",
            "search_records",
            "page_records",
            "research_gaps",
            "final_report",
        }
        events.extend(
            _message_events(
                raw_patch.get("messages"),
                node,
                has_domain_update=bool(domain_keys.intersection(raw_patch)),
            )
        )

        plan = raw_patch.get("plan")
        if isinstance(plan, dict):
            steps = plan.get("steps")
            step_count = len(steps) if isinstance(steps, list) else 0
            current_step_id = raw_patch.get("current_step_id")
            events.append(
                ResearchEvent(
                    "plan_updated",
                    f"研究计划已更新，共 {step_count} 个步骤",
                    {
                        "step_count": step_count,
                        "current_step_id": current_step_id,
                    },
                    node,
                )
            )

        search_records = raw_patch.get("search_records")
        if isinstance(search_records, list):
            for record in search_records:
                if not isinstance(record, dict):
                    continue
                query = str(record.get("query") or "")
                success = record.get("success") is True
                result_count = int(record.get("result_count") or 0)
                message = (
                    f"搜索完成：{query}，返回 {result_count} 个结果"
                    if success
                    else f"搜索失败：{query}"
                )
                events.append(
                    ResearchEvent(
                        "search_completed",
                        message,
                        {
                            "query": query,
                            "success": success,
                            "result_count": result_count,
                            "error": record.get("error"),
                        },
                        node,
                    )
                )

        page_records = raw_patch.get("page_records")
        if isinstance(page_records, list):
            for record in page_records:
                if not isinstance(record, dict):
                    continue
                url = str(record.get("requested_url") or "")
                success = record.get("success") is True
                content_chars = int(record.get("content_chars") or 0)
                message = (
                    f"网页读取完成：{url}，正文 {content_chars} 个字符"
                    if success
                    else f"网页读取失败：{url}"
                )
                events.append(
                    ResearchEvent(
                        "page_read_completed",
                        message,
                        {
                            "url": url,
                            "success": success,
                            "content_chars": content_chars,
                            "error": record.get("error"),
                        },
                        node,
                    )
                )

        gaps = raw_patch.get("research_gaps")
        if isinstance(gaps, list) and gaps:
            attempt = int(raw_patch.get("reflection_attempts") or 0)
            events.append(
                ResearchEvent(
                    "reflection",
                    f"研究检查发现 {len(gaps)} 个缺口",
                    {"attempt": attempt, "gaps": list(gaps)},
                    node,
                )
            )

        final_report = raw_patch.get("final_report")
        if isinstance(final_report, str) and final_report:
            events.append(
                ResearchEvent(
                    "report_created",
                    f"最终报告已写入 State，共 {len(final_report)} 个字符",
                    {"content_chars": len(final_report)},
                    node,
                )
            )

    return events
