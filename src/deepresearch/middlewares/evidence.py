"""Collect structured research evidence after each web search tool call."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepresearch.state import Observation, ResearchState, SearchRecord, Source


class EvidenceUpdate(TypedDict):
    """A state patch derived from one web search result."""

    search_records: list[SearchRecord]
    sources: list[Source]
    observations: list[Observation]


def _tool_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def parse_web_search_evidence(
    content: str,
    fallback_query: str = "",
) -> EvidenceUpdate:
    """Convert one web_search JSON response into a ResearchState patch."""

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        record: SearchRecord = {
            "query": fallback_query,
            "success": False,
            "result_count": 0,
            "error": f"Invalid web_search response: {exc}",
        }
        return {"search_records": [record], "sources": [], "observations": []}

    if not isinstance(payload, dict):
        record = {
            "query": fallback_query,
            "success": False,
            "result_count": 0,
            "error": "Invalid web_search response: expected a JSON object",
        }
        return {"search_records": [record], "sources": [], "observations": []}

    query = str(payload.get("query") or fallback_query).strip()
    success = payload.get("ok") is True
    raw_results = payload.get("results")
    results = raw_results if isinstance(raw_results, list) else []
    error = None if success else str(payload.get("error") or "Unknown search error")

    record = {
        "query": query,
        "success": success,
        "result_count": len(results),
        "error": error,
    }
    sources: list[Source] = []
    observations: list[Observation] = []
    if success:
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "query": query,
                }
            )
            if snippet:
                observations.append(
                    {
                        "content": snippet,
                        "source_url": url,
                        "query": query,
                    }
                )

    return {
        "search_records": [record],
        "sources": sources,
        "observations": observations,
    }


class EvidenceMiddleware(AgentMiddleware):
    """Persist web_search attempts and results alongside the message history."""

    state_schema = ResearchState

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)
        if request.tool_call.get("name") != "web_search" or not isinstance(
            result, ToolMessage
        ):
            return result

        args = request.tool_call.get("args") or {}
        fallback_query = str(args.get("query") or "")
        update = parse_web_search_evidence(_tool_text(result), fallback_query)
        return Command(update={"messages": [result], **update})

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if request.tool_call.get("name") != "web_search" or not isinstance(
            result, ToolMessage
        ):
            return result

        args = request.tool_call.get("args") or {}
        fallback_query = str(args.get("query") or "")
        update = parse_web_search_evidence(_tool_text(result), fallback_query)
        return Command(update={"messages": [result], **update})

