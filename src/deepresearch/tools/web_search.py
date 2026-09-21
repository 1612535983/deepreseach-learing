"""DuckDuckGo search exposed as a LangChain tool."""

from __future__ import annotations

import json
from typing import Any

from ddgs import DDGS
from langchain_core.tools import tool


DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10


def _error_payload(query: str, message: str) -> str:
    return json.dumps(
        {"ok": False, "query": query, "error": message},
        ensure_ascii=False,
    )


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the public web for current information and possible sources.

    Args:
        query: Concise search keywords describing the information to find.
        max_results: Number of results to return, from 1 to 10.
    """

    normalized_query = query.strip()
    if not normalized_query:
        return _error_payload(query, "Search query cannot be empty")
    if max_results < 1:
        return _error_payload(normalized_query, "max_results must be at least 1")

    result_limit = min(max_results, MAX_RESULTS_LIMIT)
    try:
        rows = DDGS(timeout=30).text(
            normalized_query,
            region="wt-wt",
            safesearch="moderate",
            max_results=result_limit,
            backend="duckduckgo",
        )
        raw_results: list[dict[str, Any]] = list(rows or [])
    except Exception as exc:
        return _error_payload(normalized_query, f"Search failed: {exc}")

    results = [
        {
            "title": str(row.get("title", "")),
            "url": str(row.get("href") or row.get("url") or row.get("link") or ""),
            "snippet": str(row.get("body") or row.get("snippet") or ""),
        }
        for row in raw_results[:result_limit]
    ]
    return json.dumps(
        {
            "ok": True,
            "query": normalized_query,
            "total_results": len(results),
            "results": results,
        },
        ensure_ascii=False,
    )

