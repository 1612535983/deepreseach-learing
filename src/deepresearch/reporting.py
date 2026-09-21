"""Deterministic reporting helpers for inspecting a completed research state."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.state import ResearchState


@dataclass(frozen=True)
class ResearchStats:
    """Counts calculated from recorded state without asking an LLM."""

    total_searches: int
    successful_searches: int
    empty_searches: int
    failed_searches: int
    returned_results: int
    unique_sources: int
    observation_count: int


def calculate_stats(state: ResearchState) -> ResearchStats:
    """Calculate mutually understandable counters from a completed state."""

    records = state.get("search_records", [])
    return ResearchStats(
        total_searches=len(records),
        successful_searches=sum(
            record["success"] and record["result_count"] > 0
            for record in records
        ),
        empty_searches=sum(
            record["success"] and record["result_count"] == 0
            for record in records
        ),
        failed_searches=sum(not record["success"] for record in records),
        returned_results=sum(record["result_count"] for record in records),
        unique_sources=len(state.get("sources", [])),
        observation_count=len(state.get("observations", [])),
    )


def format_trace(state: ResearchState) -> str:
    """Render searches and sources as a readable, non-LLM execution trace."""

    stats = calculate_stats(state)
    lines = [
        "--- Research Trace（程序统计，非 LLM 生成）---",
        f"搜索次数：{stats.total_searches}",
        f"成功并返回结果：{stats.successful_searches}",
        f"成功但无结果：{stats.empty_searches}",
        f"执行失败：{stats.failed_searches}",
        f"原始搜索结果：{stats.returned_results}",
        f"去重后来源：{stats.unique_sources}",
        f"Observation 数量：{stats.observation_count}",
        "",
        "搜索记录：",
    ]

    records = state.get("search_records", [])
    if not records:
        lines.append("（无）")
    for index, record in enumerate(records, start=1):
        if not record["success"]:
            status = "失败"
        elif record["result_count"] == 0:
            status = "成功但无结果"
        else:
            status = "成功"
        lines.append(f'{index}. {record["query"]}')
        lines.append(f'   状态：{status}；结果：{record["result_count"]}')
        if record["error"]:
            lines.append(f'   错误：{record["error"]}')

    lines.extend(["", "来源："])
    sources = state.get("sources", [])
    if not sources:
        lines.append("（无）")
    for index, source in enumerate(sources, start=1):
        lines.append(f'{index}. {source["title"] or "（无标题）"}')
        lines.append(f'   {source["url"]}')
        lines.append(f'   搜索词：{source["query"]}')

    return "\n".join(lines)

