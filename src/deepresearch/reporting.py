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
    total_page_reads: int
    successful_page_reads: int
    failed_page_reads: int


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
        total_page_reads=len(state.get("page_records", [])),
        successful_page_reads=sum(
            record["success"] for record in state.get("page_records", [])
        ),
        failed_page_reads=sum(
            not record["success"] for record in state.get("page_records", [])
        ),
    )


def format_trace(state: ResearchState) -> str:
    """Render searches and sources as a readable, non-LLM execution trace."""

    stats = calculate_stats(state)
    lines = ["--- Research Trace（程序统计，非 LLM 生成）---", "研究计划："]
    plan = state.get("plan")
    if not plan:
        lines.append("（无）")
    else:
        status_labels = {
            "pending": "待处理",
            "in_progress": "进行中",
            "completed": "已完成",
            "failed": "失败",
        }
        lines.append(f'目标：{plan["goal"]}')
        lines.append(f'当前步骤：{state.get("current_step_id") or "无"}')
        for step in plan["steps"]:
            lines.append(
                f'- [{status_labels[step["status"]]}] '
                f'{step["step_id"]}: {step["title"]}'
            )

    lines.extend(
        [
            "",
            f'反思次数：{state.get("reflection_attempts", 0)}',
            "剩余研究缺口：",
        ]
    )
    research_gaps = state.get("research_gaps", [])
    if not research_gaps:
        lines.append("（无）")
    else:
        lines.extend(f"- {gap}" for gap in research_gaps)

    lines.extend(
        [
            "",
            f"搜索次数：{stats.total_searches}",
            f"成功并返回结果：{stats.successful_searches}",
            f"成功但无结果：{stats.empty_searches}",
            f"执行失败：{stats.failed_searches}",
            f"原始搜索结果：{stats.returned_results}",
            f"去重后来源：{stats.unique_sources}",
            f"Observation 数量：{stats.observation_count}",
            f"网页读取次数：{stats.total_page_reads}",
            f"网页读取成功：{stats.successful_page_reads}",
            f"网页读取失败：{stats.failed_page_reads}",
            "",
            "搜索记录：",
        ]
    )

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
        if record.get("step_id"):
            lines.append(f'   计划步骤：{record["step_id"]}')
        if record["error"]:
            lines.append(f'   错误：{record["error"]}')

    lines.extend(["", "网页读取记录："])
    page_records = state.get("page_records", [])
    if not page_records:
        lines.append("（无）")
    for index, record in enumerate(page_records, start=1):
        status = "成功" if record["success"] else "失败"
        lines.append(f'{index}. {record["requested_url"]}')
        lines.append(
            f'   状态：{status}；正文字符：{record["content_chars"]}；'
            f'截断：{"是" if record["truncated"] else "否"}'
        )
        if record.get("step_id"):
            lines.append(f'   计划步骤：{record["step_id"]}')
        if record["final_url"] and record["final_url"] != record["requested_url"]:
            lines.append(f'   最终地址：{record["final_url"]}')
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
