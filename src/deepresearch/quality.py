"""Shared, deterministic rules for deciding whether research is complete."""

from __future__ import annotations

from deepresearch.state import ResearchState


MIN_UNIQUE_SOURCES = 2
MIN_SUCCESSFUL_PAGE_READS = 1
MIN_OBSERVATIONS = 2


def assess_research(
    state: ResearchState,
    *,
    require_final_report: bool = True,
) -> list[str]:
    """Return concrete gaps found in the structured research state."""

    gaps: list[str] = []
    plan = state.get("plan")
    if not plan:
        gaps.append("尚未创建研究计划。")
    else:
        for step in plan["steps"]:
            if step["status"] in {"pending", "in_progress"}:
                gaps.append(f'{step["step_id"]} 尚未完成：{step["title"]}')

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

    if require_final_report and not state.get("final_report"):
        gaps.append("尚未生成正式研究报告。")

    return gaps
