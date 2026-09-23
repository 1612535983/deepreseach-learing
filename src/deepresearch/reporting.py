"""Deterministic reporting helpers for inspecting a completed research state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepresearch.events import ResearchEvent
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


def format_governance_summary(state: Mapping[str, Any]) -> str:
    """Render checkpoint-safe context measurements without exposing ledgers."""

    governance = _mapping(state.get("governance"))
    context = _mapping(governance.get("context"))
    if not context:
        return "上下文治理：暂无记录"

    budget = _mapping(context.get("budget"))
    usage = _mapping(context.get("cumulative_usage"))
    externalization = _mapping(context.get("externalization"))
    compaction = _mapping(context.get("compaction"))
    finalization = _mapping(context.get("finalization"))
    model_name = context.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = "未知"

    pending_value = context.get("pending_stages")
    if isinstance(pending_value, (list, tuple)):
        pending_stages = ", ".join(
            str(stage) for stage in pending_value if str(stage).strip()
        )
    else:
        pending_stages = ""

    window_source = budget.get("window_source")
    if not isinstance(window_source, str) or not window_source:
        window_source = "unknown"
    token_count_method = budget.get("token_count_method")
    if not isinstance(token_count_method, str) or not token_count_method:
        token_count_method = "unknown"

    lines = [
        "上下文治理：",
        f"模型名称：{model_name}",
        (
            f"上下文窗口：{_non_negative_int(budget.get('window_tokens')):,} "
            f"Token（来源：{window_source}）"
        ),
        (
            f"当前上下文：{_non_negative_int(budget.get('current_tokens')):,} "
            f"Token（统计：{token_count_method}）"
        ),
        f"上下文占用：{_ratio(budget.get('utilization_ratio')):.2%}",
        (
            f"累计 Token：{_non_negative_int(usage.get('total_tokens')):,}"
            f"（输入 {_non_negative_int(usage.get('input_tokens')):,} / "
            f"输出 {_non_negative_int(usage.get('output_tokens')):,}）"
        ),
        f"模型调用次数：{_non_negative_int(context.get('model_call_count')):,}",
        f"待处理阶段：{pending_stages or '无'}",
        f"硬限制：{'是' if context.get('hard_limit_reached') is True else '否'}",
    ]
    if externalization:
        lines.append(
            "P1 外化："
            f"{_non_negative_int(externalization.get('externalized_tool_results')):,} "
            "个 Tool 结果；预计节省 "
            f"{_non_negative_int(externalization.get('estimated_tokens_saved')):,} Token"
        )
        paths = externalization.get("last_externalized_paths")
        if isinstance(paths, list) and paths:
            lines.append("最近外化文件：")
            lines.extend(f"- {path}" for path in paths)
        last_error = externalization.get("last_error")
        if isinstance(last_error, str) and last_error:
            lines.append(f"最近外化错误：{last_error}")
    if compaction:
        lines.append(
            "P4 压缩："
            f"{_non_negative_int(compaction.get('summarize_count')):,} 次摘要；"
            f"{_non_negative_int(compaction.get('snapshot_count')):,} 个快照；"
            f"累计移除 {_non_negative_int(compaction.get('removed_message_count')):,} 条消息；"
            "预计节省 "
            f"{_non_negative_int(compaction.get('estimated_tokens_saved')):,} Token"
        )
        snapshot_path = compaction.get("last_snapshot_path")
        if isinstance(snapshot_path, str) and snapshot_path:
            lines.append(f"最近 P4 快照：{snapshot_path}")
        compaction_error = compaction.get("last_error")
        if isinstance(compaction_error, str) and compaction_error:
            lines.append(f"最近 P4 错误：{compaction_error}")
    if finalization:
        lines.append(
            "P5/研究预算收尾："
            f"{'已触发' if finalization.get('active') is True else '未触发'}；"
            f"重定向 {_non_negative_int(finalization.get('redirect_count')):,} 次；"
            "拦截 "
            f"{_non_negative_int(finalization.get('blocked_tool_call_count')):,} "
            "个 Tool Call；允许 "
            f"{_non_negative_int(finalization.get('terminal_tool_call_count')):,} "
            "个收尾 Tool Call；强制停止 "
            f"{_non_negative_int(finalization.get('forced_stop_count')):,} 次"
        )
        trigger_reason = finalization.get("trigger_reason")
        if isinstance(trigger_reason, str) and trigger_reason:
            lines.append(f"收尾触发原因：{trigger_reason}")
        blocked_names = finalization.get("last_blocked_tool_names")
        if isinstance(blocked_names, list) and blocked_names:
            lines.append(
                "最近拦截 Tool："
                + ", ".join(str(name) for name in blocked_names)
            )
    return "\n".join(lines)


def format_memory_summary(state: Mapping[str, Any]) -> str:
    """Render the compact runtime view without exposing stored memory bodies."""

    memory = _mapping(state.get("memory"))
    if not memory:
        return "长期记忆：暂无运行记录"
    recalled_value = memory.get("recalled")
    recalled = recalled_value if isinstance(recalled_value, list) else []
    recalled_ids = [
        str(item.get("id"))
        for item in recalled
        if isinstance(item, Mapping) and item.get("id")
    ]
    lines = [
        "长期记忆：",
        f"Namespace：{memory.get('namespace') or 'default'}",
        f"召回执行次数：{_non_negative_int(memory.get('recall_count')):,}",
        f"当前召回数量：{len(recalled):,}",
        f"记忆注入 Token：{_non_negative_int(memory.get('injected_tokens')):,}",
        f"待处理写任务：{_non_negative_int(memory.get('pending_write_count')):,}",
        f"当前召回 ID：{', '.join(recalled_ids) if recalled_ids else '无'}",
    ]
    last_error = memory.get("last_error")
    if isinstance(last_error, str) and last_error:
        lines.append(f"最近记忆错误：{last_error}")
    return "\n".join(lines)


def format_skill_summary(state: Mapping[str, Any]) -> str:
    """Render selected immutable Skill references and runtime counters."""

    skills = _mapping(state.get("skills"))
    if not skills:
        return "Skills：暂无运行记录"
    selected_value = skills.get("selected")
    selected = selected_value if isinstance(selected_value, list) else []
    dropped_value = skills.get("dropped")
    dropped = dropped_value if isinstance(dropped_value, list) else []
    lines = [
        "Skills：",
        f"选择执行次数：{_non_negative_int(skills.get('selection_count')):,}",
        f"选中数量：{len(selected):,}",
        f"注入执行次数：{_non_negative_int(skills.get('injection_count')):,}",
        f"注入 Token：{_non_negative_int(skills.get('injected_tokens')):,}",
        f"预算丢弃数量：{len(dropped):,}",
        f"匹配 Tool Call：{_non_negative_int(skills.get('aligned_tool_calls')):,}",
        (
            "运行评估："
            f"{skills.get('evaluation_status') or '未执行'}；"
            f"{_non_negative_int(skills.get('evaluation_count')):,} 个版本"
        ),
        "选中版本：",
    ]
    if not selected:
        lines.append("- 无")
    for ref in selected:
        if not isinstance(ref, Mapping):
            continue
        skill_id = str(ref.get("skill_id") or "unknown")
        score = _ratio(ref.get("score"))
        forced = "；强制" if ref.get("forced") is True else ""
        lines.append(
            f"- {ref.get('name') or 'unknown'} [{skill_id}]：{score:.4f}{forced}"
        )
    for ref in dropped:
        if isinstance(ref, Mapping):
            lines.append(
                f"- 未注入 {ref.get('name') or 'unknown'}："
                f"{ref.get('drop_reason') or 'unknown'}"
            )
    last_error = skills.get("last_error")
    if isinstance(last_error, str) and last_error:
        lines.append(f"最近 Skill 错误：{last_error}")
    evaluation_error = skills.get("evaluation_error")
    if isinstance(evaluation_error, str) and evaluation_error:
        lines.append(f"最近 Skill 评估错误：{evaluation_error}")
    return "\n".join(lines)


def format_evaluation_summary(state: Mapping[str, Any]) -> str:
    """Render report probabilities and runtime cost without report contents."""

    report = _mapping(_mapping(state.get("evaluation")).get("report"))
    status = str(report.get("status") or "not_run")
    if status == "not_run" or not report:
        return "Jev 报告评估：未执行"

    score = report.get("composite_score")
    score_text = (
        f"{float(score):.1%}"
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else "未知"
    )
    cost = report.get("cost_usd")
    cost_text = (
        f"${float(cost):.6f}"
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        else "未提供"
    )
    lines = [
        "Jev 报告评估：",
        f"状态：{status}；模式：{report.get('mode') or 'unknown'}",
        f"Provider/模型：{report.get('provider') or '未知'} / {report.get('model') or '未知'}",
        f"综合质量分：{score_text}",
        f"建议动作：{report.get('recommended_action') or '无'}",
        f"实际动作：{report.get('runtime_action') or '无'}",
        (
            f"调用统计：{_non_negative_int(report.get('latency_ms'))} ms；"
            f"输入 {_non_negative_int(report.get('input_tokens'))} Token；"
            f"输出 {_non_negative_int(report.get('output_tokens'))} Token；"
            f"成本 {cost_text}"
        ),
        (
            f"执行次数：{_non_negative_int(report.get('evaluation_count'))}；"
            f"Gate 次数：{_non_negative_int(report.get('gate_attempts'))}"
        ),
    ]
    answer_labels = {
        "answer_relevance": "回答相关",
        "evidence_support": "证据支持",
        "citation_coverage": "引用充分",
        "evidence_sufficient": "证据足够",
        "continue_research": "继续研究",
        "source_quality": "来源质量",
    }
    answers = _mapping(report.get("answers"))
    if answers:
        lines.append("概率与评分：")
        for name, label in answer_labels.items():
            answer = _mapping(answers.get(name))
            value = answer.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            confidence = answer.get("confidence")
            suffix = (
                f"（置信度 {float(confidence):.1%}）"
                if isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                else ""
            )
            if answer.get("type") == "score":
                lines.append(f"- {label}：{float(value):.2f}{suffix}")
            else:
                lines.append(f"- {label}：{float(value):.1%}{suffix}")
    last_error = report.get("last_error")
    if isinstance(last_error, str) and last_error:
        lines.append(f"最近评估错误：{last_error}")
    return "\n".join(lines)


def format_evaluation_markdown(state: Mapping[str, Any]) -> str:
    """Render a deterministic Jev appendix for an exported Markdown report."""

    report = _mapping(_mapping(state.get("evaluation")).get("report"))
    status = str(report.get("status") or "not_run")
    if status == "not_run" or not report:
        return ""

    mode = str(report.get("mode") or "unknown")
    mode_labels = {
        "shadow": "Shadow（仅观察，不自动修改报告）",
        "gate": "Gate（可触发一次有界修订）",
    }
    action_labels = {
        "pass": "通过",
        "revise_report": "修订报告",
        "continue_research": "继续研究",
        "review": "人工复核",
        "observed": "仅记录结果",
        "review_required": "需要人工复核",
        "p5_bypass": "P5 强制收尾，未继续干预",
        "gate_exhausted": "Gate 次数已用尽",
        "fail_open": "评估失败，研究流程继续",
    }
    lines = [
        "---",
        "",
        "## Jev 报告质量评估",
        "",
        "> 本节由程序在报告生成后追加，不属于研究正文。",
        "",
        f"- 状态：{status}",
        f"- 模式：{mode_labels.get(mode, mode)}",
    ]

    if status == "error":
        error = report.get("last_error")
        lines.append(
            f"- 错误：{error if isinstance(error, str) and error else '评估失败'}"
        )
        lines.append("- 运行策略：Fail-open（评估失败不阻断报告生成）")
        return "\n".join(lines)

    score = report.get("composite_score")
    score_text = (
        f"{float(score):.1%}"
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else "未知"
    )
    recommended = str(report.get("recommended_action") or "无")
    runtime = str(report.get("runtime_action") or "无")
    lines.extend(
        [
            (
                "- Provider / 模型："
                f"{report.get('provider') or '未知'} / {report.get('model') or '未知'}"
            ),
            f"- 综合质量分：**{score_text}**",
            f"- 建议动作：{action_labels.get(recommended, recommended)}",
            f"- 实际动作：{action_labels.get(runtime, runtime)}",
            "",
            "### 分项结果",
            "",
            "| 指标 | 结果 | 置信度 |",
            "| --- | ---: | ---: |",
        ]
    )
    answer_labels = {
        "answer_relevance": "回答相关概率",
        "evidence_support": "证据支持概率",
        "citation_coverage": "引用充分概率",
        "evidence_sufficient": "证据足够概率",
        "continue_research": "继续研究概率",
        "source_quality": "来源质量评分",
    }
    answers = _mapping(report.get("answers"))
    for name, label in answer_labels.items():
        answer = _mapping(answers.get(name))
        value = answer.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        result_text = (
            f"{float(value):.2f} / 3"
            if answer.get("type") == "score"
            else f"{float(value):.1%}"
        )
        confidence = answer.get("confidence")
        confidence_text = (
            f"{float(confidence):.1%}"
            if isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            else "—"
        )
        lines.append(f"| {label} | {result_text} | {confidence_text} |")

    cost = report.get("cost_usd")
    cost_text = (
        f"${float(cost):.6f}"
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        else "Provider 未提供"
    )
    lines.extend(
        [
            "",
            "### 调用统计",
            "",
            f"- 延迟：{_non_negative_int(report.get('latency_ms'))} ms",
            (
                f"- Token：输入 {_non_negative_int(report.get('input_tokens'))} / "
                f"输出 {_non_negative_int(report.get('output_tokens'))}"
            ),
            f"- 成本：{cost_text}",
            f"- 评估时间：{report.get('evaluated_at') or '未知'}",
        ]
    )
    return "\n".join(lines)


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


def save_markdown_report(
    state: ResearchState,
    output_path: str | Path,
) -> Path:
    """Write ``final_report`` and its Jev evaluation to a new Markdown file."""

    report = state.get("final_report")
    if not report:
        raise ValueError("State 中没有可以保存的 final_report。")

    path = Path(output_path)
    if path.suffix.lower() != ".md":
        raise ValueError("报告输出路径必须以 .md 结尾。")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output_file:
            report_text = str(report)
            output_file.write(report_text)
            if not report_text.endswith("\n"):
                output_file.write("\n")
            evaluation_markdown = format_evaluation_markdown(state)
            if evaluation_markdown:
                output_file.write("\n")
                output_file.write(evaluation_markdown)
                output_file.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"报告文件已存在，不会覆盖：{path}") from exc
    except OSError as exc:
        raise ValueError(f"无法写入报告文件 {path}：{exc}") from exc
    return path


def format_research_event(event: ResearchEvent) -> str:
    """Render one stable progress event without exposing raw State or page text."""

    labels = {
        "run_started": "开始",
        "tool_requested": "工具",
        "tool_completed": "工具",
        "plan_updated": "计划",
        "search_completed": "搜索",
        "page_read_completed": "读取",
        "reflection": "反思",
        "skill_selected": "Skill",
        "skill_injected": "Skill",
        "skill_metrics": "Skill",
        "skill_evaluated": "Skill",
        "report_created": "报告",
        "report_evaluated": "评估",
        "finalization": "收尾",
        "run_completed": "完成",
        "run_failed": "失败",
    }
    lines = [f"[{labels[event.event_type]}] {event.message}"]
    if event.event_type == "run_started" and event.data.get("thread_id"):
        lines.append(f'  任务 ID：{event.data["thread_id"]}')
    if event.event_type == "reflection":
        gaps = event.data.get("gaps")
        if isinstance(gaps, list):
            lines.extend(f"  - {gap}" for gap in gaps)
    error = event.data.get("error")
    if error:
        lines.append(f"  错误：{error}")
    return "\n".join(lines)


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

    lines.extend(["", *format_governance_summary(state).splitlines()])
    lines.extend(["", *format_memory_summary(state).splitlines()])
    lines.extend(["", *format_skill_summary(state).splitlines()])
    lines.extend(["", *format_evaluation_summary(state).splitlines()])

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
            f'正式报告：{"已生成" if state.get("final_report") else "未生成"}',
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
