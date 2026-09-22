import pytest
from langchain_core.messages import HumanMessage

from deepresearch.reporting import (
    calculate_stats,
    format_governance_summary,
    format_memory_summary,
    format_skill_summary,
    format_trace,
    save_markdown_report,
)
from deepresearch.state import ResearchState


def make_state() -> ResearchState:
    return {
        "messages": [HumanMessage(content="研究问题")],
        "research_question": "研究问题",
        "plan": {
            "goal": "完成研究问题",
            "steps": [
                {"step_id": "step-1", "title": "收集资料", "status": "completed"},
                {"step_id": "step-2", "title": "核验资料", "status": "in_progress"},
            ],
        },
        "current_step_id": "step-2",
        "reflection_attempts": 1,
        "research_gaps": ["step-2 尚未完成：核验资料"],
        "search_records": [
            {
                "query": "successful query",
                "success": True,
                "result_count": 3,
                "error": None,
                "step_id": "step-1",
            },
            {
                "query": "empty query",
                "success": True,
                "result_count": 0,
                "error": None,
            },
            {
                "query": "failed query",
                "success": False,
                "result_count": 0,
                "error": "network unavailable",
            },
        ],
        "page_records": [
            {
                "requested_url": "https://example.com/docs",
                "final_url": "https://example.com/docs",
                "success": True,
                "content_chars": 1234,
                "truncated": False,
                "error": None,
                "step_id": "step-2",
            },
            {
                "requested_url": "https://example.com/missing",
                "final_url": None,
                "success": False,
                "content_chars": 0,
                "truncated": False,
                "error": "HTTP 404",
            },
        ],
        "sources": [
            {
                "title": "Official documentation",
                "url": "https://example.com/docs",
                "snippet": "Example evidence",
                "query": "successful query",
            }
        ],
        "observations": [
            {
                "content": "Example evidence",
                "source_url": "https://example.com/docs",
                "query": "successful query",
            }
        ],
        "final_report": None,
    }


def test_calculate_stats_uses_recorded_state() -> None:
    stats = calculate_stats(make_state())

    assert stats.total_searches == 3
    assert stats.successful_searches == 1
    assert stats.empty_searches == 1
    assert stats.failed_searches == 1
    assert stats.returned_results == 3
    assert stats.unique_sources == 1
    assert stats.observation_count == 1
    assert stats.total_page_reads == 2
    assert stats.successful_page_reads == 1
    assert stats.failed_page_reads == 1


def test_format_trace_includes_queries_errors_and_sources() -> None:
    trace = format_trace(make_state())

    assert "程序统计，非 LLM 生成" in trace
    assert "搜索次数：3" in trace
    assert "empty query" in trace
    assert "network unavailable" in trace
    assert "Official documentation" in trace
    assert "https://example.com/docs" in trace
    assert "网页读取次数：2" in trace
    assert "HTTP 404" in trace
    assert "目标：完成研究问题" in trace
    assert "[进行中] step-2: 核验资料" in trace
    assert "计划步骤：step-1" in trace
    assert "反思次数：1" in trace
    assert "step-2 尚未完成：核验资料" in trace
    assert "正式报告：未生成" in trace
    assert "上下文治理：暂无记录" in trace


def test_format_governance_summary_displays_recorded_metrics() -> None:
    state = make_state()
    state["governance"] = {
        "context": {
            "model_name": "deepseek-chat",
            "budget": {
                "current_tokens": 50_000,
                "window_tokens": 1_048_576,
                "utilization_ratio": 50_000 / 1_048_576,
                "token_count_method": "char_estimate",
                "window_source": "model_map",
            },
            "cumulative_usage": {
                "input_tokens": 80_000,
                "output_tokens": 8_000,
                "total_tokens": 88_000,
            },
            "model_call_count": 7,
            "pending_stages": ["P1", "P2"],
            "hard_limit_reached": False,
            "seen_message_usage": {},
            "externalization": {
                "externalized_tool_results": 2,
                "original_chars": 20_000,
                "retained_chars": 1_000,
                "estimated_tokens_saved": 4_750,
                "last_externalized_paths": [
                    ".deepresearch/externalized/run/result.json"
                ],
                "last_error": None,
            },
            "summary": "压缩摘要",
            "summary_id": "p4-summary-1",
            "compaction": {
                "snapshot_count": 1,
                "summarize_count": 1,
                "removed_message_count": 12,
                "estimated_tokens_saved": 6_200,
                "last_preserved_message_count": 6,
                "last_tokens_before": 20_000,
                "last_tokens_after": 13_800,
                "last_snapshot_path": (
                    ".deepresearch/snapshots/run/snapshot.json"
                ),
                "last_summary_id": "p4-summary-1",
                "last_error": None,
            },
            "finalization": {
                "active": True,
                "redirect_count": 1,
                "blocked_tool_call_count": 2,
                "terminal_tool_call_count": 1,
                "forced_stop_count": 0,
                "last_blocked_tool_names": ["web_search", "read_page"],
                "last_utilization_ratio": 0.92,
                "last_reason": "p5_threshold",
                "last_reminder_id": "p5-reminder-1",
            },
        }
    }

    summary = format_governance_summary(state)

    assert "模型名称：deepseek-chat" in summary
    assert "上下文窗口：1,048,576 Token（来源：model_map）" in summary
    assert "当前上下文：50,000 Token（统计：char_estimate）" in summary
    assert "上下文占用：4.77%" in summary
    assert "累计 Token：88,000（输入 80,000 / 输出 8,000）" in summary
    assert "模型调用次数：7" in summary
    assert "待处理阶段：P1, P2" in summary
    assert "硬限制：否" in summary
    assert "P1 外化：2 个 Tool 结果；预计节省 4,750 Token" in summary
    assert ".deepresearch/externalized/run/result.json" in summary
    assert "P4 压缩：1 次摘要；1 个快照；累计移除 12 条消息" in summary
    assert "预计节省 6,200 Token" in summary
    assert ".deepresearch/snapshots/run/snapshot.json" in summary
    assert "P5 收尾：已触发；重定向 1 次；拦截 2 个 Tool Call" in summary
    assert "允许 1 个收尾 Tool Call；强制停止 0 次" in summary
    assert "最近拦截 Tool：web_search, read_page" in summary
    assert "seen_message_usage" not in summary


def test_format_governance_summary_accepts_old_state() -> None:
    assert format_governance_summary(make_state()) == "上下文治理：暂无记录"


def test_format_memory_summary_displays_runtime_metrics_and_accepts_old_state() -> None:
    state = make_state()
    assert format_memory_summary(state) == "长期记忆：暂无运行记录"
    state["memory"] = {
        "namespace": "project-a",
        "last_query_hash": "hash",
        "recalled": [{"id": "trace-1", "score": 0.8, "strength": 0.7}],
        "recall_count": 2,
        "injected_tokens": 88,
        "pending_write_count": 1,
        "last_error": None,
    }

    summary = format_memory_summary(state)

    assert "Namespace：project-a" in summary
    assert "召回执行次数：2" in summary
    assert "当前召回数量：1" in summary
    assert "记忆注入 Token：88" in summary
    assert "当前召回 ID：trace-1" in summary


def test_format_skill_summary_displays_versioned_runtime_metrics() -> None:
    state = make_state()
    assert format_skill_summary(state) == "Skills：暂无运行记录"
    state["skills"] = {
        "selection_count": 1,
        "selected": [
            {
                "skill_id": "verify__abc",
                "name": "verify",
                "content_hash": "abc",
                "score": 0.75,
                "reason": "bm25",
                "forced": False,
                "allowed_tools": ["read_page"],
            }
        ],
        "dropped": [],
        "injection_count": 1,
        "injected_tokens": 80,
        "aligned_tool_calls": 2,
        "completed_recorded": True,
        "last_error": None,
    }

    summary = format_skill_summary(state)

    assert "选中数量：1" in summary
    assert "verify [verify__abc]：0.7500" in summary
    assert "注入 Token：80" in summary
    assert "匹配 Tool Call：2" in summary


def test_save_markdown_report_writes_final_report(tmp_path) -> None:  # noqa: ANN001
    state = make_state()
    state["final_report"] = "# 测试报告\n\n正文"
    output_path = tmp_path / "reports" / "result.md"

    saved_path = save_markdown_report(state, output_path)

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8") == "# 测试报告\n\n正文\n"


def test_save_markdown_report_does_not_overwrite_existing_file(tmp_path) -> None:  # noqa: ANN001
    state = make_state()
    state["final_report"] = "# 新报告"
    output_path = tmp_path / "result.md"
    output_path.write_text("原有内容", encoding="utf-8")

    with pytest.raises(ValueError, match="不会覆盖"):
        save_markdown_report(state, output_path)

    assert output_path.read_text(encoding="utf-8") == "原有内容"
