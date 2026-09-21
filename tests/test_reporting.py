import pytest
from langchain_core.messages import HumanMessage

from deepresearch.reporting import calculate_stats, format_trace, save_markdown_report
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
