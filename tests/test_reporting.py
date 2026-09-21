from langchain_core.messages import HumanMessage

from deepresearch.reporting import calculate_stats, format_trace
from deepresearch.state import ResearchState


def make_state() -> ResearchState:
    return {
        "messages": [HumanMessage(content="研究问题")],
        "research_question": "研究问题",
        "search_records": [
            {
                "query": "successful query",
                "success": True,
                "result_count": 3,
                "error": None,
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


def test_format_trace_includes_queries_errors_and_sources() -> None:
    trace = format_trace(make_state())

    assert "程序统计，非 LLM 生成" in trace
    assert "搜索次数：3" in trace
    assert "empty query" in trace
    assert "network unavailable" in trace
    assert "Official documentation" in trace
    assert "https://example.com/docs" in trace

