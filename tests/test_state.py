from deepresearch.state import (
    append_observations,
    append_search_records,
    create_initial_state,
    merge_final_report,
    merge_sources,
)


def test_create_initial_state_contains_research_fields() -> None:
    state = create_initial_state("研究 LangChain")

    assert state["research_question"] == "研究 LangChain"
    assert state["messages"][0].content == "研究 LangChain"
    assert state["search_records"] == []
    assert state["sources"] == []
    assert state["observations"] == []
    assert state["final_report"] is None


def test_search_records_and_observations_append() -> None:
    records = append_search_records(
        [{"query": "first", "success": True, "result_count": 1, "error": None}],
        [{"query": "second", "success": False, "result_count": 0, "error": "failed"}],
    )
    observations = append_observations(
        [{"content": "A", "source_url": "https://a.example", "query": "first"}],
        [{"content": "B", "source_url": "https://b.example", "query": "second"}],
    )

    assert [record["query"] for record in records] == ["first", "second"]
    assert [item["content"] for item in observations] == ["A", "B"]


def test_sources_are_deduplicated_by_url() -> None:
    first = {
        "title": "Old title",
        "url": "https://example.com/article",
        "snippet": "Old snippet",
        "query": "first query",
    }
    updated = {
        "title": "New title",
        "url": "https://example.com/article",
        "snippet": "New snippet",
        "query": "second query",
    }
    other = {
        "title": "Other",
        "url": "https://other.example/article",
        "snippet": "Other snippet",
        "query": "second query",
    }

    merged = merge_sources([first], [updated, other])

    assert merged == [updated, other]


def test_final_report_uses_latest_non_null_value() -> None:
    assert merge_final_report("old report", None) == "old report"
    assert merge_final_report("old report", "new report") == "new report"
