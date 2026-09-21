import json

from deepresearch.middlewares.evidence import (
    parse_read_page_evidence,
    parse_web_search_evidence,
)


def test_successful_search_becomes_structured_evidence() -> None:
    update = parse_web_search_evidence(
        json.dumps(
            {
                "ok": True,
                "query": "LangChain Agent",
                "total_results": 2,
                "results": [
                    {
                        "title": "LangChain Agents",
                        "url": "https://docs.langchain.com/agents",
                        "snippet": "Agent documentation",
                    },
                    {
                        "title": "No URL",
                        "url": "",
                        "snippet": "This result cannot be traced",
                    },
                ],
            }
        )
    )

    assert update["search_records"] == [
        {
            "query": "LangChain Agent",
            "success": True,
            "result_count": 2,
            "error": None,
        }
    ]
    assert len(update["sources"]) == 1
    assert update["sources"][0]["url"] == "https://docs.langchain.com/agents"
    assert update["observations"] == [
        {
            "content": "Agent documentation",
            "source_url": "https://docs.langchain.com/agents",
            "query": "LangChain Agent",
        }
    ]


def test_successful_empty_search_is_not_a_tool_failure() -> None:
    update = parse_web_search_evidence(
        json.dumps(
            {
                "ok": True,
                "query": "rare query",
                "total_results": 0,
                "results": [],
            }
        )
    )

    assert update["search_records"][0] == {
        "query": "rare query",
        "success": True,
        "result_count": 0,
        "error": None,
    }
    assert update["sources"] == []
    assert update["observations"] == []


def test_search_error_becomes_failed_record() -> None:
    update = parse_web_search_evidence(
        json.dumps(
            {
                "ok": False,
                "query": "LangChain Agent",
                "error": "network unavailable",
            }
        )
    )

    assert update["search_records"][0] == {
        "query": "LangChain Agent",
        "success": False,
        "result_count": 0,
        "error": "network unavailable",
    }


def test_invalid_tool_json_becomes_failed_record() -> None:
    update = parse_web_search_evidence("not-json", "fallback query")

    record = update["search_records"][0]
    assert record["query"] == "fallback query"
    assert record["success"] is False
    assert record["result_count"] == 0
    assert record["error"].startswith("Invalid web_search response")


def test_successful_page_read_becomes_page_evidence() -> None:
    update = parse_read_page_evidence(
        json.dumps(
            {
                "ok": True,
                "requested_url": "https://example.com/start",
                "final_url": "https://example.com/article",
                "title": "Example article",
                "content": "Full page evidence",
                "truncated": False,
            }
        ),
        step_id="step-2",
    )

    assert update["page_records"] == [
        {
            "requested_url": "https://example.com/start",
            "final_url": "https://example.com/article",
            "success": True,
            "content_chars": 18,
            "truncated": False,
            "error": None,
            "step_id": "step-2",
        }
    ]
    assert update["sources"][0]["url"] == "https://example.com/article"
    assert update["observations"] == [
        {
            "content": "Full page evidence",
            "source_url": "https://example.com/article",
            "query": "",
            "evidence_type": "page_content",
            "step_id": "step-2",
        }
    ]


def test_failed_page_read_records_error_without_evidence() -> None:
    update = parse_read_page_evidence(
        json.dumps(
            {
                "ok": False,
                "requested_url": "http://127.0.0.1/private",
                "error": "Private addresses are not allowed",
            }
        )
    )

    assert update["page_records"][0]["success"] is False
    assert update["page_records"][0]["error"] == "Private addresses are not allowed"
    assert update["sources"] == []
    assert update["observations"] == []
