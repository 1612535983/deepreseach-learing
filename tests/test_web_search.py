import importlib
import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_with_model
from deepresearch.tools import web_search_tool


search_module = importlib.import_module("deepresearch.tools.web_search")


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """A deterministic fake model that accepts LangChain tool binding."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self


class FakeDDGS:
    queries: list[str] = []

    def __init__(self, timeout: int) -> None:
        assert timeout == 30

    def text(self, query: str, **kwargs):  # noqa: ANN003, ANN201
        self.queries.append(query)
        return [
            {
                "title": "Example title",
                "href": "https://example.com/article",
                "body": "Example summary",
            }
        ]


def test_web_search_normalizes_results(monkeypatch) -> None:  # noqa: ANN001
    FakeDDGS.queries.clear()
    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)

    payload = json.loads(
        web_search_tool.invoke({"query": "agent frameworks", "max_results": 3})
    )

    assert payload["ok"] is True
    assert payload["total_results"] == 1
    assert payload["results"] == [
        {
            "title": "Example title",
            "url": "https://example.com/article",
            "snippet": "Example summary",
        }
    ]
    assert FakeDDGS.queries == ["agent frameworks"]


def test_web_search_rejects_empty_query() -> None:
    payload = json.loads(web_search_tool.invoke({"query": "   "}))

    assert payload["ok"] is False
    assert "empty" in payload["error"]


def test_search_failure_becomes_structured_tool_result(monkeypatch) -> None:  # noqa: ANN001
    class BrokenDDGS:
        def __init__(self, timeout: int) -> None:
            pass

        def text(self, query: str, **kwargs):  # noqa: ANN003, ANN201
            raise ConnectionError("network unavailable")

    monkeypatch.setattr(search_module, "DDGS", BrokenDDGS)

    payload = json.loads(web_search_tool.invoke({"query": "test"}))

    assert payload["ok"] is False
    assert "network unavailable" in payload["error"]


def test_agent_completes_tool_call_loop(monkeypatch) -> None:  # noqa: ANN001
    FakeDDGS.queries.clear()
    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "latest agent news", "max_results": 2},
                        "id": "search-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="研究完成。来源：https://example.com/article"
            ),
        ]
    )

    result = run_with_model("搜索最新消息", model, tools=[web_search_tool])

    assert FakeDDGS.queries == ["latest agent news"]
    assert "https://example.com/article" in result.answer
    assert result.state["search_records"] == [
        {
            "query": "latest agent news",
            "success": True,
            "result_count": 1,
            "error": None,
        }
    ]
    assert result.state["sources"][0]["url"] == "https://example.com/article"
    assert result.state["observations"][0]["content"] == "Example summary"


def test_repeated_searches_append_records_and_deduplicate_sources(
    monkeypatch,
) -> None:  # noqa: ANN001
    FakeDDGS.queries.clear()
    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "first query", "max_results": 2},
                        "id": "search-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "second query", "max_results": 2},
                        "id": "search-call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="两轮搜索完成"),
        ]
    )

    result = run_with_model("执行两轮搜索", model, tools=[web_search_tool])

    assert FakeDDGS.queries == ["first query", "second query"]
    assert [item["query"] for item in result.state["search_records"]] == [
        "first query",
        "second query",
    ]
    assert len(result.state["sources"]) == 1
    assert len(result.state["observations"]) == 2
