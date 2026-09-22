import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import build_agent
from deepresearch.state import ResearchState, create_initial_state
from deepresearch.tools import (
    read_page_tool,
    web_search_tool,
    write_final_report_tool,
    write_research_plan_tool,
)
from deepresearch.tools.final_report import prepare_final_report


SOURCE_A = "https://a.example/article"
SOURCE_B = "https://b.example/article"


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self


def make_ready_state() -> ResearchState:
    state = create_initial_state("研究问题")
    state["plan"] = {
        "goal": "回答研究问题",
        "steps": [
            {"step_id": "step-1", "title": "搜索", "status": "completed"},
            {"step_id": "step-2", "title": "核验", "status": "completed"},
        ],
    }
    state["current_step_id"] = None
    state["sources"] = [
        {"title": "A", "url": SOURCE_A, "snippet": "A", "query": "query"},
        {"title": "B", "url": SOURCE_B, "snippet": "B", "query": "query"},
    ]
    state["page_records"] = [
        {
            "requested_url": SOURCE_A,
            "final_url": SOURCE_A,
            "success": True,
            "content_chars": 100,
            "truncated": False,
            "error": None,
        }
    ]
    state["observations"] = [
        {"content": "Evidence A", "source_url": SOURCE_A, "query": "query"},
        {"content": "Evidence B", "source_url": SOURCE_B, "query": "query"},
    ]
    return state


def report_body() -> str:
    return (
        "## 结论\n\n测试结论。\n\n"
        "## 来源\n\n"
        f"- [Source A]({SOURCE_A})\n"
        f"- [Source B]({SOURCE_B})"
    )


def test_prepare_final_report_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="were not collected"):
        prepare_final_report(
            "研究报告",
            report_body() + "\n- https://unknown.example",
            [SOURCE_A, "https://unknown.example"],
            make_ready_state(),
        )


def test_write_final_report_updates_state_through_agent_graph() -> None:
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {
                            "title": "研究报告",
                            "report": report_body(),
                            "used_source_urls": [SOURCE_A, SOURCE_B],
                        },
                        "id": "report-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="报告已经生成。"),
        ]
    )
    graph = build_agent(
        model,
        tools=[
            write_research_plan_tool,
            web_search_tool,
            read_page_tool,
            write_final_report_tool,
        ],
    )

    final_state = graph.invoke(make_ready_state())

    assert final_state["final_report"].startswith("# 研究报告\n\n## 结论")
    assert SOURCE_A in final_state["final_report"]
    assert SOURCE_B in final_state["final_report"]
    assert final_state["final_report_source_urls"] == [SOURCE_A, SOURCE_B]
    assert final_state["research_gaps"] == []
