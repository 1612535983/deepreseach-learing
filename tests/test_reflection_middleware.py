from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_with_model
from deepresearch.middlewares.reflection import (
    MAX_REFLECTION_ATTEMPTS,
    ReflectionMiddleware,
    assess_research,
)
from deepresearch.state import ResearchState, create_initial_state
from deepresearch.tools import (
    read_page_tool,
    web_search_tool,
    write_final_report_tool,
    write_research_plan_tool,
)


class CountingToolModel(FakeMessagesListChatModel):
    call_count: ClassVar[int] = 0

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        type(self).call_count += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_complete_state() -> ResearchState:
    state = create_initial_state("研究问题")
    state["plan"] = {
        "goal": "完成研究",
        "steps": [
            {"step_id": "step-1", "title": "搜索", "status": "completed"},
            {"step_id": "step-2", "title": "核验", "status": "completed"},
        ],
    }
    state["sources"] = [
        {
            "title": "Source A",
            "url": "https://a.example",
            "snippet": "Evidence A",
            "query": "query",
        },
        {
            "title": "Source B",
            "url": "https://b.example",
            "snippet": "Evidence B",
            "query": "query",
        },
    ]
    state["page_records"] = [
        {
            "requested_url": "https://a.example",
            "final_url": "https://a.example",
            "success": True,
            "content_chars": 100,
            "truncated": False,
            "error": None,
        }
    ]
    state["observations"] = [
        {"content": "A", "source_url": "https://a.example", "query": "query"},
        {"content": "B", "source_url": "https://b.example", "query": "query"},
    ]
    state["final_report"] = "# Final report"
    return state


def test_assess_research_reports_structured_gaps() -> None:
    state = create_initial_state("研究问题")

    gaps = assess_research(state)

    assert "尚未创建研究计划。" in gaps
    assert any("去重来源不足" in gap for gap in gaps)
    assert any("网页正文读取不足" in gap for gap in gaps)
    assert any("证据观察不足" in gap for gap in gaps)
    assert "尚未生成正式研究报告。" in gaps


def test_assess_research_accepts_completed_plan_and_enough_evidence() -> None:
    assert assess_research(make_complete_state()) == []


def test_reflection_ignores_model_output_that_calls_a_tool() -> None:
    state = create_initial_state("研究问题")
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "test"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    )

    update = ReflectionMiddleware().after_model(state, None)  # type: ignore[arg-type]

    assert update is None


def test_agent_reflects_twice_then_stops() -> None:
    CountingToolModel.call_count = 0
    model = CountingToolModel(
        responses=[
            AIMessage(content="第一次过早回答"),
            AIMessage(content="第二次过早回答"),
            AIMessage(content="达到上限后的最终回答"),
        ]
    )

    result = run_with_model(
        "研究问题",
        model,
        tools=[
            write_research_plan_tool,
            web_search_tool,
            read_page_tool,
            write_final_report_tool,
        ],
    )

    assert result.answer == "达到上限后的最终回答"
    assert result.state["reflection_attempts"] == MAX_REFLECTION_ATTEMPTS
    assert "尚未创建研究计划。" in result.state["research_gaps"]
    assert CountingToolModel.call_count == 3
