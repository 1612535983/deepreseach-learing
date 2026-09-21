import importlib
from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage

from deepresearch.agent import run_with_model
from deepresearch.middlewares.plan_context import format_plan_context
from deepresearch.state import create_initial_state
from deepresearch.tools import (
    update_plan_step_tool,
    web_search_tool,
    write_research_plan_tool,
)


search_module = importlib.import_module("deepresearch.tools.web_search")


class CapturingToolModel(FakeMessagesListChatModel):
    captured_calls: ClassVar[list[list]] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.captured_calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FakeDDGS:
    def __init__(self, timeout: int) -> None:
        pass

    def text(self, query: str, **kwargs):  # noqa: ANN003, ANN201
        return [
            {
                "title": "Official docs",
                "href": "https://example.com/docs",
                "body": "Planning evidence",
            }
        ]


def system_text(messages: list) -> str:
    return "\n".join(
        message.content
        for message in messages
        if isinstance(message, SystemMessage) and isinstance(message.content, str)
    )


def test_plan_context_exposes_progress_but_not_raw_evidence() -> None:
    state = create_initial_state("研究问题")
    state["plan"] = {
        "goal": "研究 Agent",
        "steps": [
            {"step_id": "step-1", "title": "查找资料", "status": "in_progress"}
        ],
    }
    state["current_step_id"] = "step-1"
    state["sources"] = [
        {
            "title": "Secret source",
            "url": "https://secret.example/internal",
            "snippet": "RAW SECRET EVIDENCE",
            "query": "secret",
        }
    ]
    state["reflection_attempts"] = 1
    state["research_gaps"] = ["还需要读取一个网页正文。"]

    context = format_plan_context(state)

    assert "研究 Agent" in context
    assert "step-1" in context
    assert "去重来源数：1" in context
    assert "还需要读取一个网页正文。" in context
    assert "当前反思次数：1" in context
    assert "https://secret.example/internal" not in context
    assert "RAW SECRET EVIDENCE" not in context


def test_agent_creates_plan_links_evidence_and_advances_step(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    CapturingToolModel.captured_calls.clear()
    model = CapturingToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_research_plan",
                        "args": {
                            "goal": "研究 LangChain Agent",
                            "steps": ["查找官方资料", "总结执行流程"],
                        },
                        "id": "plan-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "LangChain Agent docs", "max_results": 1},
                        "id": "search-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_plan_step",
                        "args": {"step_id": "step-1", "status": "completed"},
                        "id": "update-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="第一步完成，下一步继续研究。"),
        ]
    )

    result = run_with_model(
        "研究 LangChain Agent",
        model,
        tools=[write_research_plan_tool, update_plan_step_tool, web_search_tool],
    )

    assert result.state["plan"] is not None
    assert result.state["plan"]["steps"] == [
        {"step_id": "step-1", "title": "查找官方资料", "status": "completed"},
        {"step_id": "step-2", "title": "总结执行流程", "status": "in_progress"},
    ]
    assert result.state["current_step_id"] == "step-2"
    assert result.state["search_records"][0]["step_id"] == "step-1"
    assert result.state["observations"][0]["step_id"] == "step-1"

    first_context = system_text(CapturingToolModel.captured_calls[0])
    second_context = system_text(CapturingToolModel.captured_calls[1])
    assert "No research plan exists yet" in first_context
    assert "研究目标：研究 LangChain Agent" in second_context
