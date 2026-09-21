import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import astream_with_model, stream_with_model
from deepresearch.events import ResearchEvent, events_from_update
from deepresearch.tools import update_plan_step_tool, write_research_plan_tool


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self


def test_stream_with_model_returns_result_and_lifecycle_events() -> None:
    events: list[ResearchEvent] = []
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="流式执行完成")]
    )

    result = stream_with_model("测试问题", model, on_event=events.append)

    assert result.answer == "流式执行完成"
    assert [event.event_type for event in events] == [
        "run_started",
        "run_completed",
    ]
    assert events[-1].data == {"source_count": 0, "has_report": False}


def test_stream_with_model_converts_plan_tool_updates() -> None:
    events: list[ResearchEvent] = []
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_research_plan",
                        "args": {
                            "goal": "完成研究",
                            "steps": ["搜索资料", "核验资料"],
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
                        "name": "update_plan_step",
                        "args": {"step_id": "step-1", "status": "completed"},
                        "id": "update-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="阶段完成"),
        ]
    )

    result = stream_with_model(
        "测试问题",
        model,
        tools=[write_research_plan_tool, update_plan_step_tool],
        on_event=events.append,
    )

    event_types = [event.event_type for event in events]
    assert event_types == [
        "run_started",
        "tool_requested",
        "plan_updated",
        "tool_requested",
        "plan_updated",
        "run_completed",
    ]
    assert result.state["current_step_id"] == "step-2"
    assert events[1].data == {
        "tool_name": "write_research_plan",
        "goal": "完成研究",
        "step_count": 2,
    }


def test_stream_events_do_not_expose_report_tool_body() -> None:
    secret_report = "VERY LARGE SECRET REPORT BODY"
    update = {
        "model": {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_final_report",
                            "args": {
                                "title": "研究报告",
                                "report": secret_report,
                                "used_source_urls": ["https://example.com"],
                            },
                            "id": "report-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    }

    events = events_from_update(update)

    assert len(events) == 1
    assert events[0].event_type == "tool_requested"
    assert events[0].data == {
        "tool_name": "write_final_report",
        "title": "研究报告",
        "source_count": 1,
    }
    assert secret_report not in repr(events[0])


def test_astream_with_model_delivers_async_events() -> None:
    events: list[ResearchEvent] = []

    async def collect(event: ResearchEvent) -> None:
        events.append(event)

    async def run():  # noqa: ANN202
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="异步流式执行完成")]
        )
        return await astream_with_model("异步测试", model, on_event=collect)

    result = asyncio.run(run())

    assert result.answer == "异步流式执行完成"
    assert [event.event_type for event in events] == [
        "run_started",
        "run_completed",
    ]
