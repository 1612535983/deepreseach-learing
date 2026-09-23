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


def test_skill_updates_emit_safe_progress_events() -> None:
    update = {
        "SkillSelectionMiddleware.before_model": {
            "skills": {
                "selected": [
                    {
                        "name": "source-verification",
                        "skill_id": "source-verification__abc",
                    }
                ],
                "selection_count": 1,
            }
        },
        "SkillInjectionMiddleware.before_model": {
            "skills": {
                "injection_count": 1,
                "injected_tokens": 120,
                "dropped": [],
                "render_signature": "render-1",
            }
        },
        "SkillMetricsMiddleware.after_agent": {
            "skills": {
                "aligned_tool_calls": 2,
                "completed_recorded": True,
            }
        },
    }

    events = events_from_update(update)

    assert [event.event_type for event in events] == [
        "skill_selected",
        "skill_injected",
        "skill_metrics",
    ]
    assert events[0].data["names"] == ["source-verification"]
    assert events[1].data["token_count"] == 120
    assert events[2].data["aligned_tool_calls"] == 2


def test_skill_selection_reset_does_not_emit_injection_event() -> None:
    events = events_from_update(
        {
            "selection": {
                "skills": {
                    "selected": [],
                    "selection_count": 1,
                    "injection_count": 0,
                    "render_signature": None,
                }
            }
        }
    )

    assert [event.event_type for event in events] == ["skill_selected"]


def test_skill_evaluation_emits_safe_shadow_event() -> None:
    events = events_from_update(
        {
            "SkillEvaluationMiddleware.after_agent": {
                "skills": {
                    "evaluation_recorded": True,
                    "evaluation_count": 2,
                    "evaluation_status": "completed",
                    "evaluation_error": None,
                }
            }
        }
    )

    assert len(events) == 1
    assert events[0].event_type == "skill_evaluated"
    assert events[0].data == {
        "status": "completed",
        "count": 2,
        "error": None,
    }


def test_report_evaluation_emits_safe_cost_and_quality_event() -> None:
    secret_report = "do not expose report body"
    update = {
        "ReportEvaluationMiddleware.after_model": {
            "evaluation": {
                "report": {
                    "status": "completed",
                    "mode": "shadow",
                    "composite_score": 0.8123,
                    "recommended_action": "pass",
                    "runtime_action": "observed",
                    "latency_ms": 245,
                    "input_tokens": 432,
                    "output_tokens": 12,
                    "cost_usd": 0.0042,
                }
            },
            "final_report": secret_report,
        }
    }

    events = events_from_update(update)

    assert [event.event_type for event in events] == [
        "report_evaluated",
        "report_created",
    ]
    evaluation = events[0]
    assert evaluation.data["composite_score"] == 0.8123
    assert evaluation.data["latency_ms"] == 245
    assert evaluation.data["cost_usd"] == 0.0042
    assert secret_report not in repr(evaluation)


def test_report_evaluation_error_event_does_not_fail_stream() -> None:
    events = events_from_update(
        {
            "evaluation": {
                "evaluation": {
                    "report": {
                        "status": "error",
                        "mode": "shadow",
                        "runtime_action": "fail_open",
                        "last_error": "TimeoutError: evaluation failed",
                    }
                }
            }
        }
    )

    assert len(events) == 1
    assert events[0].event_type == "report_evaluated"
    assert events[0].data["error"] == "TimeoutError: evaluation failed"


def test_research_budget_finalization_emits_safe_event() -> None:
    events = events_from_update(
        {
            "ContextFinalizationMiddleware.after_model": {
                "governance": {
                    "context": {
                        "finalization": {
                            "active": True,
                            "last_reason": "consecutive_unproductive_searches",
                            "trigger_reason": "consecutive_unproductive_searches",
                            "last_blocked_tool_names": ["web_search"],
                            "redirect_count": 1,
                        }
                    }
                }
            }
        }
    )

    assert len(events) == 1
    assert events[0].event_type == "finalization"
    assert "搜索连续失败或无结果" in events[0].message
    assert events[0].data["blocked_tool_names"] == ["web_search"]
