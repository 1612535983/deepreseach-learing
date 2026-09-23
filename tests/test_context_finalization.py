from __future__ import annotations

import asyncio
from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from deepresearch.agent import build_agent, run_with_model
from deepresearch.checkpointing import (
    create_in_memory_checkpointer,
    get_checkpoint_state,
)
from deepresearch.middlewares.context_finalization import (
    CONTEXT_BUDGET_STOP_NAME,
    DEEPRESEARCH_CONTEXT_BUDGET_STOP,
    ContextFinalizationMiddleware,
)
from deepresearch.middlewares.reflection import ReflectionMiddleware
from deepresearch.state import create_initial_state
from deepresearch.tools import update_plan_step_tool, write_final_report_tool


def p5_state(
    tool_name: str = "web_search",
    *,
    utilization_ratio: float = 0.95,
    hard_limit: bool = False,
):  # noqa: ANN201
    state = create_initial_state("研究 P5")
    state["governance"]["context"]["pending_stages"] = [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    ]
    state["governance"]["context"]["budget"] = {
        "current_tokens": int(10_000 * utilization_ratio),
        "window_tokens": 10_000,
        "utilization_ratio": utilization_ratio,
        "token_count_method": "model",
        "window_source": "model_attribute",
    }
    state["governance"]["context"]["hard_limit_reached"] = hard_limit
    state["messages"].append(
        AIMessage(
            id="p5-ai-call",
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"query": "继续搜索"},
                    "id": "p5-tool-call",
                    "type": "tool_call",
                }
            ],
            additional_kwargs={
                "tool_calls": [{"name": tool_name, "id": "p5-tool-call"}]
            },
        )
    )
    return state


def test_p5_strips_expansive_tool_call_and_redirects_to_model() -> None:
    state = p5_state()

    update = ContextFinalizationMiddleware().after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    assert update["jump_to"] == "model"
    merged = add_messages(state["messages"], update["messages"])
    cleared = merged[-2]
    reminder = merged[-1]
    assert isinstance(cleared, AIMessage)
    assert cleared.id == "p5-ai-call"
    assert cleared.tool_calls == []
    assert "tool_calls" not in cleared.additional_kwargs
    assert cleared.additional_kwargs[DEEPRESEARCH_CONTEXT_BUDGET_STOP] == 0.95
    assert isinstance(reminder, HumanMessage)
    assert reminder.name == CONTEXT_BUDGET_STOP_NAME
    assert "不得继续搜索" in str(reminder.content)
    metrics = update["governance"]["context"]["finalization"]
    assert metrics["active"] is True
    assert metrics["redirect_count"] == 1
    assert metrics["blocked_tool_call_count"] == 1
    assert metrics["last_blocked_tool_names"] == ["web_search"]
    assert metrics["last_reason"] == "p5_threshold"
    assert metrics["trigger_reason"] == "p5_threshold"


def test_consecutive_failed_searches_trigger_bounded_finalization() -> None:
    state = create_initial_state("研究循环")
    state["search_records"] = [
        {
            "query": f"failed-{index}",
            "success": False,
            "result_count": 0,
            "error": "No results",
        }
        for index in range(4)
    ]
    state["messages"].append(
        AIMessage(
            id="loop-ai-call",
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "another query"},
                    "id": "loop-tool-call",
                    "type": "tool_call",
                }
            ],
        )
    )

    update = ContextFinalizationMiddleware().after_model(  # type: ignore[arg-type]
        state, None
    )

    assert update is not None
    assert update["jump_to"] == "model"
    merged = add_messages(state["messages"], update["messages"])
    assert isinstance(merged[-2], AIMessage)
    assert merged[-2].tool_calls == []
    assert isinstance(merged[-1], HumanMessage)
    assert "连续多次失败或没有结果" in str(merged[-1].content)
    metrics = update["governance"]["context"]["finalization"]
    assert metrics["trigger_reason"] == "consecutive_unproductive_searches"
    assert metrics["last_reason"] == "consecutive_unproductive_searches"


def test_repeated_query_is_blocked_before_third_execution() -> None:
    state = create_initial_state("研究循环")
    state["search_records"] = [
        {
            "query": "LangGraph agent loop",
            "success": True,
            "result_count": 2,
            "error": None,
        },
        {
            "query": "  langgraph   AGENT loop ",
            "success": True,
            "result_count": 1,
            "error": None,
        },
    ]
    state["messages"].append(
        AIMessage(
            id="repeat-ai-call",
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "langgraph agent loop"},
                    "id": "repeat-tool-call",
                    "type": "tool_call",
                }
            ],
        )
    )

    update = ContextFinalizationMiddleware().after_model(  # type: ignore[arg-type]
        state, None
    )

    assert update is not None
    assert update["governance"]["context"]["finalization"][
        "trigger_reason"
    ] == "repeated_search_query"
    assert "同一搜索词已经重复尝试多次" in str(update["messages"][-1].content)


def test_productive_search_resets_consecutive_failure_budget() -> None:
    state = create_initial_state("正常研究")
    state["search_records"] = [
        {
            "query": "failed",
            "success": False,
            "result_count": 0,
            "error": "No results",
        },
        {
            "query": "productive",
            "success": True,
            "result_count": 2,
            "error": None,
        },
    ]
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "different query"},
                    "id": "normal-call",
                    "type": "tool_call",
                }
            ],
        )
    )

    assert ContextFinalizationMiddleware().after_model(  # type: ignore[arg-type]
        state, None
    ) is None


def test_research_budget_never_blocks_terminal_tools() -> None:
    state = create_initial_state("达到读取上限后写报告")
    state["page_records"] = [
        {
            "requested_url": f"https://example.com/{index}",
            "final_url": f"https://example.com/{index}",
            "success": True,
            "content_chars": 100,
            "truncated": False,
            "error": None,
        }
        for index in range(12)
    ]
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_final_report",
                    "args": {"report": "基于已有证据生成报告"},
                    "id": "valid-terminal-call",
                    "type": "tool_call",
                }
            ],
        )
    )

    update = ContextFinalizationMiddleware().after_model(  # type: ignore[arg-type]
        state, None
    )

    assert update is None
    assert state["governance"]["context"]["finalization"]["active"] is False


def test_search_budget_does_not_block_reading_an_existing_source() -> None:
    state = create_initial_state("搜索达到上限后读取来源")
    state["search_records"] = [
        {
            "query": f"query-{index}",
            "success": True,
            "result_count": 1,
            "error": None,
        }
        for index in range(12)
    ]
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_page",
                    "args": {"url": "https://example.com/source"},
                    "id": "read-existing-source",
                    "type": "tool_call",
                }
            ],
        )
    )

    assert ContextFinalizationMiddleware().after_model(  # type: ignore[arg-type]
        state, None
    ) is None


def test_p5_rebuilds_history_when_model_message_has_no_id() -> None:
    state = p5_state()
    state["messages"][-1].id = None

    update = ContextFinalizationMiddleware().after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    merged = add_messages(state["messages"], update["messages"])
    ai_messages = [
        message for message in merged if isinstance(message, AIMessage)
    ]
    assert len(ai_messages) == 1
    assert ai_messages[0].id is not None
    assert ai_messages[0].tool_calls == []


def test_p5_allows_terminal_report_tool() -> None:
    state = p5_state("write_final_report")

    update = ContextFinalizationMiddleware().after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    assert "messages" not in update
    assert "jump_to" not in update
    metrics = update["governance"]["context"]["finalization"]
    assert metrics["active"] is True
    assert metrics["terminal_tool_call_count"] == 1
    assert metrics["blocked_tool_call_count"] == 0
    assert metrics["last_reason"] == "terminal_tool_allowed"


def test_p5_bounds_terminal_tool_calls() -> None:
    state = p5_state("update_plan_step")
    state["governance"]["context"]["finalization"].update(
        {
            "active": True,
            "terminal_tool_call_count": 3,
        }
    )

    update = ContextFinalizationMiddleware(
        max_terminal_tool_calls=3
    ).after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    assert update["jump_to"] == "end"
    metrics = update["governance"]["context"]["finalization"]
    assert metrics["terminal_tool_call_count"] == 3
    assert metrics["forced_stop_count"] == 1
    assert metrics["last_reason"] == "terminal_tool_limit"


def test_repeated_p5_violation_stops_without_executing_tool() -> None:
    state = p5_state("read_page")
    state["governance"]["context"]["finalization"].update(
        {
            "active": True,
            "redirect_count": 1,
            "blocked_tool_call_count": 1,
            "last_blocked_tool_names": ["web_search"],
        }
    )

    update = ContextFinalizationMiddleware(max_redirects=1).after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    assert update["jump_to"] == "end"
    merged = add_messages(state["messages"], update["messages"])
    stopped = merged[-1]
    assert isinstance(stopped, AIMessage)
    assert stopped.tool_calls == []
    assert "未执行以下 Tool Call：read_page" in str(stopped.content)
    metrics = update["governance"]["context"]["finalization"]
    assert metrics["redirect_count"] == 1
    assert metrics["blocked_tool_call_count"] == 2
    assert metrics["forced_stop_count"] == 1
    assert metrics["last_reason"] == "redirect_limit"


def test_p5_hard_limit_uses_99_percent_reminder() -> None:
    state = p5_state(utilization_ratio=0.995, hard_limit=True)

    update = ContextFinalizationMiddleware().after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is not None
    reminder = update["messages"][-1]
    assert isinstance(reminder, HumanMessage)
    assert "99% 硬限制" in str(reminder.content)
    assert update["governance"]["context"]["finalization"][
        "last_reason"
    ] == "hard_limit"


def test_async_p5_uses_same_policy() -> None:
    state = p5_state()

    update = asyncio.run(
        ContextFinalizationMiddleware().aafter_model(
            state,
            None,  # type: ignore[arg-type]
        )
    )

    assert update is not None
    assert update["jump_to"] == "model"


def test_reflection_does_not_add_loops_after_p5() -> None:
    state = p5_state()
    state["messages"][-1] = AIMessage(content="基于已有信息收尾")

    update = ReflectionMiddleware().after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    assert update is None


P5_TOOL_EXECUTIONS: list[str] = []


@tool("web_search")
def tracked_web_search(query: str) -> str:
    """Record execution of a search Tool that P5 must block."""

    P5_TOOL_EXECUTIONS.append(query)
    return "不应该执行"


@tool("write_research_plan")
def tracked_write_research_plan(goal: str) -> str:
    """Represent the planning Tool in the full research middleware stack."""

    P5_TOOL_EXECUTIONS.append(goal)
    return "不应该执行"


@tool("read_page")
def tracked_read_page(url: str) -> str:
    """Represent the page-reading Tool in the full research middleware stack."""

    P5_TOOL_EXECUTIONS.append(url)
    return "不应该执行"


@tool("write_final_report")
def tracked_write_final_report(report: str) -> str:
    """Represent the terminal report Tool without executing it in this test."""

    P5_TOOL_EXECUTIONS.append(report)
    return "不应该执行"


class P5ToolModel(FakeMessagesListChatModel):
    max_input_tokens: ClassVar[int] = 1_000
    call_count: ClassVar[int] = 0

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self

    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        return 950

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        type(self).call_count += 1
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


class LoopToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self


def test_agent_blocks_p5_tool_execution_and_finishes_on_next_model_call() -> None:
    P5_TOOL_EXECUTIONS.clear()
    P5ToolModel.call_count = 0
    model = P5ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "更多材料"},
                        "id": "blocked-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="基于已有信息给出最终答案，并说明证据仍有缺口。"),
        ]
    )
    checkpointer = create_in_memory_checkpointer()

    result = run_with_model(
        "执行 P5",
        model,
        tools=[
            tracked_write_research_plan,
            tracked_web_search,
            tracked_read_page,
            tracked_write_final_report,
        ],
        checkpointer=checkpointer,
        thread_id="p5-thread",
    )

    assert result.answer == "基于已有信息给出最终答案，并说明证据仍有缺口。"
    assert P5_TOOL_EXECUTIONS == []
    assert P5ToolModel.call_count == 2
    assert result.state["reflection_attempts"] == 0
    assert not any(
        isinstance(message, ToolMessage) for message in result.state["messages"]
    )
    metrics = result.state["governance"]["context"]["finalization"]
    assert metrics["active"] is True
    assert metrics["redirect_count"] == 1
    assert metrics["blocked_tool_call_count"] == 1
    assert metrics["forced_stop_count"] == 0
    assert metrics["last_reason"] == "model_stopped_tools"
    saved_state = get_checkpoint_state(checkpointer, "p5-thread")
    assert saved_state["governance"]["context"]["finalization"] == metrics


def test_agent_allows_terminal_tool_execution_during_p5() -> None:
    P5_TOOL_EXECUTIONS.clear()
    P5ToolModel.call_count = 0
    model = P5ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {"report": "已有证据形成的报告"},
                        "id": "terminal-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="报告收尾完成。"),
        ]
    )

    result = run_with_model(
        "执行 P5 报告收尾",
        model,
        tools=[tracked_write_final_report],
    )

    assert result.answer == "报告收尾完成。"
    assert P5_TOOL_EXECUTIONS == ["已有证据形成的报告"]
    assert P5ToolModel.call_count == 2
    metrics = result.state["governance"]["context"]["finalization"]
    assert metrics["active"] is True
    assert metrics["terminal_tool_call_count"] == 1
    assert metrics["blocked_tool_call_count"] == 0


def test_agent_stops_failed_search_loop_before_tool_executes_again() -> None:
    P5_TOOL_EXECUTIONS.clear()
    model = LoopToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "another failed query"},
                        "id": "blocked-budget-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="基于已收集资料结束，并说明搜索服务不稳定。"),
        ]
    )
    state = create_initial_state("搜索服务持续失败")
    state["search_records"] = [
        {
            "query": f"failed-{index}",
            "success": False,
            "result_count": 0,
            "error": "No results",
        }
        for index in range(4)
    ]
    graph = build_agent(model, tools=[tracked_web_search])

    final_state = graph.invoke(state)

    assert P5_TOOL_EXECUTIONS == []
    assert final_state["messages"][-1].content == (
        "基于已收集资料结束，并说明搜索服务不稳定。"
    )
    metrics = final_state["governance"]["context"]["finalization"]
    assert metrics["active"] is True
    assert metrics["trigger_reason"] == "consecutive_unproductive_searches"
    assert metrics["blocked_tool_call_count"] == 1


def test_page_budget_allows_plan_correction_and_second_report_attempt() -> None:
    source_a = "https://a.example/article"
    source_b = "https://b.example/article"
    report = f"# 报告\n\n结论。\n\n- {source_a}\n- {source_b}"
    model = LoopToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {
                            "title": "研究报告",
                            "report": report,
                            "used_source_urls": [source_a, source_b],
                        },
                        "id": "rejected-report",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_plan_step",
                        "args": {"step_id": "step-2", "status": "completed"},
                        "id": "complete-plan",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {
                            "title": "研究报告",
                            "report": report,
                            "used_source_urls": [source_a, source_b],
                        },
                        "id": "accepted-report",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="报告完成。"),
        ]
    )
    state = create_initial_state("达到读取上限后完成报告")
    state["plan"] = {
        "goal": "完成研究",
        "steps": [
            {"step_id": "step-1", "title": "搜索", "status": "completed"},
            {"step_id": "step-2", "title": "报告", "status": "in_progress"},
        ],
    }
    state["current_step_id"] = "step-2"
    state["sources"] = [
        {"title": "A", "url": source_a, "snippet": "A", "query": "q"},
        {"title": "B", "url": source_b, "snippet": "B", "query": "q"},
    ]
    state["observations"] = [
        {"content": "A", "source_url": source_a, "query": "q"},
        {"content": "B", "source_url": source_b, "query": "q"},
    ]
    state["page_records"] = [
        {
            "requested_url": source_a,
            "final_url": source_a,
            "success": True,
            "content_chars": 100,
            "truncated": False,
            "error": None,
        }
        for _ in range(12)
    ]
    graph = build_agent(
        model,
        tools=[update_plan_step_tool, write_final_report_tool],
    )

    final_state = graph.invoke(state)

    assert final_state["plan"]["steps"][1]["status"] == "completed"
    assert final_state["final_report"] == report
    assert final_state["governance"]["context"]["finalization"][
        "active"
    ] is False
