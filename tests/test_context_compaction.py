from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from deepresearch.agent import run_with_model
from deepresearch.checkpointing import create_in_memory_checkpointer, get_checkpoint_state
from deepresearch.context.snapshot import ContextSnapshotter
from deepresearch.context.summarizer import ContextSummarizer
from deepresearch.context.tagged import DEEPRESEARCH_EXTERNALIZED, DEEPRESEARCH_SUMMARY
from deepresearch.middlewares.context_compaction import ContextCompactionMiddleware
from deepresearch.state import create_initial_state, merge_governance


def message_history() -> list:
    return [
        HumanMessage(id="human-1", content="研究问题"),
        AIMessage(id="old-answer", content="很长的旧结论" * 200),
        HumanMessage(id="follow-up", content="继续核验"),
        AIMessage(id="recent-answer", content="最近回答"),
    ]


def p4_state():  # noqa: ANN201
    state = create_initial_state("研究 P4")
    state["messages"] = message_history()
    state["governance"]["context"]["pending_stages"] = ["P1", "P2", "P3", "P4"]
    return state


def test_compaction_snapshots_then_replaces_old_history(tmp_path) -> None:  # noqa: ANN001
    state = p4_state()
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="# P4 摘要\n\n保留旧结论。")]
    )
    middleware = ContextCompactionMiddleware(
        model,
        snapshotter=ContextSnapshotter(tmp_path / "snapshots"),
        summarizer=ContextSummarizer(model, preserve_recent_messages=2),
    )

    update = middleware.before_model(state, None)  # type: ignore[arg-type]

    assert update is not None
    assert "messages" in update
    context_update = update["governance"]["context"]
    metrics = context_update["compaction"]
    assert metrics["snapshot_count"] == 1
    assert metrics["summarize_count"] == 1
    assert metrics["removed_message_count"] == 2
    assert metrics["estimated_tokens_saved"] > 0
    snapshot_path = Path(metrics["last_snapshot_path"])
    snapshot = ContextSnapshotter.load(snapshot_path)
    assert [message.id for message in snapshot.messages] == [
        "human-1",
        "old-answer",
        "follow-up",
        "recent-answer",
    ]

    compacted_messages = add_messages(state["messages"], update["messages"])
    assert compacted_messages[0].additional_kwargs[DEEPRESEARCH_SUMMARY] is True
    assert [message.id for message in compacted_messages[1:]] == [
        "follow-up",
        "recent-answer",
    ]
    merged_governance = merge_governance(
        state["governance"],
        update["governance"],
    )
    assert merged_governance["context"]["summary"].startswith("# P4 摘要")


def test_compaction_failure_keeps_original_messages_and_records_snapshot(tmp_path) -> None:  # noqa: ANN001
    class FailingSummaryModel(FakeMessagesListChatModel):
        def _generate(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise RuntimeError("summary unavailable")

    state = p4_state()
    model = FailingSummaryModel(responses=[AIMessage(content="unused")])
    middleware = ContextCompactionMiddleware(
        model,
        snapshotter=ContextSnapshotter(tmp_path),
        summarizer=ContextSummarizer(model, preserve_recent_messages=2),
    )

    update = middleware.before_model(state, None)  # type: ignore[arg-type]

    assert update is not None
    assert "messages" not in update
    metrics = update["governance"]["context"]["compaction"]
    assert metrics["snapshot_count"] == 1
    assert metrics["summarize_count"] == 0
    assert "summary unavailable" in metrics["last_error"]
    assert Path(metrics["last_snapshot_path"]).is_file()
    assert [message.id for message in state["messages"]] == [
        "human-1",
        "old-answer",
        "follow-up",
        "recent-answer",
    ]


def test_async_compaction_uses_async_summary_model(tmp_path) -> None:  # noqa: ANN001
    state = p4_state()
    model = FakeMessagesListChatModel(responses=[AIMessage(content="异步 P4 摘要")])
    middleware = ContextCompactionMiddleware(
        model,
        snapshotter=ContextSnapshotter(tmp_path),
        summarizer=ContextSummarizer(model, preserve_recent_messages=2),
    )

    update = asyncio.run(middleware.abefore_model(state, None))  # type: ignore[arg-type]

    assert update is not None
    assert update["governance"]["context"]["summary"] == "异步 P4 摘要"
    assert "messages" in update


@tool
def large_p4_result(label: str) -> str:
    """Return large deterministic evidence for a P4 integration test."""

    return f"{label}:" + label * 8_000


class P4ToolModel(FakeMessagesListChatModel):
    # The fourth-result request crosses P4 but remains below P5, so this
    # integration continues to isolate summary compaction.
    max_input_tokens: ClassVar[int] = 30_000

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self

    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        return sum(len(str(getattr(message, "content", ""))) for message in messages)


def test_agent_runs_p1_then_p4_and_checkpoints_compacted_state(
    monkeypatch,  # noqa: ANN001
    tmp_path,
) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    model = P4ToolModel(
        responses=[
            *[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "large_p4_result",
                            "args": {"label": label},
                            "id": f"call-{label.lower()}",
                            "type": "tool_call",
                        }
                    ],
                )
                for label in ("A", "B", "C", "D")
            ],
            AIMessage(
                content=(
                    "# 压缩摘要\n\n"
                    "- 研究目标：执行 P4\n"
                    "- 已保留外化结果路径和最近证据。"
                )
            ),
            AIMessage(content="P4 完成"),
        ]
    )
    checkpointer = create_in_memory_checkpointer()

    result = run_with_model(
        "执行 P4",
        model,
        tools=[large_p4_result],
        checkpointer=checkpointer,
        thread_id="p4-thread",
    )

    context = result.state["governance"]["context"]
    metrics = context["compaction"]
    assert metrics["snapshot_count"] == 1
    assert metrics["summarize_count"] == 1
    assert metrics["estimated_tokens_saved"] > 0
    assert Path(metrics["last_snapshot_path"]).is_file()
    assert context["summary"].startswith("# 压缩摘要")
    assert context["model_call_count"] == 5
    assert result.state["messages"][0].additional_kwargs[DEEPRESEARCH_SUMMARY] is True
    assert result.state["tagged_context"] is not None
    assert "<summary>" in result.state["tagged_context"]["rendered"]
    tool_messages = [
        message
        for message in result.state["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert tool_messages[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED] is True
    ai_tool_ids = {
        str(tool_call["id"])
        for message in result.state["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    }
    assert {message.tool_call_id for message in tool_messages} == ai_tool_ids
    saved_state = get_checkpoint_state(checkpointer, "p4-thread")
    assert saved_state["governance"]["context"]["summary"] == context["summary"]
    assert result.answer == "P4 完成"
