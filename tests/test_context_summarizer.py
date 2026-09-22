from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from deepresearch.context.snapshot import SnapshotResult
from deepresearch.context.summarizer import ContextSummarizer
from deepresearch.context.tagged import (
    DEEPRESEARCH_COMPACTION_STAGE,
    DEEPRESEARCH_SNAPSHOT_PATH,
    DEEPRESEARCH_SUMMARY,
)


def tool_pair(index: int) -> list:
    return [
        AIMessage(
            id=f"ai-{index}",
            content="",
            tool_calls=[
                {
                    "name": "read_page",
                    "args": {"url": f"https://example.com/{index}"},
                    "id": f"call-{index}",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            id=f"tool-{index}",
            content=f"正文 {index}",
            name="read_page",
            tool_call_id=f"call-{index}",
        ),
    ]


def test_partition_preserves_recent_tool_pairs() -> None:
    messages = [
        HumanMessage(id="human-1", content="研究问题"),
        *tool_pair(1),
        AIMessage(id="answer-1", content="阶段结论"),
        *tool_pair(2),
        AIMessage(id="answer-2", content="最近结论"),
    ]
    summarizer = ContextSummarizer(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        preserve_recent_messages=3,
    )

    plan = summarizer.plan(messages)

    assert plan is not None
    assert [message.id for message in plan.preserved] == [
        "ai-2",
        "tool-2",
        "answer-2",
    ]
    assert [message.id for message in plan.to_summarize] == [
        "human-1",
        "ai-1",
        "tool-1",
        "answer-1",
    ]


def test_partition_moves_orphan_tool_out_of_preserved_suffix() -> None:
    orphan = ToolMessage(
        id="orphan-tool",
        content="孤立结果",
        tool_call_id="missing-call",
    )
    messages = [
        HumanMessage(id="human-1", content="研究问题"),
        *tool_pair(1),
        orphan,
        AIMessage(id="answer-1", content="最近回答"),
    ]
    summarizer = ContextSummarizer(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        preserve_recent_messages=2,
    )

    plan = summarizer.plan(messages)

    assert plan is not None
    assert [message.id for message in plan.preserved] == ["answer-1"]
    assert "orphan-tool" in {message.id for message in plan.to_summarize}


def test_summarizer_builds_structured_prompt_and_compacted_patch() -> None:
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="# 摘要\n\n- 已完成资料收集")]
    )
    summarizer = ContextSummarizer(model, preserve_recent_messages=2)
    messages = [
        HumanMessage(id="human-1", content="研究问题"),
        *tool_pair(1),
        AIMessage(id="recent-answer", content="最近回答"),
        HumanMessage(id="recent-user", content="继续"),
    ]
    plan = summarizer.plan(messages)
    assert plan is not None

    summary = summarizer.summarize(
        plan,
        previous_summary="上一轮摘要",
        snapshot_path="snapshot.json",
    )
    compacted = summarizer.compacted_messages(
        plan,
        summary,
        SnapshotResult(Path("snapshot.json"), "snapshot-id", len(messages), 100),
    )
    merged = add_messages(messages, compacted.patch)

    assert summary.startswith("# 摘要")
    assert merged[0].id == compacted.summary_id
    assert merged[0].additional_kwargs[DEEPRESEARCH_SUMMARY] is True
    assert merged[0].additional_kwargs[DEEPRESEARCH_COMPACTION_STAGE] == "P4"
    assert merged[0].additional_kwargs[DEEPRESEARCH_SNAPSHOT_PATH] == "snapshot.json"
    assert [message.id for message in merged[1:]] == [
        message.id for message in plan.preserved
    ]


def test_async_summarizer_uses_the_same_output_contract() -> None:
    summarizer = ContextSummarizer(
        FakeMessagesListChatModel(responses=[AIMessage(content="异步摘要")]),
        preserve_recent_messages=2,
    )
    messages = [
        HumanMessage(id="human-1", content="问题"),
        *tool_pair(1),
        HumanMessage(id="recent-1", content="最近一"),
        AIMessage(id="recent-2", content="最近二"),
    ]
    plan = summarizer.plan(messages)
    assert plan is not None

    summary = asyncio.run(
        summarizer.asummarize(
            plan,
            previous_summary=None,
            snapshot_path="snapshot.json",
        )
    )

    assert summary == "异步摘要"
