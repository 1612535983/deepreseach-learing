from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deepresearch.context.snapshot import ContextSnapshotter
from deepresearch.context.tagged import (
    DEEPRESEARCH_EXTERNALIZED,
    DEEPRESEARCH_EXTERNALIZED_PATH,
)
from deepresearch.state import create_initial_state


FIXED_TIME = datetime(
    2026,
    9,
    22,
    14,
    30,
    tzinfo=timezone(timedelta(hours=8)),
)


def make_snapshot_state():  # noqa: ANN201
    state = create_initial_state("研究 P4 Snapshot")
    state["messages"].extend(
        [
            AIMessage(
                id="ai-1",
                content="",
                tool_calls=[
                    {
                        "name": "read_page",
                        "args": {"url": "https://example.com"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                id="tool-1",
                content="正文预览",
                name="read_page",
                tool_call_id="call-1",
                additional_kwargs={
                    DEEPRESEARCH_EXTERNALIZED: True,
                    DEEPRESEARCH_EXTERNALIZED_PATH: (
                        ".deepresearch/externalized/thread/result.json"
                    ),
                },
            ),
        ]
    )
    state["plan"] = {
        "goal": "验证快照",
        "steps": [
            {"step_id": "step-1", "title": "保存", "status": "in_progress"}
        ],
    }
    return state


def test_snapshot_round_trip_preserves_messages_and_state(tmp_path) -> None:  # noqa: ANN001
    state = make_snapshot_state()
    snapshotter = ContextSnapshotter(
        tmp_path / "snapshots",
        now_provider=lambda: FIXED_TIME,
    )

    result = snapshotter.create(
        state,
        state["messages"],
        namespace="thread/unsafe",
    )
    loaded = snapshotter.load(result.path)

    assert result.path.is_file()
    assert result.path.parent.name == "thread-unsafe"
    assert result.message_count == 3
    assert result.bytes_written == result.path.stat().st_size
    assert loaded.snapshot_id == result.snapshot_id
    assert loaded.created_at == FIXED_TIME.isoformat()
    assert loaded.namespace == "thread/unsafe"
    assert loaded.state["research_question"] == "研究 P4 Snapshot"
    assert loaded.state["plan"]["goal"] == "验证快照"
    assert [message.id for message in loaded.messages] == [
        state["messages"][0].id,
        "ai-1",
        "tool-1",
    ]
    restored_tool = loaded.messages[-1]
    assert isinstance(restored_tool, ToolMessage)
    assert restored_tool.tool_call_id == "call-1"
    assert restored_tool.additional_kwargs[DEEPRESEARCH_EXTERNALIZED] is True


def test_snapshot_creation_is_idempotent_for_the_same_input(tmp_path) -> None:  # noqa: ANN001
    state = make_snapshot_state()
    snapshotter = ContextSnapshotter(tmp_path, now_provider=lambda: FIXED_TIME)

    first = snapshotter.create(state, state["messages"], namespace="thread")
    second = snapshotter.create(state, state["messages"], namespace="thread")

    assert second == first
    assert len(list(tmp_path.rglob("snapshot-*.json"))) == 1


def test_snapshot_load_rejects_tampered_content(tmp_path) -> None:  # noqa: ANN001
    state = make_snapshot_state()
    snapshotter = ContextSnapshotter(tmp_path, now_provider=lambda: FIXED_TIME)
    result = snapshotter.create(state, state["messages"], namespace="thread")
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload["state"]["research_question"] = "被篡改"
    result.path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="校验失败"):
        snapshotter.load(result.path)
