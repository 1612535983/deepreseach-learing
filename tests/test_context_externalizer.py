from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from deepresearch.agent import run_with_model
from deepresearch.checkpointing import create_in_memory_checkpointer, get_checkpoint_state
from deepresearch.context.externalizer import ToolResultExternalizer
from deepresearch.context.tagged import (
    DEEPRESEARCH_EXTERNALIZED,
    DEEPRESEARCH_EXTERNALIZED_META,
    DEEPRESEARCH_EXTERNALIZED_PATH,
)
from deepresearch.middlewares.context_externalization import (
    ContextExternalizationMiddleware,
)
from deepresearch.state import create_initial_state


def tool_message(index: int, content: str) -> ToolMessage:
    return ToolMessage(
        id=f"message-{index}",
        content=content,
        name="read_page",
        tool_call_id=f"call-{index}",
    )


def test_externalizer_persists_raw_content_before_building_replacements(tmp_path) -> None:  # noqa: ANN001
    old_first = tool_message(1, "第一份正文" * 100)
    old_second = tool_message(2, "第二份正文" * 100)
    recent = tool_message(3, "最近正文" * 100)
    externalizer = ToolResultExternalizer(
        tmp_path / "externalized",
        min_chars=100,
        preview_chars=20,
        preserve_recent_results=1,
    )

    batch = externalizer.externalize_history(
        [old_first, old_second, recent],
        namespace="thread/unsafe",
    )

    assert len(batch.results) == 2
    assert batch.errors == ()
    assert {message.id for message in batch.replacements} == {
        "message-1",
        "message-2",
    }
    assert batch.replacements[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED]
    path = Path(
        batch.replacements[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED_PATH]
    )
    assert path.is_file()
    assert path.parent.name == "thread-unsafe"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["content"] == old_first.content
    assert old_first.content not in str(batch.replacements[0].content)
    assert "第一份正文" in str(batch.replacements[0].content)
    assert batch.results[0].estimated_tokens_saved > 0


def test_externalizer_is_idempotent_and_preserves_recent_results(tmp_path) -> None:  # noqa: ANN001
    messages = [
        tool_message(1, "旧正文" * 100),
        tool_message(2, "最近正文" * 100),
    ]
    externalizer = ToolResultExternalizer(
        tmp_path,
        min_chars=100,
        preserve_recent_results=1,
    )
    first = externalizer.externalize_history(messages, namespace="thread")
    merged = add_messages(messages, first.replacements)
    second = externalizer.externalize_history(merged, namespace="thread")

    assert len(first.results) == 1
    assert second.results == ()
    assert merged[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED] is True
    assert DEEPRESEARCH_EXTERNALIZED_META in merged[0].additional_kwargs
    assert DEEPRESEARCH_EXTERNALIZED not in merged[1].additional_kwargs


def test_externalizer_keeps_raw_message_when_persistence_fails(
    monkeypatch,  # noqa: ANN001
    tmp_path,
) -> None:  # noqa: ANN001
    message = tool_message(1, "必须保留的正文" * 100)
    externalizer = ToolResultExternalizer(
        tmp_path,
        min_chars=100,
        preserve_recent_results=0,
    )

    def fail_write(path, payload):  # noqa: ANN001, ANN202
        raise OSError("disk unavailable")

    monkeypatch.setattr(externalizer, "_write_once", fail_write)
    batch = externalizer.externalize_history([message], namespace="thread")

    assert batch.results == ()
    assert "disk unavailable" in batch.errors[0]
    assert message.content == "必须保留的正文" * 100
    assert DEEPRESEARCH_EXTERNALIZED not in message.additional_kwargs


def test_middleware_only_externalizes_when_p1_is_pending(tmp_path) -> None:  # noqa: ANN001
    state = create_initial_state("P1 测试")
    state["messages"].extend(
        [
            tool_message(1, "旧正文" * 100),
            tool_message(2, "最近正文" * 100),
        ]
    )
    middleware = ContextExternalizationMiddleware(
        ToolResultExternalizer(
            tmp_path,
            min_chars=100,
            preview_chars=20,
            preserve_recent_results=1,
        )
    )

    assert middleware.before_model(state, None) is None  # type: ignore[arg-type]
    state["governance"]["context"]["pending_stages"] = ["P1"]
    update = middleware.before_model(state, None)  # type: ignore[arg-type]

    assert update is not None
    assert len(update["messages"]) == 1
    metrics = update["governance"]["context"]["externalization"]
    assert metrics["externalized_tool_results"] == 1
    assert metrics["estimated_tokens_saved"] > 0
    assert len(metrics["last_externalized_paths"]) == 1


@tool
def large_result(label: str) -> str:
    """Return a large deterministic Tool result for P1 integration tests."""

    return f"{label}:" + label * 8_000


class P1ToolModel(FakeMessagesListChatModel):
    # Keep this integration focused on P1; P5 now correctly stops the same
    # expanding Tool once the request reaches 90%.
    max_input_tokens: ClassVar[int] = 40_000

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self

    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        return sum(len(str(getattr(message, "content", ""))) for message in messages)


def test_agent_externalizes_old_tool_result_before_next_model_call(
    monkeypatch,  # noqa: ANN001
    tmp_path,
) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    model = P1ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "large_result",
                        "args": {"label": "A"},
                        "id": "call-a",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "large_result",
                        "args": {"label": "B"},
                        "id": "call-b",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "large_result",
                        "args": {"label": "C"},
                        "id": "call-c",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="P1 完成"),
        ]
    )

    checkpointer = create_in_memory_checkpointer()
    result = run_with_model(
        "执行 P1",
        model,
        tools=[large_result],
        checkpointer=checkpointer,
        thread_id="p1-thread",
    )

    tool_messages = [
        message
        for message in result.state["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 3
    assert tool_messages[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED] is True
    assert DEEPRESEARCH_EXTERNALIZED not in tool_messages[-1].additional_kwargs
    path = Path(
        tool_messages[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED_PATH]
    )
    assert path.is_file()
    assert path.parent.name == "p1-thread"
    metrics = result.state["governance"]["context"]["externalization"]
    assert metrics["externalized_tool_results"] == 1
    assert metrics["estimated_tokens_saved"] > 0
    assert result.state["tagged_context"] is not None
    assert '<toolresult name="large_result" path=' in result.state["tagged_context"][
        "rendered"
    ]
    saved_state = get_checkpoint_state(checkpointer, "p1-thread")
    saved_tools = [
        message
        for message in saved_state["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert saved_tools[0].additional_kwargs[DEEPRESEARCH_EXTERNALIZED] is True
    assert result.answer == "P1 完成"
