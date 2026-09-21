import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import (
    astream_with_model,
    resume_with_model,
    run_with_model,
    stream_with_model,
)
from deepresearch.checkpointing import (
    build_thread_config,
    create_in_memory_checkpointer,
    open_async_sqlite_checkpointer,
    open_sqlite_checkpointer,
    resolve_checkpoint_run,
)
from deepresearch.events import ResearchEvent


def checkpoint_values(checkpointer, thread_id: str) -> dict:  # noqa: ANN001
    checkpoint = checkpointer.get_tuple(build_thread_config(thread_id))
    assert checkpoint is not None
    return checkpoint.checkpoint["channel_values"]


def test_invoke_automatically_saves_state_by_thread_id() -> None:
    checkpointer = create_in_memory_checkpointer()

    result = run_with_model(
        "第一个问题",
        FakeMessagesListChatModel(responses=[AIMessage(content="第一个答案")]),
        checkpointer=checkpointer,
        thread_id="research-001",
    )

    saved_state = checkpoint_values(checkpointer, "research-001")
    assert result.thread_id == "research-001"
    assert saved_state["research_question"] == "第一个问题"
    assert saved_state["messages"][-1].content == "第一个答案"


def test_in_memory_checkpointer_isolates_threads() -> None:
    checkpointer = create_in_memory_checkpointer()
    run_with_model(
        "问题 A",
        FakeMessagesListChatModel(responses=[AIMessage(content="答案 A")]),
        checkpointer=checkpointer,
        thread_id="thread-a",
    )
    run_with_model(
        "问题 B",
        FakeMessagesListChatModel(responses=[AIMessage(content="答案 B")]),
        checkpointer=checkpointer,
        thread_id="thread-b",
    )

    assert checkpoint_values(checkpointer, "thread-a")["research_question"] == "问题 A"
    assert checkpoint_values(checkpointer, "thread-b")["research_question"] == "问题 B"


def test_stream_uses_thread_config_and_keeps_final_result() -> None:
    checkpointer = create_in_memory_checkpointer()
    events: list[ResearchEvent] = []

    result = stream_with_model(
        "流式问题",
        FakeMessagesListChatModel(responses=[AIMessage(content="流式答案")]),
        on_event=events.append,
        checkpointer=checkpointer,
        thread_id="stream-thread",
    )

    assert result.answer == "流式答案"
    assert events[0].data["thread_id"] == "stream-thread"
    assert checkpoint_values(checkpointer, "stream-thread")["messages"][-1].content == "流式答案"


def test_astream_uses_thread_config() -> None:
    checkpointer = create_in_memory_checkpointer()
    events: list[ResearchEvent] = []

    async def collect(event: ResearchEvent) -> None:
        events.append(event)

    async def run():  # noqa: ANN202
        return await astream_with_model(
            "异步问题",
            FakeMessagesListChatModel(responses=[AIMessage(content="异步答案")]),
            on_event=collect,
            checkpointer=checkpointer,
            thread_id="async-thread",
        )

    result = asyncio.run(run())

    assert result.thread_id == "async-thread"
    assert events[-1].data["thread_id"] == "async-thread"
    assert checkpoint_values(checkpointer, "async-thread")["messages"][-1].content == "异步答案"


def test_checkpointer_generates_thread_id_when_omitted() -> None:
    checkpointer = create_in_memory_checkpointer()

    result = run_with_model(
        "自动 ID",
        FakeMessagesListChatModel(responses=[AIMessage(content="完成")]),
        checkpointer=checkpointer,
    )

    assert result.thread_id is not None
    assert result.thread_id.startswith("research-")
    assert checkpoint_values(checkpointer, result.thread_id)["research_question"] == "自动 ID"


def test_thread_id_without_checkpointer_is_rejected() -> None:
    with pytest.raises(ValueError, match="必须同时提供 checkpointer"):
        resolve_checkpoint_run(None, "orphan-thread")


def test_sqlite_checkpoint_survives_saver_reopen(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"

    with open_sqlite_checkpointer(database_path) as checkpointer:
        run_with_model(
            "持久化问题",
            FakeMessagesListChatModel(responses=[AIMessage(content="持久化答案")]),
            checkpointer=checkpointer,
            thread_id="persistent-thread",
        )

    with open_sqlite_checkpointer(database_path) as reopened_checkpointer:
        saved_state = checkpoint_values(reopened_checkpointer, "persistent-thread")

    assert saved_state["research_question"] == "持久化问题"
    assert saved_state["messages"][-1].content == "持久化答案"


def test_resume_reads_completed_thread_after_saver_reopen(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"
    with open_sqlite_checkpointer(database_path) as checkpointer:
        run_with_model(
            "恢复问题",
            FakeMessagesListChatModel(responses=[AIMessage(content="已保存答案")]),
            checkpointer=checkpointer,
            thread_id="resume-thread",
        )

    with open_sqlite_checkpointer(database_path) as reopened_checkpointer:
        result = resume_with_model(
            "resume-thread",
            FakeMessagesListChatModel(responses=[AIMessage(content="不应被调用")]),
            checkpointer=reopened_checkpointer,
        )

    assert result.question == "恢复问题"
    assert result.answer == "已保存答案"
    assert result.thread_id == "resume-thread"


def test_resume_continues_interrupted_thread_after_saver_reopen(tmp_path) -> None:  # noqa: ANN001
    class FailingFakeModel(FakeMessagesListChatModel):
        def _generate(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise RuntimeError("模拟模型服务中断")

    database_path = tmp_path / "interrupted.sqlite"
    with pytest.raises(RuntimeError, match="模拟模型服务中断"):
        with open_sqlite_checkpointer(database_path) as checkpointer:
            run_with_model(
                "中断问题",
                FailingFakeModel(responses=[AIMessage(content="不会返回")]),
                checkpointer=checkpointer,
                thread_id="interrupted-thread",
            )

    with open_sqlite_checkpointer(database_path) as reopened_checkpointer:
        result = resume_with_model(
            "interrupted-thread",
            FakeMessagesListChatModel(responses=[AIMessage(content="恢复成功")]),
            checkpointer=reopened_checkpointer,
        )

    assert result.question == "中断问题"
    assert result.answer == "恢复成功"


def test_new_run_rejects_existing_thread() -> None:
    checkpointer = create_in_memory_checkpointer()
    run_with_model(
        "原问题",
        FakeMessagesListChatModel(responses=[AIMessage(content="原答案")]),
        checkpointer=checkpointer,
        thread_id="existing-thread",
    )

    with pytest.raises(ValueError, match="已存在.*resume"):
        run_with_model(
            "新问题",
            FakeMessagesListChatModel(responses=[AIMessage(content="新答案")]),
            checkpointer=checkpointer,
            thread_id="existing-thread",
        )


def test_resume_rejects_missing_thread() -> None:
    with pytest.raises(ValueError, match="找不到任务"):
        resume_with_model(
            "missing-thread",
            FakeMessagesListChatModel(responses=[AIMessage(content="不会执行")]),
            checkpointer=create_in_memory_checkpointer(),
        )


def test_async_sqlite_checkpoint_survives_saver_reopen(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "async-checkpoints.sqlite"

    async def run() -> dict:  # noqa: ANN202
        async with open_async_sqlite_checkpointer(database_path) as checkpointer:
            await astream_with_model(
                "异步持久化问题",
                FakeMessagesListChatModel(responses=[AIMessage(content="异步答案")]),
                on_event=lambda event: asyncio.sleep(0),
                checkpointer=checkpointer,
                thread_id="async-persistent-thread",
            )
        async with open_async_sqlite_checkpointer(database_path) as checkpointer:
            checkpoint = await checkpointer.aget_tuple(
                build_thread_config("async-persistent-thread")
            )
            assert checkpoint is not None
            return checkpoint.checkpoint["channel_values"]

    saved_state = asyncio.run(run())

    assert saved_state["research_question"] == "异步持久化问题"
    assert saved_state["messages"][-1].content == "异步答案"
