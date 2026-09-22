from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryType
from deepresearch.memory.strategies.default.provider import build_default_provider
from deepresearch.memory.worker import MemoryTask, MemoryWorker


def _task(task_id: str = "task-1") -> MemoryTask:
    return MemoryTask(
        task_id=task_id,
        namespace="project-a",
        thread_id="thread-1",
        messages=(HumanMessage(content="请记住我的开发偏好"),),
        final_report="已完成",
    )


def test_worker_extracts_and_consolidates_without_putting_llm_in_manager(tmp_path) -> None:
    config = MemoryConfig(
        use="default",
        storage_path=tmp_path,
        consolidate_threshold=2,
    )
    provider = build_default_provider(config, now_provider=lambda: 1_000.0)
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=(
                    '[{"content":"用户喜欢先看设计","type":"episodic","importance":0.8},'
                    '{"content":"用户要求单独提交","type":"episodic","importance":0.7}]'
                )
            ),
            AIMessage(content="用户偏好先理解设计，并要求功能分别提交。"),
        ]
    )
    worker = MemoryWorker(provider.manager(), model, config)

    worker.process_now(_task())

    traces = provider.store().list_all()
    assert len(traces) == 3
    assert sum(trace.type == MemoryType.SEMANTIC for trace in traces) == 1
    assert sum(trace.metadata.get("forgotten") is True for trace in traces) == 2
    assert worker.last_error is None


def test_worker_fails_open_on_invalid_extraction_json(tmp_path) -> None:
    config = MemoryConfig(use="default", storage_path=tmp_path)
    provider = build_default_provider(config, now_provider=lambda: 1_000.0)
    worker = MemoryWorker(
        provider.manager(),
        FakeMessagesListChatModel(responses=[AIMessage(content="not json")]),
        config,
    )

    worker.process_now(_task())

    assert provider.store().list_all() == []
    assert worker.last_error is not None
    assert "JSONDecodeError" in worker.last_error


def test_worker_queue_is_bounded_deduplicated_and_drained(tmp_path) -> None:
    config = MemoryConfig(
        use="default",
        storage_path=tmp_path,
        worker_queue_size=1,
        consolidate_threshold=10,
    )
    provider = build_default_provider(config, now_provider=lambda: 1_000.0)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="[]")])
    worker = MemoryWorker(provider.manager(), model, config)

    assert worker.submit(_task()) is True
    assert worker.submit(_task()) is False
    assert worker.submit(_task("task-2")) is False

    worker.start()
    assert worker.flush(timeout=2.0) is True
    worker.shutdown()
    assert worker.pending_count == 0
