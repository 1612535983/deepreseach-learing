from typing import ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepresearch.agent import run_with_model
from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryType
from deepresearch.memory.strategies.default.provider import build_default_provider


class CapturingMemoryModel(FakeMessagesListChatModel):
    captured_calls: ClassVar[list[list]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.captured_calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_agent_recalls_into_tagged_request_without_polluting_messages(tmp_path) -> None:
    config = MemoryConfig(
        use="default",
        storage_path=tmp_path,
        namespace="project-a",
    )
    provider = build_default_provider(config, now_provider=lambda: 1_000.0)
    trace = provider.manager().encode(
        "用户偏好先查看设计方案",
        MemoryType.SEMANTIC,
        namespace="project-a",
    )
    CapturingMemoryModel.captured_calls.clear()
    model = CapturingMemoryModel(responses=[AIMessage(content="好的")])

    result = run_with_model(
        "请先给我设计方案",
        model,
        memory_provider=provider,
        memory_config=config,
    )

    system_text = str(CapturingMemoryModel.captured_calls[0][0].content)
    assert "<memory_context>" in system_text
    assert "用户偏好先查看设计方案" in system_text
    assert result.state["memory"]["recalled"][0]["id"] == trace.id
    assert [
        message
        for message in result.state["messages"]
        if isinstance(message, HumanMessage) and message.name == "memory_recall"
    ] == []
