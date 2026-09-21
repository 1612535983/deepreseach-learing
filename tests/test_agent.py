from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_demo, run_with_model


def test_offline_demo_runs_complete_agent_graph() -> None:
    result = run_demo("测试问题")

    assert result.question == "测试问题"
    assert "最小链路已跑通" in result.answer


def test_agent_returns_model_answer() -> None:
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="这是模型答案")]
    )

    result = run_with_model("什么是 ReAct？", model)

    assert result.question == "什么是 ReAct？"
    assert result.answer == "这是模型答案"

