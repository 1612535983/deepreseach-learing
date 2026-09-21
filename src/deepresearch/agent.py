"""Compile and run the smallest useful agent graph.

At this stage there are intentionally no tools, skills, custom state fields, or
middleware. Keeping the first graph small makes the core request path easy to
understand and gives later features a stable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from deepresearch.config import Settings


SYSTEM_PROMPT = """你是一个严谨的研究助手。
当前是最小可运行版本，还没有搜索工具，所以必须明确区分已知事实和不确定信息。
请直接回答问题；如果问题需要实时资料，请说明当前版本需要接入搜索工具后才能可靠回答。
"""


@dataclass(frozen=True)
class ResearchResult:
    """Stable output boundary for CLI, API, and future UI callers."""

    question: str
    answer: str


def build_model(settings: Settings) -> BaseChatModel:
    """Convert application settings into a LangChain chat model."""

    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
        temperature=0,
    )


def build_agent(model: BaseChatModel) -> Any:
    """Compile the model and prompt into LangChain's ReAct agent graph."""

    return create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def run_with_model(question: str, model: BaseChatModel) -> ResearchResult:
    """Run one question through a supplied model and return the final answer."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("研究问题不能为空。")

    graph = build_agent(model)
    state = graph.invoke(
        {"messages": [{"role": "user", "content": normalized_question}]}
    )
    final_message = state["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent 没有返回 AIMessage。")

    return ResearchResult(
        question=normalized_question,
        answer=_message_text(final_message),
    )


def run_question(question: str, settings: Settings | None = None) -> ResearchResult:
    """Run a real model request using explicit settings or the local .env file."""

    resolved_settings = settings or Settings.from_env()
    return run_with_model(question, build_model(resolved_settings))


def run_demo(question: str = "这个最小 Agent 的执行链是否已经跑通？") -> ResearchResult:
    """Exercise the complete local graph without network access or an API key."""

    fake_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=(
                    "最小链路已跑通：命令行输入已经进入 LangChain Agent Graph，"
                    "模型响应也已被封装为 ResearchResult。"
                )
            )
        ]
    )
    return run_with_model(question, fake_model)

