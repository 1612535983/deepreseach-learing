"""Compile and run the smallest useful agent graph.

At this stage there are intentionally no tools, skills, custom state fields, or
middleware. Keeping the first graph small makes the core request path easy to
understand and gives later features a stable baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from deepresearch.config import Settings
from deepresearch.middlewares import EvidenceMiddleware, PlanContextMiddleware
from deepresearch.state import ResearchState, create_initial_state
from deepresearch.tools import (
    read_page_tool,
    update_plan_step_tool,
    web_search_tool,
    write_research_plan_tool,
)


BASE_SYSTEM_PROMPT = """你是一个严谨的研究助手。
当前运行模式没有搜索工具。请明确区分已知事实和不确定信息；如果问题需要实时资料，
请说明当前模式无法可靠回答。
"""

SEARCH_SYSTEM_PROMPT = """你是一个严谨的研究助手，可以使用 web_search 搜索公开网页，
并使用 read_page 读取重要来源的正文。
开始研究前必须先调用 write_research_plan 创建 2 到 5 个有顺序的步骤；完成一个步骤后，
调用 update_plan_step 更新状态，再继续下一步。
涉及实时信息、具体事实或用户要求来源时，应先搜索再回答。使用简洁、具体的搜索关键词；
对关键结论应优先读取 2 到 3 个最相关、尽量权威的来源，而不是只依赖搜索摘要。
不得编造搜索结果或网页内容。最终回答要列出实际使用过的来源标题和 URL；如果工具失败，请明确说明。
"""


@dataclass(frozen=True)
class ResearchResult:
    """Stable output boundary for CLI, API, and future UI callers."""

    question: str
    answer: str
    state: ResearchState


def build_model(settings: Settings) -> BaseChatModel:
    """Convert application settings into a LangChain chat model."""

    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
        temperature=0,
    )


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
) -> Any:
    """Compile the model and prompt into LangChain's ReAct agent graph."""

    resolved_tools = list(tools or [])
    tool_names = {tool.name for tool in resolved_tools}
    middlewares = []
    if "write_research_plan" in tool_names:
        middlewares.append(PlanContextMiddleware())
    if {"web_search", "read_page"}.intersection(tool_names):
        middlewares.append(EvidenceMiddleware())
    return create_agent(
        model=model,
        tools=resolved_tools,
        middleware=middlewares,
        system_prompt=SEARCH_SYSTEM_PROMPT if resolved_tools else BASE_SYSTEM_PROMPT,
        state_schema=ResearchState,
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


def run_with_model(
    question: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
) -> ResearchResult:
    """Run one question through a supplied model and return the final answer."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("研究问题不能为空。")

    graph = build_agent(model, tools=tools)
    state = graph.invoke(create_initial_state(normalized_question))
    final_message = state["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent 没有返回 AIMessage。")

    return ResearchResult(
        question=normalized_question,
        answer=_message_text(final_message),
        state=state,
    )


def run_question(question: str, settings: Settings | None = None) -> ResearchResult:
    """Run a real model request using explicit settings or the local .env file."""

    resolved_settings = settings or Settings.from_env()
    return run_with_model(
        question,
        build_model(resolved_settings),
        tools=[
            write_research_plan_tool,
            update_plan_step_tool,
            web_search_tool,
            read_page_tool,
        ],
    )


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
