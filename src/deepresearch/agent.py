"""Compile and run the research agent graph and its offline demo variant."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from deepresearch.checkpointing import (
    get_default_in_memory_checkpointer,
    resolve_checkpoint_run,
)
from deepresearch.config import Settings
from deepresearch.events import ResearchEvent, events_from_update
from deepresearch.middlewares import (
    EvidenceMiddleware,
    PlanContextMiddleware,
    ReflectionMiddleware,
    SequentialToolCallMiddleware,
)
from deepresearch.state import ResearchState, create_initial_state
from deepresearch.tools import (
    read_page_tool,
    update_plan_step_tool,
    web_search_tool,
    write_final_report_tool,
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
每轮只调用一个 Tool，等待它更新 State 后再决定下一步，不要批量调用多个 Tool。
涉及实时信息、具体事实或用户要求来源时，应先搜索再回答。使用简洁、具体的搜索关键词；
对关键结论应优先读取 2 到 3 个最相关、尽量权威的来源，而不是只依赖搜索摘要。
不得编造搜索结果或网页内容。完成计划和证据收集后，必须调用 write_final_report，
提交 Markdown 报告以及报告中实际引用的来源 URL；如果工具失败，请明确说明。
"""


@dataclass(frozen=True)
class ResearchResult:
    """Stable output boundary for CLI, API, and future UI callers."""

    question: str
    answer: str
    state: ResearchState
    thread_id: str | None = None


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
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the model and prompt into LangChain's ReAct agent graph."""

    resolved_tools = list(tools or [])
    tool_names = {tool.name for tool in resolved_tools}
    middlewares = []
    state_mutating_tools = {
        "write_research_plan",
        "update_plan_step",
        "write_final_report",
    }
    if state_mutating_tools.intersection(tool_names):
        middlewares.append(SequentialToolCallMiddleware())
    if "write_research_plan" in tool_names:
        middlewares.append(PlanContextMiddleware())
    if {"web_search", "read_page"}.intersection(tool_names):
        middlewares.append(EvidenceMiddleware())
    research_tool_names = {
        "write_research_plan",
        "web_search",
        "read_page",
        "write_final_report",
    }
    if research_tool_names.issubset(tool_names):
        middlewares.append(ReflectionMiddleware())
    return create_agent(
        model=model,
        tools=resolved_tools,
        middleware=middlewares,
        system_prompt=SEARCH_SYSTEM_PROMPT if resolved_tools else BASE_SYSTEM_PROMPT,
        state_schema=ResearchState,
        checkpointer=checkpointer,
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


def _result_from_state(
    question: str,
    state: ResearchState,
    thread_id: str | None = None,
) -> ResearchResult:
    """Build the public result shared by invoke and stream execution paths."""

    final_message = state["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent 没有返回 AIMessage。")
    return ResearchResult(
        question=question,
        answer=state.get("final_report") or _message_text(final_message),
        state=state,
        thread_id=thread_id,
    )


def run_with_model(
    question: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run one question through a supplied model and return the final answer."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("研究问题不能为空。")

    resolved_thread_id, config = resolve_checkpoint_run(checkpointer, thread_id)
    graph = build_agent(model, tools=tools, checkpointer=checkpointer)
    state = graph.invoke(create_initial_state(normalized_question), config=config)
    return _result_from_state(normalized_question, state, resolved_thread_id)


def stream_with_model(
    question: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    *,
    on_event: Callable[[ResearchEvent], None],
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run the graph once while synchronously delivering progress events."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("研究问题不能为空。")

    resolved_thread_id, config = resolve_checkpoint_run(checkpointer, thread_id)
    graph = build_agent(model, tools=tools, checkpointer=checkpointer)
    final_state: ResearchState | None = None
    on_event(
        ResearchEvent(
            "run_started",
            f"研究开始：{normalized_question}",
            {
                "question": normalized_question,
                **(
                    {"thread_id": resolved_thread_id}
                    if resolved_thread_id is not None
                    else {}
                ),
            },
        )
    )
    try:
        for mode, data in graph.stream(
            create_initial_state(normalized_question),
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for event in events_from_update(data):
                    on_event(event)
            elif mode == "values" and isinstance(data, dict):
                final_state = data
    except Exception as exc:
        on_event(
            ResearchEvent(
                "run_failed",
                f"研究运行失败：{exc}",
                {"error_type": type(exc).__name__},
            )
        )
        raise

    if final_state is None:
        raise RuntimeError("Agent 流式执行没有返回最终 State。")
    result = _result_from_state(
        normalized_question,
        final_state,
        resolved_thread_id,
    )
    on_event(
        ResearchEvent(
            "run_completed",
            "研究任务已完成",
            {
                "source_count": len(final_state.get("sources", [])),
                "has_report": bool(final_state.get("final_report")),
                **(
                    {"thread_id": resolved_thread_id}
                    if resolved_thread_id is not None
                    else {}
                ),
            },
        )
    )
    return result


async def astream_with_model(
    question: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    *,
    on_event: Callable[[ResearchEvent], Awaitable[None]],
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run the graph once while asynchronously delivering progress events."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("研究问题不能为空。")

    resolved_thread_id, config = resolve_checkpoint_run(checkpointer, thread_id)
    graph = build_agent(model, tools=tools, checkpointer=checkpointer)
    final_state: ResearchState | None = None
    await on_event(
        ResearchEvent(
            "run_started",
            f"研究开始：{normalized_question}",
            {
                "question": normalized_question,
                **(
                    {"thread_id": resolved_thread_id}
                    if resolved_thread_id is not None
                    else {}
                ),
            },
        )
    )
    try:
        async for mode, data in graph.astream(
            create_initial_state(normalized_question),
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for event in events_from_update(data):
                    await on_event(event)
            elif mode == "values" and isinstance(data, dict):
                final_state = data
    except Exception as exc:
        await on_event(
            ResearchEvent(
                "run_failed",
                f"研究运行失败：{exc}",
                {"error_type": type(exc).__name__},
            )
        )
        raise

    if final_state is None:
        raise RuntimeError("Agent 流式执行没有返回最终 State。")
    result = _result_from_state(
        normalized_question,
        final_state,
        resolved_thread_id,
    )
    await on_event(
        ResearchEvent(
            "run_completed",
            "研究任务已完成",
            {
                "source_count": len(final_state.get("sources", [])),
                "has_report": bool(final_state.get("final_report")),
                **(
                    {"thread_id": resolved_thread_id}
                    if resolved_thread_id is not None
                    else {}
                ),
            },
        )
    )
    return result


def _research_tools() -> list[BaseTool]:
    """Return the complete tool set used by the real research agent."""

    return [
        write_research_plan_tool,
        update_plan_step_tool,
        web_search_tool,
        read_page_tool,
        write_final_report_tool,
    ]


def run_question(
    question: str,
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run a real model request using explicit settings or the local .env file."""

    resolved_settings = settings or Settings.from_env()
    resolved_checkpointer = (
        checkpointer
        if checkpointer is not None
        else get_default_in_memory_checkpointer()
    )
    return run_with_model(
        question,
        build_model(resolved_settings),
        tools=_research_tools(),
        checkpointer=resolved_checkpointer,
        thread_id=thread_id,
    )


def stream_question(
    question: str,
    on_event: Callable[[ResearchEvent], None],
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run a real model request and synchronously publish progress events."""

    resolved_settings = settings or Settings.from_env()
    resolved_checkpointer = (
        checkpointer
        if checkpointer is not None
        else get_default_in_memory_checkpointer()
    )
    return stream_with_model(
        question,
        build_model(resolved_settings),
        tools=_research_tools(),
        on_event=on_event,
        checkpointer=resolved_checkpointer,
        thread_id=thread_id,
    )


async def astream_question(
    question: str,
    on_event: Callable[[ResearchEvent], Awaitable[None]],
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> ResearchResult:
    """Run a real model request and asynchronously publish progress events."""

    resolved_settings = settings or Settings.from_env()
    resolved_checkpointer = (
        checkpointer
        if checkpointer is not None
        else get_default_in_memory_checkpointer()
    )
    return await astream_with_model(
        question,
        build_model(resolved_settings),
        tools=_research_tools(),
        on_event=on_event,
        checkpointer=resolved_checkpointer,
        thread_id=thread_id,
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
