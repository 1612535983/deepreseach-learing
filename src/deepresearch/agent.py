"""Compile and run the research agent graph and its offline demo variant."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from deepresearch.checkpointing import (
    ensure_new_thread,
    ensure_new_thread_async,
    get_checkpoint_state,
    normalize_thread_id,
    open_async_sqlite_checkpointer,
    open_sqlite_checkpointer,
    resolve_checkpoint_run,
)
from deepresearch.context.tagged import ContextAssembler
from deepresearch.config import Settings
from deepresearch.events import ResearchEvent, events_from_update
from deepresearch.middlewares import (
    ContextCompactionMiddleware,
    ContextExternalizationMiddleware,
    ContextFinalizationMiddleware,
    ContextGovernanceMiddleware,
    EvidenceMiddleware,
    ReflectionMiddleware,
    SequentialToolCallMiddleware,
    TaggedContextMiddleware,
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
    system_prompt = SEARCH_SYSTEM_PROMPT if resolved_tools else BASE_SYSTEM_PROMPT
    include_research_context = "write_research_plan" in tool_names
    context_assembler = ContextAssembler()

    def project_model_messages(state: ResearchState) -> list[BaseMessage]:
        assembled = context_assembler.assemble(
            state,
            list(state.get("messages", [])),
            SystemMessage(content=system_prompt),
            include_research_context=include_research_context,
        )
        return [assembled.system_message, *assembled.messages]

    research_tool_names = {
        "write_research_plan",
        "web_search",
        "read_page",
        "write_final_report",
    }
    middlewares = [ContextFinalizationMiddleware()]
    if research_tool_names.issubset(tool_names):
        # after_model hooks run in reverse registration order. Keeping Reflection
        # here makes Governance update the latest budget first, then lets P5
        # suppress reflection loops before final Tool routing is decided.
        middlewares.append(ReflectionMiddleware())
    middlewares.extend(
        [
            ContextGovernanceMiddleware(
                model,
                message_projector=project_model_messages,
            ),
            ContextExternalizationMiddleware(),
            ContextCompactionMiddleware(model),
        ]
    )
    state_mutating_tools = {
        "write_research_plan",
        "update_plan_step",
        "write_final_report",
    }
    if state_mutating_tools.intersection(tool_names):
        middlewares.append(SequentialToolCallMiddleware())
    middlewares.append(
        TaggedContextMiddleware(
            system_prompt,
            include_research_context=include_research_context,
            assembler=context_assembler,
        )
    )
    if {"web_search", "read_page"}.intersection(tool_names):
        middlewares.append(EvidenceMiddleware())
    return create_agent(
        model=model,
        tools=resolved_tools,
        middleware=middlewares,
        system_prompt=system_prompt,
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


def _question_from_checkpoint(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> str:
    """Load and validate the original question saved for a thread."""

    state = get_checkpoint_state(checkpointer, thread_id)
    question = state.get("research_question")
    if not isinstance(question, str) or not question.strip():
        raise RuntimeError(f"任务 {thread_id} 没有保存有效的研究问题。")
    return question


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
    if checkpointer is not None and resolved_thread_id is not None:
        ensure_new_thread(checkpointer, resolved_thread_id)
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
    if checkpointer is not None and resolved_thread_id is not None:
        ensure_new_thread(checkpointer, resolved_thread_id)
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
    if checkpointer is not None and resolved_thread_id is not None:
        await ensure_new_thread_async(checkpointer, resolved_thread_id)
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


def resume_with_model(
    thread_id: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
) -> ResearchResult:
    """Continue an existing graph thread from its latest saved checkpoint."""

    normalized_thread_id = normalize_thread_id(thread_id)
    question = _question_from_checkpoint(checkpointer, normalized_thread_id)
    _, config = resolve_checkpoint_run(checkpointer, normalized_thread_id)
    graph = build_agent(model, tools=tools, checkpointer=checkpointer)
    state = cast(ResearchState, graph.invoke(None, config=config))
    return _result_from_state(question, state, normalized_thread_id)


def stream_resume_with_model(
    thread_id: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    *,
    on_event: Callable[[ResearchEvent], None],
    checkpointer: BaseCheckpointSaver,
) -> ResearchResult:
    """Continue a saved graph thread while publishing synchronous events."""

    normalized_thread_id = normalize_thread_id(thread_id)
    question = _question_from_checkpoint(checkpointer, normalized_thread_id)
    _, config = resolve_checkpoint_run(checkpointer, normalized_thread_id)
    graph = build_agent(model, tools=tools, checkpointer=checkpointer)
    final_state: ResearchState | None = None
    on_event(
        ResearchEvent(
            "run_started",
            f"恢复研究：{question}",
            {
                "question": question,
                "thread_id": normalized_thread_id,
                "resumed": True,
            },
        )
    )
    try:
        for mode, data in graph.stream(
            None,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for event in events_from_update(data):
                    on_event(event)
            elif mode == "values" and isinstance(data, dict):
                final_state = cast(ResearchState, data)
    except Exception as exc:
        on_event(
            ResearchEvent(
                "run_failed",
                f"研究恢复失败：{exc}",
                {"error_type": type(exc).__name__},
            )
        )
        raise

    if final_state is None:
        raise RuntimeError("Agent 恢复执行没有返回最终 State。")
    result = _result_from_state(question, final_state, normalized_thread_id)
    on_event(
        ResearchEvent(
            "run_completed",
            "研究任务已完成",
            {
                "source_count": len(final_state.get("sources", [])),
                "has_report": bool(final_state.get("final_report")),
                "thread_id": normalized_thread_id,
                "resumed": True,
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
    model = build_model(resolved_settings)
    if checkpointer is not None:
        return run_with_model(
            question,
            model,
            tools=_research_tools(),
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    with open_sqlite_checkpointer() as sqlite_checkpointer:
        return run_with_model(
            question,
            model,
            tools=_research_tools(),
            checkpointer=sqlite_checkpointer,
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
    model = build_model(resolved_settings)
    if checkpointer is not None:
        return stream_with_model(
            question,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    with open_sqlite_checkpointer() as sqlite_checkpointer:
        return stream_with_model(
            question,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=sqlite_checkpointer,
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
    model = build_model(resolved_settings)
    if checkpointer is not None:
        return await astream_with_model(
            question,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    async with open_async_sqlite_checkpointer() as sqlite_checkpointer:
        return await astream_with_model(
            question,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=sqlite_checkpointer,
            thread_id=thread_id,
        )


def resume_question(
    thread_id: str,
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> ResearchResult:
    """Resume a real-model research thread from SQLite or an injected saver."""

    resolved_settings = settings or Settings.from_env()
    model = build_model(resolved_settings)
    if checkpointer is not None:
        return resume_with_model(
            thread_id,
            model,
            tools=_research_tools(),
            checkpointer=checkpointer,
        )
    with open_sqlite_checkpointer() as sqlite_checkpointer:
        return resume_with_model(
            thread_id,
            model,
            tools=_research_tools(),
            checkpointer=sqlite_checkpointer,
        )


def stream_resume_question(
    thread_id: str,
    on_event: Callable[[ResearchEvent], None],
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> ResearchResult:
    """Resume a real-model research thread and publish progress events."""

    resolved_settings = settings or Settings.from_env()
    model = build_model(resolved_settings)
    if checkpointer is not None:
        return stream_resume_with_model(
            thread_id,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=checkpointer,
        )
    with open_sqlite_checkpointer() as sqlite_checkpointer:
        return stream_resume_with_model(
            thread_id,
            model,
            tools=_research_tools(),
            on_event=on_event,
            checkpointer=sqlite_checkpointer,
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
