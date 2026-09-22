from __future__ import annotations

from datetime import date

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from deepresearch.context.tagged import (
    DEEPRESEARCH_EXTERNALIZED,
    DEEPRESEARCH_EXTERNALIZED_META,
    DEEPRESEARCH_EXTERNALIZED_PATH,
    ContextAssembler,
)
from deepresearch.middlewares.tagged_context import TaggedContextMiddleware
from deepresearch.state import create_initial_state


def assembler() -> ContextAssembler:
    return ContextAssembler(today_provider=lambda: date(2026, 9, 22))


def test_assembler_renders_selected_state_without_raw_evidence() -> None:
    state = create_initial_state("研究 <Agent>")
    state["plan"] = {
        "goal": "核验 Agent",
        "steps": [
            {"step_id": "step-1", "title": "读取 <docs>", "status": "in_progress"}
        ],
    }
    state["current_step_id"] = "step-1"
    state["research_gaps"] = ["还缺少正文"]
    state["sources"] = [
        {
            "title": "Secret",
            "url": "https://secret.example",
            "snippet": "RAW SECRET",
            "query": "secret",
        }
    ]

    assembled = assembler().assemble(
        state,
        state["messages"],
        SystemMessage(content="Use tools <carefully>"),
        include_research_context=True,
    )

    system_text = str(assembled.system_message.content)
    assert "<system>Use tools &lt;carefully&gt;</system>" in system_text
    assert "<goal>研究 &lt;Agent&gt;</goal>" in system_text
    assert "研究目标：核验 Agent" in system_text
    assert "读取 &lt;docs&gt;" in system_text
    assert "还缺少正文" in system_text
    assert "去重来源数：1" in system_text
    assert "<date>2026-09-22</date>" in system_text
    assert "https://secret.example" not in system_text
    assert "RAW SECRET" not in system_text


def test_assembler_rewrites_only_request_scoped_ai_content() -> None:
    original_ai = AIMessage(
        id="ai-1",
        content="结论 <待核验>",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "agent"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    tool = ToolMessage(
        id="tool-1",
        content="搜索结果",
        tool_call_id="call-1",
    )

    rewritten = assembler().rewrite_messages_for_model(
        [HumanMessage(content="问题"), original_ai, tool]
    )

    assert rewritten[1].content == "<answer>结论 &lt;待核验&gt;</answer>"
    assert isinstance(rewritten[1], AIMessage)
    assert rewritten[1].tool_calls == original_ai.tool_calls
    assert rewritten[2] is tool
    assert original_ai.content == "结论 <待核验>"


def test_audit_rendering_marks_externalized_tool_results() -> None:
    tool = ToolMessage(
        content="正文预览",
        name="read_page",
        tool_call_id="call-1",
        additional_kwargs={
            DEEPRESEARCH_EXTERNALIZED: True,
            DEEPRESEARCH_EXTERNALIZED_PATH: ".deepresearch/externalized/result.txt",
            DEEPRESEARCH_EXTERNALIZED_META: {"estimated_tokens_saved": 900},
        },
    )

    rendered = assembler().render_messages(
        [HumanMessage(content="问题"), tool]
    )

    assert '<toolresult name="read_page"' in rendered
    assert 'path=".deepresearch/externalized/result.txt"' in rendered
    assert 'tokens_saved="900"' in rendered


def test_middleware_persists_snapshot_and_overrides_only_model_request() -> None:
    state = create_initial_state("研究问题")
    middleware = TaggedContextMiddleware(
        "系统规则",
        include_research_context=False,
        assembler=assembler(),
        created_at_provider=lambda: "2026-09-22T12:00:00+08:00",
    )
    update = middleware.before_model(state, Runtime())

    assert update["tagged_context"]["message_count"] == 2  # type: ignore[index]
    assert "<goal>研究问题</goal>" in update["tagged_context"]["rendered"]  # type: ignore[index]
    assert state["tagged_context"] is None

    model = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
    request = ModelRequest(
        model=model,
        messages=list(state["messages"]),
        system_message=SystemMessage(content="系统规则"),
        state=state,
        runtime=Runtime(),
    )
    captured: list[ModelRequest] = []

    def handler(incoming: ModelRequest) -> ModelResponse:
        captured.append(incoming)
        return ModelResponse(result=[AIMessage(content="完成")])

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].content == "完成"
    assert "<system>系统规则</system>" in str(captured[0].system_message.content)
    assert request.system_message.content == "系统规则"
