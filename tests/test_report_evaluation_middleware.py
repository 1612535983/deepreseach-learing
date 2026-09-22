from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from deepresearch.agent import build_agent
from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.report import ReportEvaluator
from deepresearch.evaluation.types import (
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
)
from deepresearch.middlewares.report_evaluation import ReportEvaluationMiddleware
from deepresearch.quality import assess_research
from deepresearch.state import ResearchState, create_initial_state, merge_evaluation_state
from deepresearch.tools import (
    read_page_tool,
    web_search_tool,
    write_final_report_tool,
    write_research_plan_tool,
)


URL_A = "https://a.example/article"
URL_B = "https://b.example/article"


class ProviderStub:
    name = "stub"

    def __init__(
        self,
        *,
        error: Exception | None = None,
        values: dict[str, float] | None = None,
    ) -> None:
        self.calls = 0
        self.error = error
        self.values = values or {
            "answer_relevance": 0.91,
            "evidence_support": 0.82,
            "citation_coverage": 0.79,
            "evidence_sufficient": 0.84,
            "continue_research": 0.12,
        }

    def _response(self) -> DecisionResponse:
        answers = {
            name: DecisionAnswer("noul", value)
            for name, value in self.values.items()
        }
        answers["source_quality"] = DecisionAnswer(
            "score", 2.7, confidence=0.88
        )
        return DecisionResponse(
            provider="stub",
            model="stub-v1",
            answers=answers,
            usage=DecisionUsage(input_tokens=90, output_tokens=8, cost_usd=0.002),
            latency_ms=23,
        )

    def evaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._response()

    async def aevaluate(self, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._response()


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003
        return self


def _config(**updates) -> EvaluationConfig:  # noqa: ANN003
    values = {"use": "jev", "api_key": "test", "mode": "shadow"}
    values.update(updates)
    return EvaluationConfig(**values)


def _ready_state() -> ResearchState:
    state = create_initial_state("研究问题")
    state["messages"].append(AIMessage(content="报告已经生成。"))
    state["plan"] = {
        "goal": "完成研究",
        "steps": [
            {"step_id": "step-1", "title": "搜索", "status": "completed"},
            {"step_id": "step-2", "title": "核验", "status": "completed"},
        ],
    }
    state["sources"] = [
        {"title": "A", "url": URL_A, "snippet": "A", "query": "query"},
        {"title": "B", "url": URL_B, "snippet": "B", "query": "query"},
    ]
    state["page_records"] = [
        {
            "requested_url": URL_A,
            "final_url": URL_A,
            "success": True,
            "content_chars": 100,
            "truncated": False,
            "error": None,
        }
    ]
    state["observations"] = [
        {"content": "Evidence A", "source_url": URL_A, "query": "query"},
        {"content": "Evidence B", "source_url": URL_B, "query": "query"},
    ]
    state["final_report"] = "# 报告\n\n有依据的结论。"
    state["final_report_source_urls"] = [URL_A, URL_B]
    return state


def _middleware(provider: ProviderStub, config: EvaluationConfig | None = None):  # noqa: ANN202
    resolved = config or _config()
    return ReportEvaluationMiddleware(ReportEvaluator(provider, resolved), resolved)


def test_shadow_records_metrics_without_routing() -> None:
    provider = ProviderStub()
    update = _middleware(provider).after_model(_ready_state(), None)  # type: ignore[arg-type]

    assert update is not None
    report = update["evaluation"]["report"]
    assert report["status"] == "completed"
    assert report["mode"] == "shadow"
    assert report["recommended_action"] == "pass"
    assert report["runtime_action"] == "observed"
    assert report["input_tokens"] == 90
    assert report["cost_usd"] == 0.002
    assert "jump_to" not in update
    assert provider.calls == 1


def test_same_signature_is_not_evaluated_twice() -> None:
    provider = ProviderStub()
    middleware = _middleware(provider)
    state = _ready_state()
    first = middleware.after_model(state, None)  # type: ignore[arg-type]
    assert first is not None
    state["evaluation"] = merge_evaluation_state(
        state["evaluation"], first["evaluation"]
    )

    assert middleware.after_model(state, None) is None  # type: ignore[arg-type]
    assert provider.calls == 1


def test_provider_failure_is_recorded_and_fails_open() -> None:
    provider = ProviderStub(error=RuntimeError("secret-provider-detail"))
    update = _middleware(provider).after_model(_ready_state(), None)  # type: ignore[arg-type]

    assert update is not None
    report = update["evaluation"]["report"]
    assert report["status"] == "error"
    assert report["runtime_action"] == "fail_open"
    assert report["last_error"] == "RuntimeError: evaluation failed"
    assert "secret-provider-detail" not in str(update)
    assert "jump_to" not in update


def test_intermediate_and_incomplete_states_are_not_evaluated() -> None:
    provider = ProviderStub()
    state = _ready_state()
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "more"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert _middleware(provider).after_model(state, None) is None  # type: ignore[arg-type]

    incomplete = create_initial_state("question")
    incomplete["messages"].append(AIMessage(content="too early"))
    assert _middleware(provider).after_model(incomplete, None) is None  # type: ignore[arg-type]
    assert provider.calls == 0


def test_async_path_uses_async_provider() -> None:
    provider = ProviderStub()

    async def run():  # noqa: ANN202
        return await _middleware(provider).aafter_model(  # type: ignore[arg-type]
            _ready_state(), None
        )

    update = asyncio.run(run())
    assert update is not None
    assert update["evaluation"]["report"]["status"] == "completed"
    assert provider.calls == 1


def test_agent_graph_runs_shadow_evaluation_after_report_tool() -> None:
    provider = ProviderStub()
    config = _config()
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {
                            "title": "研究报告",
                            "report": (
                                f"结论正文\n\n- [{URL_A}]({URL_A})"
                                f"\n- [{URL_B}]({URL_B})"
                            ),
                            "used_source_urls": [URL_A, URL_B],
                        },
                        "id": "report-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="报告已经生成。"),
        ]
    )
    state = _ready_state()
    state["messages"] = state["messages"][:1]
    state["final_report"] = None
    state["final_report_source_urls"] = []
    graph = build_agent(
        model,
        tools=[
            write_research_plan_tool,
            web_search_tool,
            read_page_tool,
            write_final_report_tool,
        ],
        evaluation_provider=provider,
        evaluation_config=config,
    )

    final_state = graph.invoke(state)

    assert provider.calls == 1, assess_research(final_state)
    assert final_state["evaluation"]["report"]["status"] == "completed"
    assert final_state["evaluation"]["report"]["runtime_action"] == "observed"


def test_gate_continues_research_when_semantic_evidence_is_insufficient() -> None:
    provider = ProviderStub(
        values={
            "answer_relevance": 0.9,
            "evidence_support": 0.8,
            "citation_coverage": 0.8,
            "evidence_sufficient": 0.4,
            "continue_research": 0.91,
        }
    )
    update = _middleware(provider, _config(mode="gate")).after_model(  # type: ignore[arg-type]
        _ready_state(), None
    )

    assert update is not None
    assert update["jump_to"] == "model"
    report = update["evaluation"]["report"]
    assert report["recommended_action"] == "continue_research"
    assert report["runtime_action"] == "continue_research"
    assert report["gate_attempts"] == 1
    assert "继续研究概率 91.0%" in update["research_gaps"][0]


def test_gate_revises_report_when_report_dimensions_are_weak() -> None:
    provider = ProviderStub(
        values={
            "answer_relevance": 0.6,
            "evidence_support": 0.5,
            "citation_coverage": 0.4,
            "evidence_sufficient": 0.9,
            "continue_research": 0.2,
        }
    )
    update = _middleware(provider, _config(mode="gate")).after_model(  # type: ignore[arg-type]
        _ready_state(), None
    )

    assert update is not None
    assert update["jump_to"] == "model"
    report = update["evaluation"]["report"]
    assert report["runtime_action"] == "revise_report"
    assert "相关概率 60.0%" in update["research_gaps"][0]
    assert "再次调用 write_final_report" in update["research_gaps"][0]


def test_gate_pass_does_not_add_an_agent_loop() -> None:
    update = _middleware(ProviderStub(), _config(mode="gate")).after_model(  # type: ignore[arg-type]
        _ready_state(), None
    )

    assert update is not None
    assert update["evaluation"]["report"]["runtime_action"] == "pass"
    assert "jump_to" not in update
    assert "research_gaps" not in update


def test_gate_respects_attempt_limit_and_p5_finalization() -> None:
    values = {
        "answer_relevance": 0.9,
        "evidence_support": 0.8,
        "citation_coverage": 0.8,
        "evidence_sufficient": 0.3,
        "continue_research": 0.95,
    }
    exhausted = _middleware(
        ProviderStub(values=values),
        _config(mode="gate", max_gate_attempts=0),
    ).after_model(_ready_state(), None)  # type: ignore[arg-type]
    assert exhausted is not None
    assert exhausted["evaluation"]["report"]["runtime_action"] == "gate_exhausted"
    assert "jump_to" not in exhausted

    p5_state = _ready_state()
    p5_state["governance"]["context"]["pending_stages"] = ["P5"]
    p5 = _middleware(
        ProviderStub(values=values), _config(mode="gate")
    ).after_model(p5_state, None)  # type: ignore[arg-type]
    assert p5 is not None
    assert p5["evaluation"]["report"]["runtime_action"] == "p5_bypass"
    assert "p5_forced_finalization" in p5["evaluation"]["report"]["notes"]
    assert "jump_to" not in p5


def test_unchanged_gate_signature_is_not_charged_or_looped_again() -> None:
    provider = ProviderStub(
        values={
            "answer_relevance": 0.9,
            "evidence_support": 0.8,
            "citation_coverage": 0.8,
            "evidence_sufficient": 0.4,
            "continue_research": 0.9,
        }
    )
    middleware = _middleware(provider, _config(mode="gate"))
    state = _ready_state()
    first = middleware.after_model(state, None)  # type: ignore[arg-type]
    assert first is not None
    state["evaluation"] = merge_evaluation_state(
        state["evaluation"], first["evaluation"]
    )

    second = middleware.after_model(state, None)  # type: ignore[arg-type]

    assert second is not None
    assert second["evaluation"]["report"]["runtime_action"] == "gate_exhausted"
    assert "jump_to" not in second
    assert provider.calls == 1


def test_agent_graph_gate_adds_one_bounded_semantic_reflection_loop() -> None:
    provider = ProviderStub(
        values={
            "answer_relevance": 0.9,
            "evidence_support": 0.8,
            "citation_coverage": 0.8,
            "evidence_sufficient": 0.4,
            "continue_research": 0.9,
        }
    )
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_final_report",
                        "args": {
                            "title": "研究报告",
                            "report": (
                                f"结论正文\n\n- [{URL_A}]({URL_A})"
                                f"\n- [{URL_B}]({URL_B})"
                            ),
                            "used_source_urls": [URL_A, URL_B],
                        },
                        "id": "report-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="初次报告完成。"),
            AIMessage(content="没有新增证据，结束。"),
        ]
    )
    state = _ready_state()
    state["messages"] = state["messages"][:1]
    state["final_report"] = None
    state["final_report_source_urls"] = []
    graph = build_agent(
        model,
        tools=[
            write_research_plan_tool,
            web_search_tool,
            read_page_tool,
            write_final_report_tool,
        ],
        evaluation_provider=provider,
        evaluation_config=_config(mode="gate"),
    )

    final_state = graph.invoke(state)

    assert provider.calls == 1
    assert final_state["evaluation"]["report"]["gate_attempts"] == 1
    assert final_state["evaluation"]["report"]["runtime_action"] == "gate_exhausted"
    ai_messages = [
        message for message in final_state["messages"] if isinstance(message, AIMessage)
    ]
    assert len(ai_messages) == 3
