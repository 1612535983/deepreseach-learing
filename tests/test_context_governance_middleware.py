from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_with_model
from deepresearch.checkpointing import create_in_memory_checkpointer, get_checkpoint_state
from deepresearch.middlewares.context_governance import ContextGovernanceMiddleware
from deepresearch.state import ResearchState, create_initial_state, merge_governance


class FixedGovernanceModel:
    model_name = "governance-test-model"
    max_input_tokens = 1_000

    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        assert messages
        return 500


def state_with_usage() -> ResearchState:
    state = create_initial_state("研究上下文")
    state["messages"].append(
        AIMessage(
            id="response-1",
            content="研究结果",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 100,
                "total_tokens": 500,
            },
        )
    )
    return state


def test_middleware_records_budget_policy_and_usage_without_mutating_messages() -> None:
    state = state_with_usage()
    original_messages = list(state["messages"])
    middleware = ContextGovernanceMiddleware(FixedGovernanceModel())

    update = middleware.after_model(state, None)  # type: ignore[arg-type]

    context = update["governance"]["context"]
    assert context["model_name"] == "governance-test-model"
    assert context["budget"] == {
        "current_tokens": 500,
        "window_tokens": 1_000,
        "utilization_ratio": 0.5,
        "token_count_method": "model",
        "window_source": "model_attribute",
    }
    assert context["cumulative_usage"] == {
        "input_tokens": 400,
        "output_tokens": 100,
        "total_tokens": 500,
    }
    assert context["model_call_count"] == 1
    assert context["pending_stages"] == ["P1", "P2"]
    assert context["hard_limit_reached"] is False
    assert state["messages"] == original_messages
    assert "jump_to" not in update


def test_middleware_does_not_count_the_same_usage_twice() -> None:
    state = state_with_usage()
    middleware = ContextGovernanceMiddleware(FixedGovernanceModel())
    first = middleware.after_model(state, None)  # type: ignore[arg-type]
    state["governance"] = merge_governance(
        state["governance"],
        first["governance"],
    )

    second = middleware.after_model(state, None)  # type: ignore[arg-type]

    assert second["governance"]["context"]["cumulative_usage"] == {
        "input_tokens": 400,
        "output_tokens": 100,
        "total_tokens": 500,
    }
    assert second["governance"]["context"]["model_call_count"] == 2


def test_middleware_accepts_old_state_without_governance_or_usage_metadata() -> None:
    state = create_initial_state("旧任务")
    del state["governance"]
    state["messages"].append(AIMessage(content="旧任务答案"))

    update = ContextGovernanceMiddleware(FixedGovernanceModel()).after_model(
        state,
        None,  # type: ignore[arg-type]
    )

    context = update["governance"]["context"]
    assert context["cumulative_usage"]["total_tokens"] == 0
    assert context["seen_message_usage"] == {}
    assert context["model_call_count"] == 1


def test_async_middleware_uses_the_same_calculation() -> None:
    state = state_with_usage()
    middleware = ContextGovernanceMiddleware(FixedGovernanceModel())

    sync_update = middleware.after_model(state, None)  # type: ignore[arg-type]
    async_update = asyncio.run(
        middleware.aafter_model(state, None)  # type: ignore[arg-type]
    )

    assert async_update == sync_update


def test_build_agent_records_and_checkpoints_governance_automatically() -> None:
    checkpointer = create_in_memory_checkpointer()
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                id="integrated-response",
                content="已完成",
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                },
            )
        ]
    )

    result = run_with_model(
        "集成测试",
        model,
        checkpointer=checkpointer,
        thread_id="governance-thread",
    )
    saved_state = get_checkpoint_state(checkpointer, "governance-thread")
    context = result.state["governance"]["context"]

    assert context["model_call_count"] == 1
    assert context["cumulative_usage"]["total_tokens"] == 25
    assert context["budget"]["current_tokens"] > 0
    assert context["budget"]["window_tokens"] > 0
    assert saved_state["governance"] == result.state["governance"]
