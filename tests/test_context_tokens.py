from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepresearch.context.tokens import (
    collect_usage_delta,
    estimate_context_tokens,
)


class FixedTokenCounter:
    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        assert messages
        return 321


class FailingTokenCounter:
    def get_num_tokens_from_messages(self, messages: list[object]) -> int:
        raise RuntimeError("tokenizer unavailable")


def test_context_estimate_prefers_model_counter() -> None:
    estimate = estimate_context_tokens(
        [HumanMessage(content="研究 LangChain")],
        FixedTokenCounter(),
    )

    assert estimate.token_count == 321
    assert estimate.method == "model"


def test_context_estimate_falls_back_to_cjk_aware_character_count() -> None:
    estimate = estimate_context_tokens(
        [HumanMessage(content="研究 Agent with tools")],
        FailingTokenCounter(),
    )

    assert estimate.token_count > 0
    assert estimate.method == "char_estimate"


def test_context_estimate_skips_langchain_generic_gpt2_counter() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])

    estimate = estimate_context_tokens([HumanMessage(content="研究上下文")], model)

    assert estimate.token_count > 0
    assert estimate.method == "char_estimate"


def test_context_estimate_includes_tool_calls_and_tool_results() -> None:
    plain = estimate_context_tokens([AIMessage(content="")])
    with_tools = estimate_context_tokens(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_page",
                        "args": {"url": "https://example.com/long-article"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="这是一段从网页中读取的正文。" * 20,
                tool_call_id="call-1",
            ),
        ]
    )

    assert with_tools.token_count > plain.token_count
    assert with_tools.method == "char_estimate"


def test_context_estimate_supports_multiblock_content() -> None:
    estimate = estimate_context_tokens(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "研究上下文"},
                    {"type": "text", "text": "context governance"},
                ]
            )
        ]
    )

    assert estimate.token_count > 0


def test_usage_collection_is_idempotent_by_message_id() -> None:
    messages = [
        AIMessage(
            id="response-1",
            content="完成",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 120,
                "total_tokens": 1_120,
            },
        )
    ]

    first = collect_usage_delta(messages)
    second = collect_usage_delta(messages, first.seen_message_usage)

    assert first.delta == {
        "input_tokens": 1_000,
        "output_tokens": 120,
        "total_tokens": 1_120,
    }
    assert second.delta == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_usage_collection_records_only_incremental_usage() -> None:
    seen = {
        "id:response-1": {
            "input_tokens": 800,
            "output_tokens": 100,
            "total_tokens": 900,
        }
    }
    messages = [
        AIMessage(
            id="response-1",
            content="完成",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 140,
                "total_tokens": 1_140,
            },
        )
    ]

    update = collect_usage_delta(messages, seen)

    assert update.delta == {
        "input_tokens": 200,
        "output_tokens": 40,
        "total_tokens": 240,
    }
    assert seen["id:response-1"]["total_tokens"] == 900
    assert update.seen_message_usage["id:response-1"]["total_tokens"] == 1_140


def test_usage_collection_uses_position_when_message_id_is_missing() -> None:
    message = AIMessage(
        content="完成",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )

    first = collect_usage_delta([message])
    second = collect_usage_delta([message], first.seen_message_usage)

    assert first.seen_message_usage["position:0"]["total_tokens"] == 15
    assert second.delta["total_tokens"] == 0
