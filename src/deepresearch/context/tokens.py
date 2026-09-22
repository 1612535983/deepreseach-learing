"""Pure helpers for estimating context size and collecting model token usage."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from deepresearch.context.types import CumulativeTokenUsage, TokenCountMethod


_CJK_RANGES = (
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)


@dataclass(frozen=True)
class TokenEstimate:
    """Estimated size of one model request and the method used to count it."""

    token_count: int
    method: TokenCountMethod


@dataclass(frozen=True)
class TokenUsageUpdate:
    """New provider-reported usage plus the updated idempotency ledger."""

    delta: CumulativeTokenUsage
    seen_message_usage: dict[str, CumulativeTokenUsage]


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(_json_text(item))
        return "".join(parts)
    return str(content) if content is not None else ""


def _message_text(message: BaseMessage) -> str:
    """Render message data that contributes to the next model request."""

    parts = [message.type, _content_text(message.content)]
    if message.name:
        parts.append(message.name)

    if isinstance(message, AIMessage):
        if message.tool_calls:
            parts.append(_json_text(message.tool_calls))
        if message.invalid_tool_calls:
            parts.append(_json_text(message.invalid_tool_calls))
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning:
            parts.append(_content_text(reasoning))
    elif isinstance(message, ToolMessage):
        parts.append(message.tool_call_id)

    return "\n".join(parts)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _character_token_estimate(text: str) -> int:
    cjk_characters = sum(_is_cjk(character) for character in text)
    other_characters = len(text) - cjk_characters
    return cjk_characters + math.ceil(other_characters / 4)


def _fallback_token_count(messages: Sequence[BaseMessage]) -> int:
    if not messages:
        return 0
    # Four tokens per message and two tokens for reply priming approximate the
    # structural overhead used by common chat-model wire formats.
    return 2 + sum(
        4 + _character_token_estimate(_message_text(message))
        for message in messages
    )


def estimate_context_tokens(
    messages: Sequence[BaseMessage],
    model: Any | None = None,
) -> TokenEstimate:
    """Estimate message tokens, preferring the supplied model's own counter."""

    message_list = list(messages)
    model_counter = getattr(model, "get_num_tokens_from_messages", None)
    counter_implementation = getattr(
        type(model),
        "get_num_tokens_from_messages",
        None,
    )
    uses_generic_counter = (
        counter_implementation is BaseLanguageModel.get_num_tokens_from_messages
    )
    if callable(model_counter) and not uses_generic_counter:
        try:
            token_count = model_counter(message_list)
        except Exception:
            token_count = None
        if (
            isinstance(token_count, int)
            and not isinstance(token_count, bool)
            and token_count >= 0
        ):
            return TokenEstimate(token_count=token_count, method="model")

    return TokenEstimate(
        token_count=_fallback_token_count(message_list),
        method="char_estimate",
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _normalized_usage(message: AIMessage) -> CumulativeTokenUsage | None:
    usage = message.usage_metadata
    if not isinstance(usage, Mapping):
        return None

    input_tokens = _non_negative_int(usage.get("input_tokens"))
    output_tokens = _non_negative_int(usage.get("output_tokens"))
    reported_total = _non_negative_int(usage.get("total_tokens"))
    total_tokens = reported_total or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _usage_key(message: AIMessage, position: int) -> str:
    return f"id:{message.id}" if message.id else f"position:{position}"


def collect_usage_delta(
    messages: Sequence[BaseMessage],
    seen_message_usage: Mapping[str, CumulativeTokenUsage] | None = None,
) -> TokenUsageUpdate:
    """Collect only usage not already recorded for the same AI messages."""

    seen = deepcopy(dict(seen_message_usage or {}))
    delta: CumulativeTokenUsage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for position, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        current_usage = _normalized_usage(message)
        if current_usage is None:
            continue

        key = _usage_key(message, position)
        previous_usage = seen.get(key, {})
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            current_value = current_usage[field]
            previous_value = _non_negative_int(previous_usage.get(field))
            delta[field] += max(0, current_value - previous_value)

        seen[key] = {
            field: max(
                current_usage[field],
                _non_negative_int(previous_usage.get(field)),
            )
            for field in ("input_tokens", "output_tokens", "total_tokens")
        }

    return TokenUsageUpdate(delta=delta, seen_message_usage=seen)
