"""Serializable state types used by context-governance components."""

from __future__ import annotations

from typing import Literal, TypedDict


ContextStage = Literal["P1", "P2", "P3", "P4", "P5"]
TokenCountMethod = Literal["model", "tokenizer", "char_estimate", "unknown"]
WindowSource = Literal[
    "config",
    "model_attribute",
    "model_map",
    "fallback",
    "unknown",
]


class ContextBudget(TypedDict, total=False):
    """Current context size relative to the active model's input window."""

    current_tokens: int
    window_tokens: int
    utilization_ratio: float
    token_count_method: TokenCountMethod
    window_source: WindowSource


class CumulativeTokenUsage(TypedDict, total=False):
    """Provider-reported token usage accumulated across model calls."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ContextGovernanceState(TypedDict, total=False):
    """Checkpoint-safe measurements and decisions for context governance."""

    model_name: str | None
    budget: ContextBudget
    cumulative_usage: CumulativeTokenUsage
    model_call_count: int
    pending_stages: list[ContextStage]
    hard_limit_reached: bool
    seen_message_usage: dict[str, CumulativeTokenUsage]


class GovernanceState(TypedDict, total=False):
    """Top-level namespace reserved for governance subsystems."""

    context: ContextGovernanceState


class TaggedContextState(TypedDict, total=False):
    """Latest audit snapshot of the request-scoped context shown to the model."""

    rendered: str
    message_count: int
    created_at: str
