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


class ContextExternalizationMetrics(TypedDict, total=False):
    """Cumulative P1 results plus the outcome of the latest attempt."""

    externalized_tool_results: int
    original_chars: int
    retained_chars: int
    estimated_tokens_saved: int
    last_externalized_paths: list[str]
    last_error: str | None


class ContextCompactionMetrics(TypedDict, total=False):
    """Cumulative P4 activity plus the latest compaction measurements."""

    snapshot_count: int
    summarize_count: int
    removed_message_count: int
    estimated_tokens_saved: int
    last_preserved_message_count: int
    last_tokens_before: int
    last_tokens_after: int
    last_snapshot_path: str | None
    last_summary_id: str | None
    last_error: str | None


class ContextGovernanceState(TypedDict, total=False):
    """Checkpoint-safe measurements and decisions for context governance."""

    model_name: str | None
    budget: ContextBudget
    cumulative_usage: CumulativeTokenUsage
    model_call_count: int
    pending_stages: list[ContextStage]
    hard_limit_reached: bool
    seen_message_usage: dict[str, CumulativeTokenUsage]
    externalization: ContextExternalizationMetrics
    summary: str | None
    summary_id: str | None
    compaction: ContextCompactionMetrics


class GovernanceState(TypedDict, total=False):
    """Top-level namespace reserved for governance subsystems."""

    context: ContextGovernanceState


class TaggedContextState(TypedDict, total=False):
    """Latest audit snapshot of the request-scoped context shown to the model."""

    rendered: str
    message_count: int
    created_at: str
