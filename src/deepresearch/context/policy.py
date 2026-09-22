"""Pure threshold policy for classifying context-window pressure."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.context.types import ContextStage


@dataclass(frozen=True)
class ContextThresholds:
    """Ordered utilization thresholds for the P1-P5 governance stages."""

    p1_externalize: float = 0.40
    p2_thinking: float = 0.50
    p3_observations: float = 0.60
    p4_summarize: float = 0.80
    p5_stop_tool_calls: float = 0.90
    hard_stop: float = 0.99

    def __post_init__(self) -> None:
        values = (
            self.p1_externalize,
            self.p2_thinking,
            self.p3_observations,
            self.p4_summarize,
            self.p5_stop_tool_calls,
            self.hard_stop,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value <= 1
            for value in values
        ):
            raise ValueError("上下文治理阈值必须是大于 0 且不超过 1 的数字。")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("上下文治理阈值必须按照 P1 到 hard stop 严格递增。")


@dataclass(frozen=True)
class ContextDecision:
    """One deterministic classification of the current context pressure."""

    current_tokens: int
    window_tokens: int
    utilization_ratio: float
    pending_stages: tuple[ContextStage, ...]
    hard_limit_reached: bool


DEFAULT_CONTEXT_THRESHOLDS = ContextThresholds()


def _validate_token_count(name: str, value: int, *, allow_zero: bool) -> None:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        qualifier = "大于等于 0" if allow_zero else "大于 0"
        raise ValueError(f"{name} 必须是{qualifier}的整数。")


def evaluate_context_pressure(
    current_tokens: int,
    window_tokens: int,
    thresholds: ContextThresholds = DEFAULT_CONTEXT_THRESHOLDS,
) -> ContextDecision:
    """Return the cumulative P1-P5 stages reached by the current utilization."""

    _validate_token_count("current_tokens", current_tokens, allow_zero=True)
    _validate_token_count("window_tokens", window_tokens, allow_zero=False)
    utilization_ratio = current_tokens / window_tokens

    stage_thresholds: tuple[tuple[ContextStage, float], ...] = (
        ("P1", thresholds.p1_externalize),
        ("P2", thresholds.p2_thinking),
        ("P3", thresholds.p3_observations),
        ("P4", thresholds.p4_summarize),
        ("P5", thresholds.p5_stop_tool_calls),
    )
    pending_stages = tuple(
        stage
        for stage, threshold in stage_thresholds
        if utilization_ratio >= threshold
    )
    return ContextDecision(
        current_tokens=current_tokens,
        window_tokens=window_tokens,
        utilization_ratio=utilization_ratio,
        pending_stages=pending_stages,
        hard_limit_reached=utilization_ratio >= thresholds.hard_stop,
    )
