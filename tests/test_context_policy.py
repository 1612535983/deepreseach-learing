from __future__ import annotations

import pytest

from deepresearch.context.policy import (
    ContextThresholds,
    evaluate_context_pressure,
)


def test_context_below_first_threshold_has_no_pending_stage() -> None:
    decision = evaluate_context_pressure(399, 1_000)

    assert decision.utilization_ratio == 0.399
    assert decision.pending_stages == ()
    assert decision.hard_limit_reached is False


@pytest.mark.parametrize(
    ("current_tokens", "expected_stages"),
    [
        (400, ("P1",)),
        (500, ("P1", "P2")),
        (600, ("P1", "P2", "P3")),
        (800, ("P1", "P2", "P3", "P4")),
        (900, ("P1", "P2", "P3", "P4", "P5")),
    ],
)
def test_each_exact_threshold_adds_its_stage(
    current_tokens: int,
    expected_stages: tuple[str, ...],
) -> None:
    decision = evaluate_context_pressure(current_tokens, 1_000)

    assert decision.pending_stages == expected_stages
    assert decision.hard_limit_reached is False


def test_hard_limit_is_reported_separately_from_p_stages() -> None:
    decision = evaluate_context_pressure(990, 1_000)

    assert decision.pending_stages == ("P1", "P2", "P3", "P4", "P5")
    assert decision.hard_limit_reached is True


def test_utilization_above_window_is_not_clamped() -> None:
    decision = evaluate_context_pressure(1_250, 1_000)

    assert decision.utilization_ratio == 1.25
    assert decision.hard_limit_reached is True


def test_custom_thresholds_change_the_decision() -> None:
    thresholds = ContextThresholds(
        p1_externalize=0.10,
        p2_thinking=0.20,
        p3_observations=0.30,
        p4_summarize=0.40,
        p5_stop_tool_calls=0.50,
        hard_stop=0.60,
    )

    decision = evaluate_context_pressure(450, 1_000, thresholds)

    assert decision.pending_stages == ("P1", "P2", "P3", "P4")
    assert decision.hard_limit_reached is False


@pytest.mark.parametrize(
    ("current_tokens", "window_tokens"),
    [
        (-1, 1_000),
        (100, 0),
        (100, -1),
        (True, 1_000),
        (100, True),
    ],
)
def test_token_counts_must_be_valid_integers(
    current_tokens: int,
    window_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        evaluate_context_pressure(current_tokens, window_tokens)


def test_thresholds_must_stay_in_range() -> None:
    with pytest.raises(ValueError, match="大于 0 且不超过 1"):
        ContextThresholds(p1_externalize=0)


def test_thresholds_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="严格递增"):
        ContextThresholds(p2_thinking=0.40)
