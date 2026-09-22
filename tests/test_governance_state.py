from __future__ import annotations

import json

from deepresearch.state import (
    create_initial_governance_state,
    create_initial_state,
    merge_governance,
)


def test_initial_state_contains_complete_serializable_governance() -> None:
    state = create_initial_state("研究上下文治理")

    assert state["governance"] == create_initial_governance_state()
    assert state["governance"]["context"]["budget"] == {
        "current_tokens": 0,
        "window_tokens": 0,
        "utilization_ratio": 0.0,
        "token_count_method": "unknown",
        "window_source": "unknown",
    }
    assert state["governance"]["context"]["cumulative_usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    json.dumps(state["governance"])


def test_governance_deep_merge_preserves_unrelated_fields() -> None:
    current = create_initial_governance_state()
    current["context"]["model_name"] = "deepseek-chat"
    current["context"]["model_call_count"] = 2
    current["context"]["seen_message_usage"] = {
        "message-1": {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "total_tokens": 1_100,
        }
    }

    merged = merge_governance(
        current,
        {
            "context": {
                "budget": {
                    "current_tokens": 8_000,
                    "window_tokens": 65_536,
                    "utilization_ratio": 0.122,
                },
                "pending_stages": ["P1"],
            }
        },
    )

    context = merged["context"]
    assert context["model_name"] == "deepseek-chat"
    assert context["model_call_count"] == 2
    assert context["budget"]["current_tokens"] == 8_000
    assert context["budget"]["token_count_method"] == "unknown"
    assert context["cumulative_usage"]["total_tokens"] == 0
    assert context["seen_message_usage"]["message-1"]["total_tokens"] == 1_100
    assert context["pending_stages"] == ["P1"]


def test_governance_lists_are_replaced_and_inputs_are_not_mutated() -> None:
    current = create_initial_governance_state()
    current["context"]["pending_stages"] = ["P1", "P2", "P3", "P4"]
    incoming = {"context": {"pending_stages": ["P1"]}}

    merged = merge_governance(current, incoming)
    merged["context"]["pending_stages"].append("P2")

    assert current["context"]["pending_stages"] == ["P1", "P2", "P3", "P4"]
    assert incoming["context"]["pending_stages"] == ["P1"]
    assert merged["context"]["pending_stages"] == ["P1", "P2"]


def test_null_governance_patch_preserves_a_detached_copy() -> None:
    current = create_initial_governance_state()

    merged = merge_governance(current, None)
    merged["context"]["budget"]["current_tokens"] = 99

    assert current["context"]["budget"]["current_tokens"] == 0
