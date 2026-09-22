from deepresearch.state import (
    create_initial_memory_state,
    create_initial_state,
    merge_memory_runtime,
)


def test_initial_state_contains_memory_runtime_metrics_only() -> None:
    state = create_initial_state("研究问题")

    assert state["memory"] == create_initial_memory_state()
    assert "content" not in state["memory"]


def test_memory_runtime_patch_replaces_current_recall_and_preserves_metrics() -> None:
    current = create_initial_memory_state("project-a")
    current["recall_count"] = 2
    current["recalled"] = [{"id": "old", "score": 0.2, "strength": 0.3}]

    merged = merge_memory_runtime(
        current,
        {
            "recalled": [{"id": "new", "score": 0.9, "strength": 0.8}],
            "last_query_hash": "hash",
        },  # type: ignore[arg-type]
    )

    assert merged["namespace"] == "project-a"
    assert merged["recall_count"] == 2
    assert merged["recalled"] == [{"id": "new", "score": 0.9, "strength": 0.8}]
    assert merged["last_query_hash"] == "hash"
