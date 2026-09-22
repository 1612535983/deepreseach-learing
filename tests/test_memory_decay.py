import pytest

from deepresearch.memory.schema import MemoryTrace, MemoryType
from deepresearch.memory.strategies.default.constants import DECAY_PARAMS
from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy


def _trace(
    type: MemoryType,
    *,
    access_count: int = 0,
    importance: float = 0.0,
) -> MemoryTrace:
    params = DECAY_PARAMS[type]
    return MemoryTrace(
        id=type.value,
        namespace="default",
        content=type.value,
        type=type,
        strength=params["base_strength"],
        base_strength=params["base_strength"],
        decay_rate=params["decay_rate"],
        access_count=access_count,
        last_accessed=1_000.0,
        importance=importance,
        created_at=1_000.0,
    )


def test_strength_decreases_as_elapsed_time_grows() -> None:
    policy = EbbinghausDecayPolicy()
    trace = _trace(MemoryType.EPISODIC)

    after_one_hour = policy.compute_strength(trace, 1_000.0 + 3_600.0)
    after_ten_hours = policy.compute_strength(trace, 1_000.0 + 36_000.0)

    assert after_ten_hours < after_one_hour


def test_semantic_memory_decays_slower_than_episodic_memory() -> None:
    policy = EbbinghausDecayPolicy()
    now = 1_000.0 + 36_000.0

    episodic = policy.compute_strength(_trace(MemoryType.EPISODIC), now)
    semantic = policy.compute_strength(_trace(MemoryType.SEMANTIC), now)

    assert semantic > episodic


def test_access_and_importance_reinforce_strength() -> None:
    policy = EbbinghausDecayPolicy()
    now = 1_000.0 + 36_000.0

    plain = policy.compute_strength(_trace(MemoryType.EPISODIC), now)
    reinforced = policy.compute_strength(
        _trace(MemoryType.EPISODIC, access_count=4, importance=1.0),
        now,
    )

    assert reinforced > plain
    assert reinforced <= 1.0


def test_strength_at_creation_includes_importance_bonus() -> None:
    policy = EbbinghausDecayPolicy()
    trace = _trace(MemoryType.SEMANTIC, importance=0.8)

    assert policy.compute_strength(trace, 1_000.0) == pytest.approx(0.84)
