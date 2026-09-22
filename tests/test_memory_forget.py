import pytest

from deepresearch.memory.schema import MemoryTrace, MemoryType
from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy


def _trace(**changes: object) -> MemoryTrace:
    values = {
        "id": "trace",
        "namespace": "default",
        "content": "memory",
        "type": MemoryType.EPISODIC,
        "strength": 0.7,
        "base_strength": 0.7,
        "decay_rate": 0.1,
        "last_accessed": 1_000.0,
        "importance": 0.5,
        "created_at": 1_000.0,
        "metadata": {},
    }
    values.update(changes)
    return MemoryTrace(**values)  # type: ignore[arg-type]


def test_forget_policy_honors_explicit_soft_delete() -> None:
    policy = CompositeForgetPolicy(EbbinghausDecayPolicy())

    assert policy.should_forget(_trace(metadata={"forgotten": True}), 1_000.0)


def test_forget_policy_expires_trace_by_ttl() -> None:
    policy = CompositeForgetPolicy(
        EbbinghausDecayPolicy(),
        ttl_hours=2,
    )

    assert policy.should_forget(_trace(), 1_000.0 + 2 * 3_600.0)


def test_forget_policy_uses_lazy_decayed_strength() -> None:
    policy = CompositeForgetPolicy(
        EbbinghausDecayPolicy(),
        strength_threshold=0.2,
        ttl_hours=10_000,
    )

    assert policy.should_forget(_trace(), 1_000.0 + 100 * 3_600.0)


def test_forget_policy_validates_configuration() -> None:
    with pytest.raises(ValueError, match="strength_threshold"):
        CompositeForgetPolicy(EbbinghausDecayPolicy(), strength_threshold=1.1)
    with pytest.raises(ValueError, match="ttl_hours"):
        CompositeForgetPolicy(EbbinghausDecayPolicy(), ttl_hours=0)
