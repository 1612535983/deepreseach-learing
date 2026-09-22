"""TTL and strength-threshold forgetting decisions."""

from __future__ import annotations

import time
from collections.abc import Callable

from deepresearch.memory.schema import MemoryTrace
from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy


class CompositeForgetPolicy:
    """Forget traces that are explicitly forgotten, expired, or too weak."""

    def __init__(
        self,
        decay_policy: EbbinghausDecayPolicy,
        *,
        strength_threshold: float = 0.1,
        ttl_hours: float = 720.0,
        now_provider: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 <= strength_threshold <= 1.0:
            raise ValueError("strength_threshold 必须在 0 到 1 之间。")
        if ttl_hours <= 0:
            raise ValueError("ttl_hours 必须大于 0。")
        self._decay_policy = decay_policy
        self._strength_threshold = strength_threshold
        self._ttl_hours = ttl_hours
        self._now_provider = now_provider

    def should_forget(
        self,
        trace: MemoryTrace,
        now: float | None = None,
    ) -> bool:
        """Evaluate explicit marker, absolute age, and lazy current strength."""

        if trace.metadata.get("forgotten") is True:
            return True
        current_time = self._now_provider() if now is None else now
        if trace.created_at > 0:
            age_hours = max(0.0, current_time - trace.created_at) / 3600.0
            if age_hours >= self._ttl_hours:
                return True
        return self._decay_policy.compute_strength(trace, current_time) < self._strength_threshold
