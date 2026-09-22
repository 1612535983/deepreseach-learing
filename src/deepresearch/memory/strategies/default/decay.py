"""Lazy Ebbinghaus-style strength calculation for memory traces."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from deepresearch.memory.schema import MemoryTrace


class EbbinghausDecayPolicy:
    """Calculate current strength only when a trace is considered."""

    def __init__(self, *, now_provider: Callable[[], float] = time.time) -> None:
        self._now_provider = now_provider

    def compute_strength(
        self,
        trace: MemoryTrace,
        now: float | None = None,
    ) -> float:
        """Return decayed strength plus access and importance reinforcement."""

        current_time = self._now_provider() if now is None else now
        reference_time = trace.last_accessed or trace.created_at or current_time
        elapsed_hours = max(0.0, current_time - reference_time) / 3600.0
        retention = trace.base_strength * (1.0 - trace.decay_rate) ** elapsed_hours
        access_bonus = math.log1p(max(0, trace.access_count)) * 0.1
        importance_bonus = max(0.0, min(1.0, trace.importance)) * 0.05
        return max(0.0, min(1.0, retention + access_bonus + importance_bonus))
