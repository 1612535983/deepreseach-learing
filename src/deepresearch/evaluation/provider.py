"""Provider-neutral interface consumed by evaluation middleware."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from deepresearch.evaluation.types import DecisionQuestion, DecisionResponse


class DecisionProvider(Protocol):
    """Evaluate atomic typed questions against one shared state."""

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionResponse: ...

    async def aevaluate(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionResponse: ...
