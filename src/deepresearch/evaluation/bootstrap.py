"""Construct the optional probabilistic decision provider from settings."""

from __future__ import annotations

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.jev import JevDecisionProvider
from deepresearch.evaluation.provider import DecisionProvider


def get_evaluation_provider(config: EvaluationConfig) -> DecisionProvider | None:
    """Return the configured provider; an empty ``use`` keeps evaluation off."""

    if not config.enabled:
        return None
    if config.use == "jev":
        return JevDecisionProvider(config)
    raise ValueError(f"不支持的评估 Provider：{config.use}")
