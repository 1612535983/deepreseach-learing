"""Shared runtime identifiers for idempotent skill metrics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def resolve_skill_run_id(state: Mapping[str, Any], runtime: object) -> str:
    """Prefer the checkpoint thread id, with a stable question hash fallback."""

    try:
        config = getattr(runtime, "config", None) or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    except (AttributeError, TypeError):
        pass
    skills = state.get("skills")
    if isinstance(skills, Mapping):
        run_id = skills.get("run_id")
        if run_id:
            return str(run_id)
    question = str(state.get("research_question") or "uncheckpointed")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
