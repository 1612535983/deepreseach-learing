"""Audit bounded skill projection without modifying canonical messages."""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deepresearch.skill.context import SkillContextRenderer
from deepresearch.skill.manager import SkillManager
from deepresearch.skill.runtime import resolve_skill_run_id
from deepresearch.state import ResearchState


class SkillInjectionMiddleware(AgentMiddleware):
    """Record which immutable skills the request-scoped renderer can inject."""

    state_schema = ResearchState

    def __init__(
        self,
        manager: SkillManager,
        renderer: SkillContextRenderer,
    ) -> None:
        self._manager = manager
        self._renderer = renderer

    def _inject(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        skills = state.get("skills") or {}
        if not skills.get("selected"):
            return None
        rendered = self._renderer.render(state)
        if skills.get("render_signature") == rendered.signature:
            return None
        if self._manager.config.enable_metrics:
            run_id = resolve_skill_run_id(state, runtime)
            for ref in rendered.included:
                self._manager.store.record_injection(ref["skill_id"], run_id)
        return {
            "skills": {
                "dropped": list(rendered.dropped),
                "injection_count": int(skills.get("injection_count", 0))
                + (1 if rendered.included else 0),
                "injected_tokens": rendered.token_count,
                "render_signature": rendered.signature,
                "last_error": rendered.error,
            }
        }

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._inject(state, runtime)

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._inject(state, runtime)
