"""Persist version-scoped Skill outcomes after an Agent run completes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from deepresearch.skill.manager import SkillManager
from deepresearch.skill.runtime import resolve_skill_run_id
from deepresearch.state import ResearchState


class SkillMetricsMiddleware(AgentMiddleware):
    """Measure tool alignment and terminal outcomes for injected Skill versions."""

    state_schema = ResearchState

    def __init__(self, manager: SkillManager) -> None:
        self._manager = manager

    def _record(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        skills = state.get("skills") or {}
        if skills.get("completed_recorded"):
            return None
        selected = skills.get("selected") or []
        dropped_ids = {
            str(ref.get("skill_id"))
            for ref in skills.get("dropped", [])
            if isinstance(ref, Mapping)
        }
        included = [
            ref
            for ref in selected
            if isinstance(ref, Mapping)
            and str(ref.get("skill_id") or "") not in dropped_ids
        ]
        if not included:
            return None

        try:
            run_id = resolve_skill_run_id(state, runtime)
            aligned = 0
            for message in state.get("messages", []):
                if not isinstance(message, AIMessage):
                    continue
                for position, tool_call in enumerate(message.tool_calls):
                    tool_name = str(tool_call.get("name") or "")
                    if not tool_name:
                        continue
                    tool_call_id = str(tool_call.get("id") or "") or hashlib.sha256(
                        f"{message.id}:{position}:{tool_name}".encode("utf-8")
                    ).hexdigest()[:16]
                    for ref in included:
                        allowed_tools = {
                            str(item) for item in ref.get("allowed_tools", [])
                        }
                        if tool_name in allowed_tools and self._manager.store.record_tool_alignment(
                            str(ref["skill_id"]),
                            run_id,
                            tool_call_id,
                            tool_name,
                        ):
                            aligned += 1

            completed, note = self._outcome(state)
            for ref in included:
                self._manager.store.record_outcome(
                    str(ref["skill_id"]),
                    run_id,
                    completed=completed,
                    note=note,
                )
            return {
                "skills": {
                    "aligned_tool_calls": int(
                        skills.get("aligned_tool_calls", 0)
                    )
                    + aligned,
                    "completed_recorded": True,
                    "last_error": None,
                }
            }
        except Exception as exc:
            return {
                "skills": {
                    "completed_recorded": True,
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            }

    @staticmethod
    def _outcome(state: ResearchState) -> tuple[bool, str]:
        if str(state.get("final_report") or "").strip():
            return True, "final_report"
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and not last.tool_calls and str(last.content).strip():
            return True, "final_answer"
        return False, "no_terminal_answer"

    @override
    def after_agent(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._record(state, runtime)

    @override
    async def aafter_agent(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._record(state, runtime)
