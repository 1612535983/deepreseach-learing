"""Select skills once per distinct question and persist compact references."""

from __future__ import annotations

import hashlib
import json
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deepresearch.skill.manager import SkillManager
from deepresearch.skill.runtime import resolve_skill_run_id
from deepresearch.skill.types import SkillRef
from deepresearch.state import ResearchState


class SkillSelectionMiddleware(AgentMiddleware):
    """Bridge the external skill catalog into checkpoint-safe run state."""

    state_schema = ResearchState

    def __init__(
        self,
        manager: SkillManager,
        *,
        available_tools: tuple[str, ...],
        overrides: tuple[str, ...] = (),
    ) -> None:
        self._manager = manager
        self._available_tools = available_tools
        self._overrides = overrides

    def _select(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        question = str(state.get("research_question") or "").strip()
        if not question:
            return None
        current = state.get("skills") or {}
        if int(current.get("selection_count", 0)) > 0:
            return None
        error_hash = hashlib.sha256(
            f"{question}\x00skill-selection-error".encode("utf-8")
        ).hexdigest()
        if current.get("query_hash") == error_hash:
            return None
        try:
            catalog_hash = self._manager.selector.catalog_hash()
            fingerprint = json.dumps(
                {
                    "question": question,
                    "catalog_hash": catalog_hash,
                    "overrides": self._overrides,
                    "available_tools": self._available_tools,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            query_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
            if current.get("query_hash") == query_hash:
                return None
            selections = self._manager.select(
                question,
                overrides=self._overrides,
                available_tools=self._available_tools,
            )
            refs: list[SkillRef] = [
                {
                    "skill_id": selection.record.skill_id,
                    "name": selection.record.name,
                    "content_hash": selection.record.content_hash,
                    "score": selection.score,
                    "reason": selection.reason,
                    "forced": selection.forced,
                    "allowed_tools": list(selection.record.allowed_tools),
                }
                for selection in selections
            ]
            if self._manager.config.enable_metrics:
                run_id = resolve_skill_run_id(state, runtime)
                for selection in selections:
                    self._manager.store.record_selection(
                        selection.record.skill_id, run_id
                    )
            return {
                "skills": {
                    "query_hash": query_hash,
                    "catalog_hash": catalog_hash,
                    "selected": refs,
                    "dropped": [],
                    "selection_count": int(current.get("selection_count", 0)) + 1,
                    "injection_count": 0,
                    "injected_tokens": 0,
                    "render_signature": None,
                    "aligned_tool_calls": 0,
                    "completed_recorded": False,
                    "evaluation_recorded": False,
                    "evaluation_count": 0,
                    "evaluation_status": None,
                    "evaluation_error": None,
                    "last_error": None,
                }
            }
        except Exception as exc:
            catalog_hash = None
            query_hash = error_hash
            return {
                "skills": {
                    "query_hash": query_hash,
                    "catalog_hash": catalog_hash,
                    "selected": [],
                    "dropped": [],
                    "selection_count": int(current.get("selection_count", 0)) + 1,
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            }

    @override
    def before_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._select(state, runtime)

    @override
    async def abefore_model(
        self,
        state: ResearchState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        return self._select(state, runtime)
