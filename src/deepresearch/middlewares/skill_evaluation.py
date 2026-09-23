"""Persist fail-open, shadow-only judgments for injected skill versions."""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deepresearch.evaluation.skill import SkillRunEvaluator
from deepresearch.skill.runtime import resolve_skill_run_id
from deepresearch.state import ResearchState


class SkillEvaluationMiddleware(AgentMiddleware):
    """Evaluate once after a run; provider errors never fail the user task."""

    state_schema = ResearchState

    def __init__(self, evaluator: SkillRunEvaluator) -> None:
        self._evaluator = evaluator

    def _skip(self, state: ResearchState) -> bool:
        skills = state.get("skills") or {}
        if skills.get("evaluation_recorded"):
            return True
        selected = {
            str(item.get("skill_id") or "")
            for item in skills.get("selected", [])
            if isinstance(item, dict)
        }
        dropped = {
            str(item.get("skill_id") or "")
            for item in skills.get("dropped", [])
            if isinstance(item, dict)
        }
        return not bool(selected.difference(dropped).difference({""}))

    def _persist(
        self, state: ResearchState, evaluations: list[Any]
    ) -> dict[str, Any]:
        inserted = sum(
            self._evaluator.store.save_evaluation(evaluation)
            for evaluation in evaluations
        )
        skills = state.get("skills") or {}
        return {
            "skills": {
                "evaluation_recorded": True,
                "evaluation_count": int(skills.get("evaluation_count", 0))
                + inserted,
                "evaluation_status": "completed",
                "evaluation_error": None,
            }
        }

    def _error(
        self, state: ResearchState, runtime: Runtime, exc: Exception
    ) -> dict[str, Any]:
        skills = state.get("skills") or {}
        inserted = 0
        try:
            run_id = resolve_skill_run_id(state, runtime)
            inserted = sum(
                self._evaluator.store.save_evaluation(evaluation)
                for evaluation in self._evaluator.error_evaluations(
                    state, run_id, exc
                )
            )
        except Exception:
            inserted = 0
        return {
            "skills": {
                "evaluation_recorded": True,
                "evaluation_count": int(skills.get("evaluation_count", 0))
                + inserted,
                "evaluation_status": "error",
                "evaluation_error": f"{type(exc).__name__}: skill evaluation failed",
            }
        }

    @override
    def after_agent(
        self, state: ResearchState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if self._skip(state):
            return None
        try:
            run_id = resolve_skill_run_id(state, runtime)
            return self._persist(state, self._evaluator.evaluate(state, run_id))
        except Exception as exc:
            return self._error(state, runtime, exc)

    @override
    async def aafter_agent(
        self, state: ResearchState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if self._skip(state):
            return None
        try:
            run_id = resolve_skill_run_id(state, runtime)
            evaluations = await self._evaluator.aevaluate(state, run_id)
            return self._persist(state, evaluations)
        except Exception as exc:
            return self._error(state, runtime, exc)
