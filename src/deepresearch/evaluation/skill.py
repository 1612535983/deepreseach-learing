"""Bounded, provider-neutral evaluation of injected skill versions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.provider import DecisionProvider
from deepresearch.evaluation.types import DecisionAnswer, DecisionQuestion, DecisionResponse
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.store import SkillStore
from deepresearch.skill.types import SkillEvaluation, SkillRef
from deepresearch.state import ResearchState


SKILL_EVALUATION_SCHEMA = "skill-run-evaluation-v1"
FAILURE_CAUSES = {
    "skill_instruction": "The skill instructions were materially wrong or incomplete.",
    "model_execution": "The agent did not correctly follow otherwise useful instructions.",
    "tool_or_source": "Tool failure or unavailable/weak sources dominated the outcome.",
    "task_succeeded": "The task succeeded without a material failure.",
    "unclear": "The supplied evidence is insufficient to attribute a cause.",
}


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 2:
        return text[:limit]
    remaining = limit - len(marker)
    head = (remaining * 2) // 3
    return f"{text[:head]}{marker}{text[-(remaining - head):]}"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _included_refs(state: ResearchState) -> list[SkillRef]:
    skills = state.get("skills") or {}
    dropped = {
        str(item.get("skill_id") or "")
        for item in skills.get("dropped", [])
        if isinstance(item, Mapping)
    }
    return [
        item
        for item in skills.get("selected", [])
        if isinstance(item, dict)
        and str(item.get("skill_id") or "")
        and str(item.get("skill_id")) not in dropped
    ]


def _task_completed(state: ResearchState) -> bool:
    if str(state.get("final_report") or "").strip():
        return True
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    return bool(
        isinstance(last, AIMessage)
        and not last.tool_calls
        and str(last.content or "").strip()
    )


@dataclass(frozen=True)
class SkillEvaluationPayload:
    """Bounded run projection and dynamic question mapping for selected skills."""

    state: dict[str, Any]
    questions: dict[str, DecisionQuestion]
    skill_keys: dict[str, str]
    signature: str
    input_chars: int
    notes: tuple[str, ...] = ()


def build_skill_evaluation_payload(
    state: ResearchState,
    store: SkillStore,
    evaluation_config: EvaluationConfig,
    skill_config: SkillConfig,
) -> SkillEvaluationPayload:
    """Project only evidence needed to judge skill use; exclude raw messages/memory."""

    refs = _included_refs(state)
    if not refs:
        raise ValueError("没有实际注入的 Skill，无法执行 Skill 评估。")

    tool_calls: list[str] = []
    for message in state.get("messages", []):
        if isinstance(message, AIMessage):
            tool_calls.extend(
                str(call.get("name") or "")
                for call in message.tool_calls
                if str(call.get("name") or "")
            )
    plan = state.get("plan")
    raw_steps = plan.get("steps", []) if isinstance(plan, Mapping) else []
    steps = [
        {
            "title": _clip(step.get("title"), 200),
            "status": _clip(step.get("status"), 40),
        }
        for step in raw_steps
        if isinstance(step, Mapping)
    ]
    report_eval = _mapping(_mapping(state.get("evaluation")).get("report"))
    skill_rows: list[dict[str, Any]] = []
    skill_keys: dict[str, str] = {}
    notes: list[str] = []
    for index, ref in enumerate(refs, 1):
        skill_id = str(ref["skill_id"])
        record = store.get(skill_id)
        if record is None:
            notes.append(f"missing_skill:{skill_id}")
            continue
        key = f"skill_{index}"
        skill_keys[key] = skill_id
        skill_rows.append(
            {
                "key": key,
                "skill_id": skill_id,
                "name": record.name,
                "description": _clip(record.description, 500),
                "allowed_tools": list(record.allowed_tools),
                "body": "",
            }
        )
    if not skill_rows:
        raise ValueError("已选择的 Skill 版本均无法从仓库读取。")
    if len(skill_rows) > 1:
        notes.append("multi_skill_attribution")

    base: dict[str, Any] = {
        "research_question": _clip(state.get("research_question"), 2_000),
        "final_report": "",
        "execution": {
            "task_completed": _task_completed(state),
            "tool_calls": tool_calls[:80],
            "plan": steps[:10],
            "search_attempts": len(state.get("search_records", [])),
            "page_reads": len(state.get("page_records", [])),
            "source_count": len(state.get("sources", [])),
            "evidence_count": len(state.get("observations", [])),
            "report_evaluation": {
                "status": report_eval.get("status"),
                "composite_score": report_eval.get("composite_score"),
                "recommended_action": report_eval.get("recommended_action"),
            },
        },
        "skills": skill_rows,
    }
    overhead = len(_stable_json(base))
    if overhead >= evaluation_config.max_payload_chars:
        raise ValueError("Skill 评估基础元数据超过 max_payload_chars。")
    available = evaluation_config.max_payload_chars - overhead
    report_budget = min(evaluation_config.max_report_chars, available // 2)
    report = str(state.get("final_report") or "")
    bounded_report = _clip(report, report_budget)
    base["final_report"] = bounded_report
    if bounded_report != report.strip():
        notes.append("report_truncated")

    remaining = evaluation_config.max_payload_chars - len(_stable_json(base))
    per_skill = min(
        skill_config.evaluation_body_chars,
        max(0, remaining // len(skill_rows)),
    )
    for row in skill_rows:
        body = store.read_body(str(row["skill_id"]))
        bounded_body = _clip(body, per_skill)
        row["body"] = bounded_body
        if bounded_body != body.strip():
            notes.append("skill_body_truncated")
    serialized = _stable_json(base)
    if len(serialized) > evaluation_config.max_payload_chars:
        raise ValueError("Skill 评估输入超过 max_payload_chars。")

    questions: dict[str, DecisionQuestion] = {}
    for key in skill_keys:
        prefix = f"For `{key}` in `skills`, based only on the supplied state, "
        questions.update(
            {
                f"{key}_applicable": DecisionQuestion(
                    "noul",
                    prefix + "was this skill applicable to the research task?",
                ),
                f"{key}_followed": DecisionQuestion(
                    "noul",
                    prefix + "did the agent's execution follow its material guidance?",
                ),
                f"{key}_helpful": DecisionQuestion(
                    "noul",
                    prefix + "did the skill likely make the result materially better?",
                ),
                f"{key}_instruction_defect": DecisionQuestion(
                    "noul",
                    prefix
                    + (
                        "is a defect in the skill instruction itself a likely "
                        "cause of any weakness?"
                    ),
                ),
                f"{key}_failure_cause": DecisionQuestion(
                    "choice",
                    prefix + "which category best explains the observed outcome?",
                    FAILURE_CAUSES,
                ),
            }
        )
    signature = _hash_text(
        f"{SKILL_EVALUATION_SCHEMA}\x00{_stable_json(base)}"
    )
    return SkillEvaluationPayload(
        state=base,
        questions=questions,
        skill_keys=skill_keys,
        signature=signature,
        input_chars=len(serialized),
        notes=tuple(dict.fromkeys(notes)),
    )


@dataclass(frozen=True)
class SkillRunEvaluator:
    """Evaluate every injected skill in a single typed provider request."""

    provider: DecisionProvider
    evaluation_config: EvaluationConfig
    skill_config: SkillConfig
    store: SkillStore

    def evaluate(self, state: ResearchState, run_id: str) -> list[SkillEvaluation]:
        payload = build_skill_evaluation_payload(
            state, self.store, self.evaluation_config, self.skill_config
        )
        response = self.provider.evaluate(payload.state, payload.questions)
        return self._compose(state, run_id, payload, response)

    async def aevaluate(
        self, state: ResearchState, run_id: str
    ) -> list[SkillEvaluation]:
        payload = build_skill_evaluation_payload(
            state, self.store, self.evaluation_config, self.skill_config
        )
        response = await self.provider.aevaluate(payload.state, payload.questions)
        return self._compose(state, run_id, payload, response)

    def error_evaluations(
        self, state: ResearchState, run_id: str, exc: Exception
    ) -> list[SkillEvaluation]:
        """Create auditable, sanitized failure rows without exposing provider data."""

        payload = build_skill_evaluation_payload(
            state, self.store, self.evaluation_config, self.skill_config
        )
        created_at = datetime.now(timezone.utc).isoformat()
        return [
            SkillEvaluation(
                evaluation_id="seval_"
                + _hash_text(f"{run_id}\x00{skill_id}\x00{payload.signature}")[:20],
                run_id=run_id,
                skill_id=skill_id,
                signature=payload.signature,
                status="error",
                provider=getattr(self.provider, "name", None),
                input_chars=payload.input_chars,
                task_completed=_task_completed(state),
                notes=payload.notes,
                error=f"{type(exc).__name__}: skill evaluation failed",
                created_at=created_at,
            )
            for skill_id in payload.skill_keys.values()
        ]

    def _compose(
        self,
        state: ResearchState,
        run_id: str,
        payload: SkillEvaluationPayload,
        response: DecisionResponse,
    ) -> list[SkillEvaluation]:
        report = _mapping(_mapping(state.get("evaluation")).get("report"))
        report_score = report.get("composite_score")
        normalized_report_score = (
            float(report_score)
            if isinstance(report_score, (int, float)) and not isinstance(report_score, bool)
            else None
        )
        created_at = datetime.now(timezone.utc).isoformat()
        results: list[SkillEvaluation] = []
        for key, skill_id in payload.skill_keys.items():
            answers = {
                name.removeprefix(f"{key}_"): self._serialize(answer)
                for name, answer in response.answers.items()
                if name.startswith(f"{key}_")
            }
            results.append(
                SkillEvaluation(
                    evaluation_id="seval_"
                    + _hash_text(f"{run_id}\x00{skill_id}\x00{payload.signature}")[:20],
                    run_id=run_id,
                    skill_id=skill_id,
                    signature=payload.signature,
                    status="completed",
                    provider=response.provider,
                    model=response.model,
                    applicable=self._noul(response, f"{key}_applicable"),
                    followed=self._noul(response, f"{key}_followed"),
                    helpful=self._noul(response, f"{key}_helpful"),
                    instruction_defect=self._noul(
                        response, f"{key}_instruction_defect"
                    ),
                    failure_cause=str(
                        response.answers[f"{key}_failure_cause"].value
                    ),
                    report_score=normalized_report_score,
                    task_completed=_task_completed(state),
                    answers=answers,
                    input_chars=payload.input_chars,
                    latency_ms=response.latency_ms,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=response.usage.cost_usd,
                    notes=tuple([*payload.notes, "shared_batch_usage"]),
                    created_at=created_at,
                )
            )
        return results

    @staticmethod
    def _noul(response: DecisionResponse, name: str) -> float:
        answer = response.answers[name]
        if answer.answer_type != "noul" or not isinstance(answer.value, float):
            raise ValueError(f"{name} 必须是 Jev noul answer。")
        return answer.value

    @staticmethod
    def _serialize(answer: DecisionAnswer) -> dict[str, Any]:
        return {
            "type": answer.answer_type,
            "value": answer.value,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
            "legend": dict(answer.legend),
        }
