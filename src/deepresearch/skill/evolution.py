"""Human-governed generation, review, and promotion of skill candidates."""

from __future__ import annotations

import difflib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepresearch.evaluation.config import EvaluationConfig
from deepresearch.evaluation.provider import DecisionProvider
from deepresearch.evaluation.types import DecisionAnswer, DecisionQuestion
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.exceptions import SkillNotFoundError, SkillStoreError
from deepresearch.skill.parser import parse_skill_content
from deepresearch.skill.store import SkillStore
from deepresearch.skill.types import (
    ParsedSkill,
    SkillEvolutionExperiment,
    SkillLineage,
    SkillRecord,
)


CANDIDATE_REVIEW_QUESTIONS: dict[str, DecisionQuestion] = {
    "addresses_reason": DecisionQuestion(
        "noul",
        (
            "Does `candidate_skill` directly address the stated "
            "`evolution_reason` and observed runtime weaknesses?"
        ),
    ),
    "preserves_usefulness": DecisionQuestion(
        "noul",
        "Does `candidate_skill` preserve the useful behavior and scope of `baseline_skill`?",
    ),
    "instruction_clarity": DecisionQuestion(
        "score",
        "Rate how clear, actionable, and internally consistent `candidate_skill` is.",
        (
            "Confusing or contradictory.",
            "Partly actionable but materially ambiguous.",
            "Mostly clear and actionable.",
            "Precise, bounded, and directly actionable.",
        ),
    ),
    "regression_risk": DecisionQuestion(
        "noul",
        "Is the candidate likely to cause a material regression compared with the baseline?",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_text(message: object) -> str:
    content = message.content if isinstance(message, AIMessage) else str(message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _body_only(text: str) -> str:
    stripped = _strip_fence(text)
    if stripped.startswith("---"):
        parts = stripped.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\r\n").strip()
    return stripped


def _changed_lines(before: str, after: str) -> int:
    count = 0
    for line in difflib.ndiff(before.splitlines(), after.splitlines()):
        if line.startswith(("- ", "+ ")):
            count += 1
    return count


def _diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="baseline/SKILL.md",
            tofile="candidate/SKILL.md",
            n=3,
        )
    )


def _serialize_answer(answer: DecisionAnswer) -> dict[str, Any]:
    return {
        "type": answer.answer_type,
        "value": answer.value,
        "probabilities": dict(answer.probabilities),
        "confidence": answer.confidence,
        "legend": dict(answer.legend),
    }


class SkillEvolutionService:
    """Create inactive candidates, review them, and leave promotion to a human."""

    def __init__(
        self,
        store: SkillStore,
        skill_config: SkillConfig,
        *,
        model: BaseChatModel | None = None,
        provider: DecisionProvider | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.store = store
        self.skill_config = skill_config
        self.model = model
        self.provider = provider
        self.evaluation_config = evaluation_config or EvaluationConfig()

    def create_candidate(
        self, identifier: str, *, reason: str
    ) -> SkillEvolutionExperiment:
        """Use the research model to create one bounded, inactive child version."""

        if self.model is None:
            raise ValueError("生成 Skill 候选版本需要配置研究模型。")
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("Skill 演化原因不能为空。")
        baseline = self.store.get(identifier) or self.store.get_active(identifier)
        if baseline is None:
            raise SkillNotFoundError(f"找不到 Skill：{identifier}")
        if not baseline.is_active:
            raise ValueError("只能从当前激活的 Skill 版本生成候选。")

        baseline_body = self.store.read_body(baseline.skill_id)
        history = self._evaluation_summary(baseline.skill_id)
        prompt = (
            "你正在生成一个受控 Skill 候选版本。"
            "Skill 是过程指导，不是可执行代码。\n\n"
            f"Skill: {baseline.name}\n"
            f"演化原因: {normalized_reason}\n"
            f"最近运行评估:\n{history}\n\n"
            f"当前正文:\n{baseline_body}\n\n"
            "约束：只修复与演化原因直接相关的一个方面；"
            "保留原有适用范围；"
            f"修改总行数不得超过 {self.skill_config.evolution_max_changed_lines}；"
            "不要输出 YAML frontmatter 或代码围栏，"
            "只返回修改后的 Markdown 正文。"
        )
        response = self.model.invoke([HumanMessage(content=prompt)])
        candidate_body = _body_only(_message_text(response))
        if not candidate_body:
            raise ValueError("模型返回的 Skill 候选正文为空。")
        changed = _changed_lines(baseline_body, candidate_body)
        if changed == 0:
            raise ValueError("模型没有生成任何 Skill 修改。")
        if changed > self.skill_config.evolution_max_changed_lines:
            raise ValueError(
                "Skill 候选修改超过预算："
                f"{changed} 行 > {self.skill_config.evolution_max_changed_lines} 行。"
            )

        next_version = max(
            (item.version for item in self.store.get_versions(baseline.name)),
            default=baseline.version,
        ) + 1
        raw_content = self._candidate_content(
            baseline, candidate_body, version=next_version
        )
        parsed = parse_skill_content(
            raw_content,
            source_path=Path(".deepresearch")
            / "skills"
            / "generated"
            / baseline.name
            / "SKILL.md",
            origin="DERIVED",
            max_file_chars=self.skill_config.max_file_chars,
        )
        candidate_record = replace(
            parsed.record,
            is_active=False,
            lineage=SkillLineage(
                parent_skill_ids=(baseline.skill_id,),
                generation=baseline.lineage.generation + 1,
                origin="DERIVED",
                created_by="skill-evolution",
            ),
        )
        candidate = self.store.install(
            ParsedSkill(candidate_record, parsed.raw_content, parsed.body),
            activate=False,
        )
        experiment = SkillEvolutionExperiment(
            experiment_id=f"evo_{uuid4().hex[:16]}",
            skill_name=baseline.name,
            baseline_skill_id=baseline.skill_id,
            candidate_skill_id=candidate.skill_id,
            reason=normalized_reason,
            mutation_diff=_diff(baseline_body, candidate_body),
            changed_lines=changed,
            created_at=_utc_now(),
            notes=("inactive_candidate", "human_promotion_required"),
        )
        return self.store.create_experiment(experiment)

    def review_candidate(self, candidate_skill_id: str) -> SkillEvolutionExperiment:
        """Fail closed: deterministic checks then one bounded provider review."""

        experiment = self.store.get_experiment_for_candidate(candidate_skill_id)
        if experiment is None:
            raise SkillStoreError("找不到候选版本对应的演化实验。")
        if experiment.status != "candidate":
            raise SkillStoreError("候选版本已经完成评审。")
        baseline = self.store.get(experiment.baseline_skill_id)
        candidate = self.store.get(experiment.candidate_skill_id)
        if baseline is None or candidate is None:
            raise SkillStoreError("演化实验引用的 Skill 版本不存在。")

        rule_notes = self._rule_notes(baseline, candidate, experiment)
        if rule_notes:
            rejected = replace(
                experiment,
                status="rejected",
                rule_passed=False,
                recommendation="reject",
                notes=tuple([*experiment.notes, *rule_notes]),
                reviewed_at=_utc_now(),
            )
            return self.store.save_experiment_review(rejected)
        if self.provider is None:
            raise ValueError("评审 Skill 候选版本需要配置概率评估 Provider。")

        payload = self._review_payload(baseline, candidate, experiment)
        input_chars = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        response = self.provider.evaluate(payload, CANDIDATE_REVIEW_QUESTIONS)
        fix = self._noul(response.answers["addresses_reason"], "addresses_reason")
        preserve = self._noul(
            response.answers["preserves_usefulness"], "preserves_usefulness"
        )
        regression = self._noul(
            response.answers["regression_risk"], "regression_risk"
        )
        clarity_answer = response.answers["instruction_clarity"]
        if clarity_answer.answer_type != "score":
            raise ValueError("instruction_clarity 必须是 score answer。")
        clarity = max(0.0, min(3.0, float(clarity_answer.value))) / 3.0
        score = round(
            fix * 0.40
            + preserve * 0.30
            + clarity * 0.20
            + (1.0 - regression) * 0.10,
            6,
        )
        notes = list(experiment.notes)
        if fix < 0.65:
            notes.append("reason_not_addressed")
        if preserve < 0.75:
            notes.append("baseline_usefulness_not_preserved")
        if regression > 0.35:
            notes.append("regression_risk_high")
        if (
            clarity_answer.confidence is not None
            and clarity_answer.confidence < self.evaluation_config.confidence_floor
        ):
            notes.append("low_evaluator_confidence")
        approve = (
            score >= self.skill_config.promotion_threshold
            and not any(
                item
                in {
                    "reason_not_addressed",
                    "baseline_usefulness_not_preserved",
                    "regression_risk_high",
                    "low_evaluator_confidence",
                }
                for item in notes
            )
        )
        reviewed = replace(
            experiment,
            status="reviewed" if approve else "rejected",
            rule_passed=True,
            recommendation="approve" if approve else "reject",
            provider=response.provider,
            model=response.model,
            score=score,
            answers={
                name: _serialize_answer(answer)
                for name, answer in response.answers.items()
            },
            input_chars=input_chars,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            notes=tuple(dict.fromkeys(notes)),
            reviewed_at=_utc_now(),
        )
        return self.store.save_experiment_review(reviewed)

    def promote(self, candidate_skill_id: str) -> SkillEvolutionExperiment:
        """Activate a positively reviewed candidate through explicit human action."""

        return self.store.promote_candidate(candidate_skill_id)

    def _evaluation_summary(self, skill_id: str) -> str:
        evaluations = self.store.list_evaluations(skill_id, limit=10)
        completed = [item for item in evaluations if item.status == "completed"]
        if not completed:
            return (
                "(暂无 Jev Skill 运行评估；"
                "仅依据人工给出的演化原因生成候选)"
            )

        def average(field: str) -> float:
            values = [
                float(value)
                for item in completed
                if isinstance((value := getattr(item, field)), (int, float))
            ]
            return sum(values) / len(values) if values else 0.0

        causes: dict[str, int] = {}
        for item in completed:
            cause = item.failure_cause or "unclear"
            causes[cause] = causes.get(cause, 0) + 1
        return "\n".join(
            [
                f"样本数: {len(completed)}",
                f"平均适用概率: {average('applicable'):.1%}",
                f"平均遵循概率: {average('followed'):.1%}",
                f"平均帮助概率: {average('helpful'):.1%}",
                f"平均指令缺陷概率: {average('instruction_defect'):.1%}",
                "失败归因: "
                + ", ".join(f"{name}={count}" for name, count in sorted(causes.items())),
            ]
        )

    @staticmethod
    def _candidate_content(
        baseline: SkillRecord, body: str, *, version: int
    ) -> str:
        metadata: dict[str, Any] = {
            "name": baseline.name,
            "description": baseline.description,
            "version": version,
        }
        if baseline.tags:
            metadata["tags"] = list(baseline.tags)
        if baseline.allowed_tools:
            metadata["tools"] = list(baseline.allowed_tools)
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        return f"---\n{frontmatter}\n---\n{body.strip()}\n"

    def _rule_notes(
        self,
        baseline: SkillRecord,
        candidate: SkillRecord,
        experiment: SkillEvolutionExperiment,
    ) -> list[str]:
        notes: list[str] = []
        if candidate.name != baseline.name:
            notes.append("skill_name_changed")
        if candidate.allowed_tools != baseline.allowed_tools:
            notes.append("allowed_tools_changed")
        if candidate.tags != baseline.tags:
            notes.append("tags_changed")
        if baseline.skill_id not in candidate.lineage.parent_skill_ids:
            notes.append("baseline_not_parent")
        if experiment.changed_lines > self.skill_config.evolution_max_changed_lines:
            notes.append("change_budget_exceeded")
        if self.store.read_body(candidate.skill_id) == self.store.read_body(
            baseline.skill_id
        ):
            notes.append("candidate_unchanged")
        return notes

    def _review_payload(
        self,
        baseline: SkillRecord,
        candidate: SkillRecord,
        experiment: SkillEvolutionExperiment,
    ) -> dict[str, Any]:
        body_limit = self.skill_config.evaluation_body_chars
        payload = {
            "evolution_reason": experiment.reason,
            "baseline_skill": {
                "name": baseline.name,
                "description": baseline.description,
                "body": self.store.read_body(baseline.skill_id)[:body_limit],
            },
            "candidate_skill": {
                "name": candidate.name,
                "description": candidate.description,
                "body": self.store.read_body(candidate.skill_id)[:body_limit],
            },
            "mutation_diff": experiment.mutation_diff[
                : self.evaluation_config.max_evidence_chars * 4
            ],
            "runtime_evidence": self._evaluation_summary(baseline.skill_id),
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if len(serialized) > self.evaluation_config.max_payload_chars:
            raise ValueError("Skill 候选评审输入超过 max_payload_chars。")
        return payload

    @staticmethod
    def _noul(answer: DecisionAnswer, name: str) -> float:
        if answer.answer_type != "noul" or not isinstance(answer.value, float):
            raise ValueError(f"{name} 必须是 Jev noul answer。")
        return answer.value
