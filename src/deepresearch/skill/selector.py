"""Deterministic skill selection with explicit overrides and BM25 ranking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from deepresearch.lexical import bm25_scores
from deepresearch.skill.store import SkillStore
from deepresearch.skill.types import SkillRecord, SkillSelection


class SkillSelector:
    """Select relevant active skills without spending another model call."""

    def __init__(
        self,
        store: SkillStore,
        *,
        max_skills: int = 3,
        min_relevance: float = 0.05,
    ) -> None:
        if max_skills < 1:
            raise ValueError("max_skills 必须大于 0。")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance 必须在 0 到 1 之间。")
        self._store = store
        self._max_skills = max_skills
        self._min_relevance = min_relevance

    def catalog_hash(self) -> str:
        """Fingerprint active metadata so checkpoint selection can detect changes."""

        payload = [
            {
                "id": record.skill_id,
                "hash": record.content_hash,
                "enabled": record.enabled,
                "tools": list(record.allowed_tools),
            }
            for record in self._store.list_active()
        ]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def select_for_task(
        self,
        task_description: str,
        *,
        overrides: Iterable[str] = (),
        available_tools: Iterable[str] = (),
    ) -> list[SkillSelection]:
        """Return forced selections followed by stable relevance-ranked matches."""

        available = {name.strip() for name in available_tools if name.strip()}
        active = self._store.list_active()
        active_by_name = {record.name: record for record in active}
        active_by_id = {record.skill_id: record for record in active}

        forced: list[SkillSelection] = []
        seen: set[str] = set()
        for value in overrides:
            normalized = value.strip()
            record = active_by_id.get(normalized) or active_by_name.get(normalized)
            if record is None or not record.enabled or record.skill_id in seen:
                continue
            forced.append(
                SkillSelection(
                    record=record,
                    score=1.0,
                    reason="user_override",
                    forced=True,
                )
            )
            seen.add(record.skill_id)

        candidates = [
            record
            for record in active
            if record.enabled
            and record.skill_id not in seen
            and self._tools_are_usable(record, available)
        ]
        documents = {
            record.skill_id: " ".join(
                [record.name, record.description, *record.tags]
            )
            for record in candidates
        }
        scores = bm25_scores(task_description, documents)
        ranked = sorted(
            candidates,
            key=lambda record: (-scores.get(record.skill_id, 0.0), record.name),
        )
        automatic = [
            SkillSelection(
                record=record,
                score=scores[record.skill_id],
                reason="bm25_metadata_match",
            )
            for record in ranked
            if scores.get(record.skill_id, 0.0) >= self._min_relevance
        ]
        remaining = max(0, self._max_skills - len(forced))
        return [*forced, *automatic[:remaining]]

    @staticmethod
    def _tools_are_usable(record: SkillRecord, available: set[str]) -> bool:
        """Keep guidance skills, or tool skills with at least one usable capability."""

        if not record.allowed_tools:
            return True
        return bool(set(record.allowed_tools).intersection(available))
