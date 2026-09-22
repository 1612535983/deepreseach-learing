"""Application-lifetime facade for discovery, storage, and selection."""

from __future__ import annotations

from pathlib import Path

from deepresearch.skill.config import SkillConfig
from deepresearch.skill.parser import discover_skills
from deepresearch.skill.selector import SkillSelector
from deepresearch.skill.store import SQLiteSkillStore
from deepresearch.skill.types import SkillDiscoveryError, SkillRecord, SkillSelection


BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin_skills"


class SkillManager:
    """Own the process-wide catalog and its deterministic selector."""

    def __init__(self, config: SkillConfig) -> None:
        self.config = config
        self.store = SQLiteSkillStore(config.db_path, config.storage_path)
        self.selector = SkillSelector(
            self.store,
            max_skills=config.max_skills,
            min_relevance=config.min_relevance,
        )
        self.discovery_errors: list[SkillDiscoveryError] = []

    def load_startup(self) -> list[SkillRecord]:
        """Discover builtins first so same-name user skills may override them."""

        installed: list[SkillRecord] = []
        sources: list[tuple[tuple[str | Path, ...], str]] = []
        if self.config.include_builtin and BUILTIN_SKILLS_DIR.exists():
            sources.append(((BUILTIN_SKILLS_DIR,), "BUILTIN"))
        sources.append((self.config.skill_dirs, "IMPORTED"))
        for directories, origin in sources:
            discovery = discover_skills(
                directories,
                origin=origin,  # type: ignore[arg-type]
                max_file_chars=self.config.max_file_chars,
            )
            self.discovery_errors.extend(discovery.errors)
            for parsed in discovery.skills:
                installed.append(self.store.install(parsed))
        return installed

    def select(
        self,
        question: str,
        *,
        overrides: tuple[str, ...] = (),
        available_tools: tuple[str, ...] = (),
    ) -> list[SkillSelection]:
        return self.selector.select_for_task(
            question,
            overrides=overrides,
            available_tools=available_tools,
        )

    def list_skills(self) -> list[SkillRecord]:
        return self.store.list_active()

    def close(self) -> None:
        self.store.close()
