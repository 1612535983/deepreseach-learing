"""Strict environment-backed configuration for the skill subsystem."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_SKILL_DB_PATH = Path(".deepresearch/skills.db")
DEFAULT_SKILL_STORAGE_PATH = Path(".deepresearch/skills/objects")
DEFAULT_SKILL_DIR = Path("skills")


@dataclass(frozen=True)
class SkillConfig:
    """Startup, selection, and context-budget controls for skills."""

    use: str = ""
    db_path: str | Path = DEFAULT_SKILL_DB_PATH
    storage_path: str | Path = DEFAULT_SKILL_STORAGE_PATH
    skill_dirs: tuple[str | Path, ...] = (DEFAULT_SKILL_DIR,)
    include_builtin: bool = True
    max_skills: int = 3
    token_budget: int = 2_500
    min_relevance: float = 0.05
    max_file_chars: int = 50_000
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        if self.max_skills < 1:
            raise ValueError("DEEPRESEARCH_SKILL_MAX_SKILLS 必须大于 0。")
        if self.token_budget < 1:
            raise ValueError("DEEPRESEARCH_SKILL_TOKEN_BUDGET 必须大于 0。")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("DEEPRESEARCH_SKILL_MIN_RELEVANCE 必须在 0 到 1 之间。")
        if self.max_file_chars < 1:
            raise ValueError("DEEPRESEARCH_SKILL_MAX_FILE_CHARS 必须大于 0。")

    @property
    def enabled(self) -> bool:
        return bool(self.use.strip())

    @classmethod
    def from_env(cls) -> "SkillConfig":
        """Load optional skill settings without requiring an API key."""

        raw_dirs = os.getenv("DEEPRESEARCH_SKILL_DIRS", "").strip()
        skill_dirs: tuple[str | Path, ...] = (
            tuple(part.strip() for part in raw_dirs.split(",") if part.strip())
            if raw_dirs
            else (DEFAULT_SKILL_DIR,)
        )
        return cls(
            use=os.getenv("DEEPRESEARCH_SKILL_USE", "").strip(),
            db_path=(
                os.getenv("DEEPRESEARCH_SKILL_DB_PATH", "").strip()
                or DEFAULT_SKILL_DB_PATH
            ),
            storage_path=(
                os.getenv("DEEPRESEARCH_SKILL_STORAGE_PATH", "").strip()
                or DEFAULT_SKILL_STORAGE_PATH
            ),
            skill_dirs=skill_dirs,
            include_builtin=_env_bool(
                "DEEPRESEARCH_SKILL_INCLUDE_BUILTIN", True
            ),
            max_skills=_env_int("DEEPRESEARCH_SKILL_MAX_SKILLS", 3),
            token_budget=_env_int("DEEPRESEARCH_SKILL_TOKEN_BUDGET", 2_500),
            min_relevance=_env_float(
                "DEEPRESEARCH_SKILL_MIN_RELEVANCE", 0.05
            ),
            max_file_chars=_env_int(
                "DEEPRESEARCH_SKILL_MAX_FILE_CHARS", 50_000
            ),
            enable_metrics=_env_bool(
                "DEEPRESEARCH_SKILL_ENABLE_METRICS", True
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true 或 false。")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数。") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
