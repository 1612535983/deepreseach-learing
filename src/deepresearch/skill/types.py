"""Immutable value objects and checkpoint-safe skill runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


SkillOrigin = Literal["BUILTIN", "IMPORTED", "CAPTURED", "DERIVED"]


@dataclass(frozen=True)
class SkillLineage:
    """Version ancestry for one immutable skill record."""

    parent_skill_ids: tuple[str, ...] = ()
    generation: int = 0
    origin: SkillOrigin = "IMPORTED"
    created_by: str | None = None


@dataclass(frozen=True)
class SkillRecord:
    """Persistent metadata for one immutable ``SKILL.md`` version.

    The body is deliberately absent. ``object_path`` points at the immutable
    content-addressed copy used for checkpoint replay and audit.
    """

    skill_id: str
    name: str
    description: str
    source_path: str
    object_path: str
    content_hash: str
    version: int = 1
    tags: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    enabled: bool = True
    is_active: bool = True
    lineage: SkillLineage = field(default_factory=SkillLineage)
    created_at: str = ""
    last_updated: str = ""
    total_selections: int = 0
    total_injections: int = 0
    total_aligned_tool_calls: int = 0
    total_completed_runs: int = 0
    total_failed_runs: int = 0

    @property
    def injection_rate(self) -> float:
        return (
            self.total_injections / self.total_selections
            if self.total_selections
            else 0.0
        )

    @property
    def completion_rate(self) -> float:
        return (
            self.total_completed_runs / self.total_injections
            if self.total_injections
            else 0.0
        )


@dataclass(frozen=True)
class ParsedSkill:
    """Validated source bundle before it is installed into the store."""

    record: SkillRecord
    raw_content: str
    body: str


@dataclass(frozen=True)
class SkillDiscoveryError:
    """One bundle that discovery skipped without aborting the whole catalog."""

    path: str
    error: str


@dataclass(frozen=True)
class SkillDiscoveryResult:
    """Validated bundles and isolated parse failures from one directory scan."""

    skills: tuple[ParsedSkill, ...] = ()
    errors: tuple[SkillDiscoveryError, ...] = ()


@dataclass(frozen=True)
class SkillSelection:
    """One selector decision with enough provenance for inspection."""

    record: SkillRecord
    score: float
    reason: str
    forced: bool = False


@dataclass(frozen=True)
class SkillMetrics:
    """Cross-run counters for one immutable skill version."""

    skill_id: str
    selections: int
    injections: int
    aligned_tool_calls: int
    completed_runs: int
    failed_runs: int
    injection_rate: float
    completion_rate: float


class SkillRef(TypedDict, total=False):
    """Compact, serializable reference saved in ``ResearchState``."""

    skill_id: str
    name: str
    content_hash: str
    score: float
    reason: str
    forced: bool
    allowed_tools: list[str]
    drop_reason: str


class SkillRuntimeState(TypedDict, total=False):
    """Checkpoint-safe per-run view of selection and injection activity."""

    query_hash: str | None
    catalog_hash: str | None
    selected: list[SkillRef]
    dropped: list[SkillRef]
    selection_count: int
    injection_count: int
    injected_tokens: int
    render_signature: str | None
    aligned_tool_calls: int
    completed_recorded: bool
    last_error: str | None
