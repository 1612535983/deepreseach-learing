"""Domain-specific errors raised by the skill subsystem."""

from __future__ import annotations


class SkillError(RuntimeError):
    """Base class for recoverable skill subsystem failures."""


class SkillValidationError(SkillError, ValueError):
    """A ``SKILL.md`` bundle does not satisfy the public contract."""


class SkillNotFoundError(SkillError, LookupError):
    """A requested skill or immutable version cannot be resolved."""


class SkillStoreError(SkillError):
    """The persistent skill registry could not complete an operation."""
