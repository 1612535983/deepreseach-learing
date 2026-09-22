"""Poirot-inspired default memory strategy."""

from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy
from deepresearch.memory.strategies.default.manager import DefaultMemoryManager
from deepresearch.memory.strategies.default.store import MarkdownFileStore

__all__ = [
    "CompositeForgetPolicy",
    "DefaultMemoryManager",
    "EbbinghausDecayPolicy",
    "MarkdownFileStore",
]
