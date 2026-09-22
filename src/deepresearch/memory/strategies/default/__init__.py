"""Poirot-inspired default memory strategy."""

from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy
from deepresearch.memory.strategies.default.manager import DefaultMemoryManager

__all__ = ["CompositeForgetPolicy", "DefaultMemoryManager", "EbbinghausDecayPolicy"]
