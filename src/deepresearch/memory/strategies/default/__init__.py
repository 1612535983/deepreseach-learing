"""Poirot-inspired default memory strategy."""

from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy

__all__ = ["CompositeForgetPolicy", "EbbinghausDecayPolicy"]
