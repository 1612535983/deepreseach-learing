"""Stable defaults shared by the default memory strategy."""

from __future__ import annotations

from deepresearch.memory.schema import MemoryType


DECAY_PARAMS: dict[MemoryType, dict[str, float]] = {
    MemoryType.EPISODIC: {"base_strength": 0.7, "decay_rate": 0.1},
    MemoryType.SEMANTIC: {"base_strength": 0.8, "decay_rate": 0.02},
    MemoryType.PROCEDURAL: {"base_strength": 0.9, "decay_rate": 0.005},
}

SIMILARITY_WEIGHT = 0.7
BM25_K1 = 1.5
BM25_B = 0.75
MAX_ASSOCIATIONS_PER_TRACE = 20
MIN_TRACES_TO_CONSOLIDATE = 2
MAX_TRACES_TO_CONSOLIDATE = 10
CONSOLIDATED_IMPORTANCE_BOOST = 0.1
