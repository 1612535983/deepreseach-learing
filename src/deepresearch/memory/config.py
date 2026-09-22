"""Configuration values for the Poirot-inspired memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MEMORY_PATH = Path(".deepresearch/memory")


@dataclass(frozen=True)
class MemoryConfig:
    """Startup and runtime controls for long-term memory."""

    use: str = ""
    storage_path: str | Path = DEFAULT_MEMORY_PATH
    namespace: str = "default"
    enable_recall: bool = True
    enable_extract: bool = False
    token_budget: int = 2_000
    top_k: int = 5
    min_strength: float = 0.0
    forget_strength_threshold: float = 0.1
    forget_ttl_hours: float = 720.0
    consolidate_threshold: int = 10
    worker_queue_size: int = 32

    @property
    def enabled(self) -> bool:
        return bool(self.use.strip())
