"""Configuration values for the long-term memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass
import os
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

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("DEEPRESEARCH_MEMORY_TOKEN_BUDGET 必须大于 0。")
        if self.top_k < 1:
            raise ValueError("DEEPRESEARCH_MEMORY_TOP_K 必须大于 0。")
        if not 0.0 <= self.min_strength <= 1.0:
            raise ValueError("DEEPRESEARCH_MEMORY_MIN_STRENGTH 必须在 0 到 1 之间。")
        if not 0.0 <= self.forget_strength_threshold <= 1.0:
            raise ValueError("DEEPRESEARCH_MEMORY_FORGET_THRESHOLD 必须在 0 到 1 之间。")
        if self.forget_ttl_hours <= 0:
            raise ValueError("DEEPRESEARCH_MEMORY_TTL_HOURS 必须大于 0。")
        if self.consolidate_threshold < 2:
            raise ValueError("DEEPRESEARCH_MEMORY_CONSOLIDATE_THRESHOLD 不能小于 2。")
        if self.worker_queue_size < 1:
            raise ValueError("DEEPRESEARCH_MEMORY_WORKER_QUEUE_SIZE 必须大于 0。")

    @property
    def enabled(self) -> bool:
        return bool(self.use.strip())

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Read optional memory settings without requiring a model API key."""

        return cls(
            use=os.getenv("DEEPRESEARCH_MEMORY_USE", "").strip(),
            storage_path=(
                os.getenv("DEEPRESEARCH_MEMORY_STORAGE_PATH", "").strip()
                or DEFAULT_MEMORY_PATH
            ),
            namespace=(
                os.getenv("DEEPRESEARCH_MEMORY_NAMESPACE", "default").strip()
                or "default"
            ),
            enable_recall=_env_bool("DEEPRESEARCH_MEMORY_ENABLE_RECALL", True),
            enable_extract=_env_bool("DEEPRESEARCH_MEMORY_ENABLE_EXTRACT", False),
            token_budget=_env_int("DEEPRESEARCH_MEMORY_TOKEN_BUDGET", 2_000),
            top_k=_env_int("DEEPRESEARCH_MEMORY_TOP_K", 5),
            min_strength=_env_float("DEEPRESEARCH_MEMORY_MIN_STRENGTH", 0.0),
            forget_strength_threshold=_env_float(
                "DEEPRESEARCH_MEMORY_FORGET_THRESHOLD", 0.1
            ),
            forget_ttl_hours=_env_float("DEEPRESEARCH_MEMORY_TTL_HOURS", 720.0),
            consolidate_threshold=_env_int(
                "DEEPRESEARCH_MEMORY_CONSOLIDATE_THRESHOLD", 10
            ),
            worker_queue_size=_env_int(
                "DEEPRESEARCH_MEMORY_WORKER_QUEUE_SIZE", 32
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
