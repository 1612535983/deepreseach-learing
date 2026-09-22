"""Thread-safe lifecycle for the optional memory provider."""

from __future__ import annotations

import threading

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.provider import MemoryProvider
from deepresearch.memory.strategies.default.provider import build_default_provider
from deepresearch.memory.worker import MemoryWorker


_provider_lock = threading.Lock()
_memory_provider: MemoryProvider | None = None


def get_memory_provider(config: MemoryConfig) -> MemoryProvider | None:
    """Lazily construct the configured provider; an empty use value disables memory."""

    global _memory_provider
    if not config.enabled:
        return None
    if _memory_provider is not None:
        return _memory_provider
    with _provider_lock:
        if _memory_provider is not None:
            return _memory_provider
        if config.use != "default":
            raise ValueError(f"不支持的 MemoryProvider：{config.use}")
        _memory_provider = build_default_provider(config)
        return _memory_provider


def set_memory_provider(provider: MemoryProvider | None) -> None:
    """Inject or clear a provider, primarily for tests and application wiring."""

    global _memory_provider
    with _provider_lock:
        _memory_provider = provider


def reset_memory_provider() -> None:
    set_memory_provider(None)


_worker_lock = threading.Lock()
_memory_worker: MemoryWorker | None = None


def start_memory_worker(
    provider: MemoryProvider,
    llm: object,
    config: MemoryConfig,
) -> MemoryWorker:
    """Start and reuse the process-wide worker for the configured provider."""

    global _memory_worker
    if _memory_worker is not None:
        return _memory_worker
    with _worker_lock:
        if _memory_worker is not None:
            return _memory_worker
        worker = MemoryWorker(provider.manager(), llm, config)  # type: ignore[arg-type]
        worker.start()
        _memory_worker = worker
        return worker


def get_memory_worker() -> MemoryWorker | None:
    with _worker_lock:
        return _memory_worker


def shutdown_memory_worker(timeout: float = 5.0) -> None:
    """Drain and clear the global worker."""

    global _memory_worker
    with _worker_lock:
        worker = _memory_worker
        _memory_worker = None
    if worker is not None:
        worker.shutdown(timeout)


def reset_memory_runtime() -> None:
    """Reset provider and worker globals for isolated tests."""

    shutdown_memory_worker()
    reset_memory_provider()
