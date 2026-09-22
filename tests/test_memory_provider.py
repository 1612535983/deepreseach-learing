from deepresearch.memory.bootstrap import (
    get_memory_provider,
    reset_memory_provider,
    set_memory_provider,
)
from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.strategies.default.provider import DefaultMemoryProvider


def test_bootstrap_returns_none_when_memory_is_disabled(tmp_path) -> None:
    reset_memory_provider()

    assert get_memory_provider(MemoryConfig(storage_path=tmp_path)) is None


def test_bootstrap_lazily_reuses_default_provider(tmp_path) -> None:
    reset_memory_provider()
    config = MemoryConfig(use="default", storage_path=tmp_path)
    try:
        first = get_memory_provider(config)
        second = get_memory_provider(config)

        assert isinstance(first, DefaultMemoryProvider)
        assert first is second
        assert first.store() is not None
        assert first.retriever() is not None
        assert first.manager() is not None
    finally:
        set_memory_provider(None)
