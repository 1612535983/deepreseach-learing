from dataclasses import replace

from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryType
from deepresearch.memory.strategies.default.provider import build_default_provider
from deepresearch.memory.strategies.default.retriever import tokenize_memory_text
from deepresearch.memory.types import MemoryQuery


def test_tokenizer_supports_chinese_bigrams_and_english_words() -> None:
    tokens = tokenize_memory_text("LangGraph 用户偏好设计方案")

    assert "langgraph" in tokens
    assert "用户" in tokens
    assert "方案" in tokens


def test_retriever_ranks_relevant_chinese_memory_and_strengthens_access(tmp_path) -> None:
    provider = build_default_provider(
        MemoryConfig(use="default", storage_path=tmp_path),
        now_provider=lambda: 1_000.0,
    )
    relevant = provider.manager().encode(
        "用户偏好先阅读设计方案再修改代码",
        MemoryType.SEMANTIC,
    )
    provider.manager().encode("今天北京天气晴朗", MemoryType.EPISODIC)

    results = provider.retriever().retrieve(
        MemoryQuery(text="请先给我设计方案", namespace="default")
    )

    assert results[0].trace.id == relevant.id
    assert results[0].similarity > 0
    assert provider.store().get(relevant.id).access_count == 1  # type: ignore[union-attr]


def test_retriever_isolates_namespace_and_forgotten_traces(tmp_path) -> None:
    provider = build_default_provider(
        MemoryConfig(use="default", storage_path=tmp_path),
        now_provider=lambda: 1_000.0,
    )
    visible = provider.manager().encode(
        "项目使用 SQLite",
        MemoryType.SEMANTIC,
        namespace="a",
    )
    provider.manager().encode("项目使用 SQLite", MemoryType.SEMANTIC, namespace="b")
    provider.store().update(
        replace(visible, metadata={"forgotten": True})
    )

    assert provider.retriever().retrieve(
        MemoryQuery(text="SQLite", namespace="a")
    ) == []
    namespace_b = provider.retriever().retrieve(
        MemoryQuery(text="SQLite", namespace="b")
    )
    assert len(namespace_b) == 1
    assert namespace_b[0].trace.namespace == "b"


def test_manager_updates_incremental_retrieval_index(tmp_path) -> None:
    provider = build_default_provider(
        MemoryConfig(use="default", storage_path=tmp_path),
        now_provider=lambda: 1_000.0,
    )
    trace = provider.manager().encode("独特关键词火星基地", MemoryType.SEMANTIC)

    assert provider.retriever().retrieve(
        MemoryQuery(text="火星基地", namespace="default")
    )[0].trace.id == trace.id

    provider.manager().reconsolidate(trace.id, "独特关键词月球基地")
    assert provider.retriever().retrieve(
        MemoryQuery(text="火星", namespace="default")
    ) == []
    assert provider.retriever().retrieve(
        MemoryQuery(text="月球", namespace="default")
    )[0].trace.id == trace.id
