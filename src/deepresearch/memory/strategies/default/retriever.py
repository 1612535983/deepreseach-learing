"""Namespace-aware BM25 retrieval with lazy strength decay."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable

from deepresearch.memory.schema import MemoryTrace
from deepresearch.memory.store import MemoryStore
from deepresearch.memory.strategies.default.constants import BM25_B, BM25_K1, SIMILARITY_WEIGHT
from deepresearch.memory.strategies.default.decay import EbbinghausDecayPolicy
from deepresearch.memory.strategies.default.forget import CompositeForgetPolicy
from deepresearch.memory.types import MemoryFilter, MemoryQuery, RetrievalResult
from deepresearch.lexical import tokenize_text


def tokenize_memory_text(text: str) -> list[str]:
    """Tokenize English words and Chinese unigrams/bigrams without dependencies."""

    return tokenize_text(text)


class HybridRetriever:
    """Default retriever; vector and graph adapters remain optional extensions."""

    def __init__(
        self,
        store: MemoryStore,
        decay_policy: EbbinghausDecayPolicy,
        forget_policy: CompositeForgetPolicy,
        *,
        tokenize: Callable[[str], list[str]] = tokenize_memory_text,
        now_provider: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._decay_policy = decay_policy
        self._forget_policy = forget_policy
        self._tokenize = tokenize
        self._now_provider = now_provider
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self._trace_lengths: dict[str, int] = {}
        for trace in self._store.list_all():
            if not trace.metadata.get("forgotten"):
                self._index_trace(trace)

    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        if not query.text.strip() or query.top_k <= 0:
            return []
        now = self._now_provider()
        query_tokens = self._tokenize(query.text)
        if not query_tokens:
            return []

        candidates = self._store.list_by_filter(
            MemoryFilter(namespace=query.namespace, type=query.type_filter)
        )
        average_length = sum(self._trace_lengths.values()) / max(1, len(self._trace_lengths))
        ranked: list[RetrievalResult] = []
        for trace in candidates:
            if self._forget_policy.should_forget(trace, now):
                continue
            raw_similarity = self._bm25_score(query_tokens, trace.id, average_length)
            if raw_similarity <= 0:
                continue
            similarity = raw_similarity / (1.0 + raw_similarity)
            strength = self._decay_policy.compute_strength(trace, now)
            if strength < query.min_strength:
                continue
            ranked.append(
                RetrievalResult.compute_score(
                    trace,
                    similarity,
                    strength,
                    similarity_weight=SIMILARITY_WEIGHT,
                )
            )

        ranked.sort(key=lambda result: (result.score, result.trace.importance), reverse=True)
        results: list[RetrievalResult] = []
        for result in ranked[: query.top_k]:
            strengthened = result.trace.with_strength(result.strength, now)
            self._store.update(strengthened)
            results.append(
                RetrievalResult(
                    trace=strengthened,
                    similarity=result.similarity,
                    strength=result.strength,
                    score=result.score,
                )
            )
        return results

    def _bm25_score(
        self,
        query_tokens: list[str],
        trace_id: str,
        average_length: float,
    ) -> float:
        trace_length = self._trace_lengths.get(trace_id, 0)
        if trace_length == 0:
            return 0.0
        score = 0.0
        for token, query_frequency in Counter(query_tokens).items():
            postings = self._inverted_index.get(token, {})
            term_frequency = postings.get(trace_id, 0)
            if term_frequency == 0:
                continue
            document_frequency = len(postings)
            document_count = max(1, len(self._trace_lengths))
            inverse_frequency = math.log(
                (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
                + 1.0
            )
            normalization = (
                term_frequency
                + BM25_K1
                * (1.0 - BM25_B + BM25_B * trace_length / max(1.0, average_length))
            )
            score += (
                inverse_frequency
                * term_frequency
                * (BM25_K1 + 1.0)
                / normalization
                * query_frequency
            )
        return score

    def on_trace_added(self, trace: MemoryTrace) -> None:
        if not trace.metadata.get("forgotten"):
            self._index_trace(trace)

    def on_trace_updated(self, trace: MemoryTrace) -> None:
        self._remove_trace(trace.id)
        if not trace.metadata.get("forgotten"):
            self._index_trace(trace)

    def on_trace_removed(self, trace_id: str) -> None:
        self._remove_trace(trace_id)

    def _index_trace(self, trace: MemoryTrace) -> None:
        tokens = self._tokenize(trace.content)
        self._trace_lengths[trace.id] = len(tokens)
        for token, count in Counter(tokens).items():
            self._inverted_index[token][trace.id] = count

    def _remove_trace(self, trace_id: str) -> None:
        self._trace_lengths.pop(trace_id, None)
        for token in list(self._inverted_index):
            self._inverted_index[token].pop(trace_id, None)
            if not self._inverted_index[token]:
                del self._inverted_index[token]
