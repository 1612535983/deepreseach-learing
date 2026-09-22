"""Small dependency-free lexical retrieval primitives shared by subsystems."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping


_WORD_OR_CJK = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize English words and Chinese unigrams/bigrams."""

    tokens: list[str] = []
    for match in _WORD_OR_CJK.findall(text.lower()):
        if match and "\u3400" <= match[0] <= "\u9fff":
            characters = list(match)
            tokens.extend(characters)
            tokens.extend(
                "".join(characters[index : index + 2])
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(match)
    return tokens


def bm25_scores(
    query: str,
    documents: Mapping[str, str],
    *,
    tokenize: Callable[[str], list[str]] = tokenize_text,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Return normalized BM25 scores in ``[0, 1)`` for keyed documents."""

    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return {key: 0.0 for key in documents}
    tokenized = {key: tokenize(text) for key, text in documents.items()}
    average_length = sum(len(tokens) for tokens in tokenized.values()) / max(
        1, len(tokenized)
    )
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized.values():
        document_frequency.update(set(tokens))
    document_count = len(tokenized)
    query_counts = Counter(query_tokens)
    scores: dict[str, float] = {}
    for key, tokens in tokenized.items():
        term_counts = Counter(tokens)
        raw_score = 0.0
        for token, query_frequency in query_counts.items():
            term_frequency = term_counts.get(token, 0)
            if term_frequency == 0:
                continue
            inverse_frequency = math.log(
                (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
                + 1.0
            )
            normalization = term_frequency + k1 * (
                1.0 - b + b * len(tokens) / max(1.0, average_length)
            )
            raw_score += (
                inverse_frequency
                * term_frequency
                * (k1 + 1.0)
                / normalization
                * query_frequency
            )
        scores[key] = raw_score / (1.0 + raw_score) if raw_score > 0 else 0.0
    return scores
