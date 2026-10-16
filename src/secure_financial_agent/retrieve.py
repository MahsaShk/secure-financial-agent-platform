from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import DocumentChunk, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class TfidfRetriever:
    """Sparse TF-IDF retriever over a fixed set of chunks.

    Vectors are built once at construction, with log-scaled term frequency and
    smoothed inverse document frequency. Ranking uses cosine similarity, and ties
    are broken by chunk_id so results are deterministic. This is a bag-of-words
    model: it matches exact terms and misses synonyms.
    """

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("retriever requires at least one chunk")
        self._chunks = tuple(chunks)
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for term_frequency in self._term_frequencies:
            document_frequency.update(term_frequency.keys())
        count = len(chunks)
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._weighted_vector(frequencies) for frequencies in self._term_frequencies]

    def _weighted_vector(self, frequencies: Counter[str]) -> dict[str, float]:
        return {
            term: (1 + math.log(frequency)) * self._idf.get(term, 1.0)
            for term, frequency in frequencies.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        common = left.keys() & right.keys()
        numerator = sum(left[term] * right[term] for term in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the top_k chunks by cosine similarity. Ranks start at 1."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= len(self._chunks):
            raise ValueError(f"top_k must be between 1 and {len(self._chunks)}")
        query_vector = self._weighted_vector(Counter(tokenize(query)))
        scored = [
            (self._cosine(query_vector, vector), chunk)
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

