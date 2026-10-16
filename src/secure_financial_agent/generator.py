from __future__ import annotations

import re

from .retrieve import tokenize
from .schemas import RetrievedChunk

_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])[-+]?\$?(\d+(?:,\d{3})*(?:\.\d+)?)%?")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "did",
    "for",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "was",
    "were",
    "what",
}


class ExtractiveGenerator:
    """Fallback answerer for non-arithmetic questions.

    Runs only when the planner returns None. It does not write sentences. It
    picks one number from the retrieved chunks and returns it as a string. The
    confidence is a float.
    """

    version = "deterministic-extractive-v1"

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> tuple[str | None, float]:
        question_terms = set(tokenize(question)) - _QUESTION_STOPWORDS
        question_numbers = {
            match.group(1).replace(",", "") for match in _NUMBER_PATTERN.finditer(question)
        }
        candidates: list[tuple[bool, int, float, str]] = []
        for retrieved in contexts:
            for sentence in _SENTENCE_PATTERN.split(retrieved.chunk.text):
                overlap = len(question_terms & set(tokenize(sentence)))
                for match in _NUMBER_PATTERN.finditer(sentence):
                    normalized_number = match.group(1).replace(",", "")
                    candidates.append(
                        (
                            normalized_number in question_numbers,
                            overlap,
                            retrieved.score,
                            normalized_number,
                        )
                    )
        if not candidates:
            return None, 0.0
        # Numbers not in the question win, then word overlap, then retrieval score.
        # The final tie-break is string order, so "100" sorts before "99".
        candidates.sort(key=lambda item: (item[0], -item[1], -item[2], item[3]))
        _, overlap, retrieval_score, answer = candidates[0]
        confidence = min(0.95, 0.45 + 0.1 * overlap + 0.2 * retrieval_score)
        return answer, confidence
