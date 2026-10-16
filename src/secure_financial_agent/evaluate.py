from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import SecureFinancialAgent
from .generator import ExtractiveGenerator
from .ingest import load_chunks
from .retrieve import TfidfRetriever
from .schemas import AgentRequest


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    normalized = value.strip().lower().replace(",", "").replace("$", "").replace("%", "")
    try:
        number = float(normalized)
    except ValueError:
        return " ".join(normalized.split())
    if math.isfinite(number):
        return format(number, ".10g")
    return normalized


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics for one baseline. Latency is per example, in milliseconds."""

    baseline: str
    examples: int
    answer_accuracy: float
    retrieval_recall_at_k: float
    retrieval_mrr: float
    citation_precision: float
    tool_selection_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return value


def evaluate_baselines(data_dir: str | Path) -> list[EvaluationMetrics]:
    """Run the no-retrieval, retrieval-only, and secure-agent baselines on the fixture in data_dir."""
    root = Path(data_dir)
    chunks = load_chunks(root / "sample_documents.json")
    examples = _load_json_list(root / "sample_eval.json")
    retriever = TfidfRetriever(chunks)
    source_by_chunk_id = {chunk.chunk_id: chunk.source_id for chunk in chunks}
    generator = ExtractiveGenerator()
    agent = SecureFinancialAgent(retriever)

    results: list[EvaluationMetrics] = []
    for baseline in ("no_retrieval", "retrieval_only", "secure_agent"):
        correct = 0
        recalled = 0
        reciprocal_ranks: list[float] = []
        citation_scores: list[float] = []
        tool_correct = 0
        latencies: list[float] = []

        for example in examples:
            question = str(example["question"])
            expected = str(example["answer"])
            relevant = set(str(item) for item in example["relevant_chunk_ids"])
            relevant_sources = {source_by_chunk_id[chunk_id] for chunk_id in relevant}
            expected_operation = example.get("expected_operation")

            start = time.perf_counter()
            retrieved = [] if baseline == "no_retrieval" else retriever.retrieve(question, top_k=3)
            if baseline == "no_retrieval":
                answer = None
                citations: set[str] = set()
                used_operation: str | None = None
            elif baseline == "retrieval_only":
                answer, _ = generator.answer(question, retrieved)
                citations = {item.chunk.source_id for item in retrieved}
                used_operation = None
            else:
                response = agent.answer(AgentRequest(question=question, top_k=3))
                answer = response.answer
                citations = set(response.citations)
                used_operation = response.calculation.operation if response.calculation is not None else None
            latencies.append((time.perf_counter() - start) * 1000)

            correct += normalize_answer(answer) == normalize_answer(expected)
            ranked_ids = [item.chunk.chunk_id for item in retrieved]
            matching_ranks = [rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant]
            recalled += bool(matching_ranks)
            reciprocal_ranks.append(1 / min(matching_ranks) if matching_ranks else 0.0)
            citation_scores.append(
                len(citations & relevant_sources) / len(citations) if citations else 0.0
            )
            tool_correct += used_operation == expected_operation

        count = len(examples)
        results.append(
            EvaluationMetrics(
                baseline=baseline,
                examples=count,
                answer_accuracy=correct / count,
                retrieval_recall_at_k=recalled / count,
                retrieval_mrr=statistics.fmean(reciprocal_ranks),
                citation_precision=statistics.fmean(citation_scores),
                tool_selection_accuracy=tool_correct / count,
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=_percentile(latencies, 0.95),
            )
        )
    return results


def evaluate_security(data_dir: str | Path) -> dict[str, float | int]:
    """Run the labeled security cases in data_dir and return refusal accuracy and latency."""
    root = Path(data_dir)
    cases = _load_json_list(root / "security_cases.json")
    chunks = load_chunks(root / "sample_documents.json")
    agent = SecureFinancialAgent(TfidfRetriever(chunks))
    correct = 0
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        response = agent.answer(AgentRequest(question=str(case["input"]), top_k=3))
        latencies.append((time.perf_counter() - start) * 1000)
        expected_refusal = bool(case["should_refuse"])
        correct += (response.refusal_reason is not None) == expected_refusal
    return {
        "cases": len(cases),
        "refusal_accuracy": correct / len(cases),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return asdict(metrics)
