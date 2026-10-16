from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import CalculationRequest, RetrievedChunk

_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(\$?)(\d+(?:,\d{3})*(?:\.\d+)?)")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_FROM_TO_PATTERN = re.compile(r"\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_PERCENT_CHANGE_PATTERN = re.compile(
    r"\bby how many percent\b|\b(?:percent|percentage)\s+(?:change|increase|decrease|growth)\b"
    r"|\bgrowth rate\b|%\s*change\b",
    re.IGNORECASE,
)
_PERCENTAGE_OF_PATTERN = re.compile(r"\b(?:percentage|percent)\s+of\b|%\s*of\b", re.IGNORECASE)
_AS_PERCENTAGE_OF_PATTERN = re.compile(r"\bas\s+(?:a\s+)?(?:percentage|percent|%)\s+of\b", re.IGNORECASE)
_CHANGE_PATTERN = re.compile(
    r"\bby how many\b|\bincrease[ds]?\b|\bdecrease[ds]?\b|\bdifference\b|\bgrew\b|\bgrow\b|\bchange[ds]?\b",
    re.IGNORECASE,
)

# Canonical metric name -> phrases that refer to it. Longer phrases are matched first.
_METRICS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue",),
    "operating income": ("operating income",),
    "operating cash flow": ("operating cash flow",),
    "capital expenditures": ("capital expenditures", "capex"),
    "total assets": ("total assets", "assets"),
    "total liabilities": ("total liabilities", "liabilities"),
    "cash and cash equivalents": ("cash and cash equivalents",),
    "long-term debt": ("long-term debt",),
    "shareholders' equity": ("shareholders' equity", "shareholders equity"),
    "operating margin": ("operating margin",),
}


@dataclass(frozen=True)
class CalculationPlan:
    """Planner output: either a calculation request to run, or a refusal_reason explaining why the question can't be planned."""

    request: CalculationRequest | None = None
    refusal_reason: str | None = None


@dataclass(frozen=True)
class OperationChoice:
    """The operation chosen from the question, before operands are looked up.

    An LLM-based chooser would return this same shape. refusal_reason is set
    when no single operation can be identified.
    """

    # For "percentage", metrics is (part, whole). Otherwise it holds the single metric being compared.
    operation: str | None = None
    metrics: tuple[str, ...] = ()
    refusal_reason: str | None = None


def _metric_pattern(name: str) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(alias) for alias in sorted(_METRICS[name], key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])(?:{alternatives})(?![A-Za-z])", re.IGNORECASE)


def _find_metrics(question: str) -> list[str]:
    """Return canonical metric names in the order they appear in the question."""
    spans: list[tuple[int, int, str]] = []
    aliases = sorted(
        ((alias, name) for name, names in _METRICS.items() for alias in names),
        key=lambda item: -len(item[0]),
    )
    for alias, name in aliases:
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", re.IGNORECASE)
        for match in pattern.finditer(question):
            if not any(match.start() < end and start < match.end() for start, end, _ in spans):
                spans.append((match.start(), match.end(), name))
    return [name for _, _, name in sorted(spans)]


def _entities(question: str) -> list[str]:
    # The first word of a question is capitalized by convention, so it is skipped.
    return re.findall(r"\b[A-Z][A-Za-z]+\b", question)[1:]


def _lookup(metric: str, year: str, entities: list[str], chunks: list[RetrievedChunk]) -> float | None:
    pattern = _metric_pattern(metric)
    values: set[float] = set()
    for retrieved in chunks:
        text = retrieved.chunk.text
        if not all(re.search(rf"\b{re.escape(entity)}\b", text, re.IGNORECASE) for entity in entities):
            continue
        chunk_years = set(_YEAR_PATTERN.findall(text))
        for sentence in _SENTENCE_PATTERN.split(text):
            if not pattern.search(sentence):
                continue
            sentence_years = set(_YEAR_PATTERN.findall(sentence))
            if sentence_years:
                if sentence_years != {year}:
                    continue
            elif year not in chunk_years:
                continue
            numbers = [
                match.group(2).replace(",", "")
                for match in _NUMBER_PATTERN.finditer(sentence)
                if match.group(1) or not _YEAR_PATTERN.fullmatch(match.group(2))
            ]
            if len(numbers) == 1:
                values.add(float(numbers[0]))
    if len(values) != 1:
        return None
    return values.pop()


def _years_in_order(question: str) -> tuple[str, str] | None:
    """Return (from_year, to_year) for a question that compares two years."""
    match = _FROM_TO_PATTERN.search(question)
    if match:
        return match.group(1), match.group(2)
    distinct = sorted(set(_YEAR_PATTERN.findall(question)))
    if len(distinct) == 2:
        return distinct[0], distinct[1]
    return None


def _plan_two_year(
    operation: str,
    metric: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> CalculationPlan:
    years = _years_in_order(question)
    if years is None:
        return CalculationPlan(refusal_reason="calculation_not_understood")
    old_year, new_year = years
    entities = _entities(question)
    old_value = _lookup(metric, old_year, entities, chunks)
    new_value = _lookup(metric, new_year, entities, chunks)
    if old_value is None or new_value is None:
        return CalculationPlan(refusal_reason="operands_not_found")
    # percent_change takes (old, new); subtract takes (new, old) so the result is new minus old.
    operands = (old_value, new_value) if operation == "percent_change" else (new_value, old_value)
    return CalculationPlan(request=CalculationRequest(operation=operation, operands=operands))


def _plan_ratio(
    part_metric: str,
    whole_metric: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> CalculationPlan:
    years = set(_YEAR_PATTERN.findall(question))
    if len(years) != 1:
        return CalculationPlan(refusal_reason="calculation_not_understood")
    year = years.pop()
    entities = _entities(question)
    part = _lookup(part_metric, year, entities, chunks)
    whole = _lookup(whole_metric, year, entities, chunks)
    if part is None or whole is None:
        return CalculationPlan(refusal_reason="operands_not_found")
    return CalculationPlan(request=CalculationRequest(operation="percentage", operands=(part, whole)))


def choose_operation(question: str) -> OperationChoice | None:
    """Decide which calculator operation the question asks for and which metrics it names.

    Returns None when the question does not ask for arithmetic. This is the step an LLM could replace.
    """
    metrics = _find_metrics(question)
    if not metrics:
        return None

    if "operating margin" in metrics:
        return OperationChoice(operation="percentage", metrics=("operating income", "revenue"))

    if _PERCENT_CHANGE_PATTERN.search(question):
        operation = "percent_change"
    elif _PERCENTAGE_OF_PATTERN.search(question):
        if len(metrics) != 2 or metrics[0] == metrics[1]:
            return OperationChoice(refusal_reason="calculation_not_understood")
        if _AS_PERCENTAGE_OF_PATTERN.search(question):
            return OperationChoice(operation="percentage", metrics=(metrics[0], metrics[1]))
        return OperationChoice(operation="percentage", metrics=(metrics[1], metrics[0]))
    elif _CHANGE_PATTERN.search(question):
        operation = "subtract"
    else:
        return None

    if len(set(metrics)) != 1:
        return OperationChoice(refusal_reason="calculation_not_understood")
    return OperationChoice(operation=operation, metrics=(metrics[0],))


def extract_operands(choice: OperationChoice, question: str, chunks: list[RetrievedChunk]) -> CalculationPlan:
    """Look up the operands for a chosen operation in the retrieved chunks. This step stays deterministic."""
    if choice.operation is None:
        return CalculationPlan(refusal_reason=choice.refusal_reason)
    if choice.operation == "percentage":
        part, whole = choice.metrics
        return _plan_ratio(part, whole, question, chunks)
    return _plan_two_year(choice.operation, choice.metrics[0], question, chunks)


def plan_calculation(question: str, chunks: list[RetrievedChunk]) -> CalculationPlan | None:
    """Choose a calculator operation from the question and take its operands from the chunks.

    Returns None when the question does not ask for arithmetic.
    """
    choice = choose_operation(question)
    if choice is None:
        return None
    return extract_operands(choice, question, chunks)
