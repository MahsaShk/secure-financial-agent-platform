from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    """A passage from a source document. chunk_id must be unique across the corpus."""

    chunk_id: str
    source_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score and 1-based rank."""

    chunk: DocumentChunk
    score: float
    rank: int


@dataclass(frozen=True)
class CalculationRequest:
    """An operation name and its operands, before the calculator validates them.

    from_dict checks the types of untrusted input before constructing the request.
    """

    operation: str
    operands: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CalculationRequest:
        operation = value.get("operation")
        operands = value.get("operands")
        if not isinstance(operation, str):
            raise ValueError("calculation.operation must be a string")
        if not isinstance(operands, list):
            raise ValueError("calculation.operands must be a list")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in operands):
            raise ValueError("calculation.operands must contain only numbers")
        return cls(operation=operation, operands=tuple(float(item) for item in operands))


@dataclass(frozen=True)
class CalculationResult:
    """A completed calculation: the operation, the operands used, and the finite result."""

    operation: str
    operands: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class Evidence:
    """A retrieved chunk as reported to the caller: its ID, source, score, and an excerpt of up to 280 characters."""

    chunk_id: str
    source_id: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class AgentRequest:
    """A user question, an optional explicit calculation, and how many chunks to retrieve (1 to 10)."""

    question: str
    calculation: CalculationRequest | None = None
    top_k: int = 3

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= self.top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")


@dataclass(frozen=True)
class AgentResponse:
    """Structured output of one agent call.

    A successful answer has at least one citation. A refusal has answer=None and
    a refusal_reason. confidence must be between 0 and 1.
    """

    trace_id: str
    answer: str | None
    calculation: CalculationResult | None = None
    evidence: tuple[Evidence, ...] = ()
    citations: tuple[str, ...] = ()
    confidence: float = 0.0
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.answer is not None and not self.citations:
            raise ValueError("successful grounded answers require at least one citation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of a policy check: whether it passed, a reason code, and the rule that matched, if any."""

    allowed: bool
    reason_code: str
    matched_rule: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    """One record per agent call: outcome, reason code, tool calls, and versions.

    Stores IDs and reason codes, not the question or chunk text.
    """

    trace_id: str
    timestamp_utc: str
    outcome: str
    reason_code: str
    tool_calls: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    model_version: str = "deterministic-extractive-v1"
    prompt_version: str = "no-llm-prompt-v1"
    policy_version: str = "mvp-policy-v1"
    application_version: str = "0.1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

