from __future__ import annotations

from .guardrails import inspect_untrusted_text
from .schemas import CalculationRequest, PolicyDecision, RetrievedChunk
from .tools import Calculator


class AgentPolicy:
    """Applies the MVP allow/deny rules to user input, retrieved text, and tool calls.

    Each method returns a PolicyDecision with a reason code. The calculator is
    passed in so the tool allowlist comes from the same object that runs the tools.
    """

    version = "mvp-policy-v1"

    def __init__(self, calculator: Calculator) -> None:
        self._calculator = calculator

    def evaluate_input(self, question: str) -> PolicyDecision:
        return inspect_untrusted_text(question)

    def evaluate_retrieval(self, chunks: list[RetrievedChunk]) -> PolicyDecision:
        for retrieved in chunks:
            decision = inspect_untrusted_text(retrieved.chunk.text)
            if not decision.allowed:
                return PolicyDecision(
                    allowed=False,
                    reason_code="untrusted_retrieved_instruction",
                    matched_rule=decision.matched_rule,
                )
        return PolicyDecision(allowed=True, reason_code="allowed")

    def evaluate_calculation(self, request: CalculationRequest) -> PolicyDecision:
        if request.operation not in self._calculator.allowed_operations:
            return PolicyDecision(
                allowed=False,
                reason_code="tool_not_allowlisted",
                matched_rule=request.operation,
            )
        return PolicyDecision(allowed=True, reason_code="allowed")

