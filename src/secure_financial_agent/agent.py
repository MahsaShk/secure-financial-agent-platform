from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from .audit import AuditSink, InMemoryAuditSink
from .generator import ExtractiveGenerator
from .planner import plan_calculation
from .policy import AgentPolicy
from .retrieve import TfidfRetriever
from .schemas import (
    AgentRequest,
    AgentResponse,
    AuditEvent,
    Evidence,
    PolicyDecision,
    RetrievedChunk,
)
from .tools import Calculator


def _format_number(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f")


class SecureFinancialAgent:
    """Answers one question through guardrails, retrieval, planning, tools, and generation.

    Every call returns an AgentResponse and writes one audit event. Refusals are
    returned as structured responses rather than raised as exceptions.
    """

    def __init__(
        self,
        retriever: TfidfRetriever,
        *,
        calculator: Calculator | None = None,
        generator: ExtractiveGenerator | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._retriever = retriever
        self._calculator = calculator or Calculator()
        self._generator = generator or ExtractiveGenerator()
        self._policy = AgentPolicy(self._calculator)
        self._audit_sink = audit_sink or InMemoryAuditSink()

    def _evidence(self, chunks: list[RetrievedChunk]) -> tuple[Evidence, ...]:
        return tuple(
            Evidence(
                chunk_id=item.chunk.chunk_id,
                source_id=item.chunk.source_id,
                excerpt=item.chunk.text[:280],
                score=round(item.score, 6),
            )
            for item in chunks
        )

    def _write_audit(
        self,
        trace_id: str,
        outcome: str,
        decision: PolicyDecision,
        chunks: list[RetrievedChunk],
        tool_calls: tuple[str, ...] = (),
    ) -> None:
        self._audit_sink.write(
            AuditEvent(
                trace_id=trace_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                outcome=outcome,
                reason_code=decision.reason_code,
                tool_calls=tool_calls,
                retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                model_version=self._generator.version,
                policy_version=self._policy.version,
                metadata={"matched_rule": decision.matched_rule},
            )
        )

    def _refuse(
        self,
        trace_id: str,
        decision: PolicyDecision,
        chunks: list[RetrievedChunk] | None = None,
    ) -> AgentResponse:
        retrieved = chunks or []
        self._write_audit(trace_id, "refused", decision, retrieved)
        return AgentResponse(
            trace_id=trace_id,
            answer=None,
            evidence=self._evidence(retrieved),
            confidence=1.0,
            refusal_reason=decision.reason_code,
        )

    def answer(self, request: AgentRequest) -> AgentResponse:
        """Return a structured response for one question.

        An explicit request.calculation overrides the planner. Otherwise the
        planner decides from the question whether arithmetic is needed.
        """
        trace_id = str(uuid4())
        input_decision = self._policy.evaluate_input(request.question)
        if not input_decision.allowed:
            return self._refuse(trace_id, input_decision)

        chunks = self._retriever.retrieve(request.question, request.top_k)
        retrieval_decision = self._policy.evaluate_retrieval(chunks)
        if not retrieval_decision.allowed:
            return self._refuse(trace_id, retrieval_decision, chunks)

        citations = (chunks[0].chunk.source_id,)
        calculation_request = request.calculation
        if calculation_request is None:
            plan = plan_calculation(request.question, chunks)
            if plan is not None and plan.refusal_reason is not None:
                decision = PolicyDecision(allowed=False, reason_code=plan.refusal_reason)
                return self._refuse(trace_id, decision, chunks)
            calculation_request = plan.request if plan is not None else None
        if calculation_request is not None:
            tool_decision = self._policy.evaluate_calculation(calculation_request)
            if not tool_decision.allowed:
                return self._refuse(trace_id, tool_decision, chunks)
            try:
                calculation = self._calculator.execute(calculation_request)
            except ValueError as error:
                decision = PolicyDecision(
                    allowed=False,
                    reason_code="invalid_calculation",
                    matched_rule=str(error),
                )
                return self._refuse(trace_id, decision, chunks)
            response = AgentResponse(
                trace_id=trace_id,
                answer=_format_number(calculation.value),
                calculation=calculation,
                evidence=self._evidence(chunks),
                citations=citations,
                confidence=0.95,
            )
            self._write_audit(
                trace_id,
                "answered",
                PolicyDecision(allowed=True, reason_code="allowed"),
                chunks,
                tool_calls=(f"calculator:{calculation.operation}",),
            )
            return response

        answer, confidence = self._generator.answer(request.question, chunks)
        if answer is None:
            decision = PolicyDecision(allowed=False, reason_code="insufficient_evidence")
            return self._refuse(trace_id, decision, chunks)
        response = AgentResponse(
            trace_id=trace_id,
            answer=answer,
            evidence=self._evidence(chunks),
            citations=citations,
            confidence=confidence,
        )
        self._write_audit(
            trace_id,
            "answered",
            PolicyDecision(allowed=True, reason_code="allowed"),
            chunks,
        )
        return response
