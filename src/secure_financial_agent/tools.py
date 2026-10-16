from __future__ import annotations

import math
from collections.abc import Callable

from .schemas import CalculationRequest, CalculationResult


def _require_count(operands: tuple[float, ...], expected: int) -> None:
    if len(operands) != expected:
        raise ValueError(f"operation requires exactly {expected} operands")


def _add(operands: tuple[float, ...]) -> float:
    if len(operands) < 2:
        raise ValueError("add requires at least two operands")
    return sum(operands)


def _subtract(operands: tuple[float, ...]) -> float:
    _require_count(operands, 2)
    return operands[0] - operands[1]


def _multiply(operands: tuple[float, ...]) -> float:
    if len(operands) < 2:
        raise ValueError("multiply requires at least two operands")
    return math.prod(operands)


def _divide(operands: tuple[float, ...]) -> float:
    _require_count(operands, 2)
    if operands[1] == 0:
        raise ValueError("division by zero is not allowed")
    return operands[0] / operands[1]


def _percentage(operands: tuple[float, ...]) -> float:
    return _divide(operands) * 100


def _percent_change(operands: tuple[float, ...]) -> float:
    _require_count(operands, 2)
    old_value, new_value = operands
    if old_value == 0:
        raise ValueError("percent change from zero is undefined")
    return ((new_value - old_value) / abs(old_value)) * 100


_OPERATIONS: dict[str, Callable[[tuple[float, ...]], float]] = {
    "add": _add,
    "subtract": _subtract,
    "multiply": _multiply,
    "divide": _divide,
    "percentage": _percentage,
    "percent_change": _percent_change,
}


class Calculator:
    """Runs allowlisted arithmetic on validated numeric operands."""

    @property
    def allowed_operations(self) -> tuple[str, ...]:
        return tuple(sorted(_OPERATIONS))

    def execute(self, request: CalculationRequest) -> CalculationResult:
        """Run an allowlisted operation.

        Raises ValueError for an unknown operation, empty or non-finite operands,
        or a non-finite result.
        """
        operation = _OPERATIONS.get(request.operation)
        if operation is None:
            allowed = ", ".join(self.allowed_operations)
            raise ValueError(f"unsupported operation {request.operation!r}; allowed: {allowed}")
        if not request.operands or any(not math.isfinite(value) for value in request.operands):
            raise ValueError("operands must be finite numbers")
        value = operation(request.operands)
        if not math.isfinite(value):
            raise ValueError("calculation produced a non-finite result")
        return CalculationResult(
            operation=request.operation,
            operands=request.operands,
            value=value,
        )

