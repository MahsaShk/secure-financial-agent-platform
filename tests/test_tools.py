import math
import unittest

from secure_financial_agent.schemas import CalculationRequest
from secure_financial_agent.tools import Calculator


class CalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = Calculator()

    def test_percent_change(self) -> None:
        result = self.calculator.execute(CalculationRequest("percent_change", (100.0, 120.0)))
        self.assertEqual(result.value, 20.0)

    def test_percentage(self) -> None:
        result = self.calculator.execute(CalculationRequest("percentage", (24.0, 120.0)))
        self.assertEqual(result.value, 20.0)

    def test_rejects_division_by_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "division by zero"):
            self.calculator.execute(CalculationRequest("divide", (1.0, 0.0)))

    def test_rejects_unknown_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported operation"):
            self.calculator.execute(CalculationRequest("eval", (1.0,)))

    def test_rejects_non_finite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self.calculator.execute(CalculationRequest("add", (1.0, math.inf)))


if __name__ == "__main__":
    unittest.main()

