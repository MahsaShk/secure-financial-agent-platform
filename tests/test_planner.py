import unittest
from pathlib import Path

from secure_financial_agent.ingest import load_chunks
from secure_financial_agent.planner import plan_calculation
from secure_financial_agent.retrieve import TfidfRetriever

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_chunks(DATA_DIR / "sample_documents.json")
        cls.retriever = TfidfRetriever(cls.chunks)

    def _plan(self, question: str):
        return plan_calculation(question, self.retriever.retrieve(question, top_k=4))

    def test_growth_amount_uses_new_minus_old(self) -> None:
        plan = self._plan("By how many millions did ACME revenue increase from 2024 to 2025?")
        self.assertEqual(plan.request.operation, "subtract")
        self.assertEqual(plan.request.operands, (120.0, 100.0))

    def test_percent_change_uses_old_then_new(self) -> None:
        plan = self._plan("What was ACME revenue percent change from 2024 to 2025?")
        self.assertEqual(plan.request.operation, "percent_change")
        self.assertEqual(plan.request.operands, (100.0, 120.0))

    def test_margin_is_operating_income_over_revenue(self) -> None:
        plan = self._plan("What was ACME operating margin in 2025 as a percentage of revenue?")
        self.assertEqual(plan.request.operation, "percentage")
        self.assertEqual(plan.request.operands, (24.0, 120.0))

    def test_part_of_whole_question_reads_whole_first(self) -> None:
        plan = self._plan("What percentage of Beta assets was represented by total liabilities in 2025?")
        self.assertEqual(plan.request.operation, "percentage")
        self.assertEqual(plan.request.operands, (150.0, 250.0))

    def test_lookup_question_is_not_planned(self) -> None:
        self.assertIsNone(self._plan("What was ACME revenue in 2025?"))

    def test_missing_second_year_is_refused(self) -> None:
        plan = self._plan("How much did ACME revenue increase in 2025?")
        self.assertIsNone(plan.request)
        self.assertEqual(plan.refusal_reason, "calculation_not_understood")

    def test_unknown_year_is_refused(self) -> None:
        plan = self._plan("How much did ACME revenue increase from 2023 to 2025?")
        self.assertIsNone(plan.request)
        self.assertEqual(plan.refusal_reason, "operands_not_found")


if __name__ == "__main__":
    unittest.main()
