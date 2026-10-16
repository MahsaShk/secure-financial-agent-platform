import unittest
from pathlib import Path

from secure_financial_agent.evaluate import evaluate_baselines, evaluate_security


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = Path(__file__).resolve().parents[1] / "data"

    def test_secure_agent_improves_answer_accuracy(self) -> None:
        results = {item.baseline: item for item in evaluate_baselines(self.data_dir)}
        self.assertGreater(
            results["secure_agent"].answer_accuracy,
            results["retrieval_only"].answer_accuracy,
        )
        self.assertEqual(results["secure_agent"].retrieval_recall_at_k, 1.0)
        self.assertEqual(results["secure_agent"].tool_selection_accuracy, 1.0)

    def test_security_suite(self) -> None:
        metrics = evaluate_security(self.data_dir)
        self.assertEqual(metrics["cases"], 20)
        self.assertEqual(metrics["refusal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

