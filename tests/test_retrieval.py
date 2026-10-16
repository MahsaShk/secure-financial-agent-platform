import unittest

from secure_financial_agent.retrieve import TfidfRetriever
from secure_financial_agent.schemas import DocumentChunk


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = TfidfRetriever(
            [
                DocumentChunk("revenue", "report-a", "ACME revenue was 120 million in 2025."),
                DocumentChunk("assets", "report-b", "Beta total assets were 250 million in 2025."),
            ]
        )

    def test_ranks_relevant_chunk_first(self) -> None:
        results = self.retriever.retrieve("What was Beta total assets?", top_k=2)
        self.assertEqual(results[0].chunk.chunk_id, "assets")

    def test_rejects_invalid_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.retrieve("revenue", top_k=3)


if __name__ == "__main__":
    unittest.main()

