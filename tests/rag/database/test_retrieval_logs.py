import unittest

from app.rag.database.retrieval_logs import build_insert_log_sql, normalize_log_payload
from app.rag.retrieval_log import select_log_backend


class RetrievalLogBackendTests(unittest.TestCase):
    def test_retrieval_logs_use_postgres_backend(self):
        backend = select_log_backend()

        self.assertEqual(backend, "postgres")

    def test_normalize_log_payload_keeps_required_database_fields(self):
        payload = normalize_log_payload(
            {
                "corpus_id": "ancient-books-v1.0.0",
                "original_query": "headache",
                "rewritten_query": "headache wind",
                "retrieval_mode": "hybrid_parent",
                "degraded": False,
                "final_results": [{"citation_id": "E1"}],
            }
        )

        self.assertEqual(payload["corpus_id"], "ancient-books-v1.0.0")
        self.assertEqual(payload["chief_symptom"], None)
        self.assertEqual(payload["dense_hits"], [])
        self.assertEqual(payload["keyword_hits"], [])
        self.assertEqual(payload["final_results"], [{"citation_id": "E1"}])

    def test_insert_log_sql_targets_rag_retrieval_logs(self):
        sql = build_insert_log_sql().lower()

        self.assertIn("insert into rag_retrieval_logs", sql)
        self.assertIn("original_query", sql)
        self.assertIn("final_results", sql)


if __name__ == "__main__":
    unittest.main()
