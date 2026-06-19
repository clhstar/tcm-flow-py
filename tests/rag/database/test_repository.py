import unittest

import numpy as np

from app.rag.database.repository import (
    build_dense_search_sql,
    prepare_vector,
    rows_to_parent_map,
)


class RepositoryTests(unittest.TestCase):
    def test_prepare_vector_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "1024"):
            prepare_vector(np.asarray([1.0, 2.0], dtype=np.float32))

    def test_prepare_vector_serializes_1024_floats(self):
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1.0

        serialized = prepare_vector(vector)

        self.assertTrue(serialized.startswith("[1.0,"))
        self.assertTrue(serialized.endswith("]"))

    def test_dense_search_sql_filters_corpus_symptom_and_role(self):
        sql = build_dense_search_sql()
        normalized = sql.lower()

        self.assertIn("rag_chunk_embeddings", normalized)
        self.assertIn("rag_chunks", normalized)
        self.assertIn("c.corpus_id = $1", normalized)
        self.assertIn("$3 = any(c.symptom_tags)", normalized)
        self.assertIn("c.evidence_role = any($4::text[])", normalized)
        self.assertIn("order by e.embedding <=> $2::vector", normalized)

    def test_rows_to_parent_map_indexes_by_parent_id(self):
        rows = [
            {"parent_id": "p1", "original_text": "a"},
            {"parent_id": "p2", "original_text": "b"},
        ]

        result = rows_to_parent_map(rows)

        self.assertEqual(result["p1"]["original_text"], "a")
        self.assertEqual(result["p2"]["original_text"], "b")


if __name__ == "__main__":
    unittest.main()
