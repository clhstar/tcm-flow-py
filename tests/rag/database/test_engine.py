import unittest

import numpy as np

from app.rag.database.engine import DatabaseRetrievalEngine


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0] + [0.0] * 1023 for _ in texts], dtype=np.float32)


class FakeReranker:
    def score(self, pairs):
        return [1.0 - index / 10 for index, _ in enumerate(pairs)]


class FakeRepository:
    async def dense_search(self, corpus_id, vector, chief_symptom, top_k):
        return [
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "matched_child": "headache and wind",
                "distance": 0.1,
                "symptom_tags": ["headache"],
                "evidence_role": "syndrome_pattern",
            }
        ]

    async def load_parents(self, parent_ids):
        return {
            "p1": {
                "parent_id": "p1",
                "source_type": "ancient_book",
                "book_title": "Jing Yue Quan Shu",
                "source_file": "637-jing-yue-quan-shu.txt",
                "volume": "volume-one",
                "chapter": "headache",
                "section": "pattern",
                "symptom_tags": ["headache"],
                "evidence_role": "syndrome_pattern",
                "original_text": "headache and wind.",
            }
        }


class FakeKeywordIndex:
    async def search(
        self,
        *,
        rewritten_query,
        corpus_id,
        chief_symptom,
        evidence_roles,
        top_k,
    ):
        return [
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "matched_child": "headache and wind",
                "score": 2.0,
            }
        ]


class FailingKeywordIndex:
    async def search(self, **kwargs):
        raise RuntimeError("es unavailable")


def engine(keyword_index):
    return DatabaseRetrievalEngine(
        corpus_id="ancient-books-v1.0.0",
        repository=FakeRepository(),
        keyword_index=keyword_index,
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        settings={
            "dense_top_k": 20,
            "bm25_top_k": 20,
            "rrf_k": 60,
            "reranker_candidate_k": 40,
            "final_top_k": 5,
        },
    )


class DatabaseEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_retrieval_assigns_citations_and_sources(self):
        result = await engine(FakeKeywordIndex()).retrieve(
            "headache wind",
            chief_symptom="headache",
            mode="hybrid",
            top_k=5,
        )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["degraded"])
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(result["results"][0]["citation_id"], "E1")
        self.assertEqual(result["results"][0]["retrieval_sources"], ["bm25", "dense"])

    async def test_es_failure_degrades_to_vector(self):
        result = await engine(FailingKeywordIndex()).retrieve(
            "headache wind",
            chief_symptom="headache",
            mode="hybrid",
            top_k=5,
        )

        self.assertTrue(result["degraded"])
        self.assertEqual(result["retrieval_mode"], "vector")
        self.assertIn("es unavailable", result["degraded_reason"])


if __name__ == "__main__":
    unittest.main()
