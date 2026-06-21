import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.routers import rag


class RagRouterTests(unittest.TestCase):
    def test_query_rag_returns_production_retrieval_payload(self):
        app = FastAPI()
        app.include_router(rag.router)
        payload = {
            "status": "ok",
            "retrieval_mode": "hybrid_parent",
            "degraded": False,
            "degraded_reason": None,
            "original_query": "头痛恶风",
            "rewritten_query": "头痛恶风 头痛 头风",
            "chief_symptom": "头痛",
            "allowed_terms": ["头痛"],
            "results": [
                {
                    "citation_id": "E1",
                    "parent_id": "p1",
                    "chunk_id": "c1",
                    "book_title": "景岳全书",
                    "volume": "卷之一",
                    "chapter": "头痛",
                    "section": "头痛论",
                    "content": "头痛恶风，遇冷加重。",
                    "matched_child": "头痛恶风",
                    "evidence_role": "syndrome_pattern",
                    "symptom_tags": ["头痛"],
                    "score": 0.9,
                    "retrieval_sources": ["bm25", "dense"],
                    "bm25_rank": 1,
                    "dense_rank": 1,
                }
            ],
        }

        with (
            patch(
                "app.gateway.routers.rag.aretrieve_tcm_docs",
                new=AsyncMock(return_value=payload),
                create=True,
            ) as retrieve,
            patch(
                "app.gateway.routers.rag.format_retrieval_results",
                return_value="检索状态：ok\n\n[E1]\n原文：头痛恶风，遇冷加重。",
                create=True,
            ) as format_results,
        ):
            response = TestClient(app).post(
                "/api/rag/query",
                json={"query": "  头痛恶风  ", "top_k": 3, "mode": "hybrid"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["formatted_text"], "检索状态：ok\n\n[E1]\n原文：头痛恶风，遇冷加重。")
        self.assertEqual(body["results"][0]["citation_id"], "E1")
        retrieve.assert_awaited_once_with("头痛恶风", k=3, mode="hybrid")
        format_results.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
