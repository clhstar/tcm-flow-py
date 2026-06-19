import unittest
from unittest.mock import patch

from app.rag.retriever import format_retrieval_results
from app.tools.builtins.retrieval_tool import retrieve_tcm_knowledge


PAYLOAD = {
    "status": "ok",
    "retrieval_mode": "hybrid_parent",
    "degraded": False,
    "degraded_reason": None,
    "original_query": "头痛恶风",
    "rewritten_query": "头痛恶风 头痛 头风",
    "chief_symptom": "头痛",
    "allowed_terms": ["头痛"],
    "results": [{
        "citation_id": "E1",
        "content": "因风者恶风。",
        "matched_child": "因风者恶风。",
        "book_title": "景岳全书",
        "source_file": "637-景岳全书.txt",
        "volume": "卷之十",
        "chapter": "杂证谟",
        "section": "头痛",
        "evidence_role": "syndrome_pattern",
        "chunk_id": "c1",
        "parent_id": "p1",
        "source_type": "ancient_book",
        "symptom_tags": ["头痛"],
        "retrieval_sources": ["bm25", "dense"],
    }],
}


class RagToolTests(unittest.IsolatedAsyncioTestCase):
    def test_formatter_emits_citation_and_single_book_source(self):
        text = format_retrieval_results(PAYLOAD)

        self.assertIn("[E1]", text)
        self.assertIn("《景岳全书》", text)
        self.assertIn("主症：头痛", text)
        self.assertIn("parent_id=p1", text)
        self.assertNotIn("处方", text)

    @patch("app.tools.builtins.retrieval_tool.aretrieve_tcm_docs")
    @patch("app.tools.builtins.retrieval_tool.write_retrieval_log")
    async def test_tool_keeps_name_and_logs_stable_evidence_ids(self, log, retrieve):
        retrieve.return_value = PAYLOAD

        result = await retrieve_tcm_knowledge.ainvoke(
            {"query": "头痛恶风", "mode": "hybrid"}
        )

        self.assertEqual(retrieve_tcm_knowledge.name, "retrieve_tcm_knowledge")
        self.assertIn("[E1]", result)
        record = log.call_args.args[0]
        self.assertEqual(record["final_results"][0]["parent_id"], "p1")
        self.assertEqual(record["final_results"][0]["citation_id"], "E1")
        self.assertFalse(record["degraded"])


if __name__ == "__main__":
    unittest.main()
