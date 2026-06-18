import unittest

from app.rag.database.elasticsearch_index import (
    build_chunk_document,
    build_index_name,
    build_keyword_query,
)


class ElasticsearchIndexTests(unittest.TestCase):
    def test_build_index_name_normalizes_version(self):
        self.assertEqual(
            build_index_name("v1.0.0"),
            "tcm_rag_chunks_v1_0_0",
        )

    def test_build_chunk_document_uses_chunk_id_as_id_source(self):
        parent = {
            "book_id": "jing_yue_quan_shu",
            "book_title": "Jing Yue Quan Shu",
            "source_file": "637-jing-yue-quan-shu.txt",
            "source_hash": "A" * 64,
            "volume": "volume-one",
            "chapter": "headache",
            "section": "pattern",
        }
        chunk = {
            "chunk_id": "c1",
            "parent_id": "p1",
            "corpus_id": "ancient-books-v1.0.0",
            "row_index": 0,
            "text": "headache and wind",
            "symptom_tags": ["headache"],
            "evidence_role": "syndrome_pattern",
        }

        document = build_chunk_document(chunk, parent, "v1.0.0")

        self.assertEqual(document["chunk_id"], "c1")
        self.assertEqual(document["parent_id"], "p1")
        self.assertEqual(document["book_title"], "Jing Yue Quan Shu")
        self.assertEqual(document["index_version"], "v1.0.0")

    def test_keyword_query_filters_corpus_symptom_and_role(self):
        query = build_keyword_query(
            rewritten_query="headache wind",
            corpus_id="ancient-books-v1.0.0",
            chief_symptom="headache",
            evidence_roles=["syndrome_pattern"],
            top_k=20,
        )

        self.assertEqual(query["size"], 20)
        filters = query["query"]["bool"]["filter"]
        self.assertIn({"term": {"corpus_id": "ancient-books-v1.0.0"}}, filters)
        self.assertIn({"term": {"symptom_tags": "headache"}}, filters)
        self.assertIn({"terms": {"evidence_role": ["syndrome_pattern"]}}, filters)


if __name__ == "__main__":
    unittest.main()
