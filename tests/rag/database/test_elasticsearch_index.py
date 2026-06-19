import unittest

from app.rag.database.elasticsearch_index import (
    build_chunk_document,
    build_index_name,
    build_index_body,
    build_keyword_query,
    rebuild_keyword_index,
)


class FakeIndicesClient:
    def __init__(self):
        self.deleted = []
        self.created = []
        self.alias_updates = []

    async def delete(self, *, index, ignore_unavailable):
        self.deleted.append((index, ignore_unavailable))

    async def create(self, *, index, body):
        self.created.append((index, body))

    async def update_aliases(self, *, body):
        self.alias_updates.append(body)


class FakeElasticsearchClient:
    def __init__(self):
        self.indices = FakeIndicesClient()
        self.bulk_calls = []

    async def bulk(self, *, operations, refresh):
        self.bulk_calls.append((operations, refresh))
        return {"errors": False, "items": []}


class ElasticsearchIndexTests(unittest.IsolatedAsyncioTestCase):
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

    def test_index_body_maps_search_text_and_filter_fields(self):
        body = build_index_body(analyzer="standard")

        properties = body["mappings"]["properties"]
        self.assertEqual(properties["chunk_id"]["type"], "keyword")
        self.assertEqual(properties["corpus_id"]["type"], "keyword")
        self.assertEqual(properties["symptom_tags"]["type"], "keyword")
        self.assertEqual(properties["text"]["type"], "text")
        self.assertEqual(properties["text"]["analyzer"], "standard")

    async def test_rebuild_keyword_index_writes_documents_and_alias(self):
        client = FakeElasticsearchClient()
        bundle = type(
            "Bundle",
            (),
            {
                "corpus_id": "ancient-books-v1.0.0",
                "corpus_manifest": {"version": "v1.0.0"},
                "index_manifest": {"version": "v1.0.0"},
                "parents": [
                    {
                        "parent_id": "p1",
                        "book_id": "jing_yue_quan_shu",
                        "book_title": "Jing Yue Quan Shu",
                        "source_file": "source.txt",
                        "source_hash": "A" * 64,
                        "volume": "volume-one",
                        "chapter": "headache",
                        "section": "pattern",
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "parent_id": "p1",
                        "text": "headache and wind",
                        "symptom_tags": ["headache"],
                        "evidence_role": "syndrome_pattern",
                    }
                ],
            },
        )()

        result = await rebuild_keyword_index(
            client,
            bundle,
            alias="tcm_rag_chunks_current",
            analyzer="standard",
            batch_size=10,
        )

        self.assertEqual(
            client.indices.deleted,
            [("tcm_rag_chunks_v1_0_0", True)],
        )
        self.assertEqual(client.indices.created[0][0], "tcm_rag_chunks_v1_0_0")
        operations, refresh = client.bulk_calls[0]
        self.assertTrue(refresh)
        self.assertEqual(operations[0]["index"]["_id"], "c1")
        self.assertEqual(operations[1]["chunk_id"], "c1")
        self.assertEqual(operations[1]["row_index"], 0)
        self.assertEqual(
            client.indices.alias_updates[0]["actions"][-1],
            {
                "add": {
                    "index": "tcm_rag_chunks_v1_0_0",
                    "alias": "tcm_rag_chunks_current",
                    "is_write_index": True,
                }
            },
        )
        self.assertEqual(
            result,
            {
                "index": "tcm_rag_chunks_v1_0_0",
                "alias": "tcm_rag_chunks_current",
                "document_count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
