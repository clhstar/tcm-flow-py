import unittest
from pathlib import Path

from app.db.migrations import split_sql_statements


SCHEMA_PATH = Path("app/db/schema.sql")


class SchemaTests(unittest.TestCase):
    def test_schema_declares_required_extensions_and_tables(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8")

        self.assertIn("create extension if not exists vector", sql.lower())
        for table in (
            "app_threads",
            "app_runs",
            "app_messages",
            "app_agent_trace_events",
            "app_validation_results",
            "rag_corpora",
            "rag_sources",
            "rag_sections",
            "rag_parents",
            "rag_chunks",
            "rag_chunk_embeddings",
            "rag_bm25_tokens",
            "rag_retrieval_logs",
        ):
            self.assertIn(f"create table if not exists {table}", sql.lower())

    def test_schema_indexes_foreign_keys_and_common_filters(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

        for index_name in (
            "app_runs_thread_created_idx",
            "app_messages_thread_ordinal_idx",
            "app_messages_visible_idx",
            "rag_chunks_parent_idx",
            "rag_chunks_symptom_tags_idx",
            "rag_chunk_embeddings_hnsw_idx",
        ):
            self.assertIn(index_name, sql)

    def test_split_sql_statements_ignores_comments_and_empty_chunks(self):
        sql = """
        -- first comment
        create table one(id int);

        -- second comment
        create table two(id int);
        """

        self.assertEqual(
            split_sql_statements(sql),
            ["create table one(id int)", "create table two(id int)"],
        )


if __name__ == "__main__":
    unittest.main()
