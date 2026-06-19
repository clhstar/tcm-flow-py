import asyncio
import unittest
from pathlib import Path

from app.db.migrations import run_schema_migrations, split_sql_statements


SCHEMA_PATH = Path("app/db/schema.sql")


class FakeTransaction:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        self._events.append(("transaction", "enter"))

    async def __aexit__(self, exc_type, exc, traceback):
        self._events.append(("transaction", "exit"))


class FakeConnection:
    def __init__(self):
        self.events = []
        self.executed = []

    def transaction(self):
        return FakeTransaction(self.events)

    async def execute(self, statement):
        self.executed.append(statement)
        self.events.append(("execute", statement))


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

    def test_schema_indexes_all_foreign_key_join_paths(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

        for index_sql in (
            "create index if not exists app_agent_trace_events_run_idx on app_agent_trace_events (run_id)",
            "create index if not exists app_agent_trace_events_thread_idx on app_agent_trace_events (thread_id)",
            "create index if not exists app_validation_results_run_idx on app_validation_results (run_id)",
            "create index if not exists app_validation_results_thread_idx on app_validation_results (thread_id)",
            "create index if not exists rag_sources_corpus_idx on rag_sources (corpus_id)",
            "create index if not exists rag_sections_corpus_idx on rag_sections (corpus_id)",
            "create index if not exists rag_sections_source_idx on rag_sections (source_id)",
            "create index if not exists rag_parents_corpus_idx on rag_parents (corpus_id)",
            "create index if not exists rag_parents_source_idx on rag_parents (source_id)",
            "create index if not exists rag_parents_section_idx on rag_parents (section_id)",
            "create index if not exists rag_chunk_embeddings_corpus_idx on rag_chunk_embeddings (corpus_id)",
            "create index if not exists rag_bm25_tokens_corpus_idx on rag_bm25_tokens (corpus_id)",
            "create index if not exists rag_retrieval_logs_run_idx on rag_retrieval_logs (run_id)",
            "create index if not exists rag_retrieval_logs_thread_idx on rag_retrieval_logs (thread_id)",
            "create index if not exists rag_retrieval_logs_corpus_created_idx on rag_retrieval_logs (corpus_id, created_at desc)",
        ):
            self.assertIn(index_sql, sql)

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

    def test_split_sql_statements_ignores_inline_comments_after_semicolons(self):
        sql = """
        create table one(id int); -- inline comment
        create table two(id int);
        """

        self.assertEqual(
            split_sql_statements(sql),
            ["create table one(id int)", "create table two(id int)"],
        )

    def test_run_schema_migrations_executes_ordered_statements_in_transaction(self):
        connection = FakeConnection()
        expected_statements = split_sql_statements(SCHEMA_PATH.read_text(encoding="utf-8"))

        asyncio.run(run_schema_migrations(connection))

        self.assertEqual(connection.executed, expected_statements)
        self.assertEqual(
            connection.events,
            [
                ("transaction", "enter"),
                *((("execute", statement) for statement in expected_statements)),
                ("transaction", "exit"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
