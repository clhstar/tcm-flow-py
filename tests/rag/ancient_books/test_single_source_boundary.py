import importlib.util
import io
import unittest
from contextlib import redirect_stderr

from pydantic import ValidationError

from app.rag import terms
from app.rag import vector_store
from app.rag.ancient_books.cli import build_parser
from app.rag.ancient_books.schema import SelectedSection


class SingleSourceBoundaryTests(unittest.TestCase):
    def test_legacy_markdown_document_module_is_not_shipped(self):
        self.assertIsNone(importlib.util.find_spec("app.rag.documents"))

    def test_legacy_vector_store_alias_is_not_shipped(self):
        self.assertFalse(hasattr(vector_store, "get_vector_store"))

    def test_legacy_terms_query_helpers_are_not_shipped(self):
        self.assertFalse(hasattr(terms, "rewrite_tcm_query"))
        self.assertFalse(hasattr(terms, "detect_topic"))

    def test_build_corpus_cli_rejects_curated_markdown_root(self):
        parser = build_parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "build-corpus",
                    "--source-root",
                    "data/source",
                    "--curated-root",
                    "data/raw",
                ]
            )

    def test_schema_rejects_curated_markdown_source_type(self):
        with self.assertRaises(ValidationError):
            SelectedSection(
                section_id="section-001",
                source_type="curated_markdown",
                book_id="curated_markdown",
                book_title="curated",
                source_file="curated.md",
                source_hash="A" * 64,
                volume="",
                chapter="topic",
                section="section",
                symptom_tags=[],
                original_text="content",
            )


if __name__ == "__main__":
    unittest.main()
