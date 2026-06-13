import unittest

from pydantic import ValidationError

from experiments.rag_v1_5.schema import ChunkUnit


VALID_CHUNK = {
    "chunk_id": "c0-shl-chapter-01-000000",
    "strategy": "c0",
    "book_id": "shang_han_lun",
    "chapter_id": "shl-chapter-01",
    "clause_id": None,
    "retrieval_parent_id": None,
    "source_evidence_ids": ["shl-chapter-01-001"],
    "text": "太阳之为病，脉浮、头项强痛而恶寒。",
    "context_text": "太阳之为病，脉浮、头项强痛而恶寒。",
    "char_count": 18,
    "start_index": 0,
    "source_hash": "a" * 64,
    "corpus_version": "v1.5.0",
}


class ChunkUnitSchemaTests(unittest.TestCase):
    def test_accepts_traceable_chunk_and_normalizes_source_hash(self):
        chunk = ChunkUnit(**VALID_CHUNK)

        self.assertEqual(chunk.source_hash, "A" * 64)
        self.assertEqual(chunk.source_evidence_ids, ["shl-chapter-01-001"])

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(ValidationError):
            ChunkUnit(**{**VALID_CHUNK, "strategy": "c5"})

    def test_rejects_empty_source_evidence_ids(self):
        with self.assertRaises(ValidationError):
            ChunkUnit(**{**VALID_CHUNK, "source_evidence_ids": []})

    def test_rejects_non_positive_char_count(self):
        with self.assertRaises(ValidationError):
            ChunkUnit(**{**VALID_CHUNK, "char_count": 0})

    def test_rejects_invalid_source_hash(self):
        with self.assertRaises(ValidationError):
            ChunkUnit(**{**VALID_CHUNK, "source_hash": "not-a-sha256"})


if __name__ == "__main__":
    unittest.main()
