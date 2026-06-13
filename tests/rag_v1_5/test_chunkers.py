import unittest
from pathlib import Path

from pydantic import ValidationError

from experiments.rag_v1_5.chunkers import (
    build_chunks,
    load_chunk_config,
    load_evidence,
)
from experiments.rag_v1_5.schema import ChunkUnit


FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_PATH = (
    Path(__file__).parents[2]
    / "experiments"
    / "rag_v1_5"
    / "configs"
    / "chunks.yaml"
)


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


class CharacterChunkingTests(unittest.TestCase):
    def setUp(self):
        self.fixture_path = FIXTURES_DIR / "evidence_sample.jsonl"
        self.config = load_chunk_config(CONFIG_PATH)

    def test_load_evidence_has_stable_document_order(self):
        first = load_evidence(self.fixture_path)
        second = load_evidence(self.fixture_path)
        expected = sorted(
            first,
            key=lambda unit: (
                unit.book_id,
                unit.chapter_id,
                unit.clause_number or 0,
                unit.evidence_id,
            ),
        )

        self.assertEqual(
            [unit.evidence_id for unit in first],
            [unit.evidence_id for unit in second],
        )
        self.assertEqual(
            [unit.evidence_id for unit in first],
            [unit.evidence_id for unit in expected],
        )

    def test_c0_and_c1_chunk_chapters_without_child_duplication(self):
        units = load_evidence(self.fixture_path)
        clause_ids = {
            unit.evidence_id
            for unit in units
            if unit.content_type == "clause"
        }

        for strategy, size, overlap in (("c0", 500, 80), ("c1", 250, 40)):
            with self.subTest(strategy=strategy):
                chunks = build_chunks(units, strategy, self.config)
                parameters = self.config["strategies"][strategy]

                self.assertEqual(parameters["chunk_size"], size)
                self.assertEqual(parameters["chunk_overlap"], overlap)
                self.assertTrue(chunks)
                self.assertEqual(
                    {source_id for chunk in chunks for source_id in chunk.source_evidence_ids},
                    clause_ids,
                )
                self.assertTrue(
                    all(
                        set(chunk.source_evidence_ids) <= clause_ids
                        for chunk in chunks
                    )
                )
                self.assertTrue(
                    all(chunk.char_count <= size for chunk in chunks)
                )

    def test_c0_and_c1_preserve_chapter_boundaries_and_stable_ids(self):
        units = load_evidence(self.fixture_path)

        for strategy in ("c0", "c1"):
            with self.subTest(strategy=strategy):
                chunks = build_chunks(units, strategy, self.config)

                for chunk in chunks:
                    self.assertIsNone(chunk.clause_id)
                    self.assertIsNone(chunk.retrieval_parent_id)
                    self.assertEqual(chunk.context_text, chunk.text)
                    self.assertEqual(chunk.char_count, len(chunk.text))
                    self.assertEqual(
                        chunk.chunk_id,
                        f"{strategy}-{chunk.chapter_id}-{chunk.start_index:06d}",
                    )
                    source_units = [
                        unit
                        for unit in units
                        if unit.evidence_id in chunk.source_evidence_ids
                    ]
                    self.assertEqual(
                        {unit.book_id for unit in source_units},
                        {chunk.book_id},
                    )
                    self.assertEqual(
                        {unit.chapter_id for unit in source_units},
                        {chunk.chapter_id},
                    )


class ClauseChunkingTests(unittest.TestCase):
    def setUp(self):
        self.units = load_evidence(FIXTURES_DIR / "evidence_sample.jsonl")
        self.config = load_chunk_config(CONFIG_PATH)

    def test_c2_creates_one_chunk_per_regular_clause(self):
        clauses = [
            unit for unit in self.units if unit.content_type == "clause"
        ]
        chunks = build_chunks(self.units, "c2", self.config)

        self.assertEqual(len(chunks), len(clauses))
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                f"c2-{clause.evidence_id}-001"
                for clause in clauses
            ],
        )
        for clause, chunk in zip(clauses, chunks):
            self.assertEqual(chunk.clause_id, clause.evidence_id)
            self.assertEqual(chunk.retrieval_parent_id, clause.evidence_id)
            self.assertEqual(
                chunk.source_evidence_ids,
                [clause.evidence_id],
            )
            self.assertEqual(chunk.text, clause.normalized_text)
            self.assertEqual(chunk.context_text, clause.normalized_text)

    def test_c2_never_merges_adjacent_short_clauses(self):
        adjacent_clauses = [
            unit
            for unit in self.units
            if unit.chapter_id == "shl-chapter-01"
            and unit.content_type == "clause"
        ]
        self.assertLess(
            sum(len(unit.normalized_text) for unit in adjacent_clauses),
            500,
        )

        chunks = build_chunks(adjacent_clauses, "c2", self.config)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [chunk.source_evidence_ids for chunk in chunks],
            [[unit.evidence_id] for unit in adjacent_clauses],
        )

    def test_c2_splits_an_overlong_clause_only_within_itself(self):
        clause = next(
            unit
            for unit in self.units
            if unit.evidence_id == "shl-chapter-01-001"
        )
        long_clause = clause.model_copy(
            update={
                "normalized_text": "甲" * 1200,
                "original_text": "甲" * 1200,
            }
        )

        chunks = build_chunks([long_clause], "c2", self.config)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                f"c2-{long_clause.evidence_id}-{part:03d}"
                for part in range(1, len(chunks) + 1)
            ],
        )
        self.assertTrue(all(chunk.char_count <= 500 for chunk in chunks))
        self.assertTrue(
            all(
                chunk.source_evidence_ids == [long_clause.evidence_id]
                for chunk in chunks
            )
        )
        self.assertTrue(
            all(
                chunk.retrieval_parent_id == long_clause.evidence_id
                for chunk in chunks
            )
        )


if __name__ == "__main__":
    unittest.main()
