import unittest
from pathlib import Path

from pydantic import ValidationError

from experiments.rag_v1_5.chunkers import (
    build_chunks,
    load_chunk_config,
    load_evidence,
    summarize_chunk_statistics,
    validate_chunks,
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

    def test_accepts_c5_and_rejects_unknown_strategy(self):
        chunk = ChunkUnit(**{**VALID_CHUNK, "strategy": "c5"})

        self.assertEqual(chunk.strategy, "c5")
        with self.assertRaises(ValidationError):
            ChunkUnit(**{**VALID_CHUNK, "strategy": "c6"})

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


class StructuredChunkingTests(unittest.TestCase):
    def setUp(self):
        self.units = load_evidence(FIXTURES_DIR / "evidence_sample.jsonl")
        self.config = load_chunk_config(CONFIG_PATH)

    def test_c3_keeps_every_evidence_unit_independent(self):
        chunks = build_chunks(self.units, "c3", self.config)

        self.assertEqual(len(chunks), len(self.units))
        self.assertEqual(
            [chunk.source_evidence_ids for chunk in chunks],
            [[unit.evidence_id] for unit in self.units],
        )
        self.assertEqual(
            [chunk.retrieval_parent_id for chunk in chunks],
            [unit.parent_id for unit in self.units],
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                f"c3-{unit.evidence_id}-001"
                for unit in self.units
            ],
        )

    def test_c3_adds_title_type_and_body_context_to_every_chunk(self):
        chunks = build_chunks(self.units, "c3", self.config)
        units_by_id = {unit.evidence_id: unit for unit in self.units}

        for chunk in chunks:
            unit = units_by_id[chunk.source_evidence_ids[0]]
            self.assertIn(f"书名：{unit.book_title}", chunk.text)
            self.assertIn(f"篇名：{unit.chapter_title}", chunk.text)
            self.assertIn(f"类型：{unit.content_type}", chunk.text)
            self.assertIn(f"正文：{unit.normalized_text}", chunk.text)
            self.assertEqual(chunk.context_text, chunk.text)
            self.assertEqual(chunk.char_count, len(chunk.text))

    def test_c3_repeats_context_when_one_evidence_unit_is_split(self):
        unit = self.units[0]
        long_unit = unit.model_copy(
            update={
                "normalized_text": "甲" * 900,
                "original_text": "甲" * 900,
            }
        )

        chunks = build_chunks([long_unit], "c3", self.config)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 500 for chunk in chunks))
        for part, chunk in enumerate(chunks, start=1):
            self.assertEqual(
                chunk.chunk_id,
                f"c3-{long_unit.evidence_id}-{part:03d}",
            )
            self.assertIn(f"书名：{long_unit.book_title}", chunk.text)
            self.assertIn(f"篇名：{long_unit.chapter_title}", chunk.text)
            self.assertIn(f"类型：{long_unit.content_type}", chunk.text)
            self.assertIn("正文：", chunk.text)
            self.assertEqual(
                chunk.source_evidence_ids,
                [long_unit.evidence_id],
            )


class ParentChildChunkingTests(unittest.TestCase):
    def setUp(self):
        self.units = load_evidence(FIXTURES_DIR / "evidence_sample.jsonl")
        self.config = load_chunk_config(CONFIG_PATH)

    def test_c4_retrieves_child_and_recovers_complete_clause(self):
        chunks = build_chunks(self.units, "c4", self.config)
        chunk = next(
            chunk
            for chunk in chunks
            if chunk.source_evidence_ids
            == ["jgy-chapter-25-040-formula-01-ingredients"]
        )

        self.assertEqual(
            chunk.retrieval_parent_id,
            "jgy-chapter-25-040",
        )
        self.assertIn("苦参（三两）", chunk.text)
        self.assertIn("饮食中毒，烦满", chunk.context_text)
        self.assertIn("犀角汤亦佳", chunk.context_text)

    def test_c4_maps_clause_to_itself_and_children_to_clause(self):
        chunks = build_chunks(self.units, "c4", self.config)
        by_evidence_id = {
            chunk.source_evidence_ids[0]: chunk for chunk in chunks
        }

        self.assertEqual(
            by_evidence_id["jgy-chapter-25-040"].retrieval_parent_id,
            "jgy-chapter-25-040",
        )
        child_ids = [
            "jgy-chapter-25-040-formula-01",
            "jgy-chapter-25-040-formula-01-ingredients",
            "jgy-chapter-25-040-formula-01-preparation",
            "jgy-chapter-25-040-formula-02",
            "jgy-chapter-25-040-note-01",
        ]
        self.assertEqual(
            {
                by_evidence_id[child_id].retrieval_parent_id
                for child_id in child_ids
            },
            {"jgy-chapter-25-040"},
        )

    def test_c4_splits_only_long_child_and_keeps_full_parent_context(self):
        parent = next(
            unit
            for unit in self.units
            if unit.evidence_id == "jgy-chapter-25-040"
        )
        child = next(
            unit
            for unit in self.units
            if unit.evidence_id
            == "jgy-chapter-25-040-formula-01-ingredients"
        ).model_copy(
            update={
                "normalized_text": "苦参" * 300,
                "original_text": "苦参" * 300,
            }
        )

        chunks = build_chunks([parent, child], "c4", self.config)
        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.source_evidence_ids == [child.evidence_id]
        ]

        self.assertGreater(len(child_chunks), 1)
        self.assertTrue(
            all(chunk.char_count <= 300 for chunk in child_chunks)
        )
        self.assertTrue(
            all(
                chunk.context_text == parent.normalized_text
                for chunk in child_chunks
            )
        )

    def test_c4_rejects_child_without_clause_id(self):
        child = next(
            unit
            for unit in self.units
            if unit.content_type == "ingredients"
        )
        invalid_child = child.model_construct(
            **{**child.model_dump(), "clause_id": None}
        )

        with self.assertRaisesRegex(ValueError, "clause_id"):
            build_chunks([invalid_child], "c4", self.config)

    def test_c4_rejects_missing_clause_parent(self):
        child = next(
            unit
            for unit in self.units
            if unit.content_type == "ingredients"
        ).model_copy(update={"clause_id": "missing-clause"})

        with self.assertRaisesRegex(ValueError, "missing-clause"):
            build_chunks([child], "c4", self.config)

    def test_c4_rejects_non_clause_parent(self):
        child = next(
            unit
            for unit in self.units
            if unit.content_type == "ingredients"
        )
        invalid_parent = next(
            unit
            for unit in self.units
            if unit.content_type == "formula"
        ).model_copy(update={"evidence_id": child.clause_id})

        with self.assertRaisesRegex(ValueError, "not a clause"):
            build_chunks([invalid_parent, child], "c4", self.config)

    def test_c4_rejects_duplicate_evidence_ids(self):
        unit = self.units[0]

        with self.assertRaisesRegex(ValueError, "Duplicate Evidence ID"):
            build_chunks([unit, unit.model_copy()], "c4", self.config)


class GenericParentChildChunkingTests(unittest.TestCase):
    def setUp(self):
        self.units = load_evidence(FIXTURES_DIR / "evidence_sample.jsonl")
        self.config = load_chunk_config(CONFIG_PATH)

    def test_c5_builds_generic_recursive_parents_and_children(self):
        chunks = build_chunks(self.units, "c5", self.config)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strategy == "c5" for chunk in chunks))
        self.assertTrue(all(chunk.char_count <= 300 for chunk in chunks))
        self.assertTrue(
            all(len(chunk.context_text) <= 1000 for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.retrieval_parent_id for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.source_evidence_ids for chunk in chunks)
        )
        self.assertTrue(
            any(
                len(chunk.source_evidence_ids) > 1
                for chunk in chunks
            )
        )

    def test_c5_does_not_use_content_type_to_choose_boundaries(self):
        changed = [
            unit.model_copy(
                update={
                    "content_type": (
                        "note"
                        if unit.content_type == "clause"
                        else "clause"
                    )
                }
            )
            for unit in self.units
        ]

        original = build_chunks(self.units, "c5", self.config)
        modified = build_chunks(changed, "c5", self.config)

        self.assertEqual(
            [
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.context_text,
                    chunk.source_evidence_ids,
                )
                for chunk in original
            ],
            [
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.context_text,
                    chunk.source_evidence_ids,
                )
                for chunk in modified
            ],
        )


class ChunkValidationTests(unittest.TestCase):
    def setUp(self):
        self.units = load_evidence(FIXTURES_DIR / "evidence_sample.jsonl")
        self.config = load_chunk_config(CONFIG_PATH)
        self.valid_chunks = build_chunks(self.units, "c4", self.config)

    def test_validate_chunks_accepts_valid_parent_child_graph(self):
        validate_chunks(self.valid_chunks, self.units)

    def test_validate_chunks_rejects_duplicate_chunk_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate Chunk ID"):
            validate_chunks(
                [self.valid_chunks[0], self.valid_chunks[0].model_copy()],
                self.units,
            )

    def test_validate_chunks_rejects_unknown_source_evidence(self):
        invalid = self.valid_chunks[0].model_copy(
            update={"source_evidence_ids": ["missing-evidence"]}
        )

        with self.assertRaisesRegex(ValueError, "missing-evidence"):
            validate_chunks([invalid], self.units)

    def test_validate_chunks_rejects_cross_book_or_chapter_chunks(self):
        source = self.valid_chunks[0]
        for field, value in (
            ("book_id", "other-book"),
            ("chapter_id", "other-chapter"),
        ):
            with self.subTest(field=field):
                invalid = source.model_copy(update={field: value})
                with self.assertRaisesRegex(ValueError, field):
                    validate_chunks([invalid], self.units)

    def test_validate_chunks_rejects_c2_to_c4_cross_clause_sources(self):
        c2_chunk = next(
            chunk
            for chunk in build_chunks(self.units, "c2", self.config)
            if chunk.chunk_id == "c2-shl-chapter-01-001-001"
        )
        invalid = c2_chunk.model_copy(
            update={
                "source_evidence_ids": [
                    "shl-chapter-01-001",
                    "shl-chapter-01-002",
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "crosses clause"):
            validate_chunks([invalid], self.units)

    def test_validate_chunks_rejects_c4_missing_clause_parent(self):
        invalid = self.valid_chunks[0].model_copy(
            update={"retrieval_parent_id": "missing-clause"}
        )

        with self.assertRaisesRegex(ValueError, "missing-clause"):
            validate_chunks([invalid], self.units)

    def test_validate_chunks_rejects_incorrect_char_count(self):
        invalid = self.valid_chunks[0].model_copy(
            update={"char_count": self.valid_chunks[0].char_count + 1}
        )

        with self.assertRaisesRegex(ValueError, "char_count"):
            validate_chunks([invalid], self.units)


class ChunkStatisticsTests(unittest.TestCase):
    def test_summarizes_hand_calculated_lengths(self):
        lengths = [50, 100, 200, 400]
        chunks = [
            ChunkUnit(
                **{
                    **VALID_CHUNK,
                    "chunk_id": f"c4-test-{index:03d}",
                    "strategy": "c4",
                    "text": "甲" * length,
                    "context_text": "乙" * (10 if index < 2 else 20),
                    "char_count": length,
                    "retrieval_parent_id": (
                        "parent-1" if index < 2 else "parent-2"
                    ),
                }
            )
            for index, length in enumerate(lengths)
        ]

        statistics = summarize_chunk_statistics(chunks)

        self.assertEqual(statistics["count"], 4)
        self.assertEqual(statistics["min"], 50)
        self.assertEqual(statistics["max"], 400)
        self.assertEqual(statistics["mean"], 187.5)
        self.assertEqual(statistics["median"], 150.0)
        self.assertEqual(statistics["p95"], 400)
        self.assertEqual(statistics["short_ratio"], 0.25)
        self.assertEqual(statistics["long_ratio"], 0.0)
        self.assertEqual(statistics["unique_parent_count"], 2)
        self.assertEqual(statistics["parent_context_mean"], 15.0)
        self.assertEqual(statistics["parent_context_p95"], 20)


if __name__ == "__main__":
    unittest.main()
