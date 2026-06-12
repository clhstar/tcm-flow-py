import unittest

from pydantic import ValidationError

from experiments.rag_v1_5.schema import EvidenceUnit


class EvidenceUnitSchemaTests(unittest.TestCase):
    def test_accepts_traceable_clause_unit(self):
        unit = EvidenceUnit(
            evidence_id="shl-taiyang-001",
            book_id="shang_han_lun",
            book_title="伤寒论",
            volume="",
            chapter_id="shl-taiyang",
            chapter_title="辨太阳病脉证并治上",
            clause_id="shl-taiyang-001",
            clause_number=1,
            content_type="clause",
            parent_id="shl-taiyang-001",
            original_text="太阳之为病，脉浮、头项强痛而恶寒。",
            normalized_text="太阳之为病，脉浮、头项强痛而恶寒。",
            notes=[],
            source_file="457-伤寒论.txt",
            source_hash="A" * 64,
            corpus_version="v1.5.0",
        )

        self.assertEqual(unit.evidence_id, unit.parent_id)
        self.assertEqual(unit.content_type, "clause")

    def test_rejects_unknown_content_type(self):
        with self.assertRaises(ValidationError):
            EvidenceUnit(
                evidence_id="shl-taiyang-001",
                book_id="shang_han_lun",
                book_title="伤寒论",
                volume="",
                chapter_id="shl-taiyang",
                chapter_title="辨太阳病脉证并治上",
                clause_id="shl-taiyang-001",
                clause_number=1,
                content_type="diagnosis",
                parent_id="shl-taiyang-001",
                original_text="原文",
                normalized_text="原文",
                notes=[],
                source_file="457-伤寒论.txt",
                source_hash="A" * 64,
                corpus_version="v1.5.0",
            )

    def test_rejects_invalid_source_hash(self):
        with self.assertRaises(ValidationError):
            EvidenceUnit(
                evidence_id="shl-taiyang-001",
                book_id="shang_han_lun",
                book_title="伤寒论",
                volume="",
                chapter_id="shl-taiyang",
                chapter_title="辨太阳病脉证并治上",
                clause_id="shl-taiyang-001",
                clause_number=1,
                content_type="clause",
                parent_id="shl-taiyang-001",
                original_text="原文",
                normalized_text="原文",
                notes=[],
                source_file="457-伤寒论.txt",
                source_hash="not-a-sha256",
                corpus_version="v1.5.0",
            )


if __name__ == "__main__":
    unittest.main()
