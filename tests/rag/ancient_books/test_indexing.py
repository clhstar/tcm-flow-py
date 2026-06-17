import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.rag.ancient_books.indexing import build_index, normalize_vectors


class FakeEncoder:
    def encode(self, texts):
        return np.asarray(
            [[index + 1.0, 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


class IndexingTests(unittest.TestCase):
    def test_index_writes_sorted_rows_tokens_vectors_and_hash_manifest(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks = root / "chunks.jsonl"
            records = [
                {
                    "chunk_id": "c2",
                    "parent_id": "p2",
                    "text": "咳嗽有痰",
                    "source_type": "ancient_book",
                    "symptom_tags": ["咳嗽"],
                    "evidence_role": "syndrome_pattern",
                },
                {
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "text": "头痛恶风",
                    "source_type": "ancient_book",
                    "symptom_tags": ["头痛"],
                    "evidence_role": "syndrome_pattern",
                },
            ]
            chunks.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )

            manifest = build_index(
                chunks_path=chunks,
                corpus_manifest_sha256="A" * 64,
                output_dir=root / "index",
                encoder=FakeEncoder(),
                model_record={"model": "fake", "revision": "1" * 40},
            )

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(manifest["corpus_manifest_sha256"], "A" * 64)
            rows = [
                json.loads(line)
                for line in (root / "index" / "rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["chunk_id"] for row in rows], ["c1", "c2"])
            self.assertTrue((root / "index" / "bm25_tokens.jsonl").is_file())
            vectors = np.load(root / "index" / "dense.npy", allow_pickle=False)
            np.testing.assert_allclose(
                np.linalg.norm(vectors, axis=1),
                np.ones(2),
                rtol=1e-6,
            )
            self.assertIn("sha256", manifest["files"]["dense"])

    def test_normalize_vectors_rejects_zero_nan_and_wrong_row_count(self):
        invalid_vectors = (
            (np.asarray([[0.0, 0.0]], dtype=np.float32), 1),
            (np.asarray([[np.nan, 1.0]], dtype=np.float32), 1),
            (np.asarray([[1.0, 1.0]], dtype=np.float32), 2),
        )
        for vectors, expected_count in invalid_vectors:
            with self.subTest(vectors=vectors, expected_count=expected_count):
                with self.assertRaises(ValueError):
                    normalize_vectors(vectors, expected_count)


if __name__ == "__main__":
    unittest.main()
