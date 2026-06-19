import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.rag.database.artifacts import load_artifact_bundle


class ArtifactLoaderTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def write_bundle_files(
        self,
        corpus: Path,
        index: Path,
        *,
        dense: np.ndarray,
        vector_dimension: int = 1024,
    ):
        parent = {
            "parent_id": "p1",
            "source_type": "ancient_book",
            "book_id": "jing_yue_quan_shu",
            "book_title": "Jing Yue Quan Shu",
            "source_file": "637-jing-yue-quan-shu.txt",
            "source_hash": "A" * 64,
            "volume": "volume-one",
            "chapter": "headache",
            "section": "pattern",
            "symptom_tags": ["headache"],
            "evidence_role": "syndrome_pattern",
            "original_text": "headache and wind",
            "normalized_text": "headache and wind",
        }
        chunk = {
            "chunk_id": "c1",
            "parent_id": "p1",
            "text": "headache and wind",
            "source_type": "ancient_book",
            "symptom_tags": ["headache"],
            "evidence_role": "syndrome_pattern",
        }
        self.write_jsonl(corpus / "parents.jsonl", [parent])
        self.write_jsonl(corpus / "chunks.jsonl", [chunk])
        self.write_jsonl(index / "rows.jsonl", [chunk])
        self.write_jsonl(
            index / "bm25_tokens.jsonl",
            [{"chunk_id": "c1", "tokens": ["headache", "wind"]}],
        )
        np.save(index / "dense.npy", dense, allow_pickle=False)
        (corpus / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "parent_count": 1,
                    "chunk_count": 1,
                    "version": "v1.0.0",
                }
            ),
            encoding="utf-8",
        )
        (index / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "row_count": 1,
                    "vector_dimension": vector_dimension,
                    "embedding_model": {"model": "BAAI/bge-m3", "revision": "r1"},
                }
            ),
            encoding="utf-8",
        )

    def test_loader_rejects_vector_dimension_mismatch(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()
            self.write_bundle_files(
                corpus,
                index,
                dense=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "vector_dimension"):
                load_artifact_bundle(corpus, index)

    def test_loader_returns_ordered_rows_tokens_and_vectors(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()
            self.write_bundle_files(
                corpus,
                index,
                dense=np.zeros((1, 1024), dtype=np.float32),
            )

            bundle = load_artifact_bundle(corpus, index)

        self.assertEqual(bundle.corpus_id, "ancient-books-v1.0.0")
        self.assertEqual(bundle.parents[0]["parent_id"], "p1")
        self.assertEqual(bundle.chunks[0]["chunk_id"], "c1")
        self.assertEqual(bundle.tokens_by_chunk_id["c1"], ["headache", "wind"])
        self.assertEqual(bundle.dense.shape, (1, 1024))

    def test_loader_reorders_chunks_to_match_index_rows(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()

            parent_rows = [
                {
                    "parent_id": "p1",
                    "source_type": "ancient_book",
                    "book_id": "jing_yue_quan_shu",
                    "book_title": "Jing Yue Quan Shu",
                    "source_file": "637-jing-yue-quan-shu.txt",
                    "source_hash": "A" * 64,
                    "volume": "volume-one",
                    "chapter": "headache",
                    "section": "pattern",
                    "symptom_tags": ["headache"],
                    "evidence_role": "syndrome_pattern",
                    "original_text": "first",
                    "normalized_text": "first",
                },
                {
                    "parent_id": "p2",
                    "source_type": "ancient_book",
                    "book_id": "jing_yue_quan_shu",
                    "book_title": "Jing Yue Quan Shu",
                    "source_file": "637-jing-yue-quan-shu.txt",
                    "source_hash": "B" * 64,
                    "volume": "volume-one",
                    "chapter": "cough",
                    "section": "pattern",
                    "symptom_tags": ["cough"],
                    "evidence_role": "syndrome_pattern",
                    "original_text": "second",
                    "normalized_text": "second",
                },
            ]
            chunks_by_corpus_order = [
                {
                    "chunk_id": "c2",
                    "parent_id": "p2",
                    "text": "second",
                    "source_type": "ancient_book",
                    "symptom_tags": ["cough"],
                    "evidence_role": "syndrome_pattern",
                },
                {
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "text": "first",
                    "source_type": "ancient_book",
                    "symptom_tags": ["headache"],
                    "evidence_role": "syndrome_pattern",
                },
            ]
            rows_by_index_order = list(reversed(chunks_by_corpus_order))

            self.write_jsonl(corpus / "parents.jsonl", parent_rows)
            self.write_jsonl(corpus / "chunks.jsonl", chunks_by_corpus_order)
            self.write_jsonl(index / "rows.jsonl", rows_by_index_order)
            self.write_jsonl(
                index / "bm25_tokens.jsonl",
                [
                    {"chunk_id": "c1", "tokens": ["first"]},
                    {"chunk_id": "c2", "tokens": ["second"]},
                ],
            )
            np.save(index / "dense.npy", np.zeros((2, 1024), dtype=np.float32))
            (corpus / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "parent_count": 2,
                        "chunk_count": 2,
                        "version": "v1.0.0",
                    }
                ),
                encoding="utf-8",
            )
            (index / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "row_count": 2,
                        "vector_dimension": 1024,
                        "embedding_model": {"model": "BAAI/bge-m3", "revision": "r1"},
                    }
                ),
                encoding="utf-8",
            )

            bundle = load_artifact_bundle(corpus, index)

        self.assertEqual([chunk["chunk_id"] for chunk in bundle.chunks], ["c1", "c2"])
        self.assertEqual([row["chunk_id"] for row in bundle.rows], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
