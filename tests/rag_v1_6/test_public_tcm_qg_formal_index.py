import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rag_v1_6.common import atomic_write_json, sha256_file, write_jsonl
from experiments.rag_v1_6.public_tcm_qg_formal_index import (
    build_public_tcm_qg_formal_indexes,
)


class FakeEmbedder:
    def encode(self, texts, *, batch_size, normalize_embeddings):
        rows = []
        for text in texts:
            rows.append([float(len(text)), float(len(set(text)))])
        vectors = np.asarray(rows, dtype=np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norm, 1e-12)
        return vectors


class PublicTcmQgFormalIndexTests(unittest.TestCase):
    def test_formal_index_writes_bm25_dense_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_dir = root / "chunks"
            output_dir = root / "formal" / "indexes"
            chunk_rows = [
                {
                    "chunk_id": "c1",
                    "strategy": "b4",
                    "source_doc_id": "d1",
                    "parent_id": "p1",
                    "text": "无症状胆囊结石可不作治疗。",
                    "context_text": "无症状胆囊结石可不作治疗。",
                    "start_index": 0,
                    "char_count": 14,
                    "context_start_index": 0,
                    "context_char_count": 14,
                    "source_qa_ids": ["q1"],
                }
            ]
            child_rows = [{**chunk_rows[0], "chunk_id": "c2", "strategy": "child"}]
            chunks_dir.mkdir(parents=True)
            write_jsonl(chunks_dir / "b4.jsonl", chunk_rows)
            write_jsonl(chunks_dir / "child.jsonl", child_rows)
            chunk_manifest = root / "chunk-manifest.json"
            atomic_write_json(
                chunk_manifest,
                {
                    "status": "ready",
                    "strategies": {
                        "b4": {
                            "output_file": "b4.jsonl",
                            "output_sha256": sha256_file(chunks_dir / "b4.jsonl"),
                        },
                        "child": {
                            "output_file": "child.jsonl",
                            "output_sha256": sha256_file(chunks_dir / "child.jsonl"),
                        },
                    },
                },
            )

            manifest = build_public_tcm_qg_formal_indexes(
                chunks_dir=chunks_dir,
                chunk_manifest_path=chunk_manifest,
                output_dir=output_dir,
                manifest_path=output_dir / "manifest.json",
                embedder=FakeEmbedder(),
                embedding_model="BAAI/bge-m3",
                embedding_revision="5617a9f61b028005a4858fdac845db406aefb181",
            )

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(set(manifest["strategies"]), {"b4", "child"})
            for strategy in ("b4", "child"):
                strategy_dir = output_dir / strategy
                self.assertTrue((strategy_dir / "rows.jsonl").is_file())
                self.assertTrue((strategy_dir / "bm25_tokens.jsonl").is_file())
                self.assertTrue((strategy_dir / "dense.npy").is_file())
                self.assertTrue((strategy_dir / "manifest.json").is_file())
                self.assertEqual(
                    manifest["strategies"][strategy]["embedding_model"],
                    "BAAI/bge-m3",
                )
                self.assertEqual(
                    manifest["strategies"][strategy]["embedding_revision"],
                    "5617a9f61b028005a4858fdac845db406aefb181",
                )
                self.assertEqual(
                    manifest["strategies"][strategy]["backend"],
                    "bm25_dense_rrf_rerank_ready",
                )


if __name__ == "__main__":
    unittest.main()
