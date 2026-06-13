import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rag_v1_5.schema import ChunkUnit


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "retrieval_chunks_sample.jsonl"
)


def load_chunks() -> list[ChunkUnit]:
    return [
        ChunkUnit.model_validate_json(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


class FakeEncoder:
    def __init__(self, vectors: np.ndarray | None = None) -> None:
        self.texts = None
        self.vectors = vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts = list(texts)
        if self.vectors is not None:
            return self.vectors
        return np.asarray(
            [[index + 1.0, 1.0] for index in range(len(texts))],
            dtype=np.float64,
        )


class IndexingTests(unittest.TestCase):
    def indexing_module(self):
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.indexing"
        )
        self.assertIsNotNone(module_spec, "indexing module is not implemented")
        return importlib.import_module("experiments.rag_v1_5.indexing")

    def test_builds_aligned_normalized_deterministic_index(self) -> None:
        indexing = self.indexing_module()
        builder = getattr(indexing, "build_strategy_index", None)
        self.assertTrue(
            callable(builder),
            "build_strategy_index is not implemented",
        )
        chunks = load_chunks()
        encoder = FakeEncoder()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "first"
            second_dir = root / "second"
            first = builder(
                chunks=chunks,
                output_dir=first_dir,
                encoder=encoder,
                quality_gate_sha256="A" * 64,
                chunk_sha256="B" * 64,
                model_record={
                    "model": "BAAI/bge-m3",
                    "revision": "1" * 40,
                    "local_path": "data/models/bge-m3",
                },
            )
            second = builder(
                chunks=chunks,
                output_dir=second_dir,
                encoder=FakeEncoder(),
                quality_gate_sha256="A" * 64,
                chunk_sha256="B" * 64,
                model_record={
                    "model": "BAAI/bge-m3",
                    "revision": "1" * 40,
                    "local_path": "data/models/bge-m3",
                },
            )

            row_ids = [
                json.loads(line)["chunk_id"]
                for line in (first_dir / "rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            token_rows = [
                json.loads(line)
                for line in (first_dir / "bm25_tokens.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            vectors = np.load(first_dir / "dense.npy")

        self.assertEqual(row_ids, sorted(row_ids))
        self.assertEqual(
            [row["chunk_id"] for row in token_rows],
            row_ids,
        )
        self.assertEqual(len(vectors), len(row_ids))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1),
            np.ones(len(vectors)),
            atol=1e-6,
        )
        self.assertEqual(
            encoder.texts,
            [chunk.text for chunk in sorted(chunks, key=lambda item: item.chunk_id)],
        )
        for output_name in ("rows", "bm25_tokens", "dense"):
            self.assertEqual(
                first["files"][output_name]["sha256"],
                second["files"][output_name]["sha256"],
            )

    def test_rejects_duplicate_chunks_empty_tokens_and_bad_vectors(self) -> None:
        indexing = self.indexing_module()
        chunks = load_chunks()
        cases = {
            "duplicate": (
                [chunks[0], chunks[0]],
                FakeEncoder(),
            ),
            "empty_tokens": (
                [chunks[0].model_copy(update={"text": " \n "})],
                FakeEncoder(),
            ),
            "vector_count": (
                chunks,
                FakeEncoder(np.ones((2, 2), dtype=np.float32)),
            ),
            "nan": (
                chunks,
                FakeEncoder(
                    np.asarray(
                        [[1.0, 0.0], [np.nan, 1.0], [1.0, 1.0]],
                        dtype=np.float32,
                    )
                ),
            ),
        }
        for name, (case_chunks, encoder) in cases.items():
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaises(ValueError):
                        indexing.build_strategy_index(
                            chunks=case_chunks,
                            output_dir=Path(temporary_directory),
                            encoder=encoder,
                            quality_gate_sha256="A" * 64,
                            chunk_sha256="B" * 64,
                            model_record={
                                "model": "BAAI/bge-m3",
                                "revision": "1" * 40,
                                "local_path": "data/models/bge-m3",
                            },
                        )

    def test_real_build_rejects_non_ready_gate(self) -> None:
        indexing = self.indexing_module()
        validator = getattr(indexing, "validate_quality_gate", None)
        self.assertTrue(
            callable(validator),
            "validate_quality_gate is not implemented",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            gate_path = Path(temporary_directory) / "quality-gate.json"
            gate_path.write_text(
                json.dumps({"status": "blocked", "reviewed_count": 140}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validator(gate_path)

    def test_model_manifest_must_match_revision_and_snapshot_hash(self) -> None:
        indexing = self.indexing_module()
        loader = getattr(
            indexing,
            "load_verified_embedding_model",
            None,
        )
        self.assertTrue(
            callable(loader),
            "load_verified_embedding_model is not implemented",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.yaml"
            model_manifest_path = root / "models.json"
            config_path.write_text(
                "embedding:\n"
                "  model: BAAI/bge-m3\n"
                f"  revision: {'1' * 40}\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                loader(
                    config_path=config_path,
                    model_manifest_path=model_manifest_path,
                    repository_root=root,
                )

            model_dir = root / "data" / "models" / "bge-m3"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            model_manifest_path.write_text(
                json.dumps(
                    {
                        "embedding": {
                            "model": "BAAI/bge-m3",
                            "revision": "2" * 40,
                            "local_path": "data/models/bge-m3",
                            "snapshot_tree_sha256": "A" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                loader(
                    config_path=config_path,
                    model_manifest_path=model_manifest_path,
                    repository_root=root,
                )

            manifest = json.loads(
                model_manifest_path.read_text(encoding="utf-8")
            )
            manifest["embedding"]["revision"] = "1" * 40
            model_manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                loader(
                    config_path=config_path,
                    model_manifest_path=model_manifest_path,
                    repository_root=root,
                )


class TokenizationTests(unittest.TestCase):
    def test_tokenization_is_nonempty_and_whitespace_stable(self) -> None:
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.tokenization"
        )
        self.assertIsNotNone(
            module_spec,
            "tokenization module is not implemented",
        )
        tokenization = importlib.import_module(
            "experiments.rag_v1_5.tokenization"
        )
        tokenizer = getattr(tokenization, "tokenize_text", None)
        self.assertTrue(callable(tokenizer), "tokenize_text is not implemented")

        self.assertEqual(
            tokenizer("苦参（三两）   苦酒（一升半）"),
            tokenizer("苦参（三两） 苦酒（一升半）"),
        )
        self.assertTrue(tokenizer("苦参和苦酒如何使用"))
        self.assertEqual(tokenizer(" \n\t "), [])


if __name__ == "__main__":
    unittest.main()
