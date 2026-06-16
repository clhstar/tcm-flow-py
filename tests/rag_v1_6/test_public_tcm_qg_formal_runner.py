import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments.rag_v1_6.common import atomic_write_json, sha256_file, write_jsonl
from experiments.rag_v1_6.public_tcm_qg_formal_index import (
    build_public_tcm_qg_formal_indexes,
)
from experiments.rag_v1_6.public_tcm_qg_formal_runner import (
    freeze_public_tcm_qg_formal_prereg,
    public_tcm_qg_formal_matrix,
    run_public_tcm_qg_formal_retrieval_matrix,
    validate_formal_retrieval_inputs,
)


class FakeEmbedder:
    def encode(self, texts, *, batch_size, normalize_embeddings):
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float("胆囊" in text),
                    float("结石" in text),
                    float("治疗" in text),
                ]
            )
        array = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(array, axis=1, keepdims=True)
            array = array / np.maximum(norm, 1e-12)
        return array


class FakeReranker:
    def compute_score(self, pairs, *, batch_size):
        return [float("无症状" in pair[1]) for pair in pairs]


class PublicTcmQgFormalMatrixTests(unittest.TestCase):
    def test_formal_matrix_freezes_retrieval_and_answer_methods(self):
        matrix = public_tcm_qg_formal_matrix()

        self.assertEqual(
            [row["config_id"] for row in matrix["retrieval_configs"]],
            [
                "b1-public-bm25",
                "b2-public-dense",
                "b3-public-hybrid",
                "b4-public-hybrid-rerank",
                "p-public-hybrid-rerank",
                "p-public-no-parent",
                "p-public-no-reranker",
            ],
        )
        self.assertEqual(matrix["answer_methods"], ["B0", "B4", "P", "P-no-parent"])
        self.assertIn("strong_success", matrix["success_gates"])
        self.assertIn("parent_ablation_only", matrix["success_gates"])


class PublicTcmQgFormalPreregTests(unittest.TestCase):
    def test_prereg_manifest_excludes_private_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest = root / "source.json"
            dataset_manifest = root / "dataset.json"
            chunk_manifest = root / "chunk.json"
            model_manifest = root / "models.json"
            env_path = root / ".env"
            config_path = root / "formal.yaml"
            output_path = root / "formal-prereg.json"
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text('{"qa_id":"q1"}\n', encoding="utf-8")
            for path, stage in (
                (source_manifest, "source"),
                (dataset_manifest, "dataset"),
                (chunk_manifest, "chunk"),
                (model_manifest, "models"),
            ):
                atomic_write_json(path, {"status": "ready", "stage": stage})
            env_path.write_text(
                "OPENAI_BASE_URL=https://api.example.com/v1\n"
                "OPENAI_MODEL=formal-answer-model\n"
                "OPENAI_API_KEY=secret\n",
                encoding="utf-8",
            )
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "version": "v1.6.0",
                        "seed": 20260616,
                        "inputs": {
                            "source_manifest": source_manifest.as_posix(),
                            "dataset_manifest": dataset_manifest.as_posix(),
                            "dataset_path": dataset_path.as_posix(),
                            "chunk_manifest": chunk_manifest.as_posix(),
                            "model_manifest": model_manifest.as_posix(),
                        },
                        "answer": {
                            "methods": ["B0", "B4", "P", "P-no-parent"],
                            "temperature": 0,
                            "repeats": 1,
                            "max_tokens": 256,
                        },
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            manifest = freeze_public_tcm_qg_formal_prereg(
                config_path=config_path,
                env_path=env_path,
                output_path=output_path,
            )

            serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["stage"], "public_tcm_qg_formal_preregistered")
            self.assertEqual(manifest["retrieval_config_count"], 7)
            self.assertEqual(
                manifest["answer_methods"], ["B0", "B4", "P", "P-no-parent"]
            )
            self.assertEqual(manifest["answer_model"]["model_name"], "formal-answer-model")
            self.assertEqual(manifest["answer_model"]["base_url_origin"], "https://api.example.com")
            self.assertNotIn("secret", serialized)
            for forbidden in (
                "source_text",
                "question_text",
                "reference_answer",
                "answer_text",
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertTrue(output_path.is_file())

    def test_retrieval_validation_rejects_dataset_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text('{"qa_id":"q1"}\n', encoding="utf-8")
            prereg = root / "prereg.json"
            index_manifest = root / "indexes.json"
            atomic_write_json(
                prereg,
                {
                    "status": "ready",
                    "inputs": {
                        "dataset_sha256": "WRONG",
                    },
                },
            )
            atomic_write_json(
                index_manifest,
                {
                    "status": "ready",
                    "inputs": {
                        "prereg_manifest_sha256": sha256_file(prereg),
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "dataset sha256 mismatch"):
                validate_formal_retrieval_inputs(
                    dataset_path=dataset,
                    prereg_manifest_path=prereg,
                    index_manifest_path=index_manifest,
                )


class PublicTcmQgFormalRetrievalRunTests(unittest.TestCase):
    def test_formal_retrieval_matrix_writes_seven_config_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            text = "无症状胆囊结石可不作治疗。其他情况应区别处理。"
            answer_start = text.index("无症状胆囊结石可不作治疗")
            write_jsonl(
                dataset,
                [
                    {
                        "qa_id": "q1",
                        "source_doc_id": "d1",
                        "split": "dev",
                        "question": "什么胆囊结石可不作治疗？",
                        "answer": "无症状胆囊结石可不作治疗",
                        "source_text": text,
                        "answer_start": answer_start,
                        "answer_end": answer_start + len("无症状胆囊结石可不作治疗"),
                        "review_status": "approved",
                        "question_version": 1,
                    }
                ],
            )
            chunks_dir = root / "chunks"
            chunks_dir.mkdir()
            chunk = {
                "chunk_id": "c1",
                "strategy": "b4",
                "source_doc_id": "d1",
                "parent_id": "p1",
                "text": text,
                "context_text": text,
                "start_index": 0,
                "char_count": len(text),
                "context_start_index": 0,
                "context_char_count": len(text),
                "source_qa_ids": ["q1"],
            }
            write_jsonl(chunks_dir / "b4.jsonl", [chunk])
            write_jsonl(chunks_dir / "child.jsonl", [{**chunk, "chunk_id": "c2", "strategy": "child"}])
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
            index_manifest = root / "indexes.json"
            build_public_tcm_qg_formal_indexes(
                chunks_dir=chunks_dir,
                chunk_manifest_path=chunk_manifest,
                output_dir=root / "indexes",
                manifest_path=index_manifest,
                embedder=FakeEmbedder(),
            )

            result = run_public_tcm_qg_formal_retrieval_matrix(
                split="dev",
                dataset_path=dataset,
                indexes_dir=root / "indexes",
                output_dir=root / "runs",
                embedder=FakeEmbedder(),
                reranker=FakeReranker(),
                bm25_top_k=1,
                dense_top_k=1,
                reranker_candidate_k=1,
                final_top_k=1,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["config_count"], 7)
            per_question = root / "runs" / result["matrix_id"] / "p-public-no-parent" / "per-question.jsonl"
            self.assertTrue(per_question.is_file())
            row = json.loads(per_question.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["method_role"], "P-no-parent")
            self.assertEqual(row["hits"][0]["chunk_id"], "c2")


if __name__ == "__main__":
    unittest.main()
