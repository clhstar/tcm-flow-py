import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.rag_v1_5.indexing import build_strategy_index
from experiments.rag_v1_5.schema import RetrievalHit
from tests.rag_v1_5.test_indexing import FakeEncoder, load_chunks


def make_hit(
    chunk_id: str,
    rank: int,
    *,
    text: str | None = None,
    context_text: str | None = None,
    parent_id: str | None = None,
    bm25_score: float | None = None,
    dense_score: float | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        strategy="c4",
        rank=rank,
        text=text or chunk_id,
        context_text=context_text or f"parent:{chunk_id}",
        source_evidence_ids=[f"evidence:{chunk_id}"],
        clause_ids=[parent_id or f"clause:{chunk_id}"],
        retrieval_parent_id=parent_id,
        bm25_rank=rank if bm25_score is not None else None,
        bm25_score=bm25_score,
        dense_rank=rank if dense_score is not None else None,
        dense_score=dense_score,
    )


class RetrievalTests(unittest.TestCase):
    def retrieval_module(self):
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.retrieval"
        )
        self.assertIsNotNone(module_spec, "retrieval module is not implemented")
        return importlib.import_module("experiments.rag_v1_5.retrieval")

    def create_index(self, root: Path):
        chunks = load_chunks()
        chunks = [
            chunks[0].model_copy(update={"text": "共同 甲"}),
            chunks[1].model_copy(update={"text": "共同 乙"}),
            chunks[2].model_copy(update={"text": "其他 丙"}),
        ]
        build_strategy_index(
            chunks=chunks,
            output_dir=root,
            encoder=FakeEncoder(
                np.asarray(
                    [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
                    dtype=np.float32,
                )
            ),
            quality_gate_sha256="A" * 64,
            chunk_sha256="B" * 64,
            model_record={
                "model": "BAAI/bge-m3",
                "revision": "1" * 40,
                "local_path": "data/models/bge-m3",
            },
        )
        return chunks

    def create_large_index(self, root: Path, count: int = 12):
        template = load_chunks()[0]
        chunks = []
        vectors = []
        for index in range(1, count + 1):
            clause_id = f"jgy-chapter-25-{index:03d}"
            chunks.append(
                template.model_copy(
                    update={
                        "chunk_id": f"c4-{index:03d}",
                        "clause_id": clause_id,
                        "retrieval_parent_id": clause_id,
                        "source_evidence_ids": [clause_id],
                        "text": f"共同 测试 {index}",
                        "context_text": f"完整父级上下文 {index}",
                    }
                )
            )
            vectors.append([float(count - index + 1), float(index)])
        build_strategy_index(
            chunks=chunks,
            output_dir=root,
            encoder=FakeEncoder(np.asarray(vectors, dtype=np.float32)),
            quality_gate_sha256="A" * 64,
            chunk_sha256="B" * 64,
            model_record={
                "model": "BAAI/bge-m3",
                "revision": "1" * 40,
                "local_path": "data/models/bge-m3",
            },
        )
        return chunks

    def retrieval_config(self) -> dict:
        return {
            "bm25": {"top_k": 20},
            "dense": {"top_k": 20},
            "rrf": {"k": 60},
            "reranker": {"candidate_k": 40, "top_k": 5},
        }

    def test_bm25_and_dense_sort_scores_with_stable_ties(self) -> None:
        retrieval = self.retrieval_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            self.create_index(index_dir)
            index = retrieval.load_index(index_dir)

            bm25_hits = retrieval.search_bm25(
                index,
                "不存在",
                top_k=20,
            )
            dense_hits = retrieval.search_dense(
                index,
                np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=20,
            )

        self.assertEqual(len(bm25_hits), 3)
        self.assertEqual(
            [hit.chunk_id for hit in bm25_hits[:2]],
            sorted(hit.chunk_id for hit in bm25_hits[:2]),
        )
        self.assertGreaterEqual(
            bm25_hits[0].bm25_score,
            bm25_hits[1].bm25_score,
        )
        self.assertEqual(
            [hit.chunk_id for hit in dense_hits],
            ["c4-001", "c4-003", "c4-002"],
        )
        self.assertGreater(
            dense_hits[0].dense_score,
            dense_hits[1].dense_score,
        )
        self.assertEqual(
            {hit.chunk_id for hit in bm25_hits},
            {hit.chunk_id for hit in dense_hits},
        )

    def test_empty_query_and_corrupted_index_are_rejected(self) -> None:
        retrieval = self.retrieval_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            self.create_index(index_dir)
            index = retrieval.load_index(index_dir)
            with self.assertRaises(ValueError):
                retrieval.search_bm25(index, " \n ", top_k=5)
            (index_dir / "rows.jsonl").write_text(
                "corrupted\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                retrieval.load_index(index_dir)

    def test_rrf_matches_hand_calculation_and_preserves_source_scores(self) -> None:
        retrieval = self.retrieval_module()
        bm25 = [
            make_hit("a", 1, bm25_score=3.0),
            make_hit("b", 2, bm25_score=2.0),
            make_hit("c", 3, bm25_score=1.0),
        ]
        dense = [
            make_hit("b", 1, dense_score=0.9),
            make_hit("d", 2, dense_score=0.8),
            make_hit("a", 3, dense_score=0.7),
        ]

        fused = retrieval.reciprocal_rank_fusion(
            {"bm25": bm25, "dense": dense},
            k=60,
        )
        scores = {hit.chunk_id: hit.rrf_score for hit in fused}

        expected_a = 1 / 61 + 1 / 63
        expected_b = 1 / 62 + 1 / 61
        self.assertGreater(scores["b"], scores["a"])
        self.assertAlmostEqual(scores["a"], expected_a)
        self.assertAlmostEqual(scores["b"], expected_b)
        fused_by_id = {hit.chunk_id: hit for hit in fused}
        self.assertEqual(fused_by_id["a"].bm25_rank, 1)
        self.assertEqual(fused_by_id["a"].dense_rank, 3)
        self.assertEqual(fused_by_id["b"].bm25_score, 2.0)
        self.assertEqual(fused_by_id["b"].dense_score, 0.9)

    def test_retrieve_loaded_reports_fixed_stage_latency_without_reloading(
        self,
    ) -> None:
        retrieval = self.retrieval_module()
        latency_keys = {
            "bm25_ms",
            "dense_ms",
            "rrf_ms",
            "reranker_ms",
            "total_ms",
            "returned_context_chars",
        }
        enabled_stages = {
            "bm25": {"bm25_ms"},
            "dense": {"dense_ms"},
            "hybrid": {"bm25_ms", "dense_ms", "rrf_ms"},
            "bm25_rerank": {"bm25_ms", "reranker_ms"},
            "dense_rerank": {"dense_ms", "reranker_ms"},
            "hybrid_rerank": {
                "bm25_ms",
                "dense_ms",
                "rrf_ms",
                "reranker_ms",
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            self.create_index(index_dir)
            index = retrieval.load_index(index_dir)
            for mode in enabled_stages:
                with self.subTest(mode=mode):
                    encoder = FakeEncoder(
                        np.asarray([[1.0, 0.0]], dtype=np.float32)
                    )
                    scorer = DynamicReranker()
                    with patch.object(
                        retrieval,
                        "load_index",
                        side_effect=AssertionError(
                            "retrieve_loaded 不应重新加载索引"
                        ),
                    ):
                        result = retrieval.retrieve_loaded(
                            "共同",
                            index=index,
                            mode=mode,
                            config=self.retrieval_config(),
                            dense_encoder=(
                                encoder
                                if mode
                                in {
                                    "dense",
                                    "hybrid",
                                    "dense_rerank",
                                    "hybrid_rerank",
                                }
                                else None
                            ),
                            reranker_scorer=(
                                scorer
                                if mode
                                in {
                                    "bm25_rerank",
                                    "dense_rerank",
                                    "hybrid_rerank",
                                }
                                else None
                            ),
                        )

                    self.assertEqual(set(result.latency), latency_keys)
                    for key in latency_keys:
                        self.assertGreaterEqual(result.latency[key], 0)
                    for stage in {
                        "bm25_ms",
                        "dense_ms",
                        "rrf_ms",
                        "reranker_ms",
                    } - enabled_stages[mode]:
                        self.assertEqual(result.latency[stage], 0.0)
                    stage_sum = sum(
                        result.latency[key]
                        for key in (
                            "bm25_ms",
                            "dense_ms",
                            "rrf_ms",
                            "reranker_ms",
                        )
                    )
                    self.assertGreaterEqual(
                        result.latency["total_ms"] + 0.1,
                        stage_sum,
                    )
                    self.assertEqual(
                        result.latency["returned_context_chars"],
                        sum(
                            len(hit.context_text)
                            for hit in result.hits[:5]
                        ),
                    )
                    for hit in result.hits:
                        self.assertTrue(hit.context_text)
                        self.assertTrue(hit.clause_ids)
                        self.assertTrue(hit.source_evidence_ids)
                        self.assertTrue(hit.retrieval_parent_id)

    def test_context_policy_changes_only_returned_context(self) -> None:
        retrieval = self.retrieval_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            self.create_index(index_dir)
            index = retrieval.load_index(index_dir)

            parent = retrieval.retrieve_loaded(
                "鍏卞悓",
                index=index,
                mode="hybrid_rerank",
                config=self.retrieval_config(),
                dense_encoder=FakeEncoder(
                    np.asarray([[1.0, 0.0]], dtype=np.float32)
                ),
                reranker_scorer=DynamicReranker(),
                context_policy="parent",
            )
            child = retrieval.retrieve_loaded(
                "鍏卞悓",
                index=index,
                mode="hybrid_rerank",
                config=self.retrieval_config(),
                dense_encoder=FakeEncoder(
                    np.asarray([[1.0, 0.0]], dtype=np.float32)
                ),
                reranker_scorer=DynamicReranker(),
                context_policy="child",
            )

        self.assertEqual(
            [hit.chunk_id for hit in parent.hits],
            [hit.chunk_id for hit in child.hits],
        )
        self.assertEqual(
            [hit.rank for hit in parent.hits],
            [hit.rank for hit in child.hits],
        )
        self.assertEqual(
            [hit.reranker_score for hit in parent.hits],
            [hit.reranker_score for hit in child.hits],
        )
        self.assertTrue(
            all(hit.context_text == hit.text for hit in child.hits)
        )
        self.assertTrue(
            any(
                parent_hit.context_text != child_hit.context_text
                for parent_hit, child_hit in zip(
                    parent.hits,
                    child.hits,
                )
            )
        )

    def test_retrieve_loaded_matches_legacy_order_and_supports_top_10(
        self,
    ) -> None:
        retrieval = self.retrieval_module()
        config = self.retrieval_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index_dir = root / "c4"
            self.create_large_index(index_dir)
            index = retrieval.load_index(index_dir)

            for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                with self.subTest(mode=mode):
                    legacy_encoder = FakeEncoder(
                        np.asarray([[1.0, 0.0]], dtype=np.float32)
                    )
                    loaded_encoder = FakeEncoder(
                        np.asarray([[1.0, 0.0]], dtype=np.float32)
                    )
                    legacy_scorer = DynamicReranker()
                    loaded_scorer = DynamicReranker()
                    legacy_hits = retrieval.retrieve(
                        "共同",
                        strategy="c4",
                        mode=mode,
                        indexes_dir=root,
                        config=config,
                        dense_encoder=(
                            legacy_encoder
                            if mode
                            in {"dense", "hybrid", "hybrid_rerank"}
                            else None
                        ),
                        reranker_scorer=(
                            legacy_scorer
                            if mode == "hybrid_rerank"
                            else None
                        ),
                    )
                    default_result = retrieval.retrieve_loaded(
                        "共同",
                        index=index,
                        mode=mode,
                        config=config,
                        dense_encoder=(
                            loaded_encoder
                            if mode
                            in {"dense", "hybrid", "hybrid_rerank"}
                            else None
                        ),
                        reranker_scorer=(
                            loaded_scorer
                            if mode == "hybrid_rerank"
                            else None
                        ),
                    )
                    top10_result = retrieval.retrieve_loaded(
                        "共同",
                        index=index,
                        mode=mode,
                        config=config,
                        dense_encoder=(
                            FakeEncoder(
                                np.asarray(
                                    [[1.0, 0.0]],
                                    dtype=np.float32,
                                )
                            )
                            if mode
                            in {"dense", "hybrid", "hybrid_rerank"}
                            else None
                        ),
                        reranker_scorer=(
                            DynamicReranker()
                            if mode == "hybrid_rerank"
                            else None
                        ),
                        result_top_k=10,
                    )

                    self.assertEqual(len(legacy_hits), 5)
                    self.assertEqual(len(default_result.hits), 5)
                    self.assertEqual(len(top10_result.hits), 10)
                    self.assertEqual(
                        [hit.chunk_id for hit in legacy_hits],
                        [
                            hit.chunk_id
                            for hit in default_result.hits
                        ],
                    )
                    self.assertEqual(
                        [
                            hit.chunk_id
                            for hit in default_result.hits
                        ],
                        [
                            hit.chunk_id
                            for hit in top10_result.hits[:5]
                        ],
                    )


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def score(self, pairs):
        self.pairs = pairs
        return self.scores


class DynamicReranker:
    def __init__(self):
        self.pairs = None

    def score(self, pairs):
        self.pairs = pairs
        return [
            float(len(pairs) - index)
            for index in range(len(pairs))
        ]


class RerankerTests(unittest.TestCase):
    def reranker_module(self):
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.reranker"
        )
        self.assertIsNotNone(module_spec, "reranker module is not implemented")
        return importlib.import_module("experiments.rag_v1_5.reranker")

    def test_scores_child_text_and_preserves_shared_parent_identity(self) -> None:
        reranker = self.reranker_module()
        hits = [
            make_hit(
                "child-a",
                1,
                text="child text A",
                context_text="full shared parent",
                parent_id="parent-1",
            ),
            make_hit(
                "child-b",
                2,
                text="child text B",
                context_text="full shared parent",
                parent_id="parent-1",
            ),
        ]
        fake = FakeReranker([0.2, 0.9])

        ranked = reranker.rerank_hits(
            "query",
            hits,
            scorer=fake,
            top_k=5,
        )

        self.assertEqual(
            fake.pairs,
            [["query", "child text A"], ["query", "child text B"]],
        )
        self.assertEqual(
            [hit.chunk_id for hit in ranked],
            ["child-b", "child-a"],
        )
        self.assertEqual(
            [hit.retrieval_parent_id for hit in ranked],
            ["parent-1", "parent-1"],
        )
        self.assertEqual(
            [hit.context_text for hit in ranked],
            ["full shared parent", "full shared parent"],
        )
        self.assertEqual(
            [hit.reranker_score for hit in ranked],
            [0.9, 0.2],
        )

    def test_normalizes_single_score_and_rejects_score_count_mismatch(self) -> None:
        reranker = self.reranker_module()
        single = reranker.rerank_hits(
            "query",
            [make_hit("only", 1)],
            scorer=FakeReranker(0.7),
            top_k=5,
        )
        self.assertEqual(single[0].reranker_score, 0.7)
        with self.assertRaises(ValueError):
            reranker.rerank_hits(
                "query",
                [make_hit("a", 1), make_hit("b", 2)],
                scorer=FakeReranker([0.5]),
                top_k=5,
            )


if __name__ == "__main__":
    unittest.main()
