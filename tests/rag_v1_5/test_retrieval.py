import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

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


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def score(self, pairs):
        self.pairs = pairs
        return self.scores


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
