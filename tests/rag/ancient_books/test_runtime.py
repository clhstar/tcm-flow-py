import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.ancient_books.indexing import build_index
from app.rag.ancient_books.pipeline import build_corpus, sha256_file
from app.rag.ancient_books.runtime import (
    LoadedProductionIndex,
    ProductionRetrievalEngine,
    load_index,
    reciprocal_rank_fusion,
    recover_parents,
)
from app.rag.ancient_books.schema import RetrievalChunk


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FailingEncoder:
    def encode(self, texts):
        raise RuntimeError("model unavailable")


class FakeReranker:
    def score(self, pairs):
        return [1.0 - index / 10 for index, _ in enumerate(pairs)]


def loaded_index() -> LoadedProductionIndex:
    rows = [
        RetrievalChunk(
            chunk_id="c1",
            parent_id="p1",
            text="头痛恶风",
            source_type="ancient_book",
            symptom_tags=["头痛"],
            evidence_role="syndrome_pattern",
        ),
        RetrievalChunk(
            chunk_id="c2",
            parent_id="p1",
            text="头痛遇冷加重",
            source_type="ancient_book",
            symptom_tags=["头痛"],
            evidence_role="syndrome_pattern",
        ),
        RetrievalChunk(
            chunk_id="c3",
            parent_id="p2",
            text="咳嗽有痰",
            source_type="ancient_book",
            symptom_tags=["咳嗽"],
            evidence_role="syndrome_pattern",
        ),
    ]
    parents = {
        "p1": {
            "parent_id": "p1",
            "book_title": "景岳全书",
            "source_file": "637-景岳全书.txt",
            "volume": "卷之一",
            "chapter": "头痛",
            "section": "头痛论",
            "original_text": "头痛恶风，遇冷加重。",
            "normalized_text": "头痛恶风，遇冷加重。",
            "source_type": "ancient_book",
            "symptom_tags": ["头痛"],
            "evidence_role": "syndrome_pattern",
        },
        "p2": {
            "parent_id": "p2",
            "book_title": "景岳全书",
            "source_file": "637-景岳全书.txt",
            "volume": "卷之二",
            "chapter": "咳嗽",
            "section": "咳嗽论",
            "original_text": "咳嗽有痰。",
            "normalized_text": "咳嗽有痰。",
            "source_type": "ancient_book",
            "symptom_tags": ["咳嗽"],
            "evidence_role": "syndrome_pattern",
        },
    }
    return LoadedProductionIndex(
        rows=rows,
        row_by_id={row.chunk_id: row for row in rows},
        parents=parents,
        bm25=BM25Okapi([["头痛", "恶风"], ["头痛", "遇冷"], ["咳嗽", "痰"]]),
        dense=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        manifest={"status": "ready"},
    )


class RuntimeTests(unittest.TestCase):
    def test_load_index_verifies_corpus_parent_hash(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "637-景岳全书.txt").write_bytes(
                (
                    "<目录>卷一\\头痛\n"
                    "<篇名>头痛\n"
                    "属性：因风者恶风。\n"
                ).encode("cp936")
            )
            config = {
                "version": "v1.0.0",
                "source_encoding": "cp936",
                "symptoms": {"头痛": ["头痛"]},
                "exclude_title_patterns": [],
                "books": [{
                    "book_id": "jing_yue_quan_shu",
                    "title": "景岳全书",
                    "source_file": "637-景岳全书.txt",
                    "symptom_scan": True,
                    "method_sections": [],
                    "fixed_sections": [],
                }],
            }
            corpus_dir = root / "corpus"
            index_dir = root / "index"
            build_corpus(
                config=config,
                source_root=source,
                output_dir=corpus_dir,
            )
            build_index(
                chunks_path=corpus_dir / "chunks.jsonl",
                corpus_manifest_sha256=sha256_file(corpus_dir / "manifest.json"),
                output_dir=index_dir,
                encoder=FakeEncoder(),
                model_record={"model": "fake", "revision": "1" * 40},
            )

            loaded = load_index(index_dir, corpus_dir)
            self.assertEqual(len(loaded.rows), 1)

            parents_path = corpus_dir / "parents.jsonl"
            parents_path.write_text(
                parents_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "大小不匹配|SHA256"):
                load_index(index_dir, corpus_dir)

    def test_rrf_merges_bm25_and_dense_with_stable_ties(self):
        merged = reciprocal_rank_fusion(
            {"bm25": ["c2", "c1"], "dense": ["c1", "c3"]},
            rrf_k=60,
        )
        self.assertEqual(merged[0][0], "c1")
        self.assertEqual(
            [chunk_id for chunk_id, _ in merged[1:]],
            sorted(chunk_id for chunk_id, _ in merged[1:]),
        )

    def test_parent_recovery_deduplicates_multiple_children(self):
        hits = [
            {"chunk_id": "c1", "parent_id": "p1", "score": 0.9},
            {"chunk_id": "c2", "parent_id": "p1", "score": 0.8},
            {"chunk_id": "c3", "parent_id": "p2", "score": 0.7},
        ]
        parents = {"p1": {"parent_id": "p1"}, "p2": {"parent_id": "p2"}}

        recovered = recover_parents(hits, parents)

        self.assertEqual([row["parent_id"] for row in recovered], ["p1", "p2"])
        self.assertEqual(recovered[0]["chunk_id"], "c1")

    def test_hybrid_filters_by_symptom_and_assigns_citations(self):
        engine = ProductionRetrievalEngine(
            index=loaded_index(),
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            settings={
                "bm25_top_k": 20,
                "dense_top_k": 20,
                "rrf_k": 60,
                "reranker_candidate_k": 40,
                "final_top_k": 5,
            },
        )

        result = engine.retrieve("头痛恶风", chief_symptom="头痛")

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["degraded"])
        self.assertEqual(result["results"][0]["citation_id"], "E1")
        self.assertTrue(
            all("头痛" in row["symptom_tags"] for row in result["results"])
        )
        self.assertEqual(result["results"][0]["matched_child"], "头痛恶风")

    def test_dense_failure_is_an_explicit_keyword_degradation(self):
        engine = ProductionRetrievalEngine(
            index=loaded_index(),
            encoder=FailingEncoder(),
            reranker=FakeReranker(),
            settings={
                "bm25_top_k": 20,
                "dense_top_k": 20,
                "rrf_k": 60,
                "reranker_candidate_k": 40,
                "final_top_k": 5,
            },
        )

        result = engine.retrieve("头痛恶风", chief_symptom="头痛")

        self.assertTrue(result["degraded"])
        self.assertEqual(result["retrieval_mode"], "keyword")
        self.assertIn("model unavailable", result["degraded_reason"])


if __name__ == "__main__":
    unittest.main()
