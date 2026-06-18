import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.ancient_books.cli import SMOKE_QUERIES, build_parser, run_smoke
from app.rag.ancient_books.pipeline import (
    build_corpus,
    doctor_corpus,
    export_manifests,
)


class PipelineTests(unittest.TestCase):
    def test_export_manifests_rejects_raw_content_keys(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "content"):
                export_manifests(
                    corpus_manifest={"status": "ready", "content": "raw"},
                    index_manifest={"status": "ready"},
                    output_dir=output_dir,
                )
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_smoke_submits_all_ten_queries_and_rejects_degradation(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            def retrieve(self, query, *, chief_symptom, mode, top_k):
                self.calls.append((chief_symptom, query, mode, top_k))
                return {
                    "status": "ok",
                    "degraded": False,
                    "results": [{"symptom_tags": [chief_symptom]}],
                }

        engine = FakeEngine()
        result = run_smoke(engine)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["query_count"], 10)
        self.assertEqual(result["ok_count"], 10)
        self.assertEqual(result["insufficient_symptoms"], [])
        self.assertEqual(
            [(symptom, query) for symptom, query, _, _ in engine.calls],
            list(SMOKE_QUERIES.items()),
        )

        class DegradedEngine(FakeEngine):
            def retrieve(self, query, *, chief_symptom, mode, top_k):
                result = super().retrieve(
                    query,
                    chief_symptom=chief_symptom,
                    mode=mode,
                    top_k=top_k,
                )
                result["degraded"] = True
                return result

        with self.assertRaisesRegex(RuntimeError, "降级"):
            run_smoke(DegradedEngine())

    def test_build_corpus_writes_deterministic_single_book_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            body = (
                "<目录>卷一\\头痛\n"
                "<篇名>头痛\n"
                "属性：因风者恶风。\n"
            )
            (source / "637-景岳全书.txt").write_bytes(body.encode("cp936"))
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
            first_output = root / "first"
            second_output = root / "second"

            manifest = build_corpus(
                config=config,
                source_root=source,
                output_dir=first_output,
            )
            build_corpus(
                config=config,
                source_root=source,
                output_dir=second_output,
            )

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["book_count"], 1)
            self.assertEqual(
                manifest["sources"][0]["source_file"],
                "637-景岳全书.txt",
            )
            for filename in (
                "sections.jsonl",
                "parents.jsonl",
                "chunks.jsonl",
                "manifest.json",
            ):
                self.assertEqual(
                    (first_output / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )
            rows = [
                json.loads(line)
                for line in (first_output / "chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(rows[0]["symptom_tags"], ["头痛"])

            doctor = doctor_corpus(first_output)
            self.assertEqual(
                doctor,
                {
                    "status": "ready",
                    "source_hash_mismatch_count": 0,
                    "artifact_hash_mismatch_count": 0,
                    "count_mismatch_count": 0,
                    "duplicate_parent_count": 0,
                    "duplicate_chunk_count": 0,
                    "orphan_chunk_count": 0,
                    "excluded_content_match_count": 0,
                },
            )

    def test_build_corpus_cli_has_no_curated_markdown_argument(self):
        args = build_parser().parse_args(
            ["build-corpus", "--source-root", "data/source"]
        )

        self.assertFalse(hasattr(args, "curated_root"))


if __name__ == "__main__":
    unittest.main()
