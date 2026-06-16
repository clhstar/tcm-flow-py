import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from experiments.rag_v1_6.public_tcm_qg import (
    build_doc_split,
    freeze_public_tcm_qg_source,
    normalize_public_tcm_qg_rows,
)
from experiments.rag_v1_6.schema import PublicTcmQgDocument, PublicTcmQgQaPair


SAMPLE_TEXT = "胆囊结石的治疗应区别不同情况分别处理。无症状胆囊结石可不作治疗。"
SAMPLE_ANSWER = "无症状胆囊结石可不作治疗"


class PublicTcmQgSchemaTests(unittest.TestCase):
    def test_document_requires_annotations(self):
        doc = PublicTcmQgDocument(
            source_doc_id="1240",
            text=SAMPLE_TEXT,
            annotations=[
                {
                    "Q": "什么类型的胆囊结石可不作治疗？",
                    "A": SAMPLE_ANSWER,
                }
            ],
        )

        self.assertEqual(doc.source_doc_id, "1240")
        self.assertEqual(len(doc.annotations), 1)

    def test_qa_pair_requires_answer_span(self):
        start = SAMPLE_TEXT.index(SAMPLE_ANSWER)
        pair = PublicTcmQgQaPair(
            qa_id="tcmqg-1240-000",
            source_doc_id="1240",
            split="dev",
            question="什么类型的胆囊结石可不作治疗？",
            answer=SAMPLE_ANSWER,
            source_text=SAMPLE_TEXT,
            answer_start=start,
            answer_end=start + len(SAMPLE_ANSWER),
            review_status="approved",
        )

        self.assertEqual(pair.answer, SAMPLE_ANSWER)

        with self.assertRaises(ValidationError):
            PublicTcmQgQaPair(
                qa_id="tcmqg-1240-001",
                source_doc_id="1240",
                split="test",
                question="错误问题",
                answer=SAMPLE_ANSWER,
                source_text=SAMPLE_TEXT,
                answer_start=0,
                answer_end=len(SAMPLE_ANSWER),
                review_status="approved",
            )


class PublicTcmQgSourceTests(unittest.TestCase):
    def test_source_manifest_excludes_raw_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "train.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "text": SAMPLE_TEXT,
                            "annotations": [
                                {
                                    "Q": "什么类型的胆囊结石可不作治疗？",
                                    "A": SAMPLE_ANSWER,
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = root / "source-manifest.json"

            manifest = freeze_public_tcm_qg_source(
                source_path=source,
                output_path=output,
                public_dataset_url="https://tianchi.aliyun.com/dataset/86895",
            )
            serialized = output.read_text(encoding="utf-8")

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["document_count"], 1)
            self.assertEqual(manifest["qa_pair_count"], 1)
            self.assertNotIn(SAMPLE_TEXT, serialized)
            self.assertNotIn(SAMPLE_ANSWER, serialized)


class PublicTcmQgNormalizeTests(unittest.TestCase):
    def test_normalize_keeps_answer_substring_and_doc_split(self):
        rows = [
            {
                "id": 1240,
                "text": SAMPLE_TEXT,
                "annotations": [
                    {
                        "Q": "什么类型的胆囊结石可不作治疗？",
                        "A": SAMPLE_ANSWER,
                    }
                ],
            }
        ]

        normalized = normalize_public_tcm_qg_rows(
            rows=rows,
            split_by_doc_id={"1240": "test"},
            min_text_chars=10,
            max_text_chars=100,
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["split"], "test")
        self.assertEqual(normalized[0]["answer_start"], SAMPLE_TEXT.index(SAMPLE_ANSWER))
        self.assertEqual(
            normalized[0]["source_text"][
                normalized[0]["answer_start"] : normalized[0]["answer_end"]
            ],
            normalized[0]["answer"],
        )

    def test_normalize_drops_answer_not_in_text(self):
        rows = [
            {
                "id": 1,
                "text": "原文不含目标答案。",
                "annotations": [{"Q": "问题？", "A": "缺失答案"}],
            }
        ]

        normalized = normalize_public_tcm_qg_rows(
            rows=rows,
            split_by_doc_id={"1": "dev"},
            min_text_chars=1,
            max_text_chars=100,
        )

        self.assertEqual(normalized, [])

    def test_doc_split_is_doc_level(self):
        split = build_doc_split(
            doc_ids=["3", "1", "2", "4"],
            seed=7,
            dev_rate=0.25,
            test_rate=0.25,
        )

        self.assertEqual(set(split), {"1", "2", "3", "4"})
        self.assertEqual(list(split.values()).count("dev"), 1)
        self.assertEqual(list(split.values()).count("test"), 1)


if __name__ == "__main__":
    unittest.main()
