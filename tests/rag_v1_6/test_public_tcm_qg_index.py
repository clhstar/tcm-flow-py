import unittest

from experiments.rag_v1_6.public_tcm_qg_index import (
    build_public_tcm_qg_chunks_from_rows,
)


class PublicTcmQgIndexTests(unittest.TestCase):
    def test_chunks_include_b4_child_and_parent_context(self):
        text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
        rows = [
            {
                "qa_id": "tcmqg-1-000",
                "source_doc_id": "1",
                "split": "test",
                "question": "问题？",
                "answer": "庚辛",
                "source_text": text,
                "answer_start": text.index("庚辛"),
                "answer_end": text.index("庚辛") + 2,
                "review_status": "approved",
                "question_version": 1,
            }
        ]

        chunks = build_public_tcm_qg_chunks_from_rows(
            rows=rows,
            b4_chunk_size=10,
            b4_chunk_overlap=2,
            child_chunk_size=6,
            child_chunk_overlap=1,
        )

        self.assertEqual(set(chunks), {"b4", "child"})
        self.assertGreater(len(chunks["child"]), len(chunks["b4"]))
        self.assertEqual(chunks["child"][0]["context_text"], text)


if __name__ == "__main__":
    unittest.main()
