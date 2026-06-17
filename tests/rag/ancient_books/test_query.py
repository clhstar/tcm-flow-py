import unittest

from app.rag.ancient_books.query import detect_chief_symptom, rewrite_query


class QueryTests(unittest.TestCase):
    def test_routes_all_supported_user_phrases(self):
        cases = {
            "头疼两天": "头痛",
            "感觉天旋地转": "眩晕",
            "晚上一直咳": "咳嗽",
            "活动后气喘": "喘促",
            "心里怦怦跳": "心悸",
            "睡不着而且容易醒": "不寐",
            "胃疼饭后明显": "胃脘痛",
            "肚子痛": "腹痛",
            "最近拉肚子": "泄泻",
            "大便干结难解": "便秘",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_chief_symptom(text), expected)

    def test_rewrite_adds_aliases_but_not_unreported_syndromes(self):
        rewritten = rewrite_query("饭后胃疼，喜按")

        self.assertIn("胃脘痛", rewritten)
        self.assertIn("心痛", rewritten)
        self.assertNotIn("脾胃虚弱", rewritten)
        self.assertNotIn("肝郁气滞", rewritten)

    def test_unknown_symptom_is_not_invented(self):
        query = "最近总觉得疲劳"

        self.assertIsNone(detect_chief_symptom(query))
        self.assertEqual(rewrite_query(query), query)


if __name__ == "__main__":
    unittest.main()
