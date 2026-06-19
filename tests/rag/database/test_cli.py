import unittest

from app.rag.database.cli import build_parser


class DatabaseCliTests(unittest.TestCase):
    def test_parser_supports_import_doctor_and_smoke(self):
        parser = build_parser()

        import_args = parser.parse_args(
            ["import-artifacts", "--corpus-dir", "c", "--index-dir", "i"]
        )
        doctor_args = parser.parse_args(["doctor"])
        smoke_args = parser.parse_args(["smoke"])

        self.assertEqual(import_args.command, "import-artifacts")
        self.assertEqual(import_args.corpus_dir, "c")
        self.assertEqual(import_args.index_dir, "i")
        self.assertEqual(doctor_args.command, "doctor")
        self.assertEqual(smoke_args.command, "smoke")


if __name__ == "__main__":
    unittest.main()
