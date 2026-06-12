import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_5.cli import main
from experiments.rag_v1_5.corpus import CorpusFileSpec


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class RagExperimentCliTests(unittest.TestCase):
    def test_prepares_and_parses_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            manifest_path = root / "corpus-v1.5.0.json"
            source_dir.mkdir()

            source_text = (
                FIXTURES_DIR / "shang_han_sample.txt"
            ).read_text(encoding="utf-8")
            source_bytes = source_text.encode("cp936")
            source_path = source_dir / "457-伤寒论.txt"
            source_path.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
            specs = [
                CorpusFileSpec(
                    book_id="shang_han_lun",
                    book_title="伤寒论",
                    source_filename=source_path.name,
                    expected_sha256=source_hash,
                )
            ]

            prepare_exit = main(
                [
                    "prepare-corpus",
                    "--source-dir",
                    str(source_dir),
                    "--raw-dir",
                    str(raw_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                corpus_specs=specs,
            )
            parse_exit = main(
                [
                    "parse-corpus",
                    "--raw-dir",
                    str(raw_dir),
                    "--manifest",
                    str(manifest_path),
                    "--processed-dir",
                    str(processed_dir),
                ]
            )

            self.assertEqual(prepare_exit, 0)
            self.assertEqual(parse_exit, 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((processed_dir / "evidence.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
