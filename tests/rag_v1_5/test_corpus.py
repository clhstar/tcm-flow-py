import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.corpus import CorpusFileSpec, prepare_corpus


class CorpusImportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_dir = self.root / "source"
        self.output_dir = self.root / "output"
        self.manifest_path = self.root / "manifests" / "corpus-v1.5.0.json"
        self.source_dir.mkdir()

        self.source_text = (
            "<篇名>伤寒论\r\n"
            "<目录>\r\n"
            "<篇名>辨太阳病脉证并治上\r\n"
            "属性：1．太阳之为病，脉浮、头项强痛而恶寒。\r\n"
        )
        self.source_bytes = self.source_text.encode("cp936")
        self.source_path = self.source_dir / "457-伤寒论.txt"
        self.source_path.write_bytes(self.source_bytes)
        self.source_sha256 = hashlib.sha256(self.source_bytes).hexdigest().upper()
        self.spec = CorpusFileSpec(
            book_id="shang_han_lun",
            book_title="伤寒论",
            source_filename=self.source_path.name,
            expected_sha256=self.source_sha256,
        )
        self.generated_at = datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_imports_cp936_as_utf8_without_modifying_source(self):
        source_before = self.source_path.read_bytes()

        manifest = prepare_corpus(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            manifest_path=self.manifest_path,
            specs=[self.spec],
            generated_at=self.generated_at,
        )

        output_path = self.output_dir / self.source_path.name
        self.assertEqual(output_path.read_bytes().decode("utf-8"), self.source_text)
        self.assertEqual(self.source_path.read_bytes(), source_before)
        self.assertEqual(manifest.corpus_version, "v1.5.0")
        self.assertEqual(manifest.files[0].source_encoding, "cp936")
        self.assertEqual(manifest.files[0].output_encoding, "utf-8")
        self.assertEqual(manifest.files[0].source_sha256, self.source_sha256)
        self.assertEqual(
            manifest.generated_at,
            self.generated_at,
        )

    def test_rejects_source_hash_mismatch_before_writing_output(self):
        invalid_spec = self.spec.model_copy(
            update={"expected_sha256": "0" * 64}
        )

        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_corpus(
                source_dir=self.source_dir,
                output_dir=self.output_dir,
                manifest_path=self.manifest_path,
                specs=[invalid_spec],
                generated_at=self.generated_at,
            )

        self.assertFalse(self.output_dir.exists())
        self.assertFalse(self.manifest_path.exists())

    def test_repeated_import_produces_identical_utf8_content_and_manifest(self):
        first_manifest = prepare_corpus(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            manifest_path=self.manifest_path,
            specs=[self.spec],
            generated_at=self.generated_at,
        )
        first_output = (self.output_dir / self.source_path.name).read_bytes()
        first_manifest_bytes = self.manifest_path.read_bytes()

        second_manifest = prepare_corpus(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            manifest_path=self.manifest_path,
            specs=[self.spec],
            generated_at=self.generated_at,
        )

        self.assertEqual(
            (self.output_dir / self.source_path.name).read_bytes(),
            first_output,
        )
        self.assertEqual(self.manifest_path.read_bytes(), first_manifest_bytes)
        self.assertEqual(second_manifest, first_manifest)

        raw_manifest = json.loads(first_manifest_bytes.decode("utf-8"))
        self.assertEqual(
            raw_manifest["source"]["commit"],
            "db0155dc7c42b9c6b3736896661f317c7110038f",
        )
        self.assertEqual(
            raw_manifest["source"]["commit_verification"],
            "declared_not_locally_verified",
        )
        self.assertEqual(
            raw_manifest["source"]["license_status"],
            "not_declared_in_local_snapshot",
        )
        self.assertEqual(raw_manifest["files"][0]["output_file"], self.source_path.name)


if __name__ == "__main__":
    unittest.main()
