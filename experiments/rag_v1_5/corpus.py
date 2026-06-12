import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CORPUS_VERSION = "v1.5.0"
SOURCE_URL = "https://github.com/xiaopangxia/TCM-Ancient-Books"
SOURCE_COMMIT = "db0155dc7c42b9c6b3736896661f317c7110038f"


class CorpusFileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    book_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    expected_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")

    @field_validator("expected_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.upper()


class CorpusSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    commit: str
    commit_verification: Literal["declared_not_locally_verified", "verified"]
    license_status: Literal[
        "not_declared_in_local_snapshot",
        "declared",
    ]


class CorpusFileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    book_id: str
    book_title: str
    source_file: str
    source_sha256: str
    source_bytes: int
    source_encoding: str
    output_file: str
    output_sha256: str
    output_bytes: int
    output_encoding: str


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: str
    generated_at: datetime
    source: CorpusSource
    files: list[CorpusFileManifest]


DEFAULT_CORPUS_SPECS = [
    CorpusFileSpec(
        book_id="shang_han_lun",
        book_title="伤寒论",
        source_filename="457-伤寒论.txt",
        expected_sha256="EF2FDFA298F1367B9E7501E7C868C6BCDFE8A3ACD7C4991C9262857C606BF462",
    ),
    CorpusFileSpec(
        book_id="jin_gui_yao_lue",
        book_title="金匮要略方论",
        source_filename="499-金匮要略方论.txt",
        expected_sha256="617250F7522DA17132A97D7FE6AFD9B128F442E3980163DFAE836C1C98663F7C",
    ),
]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def prepare_corpus(
    source_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    specs: Iterable[CorpusFileSpec] = DEFAULT_CORPUS_SPECS,
    generated_at: datetime | None = None,
) -> CorpusManifest:
    prepared_files: list[tuple[CorpusFileSpec, bytes, bytes]] = []

    for spec in specs:
        source_path = source_dir / spec.source_filename
        if not source_path.is_file():
            raise FileNotFoundError(f"缺少语料文件: {source_path}")

        source_bytes = source_path.read_bytes()
        source_sha256 = sha256_bytes(source_bytes)
        if source_sha256 != spec.expected_sha256:
            raise ValueError(
                f"{spec.source_filename} SHA256 不匹配: "
                f"expected={spec.expected_sha256}, actual={source_sha256}"
            )

        try:
            decoded_text = source_bytes.decode("cp936")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{spec.source_filename} 无法按 CP936 解码"
            ) from exc

        prepared_files.append(
            (spec, source_bytes, decoded_text.encode("utf-8"))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_manifests: list[CorpusFileManifest] = []

    for spec, source_bytes, output_bytes in prepared_files:
        output_path = output_dir / spec.source_filename
        output_path.write_bytes(output_bytes)
        file_manifests.append(
            CorpusFileManifest(
                book_id=spec.book_id,
                book_title=spec.book_title,
                source_file=spec.source_filename,
                source_sha256=sha256_bytes(source_bytes),
                source_bytes=len(source_bytes),
                source_encoding="cp936",
                output_file=spec.source_filename,
                output_sha256=sha256_bytes(output_bytes),
                output_bytes=len(output_bytes),
                output_encoding="utf-8",
            )
        )

    manifest = CorpusManifest(
        corpus_version=CORPUS_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        source=CorpusSource(
            url=SOURCE_URL,
            commit=SOURCE_COMMIT,
            commit_verification="declared_not_locally_verified",
            license_status="not_declared_in_local_snapshot",
        ),
        files=file_manifests,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = manifest.model_dump(mode="json")
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest
