import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.rag_v1_5.corpus import (
    DEFAULT_CORPUS_SPECS,
    CorpusFileSpec,
    prepare_corpus,
)
from experiments.rag_v1_5.chunkers import build_chunk_artifacts
from experiments.rag_v1_5.pipeline import parse_prepared_corpus


DEFAULT_SOURCE_DIR = Path(r"G:\work\TCM-Ancient-Books-master")
DEFAULT_RAW_DIR = Path("data/rag_v1_5/raw")
DEFAULT_PROCESSED_DIR = Path("data/rag_v1_5/processed")
DEFAULT_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/corpus-v1.5.0.json"
)
DEFAULT_EVIDENCE_PATH = Path("data/rag_v1_5/processed/evidence.jsonl")
DEFAULT_CHUNK_CONFIG_PATH = Path(
    "experiments/rag_v1_5/configs/chunks.yaml"
)
DEFAULT_CHUNK_OUTPUT_DIR = Path("data/rag_v1_5/chunks")
DEFAULT_CHUNK_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/chunks-v1.5.0.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TCM-Flow V1.5 古籍语料导入与结构化解析"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-corpus",
        help="校验 CP936 原文件并导入为 UTF-8",
    )
    prepare_parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    prepare_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    prepare_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    parse_parser = subparsers.add_parser(
        "parse-corpus",
        help="依据 Manifest 解析篇章、条文、方剂和校注",
    )
    parse_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parse_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parse_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    chunk_parser = subparsers.add_parser(
        "build-chunks",
        help="构建 C0-C4 Chunk、统计和 Manifest",
    )
    chunk_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    chunk_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CHUNK_CONFIG_PATH,
    )
    chunk_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
    )
    chunk_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_CHUNK_MANIFEST_PATH,
    )
    chunk_parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    corpus_specs: Sequence[CorpusFileSpec] = DEFAULT_CORPUS_SPECS,
) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare-corpus":
        manifest = prepare_corpus(
            source_dir=args.source_dir,
            output_dir=args.raw_dir,
            manifest_path=args.manifest,
            specs=corpus_specs,
        )
        print(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "parse-corpus":
        statistics = parse_prepared_corpus(
            raw_dir=args.raw_dir,
            manifest_path=args.manifest,
            processed_dir=args.processed_dir,
        )
        print(json.dumps(statistics, ensure_ascii=False, indent=2))
        return 0

    manifest = build_chunk_artifacts(
        evidence_path=args.evidence,
        config_path=args.config,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        corpus_manifest_path=args.corpus_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
