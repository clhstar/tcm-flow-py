import argparse
import json
from pathlib import Path

from .config import load_production_config
from .indexing import build_index
from .models import BgeM3Encoder, prepare_models
from .pipeline import build_corpus, doctor_corpus, sha256_file


DEFAULT_CONFIG = Path("app/rag/config/ancient_books.yaml")
DEFAULT_CORPUS_DIR = Path("data/rag/ancient_books/corpus")
DEFAULT_INDEX_DIR = Path("data/rag/ancient_books/index")
DEFAULT_MODELS_DIR = Path("data/rag/ancient_books/models")
DEFAULT_MODELS_MANIFEST = DEFAULT_MODELS_DIR / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production ancient-book RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--source-root", type=Path, required=True)
    corpus.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    corpus.add_argument("--output-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    corpus.add_argument("--curated-root", type=Path, default=Path("data/raw"))

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)

    models = subparsers.add_parser("prepare-models")
    models.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    models.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR)
    models.add_argument(
        "--manifest", type=Path, default=DEFAULT_MODELS_MANIFEST
    )

    index = subparsers.add_parser("build-index")
    index.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    index.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    index.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX_DIR)
    index.add_argument(
        "--models-manifest", type=Path, default=DEFAULT_MODELS_MANIFEST
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-corpus":
        config = load_production_config(args.config)
        manifest = build_corpus(
            config=config,
            source_root=args.source_root,
            curated_root=args.curated_root,
            output_dir=args.output_dir,
        )
        print(
            f"status={manifest['status']} "
            f"books={manifest['book_count']} "
            f"parents={manifest['parent_count']} "
            f"chunks={manifest['chunk_count']}"
        )
        return

    if args.command == "prepare-models":
        config = load_production_config(args.config)
        manifest = prepare_models(
            config=config,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
        print(
            f"status={manifest['status']} "
            f"embedding_model={manifest['embedding']['model']} "
            f"embedding_revision={manifest['embedding']['revision']} "
            f"reranker_model={manifest['reranker']['model']} "
            f"reranker_revision={manifest['reranker']['revision']}"
        )
        return

    if args.command == "build-index":
        config = load_production_config(args.config)
        corpus_manifest_path = args.corpus_dir / "manifest.json"
        corpus_manifest = json.loads(
            corpus_manifest_path.read_text(encoding="utf-8")
        )
        chunks_record = corpus_manifest["files"]["chunks"]
        chunks_path = args.corpus_dir / chunks_record["path"]
        if sha256_file(chunks_path) != chunks_record["sha256"]:
            raise ValueError("chunks.jsonl 哈希与语料 manifest 不一致")
        models_manifest = json.loads(
            args.models_manifest.read_text(encoding="utf-8")
        )
        embedding_record = models_manifest["embedding"]
        settings = config["models"]["embedding"]
        if embedding_record["revision"] != settings["revision"]:
            raise ValueError("Embedding 模型 revision 与生产配置不一致")
        encoder = BgeM3Encoder(Path(embedding_record["local_path"]), settings)
        manifest = build_index(
            chunks_path=chunks_path,
            corpus_manifest_sha256=sha256_file(corpus_manifest_path),
            output_dir=args.output_dir,
            encoder=encoder,
            model_record=embedding_record,
        )
        print(
            f"status={manifest['status']} rows={manifest['row_count']} "
            f"dimension={manifest['vector_dimension']}"
        )
        return

    result = doctor_corpus(args.corpus_dir)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
