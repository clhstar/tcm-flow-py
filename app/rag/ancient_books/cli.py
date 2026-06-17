import argparse
from pathlib import Path

from .config import load_production_config
from .pipeline import build_corpus, doctor_corpus


DEFAULT_CONFIG = Path("app/rag/config/ancient_books.yaml")
DEFAULT_CORPUS_DIR = Path("data/rag/ancient_books/corpus")


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

    result = doctor_corpus(args.corpus_dir)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
