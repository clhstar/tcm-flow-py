import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.pool import create_pool_from_settings
from app.rag.database.artifacts import load_artifact_bundle
from app.rag.database.repository import RagPostgresRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCM-Flow database RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-artifacts")
    import_parser.add_argument("--corpus-dir", required=True)
    import_parser.add_argument("--index-dir", required=True)

    subparsers.add_parser("doctor")
    subparsers.add_parser("smoke")
    return parser


async def run_import_artifacts(corpus_dir: str, index_dir: str) -> dict:
    settings = get_settings()
    pool = await create_pool_from_settings(settings)
    try:
        bundle = load_artifact_bundle(Path(corpus_dir), Path(index_dir))
        repository = RagPostgresRepository(pool)
        return await repository.import_bundle(bundle)
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-artifacts":
        result = asyncio.run(run_import_artifacts(args.corpus_dir, args.index_dir))
        print(result)
        return 0
    if args.command == "doctor":
        print({"status": "not_connected"})
        return 0
    if args.command == "smoke":
        print({"status": "not_connected"})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
