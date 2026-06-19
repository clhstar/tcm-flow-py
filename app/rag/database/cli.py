import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.pool import create_pool_from_settings
from app.rag.database.artifacts import load_artifact_bundle
from app.rag.database.elasticsearch_index import rebuild_keyword_index
from app.rag.database.repository import RagPostgresRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCM-Flow database RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-artifacts")
    import_parser.add_argument("--corpus-dir", required=True)
    import_parser.add_argument("--index-dir", required=True)

    rebuild_parser = subparsers.add_parser("rebuild-elasticsearch")
    rebuild_parser.add_argument("--corpus-dir", required=True)
    rebuild_parser.add_argument("--index-dir", required=True)

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


async def run_rebuild_elasticsearch(corpus_dir: str, index_dir: str) -> dict:
    settings = get_settings()
    if not settings.elasticsearch_url:
        raise ValueError("ELASTICSEARCH_URL is required for Elasticsearch rebuild")

    try:
        from elasticsearch import AsyncElasticsearch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "elasticsearch is required for Elasticsearch rebuild"
        ) from exc

    client = AsyncElasticsearch(settings.elasticsearch_url)
    try:
        bundle = load_artifact_bundle(Path(corpus_dir), Path(index_dir))
        return await rebuild_keyword_index(
            client,
            bundle,
            alias=settings.elasticsearch_rag_index_alias,
            analyzer=settings.elasticsearch_analyzer,
        )
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-artifacts":
        result = asyncio.run(run_import_artifacts(args.corpus_dir, args.index_dir))
        print(result)
        return 0
    if args.command == "rebuild-elasticsearch":
        result = asyncio.run(
            run_rebuild_elasticsearch(args.corpus_dir, args.index_dir)
        )
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
