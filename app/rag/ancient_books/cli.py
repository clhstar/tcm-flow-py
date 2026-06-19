import argparse
import json
from pathlib import Path

from .config import load_production_config
from .indexing import build_index
from .models import (
    BgeM3Encoder,
    BgeReranker,
    prepare_models,
    snapshot_files,
    snapshot_tree_sha256,
)
from .pipeline import build_corpus, doctor_corpus, export_manifests, sha256_file
from .runtime import ProductionRetrievalEngine, load_index


DEFAULT_CONFIG = Path("app/rag/config/ancient_books.yaml")
DEFAULT_CORPUS_DIR = Path("data/rag/ancient_books/corpus")
DEFAULT_INDEX_DIR = Path("data/rag/ancient_books/index")
DEFAULT_MODELS_DIR = Path("data/rag/ancient_books/models")
DEFAULT_MODELS_MANIFEST = DEFAULT_MODELS_DIR / "manifest.json"
DEFAULT_COMMIT_MANIFEST_DIR = Path("app/rag/ancient_books/manifests")
SMOKE_QUERIES = {
    "头痛": "头痛恶风，遇冷加重",
    "眩晕": "眩晕伴耳鸣和乏力",
    "咳嗽": "咳嗽有痰，夜间较重",
    "喘促": "活动后喘促并有胸闷",
    "心悸": "心悸反复，劳累后明显",
    "不寐": "入睡困难并且多梦易醒",
    "胃脘痛": "胃脘痛，饭后加重",
    "腹痛": "腹痛，排便后稍缓解",
    "泄泻": "泄泻清稀，受凉后明显",
    "便秘": "大便干结，排出困难",
}


def run_smoke(engine) -> dict:
    ok_count = 0
    insufficient_symptoms = []
    for symptom, query in SMOKE_QUERIES.items():
        result = engine.retrieve(
            query,
            chief_symptom=symptom,
            mode="hybrid",
            top_k=5,
        )
        if result.get("degraded"):
            raise RuntimeError(
                f"{symptom} smoke 发生降级检索: {result.get('degraded_reason')}"
            )
        rows = result.get("results", [])
        if result.get("status") == "ok":
            if not rows or not any(
                symptom in row.get("symptom_tags", []) for row in rows
            ):
                raise RuntimeError(f"{symptom} smoke 未返回正确标注的证据")
            ok_count += 1
        elif result.get("status") == "insufficient_evidence":
            if rows:
                raise RuntimeError(f"{symptom} 证据不足时仍返回了结果")
            insufficient_symptoms.append(symptom)
        else:
            raise RuntimeError(f"{symptom} smoke 返回未知状态")
    return {
        "status": "ready",
        "query_count": len(SMOKE_QUERIES),
        "ok_count": ok_count,
        "insufficient_count": len(insufficient_symptoms),
        "insufficient_symptoms": insufficient_symptoms,
        "degraded_count": 0,
    }


class _UnavailableModel:
    def __init__(self, reason: str):
        self.reason = reason

    def encode(self, texts):
        raise RuntimeError(self.reason)

    def score(self, pairs):
        raise RuntimeError(self.reason)


def _load_model_record(manifest: dict, *, role: str, expected: dict) -> tuple[Path, dict]:
    record = manifest.get(role)
    if not isinstance(record, dict):
        raise ValueError(f"model manifest missing {role}")
    if record.get("model") != expected["model"]:
        raise ValueError(f"{role} model does not match production config")
    if record.get("revision") != expected["revision"]:
        raise ValueError(f"{role} revision does not match production config")
    path = Path(record["local_path"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"missing local model snapshot: {path}")
    actual_hash = snapshot_tree_sha256(snapshot_files(path))
    if actual_hash != record.get("snapshot_tree_sha256"):
        raise ValueError(f"{role} local model snapshot hash does not match manifest")
    return path, record


def build_local_smoke_engine(
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    config_path: Path = DEFAULT_CONFIG,
    models_manifest_path: Path = DEFAULT_MODELS_MANIFEST,
) -> ProductionRetrievalEngine:
    index = load_index(index_dir, corpus_dir)
    config = load_production_config(config_path)

    try:
        model_manifest = json.loads(models_manifest_path.read_text(encoding="utf-8"))
        embedding_path, _ = _load_model_record(
            model_manifest,
            role="embedding",
            expected=config["models"]["embedding"],
        )
        encoder = BgeM3Encoder(embedding_path, config["models"]["embedding"])
    except Exception as error:
        encoder = _UnavailableModel(f"Dense model unavailable: {error}")

    try:
        if "model_manifest" not in locals():
            model_manifest = json.loads(models_manifest_path.read_text(encoding="utf-8"))
        reranker_path, _ = _load_model_record(
            model_manifest,
            role="reranker",
            expected=config["models"]["reranker"],
        )
        reranker = BgeReranker(reranker_path, config["models"]["reranker"])
    except Exception as error:
        reranker = _UnavailableModel(f"Reranker model unavailable: {error}")

    return ProductionRetrievalEngine(
        index=index,
        encoder=encoder,
        reranker=reranker,
        settings=config["retrieval"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production ancient-book RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--source-root", type=Path, required=True)
    corpus.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    corpus.add_argument("--output-dir", type=Path, default=DEFAULT_CORPUS_DIR)

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

    export = subparsers.add_parser("export-manifests")
    export.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    export.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    export.add_argument(
        "--output-dir", type=Path, default=DEFAULT_COMMIT_MANIFEST_DIR
    )
    subparsers.add_parser("smoke")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-corpus":
        config = load_production_config(args.config)
        manifest = build_corpus(
            config=config,
            source_root=args.source_root,
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

    if args.command == "export-manifests":
        corpus_manifest = json.loads(
            (args.corpus_dir / "manifest.json").read_text(encoding="utf-8")
        )
        index_manifest = json.loads(
            (args.index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        export_manifests(
            corpus_manifest=corpus_manifest,
            index_manifest=index_manifest,
            output_dir=args.output_dir,
        )
        print(f"status=ready output_dir={args.output_dir}")
        return

    if args.command == "smoke":
        result = run_smoke(build_local_smoke_engine())
        print(" ".join(f"{key}={value}" for key, value in result.items()))
        return

    result = doctor_corpus(args.corpus_dir)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
