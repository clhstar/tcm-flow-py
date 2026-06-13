import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Callable, Sequence

import yaml

from experiments.rag_v1_5.audit import (
    build_audit_artifacts,
    freeze_quality_gate,
    import_audit_review,
)
from experiments.rag_v1_5.corpus import (
    DEFAULT_CORPUS_SPECS,
    CorpusFileSpec,
    prepare_corpus,
)
from experiments.rag_v1_5.chunkers import build_chunk_artifacts
from experiments.rag_v1_5.model_store import (
    prepare_models,
    snapshot_files,
    snapshot_tree_sha256,
    validate_revision,
)
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
DEFAULT_RETRIEVAL_CONFIG_PATH = Path(
    "experiments/rag_v1_5/configs/retrieval-pilot.yaml"
)
DEFAULT_QUALITY_GATE_PATH = Path(
    "experiments/rag_v1_5/manifests/quality-gate-v1.5.0.json"
)
DEFAULT_MODEL_OUTPUT_DIR = Path("data/rag_v1_5/models")
DEFAULT_MODEL_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/models-v1.5.0.json"
)
DEFAULT_INDEXES_DIR = Path("data/rag_v1_5/indexes")
DEFAULT_ANOMALIES_PATH = Path("data/rag_v1_5/processed/anomalies.jsonl")
DEFAULT_AUDIT_OUTPUT_DIR = Path("data/rag_v1_5/audit")
DEFAULT_AUDIT_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/audit-sample-v1.5.0.json"
)
DEFAULT_AUDIT_SOURCE_PATH = Path("data/rag_v1_5/audit/audit-140.jsonl")
DEFAULT_AUDIT_REVIEW_PATH = Path("data/rag_v1_5/audit/audit-140.csv")
DEFAULT_AUDIT_ISSUES_PATH = Path(
    "data/rag_v1_5/audit/audit-issues.jsonl"
)
DEFAULT_AUDIT_SUMMARY_PATH = Path(
    "data/rag_v1_5/audit/audit-summary.json"
)
DIRECT_DEPENDENCIES = (
    "pydantic",
    "PyYAML",
    "numpy",
    "jieba",
    "rank-bm25",
    "FlagEmbedding",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_system_info() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "python_version": platform.python_version(),
            "torch_version": None,
            "cuda_available": False,
            "gpu_name": None,
            "gpu_memory_mib": 0,
        }

    cuda_available = torch.cuda.is_available()
    gpu_memory_mib = 0
    gpu_name = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        mebibyte = 1024 * 1024
        gpu_memory_mib = (properties.total_memory + mebibyte - 1) // mebibyte
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_mib": gpu_memory_mib,
    }


def _read_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _chunk_manifest_status(
    *,
    chunks_dir: Path,
    chunk_manifest_path: Path,
) -> tuple[int, str]:
    existing = [
        strategy
        for strategy in ("c0", "c1", "c2", "c3", "c4")
        if (chunks_dir / f"{strategy}.jsonl").is_file()
    ]
    if not chunk_manifest_path.is_file():
        return len(existing), "missing"

    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    for strategy in existing:
        strategy_manifest = manifest.get("strategies", {}).get(strategy)
        if strategy_manifest is None:
            return len(existing), "mismatch"
        path = chunks_dir / strategy_manifest["output_file"]
        if (
            not path.is_file()
            or _sha256_file(path) != strategy_manifest["output_sha256"]
        ):
            return len(existing), "mismatch"
    return len(existing), "valid" if len(existing) == 5 else "incomplete"


def _model_snapshot_status(
    *,
    model_manifest_path: Path,
    repository_root: Path,
) -> dict:
    if not model_manifest_path.is_file():
        return {"embedding": "missing", "reranker": "missing"}
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    status = {}
    for role in ("embedding", "reranker"):
        model_record = manifest.get(role, {})
        local_path = model_record.get("local_path")
        if not local_path:
            status[role] = "missing"
            continue
        snapshot_path = repository_root / local_path
        if not snapshot_path.is_dir():
            status[role] = "missing"
            continue
        try:
            actual_hash = snapshot_tree_sha256(snapshot_files(snapshot_path))
        except ValueError:
            status[role] = "missing"
            continue
        status[role] = (
            "valid"
            if actual_hash == model_record.get("snapshot_tree_sha256")
            else "mismatch"
        )
    return status


def _directory_is_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".write-test"
    try:
        marker.write_text("ok", encoding="ascii")
        return True
    except OSError:
        return False
    finally:
        marker.unlink(missing_ok=True)


def build_retrieval_doctor_report(
    *,
    config_path: Path = DEFAULT_RETRIEVAL_CONFIG_PATH,
    chunks_dir: Path = DEFAULT_CHUNK_OUTPUT_DIR,
    chunk_manifest_path: Path = DEFAULT_CHUNK_MANIFEST_PATH,
    quality_gate_path: Path = DEFAULT_QUALITY_GATE_PATH,
    model_manifest_path: Path = DEFAULT_MODEL_MANIFEST_PATH,
    indexes_dir: Path = DEFAULT_INDEXES_DIR,
    system_reader: Callable[[], dict] = _read_system_info,
    package_version_reader: Callable[[str], str | None] = (
        _read_package_version
    ),
) -> dict:
    report = dict(system_reader())
    report["direct_dependencies"] = {
        package: package_version_reader(package)
        for package in DIRECT_DEPENDENCIES
    }

    chunk_count, chunk_manifest_status = _chunk_manifest_status(
        chunks_dir=chunks_dir,
        chunk_manifest_path=chunk_manifest_path,
    )
    report["chunk_strategy_count"] = chunk_count
    report["chunk_manifest_status"] = chunk_manifest_status

    if quality_gate_path.is_file():
        quality_gate = json.loads(
            quality_gate_path.read_text(encoding="utf-8")
        )
        report["quality_gate_status"] = quality_gate.get("status", "invalid")
    else:
        report["quality_gate_status"] = "missing"

    revisions = {}
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for role in ("embedding", "reranker"):
            revision = config.get(role, {}).get("revision")
            try:
                revisions[role] = validate_revision(revision)
            except (TypeError, ValueError):
                revisions[role] = None
    report["model_revisions"] = revisions
    report["model_snapshots"] = _model_snapshot_status(
        model_manifest_path=model_manifest_path,
        repository_root=Path.cwd().resolve(),
    )
    report["indexes_writable"] = _directory_is_writable(indexes_dir)
    return report


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

    doctor_parser = subparsers.add_parser(
        "retrieval-doctor",
        help="检查检索实验环境、输入哈希、模型和 Quality Gate",
    )
    doctor_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    doctor_parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
    )
    doctor_parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=DEFAULT_CHUNK_MANIFEST_PATH,
    )
    doctor_parser.add_argument(
        "--quality-gate",
        type=Path,
        default=DEFAULT_QUALITY_GATE_PATH,
    )
    doctor_parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )
    doctor_parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
    )

    model_parser = subparsers.add_parser(
        "prepare-models",
        help="下载固定 revision 的本地检索模型快照",
    )
    model_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    model_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_DIR,
    )
    model_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )

    audit_parser = subparsers.add_parser(
        "sample-audit",
        help="按两书与三类配额生成 140 组人工抽检样本",
    )
    audit_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    audit_parser.add_argument(
        "--anomalies",
        type=Path,
        default=DEFAULT_ANOMALIES_PATH,
    )
    audit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT_DIR,
    )
    audit_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_AUDIT_MANIFEST_PATH,
    )
    audit_parser.add_argument(
        "--seed",
        type=int,
        default=20260612,
    )

    review_parser = subparsers.add_parser(
        "review-audit",
        help="导入人工审核并冻结语料 Quality Gate",
    )
    review_parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_AUDIT_SOURCE_PATH,
    )
    review_parser.add_argument(
        "--reviewed-csv",
        type=Path,
        default=DEFAULT_AUDIT_REVIEW_PATH,
    )
    review_parser.add_argument(
        "--issues",
        type=Path,
        default=DEFAULT_AUDIT_ISSUES_PATH,
    )
    review_parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_AUDIT_SUMMARY_PATH,
    )
    review_parser.add_argument(
        "--quality-gate",
        type=Path,
        default=DEFAULT_QUALITY_GATE_PATH,
    )
    review_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    review_parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
    )
    review_parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=DEFAULT_CHUNK_MANIFEST_PATH,
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

    if args.command == "build-chunks":
        manifest = build_chunk_artifacts(
            evidence_path=args.evidence,
            config_path=args.config,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            corpus_manifest_path=args.corpus_manifest,
        )
    elif args.command == "retrieval-doctor":
        manifest = build_retrieval_doctor_report(
            config_path=args.config,
            chunks_dir=args.chunks_dir,
            chunk_manifest_path=args.chunk_manifest,
            quality_gate_path=args.quality_gate,
            model_manifest_path=args.model_manifest,
            indexes_dir=args.indexes_dir,
        )
    elif args.command == "prepare-models":
        manifest = prepare_models(
            config_path=args.config,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
    elif args.command == "sample-audit":
        manifest = build_audit_artifacts(
            evidence_path=args.evidence,
            anomalies_path=args.anomalies,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            seed=args.seed,
        )
    else:
        summary = import_audit_review(
            source_jsonl=args.source,
            reviewed_csv=args.reviewed_csv,
            issues_path=args.issues,
            summary_path=args.summary,
        )
        manifest = freeze_quality_gate(
            summary=summary,
            source_jsonl=args.source,
            reviewed_csv=args.reviewed_csv,
            evidence_path=args.evidence,
            chunks_dir=args.chunks_dir,
            chunk_manifest_path=args.chunk_manifest,
            quality_gate_path=args.quality_gate,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
