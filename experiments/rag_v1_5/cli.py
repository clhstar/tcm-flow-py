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
    migrate_audit_review,
)
from experiments.rag_v1_5.corpus import (
    DEFAULT_CORPUS_SPECS,
    CorpusFileSpec,
    prepare_corpus,
)
from experiments.rag_v1_5.chunkers import build_chunk_artifacts
from experiments.rag_v1_5.dataset import (
    freeze_pilot_manifest,
    import_pilot_review,
    load_dataset,
    prepare_pilot_review,
    run_smoke_dataset,
    sample_pilot_evidence_groups,
    validate_dataset,
)
from experiments.rag_v1_5.indexing import (
    BgeM3DenseEncoder,
    build_indexes,
)
from experiments.rag_v1_5.model_store import (
    prepare_models,
    snapshot_files,
    snapshot_tree_sha256,
    validate_revision,
)
from experiments.rag_v1_5.pipeline import parse_prepared_corpus
from experiments.rag_v1_5.reranker import (
    FlagRerankerScorer,
    resolve_model_snapshot,
)
from experiments.rag_v1_5.retrieval import retrieve


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
DEFAULT_INDEX_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/indexes-v1.5.0.json"
)
DEFAULT_SMOKE_REVIEW_PATH = Path(
    "data/rag_v1_5/evaluation/smoke-review.csv"
)
DEFAULT_SMOKE_OUTPUT_DIR = Path("data/rag_v1_5/runs/smoke")
DEFAULT_SMOKE_DATASET_PATH = Path(
    "data/rag_v1_5/evaluation/smoke-10.jsonl"
)
DEFAULT_PILOT_EVIDENCE_GROUPS_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-evidence-groups.jsonl"
)
DEFAULT_PILOT_EXCLUSIONS_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-exclusions.json"
)
DEFAULT_PILOT_CANDIDATE_REPORT_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-candidate-report.json"
)
DEFAULT_PILOT_DRAFT_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-40-draft.jsonl"
)
DEFAULT_PILOT_REVIEW_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-review.csv"
)
DEFAULT_PILOT_REVIEW_SUMMARY_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-review-summary.json"
)
DEFAULT_PILOT_DATASET_PATH = Path(
    "data/rag_v1_5/evaluation/pilot-40.jsonl"
)
DEFAULT_PILOT_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/pilot-40-v1.5.0.json"
)
DEFAULT_SMOKE_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/smoke-10-v1.5.0.json"
)
DIRECT_DEPENDENCIES = (
    "pydantic",
    "PyYAML",
    "numpy",
    "jieba",
    "rank-bm25",
    "FlagEmbedding",
    "transformers",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_smoke_runtime_inputs(
    *,
    quality_gate_path: Path,
    index_manifest_path: Path,
    indexes_dir: Path,
    strategy: str,
) -> dict:
    if not quality_gate_path.is_file():
        raise FileNotFoundError(f"缺少 Quality Gate: {quality_gate_path}")
    if not index_manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少索引 Manifest: {index_manifest_path}"
        )
    quality_gate = json.loads(
        quality_gate_path.read_text(encoding="utf-8")
    )
    if quality_gate.get("status") != "ready":
        raise ValueError("真实 Smoke 运行要求 Quality Gate 为 ready")
    quality_gate_sha256 = _sha256_file(quality_gate_path)
    index_manifest = json.loads(
        index_manifest_path.read_text(encoding="utf-8")
    )
    if (
        index_manifest.get("quality_gate_sha256")
        != quality_gate_sha256
    ):
        raise ValueError("索引 Manifest 与当前 Quality Gate 哈希不一致")
    strategy_record = index_manifest.get("strategies", {}).get(strategy)
    if strategy_record is None:
        raise ValueError(f"索引 Manifest 缺少策略: {strategy}")
    strategy_manifest_path = indexes_dir / strategy / "manifest.json"
    if not strategy_manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少策略索引 Manifest: {strategy_manifest_path}"
        )
    strategy_manifest_sha256 = _sha256_file(strategy_manifest_path)
    if (
        strategy_record.get("manifest_sha256")
        != strategy_manifest_sha256
    ):
        raise ValueError("策略索引 Manifest 与顶层索引 Manifest 不一致")
    strategy_manifest = json.loads(
        strategy_manifest_path.read_text(encoding="utf-8")
    )
    if (
        strategy_manifest.get("quality_gate_sha256")
        != quality_gate_sha256
    ):
        raise ValueError("策略索引 Manifest 与当前 Quality Gate 不一致")
    return {
        "quality_gate_sha256": quality_gate_sha256,
        "index_manifest_sha256": _sha256_file(index_manifest_path),
        "strategy_manifest_sha256": strategy_manifest_sha256,
    }


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

    migration_parser = subparsers.add_parser(
        "migrate-audit-review",
        help="仅为结构完全未变化的新样本继承旧人工审核结论",
    )
    migration_parser.add_argument(
        "--previous-source",
        type=Path,
        required=True,
    )
    migration_parser.add_argument(
        "--previous-reviewed-csv",
        type=Path,
        required=True,
    )
    migration_parser.add_argument(
        "--new-source",
        type=Path,
        required=True,
    )
    migration_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    migration_parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )

    index_parser = subparsers.add_parser(
        "build-indexes",
        help="构建 C0-C4 的 BM25 token 与 Dense 向量索引",
    )
    index_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    index_parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
    )
    index_parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=DEFAULT_CHUNK_MANIFEST_PATH,
    )
    index_parser.add_argument(
        "--quality-gate",
        type=Path,
        default=DEFAULT_QUALITY_GATE_PATH,
    )
    index_parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )
    index_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
    )
    index_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INDEX_MANIFEST_PATH,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="运行 BM25、Dense、Hybrid 或 Hybrid+Reranker 检索",
    )
    search_parser.add_argument(
        "--strategy",
        choices=("c0", "c1", "c2", "c3", "c4"),
        required=True,
    )
    search_parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid", "hybrid_rerank"),
        required=True,
    )
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    search_parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
    )
    search_parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )

    validate_dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="校验检索试验问题集与 Evidence Tree 的引用契约",
    )
    validate_dataset_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )
    validate_dataset_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    validate_dataset_parser.add_argument(
        "--profile",
        choices=("auto", "smoke", "pilot", "generic"),
        default="auto",
    )
    validate_dataset_parser.add_argument(
        "--evidence-groups",
        type=Path,
    )

    pilot_evidence_parser = subparsers.add_parser(
        "sample-pilot-evidence",
        help="按固定配额选择 Pilot-40 Evidence Group",
    )
    pilot_evidence_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    pilot_evidence_parser.add_argument(
        "--smoke-dataset",
        type=Path,
        default=DEFAULT_SMOKE_DATASET_PATH,
    )
    pilot_evidence_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PILOT_EVIDENCE_GROUPS_PATH,
    )
    pilot_evidence_parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_PILOT_EXCLUSIONS_PATH,
    )
    pilot_evidence_parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_PILOT_CANDIDATE_REPORT_PATH,
    )
    pilot_evidence_parser.add_argument(
        "--seed",
        type=int,
        default=20260614,
    )

    prepare_pilot_review_parser = subparsers.add_parser(
        "prepare-pilot-review",
        help="导出 Pilot-40 双轮人工审核 CSV",
    )
    prepare_pilot_review_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_PILOT_DRAFT_PATH,
    )
    prepare_pilot_review_parser.add_argument(
        "--evidence-groups",
        type=Path,
        default=DEFAULT_PILOT_EVIDENCE_GROUPS_PATH,
    )
    prepare_pilot_review_parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_PILOT_REVIEW_PATH,
    )
    prepare_pilot_review_parser.add_argument(
        "--second-review-seed",
        type=int,
        default=20260614,
    )

    import_pilot_review_parser = subparsers.add_parser(
        "import-pilot-review",
        help="导入 Pilot-40 双轮审核并冻结本地问题集",
    )
    import_pilot_review_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_PILOT_DRAFT_PATH,
    )
    import_pilot_review_parser.add_argument(
        "--evidence-groups",
        type=Path,
        default=DEFAULT_PILOT_EVIDENCE_GROUPS_PATH,
    )
    import_pilot_review_parser.add_argument(
        "--reviewed-csv",
        type=Path,
        default=DEFAULT_PILOT_REVIEW_PATH,
    )
    import_pilot_review_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PILOT_DATASET_PATH,
    )
    import_pilot_review_parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_PILOT_REVIEW_SUMMARY_PATH,
    )

    freeze_pilot_parser = subparsers.add_parser(
        "freeze-pilot-dataset",
        help="校验并冻结 Pilot-40 Manifest 与输入哈希链",
    )
    freeze_pilot_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_PILOT_DATASET_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--evidence-groups",
        type=Path,
        default=DEFAULT_PILOT_EVIDENCE_GROUPS_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--review-summary",
        type=Path,
        default=DEFAULT_PILOT_REVIEW_SUMMARY_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_PILOT_EXCLUSIONS_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PILOT_MANIFEST_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=DEFAULT_CHUNK_MANIFEST_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--quality-gate",
        type=Path,
        default=DEFAULT_QUALITY_GATE_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--index-manifest",
        type=Path,
        default=DEFAULT_INDEX_MANIFEST_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    freeze_pilot_parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=DEFAULT_SMOKE_MANIFEST_PATH,
    )

    smoke_parser = subparsers.add_parser(
        "run-smoke",
        help="运行 10 条检索烟雾测试并生成 Top 5 人工复核表",
    )
    smoke_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )
    smoke_parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    smoke_parser.add_argument(
        "--strategy",
        choices=("c0", "c1", "c2", "c3", "c4"),
        required=True,
    )
    smoke_parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid", "hybrid_rerank"),
        required=True,
    )
    smoke_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
    )
    smoke_parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=DEFAULT_INDEXES_DIR,
    )
    smoke_parser.add_argument(
        "--index-manifest",
        type=Path,
        default=DEFAULT_INDEX_MANIFEST_PATH,
    )
    smoke_parser.add_argument(
        "--quality-gate",
        type=Path,
        default=DEFAULT_QUALITY_GATE_PATH,
    )
    smoke_parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST_PATH,
    )
    smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SMOKE_OUTPUT_DIR,
    )
    smoke_parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_SMOKE_REVIEW_PATH,
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
    elif args.command == "review-audit":
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
    elif args.command == "migrate-audit-review":
        manifest = migrate_audit_review(
            previous_source_jsonl=args.previous_source,
            previous_reviewed_csv=args.previous_reviewed_csv,
            new_source_jsonl=args.new_source,
            output_csv=args.output_csv,
            summary_path=args.summary,
        )
    elif args.command == "build-indexes":
        manifest = build_indexes(
            config_path=args.config,
            chunks_dir=args.chunks_dir,
            chunk_manifest_path=args.chunk_manifest,
            quality_gate_path=args.quality_gate,
            model_manifest_path=args.model_manifest,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
    elif args.command == "validate-dataset":
        manifest = validate_dataset(
            dataset_path=args.dataset,
            evidence_path=args.evidence,
            profile=args.profile,
            evidence_groups_path=args.evidence_groups,
        )
    elif args.command == "sample-pilot-evidence":
        manifest = sample_pilot_evidence_groups(
            evidence_path=args.evidence,
            smoke_dataset_path=args.smoke_dataset,
            output_path=args.output,
            exclusions_path=args.exclusions,
            candidate_report_path=args.candidate_report,
            seed=args.seed,
        )
    elif args.command == "prepare-pilot-review":
        manifest = prepare_pilot_review(
            draft_dataset_path=args.dataset,
            evidence_groups_path=args.evidence_groups,
            review_csv_path=args.review_csv,
            second_review_seed=args.second_review_seed,
        )
    elif args.command == "import-pilot-review":
        manifest = import_pilot_review(
            draft_dataset_path=args.dataset,
            evidence_groups_path=args.evidence_groups,
            reviewed_csv_path=args.reviewed_csv,
            output_dataset_path=args.output,
            summary_path=args.summary,
        )
    elif args.command == "freeze-pilot-dataset":
        manifest = freeze_pilot_manifest(
            dataset_path=args.dataset,
            evidence_groups_path=args.evidence_groups,
            review_summary_path=args.review_summary,
            exclusions_path=args.exclusions,
            manifest_path=args.manifest,
            evidence_path=args.evidence,
            chunk_manifest_path=args.chunk_manifest,
            quality_gate_path=args.quality_gate,
            index_manifest_path=args.index_manifest,
            model_manifest_path=args.model_manifest,
            config_path=args.config,
            smoke_manifest_path=args.smoke_manifest,
        )
    elif args.command == "run-smoke":
        dataset_summary = validate_dataset(
            dataset_path=args.dataset,
            evidence_path=args.evidence,
            profile="smoke",
        )
        if (
            dataset_summary["approved_count"]
            != dataset_summary["question_count"]
        ):
            raise ValueError("Smoke 数据集必须全部审核为 approved")
        runtime_provenance = validate_smoke_runtime_inputs(
            quality_gate_path=args.quality_gate,
            index_manifest_path=args.index_manifest,
            indexes_dir=args.indexes_dir,
            strategy=args.strategy,
        )
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        dense_encoder = None
        reranker_scorer = None
        if args.mode in {"dense", "hybrid", "hybrid_rerank"}:
            embedding_path, _ = resolve_model_snapshot(
                config=config,
                role="embedding",
                model_manifest_path=args.model_manifest,
            )
            dense_encoder = BgeM3DenseEncoder(
                embedding_path,
                config["embedding"],
            )
        if args.mode == "hybrid_rerank":
            reranker_path, _ = resolve_model_snapshot(
                config=config,
                role="reranker",
                model_manifest_path=args.model_manifest,
            )
            reranker_scorer = FlagRerankerScorer(
                reranker_path,
                config["reranker"],
            )

        def retrieve_question(question):
            return retrieve(
                question.question,
                strategy=args.strategy,
                mode=args.mode,
                indexes_dir=args.indexes_dir,
                config=config,
                dense_encoder=dense_encoder,
                reranker_scorer=reranker_scorer,
                model_manifest_path=args.model_manifest,
            )

        manifest = run_smoke_dataset(
            questions=load_dataset(args.dataset),
            strategy=args.strategy,
            mode=args.mode,
            output_dir=args.output_dir,
            review_csv_path=args.review_csv,
            retriever=retrieve_question,
            provenance={
                **dataset_summary,
                **runtime_provenance,
                "config_sha256": _sha256_file(args.config),
                "model_manifest_sha256": _sha256_file(
                    args.model_manifest
                ),
            },
        )
    else:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        hits = retrieve(
            args.query,
            strategy=args.strategy,
            mode=args.mode,
            indexes_dir=args.indexes_dir,
            config=config,
            model_manifest_path=args.model_manifest,
        )
        manifest = [hit.model_dump(mode="json") for hit in hits]
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
