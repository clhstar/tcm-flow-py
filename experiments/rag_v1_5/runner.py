import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from experiments.rag_v1_5.dataset import (
    load_dataset,
    validate_dataset,
)
from experiments.rag_v1_5.indexing import BgeM3DenseEncoder
from experiments.rag_v1_5.metrics import (
    LATENCY_FIELDS,
    evaluate_rankings,
    index_size_bytes,
    summarize_latency,
    summarize_score_distribution,
)
from experiments.rag_v1_5.reranker import (
    FlagRerankerScorer,
    resolve_model_snapshot,
)
from experiments.rag_v1_5.retrieval import (
    RetrievalResult,
    load_index,
    retrieve_loaded,
)
from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit


PILOT_MATRIX = (
    ("c0-hybrid-rerank", "c0", "hybrid_rerank"),
    ("c1-hybrid-rerank", "c1", "hybrid_rerank"),
    ("c2-hybrid-rerank", "c2", "hybrid_rerank"),
    ("c3-hybrid-rerank", "c3", "hybrid_rerank"),
    ("c4-bm25", "c4", "bm25"),
    ("c4-dense", "c4", "dense"),
    ("c4-hybrid", "c4", "hybrid"),
    ("c4-hybrid-rerank", "c4", "hybrid_rerank"),
)
WARMUP_QUERY = "太阳病脉证与治法"
RUN_FILE_NAMES = (
    "run-config.json",
    "per-question.jsonl",
    "metrics.json",
    "latency.json",
    "errors.jsonl",
)
CORE_METRIC_FIELDS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_10",
)
PILOT_INPUT_HASH_FIELDS = {
    "evidence_group": "evidence_group_sha256",
    "review_summary": "review_summary_sha256",
    "quality_gate": "quality_gate_sha256",
    "smoke_manifest": "smoke_manifest_sha256",
    "index_manifest": "index_manifest_sha256",
    "model_manifest": "model_manifest_sha256",
    "config": "config_sha256",
    "evidence": "evidence_sha256",
    "chunk_manifest": "chunk_manifest_sha256",
}


@dataclass
class PilotConfigRuntime:
    retrieve: Callable[[PilotQuestion], RetrievalResult]
    index_load_ms: float
    embedding_load_ms: float
    reranker_load_ms: float
    warmup_ms: float
    index_size_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是合法 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须为 JSON object")
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_jsonl_strict(path: Path, *, label: str) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError(f"{label} JSONL 存在未完成行: {path}")
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} JSONL 第 {line_number} 行损坏"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"{label} JSONL 第 {line_number} 行必须为 object"
            )
        records.append(record)
    return records


def _append_jsonl(handle, payload: dict) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    handle.write(serialized + "\n")
    handle.flush()


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_path(path_value: str, *, repository_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _display_path(path: Path, *, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_input(
    pilot_manifest: dict,
    key: str,
    *,
    repository_root: Path,
) -> tuple[Path, str]:
    record = pilot_manifest.get("inputs", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Pilot Manifest 缺少输入记录: {key}")
    path_value = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(
        expected_sha256,
        str,
    ):
        raise ValueError(f"Pilot Manifest 输入记录非法: {key}")
    path = _resolve_path(path_value, repository_root=repository_root)
    if not path.is_file():
        raise FileNotFoundError(f"Pilot 输入文件不存在: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Pilot 输入哈希不一致: {key}")
    return path, actual_sha256


def validate_pilot_runtime_inputs(
    *,
    dataset_path: Path,
    evidence_groups_path: Path,
    pilot_manifest_path: Path,
    config_path: Path,
    indexes_dir: Path,
    repository_root: Path | None = None,
    cuda_checker: Callable[[], bool] = _cuda_available,
) -> dict:
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    pilot_manifest = _read_json(
        pilot_manifest_path,
        label="Pilot Manifest",
    )
    if pilot_manifest.get("status") != "ready":
        raise ValueError("Pilot Manifest 必须为 ready")
    if not cuda_checker():
        raise ValueError("Pilot 真实运行要求 CUDA 可用")

    dataset_path = dataset_path.resolve()
    evidence_groups_path = evidence_groups_path.resolve()
    config_path = config_path.resolve()
    indexes_dir = indexes_dir.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"缺少 Pilot dataset: {dataset_path}")
    if not evidence_groups_path.is_file():
        raise FileNotFoundError(
            f"缺少 Pilot Evidence Group: {evidence_groups_path}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少检索配置: {config_path}")

    dataset_record = pilot_manifest.get("dataset", {})
    dataset_sha256 = _sha256_file(dataset_path)
    if dataset_record.get("sha256") != dataset_sha256:
        raise ValueError("Pilot dataset 哈希与 Manifest 不一致")
    evidence_group_path, evidence_group_sha256 = _manifest_input(
        pilot_manifest,
        "evidence_group",
        repository_root=repository_root,
    )
    if evidence_group_path != evidence_groups_path:
        raise ValueError("Pilot Evidence Group 路径与 Manifest 不一致")
    review_summary_path, review_summary_sha256 = _manifest_input(
        pilot_manifest,
        "review_summary",
        repository_root=repository_root,
    )
    evidence_path, evidence_sha256 = _manifest_input(
        pilot_manifest,
        "evidence",
        repository_root=repository_root,
    )
    chunk_manifest_path, chunk_manifest_sha256 = _manifest_input(
        pilot_manifest,
        "chunk_manifest",
        repository_root=repository_root,
    )
    quality_gate_path, quality_gate_sha256 = _manifest_input(
        pilot_manifest,
        "quality_gate",
        repository_root=repository_root,
    )
    index_manifest_path, index_manifest_sha256 = _manifest_input(
        pilot_manifest,
        "index_manifest",
        repository_root=repository_root,
    )
    model_manifest_path, model_manifest_sha256 = _manifest_input(
        pilot_manifest,
        "model_manifest",
        repository_root=repository_root,
    )
    frozen_config_path, frozen_config_sha256 = _manifest_input(
        pilot_manifest,
        "config",
        repository_root=repository_root,
    )
    if frozen_config_path != config_path:
        raise ValueError("检索配置路径与 Pilot Manifest 不一致")
    smoke_manifest_path, smoke_manifest_sha256 = _manifest_input(
        pilot_manifest,
        "smoke_manifest",
        repository_root=repository_root,
    )

    quality_gate = _read_json(
        quality_gate_path,
        label="Quality Gate",
    )
    if quality_gate.get("status") != "ready":
        raise ValueError("Quality Gate 必须为 ready")
    smoke_manifest = _read_json(
        smoke_manifest_path,
        label="Smoke Manifest",
    )
    if smoke_manifest.get("status") != "passed":
        raise ValueError("Smoke Manifest 必须为 passed")
    chunk_manifest = _read_json(
        chunk_manifest_path,
        label="Chunk Manifest",
    )
    index_manifest = _read_json(
        index_manifest_path,
        label="Index Manifest",
    )
    if (
        chunk_manifest.get("evidence_sha256") != evidence_sha256
        or quality_gate.get("chunk_manifest_sha256")
        != chunk_manifest_sha256
        or quality_gate.get("evidence_sha256") != evidence_sha256
        or index_manifest.get("chunk_manifest_sha256")
        != chunk_manifest_sha256
        or index_manifest.get("quality_gate_sha256")
        != quality_gate_sha256
        or index_manifest.get("model_manifest_sha256")
        != model_manifest_sha256
    ):
        raise ValueError("Pilot 运行上游 Manifest 哈希链不一致")

    strategy_manifest_sha256 = {}
    for strategy in ("c0", "c1", "c2", "c3", "c4"):
        record = index_manifest.get("strategies", {}).get(strategy)
        if not isinstance(record, dict):
            raise ValueError(f"Index Manifest 缺少策略: {strategy}")
        strategy_manifest_path = indexes_dir / strategy / "manifest.json"
        if not strategy_manifest_path.is_file():
            raise FileNotFoundError(
                f"缺少策略索引 Manifest: {strategy_manifest_path}"
            )
        actual_sha256 = _sha256_file(strategy_manifest_path)
        if record.get("manifest_sha256") != actual_sha256:
            raise ValueError(f"{strategy} 策略 Manifest 哈希不一致")
        strategy_manifest = _read_json(
            strategy_manifest_path,
            label=f"{strategy} 策略 Manifest",
        )
        if strategy_manifest.get(
            "quality_gate_sha256"
        ) != quality_gate_sha256:
            raise ValueError(f"{strategy} 策略与 Quality Gate 不一致")
        strategy_manifest_sha256[strategy] = actual_sha256

    dataset_summary = validate_dataset(
        dataset_path=dataset_path,
        evidence_path=evidence_path,
        profile="pilot",
        evidence_groups_path=evidence_groups_path,
    )
    questions = load_dataset(dataset_path)
    if (
        dataset_summary["question_count"] != 40
        or dataset_summary["approved_count"] != 40
        or len(questions) != 40
    ):
        raise ValueError("Pilot runtime 要求恰好 40 条 approved 问题")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("检索配置顶层必须为 mapping")
    return {
        "pilot_manifest_status": pilot_manifest["status"],
        "questions": questions,
        "config": config,
        "input_hashes": {
            "pilot_manifest_sha256": _sha256_file(
                pilot_manifest_path
            ),
            "dataset_sha256": dataset_sha256,
            "evidence_group_sha256": evidence_group_sha256,
            "review_summary_sha256": review_summary_sha256,
            "quality_gate_sha256": quality_gate_sha256,
            "smoke_manifest_sha256": smoke_manifest_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "config_sha256": frozen_config_sha256,
            "evidence_sha256": evidence_sha256,
            "chunk_manifest_sha256": chunk_manifest_sha256,
            "strategy_manifest_sha256": strategy_manifest_sha256,
        },
        "paths": {
            "review_summary": review_summary_path,
            "evidence": evidence_path,
            "chunk_manifest": chunk_manifest_path,
            "quality_gate": quality_gate_path,
            "index_manifest": index_manifest_path,
            "model_manifest": model_manifest_path,
            "smoke_manifest": smoke_manifest_path,
        },
    }


class DefaultPilotRuntimeFactory:
    def __init__(
        self,
        *,
        indexes_dir: Path,
        model_manifest_path: Path,
        repository_root: Path,
    ) -> None:
        self.indexes_dir = indexes_dir
        self.model_manifest_path = model_manifest_path
        self.repository_root = repository_root
        self.dense_encoder = None
        self.reranker_scorer = None

    def __call__(
        self,
        config_id: str,
        strategy: str,
        mode: str,
        config: dict,
    ) -> PilotConfigRuntime:
        started = time.perf_counter()
        index = load_index(self.indexes_dir / strategy)
        index_load_ms = (time.perf_counter() - started) * 1000

        embedding_load_ms = 0.0
        if (
            mode in {"dense", "hybrid", "hybrid_rerank"}
            and self.dense_encoder is None
        ):
            started = time.perf_counter()
            local_path, _ = resolve_model_snapshot(
                config=config,
                role="embedding",
                model_manifest_path=self.model_manifest_path,
                repository_root=self.repository_root,
            )
            self.dense_encoder = BgeM3DenseEncoder(
                local_path,
                config["embedding"],
            )
            embedding_load_ms = (time.perf_counter() - started) * 1000

        reranker_load_ms = 0.0
        if mode == "hybrid_rerank" and self.reranker_scorer is None:
            started = time.perf_counter()
            local_path, _ = resolve_model_snapshot(
                config=config,
                role="reranker",
                model_manifest_path=self.model_manifest_path,
                repository_root=self.repository_root,
            )
            self.reranker_scorer = FlagRerankerScorer(
                local_path,
                config["reranker"],
            )
            reranker_load_ms = (time.perf_counter() - started) * 1000

        result_top_k = max(int(value) for value in config["evaluation"]["top_ks"])

        def retrieve_question(question: PilotQuestion) -> RetrievalResult:
            return retrieve_loaded(
                question.question,
                index=index,
                mode=mode,
                config=config,
                dense_encoder=self.dense_encoder,
                reranker_scorer=self.reranker_scorer,
                result_top_k=result_top_k,
            )

        started = time.perf_counter()
        retrieve_loaded(
            WARMUP_QUERY,
            index=index,
            mode=mode,
            config=config,
            dense_encoder=self.dense_encoder,
            reranker_scorer=self.reranker_scorer,
            result_top_k=result_top_k,
        )
        warmup_ms = (time.perf_counter() - started) * 1000
        return PilotConfigRuntime(
            retrieve=retrieve_question,
            index_load_ms=index_load_ms,
            embedding_load_ms=embedding_load_ms,
            reranker_load_ms=reranker_load_ms,
            warmup_ms=warmup_ms,
            index_size_bytes=index_size_bytes(
                self.indexes_dir / strategy
            ),
        )


def _matrix_definition() -> list[dict[str, str]]:
    return [
        {
            "config_id": config_id,
            "strategy": strategy,
            "mode": mode,
        }
        for config_id, strategy, mode in PILOT_MATRIX
    ]


def _compact_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _traceability_ok(hits: list[RetrievalHit]) -> bool:
    top5 = hits[:5]
    return len(top5) == 5 and all(
        hit.chunk_id
        and hit.source_evidence_ids
        and hit.clause_ids
        and hit.context_text.strip()
        for hit in top5
    )


def _c4_parent_recovery_ok(
    hits: list[RetrievalHit],
    *,
    strategy: str,
) -> bool | None:
    if strategy != "c4":
        return None
    top5 = hits[:5]
    return len(top5) == 5 and all(
        hit.retrieval_parent_id
        and hit.retrieval_parent_id in hit.clause_ids
        and hit.context_text.strip()
        for hit in top5
    )


def _question_record(
    *,
    config_id: str,
    strategy: str,
    question: PilotQuestion,
    result: RetrievalResult,
) -> dict:
    hits = [RetrievalHit.model_validate(hit) for hit in result.hits]
    if set(result.latency) != set(LATENCY_FIELDS):
        raise ValueError(
            f"{question.question_id} latency 字段不完整"
        )
    return {
        "config_id": config_id,
        "question_id": question.question_id,
        "answerable": question.answerable,
        "book_scope": question.book_scope,
        "question_type": question.question_type,
        "hits": [hit.model_dump(mode="json") for hit in hits],
        "latency": {
            field: result.latency[field] for field in LATENCY_FIELDS
        },
        "top5_traceability_ok": _traceability_ok(hits),
        "c4_parent_recovery_ok": _c4_parent_recovery_ok(
            hits,
            strategy=strategy,
        ),
    }


def _validated_per_question_records(
    path: Path,
    *,
    config_id: str,
    question_ids: set[str],
) -> list[dict]:
    records = _read_jsonl_strict(path, label=f"{config_id} per-question")
    seen = set()
    for record in records:
        question_id = record.get("question_id")
        if question_id in seen:
            raise ValueError(
                f"{config_id} per-question 存在重复 question_id: "
                f"{question_id}"
            )
        if (
            question_id not in question_ids
            or record.get("config_id") != config_id
        ):
            raise ValueError(f"{config_id} per-question 行归属不一致")
        seen.add(question_id)
    return records


def _config_hash(
    *,
    config: dict,
    strategy: str,
    mode: str,
) -> str:
    return _json_sha256(
        {
            "config": config,
            "strategy": strategy,
            "mode": mode,
        }
    )


def _metric_payload(
    *,
    questions_by_id: dict[str, PilotQuestion],
    records: list[dict],
    error_records: list[dict],
) -> tuple[dict, dict]:
    rankings = {
        record["question_id"]: [
            RetrievalHit.model_validate(hit)
            for hit in record["hits"]
        ]
        for record in records
    }
    completed_questions = [
        questions_by_id[record["question_id"]]
        for record in records
    ]
    metrics = evaluate_rankings(completed_questions, rankings)
    raw_no_answer_scores = metrics.pop("no_answer_scores")
    metrics["no_answer_score_distribution"] = {
        key: summarize_score_distribution(values)
        for key, values in raw_no_answer_scores.items()
    }
    metrics["top5_traceability_rate"] = (
        sum(record["top5_traceability_ok"] for record in records)
        / len(records)
        if records
        else 0.0
    )
    c4_values = [
        record["c4_parent_recovery_ok"]
        for record in records
        if record["c4_parent_recovery_ok"] is not None
    ]
    metrics["c4_parent_recovery_rate"] = (
        sum(c4_values) / len(c4_values) if c4_values else None
    )
    unresolved_question_ids = set(questions_by_id) - set(rankings)
    metrics.update(
        completed_count=len(records),
        error_count=len(unresolved_question_ids),
        error_attempt_count=len(error_records),
        status=(
            "completed"
            if len(records) == len(questions_by_id)
            else "failed"
        ),
    )
    latency_records = [
        {
            "question_id": record["question_id"],
            **record["latency"],
        }
        for record in records
    ]
    latency = {
        "summary": summarize_latency(latency_records),
        "records": latency_records,
    }
    return metrics, latency


def _config_summary(
    *,
    config_id: str,
    strategy: str,
    mode: str,
    metrics: dict,
    latency: dict,
    run_config: dict,
) -> dict:
    return {
        "config_id": config_id,
        "strategy": strategy,
        "mode": mode,
        **metrics,
        "latency": latency["summary"],
        "index_size_bytes": run_config["runtime"][
            "index_size_bytes"
        ],
        "index_load_ms": run_config["runtime"]["index_load_ms"],
        "embedding_load_ms": run_config["runtime"][
            "embedding_load_ms"
        ],
        "reranker_load_ms": run_config["runtime"][
            "reranker_load_ms"
        ],
        "warmup_ms": run_config["runtime"]["warmup_ms"],
    }


def run_pilot_matrix(
    *,
    dataset_path: Path,
    evidence_groups_path: Path,
    pilot_manifest_path: Path,
    config_path: Path,
    indexes_dir: Path,
    output_dir: Path,
    resume_dir: Path | None = None,
    repository_root: Path | None = None,
    input_validator: Callable[..., dict] = validate_pilot_runtime_inputs,
    runtime_factory: Callable[
        [str, str, str, dict],
        PilotConfigRuntime,
    ]
    | None = None,
    cuda_checker: Callable[[], bool] = _cuda_available,
    now: datetime | None = None,
) -> dict:
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    validated = input_validator(
        dataset_path=dataset_path,
        evidence_groups_path=evidence_groups_path,
        pilot_manifest_path=pilot_manifest_path,
        config_path=config_path,
        indexes_dir=indexes_dir,
        repository_root=repository_root,
        cuda_checker=cuda_checker,
    )
    if validated.get("pilot_manifest_status") != "ready":
        raise ValueError("Pilot Manifest 必须为 ready")
    questions = sorted(
        validated["questions"],
        key=lambda question: question.question_id,
    )
    if len(questions) != 40 or any(
        question.review_status != "approved" for question in questions
    ):
        raise ValueError("Pilot 运行要求恰好 40 条 approved 问题")
    question_ids = [question.question_id for question in questions]
    if len(set(question_ids)) != 40:
        raise ValueError("Pilot 问题存在重复 question_id")
    questions_by_id = {
        question.question_id: question for question in questions
    }
    config = validated["config"]
    input_hashes = validated["input_hashes"]
    current_time = now or datetime.now(timezone.utc)
    run_timestamp = _compact_timestamp(current_time)
    matrix_definition = _matrix_definition()

    if resume_dir is None:
        matrix_id = (
            f"pilot-{run_timestamp}-"
            f"{input_hashes['dataset_sha256'][:8]}-"
            f"{input_hashes['config_sha256'][:8]}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_dir = output_dir / matrix_id
        matrix_dir.mkdir(parents=False, exist_ok=False)
        matrix_config = {
            "version": "v1.5.0",
            "status": "running",
            "matrix_id": matrix_id,
            "created_at": current_time.astimezone(
                timezone.utc
            ).isoformat(),
            "run_timestamp": run_timestamp,
            "matrix": matrix_definition,
            "config": config,
            "input_hashes": input_hashes,
        }
        _atomic_write_json(
            matrix_dir / "matrix-config.json",
            matrix_config,
        )
    else:
        matrix_dir = resume_dir.resolve()
        matrix_config = _read_json(
            matrix_dir / "matrix-config.json",
            label="matrix-config",
        )
        if matrix_config.get("input_hashes") != input_hashes:
            raise ValueError("恢复运行的输入哈希与当前输入哈希不一致")
        if (
            matrix_config.get("config") != config
            or matrix_config.get("matrix") != matrix_definition
        ):
            raise ValueError("恢复运行的配置或固定矩阵不一致")
        matrix_id = matrix_config["matrix_id"]
        run_timestamp = matrix_config["run_timestamp"]

    if runtime_factory is None:
        runtime_factory = DefaultPilotRuntimeFactory(
            indexes_dir=indexes_dir.resolve(),
            model_manifest_path=validated["paths"]["model_manifest"],
            repository_root=repository_root,
        )

    config_summaries = []
    for config_id, strategy, mode in PILOT_MATRIX:
        config_dir = matrix_dir / config_id
        if resume_dir is None:
            config_dir.mkdir(exist_ok=False)
        else:
            config_dir.mkdir(exist_ok=True)
        per_question_path = config_dir / "per-question.jsonl"
        errors_path = config_dir / "errors.jsonl"
        run_config_path = config_dir / "run-config.json"
        per_question_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)

        config_hash = _config_hash(
            config=config,
            strategy=strategy,
            mode=mode,
        )
        existing_run_config = (
            _read_json(run_config_path, label=f"{config_id} run-config")
            if run_config_path.is_file()
            else None
        )
        if existing_run_config is not None and (
            existing_run_config.get("input_hashes") != input_hashes
            or existing_run_config.get("config_hash") != config_hash
            or existing_run_config.get("config") != config
        ):
            raise ValueError(f"{config_id} run-config 输入哈希不一致")

        records = _validated_per_question_records(
            per_question_path,
            config_id=config_id,
            question_ids=set(question_ids),
        )
        completed_ids = {
            record["question_id"] for record in records
        }
        error_records = _read_jsonl_strict(
            errors_path,
            label=f"{config_id} errors",
        )
        attempts = Counter(
            record.get("question_id") for record in error_records
        )

        runtime = None
        if len(completed_ids) < 40:
            runtime = runtime_factory(
                config_id,
                strategy,
                mode,
                config,
            )
            run_id = (
                existing_run_config["run_id"]
                if existing_run_config is not None
                else (
                    f"{run_timestamp}-{config_id}-"
                    f"{config_hash[:8]}"
                )
            )
            run_config = {
                "version": "v1.5.0",
                "status": "running",
                "run_id": run_id,
                "config_id": config_id,
                "strategy": strategy,
                "mode": mode,
                "config_hash": config_hash,
                "config": config,
                "input_hashes": input_hashes,
                "runtime": {
                    "index_load_ms": runtime.index_load_ms,
                    "embedding_load_ms": runtime.embedding_load_ms,
                    "reranker_load_ms": runtime.reranker_load_ms,
                    "warmup_ms": runtime.warmup_ms,
                    "index_size_bytes": runtime.index_size_bytes,
                },
            }
            _atomic_write_json(run_config_path, run_config)
            with per_question_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as per_question_handle, errors_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as errors_handle:
                for question in questions:
                    if question.question_id in completed_ids:
                        continue
                    attempt = attempts[question.question_id] + 1
                    try:
                        result = runtime.retrieve(question)
                        record = _question_record(
                            config_id=config_id,
                            strategy=strategy,
                            question=question,
                            result=result,
                        )
                        _append_jsonl(per_question_handle, record)
                        completed_ids.add(question.question_id)
                    except Exception as error:
                        error_record = {
                            "config_id": config_id,
                            "question_id": question.question_id,
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "recorded_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                        _append_jsonl(errors_handle, error_record)
                        attempts[question.question_id] = attempt
        else:
            if existing_run_config is None:
                raise ValueError(
                    f"{config_id} 已有结果但缺少 run-config"
                )
            run_config = existing_run_config

        records = _validated_per_question_records(
            per_question_path,
            config_id=config_id,
            question_ids=set(question_ids),
        )
        error_records = _read_jsonl_strict(
            errors_path,
            label=f"{config_id} errors",
        )
        metrics, latency = _metric_payload(
            questions_by_id=questions_by_id,
            records=records,
            error_records=error_records,
        )
        run_config = {
            **run_config,
            "status": metrics["status"],
            "completed_count": metrics["completed_count"],
            "error_count": metrics["error_count"],
        }
        _atomic_write_json(config_dir / "metrics.json", metrics)
        _atomic_write_json(config_dir / "latency.json", latency)
        _atomic_write_json(run_config_path, run_config)
        config_summaries.append(
            _config_summary(
                config_id=config_id,
                strategy=strategy,
                mode=mode,
                metrics=metrics,
                latency=latency,
                run_config=run_config,
            )
        )

    matrix_status = (
        "completed"
        if all(
            summary["status"] == "completed"
            for summary in config_summaries
        )
        else "failed"
    )
    matrix_summary = {
        "version": "v1.5.0",
        "matrix_id": matrix_id,
        "status": matrix_status,
        "config_count": len(config_summaries),
        "completed_config_count": sum(
            summary["status"] == "completed"
            for summary in config_summaries
        ),
        "failed_config_count": sum(
            summary["status"] == "failed"
            for summary in config_summaries
        ),
        "input_hashes": input_hashes,
        "configs": config_summaries,
    }
    _atomic_write_json(
        matrix_dir / "matrix-summary.json",
        matrix_summary,
    )
    matrix_config = {
        **matrix_config,
        "status": matrix_status,
    }
    _atomic_write_json(
        matrix_dir / "matrix-config.json",
        matrix_config,
    )
    return {
        **matrix_summary,
        "matrix_dir": str(matrix_dir.resolve()),
    }


def _validate_pilot_manifest_hashes(
    *,
    pilot_manifest: dict,
    pilot_manifest_sha256: str,
    input_hashes: dict,
) -> None:
    if pilot_manifest.get("status") != "ready":
        raise ValueError("Pilot Manifest 必须为 ready")
    if (
        input_hashes.get("pilot_manifest_sha256")
        != pilot_manifest_sha256
    ):
        raise ValueError("matrix input hash 与 Pilot Manifest 不一致")
    dataset = pilot_manifest.get("dataset", {})
    if (
        dataset.get("question_count") != 40
        or dataset.get("approved_count") != 40
        or dataset.get("sha256") != input_hashes.get("dataset_sha256")
    ):
        raise ValueError("Pilot dataset 与 matrix input hash 不一致")
    pilot_inputs = pilot_manifest.get("inputs", {})
    for manifest_key, input_hash_key in PILOT_INPUT_HASH_FIELDS.items():
        record = pilot_inputs.get(manifest_key)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != input_hashes.get(input_hash_key)
        ):
            raise ValueError(
                f"Pilot Manifest input hash 不一致: {manifest_key}"
            )


def _validate_frozen_config(
    *,
    config_dir: Path,
    expected: dict,
    config: dict,
    input_hashes: dict,
    summary: dict,
) -> dict:
    config_id = expected["config_id"]
    missing = [
        filename
        for filename in RUN_FILE_NAMES
        if not (config_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{config_id} 缺少运行文件: {', '.join(missing)}"
        )

    run_config = _read_json(
        config_dir / "run-config.json",
        label=f"{config_id} run-config",
    )
    metrics = _read_json(
        config_dir / "metrics.json",
        label=f"{config_id} metrics",
    )
    latency = _read_json(
        config_dir / "latency.json",
        label=f"{config_id} latency",
    )
    records = _read_jsonl_strict(
        config_dir / "per-question.jsonl",
        label=f"{config_id} per-question",
    )
    errors = _read_jsonl_strict(
        config_dir / "errors.jsonl",
        label=f"{config_id} errors",
    )

    if (
        run_config.get("config_id") != config_id
        or run_config.get("strategy") != expected["strategy"]
        or run_config.get("mode") != expected["mode"]
    ):
        raise ValueError(f"{config_id} run-config 与固定矩阵不一致")
    if run_config.get("input_hashes") != input_hashes:
        raise ValueError(f"{config_id} input hash 不一致")
    if (
        run_config.get("config") != config
        or run_config.get("config_hash")
        != _config_hash(
            config=config,
            strategy=expected["strategy"],
            mode=expected["mode"],
        )
    ):
        raise ValueError(f"{config_id} run-config 与矩阵配置不一致")
    if (
        run_config.get("status") != "completed"
        or run_config.get("completed_count") != 40
        or run_config.get("error_count") != 0
    ):
        raise ValueError(f"{config_id} 未达到 completed 40/40、0 error")
    question_ids = [record.get("question_id") for record in records]
    if (
        len(records) != 40
        or len(set(question_ids)) != 40
        or any(record.get("config_id") != config_id for record in records)
    ):
        raise ValueError(f"{config_id} per-question 必须为 40 个唯一问题")
    if errors:
        raise ValueError(f"{config_id} errors.jsonl 必须为空")
    if (
        metrics.get("status") != "completed"
        or metrics.get("completed_count") != 40
        or metrics.get("error_count") != 0
        or metrics.get("question_count") != 40
    ):
        raise ValueError(f"{config_id} metrics 未达到 completed 40/40、0 error")
    if (
        metrics.get("top5_traceability_rate") != 1.0
        or (
            expected["strategy"] == "c4"
            and metrics.get("c4_parent_recovery_rate") != 1.0
        )
    ):
        raise ValueError(f"{config_id} 可追溯性或 Parent recovery 未通过")

    comparable_fields = (
        *CORE_METRIC_FIELDS,
        "status",
        "completed_count",
        "error_count",
        "question_count",
        "top5_traceability_rate",
        "c4_parent_recovery_rate",
        "by_book",
        "by_question_type",
        "no_answer_score_distribution",
    )
    if any(
        summary.get(field) != metrics.get(field)
        for field in comparable_fields
    ):
        raise ValueError(f"{config_id} matrix summary 与 metrics 不一致")
    if summary.get("latency") != latency.get("summary"):
        raise ValueError(f"{config_id} matrix summary 与 latency 不一致")
    runtime = run_config.get("runtime", {})
    for field in (
        "index_size_bytes",
        "index_load_ms",
        "embedding_load_ms",
        "reranker_load_ms",
        "warmup_ms",
    ):
        if summary.get(field) != runtime.get(field):
            raise ValueError(f"{config_id} matrix summary 与 runtime 不一致")

    return {
        "config_id": config_id,
        "strategy": expected["strategy"],
        "mode": expected["mode"],
        "run_id": run_config.get("run_id"),
        "status": "completed",
        "completed_count": 40,
        "error_count": 0,
        "core_metrics": {
            field: metrics[field] for field in CORE_METRIC_FIELDS
        },
        "by_book": metrics["by_book"],
        "by_question_type": metrics["by_question_type"],
        "no_answer_score_distribution": metrics[
            "no_answer_score_distribution"
        ],
        "top5_traceability_rate": metrics[
            "top5_traceability_rate"
        ],
        "c4_parent_recovery_rate": metrics[
            "c4_parent_recovery_rate"
        ],
        "latency": latency["summary"],
        "index_size_bytes": runtime["index_size_bytes"],
        "runtime": {
            field: runtime[field]
            for field in (
                "index_load_ms",
                "embedding_load_ms",
                "reranker_load_ms",
                "warmup_ms",
            )
        },
        "files": {
            filename: {
                "path": f"{config_id}/{filename}",
                "sha256": _sha256_file(config_dir / filename),
            }
            for filename in RUN_FILE_NAMES
        },
    }


def freeze_pilot_runs(
    *,
    run_dir: Path,
    pilot_manifest_path: Path,
    output_path: Path,
) -> dict:
    run_dir = run_dir.resolve()
    pilot_manifest_path = pilot_manifest_path.resolve()
    repository_root = Path.cwd().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Pilot run 目录不存在: {run_dir}")

    pilot_manifest = _read_json(
        pilot_manifest_path,
        label="Pilot Manifest",
    )
    matrix_config_path = run_dir / "matrix-config.json"
    matrix_summary_path = run_dir / "matrix-summary.json"
    matrix_config = _read_json(
        matrix_config_path,
        label="matrix-config",
    )
    matrix_summary = _read_json(
        matrix_summary_path,
        label="matrix-summary",
    )
    expected_matrix = _matrix_definition()
    if matrix_config.get("matrix") != expected_matrix:
        raise ValueError("运行目录不是固定矩阵")
    if (
        matrix_config.get("matrix_id") != matrix_summary.get("matrix_id")
        or matrix_config.get("version") != "v1.5.0"
        or matrix_summary.get("version") != "v1.5.0"
    ):
        raise ValueError("matrix config 与 summary 标识不一致")
    if (
        matrix_config.get("status") != "completed"
        or matrix_summary.get("status") != "completed"
        or matrix_summary.get("config_count") != 8
        or matrix_summary.get("completed_config_count") != 8
        or matrix_summary.get("failed_config_count") != 0
    ):
        raise ValueError("Pilot 矩阵必须为 8 个 completed config")

    summaries = matrix_summary.get("configs")
    if not isinstance(summaries, list):
        raise ValueError("matrix summary configs 必须为列表")
    summary_ids = [summary.get("config_id") for summary in summaries]
    if len(summary_ids) != len(set(summary_ids)):
        raise ValueError("matrix summary 存在重复 config")
    if summary_ids != [entry["config_id"] for entry in expected_matrix]:
        raise ValueError("matrix summary 与固定矩阵不一致")

    input_hashes = matrix_config.get("input_hashes")
    if (
        not isinstance(input_hashes, dict)
        or matrix_summary.get("input_hashes") != input_hashes
    ):
        raise ValueError("matrix input hash 不一致")
    pilot_manifest_sha256 = _sha256_file(pilot_manifest_path)
    _validate_pilot_manifest_hashes(
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=pilot_manifest_sha256,
        input_hashes=input_hashes,
    )

    configs = []
    for expected, summary in zip(expected_matrix, summaries):
        if (
            summary.get("strategy") != expected["strategy"]
            or summary.get("mode") != expected["mode"]
        ):
            raise ValueError(
                f"{expected['config_id']} 与固定矩阵不一致"
            )
        configs.append(
            _validate_frozen_config(
                config_dir=run_dir / expected["config_id"],
                expected=expected,
                config=matrix_config["config"],
                input_hashes=input_hashes,
                summary=summary,
            )
        )

    manifest_core = {
        "version": "v1.5.0",
        "status": "ready",
        "matrix_id": matrix_summary.get("matrix_id"),
        "run_dir": _display_path(
            run_dir,
            repository_root=repository_root,
        ),
        "config_count": 8,
        "completed_config_count": 8,
        "failed_config_count": 0,
        "input_hashes": input_hashes,
        "matrix_files": {
            "matrix-config.json": {
                "path": "matrix-config.json",
                "sha256": _sha256_file(matrix_config_path),
            },
            "matrix-summary.json": {
                "path": "matrix-summary.json",
                "sha256": _sha256_file(matrix_summary_path),
            },
        },
        "configs": configs,
        "privacy": {
            "full_results_committed": False,
            "contains_question_text": False,
            "contains_hit_text": False,
            "contains_manual_comments": False,
        },
    }
    if output_path.is_file():
        existing = _read_json(output_path, label="Pilot runs Manifest")
        existing_core = {
            key: value
            for key, value in existing.items()
            if key != "frozen_at"
        }
        if existing_core != manifest_core:
            raise ValueError("Pilot runs Manifest 已冻结，拒绝覆盖变更")
        return existing

    manifest = {
        **manifest_core,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(output_path, manifest)
    return manifest
