import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from experiments.rag_v1_5.dataset import load_dataset
from experiments.rag_v1_5.indexing import BgeM3DenseEncoder
from experiments.rag_v1_5.metrics import (
    _ndcg,
    _recall_at,
    _reciprocal_rank,
    index_size_bytes,
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
from experiments.rag_v1_5.runner import (
    RUN_FILE_NAMES,
    WARMUP_QUERY,
    PilotConfigRuntime,
    _append_jsonl,
    _atomic_write_json,
    _compact_timestamp,
    _config_summary,
    _cuda_available,
    _json_sha256,
    _metric_payload,
    _read_json,
    _read_jsonl_strict,
    _sha256_file,
)
from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit


@dataclass(frozen=True)
class FormalMatrixConfig:
    config_id: str
    paper_role: str
    strategy: str
    mode: str
    context_policy: str
    metadata_policy: str


FORMAL_RETRIEVAL_MATRIX = (
    FormalMatrixConfig(
        "b1-c0-bm25",
        "B1",
        "c0",
        "bm25",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "b2-c0-dense",
        "B2",
        "c0",
        "dense",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "b3-c0-hybrid",
        "B3",
        "c0",
        "hybrid",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "b4-c0-hybrid-rerank",
        "B4/C0",
        "c0",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "c1-hybrid-rerank",
        "C1",
        "c1",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "c2-hybrid-rerank",
        "C2",
        "c2",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "c3-hybrid-rerank",
        "C3",
        "c3",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-c4-hybrid-rerank",
        "P/C4",
        "c4",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-parent",
        "P-Parent",
        "c4",
        "hybrid_rerank",
        "child",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-structure",
        "P-Structure",
        "c5",
        "hybrid_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-bm25",
        "P-BM25",
        "c4",
        "dense_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-dense",
        "P-Dense",
        "c4",
        "bm25_rerank",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-reranker",
        "P-Reranker",
        "c4",
        "hybrid",
        "parent",
        "with_titles",
    ),
    FormalMatrixConfig(
        "p-no-title",
        "P-Title",
        "c4",
        "hybrid_rerank",
        "parent",
        "without_titles",
    ),
)
FormalConfigRuntime = PilotConfigRuntime


def _matrix_definition() -> list[dict]:
    return [
        {
            "config_id": row.config_id,
            "paper_role": row.paper_role,
            "strategy": row.strategy,
            "mode": row.mode,
            "context_policy": row.context_policy,
            "metadata_policy": row.metadata_policy,
        }
        for row in FORMAL_RETRIEVAL_MATRIX
    ]


def _resolve_record_path(record: dict, root: Path) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def validate_formal_runtime_inputs(
    *,
    dataset_path: Path,
    formal_manifest_path: Path,
    prereg_manifest_path: Path,
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
    if not cuda_checker():
        raise ValueError("Formal 真实运行要求 CUDA 可用")
    formal_manifest = _read_json(
        formal_manifest_path,
        label="Formal Manifest",
    )
    prereg_manifest = _read_json(
        prereg_manifest_path,
        label="Formal prereg Manifest",
    )
    if formal_manifest.get("status") != "ready":
        raise ValueError("Formal Manifest 必须为 ready")
    if prereg_manifest.get("status") != "ready":
        raise ValueError("Formal prereg Manifest 必须为 ready")

    expected_dataset_hash = formal_manifest.get("dataset", {}).get(
        "sha256"
    )
    if _sha256_file(dataset_path) != expected_dataset_hash:
        raise ValueError("Formal dataset 哈希不一致")
    prereg_record = formal_manifest.get("inputs", {}).get(
        "prereg_manifest",
        {},
    )
    if _sha256_file(prereg_manifest_path) != prereg_record.get(
        "sha256"
    ):
        raise ValueError("Formal prereg 哈希不一致")
    config_record = prereg_manifest.get("inputs", {}).get("config", {})
    if _sha256_file(config_path) != config_record.get("sha256"):
        raise ValueError("Formal config 哈希不一致")

    model_manifest_path = (
        repository_root
        / "experiments/rag_v1_5/manifests/models-v1.5.0.json"
    )
    model_manifest_sha256 = _sha256_file(model_manifest_path)
    if (
        prereg_manifest.get("models", {}).get(
            "pilot_model_manifest_sha256"
        )
        != model_manifest_sha256
    ):
        raise ValueError("Formal model manifest 哈希不一致")

    index_manifest_path = indexes_dir / "manifest.json"
    index_manifest = _read_json(
        index_manifest_path,
        label="Formal index Manifest",
    )
    if index_manifest.get("status") != "ready":
        raise ValueError("Formal index Manifest 必须为 ready")
    formal_manifest_sha256 = _sha256_file(formal_manifest_path)
    if (
        index_manifest.get("formal_manifest_sha256")
        != formal_manifest_sha256
    ):
        raise ValueError("Formal index 与 Formal Manifest 哈希不一致")
    chunk_record = index_manifest.get("chunk_manifest", {})
    chunk_manifest_path = _resolve_record_path(
        chunk_record,
        repository_root,
    )
    if _sha256_file(chunk_manifest_path) != chunk_record.get("sha256"):
        raise ValueError("Formal chunk manifest 哈希不一致")
    if (
        index_manifest.get("model_manifest_sha256")
        != model_manifest_sha256
    ):
        raise ValueError("Formal index 与 model manifest 不一致")

    strategy_hashes = {}
    for key in (
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c4-no-title",
    ):
        record = index_manifest.get("strategies", {}).get(key, {})
        strategy_path = indexes_dir / key / "manifest.json"
        actual = _sha256_file(strategy_path)
        if record.get("manifest_sha256") != actual:
            raise ValueError(f"{key} 索引 Manifest 哈希不一致")
        strategy_hashes[key] = actual

    questions = load_dataset(dataset_path)
    if len(questions) != 400 or any(
        question.review_status != "approved" for question in questions
    ):
        raise ValueError("Formal runtime 要求 400 条 approved 问题")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Formal config 顶层必须为 mapping")

    return {
        "formal_manifest_status": formal_manifest["status"],
        "questions": questions,
        "config": config,
        "input_hashes": {
            "formal_manifest_sha256": formal_manifest_sha256,
            "prereg_manifest_sha256": _sha256_file(
                prereg_manifest_path
            ),
            "dataset_sha256": expected_dataset_hash,
            "config_sha256": _sha256_file(config_path),
            "chunk_manifest_sha256": _sha256_file(
                chunk_manifest_path
            ),
            "index_manifest_sha256": _sha256_file(
                index_manifest_path
            ),
            "model_manifest_sha256": model_manifest_sha256,
            "strategy_manifest_sha256": strategy_hashes,
        },
        "paths": {"model_manifest": model_manifest_path},
    }


class DefaultFormalRuntimeFactory:
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
        row: FormalMatrixConfig,
        config: dict,
    ) -> FormalConfigRuntime:
        index_key = (
            "c4-no-title"
            if row.metadata_policy == "without_titles"
            else row.strategy
        )
        started = time.perf_counter()
        index = load_index(self.indexes_dir / index_key)
        index_load_ms = (time.perf_counter() - started) * 1000

        embedding_load_ms = 0.0
        if (
            row.mode
            in {
                "dense",
                "hybrid",
                "hybrid_rerank",
                "dense_rerank",
            }
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
        if (
            row.mode
            in {"hybrid_rerank", "bm25_rerank", "dense_rerank"}
            and self.reranker_scorer is None
        ):
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

        def retrieve_question(question: PilotQuestion) -> RetrievalResult:
            return retrieve_loaded(
                question.question,
                index=index,
                mode=row.mode,
                config=config,
                dense_encoder=self.dense_encoder,
                reranker_scorer=self.reranker_scorer,
                result_top_k=10,
                context_policy=row.context_policy,
            )

        started = time.perf_counter()
        retrieve_loaded(
            WARMUP_QUERY,
            index=index,
            mode=row.mode,
            config=config,
            dense_encoder=self.dense_encoder,
            reranker_scorer=self.reranker_scorer,
            result_top_k=10,
            context_policy=row.context_policy,
        )
        warmup_ms = (time.perf_counter() - started) * 1000
        return FormalConfigRuntime(
            retrieve=retrieve_question,
            index_load_ms=index_load_ms,
            embedding_load_ms=embedding_load_ms,
            reranker_load_ms=reranker_load_ms,
            warmup_ms=warmup_ms,
            index_size_bytes=index_size_bytes(
                self.indexes_dir / index_key
            ),
        )


def _parent_recovery_ok(
    hits: list[RetrievalHit],
    row: FormalMatrixConfig,
) -> bool | None:
    if (
        row.context_policy != "parent"
        or row.strategy not in {"c4", "c5"}
    ):
        return None
    top5 = hits[:5]
    return len(top5) == 5 and all(
        hit.retrieval_parent_id and hit.context_text.strip()
        for hit in top5
    )


def _question_record(
    *,
    row: FormalMatrixConfig,
    question: PilotQuestion,
    result: RetrievalResult,
) -> dict:
    hits = [RetrievalHit.model_validate(hit) for hit in result.hits]
    top5 = hits[:5]
    traceability = len(top5) == 5 and all(
        hit.chunk_id
        and hit.source_evidence_ids
        and hit.clause_ids
        and hit.context_text.strip()
        for hit in top5
    )
    per_question_metrics = {
        "recall_at_5": None,
        "mrr_at_10": None,
        "ndcg_at_10": None,
    }
    if question.answerable:
        per_question_metrics = {
            "recall_at_5": _recall_at(question, hits, 5),
            "mrr_at_10": _reciprocal_rank(question, hits, 10),
            "ndcg_at_10": _ndcg(question, hits, 10),
        }
    return {
        "config_id": row.config_id,
        "question_id": question.question_id,
        "split": question.split,
        "answerable": question.answerable,
        "book_scope": question.book_scope,
        "question_type": question.question_type,
        **per_question_metrics,
        "hits": [hit.model_dump(mode="json") for hit in hits],
        "latency": result.latency,
        "top5_traceability_ok": traceability,
        "c4_parent_recovery_ok": _parent_recovery_ok(hits, row),
    }


def _config_hash(config: dict, row: FormalMatrixConfig) -> str:
    return _json_sha256(
        {
            "config": config,
            "matrix": _matrix_definition(),
            "row": {
                "config_id": row.config_id,
                "paper_role": row.paper_role,
                "strategy": row.strategy,
                "mode": row.mode,
                "context_policy": row.context_policy,
                "metadata_policy": row.metadata_policy,
            },
        }
    )


def run_formal_matrix(
    *,
    split: str,
    dataset_path: Path,
    formal_manifest_path: Path,
    prereg_manifest_path: Path,
    config_path: Path,
    indexes_dir: Path,
    output_dir: Path,
    resume_dir: Path | None = None,
    repository_root: Path | None = None,
    input_validator: Callable[..., dict] = (
        validate_formal_runtime_inputs
    ),
    runtime_factory: Callable | None = None,
    cuda_checker: Callable[[], bool] = _cuda_available,
    now: datetime | None = None,
) -> dict:
    if split not in {"formal_dev", "formal_test"}:
        raise ValueError("split 必须为 formal_dev 或 formal_test")
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    validated = input_validator(
        dataset_path=dataset_path,
        formal_manifest_path=formal_manifest_path,
        prereg_manifest_path=prereg_manifest_path,
        config_path=config_path,
        indexes_dir=indexes_dir,
        repository_root=repository_root,
        cuda_checker=cuda_checker,
    )
    if validated.get("formal_manifest_status") != "ready":
        raise ValueError("Formal Manifest 必须为 ready")
    questions = sorted(
        [
            question
            for question in validated["questions"]
            if question.split == split
        ],
        key=lambda question: question.question_id,
    )
    if len(questions) != 200 or any(
        question.review_status != "approved" for question in questions
    ):
        raise ValueError(f"{split} 必须包含 200 条 approved 问题")
    question_ids = {question.question_id for question in questions}
    if len(question_ids) != 200:
        raise ValueError(f"{split} 存在重复 question_id")
    questions_by_id = {
        question.question_id: question for question in questions
    }
    config = validated["config"]
    input_hashes = validated["input_hashes"]
    matrix = _matrix_definition()
    current_time = now or datetime.now(timezone.utc)
    run_timestamp = _compact_timestamp(current_time)

    if resume_dir is None:
        matrix_id = (
            f"{split}-{run_timestamp}-"
            f"{input_hashes['dataset_sha256'][:8]}-"
            f"{input_hashes['config_sha256'][:8]}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_dir = output_dir / matrix_id
        matrix_dir.mkdir(exist_ok=False)
        matrix_config = {
            "version": "v1.5.0",
            "status": "running",
            "split": split,
            "matrix_id": matrix_id,
            "created_at": current_time.isoformat(),
            "run_timestamp": run_timestamp,
            "matrix": matrix,
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
            label="formal matrix-config",
        )
        if (
            matrix_config.get("split") != split
            or matrix_config.get("input_hashes") != input_hashes
            or matrix_config.get("config") != config
            or matrix_config.get("matrix") != matrix
        ):
            raise ValueError("恢复运行的 split、哈希或矩阵不一致")
        matrix_id = matrix_config["matrix_id"]

    if runtime_factory is None:
        runtime_factory = DefaultFormalRuntimeFactory(
            indexes_dir=indexes_dir.resolve(),
            model_manifest_path=validated["paths"]["model_manifest"],
            repository_root=repository_root,
        )

    summaries = []
    for row_index, row in enumerate(FORMAL_RETRIEVAL_MATRIX):
        config_dir = matrix_dir / row.config_id
        config_dir.mkdir(exist_ok=resume_dir is not None)
        per_question_path = config_dir / "per-question.jsonl"
        errors_path = config_dir / "errors.jsonl"
        run_config_path = config_dir / "run-config.json"
        per_question_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)

        existing_records = _read_jsonl_strict(
            per_question_path,
            label=f"{row.config_id} per-question",
        )
        completed_ids = {
            record.get("question_id") for record in existing_records
        }
        if (
            len(completed_ids) != len(existing_records)
            or not completed_ids <= question_ids
        ):
            raise ValueError(f"{row.config_id} per-question 非法")
        error_records = _read_jsonl_strict(
            errors_path,
            label=f"{row.config_id} errors",
        )
        attempts = Counter(
            record.get("question_id") for record in error_records
        )
        config_hash = _config_hash(config, row)
        existing_run_config = (
            _read_json(
                run_config_path,
                label=f"{row.config_id} run-config",
            )
            if run_config_path.is_file()
            else None
        )
        if existing_run_config is not None and (
            existing_run_config.get("input_hashes") != input_hashes
            or existing_run_config.get("config_hash") != config_hash
        ):
            raise ValueError(f"{row.config_id} run-config 哈希不一致")

        if len(completed_ids) < 200:
            runtime = runtime_factory(row, config)
            run_config = {
                "version": "v1.5.0",
                "status": "running",
                "run_id": (
                    existing_run_config.get("run_id")
                    if existing_run_config
                    else f"{matrix_id}-{row.config_id}"
                ),
                **matrix[row_index],
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
            ) as per_handle, errors_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as error_handle:
                for question in questions:
                    if question.question_id in completed_ids:
                        continue
                    try:
                        record = _question_record(
                            row=row,
                            question=question,
                            result=runtime.retrieve(question),
                        )
                        _append_jsonl(per_handle, record)
                        completed_ids.add(question.question_id)
                    except Exception as error:
                        _append_jsonl(
                            error_handle,
                            {
                                "config_id": row.config_id,
                                "question_id": question.question_id,
                                "attempt": (
                                    attempts[question.question_id] + 1
                                ),
                                "error_type": type(error).__name__,
                                "message": str(error),
                                "recorded_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
        elif existing_run_config is None:
            raise ValueError(
                f"{row.config_id} 已有结果但缺少 run-config"
            )
        else:
            run_config = existing_run_config

        records = _read_jsonl_strict(
            per_question_path,
            label=f"{row.config_id} per-question",
        )
        error_records = _read_jsonl_strict(
            errors_path,
            label=f"{row.config_id} errors",
        )
        metrics, latency = _metric_payload(
            questions_by_id=questions_by_id,
            records=records,
            error_records=error_records,
        )
        metrics["parent_recovery_rate"] = metrics[
            "c4_parent_recovery_rate"
        ]
        run_config = {
            **run_config,
            "status": metrics["status"],
            "completed_count": metrics["completed_count"],
            "error_count": metrics["error_count"],
        }
        _atomic_write_json(config_dir / "metrics.json", metrics)
        _atomic_write_json(config_dir / "latency.json", latency)
        _atomic_write_json(run_config_path, run_config)
        summary = _config_summary(
            config_id=row.config_id,
            strategy=row.strategy,
            mode=row.mode,
            metrics=metrics,
            latency=latency,
            run_config=run_config,
        )
        summary.update(
            paper_role=row.paper_role,
            context_policy=row.context_policy,
            metadata_policy=row.metadata_policy,
            parent_recovery_rate=metrics["parent_recovery_rate"],
        )
        summaries.append(summary)

    status = (
        "completed"
        if all(item["status"] == "completed" for item in summaries)
        else "failed"
    )
    matrix_summary = {
        "version": "v1.5.0",
        "matrix_id": matrix_id,
        "split": split,
        "status": status,
        "config_count": 14,
        "completed_config_count": sum(
            item["status"] == "completed" for item in summaries
        ),
        "failed_config_count": sum(
            item["status"] == "failed" for item in summaries
        ),
        "total_question_runs": sum(
            item["completed_count"] for item in summaries
        ),
        "input_hashes": input_hashes,
        "configs": summaries,
    }
    _atomic_write_json(
        matrix_dir / "matrix-summary.json",
        matrix_summary,
    )
    _atomic_write_json(
        matrix_dir / "matrix-config.json",
        {**matrix_config, "status": status},
    )
    return {**matrix_summary, "matrix_dir": str(matrix_dir.resolve())}


def freeze_formal_runs(
    *,
    run_dir: Path,
    formal_manifest_path: Path,
    prereg_manifest_path: Path,
    output_path: Path,
) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    formal_manifest = _read_json(
        formal_manifest_path,
        label="Formal Manifest",
    )
    prereg_manifest = _read_json(
        prereg_manifest_path,
        label="Formal prereg Manifest",
    )
    if formal_manifest.get("status") != "ready":
        raise ValueError("Formal Manifest 必须为 ready")
    if prereg_manifest.get("status") != "ready":
        raise ValueError("Formal prereg Manifest 必须为 ready")

    matrix_config_path = run_dir / "matrix-config.json"
    matrix_summary_path = run_dir / "matrix-summary.json"
    matrix_config = _read_json(
        matrix_config_path,
        label="formal matrix-config",
    )
    matrix_summary = _read_json(
        matrix_summary_path,
        label="formal matrix-summary",
    )
    if (
        matrix_config.get("matrix") != _matrix_definition()
        or matrix_config.get("status") != "completed"
        or matrix_summary.get("status") != "completed"
        or matrix_summary.get("config_count") != 14
        or matrix_summary.get("completed_config_count") != 14
        or matrix_summary.get("failed_config_count") != 0
        or matrix_summary.get("total_question_runs") != 2800
    ):
        raise ValueError("Formal 矩阵未达到 14 x 200 完成门禁")
    input_hashes = matrix_config.get("input_hashes", {})
    if matrix_summary.get("input_hashes") != input_hashes:
        raise ValueError("Formal matrix input hashes 不一致")
    if (
        input_hashes.get("formal_manifest_sha256")
        != _sha256_file(formal_manifest_path)
        or input_hashes.get("prereg_manifest_sha256")
        != _sha256_file(prereg_manifest_path)
    ):
        raise ValueError("Formal 冻结清单哈希与运行输入不一致")

    summaries_by_id = {
        item["config_id"]: item
        for item in matrix_summary.get("configs", [])
    }
    if set(summaries_by_id) != {
        row.config_id for row in FORMAL_RETRIEVAL_MATRIX
    }:
        raise ValueError("Formal matrix summary 配置集合不一致")

    configs = []
    for row in FORMAL_RETRIEVAL_MATRIX:
        config_dir = run_dir / row.config_id
        missing = [
            filename
            for filename in RUN_FILE_NAMES
            if not (config_dir / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{row.config_id} 缺少运行文件: {missing}"
            )
        records = _read_jsonl_strict(
            config_dir / "per-question.jsonl",
            label=f"{row.config_id} per-question",
        )
        errors = _read_jsonl_strict(
            config_dir / "errors.jsonl",
            label=f"{row.config_id} errors",
        )
        metrics = _read_json(
            config_dir / "metrics.json",
            label=f"{row.config_id} metrics",
        )
        if (
            len(records) != 200
            or len(
                {record.get("question_id") for record in records}
            )
            != 200
            or errors
            or metrics.get("status") != "completed"
            or metrics.get("completed_count") != 200
            or metrics.get("error_count") != 0
            or metrics.get("top5_traceability_rate") != 1.0
        ):
            raise ValueError(f"{row.config_id} 未通过 200/200 门禁")
        if (
            row.context_policy == "parent"
            and row.strategy in {"c4", "c5"}
            and metrics.get("parent_recovery_rate") != 1.0
        ):
            raise ValueError(f"{row.config_id} Parent recovery 未通过")
        summary = summaries_by_id[row.config_id]
        configs.append(
            {
                "config_id": row.config_id,
                "paper_role": row.paper_role,
                "strategy": row.strategy,
                "mode": row.mode,
                "context_policy": row.context_policy,
                "metadata_policy": row.metadata_policy,
                "status": "completed",
                "completed_count": 200,
                "error_count": 0,
                "top5_traceability_rate": 1.0,
                "parent_recovery_rate": metrics.get(
                    "parent_recovery_rate"
                ),
                "core_metrics": {
                    key: metrics[key]
                    for key in (
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "hit_at_5",
                        "mrr_at_10",
                        "ndcg_at_10",
                    )
                },
                "latency": summary["latency"],
                "index_size_bytes": summary["index_size_bytes"],
                "files": {
                    filename: {
                        "path": f"{row.config_id}/{filename}",
                        "sha256": _sha256_file(
                            config_dir / filename
                        ),
                    }
                    for filename in RUN_FILE_NAMES
                },
            }
        )

    manifest_core = {
        "version": "v1.5.0",
        "status": "ready",
        "split": matrix_config["split"],
        "matrix_id": matrix_config["matrix_id"],
        "run_dir": run_dir.as_posix(),
        "config_count": 14,
        "completed_config_count": 14,
        "failed_config_count": 0,
        "total_question_runs": 2800,
        "input_hashes": input_hashes,
        "matrix_files": {
            "matrix-config.json": {
                "sha256": _sha256_file(matrix_config_path)
            },
            "matrix-summary.json": {
                "sha256": _sha256_file(matrix_summary_path)
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
    statistics_path = run_dir / "formal-statistics.json"
    if statistics_path.is_file():
        manifest_core["statistics"] = {
            "path": "formal-statistics.json",
            "sha256": _sha256_file(statistics_path),
        }
    if output_path.is_file():
        existing = _read_json(
            output_path,
            label="Formal runs Manifest",
        )
        existing_core = {
            key: value
            for key, value in existing.items()
            if key != "frozen_at"
        }
        if existing_core != manifest_core:
            raise ValueError(
                "Formal runs Manifest 已冻结，拒绝覆盖变更"
            )
        return existing

    manifest = {
        **manifest_core,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(output_path, manifest)
    return manifest
