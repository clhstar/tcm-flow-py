import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from experiments.rag_v1_5.runner import (
    _atomic_write_json,
    _json_sha256,
    _read_json,
    _read_jsonl_strict,
    _sha256_file,
)
from experiments.rag_v1_5.schema import (
    FormalAnswerOutput,
    FormalAnswerRunRecord,
)

ANSWER_RETRIEVAL_CONFIGS = {
    "B4": "b4-c0-hybrid-rerank",
    "P": "p-c4-hybrid-rerank",
    "P-no-parent": "p-no-parent",
}
ABSTAIN_ANSWER = "在指定古籍证据范围内未找到可靠答案。"
EVIDENCE_SYSTEM_PROMPT = """你是中医古籍证据问答评测模型。
对于 B4、P 和 P-no-parent，只能依据给定证据回答，不得补充证据之外的医学常识。
证据不足时必须拒答。
引用只能使用输入中给出的 E1、E2 等标签。
输出必须符合 JSON：{"answer":"回答正文","abstain":false,"citations":["E1"]}。
拒答时 answer 固定为“在指定古籍证据范围内未找到可靠答案。”，
abstain=true，citations=[]。"""
B0_SYSTEM_PROMPT = """你是中医古籍问答评测模型。
本配置不提供外部证据，可依据模型已有知识回答；不确定时必须拒答。
输出必须符合 JSON：{"answer":"回答正文","abstain":false,"citations":[]}。
citations 必须为空。
拒答时 answer 固定为“在指定古籍证据范围内未找到可靠答案。”，
abstain=true，citations=[]。"""


def build_answer_messages(
    *,
    question: str,
    evidence: list[dict],
    method: str,
) -> list[dict]:
    if method == "B0":
        if evidence:
            raise ValueError("B0 不允许传入检索证据")
        return [
            {"role": "system", "content": B0_SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}"},
        ]
    if method not in ANSWER_RETRIEVAL_CONFIGS:
        raise ValueError(f"不支持的回答方法: {method}")
    evidence_text = "\n\n".join(
        f"[{item['label']}]\n{item['text']}" for item in evidence
    )
    return [
        {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"问题：{question}\n\n证据：\n{evidence_text}",
        },
    ]


class FormalAnswerModel:
    def __init__(
        self,
        *,
        config: dict,
        env: Mapping[str, str] | None = None,
        chat_model_factory=None,
    ) -> None:
        if env is None:
            load_dotenv()
            environment = os.environ
        else:
            environment = env
        model_config = config["model"]
        self.model_name = environment[
            model_config["env_model_key"]
        ]
        base_url = environment[
            model_config["env_base_url_key"]
        ]
        if chat_model_factory is None:
            from langchain_openai import ChatOpenAI

            chat_model_factory = ChatOpenAI
        model = chat_model_factory(
            model=self.model_name,
            base_url=base_url,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            timeout=model_config["timeout_seconds"],
            max_retries=model_config["max_retries"],
        )
        self.structured = model.with_structured_output(
            FormalAnswerOutput,
            method=model_config["structured_output_method"],
            include_raw=True,
        )

    def invoke(
        self,
        messages: list[dict],
    ) -> tuple[FormalAnswerOutput, dict]:
        response = self.structured.invoke(messages)
        parsed = FormalAnswerOutput.model_validate(response["parsed"])
        metadata = (
            getattr(response["raw"], "response_metadata", {})
            or {}
        )
        usage = metadata.get("token_usage", {})
        return parsed, {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(
                usage.get("completion_tokens", 0)
            ),
            "system_fingerprint": metadata.get(
                "system_fingerprint"
            ),
        }


def calibrate_threshold(rows: list[dict]) -> dict:
    scored_rows = [
        {
            "answerable": bool(row["answerable"]),
            "score": float(row["score"]),
        }
        for row in rows
        if row.get("score") is not None
    ]
    positive_count = sum(
        row["answerable"] for row in scored_rows
    )
    negative_count = len(scored_rows) - positive_count
    if not positive_count or not negative_count:
        raise ValueError(
            "阈值校准需要同时包含 answerable 和 unanswerable"
        )
    scores = sorted({row["score"] for row in scored_rows})
    candidates = set(scores)
    candidates.update(
        (left + right) / 2
        for left, right in zip(scores, scores[1:])
    )
    evaluated = []
    for threshold in sorted(candidates):
        true_positive = false_positive = 0
        true_negative = false_negative = 0
        for row in scored_rows:
            predicted_answerable = row["score"] >= threshold
            if row["answerable"] and predicted_answerable:
                true_positive += 1
            elif row["answerable"]:
                false_negative += 1
            elif predicted_answerable:
                false_positive += 1
            else:
                true_negative += 1
        true_positive_rate = true_positive / positive_count
        true_negative_rate = true_negative / negative_count
        evaluated.append(
            {
                "threshold": threshold,
                "balanced_accuracy": (
                    true_positive_rate + true_negative_rate
                )
                / 2,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
            }
        )
    best = max(
        evaluated,
        key=lambda item: (
            item["balanced_accuracy"],
            item["threshold"],
        ),
    )
    return {
        "objective": "balanced_accuracy",
        "tie_break": "higher_threshold",
        "threshold": best["threshold"],
        "balanced_accuracy": best["balanced_accuracy"],
        "confusion": {
            key: best[key]
            for key in (
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            )
        },
        "scored_count": len(scored_rows),
    }


def build_evidence_items(
    record: dict,
    top_k: int = 5,
) -> list[dict]:
    config_id = record.get("config_id", "")
    dedupe_field = (
        "retrieval_parent_id"
        if config_id == ANSWER_RETRIEVAL_CONFIGS["P"]
        else "chunk_id"
    )
    items = []
    seen = set()
    for hit in record["hits"][:top_k]:
        dedupe_key = hit.get(dedupe_field) or hit["chunk_id"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        items.append(
            {
                "label": f"E{len(items) + 1}",
                "text": hit["context_text"],
                "chunk_id": hit["chunk_id"],
                "clause_ids": hit["clause_ids"],
                "retrieval_parent_id": hit.get(
                    "retrieval_parent_id"
                ),
                "reranker_score": hit.get("reranker_score"),
            }
        )
    return items


def load_frozen_answer_inputs(
    *,
    dataset_path: Path,
    matrix_dir: Path,
    answer_prereg_path: Path,
    split: str,
    formal_manifest_path: Path | None = None,
    formal_runs_manifest_path: Path | None = None,
) -> dict:
    if split not in {"formal_dev", "formal_test"}:
        raise ValueError(f"不支持的回答层 split: {split}")
    prereg = _read_json(
        answer_prereg_path,
        label="Formal answer prereg",
    )
    if prereg.get("status") != "ready":
        raise ValueError("Formal answer prereg 必须为 ready")

    matrix_config_path = matrix_dir / "matrix-config.json"
    matrix_config = _read_json(
        matrix_config_path,
        label="Formal retrieval matrix config",
    )
    if matrix_config.get("status") != "completed":
        raise ValueError("Formal retrieval matrix 必须为 completed")
    if matrix_config.get("split") != split:
        raise ValueError("回答层 split 与检索矩阵不一致")
    prereg_matrix_hash_key = (
        "dev_matrix_config_sha256"
        if split == "formal_dev"
        else "test_matrix_config_sha256"
    )
    if (
        _sha256_file(matrix_config_path)
        != prereg["inputs"][prereg_matrix_hash_key]
    ):
        raise ValueError("检索 matrix-config 哈希与回答层预注册不一致")

    matrix_input_hashes = matrix_config.get("input_hashes", {})
    if (
        _sha256_file(dataset_path)
        != matrix_input_hashes.get("dataset_sha256")
    ):
        raise ValueError("Formal 数据集哈希与检索矩阵不一致")
    if (
        prereg["inputs"].get("formal_manifest_sha256")
        != matrix_input_hashes.get("formal_manifest_sha256")
    ):
        raise ValueError("Formal Manifest 哈希链不一致")
    if formal_manifest_path is not None and (
        _sha256_file(formal_manifest_path)
        != prereg["inputs"]["formal_manifest_sha256"]
    ):
        raise ValueError("Formal Manifest 当前文件哈希已漂移")
    if formal_runs_manifest_path is not None and (
        _sha256_file(formal_runs_manifest_path)
        != prereg["inputs"]["formal_runs_manifest_sha256"]
    ):
        raise ValueError("Formal runs Manifest 当前文件哈希已漂移")

    question_rows = _read_jsonl_strict(
        dataset_path,
        label="Formal answer dataset",
    )
    questions = {}
    for row in question_rows:
        if row.get("split") != split:
            continue
        if row.get("review_status") != "approved":
            raise ValueError("回答层只允许 approved Formal 问题")
        question_id = row["question_id"]
        if question_id in questions:
            raise ValueError(f"Formal 问题 ID 重复: {question_id}")
        questions[question_id] = row
    if not questions:
        raise ValueError(f"{split} 没有 approved 问题")

    retrieval = {}
    raw_records = {}
    expected_question_ids = set(questions)
    for method, config_id in ANSWER_RETRIEVAL_CONFIGS.items():
        rows = _read_jsonl_strict(
            matrix_dir / config_id / "per-question.jsonl",
            label=f"{config_id} per-question",
        )
        records = {}
        for row in rows:
            if row.get("config_id") != config_id:
                raise ValueError(f"{config_id} 记录配置 ID 不一致")
            question_id = row["question_id"]
            if question_id in records:
                raise ValueError(
                    f"{config_id} question ID 重复: {question_id}"
                )
            records[question_id] = row
        if set(records) != expected_question_ids:
            raise ValueError(
                f"{config_id} 问题集合与 {split} 数据集不一致"
            )
        raw_records[method] = records
        retrieval[method] = {
            question_id: {
                "evidence": build_evidence_items(record),
                "top_score": (
                    record["hits"][0].get("reranker_score")
                    if record.get("hits")
                    else None
                ),
            }
            for question_id, record in records.items()
        }

    for question_id in expected_question_ids:
        p_ids = [
            hit["chunk_id"]
            for hit in raw_records["P"][question_id]["hits"]
        ]
        child_ids = [
            hit["chunk_id"]
            for hit in raw_records["P-no-parent"][question_id]["hits"]
        ]
        if p_ids != child_ids:
            raise ValueError(
                "P-no-parent 必须复用 P 的完全相同检索排序"
            )

    return {
        "split": split,
        "questions": questions,
        "retrieval": retrieval,
        "question_count": len(questions),
        "input_hashes": {
            "dataset_sha256": _sha256_file(dataset_path),
            "matrix_config_sha256": _sha256_file(
                matrix_config_path
            ),
            "answer_prereg_sha256": _sha256_file(
                answer_prereg_path
            ),
        },
    }


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as file_handle:
        file_handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            + "\n"
        )


def _completed_answer_run_exists(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    for child in output_dir.iterdir():
        summary_path = child / "matrix-summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(
            summary_path,
            label="Formal answer matrix summary",
        )
        if summary.get("status") == "completed":
            return True
    return False


def _validate_answer_output(
    *,
    output: FormalAnswerOutput,
    method: str,
    evidence: list[dict],
) -> None:
    if output.abstain and output.answer != ABSTAIN_ANSWER:
        raise ValueError("拒答 answer 必须使用冻结文本")
    if method == "B0":
        if output.citations:
            raise ValueError("B0 citations 必须为空")
        return
    allowed_labels = {item["label"] for item in evidence}
    if not output.abstain and not output.citations:
        raise ValueError("证据回答必须包含 citation")
    if not set(output.citations).issubset(allowed_labels):
        raise ValueError("回答引用了输入中不存在的证据标签")


def canonicalize_answer_output(
    output: FormalAnswerOutput,
) -> FormalAnswerOutput:
    if not output.abstain:
        return output
    return output.model_copy(
        update={
            "answer": ABSTAIN_ANSWER,
            "citations": [],
        }
    )


def run_formal_answer_matrix(
    *,
    split: str,
    output_dir: Path,
    dataset_path: Path = Path(
        "data/rag_v1_5/formal/evaluation/formal-400.jsonl"
    ),
    matrix_dir: Path | None = None,
    answer_prereg_path: Path = Path(
        "experiments/rag_v1_5/manifests/"
        "formal-answer-prereg-v1.5.0.json"
    ),
    config_path: Path = Path(
        "experiments/rag_v1_5/configs/formal-answer.yaml"
    ),
    formal_manifest_path: Path = Path(
        "experiments/rag_v1_5/manifests/formal-400-v1.5.0.json"
    ),
    formal_runs_manifest_path: Path = Path(
        "experiments/rag_v1_5/manifests/formal-runs-v1.5.0.json"
    ),
    dev_freeze_path: Path | None = None,
    resume_dir: Path | None = None,
    model_factory=None,
) -> dict:
    if split not in {"formal_dev", "formal_test"}:
        raise ValueError(f"不支持的回答层 split: {split}")
    if split == "formal_test" and (
        dev_freeze_path is None or not dev_freeze_path.is_file()
    ):
        raise ValueError("formal_test 要求已冻结 dev-freeze.json")
    if (
        split == "formal_test"
        and resume_dir is None
        and _completed_answer_run_exists(output_dir)
    ):
        raise ValueError("formal_test 已完成，禁止第二次 fresh run")

    if matrix_dir is None:
        matrix_dir = Path(
            "data/rag_v1_5/formal/runs/dev/"
            "formal_dev-20260615T100221Z-1C344CB2-D832EF32"
            if split == "formal_dev"
            else "data/rag_v1_5/formal/runs/test/"
            "formal_test-20260615T102626Z-1C344CB2-D832EF32"
        )
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    prereg = _read_json(
        answer_prereg_path,
        label="Formal answer prereg",
    )
    if _sha256_file(config_path) != prereg["inputs"].get(
        "config_sha256"
    ):
        raise ValueError("回答配置哈希与预注册不一致")
    methods = list(config["generation"]["answer_methods"])
    repeats = int(config["generation"]["repeats"])
    if methods != prereg.get("methods") or repeats != prereg.get(
        "repeats"
    ):
        raise ValueError("回答方法或重复次数与预注册不一致")

    loaded = load_frozen_answer_inputs(
        dataset_path=dataset_path,
        matrix_dir=matrix_dir,
        answer_prereg_path=answer_prereg_path,
        split=split,
        formal_manifest_path=formal_manifest_path,
        formal_runs_manifest_path=formal_runs_manifest_path,
    )
    thresholds = {}
    dev_freeze = None
    if split == "formal_test":
        dev_freeze = _read_json(
            dev_freeze_path,
            label="Formal answer dev freeze",
        )
        if dev_freeze.get("status") != "ready":
            raise ValueError("formal_test 要求 ready dev freeze")
        if dev_freeze["inputs"].get(
            "answer_prereg_sha256"
        ) != _sha256_file(answer_prereg_path):
            raise ValueError("dev freeze 与回答预注册哈希不一致")
        if dev_freeze["inputs"].get(
            "config_sha256"
        ) != _sha256_file(config_path):
            raise ValueError("dev freeze 与回答配置哈希不一致")
        thresholds = {
            "B4": dev_freeze["thresholds"]["B4"]["threshold"],
            "P": dev_freeze["thresholds"]["P"]["threshold"],
        }

    input_hashes = {
        **loaded["input_hashes"],
        "config_sha256": _sha256_file(config_path),
        "formal_manifest_sha256": _sha256_file(
            formal_manifest_path
        ),
        "formal_runs_manifest_sha256": _sha256_file(
            formal_runs_manifest_path
        ),
    }
    if dev_freeze_path is not None and dev_freeze_path.is_file():
        input_hashes["dev_freeze_sha256"] = _sha256_file(
            dev_freeze_path
        )

    if resume_dir is not None:
        run_dir = resume_dir
        run_config = _read_json(
            run_dir / "run-config.json",
            label="Formal answer run config",
        )
        if run_config.get("status") == "completed":
            raise ValueError("已完成回答矩阵不能 resume")
        if (
            run_config.get("split") != split
            or run_config.get("input_hashes") != input_hashes
        ):
            raise ValueError("resume 输入哈希或 split 不一致")
    else:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        run_id = (
            f"{split.replace('formal_', 'formal_answer_')}-"
            f"{timestamp}-{input_hashes['answer_prereg_sha256'][:8]}"
        )
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_config = {
            "version": config["version"],
            "status": "running",
            "split": split,
            "run_id": run_id,
            "input_hashes": input_hashes,
            "methods": methods,
            "repeats": repeats,
            "max_workers": int(
                config.get("execution", {}).get(
                    "max_workers",
                    1,
                )
            ),
            "prompt_sha256": {
                "B0": _json_sha256(B0_SYSTEM_PROMPT),
                "evidence": _json_sha256(
                    EVIDENCE_SYSTEM_PROMPT
                ),
            },
        }
        _atomic_write_json(
            run_dir / "run-config.json",
            run_config,
        )

    answer_path = run_dir / "per-answer.jsonl"
    error_path = run_dir / "errors.jsonl"
    completed_keys = set()
    if answer_path.is_file():
        for record in _read_jsonl_strict(
            answer_path,
            label="Formal answer records",
        ):
            completed_keys.add(
                (
                    record["question_id"],
                    record["method"],
                    record["repeat_index"],
                )
            )
    max_workers = int(
        config.get("execution", {}).get("max_workers", 1)
    )
    if max_workers < 1:
        raise ValueError("execution.max_workers 必须至少为 1")
    thread_local = threading.local()

    def get_model():
        if not hasattr(thread_local, "model"):
            thread_local.model = (
                model_factory(config=config)
                if model_factory is not None
                else FormalAnswerModel(config=config)
            )
        return thread_local.model

    def generate_record(task: dict) -> dict:
        started = time.perf_counter()
        if task["retrieval_gate_abstain"]:
            output = FormalAnswerOutput(
                answer=ABSTAIN_ANSWER,
                abstain=True,
                citations=[],
            )
            metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "system_fingerprint": None,
            }
        else:
            messages = build_answer_messages(
                question=task["question"],
                evidence=task["evidence"],
                method=task["method"],
            )
            output, metadata = get_model().invoke(messages)
        output = canonicalize_answer_output(output)
        _validate_answer_output(
            output=output,
            method=task["method"],
            evidence=task["evidence"],
        )
        record = FormalAnswerRunRecord(
            question_id=task["question_id"],
            split=split,
            method=task["method"],
            repeat_index=task["repeat_index"],
            answer=output.answer,
            abstain=output.abstain,
            citations=output.citations,
            retrieval_gate_abstain=task[
                "retrieval_gate_abstain"
            ],
            evidence_ids=[
                item["label"] for item in task["evidence"]
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=metadata["input_tokens"],
            output_tokens=metadata["output_tokens"],
            model_name=prereg["model"]["name"],
        )
        return record.model_dump(mode="json")

    error_records = (
        _read_jsonl_strict(
            error_path,
            label="Formal answer errors",
        )
        if error_path.is_file()
        else []
    )
    error_attempt_count = len(error_records)
    failed_keys = {
        (
            record["question_id"],
            record["method"],
            record["repeat_index"],
        )
        for record in error_records
    }
    question_ids = sorted(loaded["questions"])
    pending_tasks = []
    for question_id in question_ids:
        question = loaded["questions"][question_id]
        for method in methods:
            retrieval = (
                loaded["retrieval"][method]
                if method != "B0"
                else None
            )
            evidence = (
                retrieval[question_id]["evidence"]
                if retrieval is not None
                else []
            )
            top_score = (
                retrieval[question_id]["top_score"]
                if retrieval is not None
                else None
            )
            gate_method = "P" if method == "P-no-parent" else method
            retrieval_gate_abstain = (
                split == "formal_test"
                and gate_method in thresholds
                and top_score is not None
                and top_score < thresholds[gate_method]
            )
            for repeat_index in range(repeats):
                key = (question_id, method, repeat_index)
                if key in completed_keys:
                    continue
                pending_tasks.append(
                    {
                        "question_id": question_id,
                        "question": question["question"],
                        "method": method,
                        "repeat_index": repeat_index,
                        "evidence": evidence,
                        "retrieval_gate_abstain": (
                            retrieval_gate_abstain
                        ),
                    }
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_record, task): task
            for task in pending_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            key = (
                task["question_id"],
                task["method"],
                task["repeat_index"],
            )
            try:
                record = future.result()
                _append_jsonl(answer_path, record)
                completed_keys.add(key)
            except Exception as error:
                error_attempt_count += 1
                failed_keys.add(key)
                _append_jsonl(
                    error_path,
                    {
                        "question_id": task["question_id"],
                        "method": task["method"],
                        "repeat_index": task["repeat_index"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )

    expected_runs = len(question_ids) * len(methods) * repeats
    completed_count = len(completed_keys)
    error_count = len(failed_keys - completed_keys)
    status = (
        "completed"
        if completed_count == expected_runs and error_count == 0
        else "completed_with_errors"
    )
    summary = {
        "version": config["version"],
        "status": status,
        "split": split,
        "run_dir": run_dir.as_posix(),
        "question_count": len(question_ids),
        "method_count": len(methods),
        "methods": methods,
        "repeats": repeats,
        "max_workers": max_workers,
        "expected_runs": expected_runs,
        "completed_count": completed_count,
        "error_count": error_count,
        "error_attempt_count": error_attempt_count,
        "input_hashes": input_hashes,
    }
    _atomic_write_json(
        run_dir / "matrix-summary.json",
        summary,
    )
    run_config["status"] = status
    run_config["completed_count"] = completed_count
    run_config["error_count"] = error_count
    run_config["error_attempt_count"] = error_attempt_count
    _atomic_write_json(
        run_dir / "run-config.json",
        run_config,
    )
    return summary


def freeze_formal_answer_dev(
    *,
    run_dir: Path,
    dataset_path: Path,
    matrix_dir: Path,
    answer_prereg_path: Path,
    config_path: Path,
    formal_manifest_path: Path,
    formal_runs_manifest_path: Path,
    output_path: Path | None = None,
) -> dict:
    summary = _read_json(
        run_dir / "matrix-summary.json",
        label="Formal answer dev summary",
    )
    if (
        summary.get("status") != "completed"
        or summary.get("split") != "formal_dev"
        or summary.get("error_count") != 0
    ):
        raise ValueError("只允许冻结无错误的 completed formal_dev")
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    prereg = _read_json(
        answer_prereg_path,
        label="Formal answer prereg",
    )
    loaded = load_frozen_answer_inputs(
        dataset_path=dataset_path,
        matrix_dir=matrix_dir,
        answer_prereg_path=answer_prereg_path,
        split="formal_dev",
        formal_manifest_path=formal_manifest_path,
        formal_runs_manifest_path=formal_runs_manifest_path,
    )
    thresholds = {}
    for method in ("B4", "P"):
        thresholds[method] = calibrate_threshold(
            [
                {
                    "answerable": question["answerable"],
                    "score": loaded["retrieval"][method][
                        question_id
                    ]["top_score"],
                }
                for question_id, question in loaded[
                    "questions"
                ].items()
            ]
        )
    if output_path is None:
        output_path = run_dir / "dev-freeze.json"
    manifest = {
        "version": config["version"],
        "status": "ready",
        "stage": "formal_answer_dev_frozen",
        "answer_dev_frozen": True,
        "model": prereg["model"],
        "generation": {
            "temperature": config["model"]["temperature"],
            "max_tokens": config["model"]["max_tokens"],
            "structured_output_method": config["model"][
                "structured_output_method"
            ],
            "methods": config["generation"]["answer_methods"],
            "repeats": config["generation"]["repeats"],
        },
        "prompt_sha256": {
            "B0": _json_sha256(B0_SYSTEM_PROMPT),
            "evidence": _json_sha256(EVIDENCE_SYSTEM_PROMPT),
        },
        "thresholds": thresholds,
        "completion": {
            "question_count": summary["question_count"],
            "expected_runs": summary["expected_runs"],
            "completed_count": summary["completed_count"],
            "error_count": summary["error_count"],
        },
        "inputs": {
            "answer_prereg_sha256": _sha256_file(
                answer_prereg_path
            ),
            "config_sha256": _sha256_file(config_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "matrix_config_sha256": _sha256_file(
                matrix_dir / "matrix-config.json"
            ),
            "run_summary_sha256": _sha256_file(
                run_dir / "matrix-summary.json"
            ),
            "per_answer_sha256": _sha256_file(
                run_dir / "per-answer.jsonl"
            ),
        },
    }
    _atomic_write_json(output_path, manifest)
    return manifest


def freeze_formal_answer_prereg(
    *,
    config_path: Path,
    formal_manifest_path: Path,
    formal_runs_manifest_path: Path,
    dev_run_dir: Path,
    test_run_dir: Path,
    output_path: Path,
    env: Mapping[str, str] | None = None,
) -> dict:
    if env is None:
        load_dotenv()
        environment = os.environ
    else:
        environment = env
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = environment.get(
        config["model"]["env_model_key"],
        "",
    ).strip()
    base_url = environment.get(
        config["model"]["env_base_url_key"],
        "",
    ).strip()
    if not model_name or not base_url:
        raise ValueError(
            "回答层要求冻结 OPENAI_MODEL 和 OPENAI_BASE_URL"
        )

    for path, label in (
        (formal_manifest_path, "Formal Manifest"),
        (formal_runs_manifest_path, "Formal runs Manifest"),
    ):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "ready":
            raise ValueError(f"{label} 必须为 ready")

    manifest = {
        "version": config["version"],
        "status": "ready",
        "stage": "formal_answer_preregistered",
        "model": {
            "name": model_name,
            "base_url_origin": urlsplit(base_url).netloc,
        },
        "inputs": {
            "config_sha256": _sha256_file(config_path),
            "formal_manifest_sha256": _sha256_file(
                formal_manifest_path
            ),
            "formal_runs_manifest_sha256": _sha256_file(
                formal_runs_manifest_path
            ),
            "dev_matrix_config_sha256": _sha256_file(
                dev_run_dir / "matrix-config.json"
            ),
            "test_matrix_config_sha256": _sha256_file(
                test_run_dir / "matrix-config.json"
            ),
        },
        "methods": config["generation"]["answer_methods"],
        "repeats": config["generation"]["repeats"],
        "test_policy": "single_frozen_matrix",
    }
    _atomic_write_json(output_path, manifest)
    return manifest
