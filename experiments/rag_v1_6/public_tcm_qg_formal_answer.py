import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from experiments.rag_v1_6.schema import PublicTcmQgAnswerRecord, PublicTcmQgQaPair
from experiments.rag_v1_6.common import (
    VERSION,
    append_jsonl,
    atomic_write_json,
    compact_timestamp,
    json_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


FORMAL_ABSTAIN_ANSWER = "给定公开文档证据中未找到可靠答案。"
DEFAULT_PRICE_PER_1M = {
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.14, "output": 0.28},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
}
FORMAL_PROMPT_CONTRACT = {
    "evidence_methods": {
        "scope": "只能依据给定公开文档证据回答",
        "answer_rule": "证据直接支持时必须回答",
        "abstain_rule": "证据不足时拒答",
        "format": "输出 JSON",
        "citation_rule": "citations 只能使用 E1-E5",
    },
    "b0": {
        "scope": "不使用外部检索证据",
        "format": "输出 JSON",
        "fields": ["answer", "abstain"],
    },
}
FORBIDDEN_MANIFEST_FIELDS = {
    "source_text",
    "question_text",
    "reference_answer",
    "answer_text",
    "evidence_text",
    "reviewer_comment",
}
METHOD_TO_FORMAL_CONFIG = {
    "B4": "b4-public-hybrid-rerank",
    "P": "p-public-hybrid-rerank",
    "P-no-parent": "p-public-no-parent",
}


def build_public_tcm_qg_formal_prompt(
    *,
    question: str,
    method: str,
    evidence: list[dict],
) -> str:
    if method == "B0":
        return (
            "你是中医文献问答系统。本轮为 B0 直接回答基线，不使用外部检索证据。"
            "请输出 JSON，字段为 answer 和 abstain。"
            f"问题：{question}"
        )
    evidence_text = "\n".join(
        f"{item['label']} (doc={item['source_doc_id']}): {item['text']}"
        for item in evidence[:5]
    )
    return (
        "你是中医文献问答系统。只能依据给定公开文档证据回答；"
        "证据直接支持时必须回答，并优先抽取证据原文中的最短答案片段；"
        "证据不足时拒答，"
        f"固定回答为：{FORMAL_ABSTAIN_ANSWER}。只输出 JSON，不要输出解释文字；"
        "字段为 answer、abstain、citations；"
        "citations 只能使用 E1-E5。\n\n"
        f"问题：{question}\n\n证据：\n{evidence_text}"
    )


def parse_formal_answer_json(
    content: str,
    *,
    method: str,
    evidence_labels: set[str],
) -> dict:
    stripped = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("formal answer JSON must be an object")
    answer = str(payload.get("answer", "")).strip()
    abstain = bool(payload.get("abstain", False))
    citations = payload.get("citations", [])
    if method == "B0":
        citations = []
    if not isinstance(citations, list) or not all(
        isinstance(item, str) for item in citations
    ):
        raise ValueError("citations must be a list of strings")
    unknown = sorted(set(citations) - evidence_labels)
    if unknown:
        raise ValueError(f"unknown citation labels: {', '.join(unknown)}")
    if abstain:
        answer = FORMAL_ABSTAIN_ANSWER
        citations = []
    if not answer:
        raise ValueError("answer cannot be empty")
    return {
        "answer": answer,
        "abstain": abstain,
        "citations": citations,
    }


def _origin(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("base_url_origin requires a URL with scheme and host")
    return f"{parsed.scheme}://{parsed.netloc}"


def estimate_public_tcm_qg_formal_answer_cost(
    *,
    question_count: int,
    methods: list[str],
    repeats: int,
    prompt_token_estimate_per_call: int,
    completion_token_estimate_per_call: int,
    model_name: str,
    base_url_origin: str,
    prompt_cost_per_1k: float = 0.0,
    completion_cost_per_1k: float = 0.0,
    seconds_per_call: float = 1.0,
) -> dict:
    if question_count <= 0:
        raise ValueError("question_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not methods:
        raise ValueError("methods cannot be empty")
    default_price = DEFAULT_PRICE_PER_1M.get(model_name, {})
    if prompt_cost_per_1k == 0.0 and "input" in default_price:
        prompt_cost_per_1k = default_price["input"] / 1000
    if completion_cost_per_1k == 0.0 and "output" in default_price:
        completion_cost_per_1k = default_price["output"] / 1000
    expected_calls = question_count * len(methods) * repeats
    prompt_tokens = expected_calls * prompt_token_estimate_per_call
    completion_tokens = expected_calls * completion_token_estimate_per_call
    estimated_cost = (
        prompt_tokens / 1000 * prompt_cost_per_1k
        + completion_tokens / 1000 * completion_cost_per_1k
    )
    return {
        "split": "test",
        "question_count": question_count,
        "methods": len(methods),
        "method_names": list(methods),
        "repeats": repeats,
        "expected_calls": expected_calls,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_cost_by_model": {model_name: estimated_cost},
        "pricing": {
            "source": "DeepSeek official pricing, per 1M tokens",
            "input_cache_policy": "cache_miss",
            "input_usd_per_1m": prompt_cost_per_1k * 1000,
            "output_usd_per_1m": completion_cost_per_1k * 1000,
        },
        "estimated_wall_time_seconds": expected_calls * seconds_per_call,
        "model_name": model_name,
        "base_url_origin": _origin(base_url_origin),
    }


def freeze_public_tcm_qg_formal_answer_prereg(
    *,
    output_path: Path,
    retrieval_test_run_dir: Path,
    answer_methods: list[str],
    temperature: int | float,
    repeats: int,
    model_name: str,
    base_url_origin: str,
    max_tokens: int = 256,
) -> dict:
    retrieval_summary = retrieval_test_run_dir / "matrix-summary.json"
    if not retrieval_summary.is_file():
        raise FileNotFoundError(f"missing retrieval test matrix: {retrieval_summary}")
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_answer_preregistered",
        "generated_at": utc_now(),
        "answer_methods": list(answer_methods),
        "temperature": temperature,
        "repeats": repeats,
        "max_tokens": max_tokens,
        "prompt_sha256": json_sha256(FORMAL_PROMPT_CONTRACT),
        "model_name": model_name,
        "base_url_origin": _origin(base_url_origin),
        "test_policy": "single_frozen_run",
        "inputs": {
            "retrieval_test_matrix_sha256": sha256_file(retrieval_summary),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_included": False,
            "api_key_included": False,
        },
    }
    serialized = str(manifest)
    leaked = [field for field in FORBIDDEN_MANIFEST_FIELDS if field in serialized]
    if leaked:
        raise ValueError(f"manifest includes forbidden fields: {', '.join(leaked)}")
    atomic_write_json(output_path, manifest)
    return manifest


class FormalAnswerClient:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int = 120,
        max_retries: int = 2,
    ) -> None:
        load_dotenv(dotenv_path=Path(".env"))
        from langchain_openai import ChatOpenAI

        self.model_name = model_name
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def invoke_json(self, *, prompt: str, method: str) -> dict:
        response = self.model.invoke(prompt)
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage", {}) or {}
        return {
            "content": response.content,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "system_fingerprint": metadata.get("system_fingerprint"),
        }


def _load_retrieval_records(
    *,
    matrix_dir: Path,
    config_id: str,
) -> dict[str, dict]:
    rows = read_jsonl(
        matrix_dir / config_id / "per-question.jsonl",
        label=f"{config_id} formal retrieval rows",
    )
    records = {}
    for row in rows:
        qa_id = row["qa_id"]
        if qa_id in records:
            raise ValueError(f"duplicate retrieval qa_id: {qa_id}")
        records[qa_id] = row
    return records


def _evidence_from_retrieval(record: dict, *, method: str, top_k: int) -> list[dict]:
    evidence = []
    seen = set()
    for hit in record["hits"][:top_k]:
        dedupe_key = hit["parent_id"] if method == "P" else hit["chunk_id"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        evidence.append(
            {
                "label": f"E{len(evidence) + 1}",
                "source_doc_id": hit["source_doc_id"],
                "chunk_id": hit["chunk_id"],
                "parent_id": hit["parent_id"],
                "text": hit["context_text"],
                "context_start_index": hit["context_start_index"],
                "context_char_count": hit["context_char_count"],
            }
        )
        if len(evidence) == top_k:
            break
    return evidence


def load_public_tcm_qg_formal_answer_inputs(
    *,
    split: str,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    top_k: int = 5,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    matrix_config = read_json(
        retrieval_matrix_dir / "matrix-config.json",
        label="formal retrieval matrix config",
    )
    matrix_summary = read_json(
        retrieval_matrix_dir / "matrix-summary.json",
        label="formal retrieval matrix summary",
    )
    if matrix_config.get("status") != "completed" or matrix_summary.get(
        "status"
    ) != "completed":
        raise ValueError("formal retrieval matrix must be completed")
    if matrix_config.get("split") != split or matrix_summary.get("split") != split:
        raise ValueError("formal retrieval matrix split mismatch")
    questions = {
        row["qa_id"]: PublicTcmQgQaPair.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(dataset_path, label="formal public TCM-QG dataset")
        if row.get("split") == split
    }
    if not questions:
        raise ValueError(f"no questions for split={split}")
    retrieval = {}
    raw_records = {}
    for method, config_id in METHOD_TO_FORMAL_CONFIG.items():
        records = _load_retrieval_records(
            matrix_dir=retrieval_matrix_dir,
            config_id=config_id,
        )
        if set(records) != set(questions):
            raise ValueError(f"{config_id} question set mismatch")
        raw_records[method] = records
        retrieval[method] = {
            qa_id: {
                "evidence": _evidence_from_retrieval(
                    record,
                    method=method,
                    top_k=top_k,
                ),
                "hits": record["hits"],
            }
            for qa_id, record in records.items()
        }
    for qa_id in questions:
        p_ids = [hit["chunk_id"] for hit in raw_records["P"][qa_id]["hits"]]
        child_ids = [
            hit["chunk_id"] for hit in raw_records["P-no-parent"][qa_id]["hits"]
        ]
        if p_ids != child_ids:
            raise ValueError("P-no-parent must reuse the same child ranking as P")
    return {
        "split": split,
        "questions": questions,
        "retrieval": retrieval,
        "input_hashes": {
            "dataset_sha256": sha256_file(dataset_path),
            "retrieval_matrix_config_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-config.json"
            ),
            "retrieval_matrix_summary_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-summary.json"
            ),
        },
    }


def _completed_answer_keys(answer_path: Path) -> set[tuple[str, str, int]]:
    if not answer_path.is_file():
        return set()
    return {
        (row["qa_id"], row["method"], int(row["repeat_index"]))
        for row in read_jsonl(answer_path, label="formal answer records")
    }


def _error_records(error_path: Path) -> list[dict]:
    return read_jsonl(error_path, label="formal answer errors") if error_path.is_file() else []


def run_public_tcm_qg_formal_answer_matrix(
    *,
    split: str,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    answer_prereg_path: Path,
    output_dir: Path,
    resume_dir: Path | None = None,
    client_factory=None,
    max_workers: int = 1,
    top_k: int = 5,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    prereg = read_json(answer_prereg_path, label="formal answer prereg")
    if prereg.get("status") != "ready":
        raise ValueError("formal answer prereg must be ready")
    methods = list(prereg["answer_methods"])
    repeats = int(prereg["repeats"])
    loaded = load_public_tcm_qg_formal_answer_inputs(
        split=split,
        dataset_path=dataset_path,
        retrieval_matrix_dir=retrieval_matrix_dir,
        top_k=top_k,
    )
    input_hashes = {
        **loaded["input_hashes"],
        "answer_prereg_sha256": sha256_file(answer_prereg_path),
    }
    if resume_dir is None:
        run_id = (
            f"public_tcm_qg_formal_answer_{split}-"
            f"{compact_timestamp()}-{input_hashes['answer_prereg_sha256'][:8]}"
        )
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_config = {
            "version": VERSION,
            "status": "running",
            "split": split,
            "run_id": run_id,
            "methods": methods,
            "repeats": repeats,
            "model_name": prereg["model_name"],
            "base_url_origin": prereg["base_url_origin"],
            "prompt_sha256": prereg["prompt_sha256"],
            "input_hashes": input_hashes,
            "max_workers": max_workers,
            "created_at": utc_now(),
        }
        atomic_write_json(run_dir / "run-config.json", run_config)
    else:
        run_dir = resume_dir
        run_config = read_json(
            run_dir / "run-config.json",
            label="formal answer run config",
        )
        if run_config.get("split") != split or run_config.get("input_hashes") != input_hashes:
            raise ValueError("resume input hashes or split mismatch")

    answer_path = run_dir / "per-answer.jsonl"
    error_path = run_dir / "errors.jsonl"
    completed_keys = _completed_answer_keys(answer_path)
    prior_errors = _error_records(error_path)
    failed_keys = {
        (row["qa_id"], row["method"], int(row["repeat_index"]))
        for row in prior_errors
    }
    thread_local = threading.local()

    def get_client():
        if not hasattr(thread_local, "client"):
            if client_factory is not None:
                thread_local.client = client_factory()
            else:
                load_dotenv(dotenv_path=Path(".env"))
                base_url = os.getenv("OPENAI_BASE_URL", "")
                thread_local.client = FormalAnswerClient(
                    model_name=prereg["model_name"],
                    base_url=base_url,
                    temperature=float(prereg["temperature"]),
                    max_tokens=int(prereg["max_tokens"]),
                )
        return thread_local.client

    def run_task(task: dict) -> dict:
        started = time.perf_counter()
        evidence = task["evidence"]
        prompt = build_public_tcm_qg_formal_prompt(
            question=task["question"],
            method=task["method"],
            evidence=evidence,
        )
        response = get_client().invoke_json(prompt=prompt, method=task["method"])
        parsed = parse_formal_answer_json(
            response["content"],
            method=task["method"],
            evidence_labels={item["label"] for item in evidence},
        )
        record = PublicTcmQgAnswerRecord(
            qa_id=task["qa_id"],
            source_doc_id=task["source_doc_id"],
            split=split,
            method=task["method"],
            repeat_index=task["repeat_index"],
            answer=parsed["answer"],
            abstain=parsed["abstain"],
            citations=parsed["citations"],
            retrieval_supported=bool(parsed["citations"]),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=int(response.get("input_tokens", len(prompt))),
            output_tokens=int(response.get("output_tokens", len(parsed["answer"]))),
            model_name=prereg["model_name"],
        )
        return record.model_dump(mode="json")

    tasks = []
    for qa_id in sorted(loaded["questions"]):
        question = loaded["questions"][qa_id]
        for method in methods:
            evidence = [] if method == "B0" else loaded["retrieval"][method][qa_id]["evidence"]
            for repeat_index in range(repeats):
                key = (qa_id, method, repeat_index)
                if key in completed_keys:
                    continue
                tasks.append(
                    {
                        "qa_id": qa_id,
                        "source_doc_id": question["source_doc_id"],
                        "question": question["question"],
                        "method": method,
                        "repeat_index": repeat_index,
                        "evidence": evidence,
                    }
                )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            key = (task["qa_id"], task["method"], task["repeat_index"])
            try:
                record = future.result()
                append_jsonl(answer_path, record)
                completed_keys.add(key)
            except Exception as error:
                failed_keys.add(key)
                append_jsonl(
                    error_path,
                    {
                        "qa_id": task["qa_id"],
                        "method": task["method"],
                        "repeat_index": task["repeat_index"],
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "recorded_at": utc_now(),
                    },
                )
    errors = _error_records(error_path)
    parse_error_attempts = sum(
        row["error_type"] in {"JSONDecodeError", "ValueError"} for row in errors
    )
    expected_runs = len(loaded["questions"]) * len(methods) * repeats
    completed_count = len(completed_keys)
    error_count = len(failed_keys - completed_keys)
    status = (
        "completed"
        if completed_count == expected_runs and error_count == 0
        else "completed_with_errors"
    )
    summary = {
        "version": VERSION,
        "status": status,
        "stage": "public_tcm_qg_formal_answer_completed",
        "split": split,
        "run_id": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "question_count": len(loaded["questions"]),
        "methods": methods,
        "repeats": repeats,
        "expected_runs": expected_runs,
        "completed_count": completed_count,
        "error_count": error_count,
        "error_attempt_count": len(errors),
        "json_parse_error_count": parse_error_attempts,
        "json_parse_error_rate": (
            parse_error_attempts / len(errors) if errors else 0
        ),
        "model_name": prereg["model_name"],
        "input_hashes": input_hashes,
    }
    atomic_write_json(run_dir / "matrix-summary.json", summary)
    run_config.update(
        status=status,
        completed_count=completed_count,
        error_count=error_count,
        error_attempt_count=len(errors),
    )
    atomic_write_json(run_dir / "run-config.json", run_config)
    return summary


def freeze_public_tcm_qg_formal_answer_dev(
    *,
    run_dir: Path,
    output_path: Path | None = None,
) -> dict:
    summary = read_json(run_dir / "matrix-summary.json", label="formal answer dev summary")
    if (
        summary.get("status") != "completed"
        or summary.get("split") != "dev"
        or summary.get("error_count") != 0
        or summary.get("json_parse_error_rate") != 0
    ):
        raise ValueError("only completed dev answer runs with zero errors can be frozen")
    per_answer_path = run_dir / "per-answer.jsonl"
    if output_path is None:
        output_path = run_dir / "dev-freeze.json"
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_answer_dev_frozen",
        "answer_dev_frozen": True,
        "dev_freeze_status": "ready",
        "completion": {
            "question_count": summary["question_count"],
            "expected_runs": summary["expected_runs"],
            "completed_count": summary["completed_count"],
            "error_count": summary["error_count"],
            "json_parse_error_rate": summary["json_parse_error_rate"],
        },
        "inputs": {
            **summary.get("input_hashes", {}),
            "run_summary_sha256": sha256_file(run_dir / "matrix-summary.json"),
            "per_answer_sha256": sha256_file(per_answer_path),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_committed": False,
        },
    }
    atomic_write_json(output_path, manifest)
    return manifest
