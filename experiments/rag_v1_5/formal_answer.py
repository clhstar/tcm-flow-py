import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from experiments.rag_v1_5.runner import (
    _atomic_write_json,
    _read_json,
    _read_jsonl_strict,
    _sha256_file,
)
from experiments.rag_v1_5.schema import FormalAnswerOutput

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
