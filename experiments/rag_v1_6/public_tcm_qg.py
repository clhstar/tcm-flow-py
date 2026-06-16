import json
import random
from pathlib import Path

import yaml

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    sha256_file,
    utc_now,
    write_jsonl,
)
from experiments.rag_v1_6.schema import PublicTcmQgDocument, PublicTcmQgQaPair


def read_public_tcm_qg_source(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("TCM-QG source must be a JSON list")
    return rows


def freeze_public_tcm_qg_source(
    *,
    source_path: Path,
    output_path: Path,
    public_dataset_url: str,
    expected_sha256: str | None = None,
) -> dict:
    rows = read_public_tcm_qg_source(source_path)
    documents = []
    qa_pair_count = 0
    text_lengths = []
    question_lengths = []
    answer_lengths = []
    for row in rows:
        document = PublicTcmQgDocument(
            source_doc_id=str(row.get("id", "")),
            text=row.get("text", ""),
            annotations=row.get("annotations", []),
        )
        documents.append(document)
        text_lengths.append(len(document.text))
        for annotation in document.annotations:
            qa_pair_count += 1
            question_lengths.append(len(annotation["Q"]))
            answer_lengths.append(len(annotation["A"]))

    source_sha256 = sha256_file(source_path)
    if expected_sha256 and source_sha256 != expected_sha256.upper():
        raise ValueError(
            "TCM-QG source SHA256 mismatch: "
            f"expected={expected_sha256.upper()}, actual={source_sha256}"
        )

    def average(values: list[int]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_source_frozen",
        "generated_at": utc_now(),
        "public_dataset_name": "Tianchi TCM-QG",
        "public_dataset_url": public_dataset_url,
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "document_count": len(documents),
        "qa_pair_count": qa_pair_count,
        "document_mean_chars": average(text_lengths),
        "question_mean_chars": average(question_lengths),
        "gold_response_mean_chars": average(answer_lengths),
        "privacy": {
            "raw_text_included": False,
            "full_source_committed": False,
        },
    }
    atomic_write_json(output_path, manifest)
    return manifest


def normalize_text_key(value: str) -> str:
    return "".join(value.split())


def build_doc_split(
    *,
    doc_ids: list[str],
    seed: int,
    dev_rate: float,
    test_rate: float,
) -> dict[str, str]:
    if not doc_ids:
        raise ValueError("doc_ids cannot be empty")
    if dev_rate < 0 or test_rate < 0 or dev_rate + test_rate >= 1:
        raise ValueError("invalid split rates")
    ordered = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    dev_count = round(len(ordered) * dev_rate)
    test_count = round(len(ordered) * test_rate)
    split_by_doc_id = {}
    for doc_id in ordered[:dev_count]:
        split_by_doc_id[doc_id] = "dev"
    for doc_id in ordered[dev_count : dev_count + test_count]:
        split_by_doc_id[doc_id] = "test"
    for doc_id in ordered[dev_count + test_count :]:
        split_by_doc_id[doc_id] = "train_pool"
    return split_by_doc_id


def normalize_public_tcm_qg_rows(
    *,
    rows: list[dict],
    split_by_doc_id: dict[str, str],
    min_text_chars: int,
    max_text_chars: int,
) -> list[dict]:
    normalized = []
    seen = set()
    for row in rows:
        source_doc_id = str(row.get("id", ""))
        source_text = str(row.get("text", ""))
        if len(source_text) < min_text_chars or len(source_text) > max_text_chars:
            continue
        split = split_by_doc_id.get(source_doc_id)
        if split is None:
            raise ValueError(f"missing split for source_doc_id={source_doc_id}")
        annotations = row.get("annotations", [])
        if not isinstance(annotations, list):
            continue
        for annotation_index, annotation in enumerate(annotations):
            question = str(annotation.get("Q", "")).strip()
            answer = str(annotation.get("A", "")).strip()
            if not question or not answer:
                continue
            answer_start = source_text.find(answer)
            if answer_start < 0:
                continue
            dedupe_key = (
                source_doc_id,
                normalize_text_key(question),
                normalize_text_key(answer),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            pair = PublicTcmQgQaPair(
                qa_id=f"tcmqg-{source_doc_id}-{annotation_index:03d}",
                source_doc_id=source_doc_id,
                split=split,
                question=question,
                answer=answer,
                source_text=source_text,
                answer_start=answer_start,
                answer_end=answer_start + len(answer),
                review_status="approved",
            )
            normalized.append(pair.model_dump(mode="json"))
    return normalized


def prepare_public_tcm_qg_dataset(
    *,
    source_path: Path,
    config_path: Path,
    output_path: Path,
    split_path: Path,
    manifest_path: Path,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = read_public_tcm_qg_source(source_path)
    doc_ids = [str(row.get("id", "")) for row in rows]
    split_by_doc_id = build_doc_split(
        doc_ids=doc_ids,
        seed=int(config["split"]["seed"]),
        dev_rate=float(config["split"]["dev_rate"]),
        test_rate=float(config["split"]["test_rate"]),
    )
    normalized = normalize_public_tcm_qg_rows(
        rows=rows,
        split_by_doc_id=split_by_doc_id,
        min_text_chars=int(config["filtering"]["min_text_chars"]),
        max_text_chars=int(config["filtering"]["max_text_chars"]),
    )
    write_jsonl(output_path, normalized)
    atomic_write_json(
        split_path,
        {
            "version": VERSION,
            "status": "ready",
            "stage": "public_tcm_qg_doc_split_frozen",
            "split_seed": int(config["split"]["seed"]),
            "split_unit": "source_doc_id",
            "split_by_doc_id": split_by_doc_id,
            "privacy": {"raw_text_included": False},
        },
    )
    return freeze_public_tcm_qg_dataset(
        dataset_path=output_path,
        split_path=split_path,
        source_path=source_path,
        config_path=config_path,
        output_path=manifest_path,
    )


def freeze_public_tcm_qg_dataset(
    *,
    dataset_path: Path,
    split_path: Path,
    source_path: Path,
    config_path: Path,
    output_path: Path,
) -> dict:
    rows = [
        PublicTcmQgQaPair.model_validate_json(line).model_dump(mode="json")
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_split = {"train_pool": 0, "dev": 0, "test": 0}
    doc_ids_by_split = {"train_pool": set(), "dev": set(), "test": set()}
    for row in rows:
        split = row["split"]
        by_split[split] += 1
        doc_ids_by_split[split].add(row["source_doc_id"])
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_dataset_frozen",
        "generated_at": utc_now(),
        "dataset": {
            "path": dataset_path.as_posix(),
            "sha256": sha256_file(dataset_path),
            "qa_pair_count": len(rows),
            "by_split": by_split,
            "source_doc_count_by_split": {
                split: len(doc_ids)
                for split, doc_ids in doc_ids_by_split.items()
            },
        },
        "inputs": {
            "source_sha256": sha256_file(source_path),
            "split_sha256": sha256_file(split_path),
            "config_sha256": sha256_file(config_path),
        },
        "filtering": {
            "require_answer_substring": True,
            "dedupe_key": "source_doc_id + normalized_question + normalized_answer",
        },
        "privacy": {
            "raw_text_included": False,
            "qa_content_included": False,
        },
    }
    atomic_write_json(output_path, manifest)
    return manifest
