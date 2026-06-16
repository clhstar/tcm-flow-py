import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import jieba


VERSION = "v1.6.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def read_jsonl(path: Path, *, label: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    rows = []
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError(f"{label} has an incomplete trailing line: {path}")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokenize_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    tokens = [token.strip() for token in jieba.lcut(normalized, HMM=False)]
    return [token for token in tokens if token]


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "S"))
    )


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def char_f1(prediction: str, reference: str) -> float:
    prediction_counts = Counter(normalize_answer(prediction))
    reference_counts = Counter(normalize_answer(reference))
    prediction_length = sum(prediction_counts.values())
    reference_length = sum(reference_counts.values())
    if prediction_length == 0 and reference_length == 0:
        return 1.0
    if prediction_length == 0 or reference_length == 0:
        return 0.0
    overlap = sum(
        min(count, reference_counts[char])
        for char, count in prediction_counts.items()
    )
    precision = overlap / prediction_length
    recall = overlap / reference_length
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float | int | bool | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0
