import json
from pathlib import Path

from experiments.rag_v1_5.corpus import CorpusManifest, sha256_bytes
from experiments.rag_v1_5.parser import parse_corpus_file
from experiments.rag_v1_5.schema import EvidenceUnit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def validate_evidence_graph(evidence_units: list[EvidenceUnit]) -> None:
    units_by_id: dict[str, EvidenceUnit] = {}
    for unit in evidence_units:
        if unit.evidence_id in units_by_id:
            raise ValueError(f"重复 evidence_id: {unit.evidence_id}")
        units_by_id[unit.evidence_id] = unit

    expected_parent_types = {
        "formula": "clause",
        "ingredients": "formula",
        "preparation": "formula",
        "note": "clause",
    }
    for unit in evidence_units:
        if unit.content_type == "clause":
            if unit.parent_id != unit.evidence_id:
                raise ValueError(
                    f"条文 parent_id 必须指向自身: {unit.evidence_id}"
                )
            continue

        parent = units_by_id.get(unit.parent_id)
        if parent is None:
            raise ValueError(
                f"{unit.evidence_id} 找不到 parent_id: {unit.parent_id}"
            )

        expected_parent_type = expected_parent_types[unit.content_type]
        if parent.content_type != expected_parent_type:
            raise ValueError(
                f"{unit.evidence_id} parent 类型错误: "
                f"expected={expected_parent_type}, actual={parent.content_type}"
            )


def parse_prepared_corpus(
    *,
    raw_dir: Path,
    manifest_path: Path,
    processed_dir: Path,
) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = CorpusManifest.model_validate_json(manifest_bytes)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_evidence: list[dict] = []
    all_anomalies: list[dict] = []
    book_statistics: dict[str, dict] = {}
    book_outputs: dict[str, tuple[list[dict], list[dict]]] = {}
    all_units: list[EvidenceUnit] = []

    for file_manifest in manifest.files:
        input_path = raw_dir / file_manifest.output_file
        if not input_path.is_file():
            raise FileNotFoundError(f"缺少已导入语料: {input_path}")

        output_sha256 = sha256_bytes(input_path.read_bytes())
        if output_sha256 != file_manifest.output_sha256:
            raise ValueError(
                f"{file_manifest.output_file} UTF-8 输出 SHA256 不匹配: "
                f"expected={file_manifest.output_sha256}, actual={output_sha256}"
            )

        result = parse_corpus_file(
            input_path=input_path,
            book_id=file_manifest.book_id,
            book_title=file_manifest.book_title,
            source_hash=file_manifest.source_sha256,
        )
        evidence_rows = [
            unit.model_dump(mode="json") for unit in result.evidence_units
        ]
        anomaly_rows = [
            anomaly.model_dump(mode="json") for anomaly in result.anomalies
        ]

        all_units.extend(result.evidence_units)
        all_evidence.extend(evidence_rows)
        all_anomalies.extend(anomaly_rows)
        book_outputs[file_manifest.book_id] = (evidence_rows, anomaly_rows)
        book_statistics[file_manifest.book_id] = (
            result.statistics.model_dump(mode="json")
        )

    validate_evidence_graph(all_units)

    for book_id, (evidence_rows, anomaly_rows) in book_outputs.items():
        _write_jsonl(
            processed_dir / f"{book_id}.evidence.jsonl",
            evidence_rows,
        )
        _write_jsonl(
            processed_dir / f"{book_id}.anomalies.jsonl",
            anomaly_rows,
        )

    _write_jsonl(processed_dir / "evidence.jsonl", all_evidence)
    _write_jsonl(processed_dir / "anomalies.jsonl", all_anomalies)

    statistic_fields = (
        "chapter_count",
        "clause_count",
        "formula_count",
        "ingredients_count",
        "preparation_count",
        "note_count",
        "anomaly_count",
    )
    totals = {
        field: sum(stats[field] for stats in book_statistics.values())
        for field in statistic_fields
    }
    statistics = {
        "corpus_version": manifest.corpus_version,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "books": book_statistics,
        "totals": totals,
    }
    (processed_dir / "statistics.json").write_text(
        json.dumps(
            statistics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return statistics
