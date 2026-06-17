import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .chunking import build_parent_child
from .corpus import load_curated_sections, parse_tagged_book, select_sections
from .filters import contains_excluded_content
from .schema import EvidenceParent, RetrievalChunk, SelectedSection


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records,
        key=lambda item: (
            getattr(item, "section_id", "")
            or getattr(item, "parent_id", "")
            or getattr(item, "chunk_id", "")
        ),
    )
    payload = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for item in ordered
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def build_corpus_manifest(
    *,
    version: str,
    sources: list[dict],
    sections: list[SelectedSection],
    parents: list[EvidenceParent],
    chunks: list[RetrievalChunk],
    output_dir: Path,
) -> dict:
    parent_ids = [item.parent_id for item in parents]
    chunk_ids = [item.chunk_id for item in chunks]
    if _duplicate_count(parent_ids):
        raise ValueError("生产语料存在重复 parent_id")
    if _duplicate_count(chunk_ids):
        raise ValueError("生产语料存在重复 chunk_id")
    known_parents = set(parent_ids)
    orphans = [item.chunk_id for item in chunks if item.parent_id not in known_parents]
    if orphans:
        raise ValueError(f"生产语料存在孤儿 Child: {orphans[:3]}")

    artifact_paths = {
        name: output_dir / f"{name}.jsonl"
        for name in ("sections", "parents", "chunks")
    }
    return {
        "version": version,
        "status": "ready",
        "book_count": len(sources),
        "section_count": len(sections),
        "parent_count": len(parents),
        "chunk_count": len(chunks),
        "sources": sources,
        "by_source_type": dict(Counter(item.source_type for item in parents)),
        "by_evidence_role": dict(Counter(item.evidence_role for item in parents)),
        "by_symptom": dict(
            Counter(tag for item in parents for tag in item.symptom_tags)
        ),
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }


def build_corpus(
    *,
    config: dict,
    source_root: Path,
    curated_root: Path | None,
    output_dir: Path,
) -> dict:
    sections: list[SelectedSection] = []
    source_records: list[dict] = []
    for book in config["books"]:
        path = source_root / book["source_file"]
        parsed = parse_tagged_book(
            path=path,
            book_id=book["book_id"],
            book_title=book["title"],
            encoding=config["source_encoding"],
        )
        selected = select_sections(
            parsed,
            symptom_aliases=config["symptoms"],
            method_sections=book["method_sections"],
            fixed_sections=book["fixed_sections"],
            symptom_scan=book["symptom_scan"],
            exclude_title_patterns=config["exclude_title_patterns"],
        )
        if not selected:
            raise ValueError(f"{book['source_file']} 未选择到任何章节")
        sections.extend(selected)
        source_records.append(
            {
                "book_id": book["book_id"],
                "source_file": book["source_file"],
                "source_sha256": selected[0].source_hash,
                "selected_section_count": len(selected),
                "selected_sections": [
                    {
                        "volume": item.volume,
                        "chapter": item.chapter,
                        "section": item.section,
                        "symptom_tags": item.symptom_tags,
                    }
                    for item in selected
                ],
            }
        )

    if curated_root is not None:
        sections.extend(load_curated_sections(curated_root, config["symptoms"]))

    parents: list[EvidenceParent] = []
    chunks: list[RetrievalChunk] = []
    for section in sections:
        section_parents, section_chunks = build_parent_child(section)
        parents.extend(section_parents)
        chunks.extend(section_chunks)
    if not parents or not chunks:
        raise ValueError("生产语料没有生成可检索 Parent/Child")

    write_jsonl(output_dir / "sections.jsonl", sections)
    write_jsonl(output_dir / "parents.jsonl", parents)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    manifest = build_corpus_manifest(
        version=config["version"],
        sources=source_records,
        sections=sections,
        parents=parents,
        chunks=chunks,
        output_dir=output_dir,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def doctor_corpus(output_dir: Path) -> dict:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = _load_jsonl(output_dir / "sections.jsonl", SelectedSection)
    parents = _load_jsonl(output_dir / "parents.jsonl", EvidenceParent)
    chunks = _load_jsonl(output_dir / "chunks.jsonl", RetrievalChunk)

    source_hashes = {
        source["book_id"]: source["source_sha256"] for source in manifest["sources"]
    }
    source_hash_mismatch_count = sum(
        section.source_type == "ancient_book"
        and source_hashes.get(section.book_id) != section.source_hash
        for section in sections
    )
    artifact_hash_mismatch_count = sum(
        sha256_file(output_dir / record["path"]) != record["sha256"]
        or (output_dir / record["path"]).stat().st_size != record["bytes"]
        for record in manifest["files"].values()
    )
    count_mismatch_count = sum(
        manifest[key] != actual
        for key, actual in (
            ("section_count", len(sections)),
            ("parent_count", len(parents)),
            ("chunk_count", len(chunks)),
        )
    )
    parent_ids = [item.parent_id for item in parents]
    chunk_ids = [item.chunk_id for item in chunks]
    known_parent_ids = set(parent_ids)
    result = {
        "status": "ready",
        "source_hash_mismatch_count": source_hash_mismatch_count,
        "artifact_hash_mismatch_count": artifact_hash_mismatch_count,
        "count_mismatch_count": count_mismatch_count,
        "duplicate_parent_count": _duplicate_count(parent_ids),
        "duplicate_chunk_count": _duplicate_count(chunk_ids),
        "orphan_chunk_count": sum(
            chunk.parent_id not in known_parent_ids for chunk in chunks
        ),
        "excluded_content_match_count": sum(
            contains_excluded_content(parent.original_text) for parent in parents
        ),
    }
    if any(value for key, value in result.items() if key != "status"):
        result["status"] = "invalid"
    return result
