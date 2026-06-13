import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import yaml

from experiments.rag_v1_5.model_store import (
    snapshot_files,
    snapshot_tree_sha256,
)
from experiments.rag_v1_5.schema import ChunkUnit
from experiments.rag_v1_5.tokenization import tokenize_text


class DenseEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


EncoderFactory = Callable[[Path, dict], DenseEncoder]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def _normalize_dense_vectors(
    vectors: np.ndarray,
    *,
    expected_count: int,
) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Dense Encoder 必须返回二维向量")
    if array.shape[0] != expected_count:
        raise ValueError(
            f"Dense 向量数量不一致: "
            f"expected={expected_count}, actual={array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Dense 向量包含 NaN 或 Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense 向量不能为零向量")
    normalized = array / norms
    if not np.isfinite(normalized).all():
        raise ValueError("Dense 归一化结果包含 NaN 或 Inf")
    return normalized.astype(np.float32, copy=False)


def build_strategy_index(
    *,
    chunks: list[ChunkUnit],
    output_dir: Path,
    encoder: DenseEncoder,
    quality_gate_sha256: str,
    chunk_sha256: str,
    model_record: dict,
) -> dict:
    started = time.perf_counter()
    if not chunks:
        raise ValueError("索引输入 Chunk 不能为空")
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    chunk_ids = [chunk.chunk_id for chunk in ordered]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("索引输入存在重复 Chunk ID")
    strategies = {chunk.strategy for chunk in ordered}
    if len(strategies) != 1:
        raise ValueError("单策略索引不能混合多个 Chunk strategy")

    token_rows = []
    for chunk in ordered:
        tokens = tokenize_text(chunk.text)
        if not tokens:
            raise ValueError(f"{chunk.chunk_id} 分词结果为空")
        token_rows.append({"chunk_id": chunk.chunk_id, "tokens": tokens})

    vectors = _normalize_dense_vectors(
        encoder.encode([chunk.text for chunk in ordered]),
        expected_count=len(ordered),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    tokens_path = output_dir / "bm25_tokens.jsonl"
    dense_path = output_dir / "dense.npy"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(
        rows_path,
        [chunk.model_dump(mode="json") for chunk in ordered],
    )
    _write_jsonl(tokens_path, token_rows)
    with dense_path.open("wb") as file_handle:
        np.save(file_handle, vectors, allow_pickle=False)

    files = {
        "rows": {
            "path": rows_path.name,
            "sha256": _sha256_file(rows_path),
            "bytes": rows_path.stat().st_size,
        },
        "bm25_tokens": {
            "path": tokens_path.name,
            "sha256": _sha256_file(tokens_path),
            "bytes": tokens_path.stat().st_size,
        },
        "dense": {
            "path": dense_path.name,
            "sha256": _sha256_file(dense_path),
            "bytes": dense_path.stat().st_size,
        },
    }
    manifest = {
        "strategy": ordered[0].strategy,
        "quality_gate_sha256": quality_gate_sha256,
        "chunk_sha256": chunk_sha256,
        "model": model_record["model"],
        "revision": model_record["revision"],
        "local_model_path": model_record["local_path"],
        "row_count": len(ordered),
        "vector_dimension": int(vectors.shape[1]),
        "dtype": "float32",
        "normalize": True,
        "tokenizer": "jieba",
        "hmm": False,
        "files": files,
        "built_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_quality_gate(quality_gate_path: Path) -> dict:
    if not quality_gate_path.is_file():
        raise FileNotFoundError(f"缺少 Quality Gate: {quality_gate_path}")
    gate = json.loads(quality_gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "ready":
        raise ValueError(
            f"Quality Gate 未就绪: status={gate.get('status', 'missing')}"
        )
    if gate.get("reviewed_count") != 140 or gate.get("pending_count") != 0:
        raise ValueError("Quality Gate 审核数量不满足 140/0")
    return gate


def load_verified_embedding_model(
    *,
    config_path: Path,
    model_manifest_path: Path,
    repository_root: Path | None = None,
) -> tuple[Path, dict, dict]:
    if not model_manifest_path.is_file():
        raise FileNotFoundError(f"缺少模型 Manifest: {model_manifest_path}")
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        model_manifest_path.read_text(encoding="utf-8")
    )
    expected = config["embedding"]
    record = manifest.get("embedding")
    if record is None:
        raise ValueError("模型 Manifest 缺少 embedding")
    if (
        record.get("model") != expected.get("model")
        or record.get("revision") != expected.get("revision")
    ):
        raise ValueError("Embedding 模型名或 revision 与配置不一致")

    relative_path = Path(record["local_path"])
    if relative_path.is_absolute():
        raise ValueError("模型 Manifest 只能使用仓库相对路径")
    local_path = (repository_root / relative_path).resolve()
    try:
        local_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("模型路径越出仓库根目录") from error
    if not local_path.is_dir():
        raise FileNotFoundError(f"缺少本地模型快照: {local_path}")
    actual_tree_hash = snapshot_tree_sha256(snapshot_files(local_path))
    if actual_tree_hash != record.get("snapshot_tree_sha256"):
        raise ValueError("本地模型快照哈希与 Manifest 不一致")
    return local_path, record, expected


class BgeM3DenseEncoder:
    def __init__(self, local_model_path: Path, config: dict) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.batch_size = int(config["batch_size"])
        self.max_length = int(config["max_length"])
        self.model = BGEM3FlagModel(
            str(local_model_path),
            use_fp16=bool(config["use_fp16"]),
            device=config["device"],
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        batch_size = self.batch_size
        while True:
            try:
                result = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=self.max_length,
                )
                return np.asarray(result["dense_vecs"])
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()


def _default_encoder_factory(
    local_model_path: Path,
    config: dict,
) -> DenseEncoder:
    return BgeM3DenseEncoder(local_model_path, config)


def _load_chunks(path: Path) -> list[ChunkUnit]:
    return [
        ChunkUnit.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_indexes(
    *,
    config_path: Path,
    chunks_dir: Path,
    chunk_manifest_path: Path,
    quality_gate_path: Path,
    model_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    repository_root: Path | None = None,
    encoder_factory: EncoderFactory = _default_encoder_factory,
) -> dict:
    started = time.perf_counter()
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    gate = validate_quality_gate(quality_gate_path)
    gate_sha256 = _sha256_file(quality_gate_path)
    local_model_path, model_record, embedding_config = (
        load_verified_embedding_model(
            config_path=config_path,
            model_manifest_path=model_manifest_path,
            repository_root=repository_root,
        )
    )
    chunk_manifest = json.loads(
        chunk_manifest_path.read_text(encoding="utf-8")
    )
    encoder = encoder_factory(local_model_path, embedding_config)
    strategy_manifests = {}
    try:
        for strategy in ("c0", "c1", "c2", "c3", "c4"):
            chunk_record = chunk_manifest.get("strategies", {}).get(strategy)
            if chunk_record is None:
                raise ValueError(f"Chunk Manifest 缺少策略: {strategy}")
            chunk_path = chunks_dir / chunk_record["output_file"]
            actual_chunk_sha256 = _sha256_file(chunk_path)
            if actual_chunk_sha256 != chunk_record["output_sha256"]:
                raise ValueError(f"{strategy} Chunk 文件 SHA256 不匹配")
            gate_chunk_sha256 = gate.get("chunks", {}).get(strategy)
            if gate_chunk_sha256 != actual_chunk_sha256:
                raise ValueError(
                    f"{strategy} 与 Quality Gate 冻结哈希不一致"
                )
            chunks = _load_chunks(chunk_path)
            if len(chunks) != chunk_record["count"]:
                raise ValueError(f"{strategy} Chunk 行数与 Manifest 不一致")
            strategy_manifest = build_strategy_index(
                chunks=chunks,
                output_dir=output_dir / strategy,
                encoder=encoder,
                quality_gate_sha256=gate_sha256,
                chunk_sha256=actual_chunk_sha256,
                model_record=model_record,
            )
            strategy_manifest_path = output_dir / strategy / "manifest.json"
            strategy_manifests[strategy] = {
                "row_count": strategy_manifest["row_count"],
                "manifest_sha256": _sha256_file(strategy_manifest_path),
                "files": strategy_manifest["files"],
            }
    finally:
        del encoder
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    manifest = {
        "version": "v1.5.0",
        "quality_gate_sha256": gate_sha256,
        "chunk_manifest_sha256": _sha256_file(chunk_manifest_path),
        "model_manifest_sha256": _sha256_file(model_manifest_path),
        "embedding_model": model_record["model"],
        "embedding_revision": model_record["revision"],
        "strategies": strategy_manifests,
        "built_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
