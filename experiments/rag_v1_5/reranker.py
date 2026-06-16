import json
from pathlib import Path
from typing import Protocol

from experiments.rag_v1_5.model_store import (
    snapshot_files,
    snapshot_tree_sha256,
)
from experiments.rag_v1_5.schema import RetrievalHit


class RerankerScorer(Protocol):
    def score(self, pairs: list[list[str]]):
        ...


def resolve_model_snapshot(
    *,
    config: dict,
    role: str,
    model_manifest_path: Path,
    repository_root: Path | None = None,
) -> tuple[Path, dict]:
    if not model_manifest_path.is_file():
        raise FileNotFoundError(f"缺少模型 Manifest: {model_manifest_path}")
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    manifest = json.loads(
        model_manifest_path.read_text(encoding="utf-8")
    )
    expected = config[role]
    record = manifest.get(role)
    if record is None:
        raise ValueError(f"模型 Manifest 缺少 {role}")
    if (
        record.get("model") != expected.get("model")
        or record.get("revision") != expected.get("revision")
    ):
        raise ValueError(f"{role} 模型名或 revision 与配置不一致")
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
    actual_hash = snapshot_tree_sha256(snapshot_files(local_path))
    if actual_hash != record.get("snapshot_tree_sha256"):
        raise ValueError(f"{role} 本地模型快照哈希不一致")
    return local_path, record


class FlagRerankerScorer:
    def __init__(self, local_model_path: Path, config: dict) -> None:
        from FlagEmbedding import FlagReranker

        self.batch_size = int(config["batch_size"])
        self.max_length = int(config["max_length"])
        self.normalize = bool(config["normalize_score"])
        self.model = FlagReranker(
            str(local_model_path),
            use_fp16=bool(config["use_fp16"]),
            device=config["device"],
        )

    def score(self, pairs: list[list[str]]):
        import torch

        batch_size = self.batch_size
        while True:
            try:
                return self.model.compute_score(
                    pairs,
                    batch_size=batch_size,
                    max_length=self.max_length,
                    normalize=self.normalize,
                )
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    *,
    scorer: RerankerScorer,
    top_k: int,
) -> list[RetrievalHit]:
    if not query.strip():
        raise ValueError("查询不能为空")
    if not hits:
        return []
    pairs = [[query, hit.text] for hit in hits]
    raw_scores = scorer.score(pairs)
    if isinstance(raw_scores, (int, float)):
        scores = [float(raw_scores)]
    else:
        scores = [float(score) for score in raw_scores]
    if len(scores) != len(hits):
        raise ValueError(
            f"Reranker 分数数量不一致: "
            f"expected={len(hits)}, actual={len(scores)}"
        )

    scored = [
        hit.model_copy(update={"reranker_score": score})
        for hit, score in zip(hits, scores)
    ]
    scored.sort(
        key=lambda hit: (-hit.reranker_score, hit.chunk_id)
    )
    return [
        hit.model_copy(update={"rank": rank})
        for rank, hit in enumerate(scored[:top_k], start=1)
    ]
