import hashlib
import json
from pathlib import Path


class BgeM3Encoder:
    def __init__(self, model_path: Path, settings: dict):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            str(model_path),
            use_fp16=bool(settings["use_fp16"]),
            device=settings["device"],
        )
        self.batch_size = int(settings["batch_size"])
        self.max_length = int(settings["max_length"])

    def encode(self, texts: list[str]):
        result = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return result["dense_vecs"]


class BgeReranker:
    def __init__(self, model_path: Path, settings: dict):
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(
            str(model_path),
            use_fp16=bool(settings["use_fp16"]),
            device=settings["device"],
        )
        self.settings = settings

    def score(self, pairs: list[list[str]]):
        return self.model.compute_score(
            pairs,
            batch_size=int(self.settings["batch_size"]),
            max_length=int(self.settings["max_length"]),
            normalize=bool(self.settings["normalize_score"]),
        )


def snapshot_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            }
        )
    if not rows:
        raise ValueError(f"模型快照为空: {root}")
    return rows


def snapshot_tree_sha256(files: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in files
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def prepare_models(
    *,
    config: dict,
    output_dir: Path,
    manifest_path: Path,
    downloader=None,
) -> dict:
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    existing = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {"version": config["version"], "status": "ready"}
    for role in ("embedding", "reranker"):
        settings = config["models"][role]
        model_name = settings["model"].rsplit("/", 1)[-1]
        target = output_dir / model_name / settings["revision"]
        record = existing.get(role, {})
        valid_existing = False
        if target.is_dir() and record.get("revision") == settings["revision"]:
            files = snapshot_files(target)
            valid_existing = (
                snapshot_tree_sha256(files) == record.get("snapshot_tree_sha256")
            )
        if not valid_existing:
            target.mkdir(parents=True, exist_ok=True)
            downloader(
                repo_id=settings["model"],
                revision=settings["revision"],
                local_dir=target,
            )
            files = snapshot_files(target)
        manifest[role] = {
            "model": settings["model"],
            "revision": settings["revision"],
            "local_path": target.as_posix(),
            "snapshot_tree_sha256": snapshot_tree_sha256(files),
            "files": files,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
