import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Callable

import yaml


SnapshotDownloader = Callable[..., str]
VersionReader = Callable[[str], str]
REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def validate_revision(revision: str) -> str:
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "模型 revision 必须是 40 位十六进制 commit hash"
        )
    return revision.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def snapshot_files(snapshot_path: Path) -> list[dict]:
    files = []
    for path in sorted(snapshot_path.rglob("*")):
        relative_path = path.relative_to(snapshot_path)
        if not path.is_file() or ".cache" in relative_path.parts:
            continue
        files.append(
            {
                "path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"模型快照没有可用文件: {snapshot_path}")
    return files


def snapshot_tree_sha256(files: list[dict]) -> str:
    payload = "".join(
        json.dumps(file_record, sort_keys=True, separators=(",", ":")) + "\n"
        for file_record in files
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _default_snapshot_downloader(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _default_version_reader(package: str) -> str:
    return importlib.metadata.version(package)


def _snapshot_matches_manifest(
    *,
    snapshot_path: Path,
    manifest_record: dict | None,
    revision: str,
) -> bool:
    if (
        not snapshot_path.is_dir()
        or not manifest_record
        or manifest_record.get("revision") != revision
    ):
        return False
    try:
        files = snapshot_files(snapshot_path)
    except ValueError:
        return False
    return (
        snapshot_tree_sha256(files)
        == manifest_record.get("snapshot_tree_sha256")
    )


def prepare_models(
    *,
    config_path: Path,
    output_dir: Path,
    manifest_path: Path,
    repository_root: Path | None = None,
    snapshot_downloader: SnapshotDownloader = _default_snapshot_downloader,
    library_version_reader: VersionReader = _default_version_reader,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repository_root = (
        repository_root.resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    output_dir = output_dir.resolve()
    manifest = {"version": config["version"]}
    existing_manifest = {}
    if manifest_path.is_file():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    for role in ("embedding", "reranker"):
        model_config = config[role]
        repo_id = model_config["model"]
        revision = validate_revision(model_config["revision"])
        model_name = repo_id.rsplit("/", 1)[-1]
        target_dir = output_dir / model_name / revision

        if not _snapshot_matches_manifest(
            snapshot_path=target_dir,
            manifest_record=existing_manifest.get(role),
            revision=revision,
        ):
            target_dir.mkdir(parents=True, exist_ok=True)
            snapshot_downloader(
                repo_id=repo_id,
                revision=revision,
                local_dir=target_dir,
            )

        files = snapshot_files(target_dir)
        try:
            local_path = target_dir.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "模型目录必须位于仓库根目录内，Manifest 才能记录相对路径"
            ) from error

        manifest[role] = {
            "model": repo_id,
            "revision": revision,
            "local_path": local_path,
            "library": "FlagEmbedding",
            "library_version": library_version_reader("FlagEmbedding"),
            "snapshot_tree_sha256": snapshot_tree_sha256(files),
            "files": files,
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
