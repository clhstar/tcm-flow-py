import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from experiments.rag_v1_5.runner import (
    _atomic_write_json,
    _sha256_file,
)


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
