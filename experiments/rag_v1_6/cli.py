import argparse
import json
from pathlib import Path

import yaml

from experiments.rag_v1_6.public_tcm_qg import (
    freeze_public_tcm_qg_dataset,
    freeze_public_tcm_qg_source,
    prepare_public_tcm_qg_dataset,
)
from experiments.rag_v1_6.public_tcm_qg_answer import (
    run_public_tcm_qg_answer_matrix,
)
from experiments.rag_v1_6.public_tcm_qg_formal_answer import (
    estimate_public_tcm_qg_formal_answer_cost,
    freeze_public_tcm_qg_formal_answer_dev,
    freeze_public_tcm_qg_formal_answer_prereg,
    run_public_tcm_qg_formal_answer_matrix,
)
from experiments.rag_v1_6.public_tcm_qg_formal_index import (
    build_public_tcm_qg_formal_indexes,
    load_bge_m3_embedder,
)
from experiments.rag_v1_6.public_tcm_qg_formal_metrics import (
    freeze_public_tcm_qg_formal_answer_runs,
    summarize_public_tcm_qg_formal_answer_test,
)
from experiments.rag_v1_6.public_tcm_qg_formal_review import (
    import_public_tcm_qg_formal_answer_review,
    prepare_public_tcm_qg_formal_answer_review,
)
from experiments.rag_v1_6.public_tcm_qg_formal_runner import (
    freeze_public_tcm_qg_formal_prereg,
    freeze_public_tcm_qg_formal_retrieval_runs,
    load_bge_reranker,
    run_public_tcm_qg_formal_retrieval_matrix,
    summarize_public_tcm_qg_formal_retrieval_test,
)
from experiments.rag_v1_6.public_tcm_qg_index import (
    build_public_tcm_qg_chunks,
    build_public_tcm_qg_indexes,
)
from experiments.rag_v1_6.public_tcm_qg_metrics import (
    summarize_public_tcm_qg_test,
)
from experiments.rag_v1_6.public_tcm_qg_runner import (
    freeze_public_tcm_qg_runs,
    run_public_tcm_qg_retrieval_matrix,
)
from experiments.rag_v1_6.common import read_json


DEFAULT_CONFIG_PATH = Path("experiments/rag_v1_6/configs/public-tcm-qg.yaml")
DEFAULT_FORMAL_CONFIG_PATH = Path("experiments/rag_v1_6/configs/public-tcm-qg-formal.yaml")
DEFAULT_SOURCE_PATH = Path("train.json")
DEFAULT_ROOT = Path("data/rag_v1_6/public_tcm_qg")
DEFAULT_PROCESSED_DIR = DEFAULT_ROOT / "processed"
DEFAULT_CHUNKS_DIR = DEFAULT_ROOT / "chunks"
DEFAULT_INDEXES_DIR = DEFAULT_ROOT / "indexes"
DEFAULT_FORMAL_ROOT = DEFAULT_ROOT / "formal"
DEFAULT_FORMAL_INDEXES_DIR = DEFAULT_FORMAL_ROOT / "indexes"
DEFAULT_RUNS_DIR = DEFAULT_ROOT / "runs"
DEFAULT_ANSWER_DIR = DEFAULT_ROOT / "answer"
DEFAULT_MANIFEST_DIR = Path("experiments/rag_v1_6/manifests")
DEFAULT_SOURCE_MANIFEST = DEFAULT_MANIFEST_DIR / "public-tcm-qg-source-v1.6.0.json"
DEFAULT_DATASET_MANIFEST = DEFAULT_MANIFEST_DIR / "public-tcm-qg-dataset-v1.6.0.json"
DEFAULT_RUNS_MANIFEST = DEFAULT_MANIFEST_DIR / "public-tcm-qg-runs-v1.6.0.json"
DEFAULT_FORMAL_PREREG_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "public-tcm-qg-formal-prereg-v1.6.0.json"
)
DEFAULT_FORMAL_INDEX_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "public-tcm-qg-formal-indexes-v1.6.0.json"
)
DEFAULT_FORMAL_RETRIEVAL_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "public-tcm-qg-formal-retrieval-runs-v1.6.0.json"
)
DEFAULT_FORMAL_ANSWER_PREREG_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "public-tcm-qg-formal-answer-prereg-v1.6.0.json"
)
DEFAULT_FORMAL_ANSWER_RUNS_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "public-tcm-qg-formal-answer-runs-v1.6.0.json"
)
DEFAULT_FORMAL_ANSWER_DIR = DEFAULT_FORMAL_ROOT / "answer"
DEFAULT_FORMAL_REVIEW_DIR = DEFAULT_FORMAL_ANSWER_DIR / "review"
DEFAULT_DATASET_PATH = DEFAULT_PROCESSED_DIR / "public-tcm-qg.jsonl"
DEFAULT_SPLIT_PATH = DEFAULT_PROCESSED_DIR / "public-tcm-qg-split.json"
DEFAULT_CHUNK_MANIFEST = DEFAULT_CHUNKS_DIR / "manifest.json"
DEFAULT_INDEX_MANIFEST = DEFAULT_INDEXES_DIR / "manifest.json"


def _latest_directory_with_file(root: Path, filename: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"missing run root: {root}")
    candidates = [
        path for path in root.iterdir() if path.is_dir() and (path / filename).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no run in {root} contains {filename}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCM-Flow V1.6 public TCM-QG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("freeze-public-tcm-qg-source")
    source.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    source.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    source.add_argument("--output", type=Path, default=DEFAULT_SOURCE_MANIFEST)

    prepare = subparsers.add_parser("prepare-public-tcm-qg-dataset")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    prepare.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)

    dataset = subparsers.add_parser("freeze-public-tcm-qg-dataset")
    dataset.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    dataset.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    dataset.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    dataset.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    dataset.add_argument("--output", type=Path, default=DEFAULT_DATASET_MANIFEST)

    chunks = subparsers.add_parser("build-public-tcm-qg-chunks")
    chunks.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    chunks.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    chunks.add_argument("--output-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    chunks.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)

    indexes = subparsers.add_parser("build-public-tcm-qg-indexes")
    indexes.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    indexes.add_argument("--chunk-manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    indexes.add_argument("--output-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    indexes.add_argument("--manifest", type=Path, default=DEFAULT_INDEX_MANIFEST)

    formal_prereg = subparsers.add_parser("freeze-public-tcm-qg-formal-prereg")
    formal_prereg.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
    formal_prereg.add_argument("--env", type=Path, default=Path(".env"))
    formal_prereg.add_argument("--output", type=Path, default=DEFAULT_FORMAL_PREREG_MANIFEST)

    formal_indexes = subparsers.add_parser("build-public-tcm-qg-formal-indexes")
    formal_indexes.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
    formal_indexes.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    formal_indexes.add_argument("--chunk-manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    formal_indexes.add_argument("--output-dir", type=Path, default=DEFAULT_FORMAL_INDEXES_DIR)
    formal_indexes.add_argument("--manifest", type=Path, default=DEFAULT_FORMAL_INDEX_MANIFEST)
    formal_indexes.add_argument(
        "--prereg-manifest",
        type=Path,
        default=DEFAULT_FORMAL_PREREG_MANIFEST,
    )

    formal_cost = subparsers.add_parser("estimate-public-tcm-qg-formal-answer-cost")
    formal_cost.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
    formal_cost.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    formal_cost.add_argument("--env", type=Path, default=Path(".env"))
    formal_cost.add_argument("--prompt-token-estimate", type=int, default=900)
    formal_cost.add_argument("--completion-token-estimate", type=int, default=256)
    formal_cost.add_argument("--seconds-per-call", type=float, default=1.0)

    for command, split in (
        ("run-public-tcm-qg-formal-retrieval-dev", "dev"),
        ("run-public-tcm-qg-formal-retrieval-test", "test"),
    ):
        formal_run = subparsers.add_parser(command)
        formal_run.set_defaults(split=split)
        formal_run.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
        formal_run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
        formal_run.add_argument("--indexes-dir", type=Path, default=DEFAULT_FORMAL_INDEXES_DIR)
        formal_run.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_FORMAL_ROOT / "runs" / split,
        )
        formal_run.add_argument("--prereg-manifest", type=Path, default=DEFAULT_FORMAL_PREREG_MANIFEST)
        formal_run.add_argument("--index-manifest", type=Path, default=DEFAULT_FORMAL_INDEX_MANIFEST)
        formal_run.add_argument("--resume", type=Path, default=None)

    formal_summary = subparsers.add_parser("summarize-public-tcm-qg-formal-retrieval-test")
    formal_summary.add_argument("--run-dir", type=Path, default=None)

    formal_freeze = subparsers.add_parser("freeze-public-tcm-qg-formal-retrieval-runs")
    formal_freeze.add_argument("--output", type=Path, default=DEFAULT_FORMAL_RETRIEVAL_MANIFEST)
    formal_freeze.add_argument("--prereg-manifest", type=Path, default=DEFAULT_FORMAL_PREREG_MANIFEST)
    formal_freeze.add_argument("--index-manifest", type=Path, default=DEFAULT_FORMAL_INDEX_MANIFEST)
    formal_freeze.add_argument("--dev-run-dir", type=Path, default=None)
    formal_freeze.add_argument("--test-run-dir", type=Path, default=None)

    formal_answer_prereg = subparsers.add_parser(
        "freeze-public-tcm-qg-formal-answer-prereg"
    )
    formal_answer_prereg.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
    formal_answer_prereg.add_argument("--env", type=Path, default=Path(".env"))
    formal_answer_prereg.add_argument("--retrieval-test-run-dir", type=Path, default=None)
    formal_answer_prereg.add_argument("--output", type=Path, default=DEFAULT_FORMAL_ANSWER_PREREG_MANIFEST)

    for command, split in (
        ("run-public-tcm-qg-formal-answer-dev", "dev"),
        ("run-public-tcm-qg-formal-answer-test", "test"),
    ):
        formal_answer = subparsers.add_parser(command)
        formal_answer.set_defaults(split=split)
        formal_answer.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
        formal_answer.add_argument("--retrieval-run-dir", type=Path, default=None)
        formal_answer.add_argument(
            "--answer-prereg",
            type=Path,
            default=DEFAULT_FORMAL_ANSWER_PREREG_MANIFEST,
        )
        formal_answer.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_FORMAL_ANSWER_DIR / split,
        )
        formal_answer.add_argument("--resume", type=Path, default=None)
        formal_answer.add_argument("--max-workers", type=int, default=None)

    formal_answer_dev = subparsers.add_parser("freeze-public-tcm-qg-formal-answer-dev")
    formal_answer_dev.add_argument("--run-dir", type=Path, default=None)
    formal_answer_dev.add_argument("--output", type=Path, default=None)

    formal_answer_summary = subparsers.add_parser(
        "summarize-public-tcm-qg-formal-answer-test"
    )
    formal_answer_summary.add_argument("--run-dir", type=Path, default=None)
    formal_answer_summary.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    formal_answer_summary.add_argument("--retrieval-run-dir", type=Path, default=None)
    formal_answer_summary.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)

    formal_review_prepare = subparsers.add_parser(
        "prepare-public-tcm-qg-formal-answer-review"
    )
    formal_review_prepare.add_argument("--answer-run-dir", type=Path, default=None)
    formal_review_prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    formal_review_prepare.add_argument("--retrieval-run-dir", type=Path, default=None)
    formal_review_prepare.add_argument("--config", type=Path, default=DEFAULT_FORMAL_CONFIG_PATH)
    formal_review_prepare.add_argument("--output-dir", type=Path, default=DEFAULT_FORMAL_REVIEW_DIR)
    formal_review_prepare.add_argument("--main-review-questions", type=int, default=None)
    formal_review_prepare.add_argument("--second-review-rate", type=float, default=None)
    formal_review_prepare.add_argument("--parent-ablation-focus-questions", type=int, default=None)
    formal_review_prepare.add_argument("--seed", type=int, default=None)

    formal_review_import = subparsers.add_parser(
        "import-public-tcm-qg-formal-answer-review"
    )
    formal_review_import.add_argument(
        "--reviewed-csv",
        type=Path,
        default=DEFAULT_FORMAL_REVIEW_DIR / "formal-answer-review-main.csv",
    )
    formal_review_import.add_argument(
        "--second-reviewed-csv",
        type=Path,
        default=DEFAULT_FORMAL_REVIEW_DIR / "formal-answer-review-second.csv",
    )
    formal_review_import.add_argument(
        "--parent-ablation-reviewed-csv",
        type=Path,
        default=DEFAULT_FORMAL_REVIEW_DIR / "formal-answer-review-parent-ablation.csv",
    )
    formal_review_import.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FORMAL_REVIEW_DIR / "review-summary.json",
    )

    formal_answer_freeze = subparsers.add_parser(
        "freeze-public-tcm-qg-formal-answer-runs"
    )
    formal_answer_freeze.add_argument("--answer-run-dir", type=Path, default=None)
    formal_answer_freeze.add_argument(
        "--review-summary",
        type=Path,
        default=DEFAULT_FORMAL_REVIEW_DIR / "review-summary.json",
    )
    formal_answer_freeze.add_argument(
        "--answer-prereg",
        type=Path,
        default=DEFAULT_FORMAL_ANSWER_PREREG_MANIFEST,
    )
    formal_answer_freeze.add_argument(
        "--retrieval-manifest",
        type=Path,
        default=DEFAULT_FORMAL_RETRIEVAL_MANIFEST,
    )
    formal_answer_freeze.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FORMAL_ANSWER_RUNS_MANIFEST,
    )

    for command, split in (
        ("run-public-tcm-qg-retrieval-dev", "dev"),
        ("run-public-tcm-qg-retrieval-test", "test"),
    ):
        run = subparsers.add_parser(command)
        run.set_defaults(split=split)
        run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
        run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        run.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
        run.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_RUNS_DIR / split,
        )
        run.add_argument("--resume", type=Path, default=None)

    for command, split in (
        ("run-public-tcm-qg-answer-dev", "dev"),
        ("run-public-tcm-qg-answer-test", "test"),
    ):
        answer = subparsers.add_parser(command)
        answer.set_defaults(split=split)
        answer.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
        answer.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        answer.add_argument("--retrieval-run-dir", type=Path, default=None)
        answer.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_ANSWER_DIR / split,
        )
        answer.add_argument("--resume", type=Path, default=None)

    summary = subparsers.add_parser("summarize-public-tcm-qg-test")
    summary.add_argument("--run-dir", type=Path, default=None)
    summary.add_argument("--retrieval-run-dir", type=Path, default=None)
    summary.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    summary.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    freeze = subparsers.add_parser("freeze-public-tcm-qg-runs")
    freeze.add_argument("--answer-run-dir", type=Path, default=None)
    freeze.add_argument("--retrieval-run-dir", type=Path, default=None)
    freeze.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    freeze.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    freeze.add_argument("--output", type=Path, default=DEFAULT_RUNS_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest: dict
    if args.command == "freeze-public-tcm-qg-source":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        manifest = freeze_public_tcm_qg_source(
            source_path=args.source,
            output_path=args.output,
            public_dataset_url=config["source"]["public_dataset_url"],
            expected_sha256=config["source"]["expected_sha256"],
        )
    elif args.command == "prepare-public-tcm-qg-dataset":
        manifest = prepare_public_tcm_qg_dataset(
            source_path=args.source,
            config_path=args.config,
            output_path=args.output,
            split_path=args.split,
            manifest_path=args.manifest,
        )
    elif args.command == "freeze-public-tcm-qg-dataset":
        manifest = freeze_public_tcm_qg_dataset(
            dataset_path=args.dataset,
            split_path=args.split,
            source_path=args.source,
            config_path=args.config,
            output_path=args.output,
        )
    elif args.command == "build-public-tcm-qg-chunks":
        manifest = build_public_tcm_qg_chunks(
            dataset_path=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
    elif args.command == "build-public-tcm-qg-indexes":
        manifest = build_public_tcm_qg_indexes(
            chunks_dir=args.chunks_dir,
            chunk_manifest_path=args.chunk_manifest,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
    elif args.command == "freeze-public-tcm-qg-formal-prereg":
        manifest = freeze_public_tcm_qg_formal_prereg(
            config_path=args.config,
            env_path=args.env,
            output_path=args.output,
        )
    elif args.command == "build-public-tcm-qg-formal-indexes":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        manifest = build_public_tcm_qg_formal_indexes(
            chunks_dir=args.chunks_dir,
            chunk_manifest_path=args.chunk_manifest,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            embedding_model=config["embedding"]["model"],
            embedding_revision=config["embedding"]["revision"],
            device=config["embedding"]["device"],
            batch_size=int(config["embedding"]["batch_size"]),
            max_length=int(config["embedding"]["max_length"]),
            prereg_manifest_path=args.prereg_manifest,
        )
    elif args.command == "estimate-public-tcm-qg-formal-answer-cost":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        dataset_manifest = read_json(args.dataset_manifest, label="formal dataset manifest")
        from experiments.rag_v1_6.public_tcm_qg_formal_runner import (
            read_formal_answer_model_from_env,
        )

        model = read_formal_answer_model_from_env(args.env)
        manifest = estimate_public_tcm_qg_formal_answer_cost(
            question_count=int(dataset_manifest["dataset"]["by_split"]["test"]),
            methods=list(config["answer"]["methods"]),
            repeats=int(config["answer"]["repeats"]),
            prompt_token_estimate_per_call=args.prompt_token_estimate,
            completion_token_estimate_per_call=args.completion_token_estimate,
            model_name=model["model_name"],
            base_url_origin=model["base_url_origin"],
            seconds_per_call=args.seconds_per_call,
        )
    elif args.command in {
        "run-public-tcm-qg-formal-retrieval-dev",
        "run-public-tcm-qg-formal-retrieval-test",
    }:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        embedder = load_bge_m3_embedder(
            model_name=config["embedding"]["model"],
            revision=config["embedding"]["revision"],
            device=config["embedding"]["device"],
            max_length=int(config["embedding"]["max_length"]),
        )
        reranker = load_bge_reranker(
            model_name=config["reranker"]["model"],
            revision=config["reranker"]["revision"],
            device=config["reranker"]["device"],
            use_fp16=bool(config["reranker"]["use_fp16"]),
            batch_size=int(config["reranker"]["batch_size"]),
            max_length=int(config["reranker"]["max_length"]),
        )
        manifest = run_public_tcm_qg_formal_retrieval_matrix(
            split=args.split,
            dataset_path=args.dataset,
            indexes_dir=args.indexes_dir,
            output_dir=args.output_dir,
            embedder=embedder,
            reranker=reranker,
            prereg_manifest_path=args.prereg_manifest,
            index_manifest_path=args.index_manifest,
            resume_dir=args.resume,
            bm25_top_k=int(config["retrieval"]["bm25_top_k"]),
            dense_top_k=int(config["retrieval"]["dense_top_k"]),
            rrf_k=int(config["retrieval"]["rrf_k"]),
            reranker_candidate_k=int(config["retrieval"]["reranker_candidate_k"]),
            final_top_k=int(config["retrieval"]["final_top_k"]),
            embedding_batch_size=int(config["embedding"]["batch_size"]),
            reranker_batch_size=int(config["reranker"]["batch_size"]),
        )
    elif args.command == "summarize-public-tcm-qg-formal-retrieval-test":
        run_dir = args.run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ROOT / "runs" / "test",
            "matrix-summary.json",
        )
        manifest = summarize_public_tcm_qg_formal_retrieval_test(run_dir=run_dir)
    elif args.command == "freeze-public-tcm-qg-formal-retrieval-runs":
        test_run_dir = args.test_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ROOT / "runs" / "test",
            "matrix-summary.json",
        )
        dev_run_dir = args.dev_run_dir
        if dev_run_dir is None:
            try:
                dev_run_dir = _latest_directory_with_file(
                    DEFAULT_FORMAL_ROOT / "runs" / "dev",
                    "matrix-summary.json",
                )
            except FileNotFoundError:
                dev_run_dir = None
        manifest = freeze_public_tcm_qg_formal_retrieval_runs(
            output_path=args.output,
            prereg_manifest_path=args.prereg_manifest,
            index_manifest_path=args.index_manifest,
            dev_run_dir=dev_run_dir,
            test_run_dir=test_run_dir,
        )
    elif args.command == "freeze-public-tcm-qg-formal-answer-prereg":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        from experiments.rag_v1_6.public_tcm_qg_formal_runner import (
            read_formal_answer_model_from_env,
        )

        model = read_formal_answer_model_from_env(args.env)
        retrieval_test_run_dir = args.retrieval_test_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ROOT / "runs" / "test",
            "matrix-summary.json",
        )
        manifest = freeze_public_tcm_qg_formal_answer_prereg(
            output_path=args.output,
            retrieval_test_run_dir=retrieval_test_run_dir,
            answer_methods=list(config["answer"]["methods"]),
            temperature=config["answer"]["temperature"],
            repeats=int(config["answer"]["repeats"]),
            max_tokens=int(config["answer"]["max_tokens"]),
            model_name=model["model_name"],
            base_url_origin=model["base_url_origin"],
        )
    elif args.command in {
        "run-public-tcm-qg-formal-answer-dev",
        "run-public-tcm-qg-formal-answer-test",
    }:
        config = yaml.safe_load(DEFAULT_FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))
        retrieval_root = DEFAULT_FORMAL_ROOT / "runs" / args.split
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            retrieval_root,
            "matrix-summary.json",
        )
        max_workers = (
            args.max_workers
            if args.max_workers is not None
            else int(config["answer"]["max_workers"])
        )
        manifest = run_public_tcm_qg_formal_answer_matrix(
            split=args.split,
            dataset_path=args.dataset,
            retrieval_matrix_dir=retrieval_run_dir,
            answer_prereg_path=args.answer_prereg,
            output_dir=args.output_dir,
            resume_dir=args.resume,
            max_workers=max_workers,
            top_k=int(config["retrieval"]["answer_context_top_k"]),
        )
    elif args.command == "freeze-public-tcm-qg-formal-answer-dev":
        run_dir = args.run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ANSWER_DIR / "dev",
            "matrix-summary.json",
        )
        manifest = freeze_public_tcm_qg_formal_answer_dev(
            run_dir=run_dir,
            output_path=args.output,
        )
    elif args.command == "summarize-public-tcm-qg-formal-answer-test":
        run_dir = args.run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ANSWER_DIR / "test",
            "matrix-summary.json",
        )
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ROOT / "runs" / "test",
            "matrix-summary.json",
        )
        manifest = summarize_public_tcm_qg_formal_answer_test(
            run_dir=run_dir,
            dataset_path=args.dataset,
            retrieval_matrix_dir=retrieval_run_dir,
            config_path=args.config,
        )
    elif args.command == "prepare-public-tcm-qg-formal-answer-review":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        review_config = config["human_review"]
        statistics = config["statistics"]
        answer_run_dir = args.answer_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ANSWER_DIR / "test",
            "matrix-summary.json",
        )
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ROOT / "runs" / "test",
            "matrix-summary.json",
        )
        manifest = prepare_public_tcm_qg_formal_answer_review(
            answer_run_dir=answer_run_dir,
            dataset_path=args.dataset,
            retrieval_matrix_dir=retrieval_run_dir,
            output_dir=args.output_dir,
            main_review_questions=(
                args.main_review_questions
                if args.main_review_questions is not None
                else int(review_config["main_review_questions"])
            ),
            second_review_rate=(
                args.second_review_rate
                if args.second_review_rate is not None
                else float(review_config["second_review_rate"])
            ),
            parent_ablation_focus_questions=(
                args.parent_ablation_focus_questions
                if args.parent_ablation_focus_questions is not None
                else int(review_config["parent_ablation_focus_questions"])
            ),
            seed=args.seed if args.seed is not None else int(statistics["bootstrap_seed"]),
        )
    elif args.command == "import-public-tcm-qg-formal-answer-review":
        manifest = import_public_tcm_qg_formal_answer_review(
            reviewed_csv_path=args.reviewed_csv,
            second_reviewed_csv_path=args.second_reviewed_csv,
            parent_ablation_reviewed_csv_path=args.parent_ablation_reviewed_csv,
            output_path=args.output,
        )
    elif args.command == "freeze-public-tcm-qg-formal-answer-runs":
        answer_run_dir = args.answer_run_dir or _latest_directory_with_file(
            DEFAULT_FORMAL_ANSWER_DIR / "test",
            "success-gate.json",
        )
        manifest = freeze_public_tcm_qg_formal_answer_runs(
            answer_run_dir=answer_run_dir,
            review_summary_path=args.review_summary,
            output_path=args.output,
            answer_prereg_path=args.answer_prereg,
            retrieval_runs_manifest_path=args.retrieval_manifest,
        )
    elif args.command in {
        "run-public-tcm-qg-retrieval-dev",
        "run-public-tcm-qg-retrieval-test",
    }:
        manifest = run_public_tcm_qg_retrieval_matrix(
            split=args.split,
            dataset_path=args.dataset,
            config_path=args.config,
            indexes_dir=args.indexes_dir,
            output_dir=args.output_dir,
            resume_dir=args.resume,
        )
    elif args.command in {
        "run-public-tcm-qg-answer-dev",
        "run-public-tcm-qg-answer-test",
    }:
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            DEFAULT_RUNS_DIR / args.split,
            "matrix-summary.json",
        )
        manifest = run_public_tcm_qg_answer_matrix(
            split=args.split,
            dataset_path=args.dataset,
            retrieval_matrix_dir=retrieval_run_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            resume_dir=args.resume,
        )
    elif args.command == "summarize-public-tcm-qg-test":
        run_dir = args.run_dir or _latest_directory_with_file(
            DEFAULT_ANSWER_DIR / "test",
            "matrix-summary.json",
        )
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            DEFAULT_RUNS_DIR / "test",
            "matrix-summary.json",
        )
        manifest = summarize_public_tcm_qg_test(
            run_dir=run_dir,
            dataset_path=args.dataset,
            retrieval_matrix_dir=retrieval_run_dir,
            config_path=args.config,
        )
    elif args.command == "freeze-public-tcm-qg-runs":
        answer_run_dir = args.answer_run_dir or _latest_directory_with_file(
            DEFAULT_ANSWER_DIR / "test",
            "success-gate.json",
        )
        retrieval_run_dir = args.retrieval_run_dir or _latest_directory_with_file(
            DEFAULT_RUNS_DIR / "test",
            "matrix-summary.json",
        )
        automatic = read_json(
            answer_run_dir / "automatic-metrics.json",
            label="automatic metrics",
        )
        paired = read_json(answer_run_dir / "paired-bootstrap.json", label="paired bootstrap")
        gate = read_json(answer_run_dir / "success-gate.json", label="success gate")
        manifest = freeze_public_tcm_qg_runs(
            metrics={
                "status": "ready",
                "answer_mode": automatic.get("answer_mode"),
                "by_method": automatic.get("by_method", {}),
                "paired_comparisons": paired.get("comparisons", []),
                "success_gate": gate,
            },
            output_path=args.output,
            source_manifest_path=args.source_manifest,
            dataset_manifest_path=args.dataset_manifest,
            retrieval_run_dir=retrieval_run_dir,
            answer_run_dir=answer_run_dir,
        )
    else:
        parser.error(f"unknown command: {args.command}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
