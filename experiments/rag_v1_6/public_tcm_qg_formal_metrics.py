from collections import defaultdict
from pathlib import Path

import yaml

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    char_f1,
    exact_match,
    mean,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_jsonl,
)
from experiments.rag_v1_6.public_tcm_qg_formal_answer import (
    load_public_tcm_qg_formal_answer_inputs,
)
from experiments.rag_v1_6.public_tcm_qg_metrics import (
    paired_bootstrap_by_doc,
    rouge_l_f1,
)
from experiments.rag_v1_6.schema import PublicTcmQgAnswerRecord


def classify_success_gate(comparisons: dict) -> dict:
    p_b4 = comparisons["P-B4"]
    p_child = comparisons["P-P-no-parent"]
    parent_ablation = (
        p_child["char_f1_delta"] > 0
        and p_child["char_f1_ci_lower"] > 0
        and p_child["citation_recall_delta"] > 0
        and p_child["citation_recall_ci_lower"] > 0
    )
    strong = (
        parent_ablation
        and p_b4["char_f1_delta"] > 0
        and p_b4["char_f1_ci_lower"] > 0
        and p_b4["citation_recall_delta"] >= 0
        and p_b4["unsupported_answer_rate_delta"] <= 0
    )
    if strong:
        gate = "strong_success"
        interpretation = "Parent-Child may be reported as better than B4."
    elif parent_ablation:
        gate = "parent_ablation_only"
        interpretation = "Only parent context over child-only can be claimed."
    else:
        gate = "failed"
        interpretation = "Report failure analysis; do not package as success."
    return {
        "status": "ready",
        "success_gate": gate,
        "strong_success": strong,
        "parent_ablation_only": parent_ablation and not strong,
        "failed": gate == "failed",
        "interpretation": interpretation,
    }


def _citation_metrics(
    *,
    citations: list[str],
    evidence: list[dict],
    reference_answer: str,
) -> dict:
    evidence_by_label = {item["label"]: item for item in evidence}
    cited = [evidence_by_label[label] for label in citations if label in evidence_by_label]
    supported = [item for item in cited if reference_answer in item["text"]]
    return {
        "citation_precision": len(supported) / len(citations) if citations else 0.0,
        "citation_recall": 1.0 if supported else 0.0,
        "supported_citation_count": len(supported),
        "citation_count": len(citations),
    }


def summarize_formal_answer_rows(
    *,
    questions: dict[str, dict],
    answers: list[dict],
    retrieval: dict[str, dict],
) -> dict:
    per_answer = []
    grouped = defaultdict(list)
    for answer in answers:
        answer_record = PublicTcmQgAnswerRecord.model_validate(answer)
        question = questions[answer_record.qa_id]
        evidence = (
            retrieval.get(answer_record.method, {})
            .get(answer_record.qa_id, {})
            .get("evidence", [])
        )
        citation = (
            _citation_metrics(
                citations=answer_record.citations,
                evidence=evidence,
                reference_answer=question["answer"],
            )
            if answer_record.method != "B0"
            else {
                "citation_precision": None,
                "citation_recall": None,
                "supported_citation_count": 0,
                "citation_count": 0,
            }
        )
        unsupported = (
            None
            if answer_record.method == "B0"
            else int(not answer_record.abstain and citation["citation_recall"] < 1.0)
        )
        row = {
            "qa_id": answer_record.qa_id,
            "source_doc_id": answer_record.source_doc_id,
            "split": answer_record.split,
            "method": answer_record.method,
            "repeat_index": answer_record.repeat_index,
            "exact_match": exact_match(answer_record.answer, question["answer"]),
            "char_f1": char_f1(answer_record.answer, question["answer"]),
            "rouge_l_f1": rouge_l_f1(answer_record.answer, question["answer"]),
            "citation_precision": citation["citation_precision"],
            "citation_recall": citation["citation_recall"],
            "unsupported_answer": unsupported,
            "abstain": answer_record.abstain,
            "retrieval_supported": answer_record.retrieval_supported,
            "latency_ms": answer_record.latency_ms,
            "input_tokens": answer_record.input_tokens,
            "output_tokens": answer_record.output_tokens,
        }
        per_answer.append(row)
        grouped[answer_record.method].append(row)

    by_method = {}
    for method, rows in sorted(grouped.items()):
        citation_rows = [row for row in rows if row["citation_recall"] is not None]
        unsupported_rows = [row for row in rows if row["unsupported_answer"] is not None]
        by_method[method] = {
            "run_count": len(rows),
            "question_count": len({row["qa_id"] for row in rows}),
            "exact_match": mean(row["exact_match"] for row in rows),
            "char_f1": mean(row["char_f1"] for row in rows),
            "rouge_l_f1": mean(row["rouge_l_f1"] for row in rows),
            "citation_precision": (
                mean(row["citation_precision"] for row in citation_rows)
                if citation_rows
                else None
            ),
            "citation_recall": (
                mean(row["citation_recall"] for row in citation_rows)
                if citation_rows
                else None
            ),
            "unsupported_answer_rate": (
                mean(row["unsupported_answer"] for row in unsupported_rows)
                if unsupported_rows
                else None
            ),
            "abstain_rate": mean(row["abstain"] for row in rows),
            "retrieval_supported_rate": mean(row["retrieval_supported"] for row in rows),
            "latency_ms_mean": mean(row["latency_ms"] for row in rows),
            "input_tokens_mean": mean(row["input_tokens"] for row in rows),
            "output_tokens_mean": mean(row["output_tokens"] for row in rows),
        }
    return {"by_method": by_method, "per_answer": per_answer}


def _bootstrap_comparisons(
    *,
    rows: list[dict],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[list[dict], dict]:
    comparison_specs = [
        ("B4", "B0", ["char_f1"]),
        ("P", "B4", ["char_f1", "citation_recall", "unsupported_answer"]),
        ("P", "P-no-parent", ["char_f1", "citation_recall", "unsupported_answer"]),
    ]
    bootstrap_rows = []
    comparison_lookup = {}
    for method_a, method_b, metrics in comparison_specs:
        key = f"{method_a}-{method_b}"
        comparison_lookup[key] = {}
        for metric in metrics:
            boot = paired_bootstrap_by_doc(
                rows=rows,
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                seed=seed,
                resamples=resamples,
                confidence_level=confidence_level,
            )
            bootstrap_rows.append(boot)
            metric_key = "unsupported_answer_rate" if metric == "unsupported_answer" else metric
            comparison_lookup[key][f"{metric_key}_delta"] = boot["delta"]
            comparison_lookup[key][f"{metric_key}_ci_lower"] = boot["ci_lower"]
            comparison_lookup[key][f"{metric_key}_ci_upper"] = boot["ci_upper"]
    return bootstrap_rows, comparison_lookup


def summarize_public_tcm_qg_formal_answer_test(
    *,
    run_dir: Path,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    config_path: Path,
) -> dict:
    matrix_summary = read_json(run_dir / "matrix-summary.json", label="formal answer summary")
    if (
        matrix_summary.get("status") != "completed"
        or matrix_summary.get("split") != "test"
        or matrix_summary.get("error_count", 0) != 0
    ):
        raise ValueError("only a completed zero-error formal test answer run can be summarized")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = load_public_tcm_qg_formal_answer_inputs(
        split="test",
        dataset_path=dataset_path,
        retrieval_matrix_dir=retrieval_matrix_dir,
        top_k=int(config["retrieval"]["answer_context_top_k"]),
    )
    answers = read_jsonl(run_dir / "per-answer.jsonl", label="formal answer rows")
    result = summarize_formal_answer_rows(
        questions=loaded["questions"],
        answers=answers,
        retrieval=loaded["retrieval"],
    )
    statistics = config["statistics"]
    bootstrap_rows, comparisons = _bootstrap_comparisons(
        rows=result["per_answer"],
        seed=int(statistics["bootstrap_seed"]),
        resamples=int(statistics["bootstrap_resamples"]),
        confidence_level=float(statistics["confidence_level"]),
    )
    gate = classify_success_gate(comparisons)
    safe_fields = (
        "qa_id",
        "source_doc_id",
        "split",
        "method",
        "repeat_index",
        "exact_match",
        "char_f1",
        "rouge_l_f1",
        "citation_precision",
        "citation_recall",
        "unsupported_answer",
        "abstain",
        "retrieval_supported",
        "latency_ms",
        "input_tokens",
        "output_tokens",
    )
    per_question_path = run_dir / "per-question-metrics.jsonl"
    write_jsonl(
        per_question_path,
        [{field: row.get(field) for field in safe_fields} for row in result["per_answer"]],
    )
    automatic = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_answer_metrics_ready",
        "generated_at": utc_now(),
        "answer_mode": "formal_llm",
        "run_count": len(answers),
        "by_method": result["by_method"],
        "definitions": {
            "exact_match": "normalized exact match against reference answer",
            "char_f1": "character-level F1 against reference answer",
            "rouge_l_f1": "longest common subsequence F1",
            "citation_precision": "share of cited public evidence items containing the reference answer span",
            "citation_recall": "1 when at least one cited public evidence item contains the reference answer span",
            "unsupported_answer_rate": "answered evidence method without a cited supporting span",
        },
        "inputs": {
            "run_summary_sha256": sha256_file(run_dir / "matrix-summary.json"),
            "per_answer_sha256": sha256_file(run_dir / "per-answer.jsonl"),
            "dataset_sha256": sha256_file(dataset_path),
            "retrieval_matrix_summary_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-summary.json"
            ),
            "config_sha256": sha256_file(config_path),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_included": False,
        },
    }
    paired = {
        "version": VERSION,
        "status": "ready",
        "paired": True,
        "resample_unit": "source_doc_id",
        "comparisons": bootstrap_rows,
    }
    atomic_write_json(run_dir / "automatic-metrics.json", automatic)
    atomic_write_json(run_dir / "paired-bootstrap.json", paired)
    atomic_write_json(run_dir / "success-gate.json", gate)
    return {
        "version": VERSION,
        "status": "ready",
        "run_dir": run_dir.as_posix(),
        "answer_mode": "formal_llm",
        "by_method": result["by_method"],
        "paired_comparisons": comparisons,
        "success_gate": gate,
        "files": {
            "automatic_metrics_sha256": sha256_file(run_dir / "automatic-metrics.json"),
            "paired_bootstrap_sha256": sha256_file(run_dir / "paired-bootstrap.json"),
            "success_gate_sha256": sha256_file(run_dir / "success-gate.json"),
            "per_question_metrics_sha256": sha256_file(per_question_path),
        },
    }


def freeze_public_tcm_qg_formal_answer_runs(
    *,
    answer_run_dir: Path,
    review_summary_path: Path,
    output_path: Path,
    answer_prereg_path: Path,
    retrieval_runs_manifest_path: Path,
) -> dict:
    automatic_path = answer_run_dir / "automatic-metrics.json"
    paired_path = answer_run_dir / "paired-bootstrap.json"
    gate_path = answer_run_dir / "success-gate.json"
    automatic = read_json(automatic_path, label="formal automatic metrics")
    paired = read_json(paired_path, label="formal paired bootstrap")
    gate = read_json(gate_path, label="formal success gate")
    review = read_json(review_summary_path, label="formal answer review summary")
    if automatic.get("status") != "ready":
        raise ValueError("formal automatic metrics must be ready")
    if paired.get("status") != "ready":
        raise ValueError("formal paired bootstrap must be ready")
    if gate.get("status") != "ready":
        raise ValueError("formal success gate must be ready")
    if review.get("status") != "ready" or not review.get("answer_review_completed"):
        raise ValueError("formal answer review must be completed before freezing runs")
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_answer_runs_frozen",
        "generated_at": utc_now(),
        "answer_run": {
            "path": answer_run_dir.as_posix(),
            "matrix_summary_sha256": sha256_file(answer_run_dir / "matrix-summary.json"),
            "per_answer_sha256": sha256_file(answer_run_dir / "per-answer.jsonl"),
        },
        "automatic_metrics": {
            "path": automatic_path.as_posix(),
            "sha256": sha256_file(automatic_path),
            "by_method": automatic.get("by_method", {}),
        },
        "paired_bootstrap": {
            "path": paired_path.as_posix(),
            "sha256": sha256_file(paired_path),
            "comparisons": paired.get("comparisons", []),
        },
        "success_gate": gate,
        "human_review": {
            "path": review_summary_path.as_posix(),
            "sha256": sha256_file(review_summary_path),
            "reviewed_count": review["reviewed_count"],
            "second_review_count": review["second_review_count"],
            "disagreement_count": review["disagreement_count"],
            "metrics": review.get("metrics", {}),
        },
        "inputs": {
            "answer_prereg_sha256": sha256_file(answer_prereg_path),
            "retrieval_runs_manifest_sha256": sha256_file(retrieval_runs_manifest_path),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_included": False,
            "review_free_text_included": False,
        },
    }
    serialized = str(manifest)
    forbidden = [
        "source_text",
        "question_text",
        "reference_answer",
        "answer_text",
        "evidence_text",
        "reviewer_comment",
    ]
    leaked = [field for field in forbidden if field in serialized]
    if leaked:
        raise ValueError(f"manifest includes forbidden fields: {', '.join(leaked)}")
    atomic_write_json(output_path, manifest)
    return manifest
