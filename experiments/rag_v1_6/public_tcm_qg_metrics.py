import random
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
from experiments.rag_v1_6.public_tcm_qg_answer import load_public_answer_inputs
from experiments.rag_v1_6.schema import PublicTcmQgAnswerRecord


def _lcs_length(left: str, right: str) -> int:
    dp = [0] * (len(right) + 1)
    for left_char in left:
        previous = 0
        for index, right_char in enumerate(right, start=1):
            saved = dp[index]
            if left_char == right_char:
                dp[index] = previous + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous = saved
    return dp[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    lcs = _lcs_length(prediction, reference)
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _citation_metrics(
    *,
    citations: list[str],
    evidence: list[dict],
    reference_answer: str,
) -> dict:
    evidence_by_label = {item["label"]: item for item in evidence}
    cited = [evidence_by_label[label] for label in citations if label in evidence_by_label]
    supported = [
        item
        for item in cited
        if reference_answer in item["text"]
    ]
    return {
        "citation_precision": len(supported) / len(citations) if citations else 0.0,
        "citation_recall": 1.0 if supported else 0.0,
        "supported_citation_count": len(supported),
        "citation_count": len(citations),
    }


def summarize_public_answer_rows(
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
        exact = exact_match(answer_record.answer, question["answer"])
        f1 = char_f1(answer_record.answer, question["answer"])
        rouge = rouge_l_f1(answer_record.answer, question["answer"])
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
            "exact_match": exact,
            "char_f1": f1,
            "rouge_l_f1": rouge,
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
        unsupported_rows = [
            row for row in rows if row["unsupported_answer"] is not None
        ]
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
            "retrieval_supported_rate": mean(
                row["retrieval_supported"] for row in rows
            ),
            "latency_ms_mean": mean(row["latency_ms"] for row in rows),
            "input_tokens_mean": mean(row["input_tokens"] for row in rows),
            "output_tokens_mean": mean(row["output_tokens"] for row in rows),
        }
    return {"by_method": by_method, "per_answer": per_answer}


def _paired_metric_by_doc(
    *,
    rows: list[dict],
    method_a: str,
    method_b: str,
    metric: str,
) -> dict[str, list[float]]:
    grouped = defaultdict(list)
    values = defaultdict(list)
    for row in rows:
        if row["method"] in {method_a, method_b} and row.get(metric) is not None:
            values[(row["qa_id"], row["method"])].append(float(row[metric]))
            grouped[row["qa_id"]].append(row)
    by_doc = defaultdict(list)
    for qa_id, qa_rows in grouped.items():
        if (qa_id, method_a) not in values or (qa_id, method_b) not in values:
            continue
        source_doc_id = qa_rows[0]["source_doc_id"]
        by_doc[source_doc_id].append(
            mean(values[(qa_id, method_a)]) - mean(values[(qa_id, method_b)])
        )
    if not by_doc:
        raise ValueError(f"no paired rows for {method_a}-{method_b} {metric}")
    return dict(by_doc)


def paired_bootstrap_by_doc(
    *,
    rows: list[dict],
    method_a: str,
    method_b: str,
    metric: str,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict:
    by_doc = _paired_metric_by_doc(
        rows=rows,
        method_a=method_a,
        method_b=method_b,
        metric=metric,
    )
    doc_ids = sorted(by_doc)
    observed = mean(delta for deltas in by_doc.values() for delta in deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        sampled = [doc_ids[rng.randrange(len(doc_ids))] for _ in doc_ids]
        samples.append(mean(delta for doc_id in sampled for delta in by_doc[doc_id]))
    samples.sort()
    alpha = 1 - confidence_level
    lower_index = max(0, min(resamples - 1, int((alpha / 2) * resamples)))
    upper_index = max(
        0,
        min(resamples - 1, int((1 - alpha / 2) * resamples) - 1),
    )
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "resample_unit": "source_doc_id",
        "source_doc_count": len(doc_ids),
        "delta": observed,
        "ci_lower": samples[lower_index],
        "ci_upper": samples[upper_index],
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
    }


def success_gate(*, comparisons: dict, by_method: dict) -> dict:
    p_b4 = comparisons["P-B4"]
    p_child = comparisons["P-P-no-parent"]
    success = (
        p_b4["char_f1_delta"] > 0
        and p_b4["char_f1_ci_low"] > 0
        and p_child["char_f1_delta"] > 0
        and p_child["char_f1_ci_low"] > 0
        and by_method["P"]["citation_recall"]
        >= by_method["P-no-parent"]["citation_recall"]
        and by_method["P"]["unsupported_answer_rate"]
        <= by_method["P-no-parent"]["unsupported_answer_rate"]
    )
    return {
        "public_tcm_qg_success": bool(success),
        "skip_mtcmb_tcm_litdata": bool(success),
        "mode": "extractive_oracle_proxy",
    }


def summarize_public_tcm_qg_test(
    *,
    run_dir: Path,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    config_path: Path,
) -> dict:
    matrix_summary = read_json(run_dir / "matrix-summary.json", label="answer summary")
    if matrix_summary.get("status") != "completed" or matrix_summary.get("split") != "test":
        raise ValueError("only a completed test answer run can be summarized")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = load_public_answer_inputs(
        split="test",
        dataset_path=dataset_path,
        retrieval_matrix_dir=retrieval_matrix_dir,
        top_k=int(config["retrieval"]["answer_context_top_k"]),
    )
    answers = read_jsonl(run_dir / "per-answer.jsonl", label="public answer rows")
    result = summarize_public_answer_rows(
        questions=loaded["questions"],
        answers=answers,
        retrieval=loaded["retrieval"],
    )
    evaluation = config["statistics"]
    comparison_specs = [
        ("B4", "B0", ["char_f1"]),
        ("P", "B4", ["char_f1", "citation_recall", "unsupported_answer"]),
        ("P", "P-no-parent", ["char_f1", "citation_recall", "unsupported_answer"]),
    ]
    bootstrap_rows = []
    comparison_lookup = {}
    for method_a, method_b, metrics in comparison_specs:
        comparison_key = f"{method_a}-{method_b}"
        comparison_lookup[comparison_key] = {}
        for metric in metrics:
            boot = paired_bootstrap_by_doc(
                rows=result["per_answer"],
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                seed=int(evaluation["bootstrap_seed"]),
                resamples=int(evaluation["bootstrap_resamples"]),
                confidence_level=float(evaluation["confidence_level"]),
            )
            bootstrap_rows.append(boot)
            comparison_lookup[comparison_key][f"{metric}_delta"] = boot["delta"]
            comparison_lookup[comparison_key][f"{metric}_ci_low"] = boot["ci_lower"]
            comparison_lookup[comparison_key][f"{metric}_ci_high"] = boot["ci_upper"]
    gate = success_gate(
        comparisons=comparison_lookup,
        by_method=result["by_method"],
    )
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
        "generated_at": utc_now(),
        "answer_mode": matrix_summary.get("answer_mode"),
        "run_count": len(answers),
        "by_method": result["by_method"],
        "definitions": {
            "char_f1": "character-level F1 against reference answer",
            "rouge_l_f1": "longest common subsequence F1",
            "citation_recall": "1 when a cited public evidence item contains the reference answer span",
            "unsupported_answer_rate": "answered evidence method without cited supporting span",
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
        "answer_mode": matrix_summary.get("answer_mode"),
        "by_method": result["by_method"],
        "paired_comparisons": comparison_lookup,
        "success_gate": gate,
        "files": {
            "automatic_metrics_sha256": sha256_file(run_dir / "automatic-metrics.json"),
            "paired_bootstrap_sha256": sha256_file(run_dir / "paired-bootstrap.json"),
            "success_gate_sha256": sha256_file(run_dir / "success-gate.json"),
            "per_question_metrics_sha256": sha256_file(per_question_path),
        },
    }
