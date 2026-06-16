import json
import random
import unicodedata
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev

import yaml

from experiments.rag_v1_5.runner import (
    _atomic_write_json,
    _read_json,
    _read_jsonl_strict,
    _sha256_file,
)


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "S"))
    )


def exact_match(prediction: str, reference: str) -> float:
    return float(
        normalize_answer(prediction) == normalize_answer(reference)
    )


def char_f1(prediction: str, reference: str) -> float:
    prediction_counts = Counter(normalize_answer(prediction))
    reference_counts = Counter(normalize_answer(reference))
    prediction_length = sum(prediction_counts.values())
    reference_length = sum(reference_counts.values())
    if prediction_length == 0 and reference_length == 0:
        return 1.0
    if prediction_length == 0 or reference_length == 0:
        return 0.0
    overlap = sum(
        min(count, reference_counts[char])
        for char, count in prediction_counts.items()
    )
    precision = overlap / prediction_length
    recall = overlap / reference_length
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def citation_metrics(
    *,
    citations: list[str],
    evidence: dict[str, dict],
    gold_clause_ids: list[str],
) -> dict:
    gold = set(gold_clause_ids)
    cited_items = [
        evidence[label] for label in citations if label in evidence
    ]
    supported_citations = sum(
        bool(set(item.get("clause_ids", ())) & gold)
        for item in cited_items
    )
    precision = (
        supported_citations / len(citations) if citations else 0.0
    )
    covered_gold = set()
    for item in cited_items:
        covered_gold.update(
            set(item.get("clause_ids", ())) & gold
        )
    recall = len(covered_gold) / len(gold) if gold else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "supported_citation_count": supported_citations,
        "citation_count": len(citations),
        "covered_gold_clause_count": len(covered_gold),
        "gold_clause_count": len(gold),
    }


def refusal_metrics(
    *,
    answerable: bool,
    abstain: bool,
) -> dict:
    correct = (answerable and not abstain) or (
        not answerable and abstain
    )
    return {
        "correct": int(correct),
        "answerable_answered": int(answerable and not abstain),
        "answerable_refused": int(answerable and abstain),
        "unanswerable_refused": int(
            not answerable and abstain
        ),
        "unsupported_answer": int(
            not answerable and not abstain
        ),
    }


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def summarize_answer_rows(
    *,
    questions: dict[str, dict],
    answers: list[dict],
    retrieval: dict[str, dict],
) -> dict:
    per_answer = []
    grouped = defaultdict(list)
    for answer in answers:
        question = questions[answer["question_id"]]
        method = answer["method"]
        evidence_items = (
            retrieval.get(method, {})
            .get(answer["question_id"], {})
            .get("evidence", [])
        )
        evidence = {
            item["label"]: item for item in evidence_items
        }
        refusal = refusal_metrics(
            answerable=question["answerable"],
            abstain=answer["abstain"],
        )
        row = {
            **answer,
            "answerable": question["answerable"],
            "exact_match": (
                exact_match(
                    answer["answer"],
                    question["reference_answer"],
                )
                if question["answerable"]
                else None
            ),
            "char_f1": (
                char_f1(
                    answer["answer"],
                    question["reference_answer"],
                )
                if question["answerable"]
                else None
            ),
            "refusal_correct": refusal["correct"],
            "unsupported_answer": (
                refusal["unsupported_answer"]
                if not question["answerable"]
                else None
            ),
        }
        if method != "B0" and question["answerable"]:
            citation = citation_metrics(
                citations=answer["citations"],
                evidence=evidence,
                gold_clause_ids=question["gold_clause_ids"],
            )
            row["citation_precision"] = citation["precision"]
            row["citation_recall"] = citation["recall"]
        else:
            row["citation_precision"] = None
            row["citation_recall"] = None
        per_answer.append(row)
        grouped[method].append(row)

    by_method = {}
    for method, rows in sorted(grouped.items()):
        answerable_rows = [
            row for row in rows if row["answerable"]
        ]
        unanswerable_rows = [
            row for row in rows if not row["answerable"]
        ]
        citation_rows = [
            row
            for row in answerable_rows
            if row["citation_precision"] is not None
        ]
        per_question = defaultdict(list)
        for row in rows:
            per_question[row["question_id"]].append(row)
        stability = []
        repeat_char_f1_std = []
        for question_rows in per_question.values():
            signatures = {
                (
                    normalize_answer(row["answer"]),
                    row["abstain"],
                    tuple(sorted(row["citations"])),
                )
                for row in question_rows
            }
            stability.append(float(len(signatures) == 1))
            char_values = [
                row["char_f1"]
                for row in question_rows
                if row["char_f1"] is not None
            ]
            if char_values:
                repeat_char_f1_std.append(
                    pstdev(char_values)
                    if len(char_values) > 1
                    else 0.0
                )
        by_method[method] = {
            "run_count": len(rows),
            "answerable_run_count": len(answerable_rows),
            "unanswerable_run_count": len(unanswerable_rows),
            "exact_match": _mean(
                [row["exact_match"] for row in answerable_rows]
            ),
            "char_f1": _mean(
                [row["char_f1"] for row in answerable_rows]
            ),
            "citation_precision": (
                _mean(
                    [
                        row["citation_precision"]
                        for row in citation_rows
                    ]
                )
                if citation_rows
                else None
            ),
            "citation_recall": (
                _mean(
                    [
                        row["citation_recall"]
                        for row in citation_rows
                    ]
                )
                if citation_rows
                else None
            ),
            "refusal_accuracy": _mean(
                [row["refusal_correct"] for row in rows]
            ),
            "answerable_response_rate": _mean(
                [
                    float(not row["abstain"])
                    for row in answerable_rows
                ]
            ),
            "unanswerable_refusal_accuracy": _mean(
                [
                    float(row["abstain"])
                    for row in unanswerable_rows
                ]
            ),
            "unsupported_answer_rate": _mean(
                [
                    row["unsupported_answer"]
                    for row in unanswerable_rows
                ]
            ),
            "latency_ms_mean": _mean(
                [float(row["latency_ms"]) for row in rows]
            ),
            "input_tokens_mean": _mean(
                [float(row["input_tokens"]) for row in rows]
            ),
            "output_tokens_mean": _mean(
                [float(row["output_tokens"]) for row in rows]
            ),
            "answer_stability": _mean(stability),
            "repeat_char_f1_std_mean": _mean(
                repeat_char_f1_std
            ),
        }
    return {
        "by_method": by_method,
        "per_answer": per_answer,
    }


def paired_bootstrap(
    *,
    rows: list[dict],
    method_a: str,
    method_b: str,
    metric: str,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if (
            row["method"] not in {method_a, method_b}
            or row.get(metric) is None
        ):
            continue
        grouped[
            (row["question_id"], row["method"])
        ].append(float(row[metric]))
    question_ids = sorted(
        {
            question_id
            for question_id, method in grouped
            if (question_id, method_a) in grouped
            and (question_id, method_b) in grouped
        }
    )
    if not question_ids:
        raise ValueError(
            f"{method_a}/{method_b} 的 {metric} 没有配对问题"
        )
    deltas = {
        question_id: (
            _mean(grouped[(question_id, method_a)])
            - _mean(grouped[(question_id, method_b)])
        )
        for question_id in question_ids
    }
    observed = _mean(list(deltas.values()))
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        sampled_ids = [
            rng.choice(question_ids) for _ in question_ids
        ]
        samples.append(
            _mean([deltas[question_id] for question_id in sampled_ids])
        )
    samples.sort()
    alpha = 1 - confidence_level
    lower_index = max(
        0,
        int((alpha / 2) * resamples),
    )
    upper_index = min(
        resamples - 1,
        int((1 - alpha / 2) * resamples) - 1,
    )
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "question_count": len(question_ids),
        "delta": observed,
        "confidence_level": confidence_level,
        "ci_lower": samples[lower_index],
        "ci_upper": samples[upper_index],
        "resamples": resamples,
        "seed": seed,
    }


def summarize_formal_answer_test(
    *,
    run_dir: Path,
    dataset_path: Path,
    matrix_dir: Path,
    answer_prereg_path: Path,
    config_path: Path,
    formal_manifest_path: Path,
    formal_runs_manifest_path: Path,
) -> dict:
    from experiments.rag_v1_5.formal_answer import (
        load_frozen_answer_inputs,
    )

    matrix_summary = _read_json(
        run_dir / "matrix-summary.json",
        label="Formal answer test summary",
    )
    if (
        matrix_summary.get("status") != "completed"
        or matrix_summary.get("split") != "formal_test"
        or matrix_summary.get("error_count") != 0
    ):
        raise ValueError(
            "自动指标只允许汇总无未解决错误的 completed formal_test"
        )
    answers = _read_jsonl_strict(
        run_dir / "per-answer.jsonl",
        label="Formal answer test records",
    )
    expected_runs = matrix_summary["expected_runs"]
    keys = {
        (
            row["question_id"],
            row["method"],
            row["repeat_index"],
        )
        for row in answers
    }
    if len(answers) != expected_runs or len(keys) != expected_runs:
        raise ValueError("Formal answer test 记录数量或唯一性不一致")
    loaded = load_frozen_answer_inputs(
        dataset_path=dataset_path,
        matrix_dir=matrix_dir,
        answer_prereg_path=answer_prereg_path,
        split="formal_test",
        formal_manifest_path=formal_manifest_path,
        formal_runs_manifest_path=formal_runs_manifest_path,
    )
    result = summarize_answer_rows(
        questions=loaded["questions"],
        answers=answers,
        retrieval=loaded["retrieval"],
    )
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    evaluation = config["evaluation"]
    comparisons = (
        ("B4", "B0", ("char_f1", "refusal_correct")),
        (
            "P",
            "B4",
            (
                "char_f1",
                "citation_recall",
                "refusal_correct",
            ),
        ),
        (
            "P",
            "P-no-parent",
            (
                "char_f1",
                "citation_recall",
                "refusal_correct",
            ),
        ),
    )
    bootstrap_results = []
    for method_a, method_b, metrics in comparisons:
        for metric in metrics:
            bootstrap_results.append(
                paired_bootstrap(
                    rows=result["per_answer"],
                    method_a=method_a,
                    method_b=method_b,
                    metric=metric,
                    seed=evaluation["bootstrap_seed"],
                    resamples=evaluation["bootstrap_resamples"],
                    confidence_level=evaluation[
                        "confidence_level"
                    ],
                )
            )
    for method_a, method_b in (
        ("B4", "B0"),
        ("P", "B4"),
        ("P", "P-no-parent"),
    ):
        bootstrap_results.append(
            paired_bootstrap(
                rows=result["per_answer"],
                method_a=method_a,
                method_b=method_b,
                metric="unsupported_answer",
                seed=evaluation["bootstrap_seed"],
                resamples=evaluation["bootstrap_resamples"],
                confidence_level=evaluation["confidence_level"],
            )
        )

    per_question_path = run_dir / "per-question-metrics.jsonl"
    safe_metric_fields = (
        "question_id",
        "method",
        "repeat_index",
        "answerable",
        "exact_match",
        "char_f1",
        "citation_precision",
        "citation_recall",
        "refusal_correct",
        "unsupported_answer",
        "latency_ms",
        "input_tokens",
        "output_tokens",
    )
    per_question_path.write_text(
        "".join(
            json.dumps(
                {
                    field: row.get(field)
                    for field in safe_metric_fields
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in result["per_answer"]
        ),
        encoding="utf-8",
    )
    paired_path = run_dir / "paired-bootstrap.json"
    _atomic_write_json(
        paired_path,
        {
            "version": config["version"],
            "paired": True,
            "resample_unit": "question_id",
            "comparisons": bootstrap_results,
        },
    )
    automatic_path = run_dir / "automatic-metrics.json"
    automatic = {
        "version": config["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "run_count": len(answers),
        "by_method": result["by_method"],
        "definitions": {
            "unsupported_answer_rate": (
                "unanswerable and abstain=false"
            ),
            "hallucination": "human_review_only",
            "medical_correctness": "not_claimed_by_string_metrics",
        },
        "inputs": {
            "run_summary_sha256": _sha256_file(
                run_dir / "matrix-summary.json"
            ),
            "per_answer_sha256": _sha256_file(
                run_dir / "per-answer.jsonl"
            ),
            "dataset_sha256": _sha256_file(dataset_path),
            "config_sha256": _sha256_file(config_path),
            "answer_prereg_sha256": _sha256_file(
                answer_prereg_path
            ),
        },
        "files": {
            "per_question_metrics": {
                "path": per_question_path.name,
                "sha256": _sha256_file(per_question_path),
            },
            "paired_bootstrap": {
                "path": paired_path.name,
                "sha256": _sha256_file(paired_path),
            },
        },
    }
    _atomic_write_json(automatic_path, automatic)
    return {
        "status": "ready",
        "run_dir": run_dir.as_posix(),
        "automatic_metrics": automatic,
        "automatic_metrics_sha256": _sha256_file(automatic_path),
        "per_question_metrics_sha256": _sha256_file(
            per_question_path
        ),
        "paired_bootstrap_sha256": _sha256_file(paired_path),
    }
