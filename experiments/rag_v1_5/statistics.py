import random
import statistics
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _indexed_rows(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("per-question row 缺少 question_id")
        if question_id in indexed:
            raise ValueError(f"重复 question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def paired_stratified_bootstrap(
    *,
    per_question_a: list[dict],
    per_question_b: list[dict],
    metric_fields: tuple[str, ...],
    strata_fields: tuple[str, ...],
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict:
    if resamples <= 0:
        raise ValueError("resamples 必须大于 0")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level 必须位于 0 和 1 之间")
    if not metric_fields or not strata_fields:
        raise ValueError("metric_fields 和 strata_fields 不能为空")

    rows_a = _indexed_rows(per_question_a)
    rows_b = _indexed_rows(per_question_b)
    if set(rows_a) != set(rows_b):
        raise ValueError("配对 Bootstrap 的 question_id 集合不一致")

    strata: dict[tuple, list[str]] = defaultdict(list)
    for question_id in sorted(rows_a):
        row_a = rows_a[question_id]
        row_b = rows_b[question_id]
        stratum_a = tuple(row_a.get(field) for field in strata_fields)
        stratum_b = tuple(row_b.get(field) for field in strata_fields)
        if stratum_a != stratum_b or any(
            value is None for value in stratum_a
        ):
            raise ValueError(
                f"{question_id} 的分层字段不一致或缺失"
            )
        for field in metric_fields:
            if row_a.get(field) is None or row_b.get(field) is None:
                raise ValueError(
                    f"{question_id} 缺少配对指标 {field}"
                )
        strata[stratum_a].append(question_id)

    rng = random.Random(seed)
    samples = {
        metric: np.empty(resamples, dtype=np.float64)
        for metric in metric_fields
    }
    for sample_index in range(resamples):
        sampled_ids = []
        for stratum in sorted(strata, key=repr):
            members = strata[stratum]
            sampled_ids.extend(
                members[rng.randrange(len(members))]
                for _ in members
            )
        for metric in metric_fields:
            samples[metric][sample_index] = statistics.fmean(
                float(rows_a[question_id][metric])
                - float(rows_b[question_id][metric])
                for question_id in sampled_ids
            )

    alpha = 1.0 - confidence_level
    metrics = {}
    for metric in metric_fields:
        observed = statistics.fmean(
            float(rows_a[question_id][metric])
            - float(rows_b[question_id][metric])
            for question_id in sorted(rows_a)
        )
        lower, upper = np.quantile(
            samples[metric],
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        metrics[metric] = {
            "delta": float(observed),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "inconclusive_at_95pct": bool(
                float(lower) <= 0.0 <= float(upper)
            ),
        }

    return {
        "paired": True,
        "seed": seed,
        "resamples": resamples,
        "confidence_level": confidence_level,
        "strata_fields": list(strata_fields),
        "question_count": len(rows_a),
        "stratum_count": len(strata),
        "metrics": metrics,
    }


def _stratified_mean_bootstrap(
    *,
    rows: list[dict],
    metric_fields: tuple[str, ...],
    strata_fields: tuple[str, ...],
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict:
    indexed = _indexed_rows(rows)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for question_id, row in indexed.items():
        stratum = tuple(row.get(field) for field in strata_fields)
        if any(value is None for value in stratum):
            raise ValueError(f"{question_id} 缺少分层字段")
        strata[stratum].append(question_id)
    rng = random.Random(seed)
    samples = {
        metric: np.empty(resamples, dtype=np.float64)
        for metric in metric_fields
    }
    for sample_index in range(resamples):
        sampled_ids = []
        for stratum in sorted(strata, key=repr):
            members = strata[stratum]
            sampled_ids.extend(
                members[rng.randrange(len(members))]
                for _ in members
            )
        for metric in metric_fields:
            samples[metric][sample_index] = statistics.fmean(
                float(indexed[question_id][metric])
                for question_id in sampled_ids
            )
    alpha = 1.0 - confidence_level
    return {
        metric: {
            "value": statistics.fmean(
                float(row[metric]) for row in rows
            ),
            "ci_lower": float(
                np.quantile(samples[metric], alpha / 2.0)
            ),
            "ci_upper": float(
                np.quantile(
                    samples[metric],
                    1.0 - alpha / 2.0,
                )
            ),
        }
        for metric in metric_fields
    }


def summarize_formal_test(
    *,
    run_dir: Path,
    prereg_manifest_path: Path,
    output_path: Path | None = None,
) -> dict:
    matrix_config = json.loads(
        (run_dir / "matrix-config.json").read_text(encoding="utf-8")
    )
    matrix_summary = json.loads(
        (run_dir / "matrix-summary.json").read_text(encoding="utf-8")
    )
    prereg = json.loads(
        prereg_manifest_path.read_text(encoding="utf-8")
    )
    if (
        matrix_config.get("status") != "completed"
        or matrix_config.get("split") != "formal_test"
        or matrix_summary.get("status") != "completed"
        or matrix_summary.get("config_count") != 14
        or prereg.get("status") != "ready"
    ):
        raise ValueError("Formal test 矩阵或预注册未就绪")

    statistics_config = prereg["statistics"]
    metric_fields = tuple(statistics_config["primary_metrics"])
    strata_fields = tuple(statistics_config["strata"])
    resamples = int(statistics_config["bootstrap_resamples"])
    seed = int(statistics_config["bootstrap_seed"])
    confidence_level = float(
        statistics_config["confidence_level"]
    )

    records_by_config = {}
    absolute = {}
    details = {}
    for row in matrix_config["matrix"]:
        config_id = row["config_id"]
        records = [
            json.loads(line)
            for line in (
                run_dir / config_id / "per-question.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        answerable = [
            record
            for record in records
            if record.get("answerable")
            and all(
                record.get(metric) is not None
                for metric in metric_fields
            )
        ]
        records_by_config[config_id] = answerable
        absolute[config_id] = _stratified_mean_bootstrap(
            rows=answerable,
            metric_fields=metric_fields,
            strata_fields=strata_fields,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        metrics = json.loads(
            (run_dir / config_id / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        summary = next(
            item
            for item in matrix_summary["configs"]
            if item["config_id"] == config_id
        )
        details[config_id] = {
            "by_book": metrics.get("by_book", {}),
            "by_question_type": metrics.get(
                "by_question_type",
                {},
            ),
            "no_answer_score_distribution": metrics.get(
                "no_answer_score_distribution",
                {},
            ),
            "latency": summary.get("latency", {}),
            "returned_context_chars": summary.get(
                "latency",
                {},
            ).get("returned_context_chars", {}),
            "index_size_bytes": summary.get(
                "index_size_bytes"
            ),
        }

    comparison_specs = [
        {
            "comparison_id": "primary",
            **prereg["comparisons"]["primary"],
        },
        *[
            {
                "comparison_id": f"ablation-{index + 1}",
                **comparison,
            }
            for index, comparison in enumerate(
                prereg["comparisons"]["ablations"]
            )
        ],
    ]
    paired = []
    for comparison in comparison_specs:
        result = paired_stratified_bootstrap(
            per_question_a=records_by_config[comparison["a"]],
            per_question_b=records_by_config[comparison["b"]],
            metric_fields=metric_fields,
            strata_fields=strata_fields,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        paired.append({**comparison, **result})

    summary = {
        "version": "v1.5.0",
        "status": "ready",
        "split": "formal_test",
        "bootstrap": {
            "paired": True,
            "seed": seed,
            "resamples": resamples,
            "confidence_level": confidence_level,
            "strata": list(strata_fields),
            "primary_metrics": list(metric_fields),
        },
        "absolute": absolute,
        "paired_comparisons": paired,
        "details": details,
    }
    destination = output_path or run_dir / "formal-statistics.json"
    destination.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
