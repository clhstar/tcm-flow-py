import math
import statistics
from pathlib import Path

from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit


LATENCY_FIELDS = (
    "bm25_ms",
    "dense_ms",
    "rrf_ms",
    "reranker_ms",
    "total_ms",
    "returned_context_chars",
)


def hit_clause_ids(hit: RetrievalHit) -> set[str]:
    return set(hit.clause_ids)


def _score(hit: RetrievalHit) -> float | None:
    for value in (
        hit.reranker_score,
        hit.rrf_score,
        hit.dense_score,
        hit.bm25_score,
    ):
        if value is not None:
            return float(value)
    return None


def _recall_at(
    question: PilotQuestion,
    hits: list[RetrievalHit],
    top_k: int,
) -> float:
    gold = set(question.gold_clause_ids)
    retrieved = set().union(
        *(hit_clause_ids(hit) for hit in hits[:top_k])
    ) if hits[:top_k] else set()
    return len(gold & retrieved) / len(gold)


def _reciprocal_rank(
    question: PilotQuestion,
    hits: list[RetrievalHit],
    top_k: int,
) -> float:
    gold = set(question.gold_clause_ids)
    for rank, hit in enumerate(hits[:top_k], start=1):
        if gold & hit_clause_ids(hit):
            return 1.0 / rank
    return 0.0


def _hit_relevance_keys(
    hit: RetrievalHit,
    question: PilotQuestion,
) -> set[str]:
    keys = (
        set(hit.clause_ids)
        | set(hit.source_evidence_ids)
        | ({hit.retrieval_parent_id} if hit.retrieval_parent_id else set())
    )
    return keys & set(question.graded_relevance)


def _ndcg(
    question: PilotQuestion,
    hits: list[RetrievalHit],
    top_k: int,
) -> float:
    seen_relevance_keys = set()
    dcg = 0.0
    for rank, hit in enumerate(hits[:top_k], start=1):
        matching = (
            _hit_relevance_keys(hit, question) - seen_relevance_keys
        )
        relevance = max(
            (
                question.graded_relevance[key]
                for key in matching
            ),
            default=0,
        )
        seen_relevance_keys.update(matching)
        dcg += (2**relevance - 1) / math.log2(rank + 1)

    ideal_relevances = sorted(
        question.graded_relevance.values(),
        reverse=True,
    )[:top_k]
    ideal_dcg = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _parent_recovery_rate(
    rankings: dict[str, list[RetrievalHit]],
) -> float | None:
    c4_hits = [
        hit
        for hits in rankings.values()
        for hit in hits
        if hit.strategy == "c4" and hit.retrieval_parent_id
    ]
    if not c4_hits:
        return None
    recovered = sum(
        hit.retrieval_parent_id in hit_clause_ids(hit)
        and bool(hit.context_text.strip())
        for hit in c4_hits
    )
    return recovered / len(c4_hits)


def _evaluate_core(
    questions: list[PilotQuestion],
    rankings: dict[str, list[RetrievalHit]],
    *,
    top_ks: tuple[int, ...],
) -> dict:
    approved = [
        question
        for question in questions
        if question.review_status == "approved"
    ]
    answerable = [
        question for question in approved if question.answerable
    ]
    result = {
        "question_count": len(approved),
        "answerable_question_count": len(answerable),
        "unanswerable_question_count": len(approved) - len(answerable),
    }
    for top_k in top_ks:
        values = [
            _recall_at(
                question,
                rankings.get(question.question_id, []),
                top_k,
            )
            for question in answerable
        ]
        result[f"recall_at_{top_k}"] = (
            statistics.fmean(values) if values else 0.0
        )

    hit_values = [
        float(
            bool(
                set(question.gold_clause_ids)
                & set().union(
                    *(
                        hit_clause_ids(hit)
                        for hit in rankings.get(question.question_id, [])[:5]
                    )
                )
            )
        )
        for question in answerable
    ]
    result["hit_at_5"] = (
        statistics.fmean(hit_values) if hit_values else 0.0
    )
    reciprocal_ranks = [
        _reciprocal_rank(
            question,
            rankings.get(question.question_id, []),
            10,
        )
        for question in answerable
    ]
    result["mrr_at_10"] = (
        statistics.fmean(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )
    ndcg_values = [
        _ndcg(
            question,
            rankings.get(question.question_id, []),
            10,
        )
        for question in answerable
    ]
    result["ndcg_at_10"] = (
        statistics.fmean(ndcg_values) if ndcg_values else 0.0
    )
    return result


def evaluate_rankings(
    questions: list[PilotQuestion],
    rankings: dict[str, list[RetrievalHit]],
    *,
    top_ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    result = _evaluate_core(
        questions,
        rankings,
        top_ks=top_ks,
    )
    result["c4_parent_recovery_rate"] = _parent_recovery_rate(rankings)

    approved = [
        question
        for question in questions
        if question.review_status == "approved"
    ]
    unanswerable = [
        question for question in approved if not question.answerable
    ]
    top1_scores = []
    top5_scores = []
    for question in unanswerable:
        scores = [
            score
            for score in (
                _score(hit)
                for hit in rankings.get(question.question_id, [])[:5]
            )
            if score is not None
        ]
        if scores:
            top1_scores.append(scores[0])
            top5_scores.extend(scores)
    result["no_answer_scores"] = {
        "top1": top1_scores,
        "top5": top5_scores,
    }

    result["by_question_type"] = {
        question_type: _evaluate_core(
            [
                question
                for question in approved
                if question.question_type == question_type
            ],
            rankings,
            top_ks=top_ks,
        )
        for question_type in sorted(
            {question.question_type for question in approved}
        )
    }
    result["by_book"] = {
        book_scope: _evaluate_core(
            [
                question
                for question in approved
                if question.book_scope == book_scope
            ],
            rankings,
            top_ks=top_ks,
        )
        for book_scope in sorted(
            {question.book_scope for question in approved}
        )
    }
    return result


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_latency(records: list[dict]) -> dict:
    summary = {}
    for field in LATENCY_FIELDS:
        values = [float(record[field]) for record in records]
        if not values:
            summary[field] = {
                "count": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }
            continue
        summary[field] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": _nearest_rank_percentile(values, 0.95),
            "max": max(values),
        }
    return summary


def index_size_bytes(path: Path) -> int:
    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )
