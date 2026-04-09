"""Evaluation metrics."""

from __future__ import annotations

import math

from bidmate_rag.schema import GenerationResult, RetrievedChunk


def _match_expected(chunk: RetrievedChunk, expected_doc_ids: list[str]) -> bool:
    return (
        chunk.chunk.doc_id in expected_doc_ids
        or chunk.chunk.metadata.get("사업명") in expected_doc_ids
    )


def calc_hit_rate(
    chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5
) -> float | None:
    if not expected_doc_ids:
        return None
    return 1.0 if any(_match_expected(chunk, expected_doc_ids) for chunk in chunks[:k]) else 0.0


def calc_mrr(chunks: list[RetrievedChunk], expected_doc_ids: list[str]) -> float | None:
    if not expected_doc_ids:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if _match_expected(chunk, expected_doc_ids):
            return 1.0 / index
    return 0.0


def calc_ndcg(
    chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5
) -> float | None:
    if not expected_doc_ids:
        return None
    relevances = [2 if _match_expected(chunk, expected_doc_ids) else 0 for chunk in chunks[:k]]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def summarize_generation_results(results: list[GenerationResult]) -> dict[str, float]:
    if not results:
        return {"avg_latency_ms": 0.0, "total_cost_usd": 0.0}
    return {
        "avg_latency_ms": round(sum(result.latency_ms for result in results) / len(results), 3),
        "total_cost_usd": round(sum(result.cost_usd for result in results), 6),
    }
