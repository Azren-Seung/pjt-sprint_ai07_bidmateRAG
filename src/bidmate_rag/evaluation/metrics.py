"""Evaluation metrics."""

from __future__ import annotations

import math

from bidmate_rag.schema import GenerationResult, RetrievedChunk


def _match_expected(chunk: RetrievedChunk, expected_doc_ids: list[str]) -> bool:
    """Check whether a retrieved chunk matches any expected document identifier.

    Accepts three identifier types so that callers can pass `doc_id` (공고 번호),
    `사업명`, or `파일명` interchangeably:
    - `doc_id`: ChromaDB-stored doc id (typically `공고 번호`)
    - `사업명`: business/project name from metadata
    - `파일명`: original filename (e.g. `기관명_사업명.hwp`)

    The eval CSVs (`data/eval/eval_v*/eval_batch_*.csv`) populate `ground_truth_docs`
    with `파일명` strings, so this function must compare against `파일명` to
    make Hit Rate / MRR / nDCG metrics meaningful.
    부분 매칭 추가: 사업명이 잘려있어도 파일명에 포함되면 매칭 성공
    """
    사업명 = chunk.chunk.metadata.get("사업명") or ""
    파일명 = chunk.chunk.metadata.get("파일명") or ""
    doc_id = chunk.chunk.doc_id or ""

    for exp in expected_doc_ids:
        if not exp:
            continue
        if (
            doc_id == exp
            or 사업명 == exp
            or 파일명 == exp
            or (사업명 and 사업명 in exp)  # 사업명이 기대값에 포함
            or (exp in 파일명)             # 기대값이 파일명에 포함
        ):
            return True
    return False


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
