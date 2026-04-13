"""Evaluation metrics."""

from __future__ import annotations

import math

from bidmate_rag.schema import GenerationResult, RetrievedChunk


def _match_expected(chunk: RetrievedChunk, expected_doc_ids: list[str]) -> bool:
    """검색된 청크가 정답 문서에 해당하는지 확인

    Args:
        chunk: 검색된 청크.
        expected_doc_ids: 정답 문서 식별자 리스트 (doc_id, 사업명, 파일명 중 하나).

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
    """상위 k개 결과 중 정답 문서가 하나라도 포함되었는지 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 확인할지.

    Returns:
        정답 포함 시 1.0, 미포함 시 0.0, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 상위 k개 중 정답이 하나라도 있으면 1.0
    return 1.0 if any(_match_expected(chunk, expected_doc_ids) for chunk in chunks[:k]) else 0.0


def calc_mrr(chunks: list[RetrievedChunk], expected_doc_ids: list[str]) -> float | None:
    """정답 문서가 처음 등장하는 순위의 역수(Mean Reciprocal Rank)를 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.

    Returns:
        1/순위 값 (1위=1.0, 2위=0.5, ...), 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 첫 번째 정답이 나타나는 순위의 역수를 반환
    for index, chunk in enumerate(chunks, start=1):
        if _match_expected(chunk, expected_doc_ids):
            return 1.0 / index
    return 0.0


def calc_ndcg(
    chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5
) -> float | None:
    """상위 k개 결과의 순위 품질을 nDCG(normalized Discounted Cumulative Gain)로 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 평가할지.

    Returns:
        0~1 사이 nDCG 점수, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 각 청크의 관련도: 정답이면 2, 아니면 0
    relevances = [2 if _match_expected(chunk, expected_doc_ids) else 0 for chunk in chunks[:k]]
    # DCG: 순위가 낮을수록 log로 할인
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))
    # iDCG: 이상적인 순서(정답이 모두 상위에 위치)로 계산
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def calc_map(chunks: list[RetrievedChunk], expected_doc_ids: list[str], k: int = 5) -> float | None:
    """상위 k개 결과의 Mean Average Precision을 계산

    Args:
        chunks: 검색된 청크 리스트.
        expected_doc_ids: 정답 문서 식별자 리스트.
        k: 상위 몇 개까지 평가할지.

    Returns:
        0~1 사이 MAP 점수, 정답 없으면 None.
    """
    if not expected_doc_ids:
        return None
    # 정답 수 (중복 문서 제거를 위해 set 사용하지 않음 — 평가셋 기준 그대로)
    num_relevant = len(expected_doc_ids)
    hits = 0
    precision_sum = 0.0
    # 이미 매칭된 doc을 추적하여 같은 문서의 청크 중복 카운트 방지
    seen_docs: set[str] = set()
    for index, chunk in enumerate(chunks[:k], start=1):
        if _match_expected(chunk, expected_doc_ids):
            # 같은 문서의 다른 청크가 이미 매칭되었으면 스킵
            doc_key = chunk.chunk.doc_id
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
            hits += 1
            # 해당 순위에서의 precision을 누적
            precision_sum += hits / index
    return precision_sum / num_relevant if num_relevant else 0.0


def summarize_generation_results(results: list[GenerationResult]) -> dict[str, float]:
    """생성 결과들의 평균 지연시간과 총 비용을 요약

    Args:
        results: GenerationResult 리스트.

    Returns:
        avg_latency_ms와 total_cost_usd를 포함하는 딕셔너리.
    """
    if not results:
        return {"avg_latency_ms": 0.0, "total_cost_usd": 0.0}
    return {
        # 전체 결과의 평균 응답 시간 (밀리초)
        "avg_latency_ms": round(sum(result.latency_ms for result in results) / len(results), 3),
        # 전체 결과의 누적 API 비용 (USD)
        "total_cost_usd": round(sum(result.cost_usd for result in results), 6),
    }
