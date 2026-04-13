"""Tests for evaluation/metrics.py — focused on identifier matching."""

from __future__ import annotations

from bidmate_rag.evaluation.metrics import calc_hit_rate, calc_map, calc_mrr, calc_ndcg
from bidmate_rag.schema import Chunk, RetrievedChunk


def _make_chunk(
    *,
    doc_id: str = "DOC-001",
    사업명: str = "샘플 사업",
    파일명: str = "기관명_샘플 사업.hwp",
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=0.9,
        chunk=Chunk(
            chunk_id=f"chunk-{rank}",
            doc_id=doc_id,
            text="본문",
            text_with_meta="본문",
            char_count=2,
            section="",
            content_type="text",
            chunk_index=0,
            metadata={"사업명": 사업명, "파일명": 파일명},
        ),
    )


def test_match_by_doc_id():
    chunks = [_make_chunk(doc_id="DOC-001")]
    assert calc_hit_rate(chunks, ["DOC-001"], k=5) == 1.0


def test_match_by_사업명():
    chunks = [_make_chunk(사업명="공공포털 구축")]
    assert calc_hit_rate(chunks, ["공공포털 구축"], k=5) == 1.0


def test_match_by_파일명():
    """Eval CSV ground_truth_docs는 파일명 형식이라 이 매칭이 가장 중요."""
    chunks = [_make_chunk(파일명="한국가스공사_차세대 ERP.hwp")]
    assert calc_hit_rate(chunks, ["한국가스공사_차세대 ERP.hwp"], k=5) == 1.0


def test_no_match_returns_zero():
    chunks = [_make_chunk(doc_id="DOC-001", 사업명="A", 파일명="A.hwp")]
    assert calc_hit_rate(chunks, ["완전히 다른 파일.hwp"], k=5) == 0.0


def test_empty_expected_returns_none():
    chunks = [_make_chunk()]
    assert calc_hit_rate(chunks, [], k=5) is None
    assert calc_mrr(chunks, []) is None
    assert calc_ndcg(chunks, [], k=5) is None
    assert calc_map(chunks, [], k=5) is None


def test_mrr_uses_파일명_at_rank_3():
    chunks = [
        _make_chunk(doc_id="A", 사업명="A", 파일명="a.hwp", rank=1),
        _make_chunk(doc_id="B", 사업명="B", 파일명="b.hwp", rank=2),
        _make_chunk(doc_id="C", 사업명="C", 파일명="target.hwp", rank=3),
    ]
    # rank 3에서 매칭 → MRR = 1/3
    assert calc_mrr(chunks, ["target.hwp"]) == 1 / 3


def test_ndcg_파일명_top_position():
    chunks = [
        _make_chunk(파일명="hit.hwp", rank=1),
        _make_chunk(파일명="miss1.hwp", rank=2),
        _make_chunk(파일명="miss2.hwp", rank=3),
    ]
    # 1위에서 hit → DCG = 2/log2(2) = 2.0, iDCG = 2.0 → nDCG = 1.0
    assert calc_ndcg(chunks, ["hit.hwp"], k=5) == 1.0


def test_map_single_doc():
    """정답 문서 1개 → MAP은 MRR과 동일한 값."""
    chunks = [
        _make_chunk(파일명="miss.hwp", rank=1),
        _make_chunk(파일명="hit.hwp", rank=2),
    ]
    # 2위에서 매칭 → precision=1/2 → AP=0.5/1=0.5
    assert calc_map(chunks, ["hit.hwp"], k=5) == 0.5


def test_map_multi_doc():
    """정답 문서 2개 → 두 문서를 모두 상위에서 찾았는지 평가."""
    chunks = [
        _make_chunk(doc_id="A", 파일명="hit1.hwp", rank=1),
        _make_chunk(doc_id="B", 파일명="miss.hwp", rank=2),
        _make_chunk(doc_id="C", 파일명="hit2.hwp", rank=3),
    ]
    # 1위: precision=1/1, 3위: precision=2/3 → AP=(1+2/3)/2=0.8333...
    result = calc_map(chunks, ["hit1.hwp", "hit2.hwp"], k=5)
    assert round(result, 4) == 0.8333


def test_map_duplicate_chunk_same_doc():
    """같은 문서의 청크가 여러 개 검색되어도 중복 카운트하지 않음."""
    chunks = [
        _make_chunk(doc_id="A", 파일명="hit.hwp", rank=1),
        _make_chunk(doc_id="A", 파일명="hit.hwp", rank=2),
    ]
    # 같은 doc_id → 1위에서만 카운트 → AP=1/1=1.0
    assert calc_map(chunks, ["hit.hwp"], k=5) == 1.0


def test_hit_rate_outside_topk():
    chunks = [
        _make_chunk(파일명="miss1.hwp", rank=1),
        _make_chunk(파일명="miss2.hwp", rank=2),
        _make_chunk(파일명="miss3.hwp", rank=3),
        _make_chunk(파일명="miss4.hwp", rank=4),
        _make_chunk(파일명="miss5.hwp", rank=5),
        _make_chunk(파일명="hit.hwp", rank=6),  # k=5 밖
    ]
    assert calc_hit_rate(chunks, ["hit.hwp"], k=5) == 0.0
