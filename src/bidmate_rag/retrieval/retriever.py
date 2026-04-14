"""Retriever orchestration.

검색 흐름:
  1. 질문에서 메타데이터 필터 자동 추출 (기관명, 도메인, 금액 범위 등)
  2. 임베딩 벡터 검색으로 후보 청크 조회 (Cross-Encoder 있으면 4배 많이)
  3. 비교 질문이면 기관별 fan-out 검색 → round-robin 병합
  4. Cross-Encoder 리랭킹으로 질문-청크 관련성 재정렬
  5. 섹션/테이블 부스팅으로 RFP 특화 미세 조정
"""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_matched_agencies,
    extract_metadata_filters,
    extract_range_filters,
    extract_section_hint,
    is_comparison_query,
)
from bidmate_rag.retrieval.multiturn import (
    extract_recent_agency_filter,
    rewrite_query_with_history,
)
from bidmate_rag.retrieval.reranker import (
    _assign_ranks,
    cross_encoder_rerank,
    rerank_with_boost,
)


class RAGRetriever:
    """메타데이터 필터와 벡터 검색을 결합하는 RAG 리트리버."""

    def __init__(
        self,
        vector_store,
        embedder,
        metadata_store=None,
        reranker_model=None,
        enable_multiturn: bool = True,
    ) -> None:
        """RAGRetriever를 초기화

        Args:
            vector_store: 벡터 검색에 사용할 벡터 스토어.
            embedder: 쿼리 임베딩 생성기.
            metadata_store: 메타데이터 기반 문서 필터링 스토어.
            reranker_model: Cross-Encoder 리랭킹 모델. None이면 Cross-Encoder 없이 부스팅만 적용.
            enable_multiturn: 이전 대화를 이용한 검색문 보강 사용 여부.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store
        self.reranker = reranker_model
        self.enable_multiturn = enable_multiturn

    # ── fan-out 관련 메서드 ──
    # 비교 질문("A와 B의 예산 차이는?")에서 기관별로 따로 검색하는 로직.
    # $in 필터로 한번에 검색하면 한쪽 기관 문서만 상위에 올 수 있어서,
    # 기관마다 따로 검색 후 round-robin으로 골고루 병합한다.

    def _extract_scope_key(self, where: dict | None) -> tuple[str, list[str]] | None:
        """where 필터에서 $in 다중 기관 목록을 추출한다.
        예: {"발주 기관": {"$in": ["가스공사", "고려대"]}} → ("발주 기관", ["가스공사", "고려대"])
        """
        if not where:
            return None
        value = where.get("발주 기관")
        if isinstance(value, dict):
            scoped_values = value.get("$in")
            if isinstance(scoped_values, list) and len(scoped_values) >= 2:
                return "발주 기관", scoped_values
        return None

    def _should_run_scoped_queries(self, query: str, where: dict | None) -> bool:
        """비교 질문 + 다중 기관 필터일 때만 fan-out 실행."""
        return is_comparison_query(query) and self._extract_scope_key(where) is not None

    def _build_scoped_filters(self, where: dict) -> list[dict]:
        """$in 필터를 기관별 개별 필터로 분리한다.
        예: {"발주 기관": {"$in": ["A", "B"]}} → [{"발주 기관": "A"}, {"발주 기관": "B"}]
        """
        scoped_target = self._extract_scope_key(where)
        if scoped_target is None:
            return [where]
        scope_key, scoped_values = scoped_target
        shared_filters = {key: value for key, value in where.items() if key != scope_key}
        return [{**shared_filters, scope_key: value} for value in scoped_values]

    def _merge_round_robin(self, grouped_results: list[list], top_k: int) -> list:
        """기관별 검색 결과를 번갈아 가며 병합한다.
        A1, B1, A2, B2, ... 순서로 양쪽 문서가 골고루 포함되도록 한다.
        """
        merged: list = []
        seen_chunk_ids: set[str] = set()
        max_group_len = max((len(results) for results in grouped_results), default=0)
        for index in range(max_group_len):
            for results in grouped_results:
                if index >= len(results):
                    continue
                item = results[index]
                chunk_id = item.chunk.chunk_id
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                merged.append(item)
                if len(merged) >= top_k:
                    return _assign_ranks(merged)
        return _assign_ranks(merged)

    # ── 메인 검색 메서드 ──

    def retrieve(
        self,
        query: str,
        chat_history=None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        """쿼리에 대해 메타데이터 필터링 후 벡터 검색을 수행

        Args:
            query: 사용자 질의 문자열.
            chat_history: 이전 대화 이력.
            top_k: 반환할 최대 결과 수.
            metadata_filter: ChromaDB ``where`` 절로 직접 사용할 필터.
                평가셋의 ``metadata_filter`` 또는 Streamlit UI의 수동 필터를
                위한 explicit override. 지정 시 query 기반 자동 추출은 무시되고
                이 값이 그대로 사용됩니다.

        Returns:
            RetrievedChunk 리스트.
        """
        agency_list = getattr(self.metadata_store, "agency_list", [])
        resolved_query = (
            rewrite_query_with_history(query, chat_history, agency_list)
            if self.enable_multiturn
            else query
        )

        # ── 1단계: 메타데이터 필터 결정 ──
        # metadata_filter is None  → 질문에서 자동 추출
        # metadata_filter == {}    → 필터 없이 전체 검색
        # metadata_filter == {...} → 지정된 필터 그대로 사용
        if metadata_filter is not None:
            where = dict(metadata_filter) if metadata_filter else None
        else:
            # 자동 추출: 기관명 → 도메인 → 기관유형 순으로 시도
            where = extract_metadata_filters(resolved_query, agency_list, chat_history=chat_history)
            # 기관이 2개 이상 + 비교 질문이면 $in 필터 생성
            matched_agencies = extract_matched_agencies(resolved_query, agency_list)
            if where is None and len(matched_agencies) >= 2 and is_comparison_query(resolved_query):
                where = {"발주 기관": {"$in": matched_agencies}}
            if self.enable_multiturn and (where is None or "발주 기관" not in where):
                history_agency_filter = extract_recent_agency_filter(chat_history, agency_list)
                if history_agency_filter:
                    where = {**history_agency_filter, **(where or {})}
            # 금액/연도 범위 필터 추가
            range_filter = extract_range_filters(resolved_query)
            if range_filter:
                where = {**(where or {}), **range_filter}
            # 위 필터가 모두 실패하면 MetadataStore에서 관련 문서 추정
            if where is None and self.metadata_store is not None:
                relevant_docs = self.metadata_store.find_relevant_docs(resolved_query, top_n=3)
                if relevant_docs:
                    where = {"파일명": {"$in": relevant_docs}}

        # ── 2단계: 벡터 검색 ──
        # Cross-Encoder가 있으면 후보를 4배 넓게 가져와서 정밀 재정렬한다
        final_top_k = top_k
        rerank_pool_k = final_top_k * 4 if self.reranker else final_top_k
        section_hint = extract_section_hint(resolved_query)
        where_document = {"$contains": section_hint} if section_hint else None
        query_embedding = self.embedder.embed_query(resolved_query)

        # 비교 질문이면 기관별 fan-out 검색 → round-robin 병합
        if self._should_run_scoped_queries(resolved_query, where):
            grouped_results = [
                self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=rerank_pool_k,
                    where=scoped_where,
                    where_document=where_document,
                )
                for scoped_where in self._build_scoped_filters(where)
            ]
            results = self._merge_round_robin(grouped_results, rerank_pool_k)
        else:
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=rerank_pool_k,
                where=where,
                where_document=where_document,
            )

        # ── 3단계: Cross-Encoder 리랭킹 ──
        # 넓게 가져온 후보 중 질문과 실제로 관련 있는 top_k개만 선별
        results = cross_encoder_rerank(self.reranker, resolved_query, results, final_top_k)

        # ── 4단계: 섹션/테이블 부스팅 (RFP 특화 미세 조정) ──
        return rerank_with_boost(results, query=resolved_query, section_hint=section_hint)
