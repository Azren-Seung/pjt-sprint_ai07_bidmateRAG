"""Retriever orchestration."""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_metadata_filters,
    extract_range_filters,
    extract_section_hint,
    is_comparison_query,
    should_boost_tables,
)


class RAGRetriever:
    """메타데이터 필터와 벡터 검색을 결합하는 RAG 리트리버."""

    def __init__(self, vector_store, embedder, metadata_store=None) -> None:
        """RAGRetriever를 초기화

        Args:
            vector_store: 벡터 검색에 사용할 벡터 스토어.
            embedder: 쿼리 임베딩 생성기.
            metadata_store: 메타데이터 기반 문서 필터링 스토어.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store

    def _extract_scope_key(self, where: dict | None) -> tuple[str, list[str]] | None:
        if not where:
            return None
        for key in ("발주 기관", "파일명"):
            value = where.get(key)
            if not isinstance(value, dict):
                continue
            scoped_values = value.get("$in")
            if isinstance(scoped_values, list) and len(scoped_values) >= 2:
                return key, scoped_values
        return None

    def _should_run_scoped_queries(self, query: str, where: dict | None) -> bool:
        return is_comparison_query(query) and self._extract_scope_key(where) is not None

    def _build_scoped_filters(self, where: dict) -> list[dict]:
        scoped_target = self._extract_scope_key(where)
        if scoped_target is None:
            return [where]
        scope_key, scoped_values = scoped_target
        shared_filters = {key: value for key, value in where.items() if key != scope_key}
        return [{**shared_filters, scope_key: value} for value in scoped_values]

    def _merge_round_robin(self, grouped_results: list[list], top_k: int) -> list:
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
                    return self._assign_ranks(merged)
        return self._assign_ranks(merged)

    def _assign_ranks(self, results: list) -> list:
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results

    def _rerank_results(self, results: list, query: str, section_hint: str | None) -> list:
        if not results:
            return results

        table_boost = should_boost_tables(query)
        if not section_hint and not table_boost:
            return self._assign_ranks(results)

        def boosted_score(result) -> float:
            score = result.score
            if section_hint and section_hint in result.chunk.section:
                score += 0.1
            if table_boost and result.chunk.content_type == "table":
                score += 0.1
            return score

        ordered = sorted(
            enumerate(results),
            key=lambda item: (boosted_score(item[1]), item[1].score, -item[0]),
            reverse=True,
        )
        reranked = []
        for index, (_, result) in enumerate(ordered, start=1):
            result.rank = index
            reranked.append(result)
        return reranked

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
        # ``metadata_filter is None`` → 자동 추출 (legacy 기본 동작)
        # ``metadata_filter == {}``   → "필터 없음" 명시 (자동 추출도 비활성화)
        # ``metadata_filter == {...}`` → explicit override
        if metadata_filter is not None:
            where = dict(metadata_filter) if metadata_filter else None
        else:
            agency_list = getattr(self.metadata_store, "agency_list", [])
            where = extract_metadata_filters(query, agency_list, chat_history=chat_history)
            range_filter = extract_range_filters(query)
            if range_filter:
                where = {**(where or {}), **range_filter}
            if where is None and self.metadata_store is not None:
                relevant_docs = self.metadata_store.find_relevant_docs(query, top_n=3)
                if relevant_docs:
                    where = {"파일명": {"$in": relevant_docs}}
        section_hint = extract_section_hint(query)
        where_document = {"$contains": section_hint} if section_hint else None
        query_embedding = self.embedder.embed_query(query)
        if self._should_run_scoped_queries(query, where):
            grouped_results = [
                self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    where=scoped_where,
                    where_document=where_document,
                )
                for scoped_where in self._build_scoped_filters(where)
            ]
            results = self._merge_round_robin(grouped_results, top_k)
        else:
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                where=where,
                where_document=where_document,
            )
        return self._rerank_results(results, query=query, section_hint=section_hint)
