"""Retriever orchestration."""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_metadata_filters,
    extract_range_filters,
    extract_section_hint,
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

    def retrieve(self, query: str, chat_history=None, top_k: int = 5):
        """쿼리에 대해 메타데이터 필터링 후 벡터 검색을 수행

        Args:
            query: 사용자 질의 문자열.
            chat_history: 이전 대화 이력.
            top_k: 반환할 최대 결과 수.

        Returns:
            RetrievedChunk 리스트.
        """
        agency_list = getattr(self.metadata_store, "agency_list", [])
        where = extract_metadata_filters(query, agency_list, chat_history=chat_history)
        range_filter = extract_range_filters(query)
        if range_filter:
            where = {**(where or {}), **range_filter}
        section_hint = extract_section_hint(query)
        where_document = {"$contains": section_hint} if section_hint else None
        if where is None and self.metadata_store is not None:
            relevant_docs = self.metadata_store.find_relevant_docs(query, top_n=3)
            if relevant_docs:
                where = {"파일명": {"$in": relevant_docs}}
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
            where_document=where_document,
        )
