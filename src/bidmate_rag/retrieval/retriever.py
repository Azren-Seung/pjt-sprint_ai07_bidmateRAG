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
        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
            where_document=where_document,
        )
