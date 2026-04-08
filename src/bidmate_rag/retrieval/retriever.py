"""Retriever orchestration."""

from __future__ import annotations

from bidmate_rag.retrieval.filters import (
    extract_metadata_filters,
    extract_range_filters,
    extract_section_hint,
)


class RAGRetriever:
    def __init__(self, vector_store, embedder, metadata_store=None) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store

    def retrieve(self, query: str, chat_history=None, top_k: int = 5):
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
