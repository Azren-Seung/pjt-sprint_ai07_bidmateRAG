"""Retrieval helpers for the web_api adapter.

주요 기능: `@` 멘션이 2개 이상일 때 ChromaDB의 단일 `$in` 필터가 한 문서에만
유사도 점수를 몰아줄 수 있으므로, 문서별로 따로 검색한 뒤 점수 기준으로 병합한다.
"""

from __future__ import annotations

from typing import Protocol

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.web_api.pipeline_cache import get_pipeline


class _RetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        chat_history: list | None = None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]: ...


def split_and_merge_chunks(
    retriever: _RetrieverProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """문서별로 top_k//N + 2개씩 검색한 뒤 점수로 정렬·절단한다.

    이 함수는 retriever 객체만 의존하므로 테스트에서 FakeRetriever 주입 가능.
    """
    if not mentioned_doc_ids:
        raise ValueError("mentioned_doc_ids must be non-empty")
    per_doc_k = max(top_k // len(mentioned_doc_ids), 3) + 2
    all_chunks: list[RetrievedChunk] = []
    for doc_id in mentioned_doc_ids:
        chunks = retriever.retrieve(
            query,
            chat_history=None,
            top_k=per_doc_k,
            metadata_filter={"doc_id": doc_id},
        )
        all_chunks.extend(chunks)
    all_chunks.sort(key=lambda c: -c.score)
    return all_chunks[:top_k]


def per_doc_split_query(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> GenerationResult:
    """멘션 2개 이상일 때 실행되는 RAG 우회 경로.

    - retriever는 각 문서별로 따로 호출 (per-doc split)
    - LLM은 merged chunks를 받아 단일 generate() 호출
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    merged = split_and_merge_chunks(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )

    return llm.generate(
        question=question,
        context_chunks=merged,
        history=[],
        generation_config={
            "max_context_chars": max_context_chars,
            "scenario": runtime.provider.scenario or runtime.provider.provider,
            "run_id": f"web-{provider_config}",
            "embedding_provider": embedder.provider_name,
            "embedding_model": embedder.model_name,
        },
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )
