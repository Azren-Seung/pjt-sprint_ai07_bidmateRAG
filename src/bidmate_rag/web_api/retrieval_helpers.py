"""Retrieval helpers for the web_api adapter.

web_api는 `@` 멘션과 `/` 슬래시 커맨드로 사용자 의도를 명시적으로 수집한다.
`RAGRetriever.retrieve`를 사용하되, 멘션이 있으면 `metadata_filter`를 명시적으로
전달해서 자동 추출 경로를 건너뛴다. 멘션이 없으면 `metadata_filter={}`를 넘겨
retriever의 자동 필터 추출 없이 순수 벡터 검색을 수행한다.

멘션 2개+ 일 때는 문서별 loop 후 점수로 병합한다 (per-doc split).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.providers.llm.base import StreamDelta
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


def _retriever_search(
    retriever: _RetrieverProtocol,
    *,
    query: str,
    mentioned_doc_ids: list[str],
    top_k: int,
) -> list[RetrievedChunk]:
    """RAGRetriever.retrieve를 통한 검색. reranker·부스팅·폴백 자동 적용.

    - 멘션 0개: metadata_filter={} (자동 추출 없이 순수 벡터 검색)
    - 멘션 1개: metadata_filter={"파일명": doc_id}
    - 멘션 2개+: 문서별 loop + 점수 병합 (per-doc split)
    """
    if not mentioned_doc_ids:
        return retriever.retrieve(
            query,
            chat_history=None,
            top_k=top_k,
            metadata_filter={},
        )

    if len(mentioned_doc_ids) == 1:
        return retriever.retrieve(
            query,
            chat_history=None,
            top_k=top_k,
            metadata_filter={"doc_id": mentioned_doc_ids[0]},
        )

    # per-doc split
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


# Backward-compat alias
split_and_merge_chunks = _retriever_search


def web_query(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> GenerationResult:
    """Web API의 통합 RAG 경로.

    `RAGRetriever.retrieve`를 통해 검색하고 `llm.generate`로 응답 생성.
    reranker, section 부스팅, where_document 폴백 등 retriever의 모든
    개선사항이 자동으로 적용된다.
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = _retriever_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )

    return llm.generate(
        question=question,
        context_chunks=chunks,
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


def web_query_stream(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> Iterator[tuple[str, object]]:
    """Streaming 버전의 `web_query`.

    이벤트 스트림을 (event_type, payload) 튜플로 방출:
      1. ("retrieval", list[RetrievedChunk]) — 검색 완료 직후 1회
      2. ("token", str)                      — LLM delta마다
      3. ("done", GenerationResult)          — 스트림 종료 시 1회
    """
    pipeline, runtime, embedder, llm = get_pipeline(provider_config, chunking_config)
    retriever = pipeline.retriever

    chunks = _retriever_search(
        retriever,
        query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        top_k=top_k,
    )
    yield ("retrieval", chunks)

    gen_config = {
        "max_context_chars": max_context_chars,
        "scenario": runtime.provider.scenario or runtime.provider.provider,
        "run_id": f"web-{provider_config}",
        "embedding_provider": embedder.provider_name,
        "embedding_model": embedder.model_name,
    }
    for item in llm.generate_stream(
        question=question,
        context_chunks=chunks,
        history=[],
        generation_config=gen_config,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    ):
        if isinstance(item, StreamDelta):
            yield ("token", item.text)
        elif isinstance(item, GenerationResult):
            yield ("done", item)


# Backward-compat alias: kept for tests + existing imports
def per_doc_split_query(
    *,
    question: str,
    augmented_query: str,
    mentioned_doc_ids: list[str],
    provider_config: str,
    chunking_config: str | None,
    system_prompt: str | None,
    top_k: int,
    max_context_chars: int,
) -> GenerationResult:
    """멘션 2개 이상일 때 진입하던 기존 경로 — 이제 `web_query`로 위임."""
    return web_query(
        question=question,
        augmented_query=augmented_query,
        mentioned_doc_ids=mentioned_doc_ids,
        provider_config=provider_config,
        chunking_config=chunking_config,
        system_prompt=system_prompt,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )
