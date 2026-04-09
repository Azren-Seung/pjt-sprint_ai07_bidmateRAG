"""런타임 조립 헬퍼.

CLI 스크립트와 Streamlit UI가 공유하는 파이프라인 조립 로직.
설정 파일 → 프로바이더/리트리버/LLM 생성 → RAGChatPipeline 반환.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.config.settings import RuntimeConfig, load_runtime_config
from bidmate_rag.pipelines.chat import RAGChatPipeline
from bidmate_rag.providers.llm.registry import build_embedding_provider, build_llm_provider
from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.retrieval.vector_store import ChromaVectorStore
from bidmate_rag.storage.metadata_store import MetadataStore


def collection_name_for_config(runtime: RuntimeConfig) -> str:
    """RuntimeConfig에서 ChromaDB 컬렉션 이름을 생성한다.

    Args:
        runtime: 런타임 설정.

    Returns:
        'bidmate-{provider}-{model}' 형식의 컬렉션 이름.
    """
    if runtime.provider.collection_name:
        return runtime.provider.collection_name
    model = (
        (runtime.provider.embedding_model or runtime.provider.model)
        .replace("/", "-")
        .replace(" ", "-")
    )
    return f"bidmate-{runtime.provider.provider}-{model}".lower()


def build_runtime_pipeline(
    base_config_path: str | Path,
    provider_config_path: str | Path,
    experiment_config_path: str | Path | None = None,
    persist_dir: str | Path = "artifacts/chroma_db",
    metadata_path: str | Path = "data/processed/cleaned_documents.parquet",
):
    """설정 파일들로부터 RAGChatPipeline을 조립한다.

    Args:
        base_config_path: 기본 설정 YAML 경로.
        provider_config_path: 프로바이더 설정 YAML 경로.
        experiment_config_path: 실험 설정 YAML 경로 (선택).
        persist_dir: ChromaDB 저장 디렉터리.
        metadata_path: 정제된 문서 메타데이터 parquet 경로.

    Returns:
        (pipeline, runtime, embedder, llm) 튜플.
    """
    runtime = load_runtime_config(base_config_path, provider_config_path, experiment_config_path)
    embedder = build_embedding_provider(runtime.provider)
    llm = build_llm_provider(runtime.provider)
    vector_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name_for_config(runtime),
    )
    metadata_file = Path(metadata_path)
    metadata_store = (
        MetadataStore.from_parquet(metadata_file)
        if metadata_file.exists()
        else MetadataStore(pd.DataFrame())
    )
    retriever = RAGRetriever(
        vector_store=vector_store, embedder=embedder, metadata_store=metadata_store
    )
    pipeline = RAGChatPipeline(retriever=retriever, llm=llm)
    return pipeline, runtime, embedder, llm
