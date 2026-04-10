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

    격리 규칙:
      - ``experiment.mode == "full_rag"`` (default): chunking을 바꿔가며 실험할
        가능성이 있으므로 ``실험명-…`` prefix로 격리.
      - ``experiment.mode == "generation_only"``: 동일 인덱스에 다른 LLM만
        붙이는 실험이므로 collection을 **공유** (재빌드 비용 절약).
      - 실험 config가 없는 경우 (``ad-hoc``): legacy 동작 보존.

    ``provider.collection_name``이 명시된 경우:
      - 격리가 필요 없는 모드(generation_only / ad-hoc)에서는 그것을 그대로 사용
      - 격리가 필요한 모드(full_rag)에서는 ``{실험명}-{명시이름}`` 형식으로 prefix
    """
    model = (
        (runtime.provider.embedding_model or runtime.provider.model)
        .replace("/", "-")
        .replace(" ", "-")
    )
    exp_name = (runtime.experiment.name or "ad-hoc").replace("/", "-").replace(" ", "-")
    mode = runtime.experiment.mode or "full_rag"
    is_shared = mode == "generation_only" or exp_name in ("ad-hoc", "default", "")

    explicit = runtime.provider.collection_name
    if explicit:
        if is_shared:
            return explicit
        return f"{exp_name}-{explicit}".lower()

    if is_shared:
        return f"bidmate-{runtime.provider.provider}-{model}".lower()
    return f"bidmate-{exp_name}-{runtime.provider.provider}-{model}".lower()


def build_runtime_pipeline(
    base_config_path: str | Path,
    provider_config_path: str | Path,
    experiment_config_path: str | Path | None = None,
    persist_dir: str | Path | None = None, # None으로 변경 - provider yaml의 persist_dir이 우선 적용되도록
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
    # persist_dir 우선순위: 함수 인자 → provider yaml → 기본값
    resolved_persist_dir = persist_dir or runtime.provider.persist_dir
    embedder = build_embedding_provider(runtime.provider)
    llm = build_llm_provider(runtime.provider)
    vector_store = ChromaVectorStore(
        persist_dir=resolved_persist_dir,
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
