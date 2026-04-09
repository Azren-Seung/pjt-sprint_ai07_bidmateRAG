"""Runtime assembly helpers shared by CLI and UI."""

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
