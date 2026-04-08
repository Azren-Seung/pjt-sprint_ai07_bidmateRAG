"""Chroma vector store integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from bidmate_rag.schema import Chunk, RetrievedChunk


def _primitive_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class ChromaVectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str) -> None:
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 5000) -> None:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=batch_embs,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    _primitive_metadata(
                        chunk.metadata
                        | {
                            "doc_id": chunk.doc_id,
                            "section": chunk.section,
                            "content_type": chunk.content_type,
                            "chunk_index": chunk.chunk_index,
                        }
                    )
                    for chunk in batch_chunks
                ],
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[RetrievedChunk]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        results = self.collection.query(**kwargs)
        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][index]
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", metadata.get("공고 번호", ""))),
                text=results["documents"][0][index],
                text_with_meta=results["documents"][0][index],
                char_count=len(results["documents"][0][index]),
                section=str(metadata.get("section", "")),
                content_type=str(metadata.get("content_type", "text")),
                chunk_index=int(metadata.get("chunk_index", index)),
                metadata=metadata,
            )
            retrieved.append(
                RetrievedChunk(
                    rank=index + 1,
                    score=round(1 - results["distances"][0][index], 4),
                    chunk=chunk,
                )
            )
        return retrieved
