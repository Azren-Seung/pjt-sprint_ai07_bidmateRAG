"""Index building pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.schema import Chunk


def _row_to_chunk(row: dict) -> Chunk:
    metadata_keys = {
        key
        for key in row
        if key
        not in {
            "chunk_id",
            "doc_id",
            "text",
            "text_with_meta",
            "char_count",
            "section",
            "content_type",
            "chunk_index",
        }
    }
    metadata = {key: row[key] for key in metadata_keys}
    return Chunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row.get("doc_id") or metadata.get("공고 번호") or metadata.get("파일명")),
        text=str(row["text"]),
        text_with_meta=str(row["text_with_meta"]),
        char_count=int(row["char_count"]),
        section=str(row.get("section", "")),
        content_type=str(row.get("content_type", "text")),
        chunk_index=int(row.get("chunk_index", 0)),
        metadata=metadata,
    )


def build_index_from_parquet(
    chunks_path: str | Path,
    embedder,
    vector_store,
    min_chars: int = 50,
) -> dict[str, int | str]:
    frame = pd.read_parquet(chunks_path)
    filtered = frame[frame["char_count"] >= min_chars].copy()
    chunks = [_row_to_chunk(row) for row in filtered.to_dict(orient="records")]

    # 배치 임베딩 (API 토큰 한도 대응, 100개씩)
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embeddings = embedder.embed_documents([c.text_with_meta for c in batch])
        all_embeddings.extend(batch_embeddings)

    vector_store.upsert(chunks, all_embeddings)
    return {
        "input_chunks": int(len(frame)),
        "indexed_chunks": len(chunks),
        "embedding_provider": getattr(embedder, "provider_name", ""),
        "embedding_model": getattr(embedder, "model_name", ""),
    }
