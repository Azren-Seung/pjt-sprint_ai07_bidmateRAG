"""벡터 인덱스 빌드 파이프라인.

chunks.parquet를 읽어 임베딩을 생성하고 ChromaDB에 저장한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.schema import Chunk


def _row_to_chunk(row: dict) -> Chunk:
    """parquet 행(dict)을 Chunk 객체로 변환한다.

    Args:
        row: chunks.parquet의 한 행.

    Returns:
        Chunk 인스턴스.
    """
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
    """parquet 파일에서 청크를 읽어 벡터 인덱스를 생성한다.

    Args:
        chunks_path: chunks.parquet 경로.
        embedder: 임베딩 프로바이더 (embed_documents 메서드 필요).
        vector_store: 벡터 저장소 (upsert 메서드 필요).
        min_chars: 최소 글자수 미만의 청크는 제외.

    Returns:
        입력/인덱싱 청크 수, 임베딩 모델 정보 딕셔너리.
    """
    frame = pd.read_parquet(chunks_path, dtype_backend="numpy_nullable")
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
