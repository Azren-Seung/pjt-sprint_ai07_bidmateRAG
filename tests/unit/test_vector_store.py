"""Tests for retrieval/vector_store.py — stale chunk replacement."""

from __future__ import annotations

import chromadb

from bidmate_rag.retrieval.vector_store import ChromaVectorStore
from bidmate_rag.schema import Chunk


def _make_chunk(doc_id: str, chunk_index: int, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_{chunk_index}",
        doc_id=doc_id,
        text=text,
        text_with_meta=text,
        char_count=len(text),
        section="",
        content_type="text",
        chunk_index=chunk_index,
        metadata={"사업명": "test", "발주 기관": "org", "파일명": f"{doc_id}.hwp"},
    )


def _make_store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(str(tmp_path / "chroma"), "test-collection")


def _fake_embeddings(n: int) -> list[list[float]]:
    return [[float(i)] * 8 for i in range(n)]


# ---------------------------------------------------------------------------
# replace_documents: stale chunk 제거
# ---------------------------------------------------------------------------


def test_replace_documents_removes_stale_chunks(tmp_path):
    """같은 doc_id로 청크 수가 줄어도 stale 없어야 함."""
    store = _make_store(tmp_path)

    # 1차 빌드: docA → 5 chunks
    chunks_v1 = [_make_chunk("docA", i) for i in range(5)]
    store.replace_documents(chunks_v1, _fake_embeddings(5))
    assert store.collection.count() == 5

    # 2차 빌드: docA → 3 chunks (줄어듦)
    chunks_v2 = [_make_chunk("docA", i) for i in range(3)]
    store.replace_documents(chunks_v2, _fake_embeddings(3))
    assert store.collection.count() == 3  # stale 2개 제거됨

    # 남은 ID는 정확히 docA_0, docA_1, docA_2
    ids = sorted(store.collection.get()["ids"])
    assert ids == ["docA_0", "docA_1", "docA_2"]


def test_replace_documents_preserves_other_docs(tmp_path):
    """docA만 다시 빌드해도 docB 청크는 보존."""
    store = _make_store(tmp_path)

    # docA 3개 + docB 2개
    chunks = [_make_chunk("docA", i) for i in range(3)] + [
        _make_chunk("docB", i) for i in range(2)
    ]
    store.replace_documents(chunks, _fake_embeddings(5))
    assert store.collection.count() == 5

    # docA만 1개로 줄여서 다시 빌드
    chunks_v2 = [_make_chunk("docA", 0)]
    store.replace_documents(chunks_v2, _fake_embeddings(1))
    assert store.collection.count() == 3  # docA 1 + docB 2 = 3

    ids = sorted(store.collection.get()["ids"])
    assert ids == ["docA_0", "docB_0", "docB_1"]


def test_upsert_alone_leaves_stale_chunks(tmp_path):
    """upsert만 쓰면 줄어든 청크가 남는 것을 증명 (replace가 필요한 이유)."""
    store = _make_store(tmp_path)

    # 1차 빌드
    chunks_v1 = [_make_chunk("docA", i) for i in range(5)]
    store.upsert(chunks_v1, _fake_embeddings(5))
    assert store.collection.count() == 5

    # 2차 빌드 (upsert만)
    chunks_v2 = [_make_chunk("docA", i) for i in range(3)]
    store.upsert(chunks_v2, _fake_embeddings(3))
    assert store.collection.count() == 5  # ← stale 2개 남아 있음!


def test_replace_documents_with_empty_chunks_is_noop(tmp_path):
    store = _make_store(tmp_path)
    # 아무 청크 없으면 collection 건드리지 않아야 함
    store.replace_documents([], [])
    assert store.collection.count() == 0
