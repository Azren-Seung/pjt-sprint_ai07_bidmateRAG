from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.schema import Chunk, RetrievedChunk


class FakeEmbedder:
    provider_name = "fake-embedder"
    model_name = "fake-embedding-model"

    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return [
            RetrievedChunk(
                rank=1,
                score=0.9,
                chunk=Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="요구사항",
                    text_with_meta="[발주기관: 기관 | 사업명: 사업]\n요구사항",
                    char_count=4,
                    section="요구사항",
                    content_type="table",
                    chunk_index=0,
                    metadata={"사업명": "사업", "발주 기관": "국민연금공단", "파일명": "doc-1.hwp"},
                ),
            )
        ]


class FakeMetadataStore:
    agency_list = ["국민연금공단", "기초과학연구원"]

    def find_relevant_docs(self, query: str, top_n: int = 3):
        return ["doc-1.hwp", "doc-2.hwp"]


def test_retriever_merges_metadata_range_and_section_filters() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve("국민연금공단 2024년 5억 이상 보안 요구사항 알려줘", top_k=3)

    assert vector_store.last_kwargs["where"] == {
        "발주 기관": "국민연금공단",
        "사업 금액": {"$gte": 500000000},
        "공개연도": 2024,
    }
    assert vector_store.last_kwargs["where_document"] == {"$contains": "보안"}


def test_retriever_uses_metadata_store_shortlist_when_no_explicit_filter() -> None:
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve("비슷한 사업 비교해줘", top_k=2)

    assert vector_store.last_kwargs["where"] == {"파일명": {"$in": ["doc-1.hwp", "doc-2.hwp"]}}
    assert results[0].score > 0
