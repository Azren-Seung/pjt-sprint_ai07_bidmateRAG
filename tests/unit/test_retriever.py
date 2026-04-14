from bidmate_rag.retrieval.retriever import RAGRetriever
from bidmate_rag.schema import Chunk, RetrievedChunk


class FakeEmbedder:
    provider_name = "fake-embedder"
    model_name = "fake-embedding-model"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, query_results: list[RetrievedChunk] | None = None):
        self.last_kwargs = None
        self.calls = []
        self.query_results = query_results

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        if self.query_results is not None:
            return self.query_results
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


class EmptyShortlistMetadataStore:
    agency_list = ["국민연금공단", "기초과학연구원"]

    def find_relevant_docs(self, query: str, top_n: int = 3):
        return []


class FakeReranker:
    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return [self.scores_by_text[text] for _, text in pairs]


def _retrieved_chunk(
    chunk_id: str,
    score: float,
    *,
    agency: str,
    section: str = "",
    content_type: str = "text",
    doc_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id or f"{chunk_id}-doc",
            text=chunk_id,
            text_with_meta=chunk_id,
            char_count=len(chunk_id),
            section=section,
            content_type=content_type,
            chunk_index=0,
            metadata={
                "사업명": f"{chunk_id}-사업",
                "발주 기관": agency,
                "파일명": f"{chunk_id}.hwp",
            },
        ),
    )


class ScopedFakeVectorStore:
    def __init__(self, results_by_agency: dict[str, list[RetrievedChunk]]):
        self.results_by_agency = results_by_agency
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        agency = kwargs["where"]["발주 기관"]
        return self.results_by_agency[agency][: kwargs["top_k"]]


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

    assert len(vector_store.calls) == 1
    assert vector_store.last_kwargs["where"] == {"파일명": {"$in": ["doc-1.hwp", "doc-2.hwp"]}}
    assert results[0].score > 0


def test_retriever_merges_multi_agency_filter_results_round_robin() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk("nps-1", 0.99, agency="국민연금공단"),
                _retrieved_chunk("nps-2", 0.98, agency="국민연금공단"),
            ],
            "기초과학연구원": [
                _retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원"),
                _retrieved_chunk("ibs-2", 0.96, agency="기초과학연구원"),
            ],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원 사업을 비교해줘",
        top_k=4,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1", "nps-2", "ibs-2"]
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_retriever_keeps_explicit_multi_source_filter_single_query_without_comparison_intent() -> (
    None
):
    vector_store = FakeVectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "두 기관 요구사항 정리해줘",
        top_k=4,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert len(vector_store.calls) == 1
    assert vector_store.last_kwargs["where"] == {
        "발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}
    }


def test_retriever_fans_out_multi_agency_filter_for_per_source_each_query() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve(
        "각 기관 요구사항을 각각 정리해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]


def test_retriever_fans_out_query_named_multi_agency_comparison_without_explicit_filter() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [_retrieved_chunk("nps-1", 0.99, agency="국민연금공단")],
            "기초과학연구원": [_retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원")],
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
    )

    results = retriever.retrieve("국민연금공단과 기초과학연구원 사업을 비교해줘", top_k=2)

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [result.chunk.chunk_id for result in results] == ["nps-1", "ibs-1"]


def test_retriever_applies_cross_encoder_after_multi_agency_fan_out_merge() -> None:
    vector_store = ScopedFakeVectorStore(
        {
            "국민연금공단": [
                _retrieved_chunk("nps-1", 0.99, agency="국민연금공단"),
                _retrieved_chunk("nps-2", 0.98, agency="국민연금공단"),
            ],
            "기초과학연구원": [
                _retrieved_chunk("ibs-1", 0.97, agency="기초과학연구원"),
                _retrieved_chunk("ibs-2", 0.96, agency="기초과학연구원"),
            ],
        }
    )
    reranker = FakeReranker(
        {
            "[발주기관: 국민연금공단 | 사업명: nps-1-사업]\nnps-1": 0.20,
            "[발주기관: 기초과학연구원 | 사업명: ibs-1-사업]\nibs-1": 0.95,
            "[발주기관: 국민연금공단 | 사업명: nps-2-사업]\nnps-2": 0.90,
            "[발주기관: 기초과학연구원 | 사업명: ibs-2-사업]\nibs-2": 0.10,
        }
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=FakeEmbedder(),
        metadata_store=FakeMetadataStore(),
        reranker_model=reranker,
    )

    results = retriever.retrieve(
        "국민연금공단과 기초과학연구원 사업을 비교해줘",
        top_k=2,
        metadata_filter={"발주 기관": {"$in": ["국민연금공단", "기초과학연구원"]}},
    )

    assert [call["where"]["발주 기관"] for call in vector_store.calls] == [
        "국민연금공단",
        "기초과학연구원",
    ]
    assert [call["top_k"] for call in vector_store.calls] == [8, 8]
    assert reranker.calls == [
        [
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 국민연금공단 | 사업명: nps-1-사업]\nnps-1",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 기초과학연구원 | 사업명: ibs-1-사업]\nibs-1",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 국민연금공단 | 사업명: nps-2-사업]\nnps-2",
            ],
            [
                "국민연금공단과 기초과학연구원 사업을 비교해줘",
                "[발주기관: 기초과학연구원 | 사업명: ibs-2-사업]\nibs-2",
            ],
        ]
    ]
    assert [result.chunk.chunk_id for result in results] == ["ibs-1", "nps-2"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == [0.95, 0.9]


def test_retriever_rewrites_follow_up_query_and_inherits_recent_agency_filter() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=FakeMetadataStore(),
    )

    retriever.retrieve(
        "그 사업 예산은?",
        top_k=2,
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert embedder.queries == ["국민연금공단 차세대 ERP 사업 예산은?"]
    assert vector_store.last_kwargs["where"] == {"발주 기관": "국민연금공단"}


def test_retriever_can_disable_multiturn_rewrite_and_history_filter() -> None:
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        metadata_store=EmptyShortlistMetadataStore(),
        enable_multiturn=False,
    )

    retriever.retrieve(
        "그 사업 예산은?",
        top_k=2,
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert embedder.queries == ["그 사업 예산은?"]
    assert vector_store.last_kwargs["where"] is None
