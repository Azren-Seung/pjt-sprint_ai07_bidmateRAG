from bidmate_rag.pipelines.chat import RAGChatPipeline
from bidmate_rag.retrieval.memory import ConversationMemory
from bidmate_rag.schema import Chunk, GenerationResult, RetrievedChunk


class FakeRetriever:
    _last_debug = {
        "original_query": "요구사항 알려줘",
        "rewritten_query": "요구사항 알려줘",
        "rewrite_applied": False,
        "rewrite_reason": "original",
        "rewrite_prompt_tokens": 0,
        "rewrite_completion_tokens": 0,
        "rewrite_total_tokens": 0,
        "rewrite_cost_usd": 0.0,
        "retrieved_chunks_before_rerank": [],
        "retrieved_chunks_after_rerank": [],
    }

    def retrieve(
        self,
        query: str,
        chat_history=None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):
        chunk = Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="핵심 요구사항",
            text_with_meta="[발주기관: 기관 | 사업명: 사업]\n핵심 요구사항",
            char_count=7,
            section="요구사항",
            content_type="text",
            chunk_index=0,
            metadata={"사업명": "사업", "발주 기관": "기관", "파일명": "sample.hwp"},
        )
        return [RetrievedChunk(rank=1, score=0.95, chunk=chunk)]


class FakeLLM:
    provider_name = "fake-llm"
    model_name = "fake-model"

    def generate(self, question, context_chunks, history, generation_config, system_prompt):
        return GenerationResult(
            question_id="q-1",
            question=question,
            scenario="scenario_b",
            run_id="run-1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer="문서 기준 답변",
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in context_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in context_chunks],
            retrieved_chunks=context_chunks,
            latency_ms=10,
            token_usage={"total": 5},
        )


def test_chat_pipeline_returns_generation_result() -> None:
    pipeline = RAGChatPipeline(retriever=FakeRetriever(), llm=FakeLLM())

    result = pipeline.answer("요구사항 알려줘")

    assert result.answer == "문서 기준 답변"
    assert result.retrieved_chunk_ids == ["chunk-1"]
    assert result.llm_model == "fake-model"


def test_chat_pipeline_includes_memory_debug() -> None:
    pipeline = RAGChatPipeline(
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        memory=ConversationMemory(
            max_recent_turns=4,
            max_summary_chars=120,
            agency_list=["국민연금공단"],
        ),
    )

    result = pipeline.answer(
        "평가기준은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
    )

    assert "memory_summary" in result.debug
    assert "memory_slots" in result.debug
    assert result.debug["generation_cost_usd"] == 0.0
