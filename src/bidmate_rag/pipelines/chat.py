"""RAG 채팅 파이프라인.

검색(retrieval) → LLM 생성(generation)을 연결하여
사용자 질문에 문서 근거 기반 답변을 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.schema import GenerationResult


@dataclass(slots=True)
class RAGChatPipeline:
    """RAG 질의응답 파이프라인.

    retriever로 관련 청크를 검색하고 llm으로 답변을 생성한다.

    Attributes:
        retriever: 벡터 검색 + 메타데이터 필터링 리트리버.
        llm: LLM 프로바이더 (generate 메서드 필요).
        system_prompt: 시스템 프롬프트.
        default_generation_config: 기본 생성 설정.
    """

    retriever: object
    llm: object
    system_prompt: str = SYSTEM_PROMPT
    default_generation_config: dict = field(default_factory=dict)

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        top_k: int = 5,
        generation_config: dict | None = None,
        question_id: str | None = None,
        scenario: str | None = None,
        run_id: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> GenerationResult:
        """사용자 질문에 대해 RAG 기반 답변을 생성한다.

        Args:
            question: 사용자 질문.
            chat_history: 이전 대화 히스토리.
            top_k: 검색할 청크 수.
            generation_config: LLM 생성 설정 오버라이드.
            question_id: 평가용 질문 ID.
            scenario: 실험 시나리오 (A/B).
            run_id: 실행 ID.
            embedding_provider: 임베딩 프로바이더명.
            embedding_model: 임베딩 모델명.

        Returns:
            GenerationResult (답변, 검색 청크, 토큰 사용량 등).
        """
        retrieved = self.retriever.retrieve(question, chat_history=chat_history, top_k=top_k)
        config = {**self.default_generation_config, **(generation_config or {})}
        if question_id is not None:
            config["question_id"] = question_id
        if scenario is not None:
            config["scenario"] = scenario
        if run_id is not None:
            config["run_id"] = run_id
        if embedding_provider is not None:
            config["embedding_provider"] = embedding_provider
        if embedding_model is not None:
            config["embedding_model"] = embedding_model
        return self.llm.generate(
            question=question,
            context_chunks=retrieved,
            history=chat_history or [],
            generation_config=config,
            system_prompt=self.system_prompt,
        )
