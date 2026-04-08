"""RAG chat orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from bidmate_rag.config.prompts import SYSTEM_PROMPT
from bidmate_rag.schema import GenerationResult


@dataclass(slots=True)
class RAGChatPipeline:
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
