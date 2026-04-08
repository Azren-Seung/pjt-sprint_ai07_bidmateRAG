"""Common interfaces for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bidmate_rag.schema import GenerationResult, RetrievedChunk


class BaseLLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        raise NotImplementedError
