"""Local Hugging Face generation provider."""

from __future__ import annotations

from uuid import uuid4

from bidmate_rag.config.prompts import build_rag_user_prompt
from bidmate_rag.providers.llm.base import BaseLLMProvider
from bidmate_rag.schema import GenerationResult, RetrievedChunk


def _build_context(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for retrieved in chunks:
        metadata = retrieved.chunk.metadata
        source = f"[출처: {metadata.get('사업명', '')} | {metadata.get('발주 기관', '')}]"
        chunk_text = f"{source}\n{retrieved.chunk.text}"
        if total + len(chunk_text) > max_chars:
            break
        parts.append(chunk_text)
        total += len(chunk_text)
    return "\n\n---\n\n".join(parts)


class HFLocalLLM(BaseLLMProvider):
    def __init__(self, model_name: str, provider_name: str = "huggingface", generator=None) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._generator = generator

    def _get_generator(self):
        if self._generator is not None:
            return self._generator
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "transformers is required for the local HF provider. "
                "Install the ml dependency group."
            ) from exc
        self._generator = pipeline("text-generation", model=self.model_name)
        return self._generator

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        prompt = build_rag_user_prompt(question, _build_context(context_chunks))
        generator = self._get_generator()
        response = generator(
            f"{system_prompt}\n\n{prompt}",
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            do_sample=False,
        )
        generated_text = response[0]["generated_text"] if response else ""
        return GenerationResult(
            question_id=generation_config.get("question_id", f"q-{uuid4().hex[:8]}"),
            question=question,
            scenario=generation_config.get("scenario", "scenario_a"),
            run_id=generation_config.get("run_id", f"run-{uuid4().hex[:8]}"),
            embedding_provider=generation_config.get("embedding_provider", ""),
            embedding_model=generation_config.get("embedding_model", ""),
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer=generated_text,
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in context_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in context_chunks],
            retrieved_chunks=context_chunks,
            latency_ms=float(generation_config.get("latency_ms", 0.0)),
            token_usage={},
            cost_usd=0.0,
            context=_build_context(context_chunks),
        )
