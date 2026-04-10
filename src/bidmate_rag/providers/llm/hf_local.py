"""Local Hugging Face generation provider."""

from __future__ import annotations

import time
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
        tokenizer = generator.tokenizer
        full_input = f"{system_prompt}\n\n{prompt}"

        # 입력 토큰 수 측정
        input_tokens = len(tokenizer.encode(full_input))

        # 지연 시간 측정
        start = time.time()
        response = generator(
            full_input,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            do_sample=False,
            return_full_text=False,
        )
        latency_ms = (time.time() - start) * 1000

        generated_text = response[0]["generated_text"].strip() if response else ""

        # 출력 토큰 수 측정
        output_tokens = len(tokenizer.encode(generated_text)) if generated_text else 0

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
            latency_ms=latency_ms,
            token_usage={
                "prompt": input_tokens,
                "completion": output_tokens,
                "total": input_tokens + output_tokens,
            },
            cost_usd=0.0,
            context=_build_context(context_chunks),
        )
