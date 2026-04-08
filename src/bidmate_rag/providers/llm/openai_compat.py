"""OpenAI-compatible client adapter for OpenAI and local endpoints."""

from __future__ import annotations

import os
from uuid import uuid4

from openai import OpenAI

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


class OpenAICompatibleLLM(BaseLLMProvider):
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_base = api_base
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        context = _build_context(
            context_chunks, max_chars=generation_config.get("max_context_chars", 8000)
        )
        messages = [{"role": "system", "content": system_prompt}]
        for item in history[-4:]:
            messages.append({"role": "user", "content": item["user"]})
            messages.append({"role": "assistant", "content": item["assistant"]})
        messages.append({"role": "user", "content": build_rag_user_prompt(question, context)})
        import time as _time
        _start = _time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=generation_config.get("max_completion_tokens", 16000),
        )
        _elapsed_ms = (_time.time() - _start) * 1000
        usage = getattr(response, "usage", None)
        return GenerationResult(
            question_id=generation_config.get("question_id", f"q-{uuid4().hex[:8]}"),
            question=question,
            scenario=generation_config.get("scenario", "ad-hoc"),
            run_id=generation_config.get("run_id", f"run-{uuid4().hex[:8]}"),
            embedding_provider=generation_config.get("embedding_provider", ""),
            embedding_model=generation_config.get("embedding_model", ""),
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer=response.choices[0].message.content or "(응답 없음)",
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in context_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in context_chunks],
            retrieved_chunks=context_chunks,
            latency_ms=round(_elapsed_ms, 1),
            token_usage={
                "prompt": getattr(usage, "prompt_tokens", 0),
                "completion": getattr(usage, "completion_tokens", 0),
                "total": getattr(usage, "total_tokens", 0),
            },
            cost_usd=float(generation_config.get("cost_usd", 0.0)),
            context=context,
        )
