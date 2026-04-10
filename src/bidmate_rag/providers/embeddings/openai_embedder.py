"""OpenAI embedding adapter."""

from __future__ import annotations

import os

from openai import OpenAI

from bidmate_rag.providers.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbedder(BaseEmbeddingProvider):
    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
        provider_name: str = "openai",
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_base = api_base
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )
        self.cumulative_tokens: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        usage = getattr(response, "usage", None)
        self.cumulative_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]
