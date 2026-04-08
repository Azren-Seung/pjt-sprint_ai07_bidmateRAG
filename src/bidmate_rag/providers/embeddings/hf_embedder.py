"""Hugging Face embedding adapter."""

from __future__ import annotations

from bidmate_rag.providers.embeddings.base import BaseEmbeddingProvider


class HFEmbedder(BaseEmbeddingProvider):
    def __init__(self, model_name: str, encode_fn=None, provider_name: str = "huggingface") -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._encode_fn = encode_fn

    def _get_encode_fn(self):
        if self._encode_fn is not None:
            return self._encode_fn
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "sentence-transformers is required for HF embedding. "
                "Install the ml dependency group."
            ) from exc
        model = SentenceTransformer(self.model_name)
        self._encode_fn = model.encode
        return self._encode_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_encode_fn()(texts)
        return [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]
