"""Embedding generation via the LLM provider -- kept available (not
deleted) for interface compatibility, but no longer constructed by
default anywhere in this project; the active embedding path is the
local BGE model in bge_embeddings.py (no API key required).

Provider migration note: this project's LLM calls moved from Gemini to
Groq (see gemini_client.py). Groq's API does not offer an embeddings
endpoint, so unlike GeminiHypothesisClient/GeminiIntakeClient (which have
a direct Groq equivalent), GeminiEmbeddingClient has none -- it now
raises NotImplementedError if ever constructed, rather than silently
being broken. This was already dormant/unused before the migration (see
above), so this has no effect on any current call path.
"""
from __future__ import annotations

from typing import Protocol

from logisense.config import Settings

_DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


class EmbeddingClient(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class GeminiEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set before generating embeddings")
        raise NotImplementedError(
            "Groq has no embeddings API. This client was already unused before the "
            "Gemini -> Groq migration (see bge_embeddings.py for the active local "
            "embedding path) and has no Groq equivalent to migrate to."
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Groq has no embeddings API; use the local BGE embedding client instead.")


def embed_query_text(embedding_client, text: str) -> list[float]:
    """Embeds a single query string, using embed_query() (with its
    model-specific instruction prefix, e.g. BGE's) when the client
    supports it, falling back to plain embed() for clients that only
    implement the basic EmbeddingClient protocol (e.g. GeminiEmbeddingClient,
    which has no query/document distinction). Centralizes this dispatch
    so every query-embedding call site in the project (hybrid_context,
    qa_service, evidence_validation) does the right thing automatically
    regardless of which embedding client is configured."""
    if hasattr(embedding_client, "embed_query"):
        return embedding_client.embed_query(text)
    embeddings = embedding_client.embed([text])
    return embeddings[0] if embeddings else []
