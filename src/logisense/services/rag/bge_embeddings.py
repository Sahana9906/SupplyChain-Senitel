"""Local embedding generation via a Sentence Transformers BGE model --
no Google API key, no network call to any external embedding API.

BAAI/bge-* models are documented to perform noticeably better on
retrieval when the QUERY (not the documents/passages) is prefixed with
an instruction describing the retrieval task. Documents are embedded as
plain text; only queries get the prefix. Mixing this up (or applying it
to both) measurably hurts retrieval quality, so embed_documents() and
embed_query() are kept as distinct methods rather than one that guesses.

Error handling is deliberately split into two exception types so a
caller (and a human debugging a failure) can immediately tell whether
the problem is "the model couldn't be loaded at all" (missing package,
no network to Hugging Face on first run, corrupt cache) versus "the
model loaded fine but this specific encode() call failed" (bad input,
out of memory, etc.) -- these need different fixes.
"""
from __future__ import annotations

from logisense.config import Settings

# BGE's documented query-side instruction for retrieval tasks. Applied
# ONLY to queries, never to indexed documents/passages -- this asymmetry
# is intentional and matches BAAI's own usage recommendation for the
# bge-base-en-v1.5 / bge-large-en-v1.5 family.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModelLoadError(RuntimeError):
    """The local embedding model could not be loaded -- e.g. the
    sentence-transformers package is missing, or (on first run) the
    model couldn't be downloaded from Hugging Face and isn't cached
    locally yet. Distinct from EmbeddingGenerationError so a caller can
    tell "nothing will ever work until this is fixed" apart from
    "this one encode() call failed"."""


class EmbeddingGenerationError(RuntimeError):
    """The model loaded successfully, but a specific .encode() call
    failed (e.g. bad input, out-of-memory). The model itself is fine --
    this failure is call-specific."""


class BGEEmbeddingClient:
    model_name: str

    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "sentence-transformers is not installed. Install it with "
                "`pip install -r requirements.txt` (or `pip install sentence-transformers`)."
            ) from exc

        self.model_name = settings.embedding_model
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Failed to load local embedding model '{self.model_name}'. If this is the "
                f"first run, it needs to download from Hugging Face and requires network "
                f"access to huggingface.co; if the model is already cached, this may be a "
                f"corrupt cache or an invalid model name. Original error: {exc}"
            ) from exc

    def get_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds document/passage text -- no instruction prefix. This
        is what gets indexed into the vector store."""
        if not texts:
            return []
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingGenerationError(f"Local embedding generation failed for {len(texts)} document(s): {exc}") from exc
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query -- prefixed with BGE's
        documented retrieval instruction, which document embeddings do
        NOT get. Must use the same model as embed_documents() (it does,
        since both use self._model) so the two vector spaces match."""
        try:
            embeddings = self._model.encode(
                [f"{_BGE_QUERY_INSTRUCTION}{text}"], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as exc:
            raise EmbeddingGenerationError(f"Local embedding generation failed for query: {exc}") from exc
        return embeddings[0].tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias matching the existing EmbeddingClient
        protocol used elsewhere in this project (KnowledgeBaseIndexer
        etc). Treats input as DOCUMENTS -- the majority existing call
        pattern (chunk indexing embeds a list of passages). Code
        embedding a single search QUERY should call embed_query()
        instead so the instruction prefix is applied correctly; see
        services/rag/embeddings.py::embed_query_text() for a helper that
        picks the right method automatically."""
        return self.embed_documents(texts)
