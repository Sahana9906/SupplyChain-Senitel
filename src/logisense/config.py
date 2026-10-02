from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Phase 1 runtime configuration. Deliberately has no reference to any
    particular source dataset -- ingestion/normalization is a separate
    concern that maps a dataset into the canonical schema before Phase 1
    ever runs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str = Field(
        "postgresql+psycopg://logisense:logisense@localhost:5432/logisense",
        validation_alias="DATABASE_URL",
    )

    # Provider migration note: this project's LLM calls moved from Google
    # Gemini to Groq. Only the provider changed -- everything downstream
    # of the LLM (Pydantic response models, structured-output validation,
    # retrieval, business logic) is untouched.
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")

    # External data connectors for multi-source grounded investigation.
    # Open-Meteo and GDELT need no key (free/public APIs). AISstream.io
    # is also free but requires registering for a key at aisstream.io --
    # when unset, that connector marks itself unavailable rather than
    # crashing or fabricating data.
    aisstream_api_key: str | None = Field(default=None, validation_alias="AISSTREAM_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", validation_alias="GROQ_MODEL")
    groq_timeout_seconds: int = 45

    # Groq's Python SDK already retries 429/5xx responses internally with
    # exponential backoff + jitter (see groq._base_client) -- this only
    # configures how many attempts it makes, exactly like the previous
    # Gemini retry setting did, rather than adding a second, duplicate
    # retry loop on top of the SDK's own.
    groq_retry_max_attempts: int = Field(default=5, validation_alias="GROQ_RETRY_MAX_ATTEMPTS")

    # RAG layer (document chunking, embeddings, vector search).
    #
    # Embedding is a LOCAL Sentence Transformer model -- no API key
    # required for embedding generation. This is deliberately separate
    # from groq_model above, which powers LLM generation (hypothesis
    # validation, Q&A answers) via Groq.
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", validation_alias="EMBEDDING_MODEL")
    chroma_persist_dir: str = Field(default=".chroma_data", validation_alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="logisense_documents", validation_alias="CHROMA_COLLECTION_NAME"
    )
    rag_chunk_size: int = Field(default=800, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=150, validation_alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    rag_min_similarity: float = Field(default=0.25, validation_alias="RAG_MIN_SIMILARITY")

    # Phase 2: hybrid (dense + BM25) retrieval.
    retrieval_mode: str = Field(default="hybrid", validation_alias="RETRIEVAL_MODE")  # semantic | bm25 | hybrid
    dense_top_k: int = Field(default=20, validation_alias="DENSE_TOP_K")
    bm25_top_k: int = Field(default=20, validation_alias="BM25_TOP_K")
    hybrid_top_k: int = Field(default=5, validation_alias="HYBRID_TOP_K")
    rrf_k: int = Field(default=60, validation_alias="RRF_K")
    # Co-located with chroma_persist_dir (the project's existing
    # persistence directory for retrieval indexes) rather than
    # introducing a new persistence location.
    bm25_index_path: str = Field(default=".chroma_data/bm25_index.pkl", validation_alias="BM25_INDEX_PATH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
