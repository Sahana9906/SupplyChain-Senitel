"""Tests for the local BGE embedding client. Uses a mocked
SentenceTransformer throughout -- these tests must run without
downloading BAAI/bge-base-en-v1.5 from Hugging Face, per the project's
own instruction not to make the whole suite depend on a large model
download.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from logisense import models as m  # noqa: F401 -- ensures all tables are registered on Base.metadata before the `session` fixture creates them, even when this file is run in isolation
from logisense.config import Settings


class _FakeSentenceTransformer:
    """Deterministic fake standing in for the real BAAI/bge-base-en-v1.5
    model -- fixed 768-dim output (matching the real model's documented
    dimensionality), records exactly what text it was asked to encode
    so tests can verify the query/document asymmetry."""

    DIM = 768

    def __init__(self, model_name):
        self.model_name_loaded = model_name
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        return np.ones((len(texts), self.DIM), dtype=float)

    def get_sentence_embedding_dimension(self):
        return self.DIM


def _settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", gemini_api_key="x")


@pytest.fixture()
def bge_client():
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
        yield BGEEmbeddingClient(_settings())


# ---------------------------------------------------------------------------
# Test 1: local embedding
# ---------------------------------------------------------------------------

def test_single_text_returns_a_vector(bge_client):
    result = bge_client.embed(["Supplier A has a delivery delay."])
    assert len(result) == 1
    assert len(result[0]) == 768
    assert all(isinstance(v, float) for v in result[0])


# ---------------------------------------------------------------------------
# Test 2: batch embedding
# ---------------------------------------------------------------------------

def test_batch_embedding_returns_one_vector_per_input(bge_client):
    texts = [
        "Supplier A has a delivery delay.",
        "Warehouse WH-01 is below reorder point.",
        "Carrier CAR-01 missed its SLA commitment.",
    ]
    result = bge_client.embed_documents(texts)
    assert len(result) == len(texts)
    assert all(len(vec) == 768 for vec in result)


# ---------------------------------------------------------------------------
# Test 3: query embedding
# ---------------------------------------------------------------------------

def test_query_embedding_uses_same_local_model_with_instruction_prefix(bge_client):
    vector = bge_client.embed_query("Which suppliers are affected by delivery delays?")
    assert len(vector) == 768

    # The BGE-documented retrieval instruction must be applied to the
    # query text, and ONLY the query -- never plain, unprefixed text.
    encoded_texts = bge_client._model.encode_calls[-1]
    assert len(encoded_texts) == 1
    assert encoded_texts[0].startswith("Represent this sentence for searching relevant passages: ")
    assert "Which suppliers are affected by delivery delays?" in encoded_texts[0]


def test_document_embedding_does_not_get_the_query_instruction_prefix(bge_client):
    bge_client.embed_documents(["Supplier A has a delivery delay."])
    encoded_texts = bge_client._model.encode_calls[-1]
    assert encoded_texts == ["Supplier A has a delivery delay."]
    assert "Represent this sentence" not in encoded_texts[0]


# ---------------------------------------------------------------------------
# Test 4: no Google embedding API
# ---------------------------------------------------------------------------

def test_bge_client_never_touches_the_google_genai_client():
    exploding_genai_client = MagicMock(side_effect=AssertionError("BGEEmbeddingClient must never construct a Google genai.Client"))
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer), \
         patch("google.genai.Client", exploding_genai_client):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient

        client = BGEEmbeddingClient(_settings())
        client.embed_documents(["some text"])
        client.embed_query("some query")

    exploding_genai_client.assert_not_called()


def test_bge_client_works_without_a_gemini_api_key():
    """Embedding generation must not require GEMINI_API_KEY -- only
    Gemini generation (hypothesis validation / Q&A) needs one."""
    settings = Settings(database_url="sqlite:///:memory:", gemini_api_key=None)
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient

        client = BGEEmbeddingClient(settings)  # must not raise
        result = client.embed_documents(["text"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 5: Chroma integration
# ---------------------------------------------------------------------------

def test_documents_can_be_indexed_and_retrieved_via_chroma(session, vector_store):
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
        from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer

        embedding_client = BGEEmbeddingClient(_settings())

        # Make the fake model actually distinguish content (constant
        # vectors from _FakeSentenceTransformer above would tie every
        # similarity score) so retrieval ranking is meaningfully testable.
        def varied_encode(texts, **kwargs):
            embedding_client._model.encode_calls.append(list(texts))
            vectors = []
            for t in texts:
                vec = np.zeros(768)
                for i, ch in enumerate(t):
                    vec[ord(ch) % 768] += 1.0
                vectors.append(vec)
            return np.array(vectors)
        embedding_client._model.encode = varied_encode

        import tempfile, os
        knowledge_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(knowledge_dir, "supplier_contracts"))
        with open(os.path.join(knowledge_dir, "supplier_contracts", "SUP-099_contract.md"), "w") as f:
            f.write("# Supply Agreement SUP-099\n\n## Penalty Terms\n\nLate delivery incurs a 5% penalty per day.\n")

        indexer = KnowledgeBaseIndexer(session, embedding_client, vector_store, chunk_size=2000, chunk_overlap=50)
        result = indexer.index_directory(knowledge_dir)

        assert result.documents_indexed == 1
        assert result.total_chunks >= 1
        assert vector_store.collection.count() == result.total_chunks

        query_embedding = embedding_client.embed_query("What is the penalty for late delivery?")
        matches = vector_store.query(query_embedding, top_k=5)
        assert len(matches) >= 1
        assert matches[0].metadata.get("filename") == "SUP-099_contract.md"


# ---------------------------------------------------------------------------
# Chroma dimension-safety (rebuild mechanism)
# ---------------------------------------------------------------------------

def test_incompatible_dimension_collection_is_rebuilt_not_mixed(vector_store):
    # Simulate a collection previously populated by a DIFFERENT
    # embedding model (e.g. the old Gemini one) with a different
    # dimensionality than the current BGE model produces.
    vector_store.upsert(
        ids=["old-vec"], embeddings=[[0.1] * 1536], documents=["old content"],
        metadatas=[{"document_id": 1, "chunk_id": 1}],
    )
    assert vector_store.collection.count() == 1

    was_rebuilt = vector_store.ensure_compatible_dimension(expected_dim=768)

    assert was_rebuilt is True
    assert vector_store.collection.count() == 0  # old, incompatible vectors are gone


def test_compatible_dimension_collection_is_left_alone(vector_store):
    vector_store.upsert(
        ids=["vec-1"], embeddings=[[0.1] * 768], documents=["content"],
        metadatas=[{"document_id": 1, "chunk_id": 1}],
    )
    was_rebuilt = vector_store.ensure_compatible_dimension(expected_dim=768)

    assert was_rebuilt is False
    assert vector_store.collection.count() == 1  # untouched
