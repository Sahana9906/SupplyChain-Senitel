"""Tests for Phase 2 hybrid retrieval: BM25 lexical search + dense
(BGE/Chroma) semantic search combined via Reciprocal Rank Fusion.

Mocked SentenceTransformer throughout -- no real model download needed,
matching the same convention as test_bge_embeddings.py.
"""
from unittest.mock import patch

import numpy as np
import pytest

from logisense import models as m  # noqa: F401 -- registers all tables on Base.metadata when this file runs in isolation
from logisense.config import Settings
from logisense.services.rag.bm25_index import BM25Index, IndexedChunk
from logisense.services.rag.hybrid_retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalCandidate,
    reciprocal_rank_fusion,
)


class _FakeSentenceTransformer:
    DIM = 768

    def __init__(self, model_name):
        pass

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for t in texts:
            vec = np.zeros(self.DIM)
            for ch in t.lower():
                vec[ord(ch) % self.DIM] += 1.0
            vectors.append(vec)
        return np.array(vectors)

    def get_sentence_embedding_dimension(self):
        return self.DIM


def _settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", gemini_api_key="x")


# ---------------------------------------------------------------------------
# Dense retrieval test
# ---------------------------------------------------------------------------

def test_dense_retrieval_still_works(vector_store):
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
        embedding_client = BGEEmbeddingClient(_settings())

    [doc_embedding] = embedding_client.embed_documents(["Supplier SUP-014 has repeated delivery delays this quarter."])
    vector_store.upsert(
        ids=["document:1:chunk:0"], embeddings=[doc_embedding],
        documents=["Supplier SUP-014 has repeated delivery delays this quarter."],
        metadatas=[{"document_id": 1, "chunk_id": 1, "category": "historical_incidents", "filename": "SUP-014_history.md"}],
    )

    retriever = DenseRetriever(embedding_client, vector_store)
    results = retriever.retrieve("Which suppliers are causing delivery problems?", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "document:1:chunk:0"
    assert results[0].dense_rank == 1
    assert results[0].dense_score is not None
    assert results[0].bm25_rank is None


# ---------------------------------------------------------------------------
# BM25 test
# ---------------------------------------------------------------------------

def test_bm25_retrieves_exact_identifier_match():
    chunks = [
        IndexedChunk(
            vector_id="document:1:chunk:0", text="Supplier SUP-104 has a 7-day SLA.",
            metadata={"document_id": 1, "chunk_id": 1, "category": "supplier_contracts", "filename": "SUP-104_contract.md"},
        ),
        IndexedChunk(
            vector_id="document:2:chunk:0", text="Supplier SUP-205 has a 10-day SLA.",
            metadata={"document_id": 2, "chunk_id": 2, "category": "supplier_contracts", "filename": "SUP-205_contract.md"},
        ),
    ]
    index = BM25Index(chunks)
    retriever = BM25Retriever(index)

    results = retriever.retrieve("SUP-104 7-day SLA", top_k=5)

    assert len(results) >= 1
    assert results[0].chunk_id == "document:1:chunk:0"
    assert results[0].bm25_rank == 1
    assert results[0].dense_rank is None


def test_bm25_index_persists_and_loads(tmp_path):
    chunks = [IndexedChunk(vector_id="document:1:chunk:0", text="Supplier SUP-104 has a 7-day SLA.", metadata={"document_id": 1})]
    index = BM25Index(chunks)
    path = str(tmp_path / "bm25.pkl")
    index.save(path)

    loaded = BM25Index.load(path)
    results = loaded.query("SUP-104", top_k=5)
    assert len(results) == 1
    assert results[0][0].vector_id == "document:1:chunk:0"


def test_bm25_load_missing_index_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="rag-index-knowledge"):
        BM25Index.load(str(tmp_path / "does_not_exist.pkl"))


# ---------------------------------------------------------------------------
# RRF test
# ---------------------------------------------------------------------------

def test_rrf_ranks_chunks_found_by_both_retrievers_highest():
    dense_results = [
        RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={}, dense_rank=1, dense_score=0.9),
        RetrievalCandidate(chunk_id="chunk-B", text="b", metadata={}, dense_rank=2, dense_score=0.8),
    ]
    bm25_results = [
        RetrievalCandidate(chunk_id="chunk-C", text="c", metadata={}, bm25_rank=1, bm25_score=5.0),
        RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={}, bm25_rank=2, bm25_score=4.0),
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert fused[0].chunk_id == "chunk-A"
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[0].hybrid_score > fused[1].hybrid_score > fused[2].hybrid_score


def test_rrf_k_constant_changes_relative_weighting():
    fused_k60 = reciprocal_rank_fusion(
        [RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={}, dense_rank=1, dense_score=0.9)],
        [RetrievalCandidate(chunk_id="chunk-B", text="b", metadata={}, bm25_rank=1, bm25_score=5.0)],
        k=60,
    )
    fused_k1 = reciprocal_rank_fusion(
        [RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={}, dense_rank=1, dense_score=0.9)],
        [RetrievalCandidate(chunk_id="chunk-B", text="b", metadata={}, bm25_rank=1, bm25_score=5.0)],
        k=1,
    )
    assert fused_k60[0].hybrid_score != fused_k1[0].hybrid_score


# ---------------------------------------------------------------------------
# Deduplication test
# ---------------------------------------------------------------------------

def test_chunk_found_by_both_retrievers_appears_only_once():
    dense_results = [RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={"x": 1}, dense_rank=1, dense_score=0.9)]
    bm25_results = [RetrievalCandidate(chunk_id="chunk-A", text="a", metadata={"x": 1}, bm25_rank=3, bm25_score=2.0)]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert len(fused) == 1
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 3


# ---------------------------------------------------------------------------
# Metadata test
# ---------------------------------------------------------------------------

def test_metadata_survives_hybrid_fusion():
    real_metadata = {"document_id": 7, "chunk_id": 42, "category": "supplier_contracts", "filename": "SUP-104_contract.md", "section": "SLA Terms"}
    dense_results = [RetrievalCandidate(chunk_id="document:7:chunk:0", text="text", metadata=real_metadata, dense_rank=1, dense_score=0.9)]
    bm25_results = []

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert fused[0].metadata == real_metadata
    assert fused[0].source == "supplier_contracts/SUP-104_contract.md"
    assert fused[0].document_id == 7


# ---------------------------------------------------------------------------
# No-Google test
# ---------------------------------------------------------------------------

def test_hybrid_retrieval_never_calls_google_embedding_api():
    from unittest.mock import MagicMock
    exploding_client = MagicMock(side_effect=AssertionError("must never construct google.genai.Client"))

    chunks = [IndexedChunk(vector_id="document:1:chunk:0", text="Supplier SUP-104 has a 7-day SLA.", metadata={"document_id": 1})]
    bm25_index = BM25Index(chunks)

    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer), \
         patch("google.genai.Client", exploding_client):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
        import chromadb
        from logisense.services.rag.vector_store import ChromaVectorStore

        embedding_client = BGEEmbeddingClient(_settings())
        vs = ChromaVectorStore(_settings(), client=chromadb.EphemeralClient())
        [emb] = embedding_client.embed_documents(["Supplier SUP-104 has a 7-day SLA."])
        vs.upsert(ids=["document:1:chunk:0"], embeddings=[emb], documents=["Supplier SUP-104 has a 7-day SLA."], metadatas=[{"document_id": 1}])

        dense = DenseRetriever(embedding_client, vs)
        bm25 = BM25Retriever(bm25_index)
        hybrid = HybridRetriever(dense, bm25, rrf_k=60)
        results = hybrid.retrieve("SUP-104 7-day SLA", dense_top_k=5, bm25_top_k=5, final_top_k=5)

    exploding_client.assert_not_called()
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

def test_end_to_end_documents_to_hybrid_evidence(session, vector_store, tmp_path):
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
        from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer

        embedding_client = BGEEmbeddingClient(_settings())

        knowledge_dir = tmp_path / "company_knowledge"
        (knowledge_dir / "supplier_contracts").mkdir(parents=True)
        (knowledge_dir / "supplier_contracts" / "SUP-104_contract.md").write_text(
            "# Supply Agreement SUP-104\n\n## SLA Terms\n\nSupplier SUP-104 has a 7-day SLA for delivery.\n"
        )
        (knowledge_dir / "supplier_contracts" / "SUP-205_contract.md").write_text(
            "# Supply Agreement SUP-205\n\n## SLA Terms\n\nSupplier SUP-205 has a 10-day SLA for delivery.\n"
        )

        indexer = KnowledgeBaseIndexer(session, embedding_client, vector_store, chunk_size=2000, chunk_overlap=50)
        index_result = indexer.index_directory(str(knowledge_dir))
        assert index_result.documents_indexed == 2

        bm25_index = BM25Index.build_from_vector_store(vector_store)
        assert len(bm25_index.chunks) == index_result.total_chunks

        dense = DenseRetriever(embedding_client, vector_store)
        bm25 = BM25Retriever(bm25_index)
        hybrid = HybridRetriever(dense, bm25, rrf_k=60)

        results = hybrid.retrieve("Why did SUP-104 breach the 7-day SLA?", dense_top_k=10, bm25_top_k=10, final_top_k=5)

        assert len(results) >= 1
        assert results[0].metadata.get("filename") == "SUP-104_contract.md"
