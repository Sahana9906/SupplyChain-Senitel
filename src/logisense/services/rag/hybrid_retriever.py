"""Hybrid retrieval: combines dense (BGE + Chroma) semantic search with
BM25 lexical search using Reciprocal Rank Fusion (RRF).

RRF, not a raw score sum, because BM25 scores and cosine similarity
scores live on completely different, uncalibrated scales -- adding them
directly would let whichever score happens to have a larger typical
magnitude dominate the ranking for no principled reason. RRF only uses
each retriever's RANK ORDER, which is scale-free by construction:

    RRF(chunk) = sum over retrievers of  1 / (k + rank_in_that_retriever)

A chunk that ranks well in BOTH retrievers accumulates a higher score
than one that ranks well in only one -- this is the actual mechanism
that makes "found by both dense and lexical search" a strong signal.
"""
from __future__ import annotations

from dataclasses import dataclass

from logisense.services.rag.bm25_index import BM25Index
from logisense.services.rag.embeddings import embed_query_text
from logisense.services.rag.vector_store import ChromaVectorStore


@dataclass
class RetrievalCandidate:
    """One retrieved chunk, with full provenance from whichever
    retriever(s) found it -- NOT frozen, since hybrid fusion fills in
    fields incrementally as the same chunk is found by more than one
    retriever."""

    chunk_id: str  # the Chroma vector_id, e.g. "document:5:chunk:2" -- the canonical identity shared by both retrievers
    text: str
    metadata: dict
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    hybrid_score: float = 0.0

    @property
    def document_id(self):
        return self.metadata.get("document_id")

    @property
    def source(self) -> str:
        return f"{self.metadata.get('category')}/{self.metadata.get('filename')}"


class DenseRetriever:
    def __init__(self, embedding_client, vector_store: ChromaVectorStore) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalCandidate]:
        query_embedding = embed_query_text(self.embedding_client, query)
        if not query_embedding:
            return []
        matches = self.vector_store.query(query_embedding, top_k=top_k, where=where)
        return [
            RetrievalCandidate(
                chunk_id=match.vector_id, text=match.document_text, metadata=match.metadata,
                dense_rank=rank, dense_score=round(match.similarity, 4),
            )
            for rank, match in enumerate(matches, start=1)
        ]


class BM25Retriever:
    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalCandidate]:
        # Over-fetch when filtering post-hoc, so a metadata filter
        # doesn't just shrink an already-small top_k down to nothing --
        # BM25 has no native filtered-search API the way Chroma does.
        results = self.index.query(query, top_k=top_k * 5 if where else top_k)
        if where:
            results = [(chunk, score) for chunk, score in results if all(chunk.metadata.get(k) == v for k, v in where.items())]
        results = results[:top_k]
        return [
            RetrievalCandidate(
                chunk_id=chunk.vector_id, text=chunk.text, metadata=chunk.metadata,
                bm25_rank=rank, bm25_score=round(score, 4),
            )
            for rank, (chunk, score) in enumerate(results, start=1)
        ]


def reciprocal_rank_fusion(
    dense_results: list[RetrievalCandidate], bm25_results: list[RetrievalCandidate], k: int = 60,
) -> list[RetrievalCandidate]:
    """Fuses two ranked lists by RRF, deduplicating by chunk_id -- a
    chunk found by both retrievers is merged into ONE candidate
    carrying both sets of rank/score information, not two."""
    merged: dict[str, RetrievalCandidate] = {}

    for candidate in dense_results:
        merged[candidate.chunk_id] = candidate
        candidate.hybrid_score += 1.0 / (k + candidate.dense_rank)

    for candidate in bm25_results:
        existing = merged.get(candidate.chunk_id)
        if existing is not None:
            existing.bm25_rank = candidate.bm25_rank
            existing.bm25_score = candidate.bm25_score
            existing.hybrid_score += 1.0 / (k + candidate.bm25_rank)
        else:
            candidate.hybrid_score += 1.0 / (k + candidate.bm25_rank)
            merged[candidate.chunk_id] = candidate

    return sorted(merged.values(), key=lambda c: c.hybrid_score, reverse=True)


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, dense_top_k: int = 20, bm25_top_k: int = 20, final_top_k: int = 5, where: dict | None = None,
    ) -> list[RetrievalCandidate]:
        dense_results = self.dense.retrieve(query, top_k=dense_top_k, where=where)
        bm25_results = self.bm25.retrieve(query, top_k=bm25_top_k, where=where)
        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=self.rrf_k)
        return fused[:final_top_k]
