"""Local vector store, backed by ChromaDB (no server required -- runs as
an embedded library, persisted to disk). This is the actual embedding +
ANN similarity search layer of the RAG pipeline; Postgres (Document/
DocumentChunk) remains the provenance/traceability record.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from logisense.config import Settings


@dataclass(frozen=True)
class VectorMatch:
    vector_id: str
    document_text: str
    metadata: dict
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine distance -> a rough [0, 1] similarity score. Not a
        calibrated probability -- a monotonic proxy for ranking and the
        min-similarity cutoff."""
        return max(0.0, 1.0 - self.distance)


class ChromaVectorStore:
    def __init__(self, settings: Settings, client: chromadb.ClientAPI | None = None) -> None:
        self.client = client or chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            settings.chroma_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[VectorMatch]:
        if self.collection.count() == 0:
            return []
        kwargs = {"query_embeddings": [query_embedding], "n_results": min(top_k, self.collection.count())}
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            VectorMatch(vector_id=i, document_text=d, metadata=meta or {}, distance=dist)
            for i, d, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    def delete_by_document(self, document_id: int) -> None:
        self.collection.delete(where={"document_id": document_id})

    def get_all(self) -> list[tuple[str, str, dict]]:
        """Returns (vector_id, document_text, metadata) for every chunk
        currently in this collection. Used to build the BM25 index over
        the EXACT SAME canonical chunks Chroma holds -- reading straight
        from Chroma (rather than re-reading Postgres or re-chunking
        files separately) guarantees both retrieval systems operate on
        byte-identical text and metadata, with no risk of drift."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        return list(zip(result["ids"], result["documents"], result["metadatas"]))

    def ensure_compatible_dimension(self, expected_dim: int) -> bool:
        """Checks whether existing vectors in this collection match the
        dimensionality the CURRENT embedding model produces. If a
        previous embedding model (different dimensionality) populated
        this collection, its vectors are structurally incompatible with
        new ones -- Chroma would either error or silently corrupt
        similarity search. Rather than mixing vectors from different
        models, this deletes and recreates the collection cleanly.

        Returns True if the collection was reset, False if it was
        already empty or already compatible."""
        if self.collection.count() == 0:
            return False

        peek = self.collection.peek(limit=1)
        existing_embeddings = peek.get("embeddings")
        if existing_embeddings is None or len(existing_embeddings) == 0:
            return False

        existing_dim = len(existing_embeddings[0])
        if existing_dim == expected_dim:
            return False

        name = self.collection.name
        metadata = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name, metadata=metadata)
        return True
