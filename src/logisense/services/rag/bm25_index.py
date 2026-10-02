"""BM25 lexical retrieval over the SAME canonical chunks stored in
Chroma (built by reading Chroma's own contents via
ChromaVectorStore.get_all() -- never a separately-chunked corpus).

Tokenization is deliberately simple (lowercase, split on anything that
isn't alphanumeric-or-hyphen) so that hyphenated business identifiers
like "SUP-104", "PO-83921", "7-day" survive as single tokens -- this is
exactly what makes BM25 useful here: exact identifier matches that
semantic/dense retrieval can miss or dilute.

Persisted to a single pickle file (rank_bm25's BM25Okapi is a small,
plain-Python object -- no need for a database table or a second
service). Rebuilt in full every time the knowledge base is re-indexed,
never incrementally -- this guarantees it can never silently drift out
of sync with whatever Chroma currently holds, at the cost of an
O(corpus size) rebuild, which is cheap at this corpus's scale (hundreds
of chunks, not millions).
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class IndexedChunk:
    """One canonical chunk as BM25 sees it -- vector_id is the SAME id
    Chroma uses (e.g. "document:5:chunk:2"), so a BM25 result and a
    Chroma result for the same chunk can always be matched up exactly."""

    vector_id: str
    text: str
    metadata: dict


class BM25Index:
    def __init__(self, chunks: list[IndexedChunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def build_from_vector_store(cls, vector_store) -> "BM25Index":
        entries = vector_store.get_all()
        chunks = [IndexedChunk(vector_id=vid, text=text, metadata=meta) for vid, text, meta in entries]
        return cls(chunks)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No BM25 index found at '{path}'. Run rag-index-knowledge first to build it."
            )
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        return cls(chunks)

    def query(self, text: str, top_k: int = 20) -> list[tuple[IndexedChunk, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # NOTE: BM25's standard IDF term can go negative for terms that
        # appear in most/all documents in a small corpus -- this is a
        # real, documented property of the formula, not a sign that
        # there's "no match". RRF fusion only uses rank order, so we
        # return the top_k by rank regardless of raw score sign, rather
        # than filtering on score > 0 (which would silently drop
        # legitimate top matches in small corpora).
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices[:top_k]]
