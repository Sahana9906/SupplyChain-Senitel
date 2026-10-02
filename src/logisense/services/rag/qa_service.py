"""Evidence-grounded Q&A over the internal knowledge base.

Retrieval and generation are deliberately separate steps: vector search
finds candidate chunks first, and the LLM is instructed to answer using
ONLY those chunks -- never its own general knowledge -- and to say so
explicitly when the retrieved evidence is insufficient. Every claim in
the answer must be traceable to a specific retrieved chunk (document +
category + filename), which the caller can display as a citation.

Entity matching is mandatory, not a ranking nudge: if the question names
a specific supplier/contract (e.g. "SUP-104"), retrieval is hard-filtered
to that entity's own documents. A generically-similar document about a
DIFFERENT supplier is never cited as evidence for this one, no matter
how high its embedding similarity score is -- similarity alone is not
sufficient grounds for a citation when the question is about a named
entity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import groq

from logisense.config import Settings
from logisense.services.gemini_client import GroqClient, build_gemini_client
from logisense.services.rag.embeddings import EmbeddingClient, embed_query_text
from logisense.services.rag.entity_extraction import extract_entities
from logisense.services.rag.vector_store import ChromaVectorStore, VectorMatch

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are answering questions using ONLY the evidence passages provided below.

Rules:
1. Base your answer strictly on the provided evidence. Do not use outside knowledge, do not invent policies, contract terms, or facts not present in the evidence.
2. If the evidence does not contain enough information to answer the question, say so explicitly (e.g. "The knowledge base does not contain sufficient evidence to answer this."). Do not guess or fill gaps.
3. When you state a fact, cite it using the exact bracketed source shown for that passage (e.g. [DOC: category/filename.md]) so the user can see exactly where each claim came from.
4. Keep the answer factual and concise. Do not speculate about causes or recommend actions unless the evidence itself explicitly states them.
"""


@dataclass(frozen=True)
class Citation:
    reference_number: int
    document_id: int
    category: str
    filename: str
    title: str
    chunk_text: str
    relevance_score: float
    section: str | None = None

    @property
    def citation_text(self) -> str:
        """Explicit, traceable source marker -- e.g.
        "[DOC: supplier_contracts/SUP-014_contract.md]" or, when the
        chunk came from a specific clause/section,
        "[DOC: supplier_contracts/SUP-014_contract.md, section: Penalty Terms]".
        This is what should actually be shown to the user, not just the
        internal reference number."""
        section_part = f", section: {self.section}" if self.section else ""
        return f"[DOC: {self.category}/{self.filename}{section_part}]"


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sufficient_evidence: bool = True


class EvidenceGroundedQAService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: ChromaVectorStore,
        settings: Settings,
        top_k: int = 5,
        min_similarity: float = 0.25,
        gemini_client: GroqClient | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.settings = settings
        self.top_k = top_k
        self.min_similarity = min_similarity
        self._client = gemini_client

    def _gemini_client(self) -> GroqClient:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set to generate grounded answers")
            self._client = build_gemini_client(self.settings)
        return self._client

    def answer(self, question: str) -> GroundedAnswer:
        query_embedding = embed_query_text(self.embedding_client, question)
        if not query_embedding:
            return GroundedAnswer(question=question, answer="Unable to generate an embedding for the question.", sufficient_evidence=False)

        # Entity matching is mandatory: if the question names a specific
        # supplier, retrieval is hard-filtered to that supplier's own
        # documents -- a document about a different supplier is never
        # retrieved as a candidate for this question, regardless of how
        # semantically similar its text happens to be.
        entities = extract_entities(question)
        where = {"supplier_id": entities.primary_supplier_id} if entities.primary_supplier_id else None

        matches = self.vector_store.query(query_embedding, top_k=self.top_k, where=where)
        matches = [m for m in matches if m.similarity >= self.min_similarity]

        if not matches:
            missing_entity_note = (
                f" No documents were found for {entities.primary_supplier_id} specifically."
                if entities.primary_supplier_id else ""
            )
            return GroundedAnswer(
                question=question,
                answer=f"The knowledge base does not contain sufficient evidence to answer this question.{missing_entity_note}",
                citations=[], sufficient_evidence=False,
            )

        citations = self._build_citations(matches)
        prompt = self._build_prompt(question, citations)

        client = self._gemini_client()
        try:
            response = client.models.generate_content(model=self.settings.groq_model, contents=prompt)
        except groq.APIError as exc:
            logger.warning("QA call failed after all retry attempts (HTTP %s): %s", getattr(exc, "status_code", None), exc)
            raise
        answer_text = (response.text or "").strip()

        sufficient = "does not contain sufficient evidence" not in answer_text.lower()
        return GroundedAnswer(question=question, answer=answer_text, citations=citations, sufficient_evidence=sufficient)

    @staticmethod
    def _build_citations(matches: list[VectorMatch]) -> list[Citation]:
        citations = []
        for i, match in enumerate(matches, start=1):
            meta = match.metadata
            citations.append(Citation(
                reference_number=i, document_id=meta.get("document_id"), category=meta.get("category", "unknown"),
                filename=meta.get("filename", "unknown"), title=meta.get("title", "Untitled"),
                chunk_text=match.document_text, relevance_score=round(match.similarity, 3),
                section=meta.get("section") or None,
            ))
        return citations

    @staticmethod
    def _build_prompt(question: str, citations: list[Citation]) -> str:
        evidence_block = "\n\n".join(
            f"{c.citation_text} (\"{c.title}\")\n{c.chunk_text}"
            for c in citations
        )
        return f"{SYSTEM_INSTRUCTIONS}\n\nEVIDENCE:\n{evidence_block}\n\nQUESTION: {question}\n\nANSWER:"
