"""LLM-powered investigation answering, grounded in hybrid context
(structured database facts + retrieved document evidence).

The prompt makes the model treat the two evidence kinds differently:
structured facts are presented as authoritative database records (exact,
not paraphrased); document evidence is presented as retrieved passages
that may be prose/policy language. The model is told to answer using
ONLY what's provided and to say so explicitly when neither source has
enough to answer -- this is the actual anti-hallucination mechanism,
not just an instruction: if HybridContext.has_any_evidence is False,
the LLM is never even called.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import groq

from logisense.config import Settings
from logisense.services.gemini_client import GroqClient, build_gemini_client
from logisense.services.rag.hybrid_context import HybridContext, HybridContextBuilder, StructuredFact
from logisense.services.rag.vector_store import VectorMatch

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are answering a supply-chain question using ONLY the evidence provided below.

Evidence comes in two kinds:
- DATABASE RECORDS: exact, authoritative facts from the operational database (suppliers, purchase orders, shipments, inventory, contracts). Treat these as ground truth.
- DOCUMENT PASSAGES: retrieved excerpts from contracts, policies, correspondence, and historical records. These are prose -- quote or closely paraphrase them, don't invent additional detail.

Rules:
1. Base your answer strictly on the evidence below. Do not use outside knowledge. Do not invent supplier names, dates, quantities, contract terms, or policies not present in the evidence.
2. If the evidence does not contain enough information to answer, say so explicitly (e.g. "The available evidence does not contain sufficient information to answer this."). Do not guess or fill gaps.
3. When you state a fact, cite it using the exact bracketed source shown for that item (e.g. [DB: table.record_id] or [DOC: category/filename.md]) so the user can see exactly where each claim came from.
4. Keep the answer factual and concise. Do not recommend actions or speculate about root causes unless the evidence itself explicitly states them.
"""


@dataclass(frozen=True)
class EvidenceCitation:
    reference_number: int
    kind: str  # "database" | "document"
    label: str
    source_table: str | None = None
    source_id: str | None = None
    category: str | None = None
    filename: str | None = None
    section: str | None = None
    text: str | None = None
    relevance_score: float | None = None

    @property
    def citation_text(self) -> str:
        """Explicit, traceable source marker for display -- e.g.
        "[DB: purchase_orders.PO-04370]" or
        "[DOC: supplier_contracts/SUP-014_contract.md, section: Penalty Terms]"."""
        if self.kind == "database":
            return f"[DB: {self.source_table}.{self.source_id}]"
        section_part = f", section: {self.section}" if self.section else ""
        return f"[DOC: {self.category}/{self.filename}{section_part}]"


@dataclass(frozen=True)
class HybridAnswer:
    question: str
    answer: str
    citations: list[EvidenceCitation] = field(default_factory=list)
    sufficient_evidence: bool = True


class HybridInvestigationService:
    def __init__(
        self,
        context_builder: HybridContextBuilder,
        settings: Settings,
        gemini_client: GroqClient | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.settings = settings
        self._client = gemini_client

    def _gemini_client(self) -> GroqClient:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set to generate grounded answers")
            self._client = build_gemini_client(self.settings)
        return self._client

    def answer(self, question: str) -> HybridAnswer:
        context = self.context_builder.build(question)

        if not context.has_any_evidence:
            return HybridAnswer(
                question=question,
                answer="The available evidence does not contain sufficient information to answer this question. "
                       "No matching database records or document passages were found.",
                citations=[], sufficient_evidence=False,
            )

        citations = self._build_citations(context)
        prompt = self._build_prompt(context, citations)

        client = self._gemini_client()
        try:
            response = client.models.generate_content(model=self.settings.groq_model, contents=prompt)
        except groq.APIError as exc:
            logger.warning("Hybrid-investigation call failed after all retry attempts (HTTP %s): %s", getattr(exc, "status_code", None), exc)
            raise
        answer_text = (response.text or "").strip()

        sufficient = "does not contain sufficient" not in answer_text.lower()
        return HybridAnswer(question=question, answer=answer_text, citations=citations, sufficient_evidence=sufficient)

    @staticmethod
    def _build_citations(context: HybridContext) -> list[EvidenceCitation]:
        citations: list[EvidenceCitation] = []
        n = 1
        for fact in context.structured_facts:
            citations.append(EvidenceCitation(
                reference_number=n, kind="database", label=fact.label,
                source_table=fact.source_table, source_id=fact.source_id, text=fact.value,
            ))
            n += 1
        for match in context.document_matches:
            meta = match.metadata
            citations.append(EvidenceCitation(
                reference_number=n, kind="document", label=meta.get("title", "Untitled"),
                category=meta.get("category"), filename=meta.get("filename"), section=meta.get("section") or None,
                text=match.document_text, relevance_score=round(match.similarity, 3),
            ))
            n += 1
        return citations

    @staticmethod
    def _build_prompt(context: HybridContext, citations: list[EvidenceCitation]) -> str:
        blocks = []
        for c in citations:
            kind_label = "DATABASE RECORD" if c.kind == "database" else "DOCUMENT PASSAGE"
            blocks.append(f"{c.citation_text} {kind_label}\n{c.text}")
        evidence_block = "\n\n".join(blocks)
        return f"{SYSTEM_INSTRUCTIONS}\n\nEVIDENCE:\n{evidence_block}\n\nQUESTION: {context.question}\n\nANSWER:"
