"""Multi-source grounded investigation: combines internal database
facts, internal knowledge-base documents (both via the existing
HybridContextBuilder -- reused, not duplicated), and external APIs
(Open-Meteo weather, GDELT news/events, AISstream.io maritime data)
into ONE evidence-grounded answer with explicit, traceable citations.

Evidence is collected from every available source BEFORE the LLM is
asked anything. The LLM never answers first and gets citations bolted
on afterward -- if there is no evidence at all, the LLM is never called.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import groq
from sqlalchemy import select
from sqlalchemy.orm import Session

from logisense import models as m
from logisense.config import Settings
from logisense.services.external.evidence_item import EvidenceItem
from logisense.services.external.gdelt_connector import GDELTConnector
from logisense.services.external.maritime_connector import AISStreamConnector
from logisense.services.external.weather_connector import OpenMeteoConnector
from logisense.services.gemini_client import GroqClient, build_gemini_client
from logisense.services.rag.hybrid_context import HybridContextBuilder

logger = logging.getLogger(__name__)

_WEATHER_KEYWORDS = re.compile(r"\b(weather|storm|rain|snow|flood|wind|hurricane|typhoon|delay|disrupt|transport|route|climate)\b", re.IGNORECASE)
_MARITIME_KEYWORDS = re.compile(r"\b(ship|vessel|port|ocean|maritime|sea|cargo)\b", re.IGNORECASE)

_SYSTEM_INSTRUCTIONS = """You are answering a supply-chain question using ONLY the evidence provided below, gathered from internal company records, internal documents, and external real-world data sources.

Rules:
1. Answer only using the supplied evidence. Do not invent missing facts.
2. Clearly distinguish stated facts from your own inference.
3. If evidence is insufficient to answer, say so explicitly.
4. Do not treat an external event (weather, news) as the cause of an operational problem unless it is both temporally and geographically relevant to the specific supplier/shipment/PO in question.
5. Do not treat correlation as proof of causation.
6. Prefer exact entity matches (the specific supplier/PO named in the question) over general or tangential evidence.
7. Cite every important factual claim using the exact bracketed source shown for that item (e.g. [DB: table.id], [DOC: category/file.md], [EXTERNAL: Open-Meteo — location — time]).
8. Use internal database facts as authoritative for operational values (dates, quantities, status). Use internal documents for contractual/procedural context. Use external APIs only for real-world conditions (weather, events, vessel data) -- never as a substitute for an internal fact you don't have.
"""


@dataclass(frozen=True)
class MultiSourceGroundedAnswer:
    question: str
    answer: str
    citations: list[EvidenceItem] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    sufficient_evidence: bool = True


class MultiSourceInvestigationService:
    def __init__(
        self,
        session: Session,
        context_builder: HybridContextBuilder,
        settings: Settings,
        weather_connector: OpenMeteoConnector | None = None,
        gdelt_connector: GDELTConnector | None = None,
        maritime_connector: AISStreamConnector | None = None,
        gemini_client: GroqClient | None = None,
    ) -> None:
        self.session = session
        self.context_builder = context_builder
        self.settings = settings
        self.weather_connector = weather_connector or OpenMeteoConnector()
        self.gdelt_connector = gdelt_connector or GDELTConnector()
        self.maritime_connector = maritime_connector or AISStreamConnector(settings)
        self._client = gemini_client

    def _gemini_client(self) -> GroqClient:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set to generate grounded answers")
            self._client = build_gemini_client(self.settings)
        return self._client

    def answer(self, question: str) -> MultiSourceGroundedAnswer:
        source_status: dict[str, str] = {}

        # -- Internal DB + internal RAG: reuse HybridContextBuilder, ----
        # -- which already does entity-filtered retrieval. -----------------
        internal_context = self.context_builder.build(question)
        source_status["internal_db"] = "available" if internal_context.structured_facts else "no_evidence_found"
        source_status["internal_rag"] = "available" if internal_context.document_matches else "no_evidence_found"

        evidence: list[EvidenceItem] = []
        for fact in internal_context.structured_facts:
            evidence.append(EvidenceItem(
                source_type="internal_db", source_name=fact.source_table, source_id=fact.source_id,
                content=fact.value, confidence=0.95,
            ))
        for match in internal_context.document_matches:
            meta = match.metadata
            evidence.append(EvidenceItem(
                source_type="internal_document", source_name=meta.get("category", "unknown"), source_id=meta.get("filename", "unknown"),
                content=match.document_text, title=meta.get("title"), relevance_score=round(match.similarity, 3),
                confidence=0.75, metadata=dict(meta),
            ))

        # -- Derive external retrieval parameters from the internal DB ----
        # -- (entity, location, time window) -- never ask the user for ----
        # -- something already known internally. ---------------------------
        entities = internal_context.entities
        location_name = None
        entity_name = None
        if entities.primary_supplier_id:
            supplier = self.session.scalar(select(m.Supplier).where(m.Supplier.source_supplier_id == entities.primary_supplier_id))
            if supplier:
                entity_name = supplier.name
                if supplier.location:
                    location_name = ", ".join(filter(None, [supplier.location.city, supplier.location.country]))

        # -- External: weather ---------------------------------------------
        if location_name and _WEATHER_KEYWORDS.search(question):
            result = self.weather_connector.get_weather_for_location(location_name)
            source_status["open_meteo"] = result.status.value
            evidence.extend(result.items)
        else:
            source_status["open_meteo"] = "not_queried"

        # -- External: maritime ----------------------------------------------
        if location_name and _MARITIME_KEYWORDS.search(question):
            result = self.maritime_connector.get_vessel_activity_near(location_name)
            source_status["aisstream"] = result.status.value
            evidence.extend(result.items)
        else:
            source_status["aisstream"] = "not_queried"

        # -- External: GDELT ---------------------------------------------
        if entity_name or entities.primary_supplier_id:
            result = self.gdelt_connector.search_events(entity_name or entities.primary_supplier_id)
            source_status["gdelt"] = result.status.value
            evidence.extend(result.items)
        else:
            source_status["gdelt"] = "not_queried"

        if not evidence:
            return MultiSourceGroundedAnswer(
                question=question,
                answer="No internal or external evidence could be found to answer this question.",
                citations=[], source_status=source_status, sufficient_evidence=False,
            )

        evidence.sort(key=lambda e: (e.ranking_priority, -(e.relevance_score or e.confidence or 0)))

        prompt = self._build_prompt(question, evidence, source_status)
        try:
            response = self._gemini_client().models.generate_content(model=self.settings.groq_model, contents=prompt)
        except groq.APIError as exc:
            logger.warning("Multi-source-investigation call failed after all retry attempts (HTTP %s): %s", getattr(exc, "status_code", None), exc)
            raise
        answer_text = (response.text or "").strip()
        sufficient = "insufficient" not in answer_text.lower()[:400]  # only checks near the start, avoiding false positives from later hedging language

        return MultiSourceGroundedAnswer(
            question=question, answer=answer_text, citations=evidence,
            source_status=source_status, sufficient_evidence=sufficient,
        )

    @staticmethod
    def _build_prompt(question: str, evidence: list[EvidenceItem], source_status: dict[str, str]) -> str:
        blocks = [f"{item.citation_text}\n{item.content}" for item in evidence]
        evidence_block = "\n\n".join(blocks)
        unavailable = [f"{k}: {v}" for k, v in source_status.items() if v in ("unavailable", "error")]
        unavailable_note = f"\n\nSOURCES NOT AVAILABLE FOR THIS ANALYSIS: {', '.join(unavailable)}" if unavailable else ""
        return f"{_SYSTEM_INSTRUCTIONS}\n\nEVIDENCE:\n{evidence_block}{unavailable_note}\n\nQUESTION: {question}\n\nANSWER:"
