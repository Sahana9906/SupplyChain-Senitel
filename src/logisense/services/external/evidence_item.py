"""One canonical evidence structure shared by every source this project
can retrieve from -- internal database facts, internal knowledge-base
documents, and external APIs (weather, maritime, news/events). Every
piece of evidence, regardless of where it came from, becomes exactly
one EvidenceItem, so ranking, LLM prompting, and citation display never
need to special-case the source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["internal_db", "internal_document", "weather_api", "maritime_api", "gdelt"]


@dataclass(frozen=True)
class EvidenceItem:
    source_type: SourceType
    source_name: str
    source_id: str
    content: str
    title: str | None = None
    entity: str | None = None
    timestamp: str | None = None
    location: str | None = None
    relevance_score: float | None = None
    url: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def citation_text(self) -> str:
        """Explicit, traceable source marker for display -- tells the
        user exactly WHERE this claim came from, in the format required
        for each source type. Never a vague "according to the knowledge
        base" -- always a concrete, checkable path."""
        if self.source_type == "internal_db":
            return f"[DB: {self.source_name}.{self.source_id}]"
        if self.source_type == "internal_document":
            return f"[DOC: {self.source_name}/{self.source_id}]"
        if self.source_type == "weather_api":
            return f"[EXTERNAL: Open-Meteo — {self.location} — {self.timestamp}]"
        if self.source_type == "maritime_api":
            return f"[EXTERNAL: AISstream.io — {self.entity or self.location} — {self.timestamp}]"
        if self.source_type == "gdelt":
            return f"[EXTERNAL: GDELT — {self.title} — {self.timestamp}]"
        return f"[{self.source_type}: {self.source_name}]"

    @property
    def ranking_priority(self) -> int:
        """Lower is higher priority. Exact internal DB facts rank above
        directly-matching internal documents, which rank above external
        evidence -- matches the required priority order (structured
        facts are authoritative; external evidence is corroborating
        context, not the primary source of truth)."""
        return {"internal_db": 0, "internal_document": 1, "weather_api": 2, "maritime_api": 2, "gdelt": 2}.get(self.source_type, 3)
