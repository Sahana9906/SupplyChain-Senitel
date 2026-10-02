"""GDELT connector -- external events/news intelligence. Free, public
API, no key required.

Real GDELT DOC 2.0 API (documented, public):
    https://api.gdeltproject.org/api/v2/doc/doc

NOTE: same network restriction as weather_connector.py -- this sandbox
cannot reach api.gdeltproject.org (confirmed 403). Implemented against
GDELT's real, documented query/response format and unit-tested against
mocked responses; not exercised live from within this sandbox.
"""
from __future__ import annotations

import httpx

from logisense.models import ProviderStatus
from logisense.services.external.evidence_item import EvidenceItem

_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTConnectorResult:
    def __init__(self, status: ProviderStatus, items: list[EvidenceItem], error_message: str | None = None) -> None:
        self.status = status
        self.items = items
        self.error_message = error_message


class GDELTConnector:
    def __init__(self, timeout_seconds: float = 10.0, max_records: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_records = max_records

    def search_events(self, entity: str, extra_terms: list[str] | None = None) -> GDELTConnectorResult:
        """entity should be a real, specific name -- a supplier's actual
        company name or a real port/city, not a generic term. Do not
        just dump every returned article as evidence: only articles
        whose title genuinely mentions the entity are kept."""
        terms = [entity] + (extra_terms or [])
        query = " ".join(f'"{t}"' if " " in t else t for t in terms)

        try:
            response = httpx.get(
                _GDELT_DOC_URL,
                params={"query": query, "mode": "artlist", "format": "json", "maxrecords": self.max_records, "sort": "hybridrel"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            return GDELTConnectorResult(ProviderStatus.ERROR, [], f"GDELT request timed out: {exc}")
        except httpx.HTTPError as exc:
            return GDELTConnectorResult(ProviderStatus.ERROR, [], f"GDELT request failed: {exc}")
        except ValueError as exc:
            return GDELTConnectorResult(ProviderStatus.ERROR, [], f"GDELT returned a malformed response: {exc}")

        articles = data.get("articles", [])
        if not articles:
            return GDELTConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], f"No GDELT articles found for '{entity}'.")

        # Relevance filter: only keep articles that genuinely mention the
        # entity by name -- GDELT's own relevance sort is not a
        # substitute for checking the entity actually appears, since a
        # broad query can still surface tangentially-related results.
        items = []
        for article in articles:
            title = article.get("title", "")
            if entity.lower() not in title.lower():
                continue
            items.append(EvidenceItem(
                source_type="gdelt", source_name="GDELT", source_id=article.get("url", title),
                content=title, title=title, entity=entity, timestamp=article.get("seendate"),
                url=article.get("url"), confidence=0.6,  # news relevance is inherently less certain than a DB record
                metadata={"domain": article.get("domain"), "language": article.get("language")},
            ))

        if not items:
            return GDELTConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], f"GDELT returned results, but none genuinely mention '{entity}' by name.")

        return GDELTConnectorResult(ProviderStatus.AVAILABLE, items)
