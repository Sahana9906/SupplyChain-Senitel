"""Extracts real entity IDs (SUP-014, PO-04370, SHIP-01203, CTR-014,
CAR-07, SKU-00042) from free text -- used two ways:

1. When indexing a document, to tag each chunk with which real entities
   it's actually about (supplier_id/contract_id/etc in metadata).
2. When a user asks a question, to figure out which structured records
   to pull from the database and which document chunks to prioritize --
   the mechanism hybrid retrieval is built on.

Pure regex, no LLM call -- fast, deterministic, and exact (an LLM might
paraphrase an ID; regex either finds the literal string or it doesn't).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS = {
    "supplier_id": re.compile(r"\bSUP-\d+\b"),
    "contract_id": re.compile(r"\bCTR-\d+\b"),
    "purchase_order_id": re.compile(r"\bPO-\d+\b"),
    "shipment_id": re.compile(r"\bSHIP-\d+\b"),
    "carrier_id": re.compile(r"\bCAR-\d+\b"),
    "sku_id": re.compile(r"\bSKU-\d+\b"),
    "customer_order_id": re.compile(r"\bORD-\d+\b"),
}


@dataclass(frozen=True)
class ExtractedEntities:
    supplier_ids: list[str] = field(default_factory=list)
    contract_ids: list[str] = field(default_factory=list)
    purchase_order_ids: list[str] = field(default_factory=list)
    shipment_ids: list[str] = field(default_factory=list)
    carrier_ids: list[str] = field(default_factory=list)
    sku_ids: list[str] = field(default_factory=list)
    customer_order_ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.supplier_ids, self.contract_ids, self.purchase_order_ids,
            self.shipment_ids, self.carrier_ids, self.sku_ids, self.customer_order_ids,
        ])

    @property
    def primary_supplier_id(self) -> str | None:
        return self.supplier_ids[0] if self.supplier_ids else None


def extract_entities(text: str) -> ExtractedEntities:
    if not text:
        return ExtractedEntities()

    def unique_ordered(pattern: re.Pattern) -> list[str]:
        seen: list[str] = []
        for match in pattern.finditer(text):
            if match.group(0) not in seen:
                seen.append(match.group(0))
        return seen

    return ExtractedEntities(
        supplier_ids=unique_ordered(_PATTERNS["supplier_id"]),
        contract_ids=unique_ordered(_PATTERNS["contract_id"]),
        purchase_order_ids=unique_ordered(_PATTERNS["purchase_order_id"]),
        shipment_ids=unique_ordered(_PATTERNS["shipment_id"]),
        carrier_ids=unique_ordered(_PATTERNS["carrier_id"]),
        sku_ids=unique_ordered(_PATTERNS["sku_id"]),
        customer_order_ids=unique_ordered(_PATTERNS["customer_order_id"]),
    )
