# Company Knowledge Base

A small, curated internal-document corpus for the SupplyChain Sentinel
GenAI agent to retrieve evidence from. Generated from the real IDs already
present in `company_data/*.csv` -- every supplier, PO, shipment, and
carrier referenced below actually exists in that dataset.

This corpus is deliberately small (quality and cross-document
consistency over volume): roughly 5-10 documents per category rather
than hundreds.

## Critical: no predefined incidents

No document here encodes a root cause, an expected answer, or a
recommended mitigation. Policy/SOP/agreement documents describe
standing business rules, not a response to any specific event.
Historical incident records describe REAL statistical outliers already
present in `inbound_shipments.csv` (the slowest ~3% of shipments by
natural, unscripted delay) -- they report what was observed, what was
checked, and what evidence exists on file, without asserting a single
definitive cause or prescribing what should be done differently. Sentinel
has to reason over this evidence itself.

## Categories

| Category | Documents |
|---|---|
| `supplier_contracts/` | 8 |
| `procurement_policies/` | 7 |
| `shipping_policies/` | 6 |
| `warehouse_sops/` | 6 |
| `supplier_communications/` | 8 |
| `procurement_notes/` | 6 |
| `historical_incidents/` | 8 |
| `supplier_performance_reviews/` | 8 |
| `logistics_agreements/` | 8 |
| `escalation_procedures/` | 6 |

## Regenerating

```bash
python generate_knowledge_base.py
```

Deterministic (fixed seed) -- re-running produces the same document
set. Reads `company_data/*.csv` only; never modifies it.
