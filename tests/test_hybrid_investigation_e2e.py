"""End-to-end tests over the REAL company_data/ CSVs and REAL
company_knowledge/ documents -- not tiny synthetic fixtures. Proves the
whole hybrid pipeline (CSV load -> knowledge index -> hybrid retrieval
-> grounded answer -> citations) works together on the actual dataset
this project ships with.

Uses a deterministic word-hash embedding client (meaningful similarity
without a live Gemini call) and a fake Gemini client for the generation
step (no live network needed, consistent with every other test in this
project).
"""
import hashlib
import os
from dataclasses import dataclass

import pytest

from logisense.services.erp.csv_loader import load_company_data
from logisense.services.rag.hybrid_context import HybridContextBuilder
from logisense.services.rag.hybrid_investigation import HybridInvestigationService
from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMPANY_DATA_DIR = os.path.join(_REPO_ROOT, "company_data")
_KNOWLEDGE_DIR = os.path.join(_COMPANY_DATA_DIR, "company_knowledge")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_COMPANY_DATA_DIR), reason="company_data/ not present in this checkout"
)


def _word_hash_embed(text: str, dim: int = 48) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


class WordHashEmbeddingClient:
    model_name = "word-hash-test-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_word_hash_embed(t) for t in texts]


@dataclass
class _FakeGenAIResponse:
    text: str


class FakeGeminiModels:
    def __init__(self):
        self.last_prompt: str | None = None

    def generate_content(self, model, contents):
        self.last_prompt = contents
        # Echo back a plausible grounded-sounding answer referencing the
        # evidence numbers actually present in the prompt, so tests can
        # assert on citation wiring without needing a live LLM call.
        return _FakeGenAIResponse(text="Based on the evidence provided: see [1] for details.")


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()


@pytest.fixture(scope="module")
def loaded_session_and_store():
    """Loads the REAL CSVs and indexes the REAL knowledge base ONCE per
    test module (both are moderately large -- 5000 POs, 71 documents --
    so this is shared across tests rather than repeated per-test)."""
    import chromadb
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from logisense.config import Settings
    from logisense.db import Base
    from logisense.services.rag.vector_store import ChromaVectorStore

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True, expire_on_commit=False)()

    load_company_data(session, _COMPANY_DATA_DIR)

    settings = Settings(database_url="sqlite:///:memory:", gemini_api_key="x", chroma_collection_name="e2e-test-hybrid")
    vector_store = ChromaVectorStore(settings, client=chromadb.EphemeralClient())
    KnowledgeBaseIndexer(session, WordHashEmbeddingClient(), vector_store).index_directory(_KNOWLEDGE_DIR)

    yield session, vector_store
    session.close()
    engine.dispose()


def _make_service(session, vector_store) -> tuple[HybridInvestigationService, FakeGeminiClient]:
    from logisense.config import Settings

    settings = Settings(database_url="sqlite:///:memory:", gemini_api_key="x")
    context_builder = HybridContextBuilder(
        session, WordHashEmbeddingClient(), vector_store, top_k=6, min_similarity=0.0
    )
    fake_gemini = FakeGeminiClient()
    service = HybridInvestigationService(context_builder, settings, gemini_client=fake_gemini)
    return service, fake_gemini


# ---------------------------------------------------------------------------
# Supplier disruption
# ---------------------------------------------------------------------------

def test_supplier_disruption_question_returns_real_structured_facts_and_citations(loaded_session_and_store):
    session, vector_store = loaded_session_and_store
    service, fake_gemini = _make_service(session, vector_store)

    result = service.answer("What is happening with supplier SUP-014? Are there any delayed shipments or open purchase orders?")

    assert result.sufficient_evidence is True
    database_citations = [c for c in result.citations if c.kind == "database"]
    assert any(c.source_table == "suppliers" and c.source_id == "SUP-014" for c in database_citations)
    assert any(c.source_table == "purchase_orders" for c in database_citations)

    # The evidence sent to the LLM must contain the real facts, not a
    # paraphrase -- e.g. a real PO reference number for this supplier.
    assert "PO-" in fake_gemini.models.last_prompt
    assert "SUP-014" in fake_gemini.models.last_prompt


def test_supplier_disruption_question_retrieves_that_suppliers_own_contract(loaded_session_and_store):
    session, vector_store = loaded_session_and_store
    service, _ = _make_service(session, vector_store)

    result = service.answer("Does supplier SUP-014 have any contractual delivery obligations?")

    document_citations = [c for c in result.citations if c.kind == "document"]
    assert any(c.filename == "SUP-014_contract.md" for c in document_citations), (
        f"Expected SUP-014's own contract in results, got: {[c.filename for c in document_citations]}"
    )


# ---------------------------------------------------------------------------
# PO impact
# ---------------------------------------------------------------------------

def test_po_impact_question_returns_po_shipment_and_inventory_facts(loaded_session_and_store):
    session, vector_store = loaded_session_and_store

    from sqlalchemy import select
    from logisense import models as m
    # Pick a real PO that has both a shipment AND at least one line item
    # (products link to a PO via PurchaseOrderItem, not a direct FK on
    # the PO header itself).
    po = session.scalar(
        select(m.PurchaseOrder)
        .join(m.Shipment, m.Shipment.purchase_order_id == m.PurchaseOrder.id)
        .join(m.PurchaseOrderItem, m.PurchaseOrderItem.purchase_order_id == m.PurchaseOrder.id)
        .limit(1)
    )
    assert po is not None, "fixture data should contain at least one PO with a shipment and items"

    service, fake_gemini = _make_service(session, vector_store)
    result = service.answer(f"What is the status of purchase order {po.source_reference} and how does it affect inventory?")

    assert result.sufficient_evidence is True
    database_citations = [c for c in result.citations if c.kind == "database"]
    labels = {(c.source_table, c.source_id) for c in database_citations}
    assert ("purchase_orders", po.source_reference) in labels
    # A shipment fact should be present too (joined via the PO reference).
    assert any(c.source_table == "shipments" for c in database_citations)
    # And, if the PO's items reference products with real inventory rows,
    # those should be included too (via PurchaseOrderItem, not a direct
    # PO.product_id field, which the loader never populates).
    inventory_citations = [c for c in database_citations if c.source_table == "inventory"]
    if inventory_citations:
        assert all("on_hand=" in c.text for c in inventory_citations)


# ---------------------------------------------------------------------------
# Contract obligations
# ---------------------------------------------------------------------------

def test_contract_obligations_question_cites_specific_section(loaded_session_and_store):
    session, vector_store = loaded_session_and_store
    service, _ = _make_service(session, vector_store)

    result = service.answer("What are the penalty terms in supplier SUP-014's contract if a shipment is late?")

    document_citations = [c for c in result.citations if c.kind == "document" and c.filename == "SUP-014_contract.md"]
    assert document_citations, "expected at least one chunk from SUP-014's contract document"
    # Section-aware chunking should let us cite the specific clause, not
    # just "somewhere in this document".
    sections = {c.section for c in document_citations if c.section}
    assert sections, f"expected section metadata on contract citations, got: {document_citations}"


# ---------------------------------------------------------------------------
# Insufficient evidence
# ---------------------------------------------------------------------------

def test_insufficient_evidence_for_nonexistent_supplier(loaded_session_and_store):
    session, vector_store = loaded_session_and_store
    service, fake_gemini = _make_service(session, vector_store)

    result = service.answer("What are supplier SUP-999's contractual obligations regarding purple unicorn delivery penalties?")

    assert result.sufficient_evidence is False
    assert result.citations == []
    assert "does not contain sufficient" in result.answer.lower()
    # The LLM must never be called when there's no evidence to ground on --
    # this IS the anti-hallucination mechanism, not just a nice message.
    assert fake_gemini.models.last_prompt is None
