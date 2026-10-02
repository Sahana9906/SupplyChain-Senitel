"""End-to-end tests for MultiSourceInvestigationService -- combines
REAL internal ERP + RAG data (loaded exactly as the CLI would) with
MOCKED external connectors (since real external network access isn't
available in this sandbox), proving the whole pipeline genuinely wires
together: internal DB + internal RAG + external, ranked, cited, with a
grounded (fake, for determinism) LLM answer.
"""
import hashlib
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from logisense import models as m
from logisense.models import ProviderStatus
from logisense.services.erp.csv_loader import load_company_data
from logisense.services.external.evidence_item import EvidenceItem
from logisense.services.external.gdelt_connector import GDELTConnectorResult
from logisense.services.external.maritime_connector import AISStreamConnectorResult
from logisense.services.external.weather_connector import OpenMeteoConnectorResult
from logisense.services.multi_source_investigation import MultiSourceInvestigationService
from logisense.services.rag.hybrid_context import HybridContextBuilder
from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMPANY_DATA_DIR = os.path.join(_REPO_ROOT, "company_data")
_KNOWLEDGE_DIR = os.path.join(_COMPANY_DATA_DIR, "company_knowledge")

pytestmark = pytest.mark.skipif(not os.path.isdir(_COMPANY_DATA_DIR), reason="company_data/ not present in this checkout")


def _word_hash_embed(text: str, dim: int = 48) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


class WordHashEmbeddingClient:
    model_name = "word-hash-test-embedding"

    def embed(self, texts): return [_word_hash_embed(t) for t in texts]
    def embed_documents(self, texts): return self.embed(texts)
    def embed_query(self, text): return _word_hash_embed(text)


@dataclass
class _FakeGenAIResponse:
    text: str


class FakeGeminiModels:
    def __init__(self, response_text="Grounded answer citing the evidence provided."):
        self.response_text = response_text
        self.last_prompt = None

    def generate_content(self, model, contents):
        self.last_prompt = contents
        return _FakeGenAIResponse(text=self.response_text)


class FakeGeminiClient:
    def __init__(self, **kwargs):
        self.models = FakeGeminiModels(**kwargs)


@pytest.fixture(scope="module")
def loaded_env():
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

    settings = Settings(database_url="sqlite:///:memory:", gemini_api_key="x", chroma_collection_name="e2e-multisource")
    vector_store = ChromaVectorStore(settings, client=chromadb.EphemeralClient())
    KnowledgeBaseIndexer(session, WordHashEmbeddingClient(), vector_store).index_directory(_KNOWLEDGE_DIR)

    yield session, vector_store, settings
    session.close()
    engine.dispose()


def _real_supplier_id(session) -> str:
    from sqlalchemy import select
    supplier = session.scalar(select(m.Supplier).where(m.Supplier.source_supplier_id.is_not(None)).limit(1))
    return supplier.source_supplier_id


def _make_service(session, vector_store, settings, weather_result=None, gdelt_result=None, maritime_result=None):
    context_builder = HybridContextBuilder(session, WordHashEmbeddingClient(), vector_store, top_k=6, min_similarity=0.0)
    fake_gemini = FakeGeminiClient()

    weather_mock = MagicMock()
    weather_mock.get_weather_for_location.return_value = weather_result or OpenMeteoConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [])
    gdelt_mock = MagicMock()
    gdelt_mock.search_events.return_value = gdelt_result or GDELTConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [])
    maritime_mock = MagicMock()
    maritime_mock.get_vessel_activity_near.return_value = maritime_result or AISStreamConnectorResult(ProviderStatus.UNAVAILABLE, [], "no key")

    service = MultiSourceInvestigationService(
        session, context_builder, settings,
        weather_connector=weather_mock, gdelt_connector=gdelt_mock, maritime_connector=maritime_mock,
        gemini_client=fake_gemini,
    )
    return service, fake_gemini, weather_mock, gdelt_mock, maritime_mock


# ---------------------------------------------------------------------------
# Test 1: internal-only question -- external sources should not even be queried
# ---------------------------------------------------------------------------

def test_contractual_question_does_not_trigger_external_calls(loaded_env):
    session, vector_store, settings = loaded_env
    supplier_id = _real_supplier_id(session)
    service, fake_gemini, weather_mock, gdelt_mock, maritime_mock = _make_service(session, vector_store, settings)

    result = service.answer(f"What are {supplier_id}'s contractual penalty terms?")

    assert result.sufficient_evidence is True
    assert any(c.source_type == "internal_db" for c in result.citations)
    weather_mock.get_weather_for_location.assert_not_called()
    maritime_mock.get_vessel_activity_near.assert_not_called()
    assert result.source_status["open_meteo"] == "not_queried"
    assert result.source_status["aisstream"] == "not_queried"


# ---------------------------------------------------------------------------
# Test 2: weather-relevant question -- external weather IS queried, combined
# ---------------------------------------------------------------------------

def test_weather_relevant_question_combines_internal_and_external_evidence(loaded_env):
    session, vector_store, settings = loaded_env
    supplier_id = _real_supplier_id(session)

    weather_evidence = OpenMeteoConnectorResult(ProviderStatus.AVAILABLE, [
        EvidenceItem(source_type="weather_api", source_name="Open-Meteo", source_id="loc", content="Heavy rain reported.", location="Test City", timestamp="2026-01-01"),
    ])
    service, fake_gemini, weather_mock, gdelt_mock, maritime_mock = _make_service(session, vector_store, settings, weather_result=weather_evidence)

    result = service.answer(f"Did weather delay {supplier_id}'s shipment?")

    weather_mock.get_weather_for_location.assert_called_once()
    assert any(c.source_type == "weather_api" for c in result.citations)
    assert any(c.source_type == "internal_db" for c in result.citations)
    kinds_in_order = [c.source_type for c in result.citations]
    assert kinds_in_order.index("internal_db") < kinds_in_order.index("weather_api")


# ---------------------------------------------------------------------------
# Test 3: missing AISstream key -- system still works, discloses unavailability
# ---------------------------------------------------------------------------

def test_missing_aisstream_key_does_not_break_the_answer(loaded_env):
    session, vector_store, settings = loaded_env
    supplier_id = _real_supplier_id(session)
    service, fake_gemini, weather_mock, gdelt_mock, maritime_mock = _make_service(
        session, vector_store, settings,
        maritime_result=AISStreamConnectorResult(ProviderStatus.UNAVAILABLE, [], "AISSTREAM_API_KEY is not configured."),
    )

    result = service.answer(f"Is the vessel carrying {supplier_id}'s shipment delayed at port?")

    assert result.sufficient_evidence is True
    assert result.source_status["aisstream"] == "unavailable"
    assert "SOURCES NOT AVAILABLE" in fake_gemini.models.last_prompt
    assert "aisstream" in fake_gemini.models.last_prompt.lower()


# ---------------------------------------------------------------------------
# Test 4: no evidence anywhere -- LLM never called, explicit insufficient-evidence
# ---------------------------------------------------------------------------

def test_no_evidence_anywhere_returns_insufficient_without_calling_llm(loaded_env):
    session, vector_store, settings = loaded_env
    service, fake_gemini, weather_mock, gdelt_mock, maritime_mock = _make_service(session, vector_store, settings)

    result = service.answer("What is the capital of a country with no relation to this data whatsoever, SUP-999999?")

    assert result.sufficient_evidence is False
    assert result.citations == []
    assert fake_gemini.models.last_prompt is None


# ---------------------------------------------------------------------------
# Test 5: external evidence never fabricated when not available
# ---------------------------------------------------------------------------

def test_external_evidence_absent_from_citations_when_not_available(loaded_env):
    session, vector_store, settings = loaded_env
    supplier_id = _real_supplier_id(session)
    service, fake_gemini, weather_mock, gdelt_mock, maritime_mock = _make_service(
        session, vector_store, settings,
        weather_result=OpenMeteoConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, []),
    )

    result = service.answer(f"Did weather affect {supplier_id}'s shipment delay?")

    assert "weather_api" not in [c.source_type for c in result.citations]
