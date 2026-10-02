from dataclasses import dataclass

from logisense.config import Settings
from logisense.services.rag.qa_service import EvidenceGroundedQAService


class FakeEmbeddingClient:
    model_name = "fake-embedding-model"

    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        return [self._vector for _ in texts]


@dataclass
class _FakeGenAIResponse:
    text: str


class FakeGeminiModels:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_prompt: str | None = None

    def generate_content(self, model, contents):
        self.last_prompt = contents
        return _FakeGenAIResponse(text=self._response_text)


class FakeGeminiClient:
    def __init__(self, response_text: str):
        self.models = FakeGeminiModels(response_text)


def _settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", gemini_api_key="x")


def test_returns_insufficient_evidence_when_no_matches(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0])
    fake_gemini = FakeGeminiClient(response_text="should not be reached")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), gemini_client=fake_gemini)

    result = qa.answer("What is supplier SUP-014's lead time?")

    assert result.sufficient_evidence is False
    assert result.citations == []
    assert "does not contain sufficient evidence" in result.answer.lower()
    # Gemini should never even be called when there's no evidence to ground on.
    assert fake_gemini.models.last_prompt is None


def test_grounded_answer_includes_citations_from_real_matches(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0, 0.0])
    vector_store.upsert(
        ids=["chunk-1"], embeddings=[[1.0, 0.0, 0.0]], documents=["Agreed lead time is 20 days per the SUP-014 contract."],
        metadatas=[{"document_id": 7, "chunk_id": 42, "category": "supplier_contracts", "filename": "SUP-014_contract.md", "title": "Supply Agreement — SUP-014", "chunk_index": 0, "supplier_id": "SUP-014"}],
    )
    fake_gemini = FakeGeminiClient(response_text="Supplier SUP-014's agreed lead time is 20 days [DOC: supplier_contracts/SUP-014_contract.md].")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), min_similarity=0.0, gemini_client=fake_gemini)

    result = qa.answer("What is supplier SUP-014's lead time?")

    assert result.sufficient_evidence is True
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.reference_number == 1
    assert citation.category == "supplier_contracts"
    assert citation.filename == "SUP-014_contract.md"
    assert citation.document_id == 7
    assert "20 days" in result.answer

    # The prompt actually sent to Gemini must contain the real evidence text,
    # not a paraphrase or a fabricated summary -- this is what "grounded" means.
    assert "Agreed lead time is 20 days per the SUP-014 contract." in fake_gemini.models.last_prompt
    assert "supplier_contracts/SUP-014_contract.md" in fake_gemini.models.last_prompt


def test_min_similarity_filters_weak_matches(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0])
    vector_store.upsert(
        ids=["chunk-1"], embeddings=[[0.0, 1.0]],  # orthogonal -- similarity ~0
        documents=["Unrelated content."],
        metadatas=[{"document_id": 1, "chunk_id": 1, "category": "x", "filename": "y.md", "title": "t", "chunk_index": 0}],
    )
    fake_gemini = FakeGeminiClient(response_text="should not be reached")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), min_similarity=0.5, gemini_client=fake_gemini)

    result = qa.answer("irrelevant question")

    assert result.sufficient_evidence is False
    assert result.citations == []


def test_citation_reference_numbers_are_sequential_and_match_prompt_order(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0])
    vector_store.upsert(
        ids=["chunk-a", "chunk-b"], embeddings=[[1.0, 0.0], [0.9, 0.1]],
        documents=["First relevant passage.", "Second relevant passage."],
        metadatas=[
            {"document_id": 1, "chunk_id": 1, "category": "a", "filename": "a.md", "title": "A", "chunk_index": 0},
            {"document_id": 2, "chunk_id": 2, "category": "b", "filename": "b.md", "title": "B", "chunk_index": 0},
        ],
    )
    fake_gemini = FakeGeminiClient(response_text="Answer referencing [DOC: a/a.md] and [DOC: b/b.md].")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), min_similarity=0.0, top_k=2, gemini_client=fake_gemini)

    result = qa.answer("question")

    assert [c.reference_number for c in result.citations] == [1, 2]
    assert "[DOC: a/a.md]" in fake_gemini.models.last_prompt and "[DOC: b/b.md]" in fake_gemini.models.last_prompt


# ---------------------------------------------------------------------------
# Mandatory entity matching -- a document about a DIFFERENT supplier must
# never be cited as evidence for a question about THIS supplier, even with
# a high raw embedding similarity score.
# ---------------------------------------------------------------------------

def test_entity_matching_excludes_documents_about_a_different_supplier(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0])
    # Both chunks are semantically near-identical to the query embedding
    # (same vector), so similarity alone cannot distinguish them -- only
    # the supplier_id metadata does.
    vector_store.upsert(
        ids=["chunk-sup104", "chunk-sup014"], embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["Supplier SUP-104 breached its 7-day SLA due to a production delay.",
                   "Supplier SUP-014 breached its 7-day SLA due to a customs delay."],
        metadatas=[
            {"document_id": 1, "chunk_id": 1, "category": "historical_incidents", "filename": "SUP-104_incident.md", "title": "SUP-104 Incident", "chunk_index": 0, "supplier_id": "SUP-104"},
            {"document_id": 2, "chunk_id": 2, "category": "historical_incidents", "filename": "SUP-014_incident.md", "title": "SUP-014 Incident", "chunk_index": 0, "supplier_id": "SUP-014"},
        ],
    )
    fake_gemini = FakeGeminiClient(response_text="SUP-104 breached its SLA due to a production delay [DOC: historical_incidents/SUP-104_incident.md].")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), min_similarity=0.0, top_k=5, gemini_client=fake_gemini)

    result = qa.answer("Why did SUP-104 breach the 7-day SLA?")

    assert len(result.citations) == 1
    assert result.citations[0].filename == "SUP-104_incident.md"
    # The SUP-014 document must never even reach Gemini's prompt, despite
    # having an identical embedding similarity score. (Checking the real
    # filename, not the bare string "SUP-014" -- the system instructions'
    # own example text coincidentally mentions a SUP-014 contract.)
    assert "SUP-014_incident.md" not in fake_gemini.models.last_prompt
    assert "customs delay" not in fake_gemini.models.last_prompt


def test_citation_text_property_formats_as_explicit_traceable_source(vector_store):
    embedding_client = FakeEmbeddingClient([1.0, 0.0])
    vector_store.upsert(
        ids=["chunk-1"], embeddings=[[1.0, 0.0]], documents=["Penalty is 5% per day late."],
        metadatas=[{"document_id": 1, "chunk_id": 1, "category": "supplier_contracts", "filename": "SUP-014_contract.md", "title": "T", "chunk_index": 0, "section": "Penalty Terms"}],
    )
    fake_gemini = FakeGeminiClient(response_text="Answer.")
    qa = EvidenceGroundedQAService(embedding_client, vector_store, _settings(), min_similarity=0.0, gemini_client=fake_gemini)

    result = qa.answer("question")

    assert result.citations[0].citation_text == "[DOC: supplier_contracts/SUP-014_contract.md, section: Penalty Terms]"
