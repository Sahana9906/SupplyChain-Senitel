import os

from logisense.repositories import DocumentRepository
from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer, discover_documents


class FakeEmbeddingClient:
    model_name = "fake-embedding-model"

    def embed(self, texts):
        return [[float(len(t))] for t in texts]  # trivial but deterministic


def _build_knowledge_dir(tmp_path) -> str:
    base = tmp_path / "company_knowledge"
    (base / "supplier_contracts").mkdir(parents=True)
    (base / "escalation_procedures").mkdir(parents=True)

    (base / "supplier_contracts" / "SUP-014_contract.md").write_text(
        "# Supply Agreement — Meridian Supply Co. (SUP-014)\n\n"
        "**Contract ID:** CTR-014\n\n## Terms\n\n"
        "Agreed lead time is 20 days. Penalty terms: 2% order value per day late.\n"
    )
    (base / "escalation_procedures" / "delayed_shipment_escalation.md").write_text(
        "# Delayed Shipment Escalation\n\n"
        "1. Contact carrier for status within 1 business day.\n"
        "2. Escalate to Logistics Manager if unresolved after 2 more days.\n"
    )
    (base / "README.md").write_text("# Company Knowledge Base\n\nThis file should NOT be indexed.\n")
    return str(base)


def test_discover_documents_finds_category_files_and_skips_readme(tmp_path):
    knowledge_dir = _build_knowledge_dir(tmp_path)
    found = discover_documents(knowledge_dir)

    filenames = {f for _, f, _ in found}
    assert "SUP-014_contract.md" in filenames
    assert "delayed_shipment_escalation.md" in filenames
    assert "README.md" not in filenames
    assert len(found) == 2


def test_index_directory_creates_documents_and_chunks(session, vector_store, tmp_path):
    knowledge_dir = _build_knowledge_dir(tmp_path)
    indexer = KnowledgeBaseIndexer(session, FakeEmbeddingClient(), vector_store, chunk_size=2000, chunk_overlap=100)

    result = indexer.index_directory(knowledge_dir)

    assert result.documents_indexed == 2
    assert result.total_chunks >= 2
    assert set(result.categories) == {"supplier_contracts", "escalation_procedures"}

    doc = DocumentRepository(session).get_by_source("knowledge_base_document", "supplier_contracts/SUP-014_contract.md")
    assert doc is not None
    assert doc.title == "Supply Agreement — Meridian Supply Co. (SUP-014)"
    assert doc.doc_metadata["category"] == "supplier_contracts"

    chunks = DocumentRepository(session).get_chunks(doc.id)
    assert len(chunks) >= 1
    assert all(c.vector_store_id for c in chunks)

    assert vector_store.collection.count() == result.total_chunks


def test_chunk_metadata_traces_back_to_source_file(session, vector_store, tmp_path):
    knowledge_dir = _build_knowledge_dir(tmp_path)
    indexer = KnowledgeBaseIndexer(session, FakeEmbeddingClient(), vector_store, chunk_size=2000, chunk_overlap=100)
    indexer.index_directory(knowledge_dir)

    matches = vector_store.query([26.0], top_k=5)  # matches FakeEmbeddingClient's trivial embedding shape
    assert matches
    for match in matches:
        assert "category" in match.metadata
        assert "filename" in match.metadata
        assert "document_id" in match.metadata
        assert "chunk_id" in match.metadata


def test_reindexing_replaces_not_duplicates(session, vector_store, tmp_path):
    knowledge_dir = _build_knowledge_dir(tmp_path)
    indexer = KnowledgeBaseIndexer(session, FakeEmbeddingClient(), vector_store, chunk_size=2000, chunk_overlap=100)

    first = indexer.index_directory(knowledge_dir)
    second = indexer.index_directory(knowledge_dir)

    assert first.documents_indexed == second.documents_indexed
    assert vector_store.collection.count() == second.total_chunks  # not doubled
