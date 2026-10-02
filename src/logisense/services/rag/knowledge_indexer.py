"""Discovers and indexes the internal knowledge-base documents
(company_knowledge/<category>/*.md) into Postgres (Document/DocumentChunk
for provenance) and the vector store (embeddings for retrieval).

Category (the subfolder name) becomes each document's document_type.
Chunking is section-aware (splits on markdown "## " headings first) so a
chunk from a contract's "Penalty Terms" section is tagged with that
section, not just an arbitrary character offset -- required for
"which clause/section supports this claim" citations. Every chunk also
carries the real supplier/contract/PO/etc IDs found in that document, so
retrieval can be filtered to a specific entity, not just semantically
similar text in general.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from logisense.repositories import DocumentRepository
from logisense.services.rag.chunker import chunk_by_sections
from logisense.services.rag.embeddings import EmbeddingClient
from logisense.services.rag.entity_extraction import extract_entities
from logisense.services.rag.vector_store import ChromaVectorStore

SOURCE_TYPE_KNOWLEDGE_DOC = "knowledge_base_document"


@dataclass(frozen=True)
class IndexedDocument:
    category: str
    filename: str
    title: str
    document_id: int
    chunk_count: int


@dataclass(frozen=True)
class KnowledgeIndexResult:
    documents_indexed: int
    total_chunks: int
    categories: list[str]


def discover_documents(knowledge_dir: str) -> list[tuple[str, str, str]]:
    """Returns (category, filename, full_path) for every .md file under
    knowledge_dir/<category>/*.md. README.md at the root is skipped --
    it's documentation about the corpus, not evidence within it."""
    found = []
    if not os.path.isdir(knowledge_dir):
        return found
    for category in sorted(os.listdir(knowledge_dir)):
        category_path = os.path.join(knowledge_dir, category)
        if not os.path.isdir(category_path):
            continue
        for filename in sorted(os.listdir(category_path)):
            if filename.endswith((".md", ".txt")):
                found.append((category, filename, os.path.join(category_path, filename)))
    return found


def _extract_title(content: str, fallback: str) -> str:
    match = re.match(r"^#\s+(.+)$", content.strip().splitlines()[0] if content.strip() else "")
    return match.group(1).strip() if match else fallback


class KnowledgeBaseIndexer:
    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
        vector_store: ChromaVectorStore,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.session = session
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = DocumentRepository(session)

    def index_directory(self, knowledge_dir: str) -> KnowledgeIndexResult:
        found = discover_documents(knowledge_dir)
        indexed: list[IndexedDocument] = []
        for category, filename, full_path in found:
            indexed.append(self._index_file(category, filename, full_path))
        self.session.commit()

        return KnowledgeIndexResult(
            documents_indexed=len(indexed),
            total_chunks=sum(d.chunk_count for d in indexed),
            categories=sorted({d.category for d in indexed}),
        )

    def _index_file(self, category: str, filename: str, full_path: str) -> IndexedDocument:
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        source_reference = f"{category}/{filename}"
        title = _extract_title(content, fallback=filename)
        doc_entities = extract_entities(content)

        document = self.documents.upsert(
            source_type=SOURCE_TYPE_KNOWLEDGE_DOC,
            source_reference=source_reference,
            title=title,
            content=content,
            doc_metadata={
                "category": category, "filename": filename,
                "supplier_id": doc_entities.primary_supplier_id,
                "contract_id": doc_entities.contract_ids[0] if doc_entities.contract_ids else None,
            },
        )

        sectioned_chunks = chunk_by_sections(content, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_texts = [c for _, c in sectioned_chunks]
        section_titles = [s for s, _ in sectioned_chunks]
        vector_ids = [f"document:{document.id}:chunk:{i}" for i in range(len(chunk_texts))]

        self.vector_store.delete_by_document(document.id)

        chunk_rows = self.documents.replace_chunks(
            document_id=document.id, chunk_texts=chunk_texts,
            embedding_model=self.embedding_client.model_name, vector_ids=vector_ids,
        )

        embeddings = self.embedding_client.embed(chunk_texts) if chunk_texts else []
        if chunk_texts:
            metadatas = []
            for i, chunk_text_value in enumerate(chunk_texts):
                chunk_entities = extract_entities(chunk_text_value)
                metadatas.append({
                    "source_type": SOURCE_TYPE_KNOWLEDGE_DOC, "document_id": document.id,
                    "chunk_id": chunk_rows[i].id, "document_type": category, "category": category,
                    "filename": filename, "title": title, "chunk_index": i,
                    "section": section_titles[i] or "",
                    # Chunk-level entities (what THIS passage mentions) fall
                    # back to the document-level entity when the chunk
                    # itself doesn't repeat the ID (e.g. a contract's
                    # "Penalty Terms" section chunk that doesn't restate
                    # "SUP-014") -- still correctly attributable to the doc.
                    "supplier_id": chunk_entities.primary_supplier_id or doc_entities.primary_supplier_id or "",
                    "contract_id": (chunk_entities.contract_ids[0] if chunk_entities.contract_ids
                                    else (doc_entities.contract_ids[0] if doc_entities.contract_ids else "")),
                })
            self.vector_store.upsert(ids=vector_ids, embeddings=embeddings, documents=chunk_texts, metadatas=metadatas)

        return IndexedDocument(category=category, filename=filename, title=title, document_id=document.id, chunk_count=len(chunk_texts))
