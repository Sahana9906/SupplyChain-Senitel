import typer
from alembic import command
from alembic.config import Config

from logisense.config import get_settings
from logisense.db import SessionLocal
from logisense.logging import configure_logging
from logisense.repositories import HypothesisRepository, InvestigationRunRepository
from logisense.services.gemini_client import GeminiHypothesisClient, GeminiIntakeClient
from logisense.services.hypothesis import HypothesisGenerationService
from logisense.services.incident_detection import IncidentDetectionService
from logisense.services.ingestion.dataco_loader import DataCoLoader
from logisense.services.ingestion.olist_loader import OlistLoader
from logisense.services.intake import IncidentIntakeService

app = typer.Typer(help="LogiSense AI Phase 1 operational commands.")


@app.callback()
def main() -> None:
    configure_logging(get_settings().log_level)


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Runs Alembic migrations to head, then verifies the resulting
    schema actually matches what the models declare -- and self-heals
    if not (e.g. a stale alembic_version stamp left over from a prior
    `stamp` without a matching `upgrade`, or a partially-applied
    migration). This makes db-upgrade itself robust to that class of
    bug rather than requiring a separate manual repair step."""
    from sqlalchemy import create_engine
    from logisense.config import get_settings
    from logisense.db import Base
    from logisense.services.db_repair import diagnose_schema, repair_schema
    import logisense.models  # noqa: F401 -- registers all tables onto Base.metadata

    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")

    settings = get_settings()
    engine = create_engine(settings.database_url)
    expected_tables = set(Base.metadata.tables.keys())

    diagnosis = diagnose_schema(engine, cfg, expected_tables)
    if diagnosis.is_consistent:
        typer.echo(f"Schema verified: all {len(expected_tables)} tables present.")
        return

    typer.echo(
        f"WARNING: after upgrade, {len(diagnosis.missing_tables)} table(s) are still missing "
        f"({sorted(diagnosis.missing_tables)}) -- the database's revision stamp did not match its "
        f"actual tables (e.g. from a prior `stamp` without a matching `upgrade`). Self-healing..."
    )
    result = repair_schema(engine, cfg, expected_tables)
    typer.echo(result.action_taken)
    if result.diagnosis_after.is_consistent:
        typer.echo(f"Schema repaired: all {len(expected_tables)} tables now present.")
    else:
        typer.echo(f"Repair incomplete -- still missing: {sorted(result.diagnosis_after.missing_tables)}")
        raise typer.Exit(code=1)


@app.command("load-olist")
def load_olist(
    data_dir: str = typer.Option(..., help="Directory containing the extracted Olist CSV files."),
) -> None:
    """Load the public Olist Brazilian E-Commerce dataset into the
    canonical schema. Does not detect incidents -- run detect-incidents
    afterward."""
    from pathlib import Path

    with SessionLocal() as session:
        result = OlistLoader(session, Path(data_dir)).load()
    typer.echo(
        f"order_items_read={result.order_items_read} "
        f"canonical_orders_upserted={result.canonical_orders_upserted} "
        f"shipments_upserted={result.shipments_upserted} "
        f"skipped_rows={result.skipped_rows}"
    )
    typer.echo(
        f"totals: locations={result.locations_total} products={result.products_total} "
        f"suppliers={result.suppliers_total} customers={result.customers_total}"
    )
    for warning in result.warnings[:20]:
        typer.echo(f"warning: {warning}")
    if len(result.warnings) > 20:
        typer.echo(f"... and {len(result.warnings) - 20} more warnings")


@app.command("load-dataco")
def load_dataco(
    csv_path: str = typer.Option(..., help="Path to DataCoSupplyChainDataset.csv"),
    batch_size: int = typer.Option(2000, min=100, help="Rows per commit batch."),
) -> None:
    """Load the DataCo Smart Supply Chain dataset into the canonical
    schema. Does not detect incidents -- run detect-incidents afterward."""
    from pathlib import Path

    with SessionLocal() as session:
        result = DataCoLoader(session, Path(csv_path), batch_size=batch_size).load()
    typer.echo(
        f"rows_read={result.rows_read} "
        f"canonical_orders_upserted={result.canonical_orders_upserted} "
        f"shipments_upserted={result.shipments_upserted} "
        f"skipped_rows={result.skipped_rows}"
    )
    typer.echo(
        f"totals: locations={result.locations_total} products={result.products_total} "
        f"customers={result.customers_total}"
    )
    for warning in result.warnings[:20]:
        typer.echo(f"warning: {warning}")
    if len(result.warnings) > 20:
        typer.echo(f"... and {len(result.warnings) - 20} more warnings")


@app.command("detect-incidents")
def detect_incidents(
    limit: int = typer.Option(1000, min=1, help="Max shipments to scan for new incidents."),
) -> None:
    """Scan the canonical shipments table and create incidents for real,
    data-derived delivery deviations. Never fabricates data."""
    with SessionLocal() as session:
        result = IncidentDetectionService(session).detect(limit=limit)
    typer.echo(
        f"Scanned {result.shipments_scanned} shipments, "
        f"created {result.incidents_created} incident(s)."
    )


@app.command("start-investigation")
def start_investigation(
    incident_id: int = typer.Option(..., min=1, help="Incident id from the incidents table."),
) -> None:
    settings = get_settings()
    with SessionLocal() as session:
        result = IncidentIntakeService(
            session=session,
            model_client=GeminiIntakeClient(settings),
        ).start_investigation(incident_id=incident_id)

        typer.echo(
            f"run_id={result.run_id} status={result.status} current_stage={result.current_stage}"
        )

        if result.status == "completed":
            intake = InvestigationRunRepository(session).get_intake(result.run_id)
            if intake is not None:
                typer.echo("")
                typer.echo(intake.triage_assessment)


@app.command("generate-hypotheses")
def generate_hypotheses(
    run_id: str = typer.Option(..., help="run_id from a completed Phase 1 investigation."),
) -> None:
    """Phase 2: generate candidate root-cause hypotheses for a run that
    has already completed Phase 1 triage."""
    settings = get_settings()
    with SessionLocal() as session:
        result = HypothesisGenerationService(
            session=session,
            model_client=GeminiHypothesisClient(settings),
        ).generate(run_id=run_id)

        typer.echo(
            f"run_id={result.run_id} current_stage={result.current_stage} "
            f"hypotheses={result.hypothesis_count} ruled_out={result.ruled_out_count} "
            f"similar_past_incidents_used={result.similar_past_incidents_used}"
        )

        hypotheses = HypothesisRepository(session).get_by_run_id(result.run_id)
        for h in hypotheses:
            typer.echo("")
            typer.echo(f"[{h.rank}] ({h.confidence.value.upper()} confidence) {h.statement}")
            typer.echo(f"    Rationale: {h.rationale}")
            typer.echo(f"    Why ranked here: {h.why_ranked_here}")
            typer.echo(f"    Supporting signals: {h.supporting_signals}")
            typer.echo(f"    Would confirm: {h.what_would_confirm}")
            typer.echo(f"    Would refute: {h.what_would_refute}")

        run = InvestigationRunRepository(session).get_by_run_id(result.run_id)
        if run and run.ruled_out_hypotheses:
            typer.echo("")
            typer.echo("Ruled out:")
            for item in run.ruled_out_hypotheses:
                typer.echo(f"  - {item['statement']}")
                typer.echo(f"    Reason: {item['reason_ruled_out']}")


@app.command("erp-load-csv")
def erp_load_csv(
    data_dir: str = typer.Option("company_data", help="Directory containing the ERP CSVs."),
) -> None:
    """Load company_data/*.csv into the ERP database. CSVs are seed data
    only -- nothing downstream reads them directly after this. Idempotent."""
    from logisense.services.erp.csv_loader import load_company_data

    with SessionLocal() as session:
        result = load_company_data(session, data_dir)

    typer.echo(f"suppliers={result.suppliers}  products={result.products}  warehouses={result.warehouses}")
    typer.echo(f"carriers={result.carriers}  customers={result.customers}  contracts={result.contracts}")
    typer.echo(f"purchase_orders={result.purchase_orders}  purchase_order_items={result.purchase_order_items}")
    typer.echo(f"inbound_shipments={result.inbound_shipments}  shipment_events={result.shipment_events}")
    typer.echo(f"customer_orders={result.customer_orders}  inventory_rows={result.inventory_rows}")
    if result.skipped_rows:
        typer.echo(f"skipped (unresolved references): {result.skipped_rows}")


@app.command("rag-index-knowledge")
def rag_index_knowledge(
    knowledge_dir: str = typer.Option("company_data/company_knowledge", help="Directory of category subfolders of .md files."),
) -> None:
    """Discover, chunk, embed, and index the internal knowledge-base
    documents into Postgres (provenance) + the vector store (dense
    retrieval), then build the BM25 lexical index over the SAME
    canonical chunks (read directly from Chroma, guaranteeing both
    retrieval systems see identical text). Embedding is done locally
    via a Sentence Transformers BGE model -- NO Google API key
    required. If the collection already holds vectors from a different
    (incompatible) embedding model, it is cleanly rebuilt rather than
    silently mixing vector spaces. The BM25 index is always rebuilt in
    full here, never incrementally, so it can never silently drift out
    of sync with what Chroma currently holds."""
    from logisense.config import get_settings
    from logisense.services.rag.bge_embeddings import EmbeddingGenerationError, EmbeddingModelLoadError, BGEEmbeddingClient
    from logisense.services.rag.bm25_index import BM25Index
    from logisense.services.rag.knowledge_indexer import KnowledgeBaseIndexer
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()

    try:
        embedding_client = BGEEmbeddingClient(settings)
    except EmbeddingModelLoadError as exc:
        typer.echo(f"ERROR loading embedding model: {exc}")
        raise typer.Exit(code=1)

    vector_store = ChromaVectorStore(settings)
    if vector_store.ensure_compatible_dimension(embedding_client.get_embedding_dimension()):
        typer.echo(
            f"NOTE: existing Chroma collection '{settings.chroma_collection_name}' held vectors from a "
            f"different embedding model (incompatible dimensionality) -- it was rebuilt cleanly."
        )

    try:
        with SessionLocal() as session:
            indexer = KnowledgeBaseIndexer(
                session=session, embedding_client=embedding_client, vector_store=vector_store,
                chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap,
            )
            result = indexer.index_directory(knowledge_dir)
    except EmbeddingGenerationError as exc:
        typer.echo(f"ERROR generating embeddings: {exc}")
        raise typer.Exit(code=1)

    bm25_index = BM25Index.build_from_vector_store(vector_store)
    bm25_index.save(settings.bm25_index_path)

    typer.echo(f"documents_indexed={result.documents_indexed}  total_chunks={result.total_chunks}")
    typer.echo(f"categories={result.categories}")
    typer.echo(f"embedding_model={settings.embedding_model}  (local, no Google API key used)")
    typer.echo(f"chroma_collection={settings.chroma_collection_name}")
    typer.echo(f"bm25_index_chunks={len(bm25_index.chunks)}  bm25_index_path={settings.bm25_index_path}")


@app.command("rag-retrieve")
def rag_retrieve(
    query: str = typer.Argument(..., help="Hypothesis or free-text query to retrieve evidence for."),
    top_k: int = typer.Option(None, help="Final number of results to return. Defaults to settings.hybrid_top_k."),
    mode: str = typer.Option("semantic", help="semantic | bm25 | hybrid."),
) -> None:
    """Pure retrieval, no LLM call. Prints the actual retrieved context
    + full metadata + scores/ranks for each result -- nothing is
    generated or summarized.

    mode=semantic : dense (BGE + Chroma) only (default).
    mode=bm25     : lexical (BM25) only.
    mode=hybrid   : both, combined via Reciprocal Rank Fusion.
    """
    from logisense.config import get_settings
    from logisense.services.rag.bge_embeddings import EmbeddingGenerationError, EmbeddingModelLoadError, BGEEmbeddingClient
    from logisense.services.rag.bm25_index import BM25Index
    from logisense.services.rag.hybrid_retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    final_top_k = top_k or settings.hybrid_top_k

    if mode not in ("semantic", "bm25", "hybrid"):
        typer.echo(f"ERROR: unknown mode '{mode}' -- must be semantic, bm25, or hybrid.")
        raise typer.Exit(code=1)

    dense_retriever = None
    bm25_retriever = None

    if mode in ("semantic", "hybrid"):
        try:
            embedding_client = BGEEmbeddingClient(settings)
        except EmbeddingModelLoadError as exc:
            typer.echo(f"ERROR loading embedding model: {exc}")
            raise typer.Exit(code=1)
        dense_retriever = DenseRetriever(embedding_client, ChromaVectorStore(settings))

    if mode in ("bm25", "hybrid"):
        try:
            bm25_index = BM25Index.load(settings.bm25_index_path)
        except FileNotFoundError as exc:
            typer.echo(f"ERROR: {exc}")
            raise typer.Exit(code=1)
        bm25_retriever = BM25Retriever(bm25_index)

    try:
        if mode == "semantic":
            results = dense_retriever.retrieve(query, top_k=final_top_k)
        elif mode == "bm25":
            results = bm25_retriever.retrieve(query, top_k=final_top_k)
        else:
            hybrid = HybridRetriever(dense_retriever, bm25_retriever, rrf_k=settings.rrf_k)
            results = hybrid.retrieve(
                query, dense_top_k=settings.dense_top_k, bm25_top_k=settings.bm25_top_k, final_top_k=final_top_k,
            )
    except EmbeddingGenerationError as exc:
        typer.echo(f"ERROR generating query embedding: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Query: {query}")
    typer.echo(f"Mode: {mode}")
    if mode != "bm25":
        typer.echo(f"Embedding model: {settings.embedding_model}  (local, no Google API key used)")
    typer.echo(f"Results: {len(results)}\n")

    for i, cand in enumerate(results, start=1):
        meta = cand.metadata
        typer.echo(f"--- Rank {i} (hybrid_score={round(cand.hybrid_score, 5)}) ---")
        typer.echo(f"  chunk_id: {cand.chunk_id}")
        typer.echo(f"  document_id: {cand.document_id}")
        typer.echo(f"  source: {cand.source}")
        typer.echo(f"  document_type: {meta.get('document_type')}")
        typer.echo(f"  section: {meta.get('section') or '(none)'}")
        typer.echo(f"  supplier_id: {meta.get('supplier_id') or '(none)'}")
        typer.echo(f"  contract_id: {meta.get('contract_id') or '(none)'}")
        typer.echo(f"  dense_rank: {cand.dense_rank}  dense_score: {cand.dense_score}")
        typer.echo(f"  bm25_rank: {cand.bm25_rank}  bm25_score: {cand.bm25_score}")
        typer.echo(f"  context: {cand.text}")
        typer.echo("")


@app.command("rag-query")
def rag_query(question: str = typer.Argument(..., help="Question to answer from the knowledge base.")) -> None:
    """Evidence-grounded Q&A over the indexed knowledge base. Answer is
    grounded ONLY in retrieved evidence, with citations."""
    from logisense.config import get_settings
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
    from logisense.services.rag.qa_service import EvidenceGroundedQAService
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    embedding_client = BGEEmbeddingClient(settings)
    qa = EvidenceGroundedQAService(
        embedding_client=embedding_client, vector_store=ChromaVectorStore(settings), settings=settings,
        top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
    )
    result = qa.answer(question)

    typer.echo(f"Q: {result.question}")
    typer.echo(f"\nA: {result.answer}")
    typer.echo(f"\nSufficient evidence: {result.sufficient_evidence}")
    if result.citations:
        typer.echo("\nCitations:")
        for c in result.citations:
            typer.echo(f"  {c.citation_text} -- \"{c.title}\" (relevance={c.relevance_score})")


@app.command("investigate")
def investigate(
    supplier_id: str = typer.Option(..., "--supplier", help="Real supplier source id, e.g. SUP-014."),
) -> None:
    """Combined ERP + RAG investigation demo: operational facts from the
    database, plus evidence-grounded answers from the knowledge base."""
    from logisense.config import get_settings
    from logisense.services.investigate import InvestigationService
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
    from logisense.services.rag.qa_service import EvidenceGroundedQAService
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    qa_service = None
    try:
        qa_service = EvidenceGroundedQAService(
            embedding_client=BGEEmbeddingClient(settings), vector_store=ChromaVectorStore(settings),
            settings=settings, top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
        )
    except ValueError:
        typer.echo("(GROQ_API_KEY not set -- knowledge-base questions will be skipped, ERP facts only)\n")

    with SessionLocal() as session:
        svc = InvestigationService(session, qa_service=qa_service)
        investigation = svc.investigate_supplier(supplier_id)

    typer.echo("=== ERP: Operational Facts ===")
    if investigation.operational is None:
        typer.echo(f"No supplier found with id {supplier_id}")
        return
    op = investigation.operational
    typer.echo(f"Supplier: {op.supplier_name} ({op.supplier_id})")
    typer.echo(f"Contract: {op.contract_id} -- {op.contract_sla}")
    typer.echo(f"Open purchase orders: {len(op.open_purchase_orders)}")
    for po in op.open_purchase_orders[:5]:
        typer.echo(f"  {po['po_id']}: {po['status']}, expected {po['expected_ready_date']}, value ${po['total_value']}")
    typer.echo(f"Recent shipments: {len(op.recent_shipments)}")
    for s in op.recent_shipments[:5]:
        typer.echo(f"  {s['po_id']}: delay_days={s['delay_days']}, carrier={s['carrier']}")

    if investigation.knowledge_answers:
        typer.echo("\n=== RAG: Organizational Knowledge ===")
        for answer in investigation.knowledge_answers:
            typer.echo(f"\nQ: {answer.question}")
            typer.echo(f"A: {answer.answer}")
            for c in answer.citations:
                typer.echo(f"  [{c.reference_number}] {c.category}/{c.filename}")


@app.command("analyze")
def analyze(question: str = typer.Argument(..., help="Free-text supply-chain question -- combines internal DB + internal RAG + external weather/maritime/news evidence.")) -> None:
    """Multi-source grounded investigation: internal database facts +
    internal knowledge-base documents + external real-time data
    (Open-Meteo weather, GDELT events/news, AISstream.io maritime data
    where relevant and available) -- combined into one evidence-grounded
    answer with explicit, traceable citations for every claim. External
    sources are only queried when the question is actually relevant to
    them (e.g. weather is skipped for a purely contractual question),
    and entity/location parameters are derived from the internal
    database automatically -- never asked of the user."""
    from logisense.config import get_settings
    from logisense.services.multi_source_investigation import MultiSourceInvestigationService
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
    from logisense.services.rag.hybrid_context import HybridContextBuilder
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    with SessionLocal() as session:
        embedding_client = BGEEmbeddingClient(settings)
        context_builder = HybridContextBuilder(
            session, embedding_client, ChromaVectorStore(settings),
            top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
        )
        service = MultiSourceInvestigationService(session, context_builder, settings)
        result = service.answer(question)

    typer.echo(f"Q: {result.question}")
    typer.echo(f"\nA: {result.answer}")
    typer.echo(f"\nSufficient evidence: {result.sufficient_evidence}")
    typer.echo("\nSource status:")
    for source, status in result.source_status.items():
        typer.echo(f"  {source}: {status}")
    if result.citations:
        typer.echo("\nCitations:")
        for c in result.citations:
            typer.echo(f"  {c.citation_text}")


@app.command("ask")
def ask(question: str = typer.Argument(..., help="Free-text supply-chain question, e.g. about a supplier, PO, or contract.")) -> None:
    """Hybrid investigation: answers a free-text question using BOTH real
    structured ERP facts (exact database lookups for any SUP-/PO-/SKU-/
    CTR- ids mentioned) and semantically retrieved, entity-filtered
    document evidence -- with citations for both. Says so explicitly if
    neither source has enough evidence, rather than guessing."""
    from logisense.config import get_settings
    from logisense.services.investigate import InvestigationService
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
    from logisense.services.rag.hybrid_context import HybridContextBuilder
    from logisense.services.rag.hybrid_investigation import HybridInvestigationService
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()

    with SessionLocal() as session:
        embedding_client = BGEEmbeddingClient(settings)
        context_builder = HybridContextBuilder(
            session, embedding_client, ChromaVectorStore(settings),
            top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
        )
        hybrid_service = HybridInvestigationService(context_builder, settings)
        svc = InvestigationService(session, hybrid_service=hybrid_service)
        result = svc.ask(question)

    typer.echo(f"Q: {result.question}")
    typer.echo(f"\nA: {result.answer}")
    typer.echo(f"\nSufficient evidence: {result.sufficient_evidence}")
    if result.citations:
        typer.echo("\nCitations:")
        for c in result.citations:
            if c.kind == "database":
                typer.echo(f"  {c.citation_text} -- {c.label}")
            else:
                typer.echo(f"  {c.citation_text} (relevance={c.relevance_score})")


@app.command("validate-hypotheses")
def validate_hypotheses(
    run_id: str = typer.Option(..., "--run-id", help="Investigation run id that already has hypotheses generated."),
    include_external: bool = typer.Option(False, "--include-external", help="Also query weather/maritime/news APIs where relevant to each hypothesis. Off by default (adds real network calls); gracefully degrades if a source's API key isn't configured."),
) -> None:
    """Hypothesis -> Evidence Retrieval -> Hypothesis Validation ->
    Grounded Finding. For each hypothesis on this run, retrieves
    incident-specific structured evidence (supplier, contract, PO,
    shipment, inventory), semantic evidence (contract clauses,
    historical incidents -- via hybrid dense+BM25 retrieval when a BM25
    index is available, dense-only otherwise), and optionally external
    evidence (weather/maritime/news, with --include-external), validates
    the hypothesis against ONLY that evidence, and produces a final
    grounded investigation finding. Persists to the existing
    Evidence/EvidenceAssessment/Verdict tables."""
    from logisense.config import get_settings
    from logisense.services.evidence_validation import HypothesisEvidenceValidator
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient
    from logisense.services.rag.bm25_index import BM25Index
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()

    bm25_index = None
    try:
        bm25_index = BM25Index.load(settings.bm25_index_path)
    except FileNotFoundError:
        typer.echo(f"NOTE: no BM25 index found at '{settings.bm25_index_path}' -- using dense-only retrieval. Run rag-index-knowledge to enable hybrid retrieval here.")

    with SessionLocal() as session:
        validator = HypothesisEvidenceValidator(
            session, BGEEmbeddingClient(settings), ChromaVectorStore(settings), settings,
            bm25_index=bm25_index, rrf_k=settings.rrf_k, enable_external_evidence=include_external,
        )
        result = validator.validate_run(run_id)

    typer.echo("=== Hypothesis Validation ===")
    for r in result.hypothesis_results:
        typer.echo(f"\n[{r.rank}] {r.statement}")
        typer.echo(f"  Status: {r.status}  Confidence: {r.confidence}")
        typer.echo(f"  Reasoning: {r.reasoning}")
        if r.contradicting_evidence_summary:
            typer.echo(f"  Contradicting: {r.contradicting_evidence_summary}")
        typer.echo(f"  Evidence items: {len(r.evidence_items)}  Citations: {len(r.citations)}")

    typer.echo("\n=== Grounded Finding ===")
    typer.echo(f"Root cause hypothesis id: {result.root_cause_hypothesis_id}")
    typer.echo(f"Confidence: {result.confidence}")
    typer.echo(f"Explanation: {result.explanation}")
    typer.echo(f"Supporting evidence: {result.supporting_evidence_summary}")
    typer.echo(f"Contradicting evidence: {result.contradicting_evidence_summary}")
    typer.echo(f"Remaining uncertainty: {result.remaining_uncertainty}")
    typer.echo(f"Evidence gaps: {result.evidence_gaps}")
    typer.echo(f"Recommended next action: {result.recommended_next_action}")


@app.command("generate-mitigations")
def generate_mitigations(
    run_id: str = typer.Option(..., "--run-id", help="Investigation run id that already has a grounded finding (run validate-hypotheses first)."),
) -> None:
    """Phase 4: Mitigation Strategy Generation. Consumes ONLY the
    persisted Investigation output (Verdict + Hypothesis + Evidence rows
    for this run -- never re-investigates, never regenerates hypotheses,
    never determines a new root cause), identifies applicable mitigation
    categories deterministically, asks Groq for descriptive,
    evidence-grounded strategies for each, validates them, scores
    priority/urgency/confidence deterministically, and persists to the
    mitigation_strategies table."""
    from logisense.services.mitigation import MitigationGenerationService

    settings = get_settings()
    with SessionLocal() as session:
        result = MitigationGenerationService(session, settings).generate(run_id)

    typer.echo("=== Mitigation Strategy Generation ===")
    typer.echo(f"Root cause status: {result.root_cause_status}")
    typer.echo(f"Mitigation categories identified: {', '.join(result.categories_identified)}")
    if result.rejected_count:
        typer.echo(f"Strategies rejected by validation: {result.rejected_count}")

    for s in result.strategies:
        typer.echo("")
        flag = "" if s.validation_status.value == "valid" else f"  [{s.validation_status.value.upper()}: {s.validation_notes}]"
        action_kind = "CONFIRMED CORRECTIVE ACTION" if s.is_confirmed_corrective_action else "PREVENTIVE / RISK-REDUCTION"
        typer.echo(f"[{s.priority.value.upper()}/{s.urgency.value}] {s.title} ({action_kind}, confidence={s.confidence}){flag}")
        typer.echo(f"    Type: {s.strategy_type}")
        typer.echo(f"    Description: {s.description}")
        typer.echo(f"    Why this mitigation: {s.why_this_mitigation}")
        typer.echo(f"    Investigation basis:")
        for basis in s.investigation_basis:
            typer.echo(f"      - {basis}")
        typer.echo(f"    Risk addressed: {s.risk_addressed}")
        typer.echo(f"    Implementation steps:")
        for i, step in enumerate(s.implementation_steps, start=1):
            typer.echo(f"      {i}. {step}")
        typer.echo(f"    Expected impact: {s.expected_impact}")
        typer.echo(f"    Limitations: {s.limitations}")
        if s.dependencies:
            typer.echo(f"    Dependencies: {', '.join(s.dependencies)}")
        typer.echo(f"    Success metrics: {', '.join(s.success_metrics)}")

    if not result.strategies:
        typer.echo("No mitigation strategies survived validation -- see logs for rejection reasons.")


@app.command("db-verify")
def db_verify() -> None:
    """Checks whether the database's actual tables match what the
    Alembic-stamped revision claims to have created. Catches the class
    of bug where `alembic_version` says migrations ran but a table is
    actually missing (e.g. from a `stamp` without a matching `upgrade`)."""
    from alembic.config import Config
    from sqlalchemy import create_engine
    from logisense.config import get_settings
    from logisense.db import Base
    from logisense.services.db_repair import diagnose_schema
    import logisense.models  # noqa: F401 -- registers all tables onto Base.metadata

    settings = get_settings()
    engine = create_engine(settings.database_url)
    cfg = Config("alembic.ini")
    expected_tables = set(Base.metadata.tables.keys())

    diagnosis = diagnose_schema(engine, cfg, expected_tables)

    typer.echo(f"Stamped revision: {diagnosis.stamped_revision}")
    typer.echo(f"Expected tables: {len(diagnosis.expected_tables)}")
    typer.echo(f"Actual tables: {len(diagnosis.actual_tables)}")
    if diagnosis.is_consistent:
        typer.echo("Schema is CONSISTENT -- all expected tables are present.")
    else:
        typer.echo(f"Schema is INCONSISTENT -- {len(diagnosis.missing_tables)} missing table(s): {sorted(diagnosis.missing_tables)}")
        if diagnosis.repair_target_revision:
            typer.echo(f"Run `db-repair` to fix (would re-stamp to '{diagnosis.repair_target_revision}' and re-run upgrade).")
        else:
            typer.echo("Could not determine a safe automatic repair point -- manual review needed.")
        raise typer.Exit(code=1)


@app.command("db-repair")
def db_repair() -> None:
    """Repairs a database whose alembic_version stamp doesn't match its
    actual tables. Works entirely through Alembic's own stamp + upgrade
    commands (re-stamps to the last revision that genuinely matches the
    database's real state, then runs a real upgrade) -- never creates
    tables manually, never bypasses the migration system."""
    from alembic.config import Config
    from sqlalchemy import create_engine
    from logisense.config import get_settings
    from logisense.db import Base
    from logisense.services.db_repair import repair_schema
    import logisense.models  # noqa: F401

    settings = get_settings()
    engine = create_engine(settings.database_url)
    cfg = Config("alembic.ini")
    expected_tables = set(Base.metadata.tables.keys())

    result = repair_schema(engine, cfg, expected_tables)

    typer.echo(result.action_taken)
    typer.echo("")
    if result.diagnosis_after.is_consistent:
        typer.echo(f"Repair successful -- all {len(result.diagnosis_after.expected_tables)} expected tables now present.")
    else:
        typer.echo(f"Repair incomplete -- still missing: {sorted(result.diagnosis_after.missing_tables)}")
        raise typer.Exit(code=1)


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("logisense.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
