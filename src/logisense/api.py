from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from logisense import models as m
from logisense.config import get_settings
from logisense.db import get_session
from logisense.repositories import (
    ContextualMetricsRepository,
    HypothesisRepository,
    IncidentRepository,
    InvestigationRunRepository,
)
from logisense.services.gemini_client import GeminiHypothesisClient, GeminiIntakeClient
from logisense.services.hypothesis import HypothesisGenerationService
from logisense.services.intake import IncidentIntakeService

app = FastAPI(title="LogiSense AI - Phase 1: Incident Intake & Triage", version="1.0.0")

# The static frontend runs on a fixed local origin -- scope CORS to it
# specifically rather than allowing any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InvestigationCreateRequest(BaseModel):
    incident_id: int = Field(gt=0)


class InvestigationCreateResponse(BaseModel):
    run_id: str
    incident_id: int
    status: str
    current_stage: str


class OperationalDetails(BaseModel):
    expected_delivery_date: date | None
    actual_delivery_date: date | None
    delay_days: int | None
    shipping_mode: str | None
    carrier: str | None = None
    shipment_id: int | None = None


class BusinessContext(BaseModel):
    order_value: float | None
    supplier: str | None
    region: str | None


class IncidentListItem(BaseModel):
    incident_id: int
    incident_type: str
    shipment_id: int
    deviation_value: float | None
    detected_at: str
    region: str | None
    supplier: str | None
    latest_run_id: str | None = None
    latest_status: str | None = None
    latest_severity: str | None = None


class InvestigationDetailResponse(BaseModel):
    run_id: str
    incident_id: int
    status: str
    current_stage: str
    error_message: str | None = None

    operational_details: OperationalDetails
    business_context: BusinessContext

    incident_type: str | None = None
    severity: str | None = None
    impact: str | None = None
    priority: str | None = None
    rationale: str | None = None
    recommended_next_step: str | None = None
    normalized_summary: str | None = None
    triage_assessment: str | None = None

    # Dashboard data: the metrics snapshot Gemini actually saw (frozen at
    # triage time), plus two live comparison views for charts.
    contextual_metrics: dict | None = None
    region_comparison: list[dict] = Field(default_factory=list)
    region_trend: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class IncidentDetailResponse(BaseModel):
    incident_id: int
    incident_type: str
    deviation_value: float | None
    rationale: str
    detected_at: str
    operational_details: OperationalDetails
    business_context: BusinessContext
    latest_run_id: str | None = None
    latest_status: str | None = None
    latest_severity: str | None = None


@app.get("/incidents/{incident_id}", response_model=IncidentDetailResponse)
def get_incident(
    incident_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> IncidentDetailResponse:
    """Real incident facts straight from the database, independent of
    whether an investigation has been started yet -- lets the frontend
    show what an incident IS before the user decides to investigate it."""
    incident = IncidentRepository(session).get_with_context(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")

    shipment = incident.shipment
    order = shipment.order
    supplier = order.supplier if order else None
    customer = order.customer if order else None
    region = customer.location.region if customer and customer.location else None

    runs_repo = InvestigationRunRepository(session)
    latest_run = runs_repo.get_latest_for_incidents([incident.id]).get(incident.id)
    latest_severity = None
    if latest_run and latest_run.status == m.InvestigationRunStatus.COMPLETED:
        intake = runs_repo.get_intake(latest_run.run_id)
        latest_severity = intake.severity.value if intake else None

    return IncidentDetailResponse(
        incident_id=incident.id,
        incident_type=incident.incident_type,
        deviation_value=float(incident.deviation_value) if incident.deviation_value is not None else None,
        rationale=incident.rationale,
        detected_at=incident.detected_at.isoformat(),
        operational_details=OperationalDetails(
            expected_delivery_date=shipment.expected_delivery_date,
            actual_delivery_date=shipment.actual_delivery_date,
            delay_days=shipment.delay_days,
            shipping_mode=shipment.shipping_mode,
            carrier=shipment.carrier.name if shipment.carrier else None,
            shipment_id=shipment.id,
        ),
        business_context=BusinessContext(
            order_value=float(order.order_value) if order and order.order_value is not None else None,
            supplier=supplier.name if supplier else None,
            region=region,
        ),
        latest_run_id=latest_run.run_id if latest_run else None,
        latest_status=latest_run.status.value if latest_run else None,
        latest_severity=latest_severity,
    )


@app.get("/incidents", response_model=list[IncidentListItem])
def list_incidents(
    session: Annotated[Session, Depends(get_session)],
    limit: int = 50,
) -> list[IncidentListItem]:
    incidents = IncidentRepository(session).list_recent(limit=limit)
    latest_runs = InvestigationRunRepository(session).get_latest_for_incidents(
        [incident.id for incident in incidents]
    )
    runs_repo = InvestigationRunRepository(session)

    items = []
    for incident in incidents:
        order = incident.shipment.order
        customer = order.customer if order else None
        supplier = order.supplier if order else None
        region = customer.location.region if customer and customer.location else None

        latest_run = latest_runs.get(incident.id)
        latest_severity = None
        if latest_run and latest_run.status == m.InvestigationRunStatus.COMPLETED:
            intake = runs_repo.get_intake(latest_run.run_id)
            latest_severity = intake.severity.value if intake else None

        items.append(
            IncidentListItem(
                incident_id=incident.id,
                incident_type=incident.incident_type,
                shipment_id=incident.shipment_id,
                deviation_value=float(incident.deviation_value) if incident.deviation_value is not None else None,
                detected_at=incident.detected_at.isoformat(),
                region=region,
                supplier=supplier.name if supplier else None,
                latest_run_id=latest_run.run_id if latest_run else None,
                latest_status=latest_run.status.value if latest_run else None,
                latest_severity=latest_severity,
            )
        )
    return items


@app.post("/investigations", response_model=InvestigationCreateResponse)
def create_investigation(
    request: InvestigationCreateRequest,
    session: Annotated[Session, Depends(get_session)],
) -> InvestigationCreateResponse:
    settings = get_settings()
    try:
        result = IncidentIntakeService(
            session=session,
            model_client=GeminiIntakeClient(settings),
        ).start_investigation(request.incident_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Incident intake failed: {exc}") from exc

    return InvestigationCreateResponse(
        run_id=result.run_id,
        incident_id=result.incident_id,
        status=result.status,
        current_stage=result.current_stage,
    )


@app.get("/investigations/{run_id}", response_model=InvestigationDetailResponse)
def get_investigation(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> InvestigationDetailResponse:
    runs = InvestigationRunRepository(session)
    run = runs.get_by_run_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Investigation run not found: {run_id}")

    incident = IncidentRepository(session).get_with_context(run.incident_id)
    shipment = incident.shipment
    order = shipment.order
    supplier = order.supplier if order else None
    customer = order.customer if order else None
    region = customer.location.region if customer and customer.location else None

    intake = runs.get_intake(run_id)

    region_comparison: list[dict] = []
    region_trend: list[dict] = []
    if region:
        metrics_repo = ContextualMetricsRepository(session)
        region_comparison = metrics_repo.region_comparison(current_region=region)
        region_trend = metrics_repo.region_trend(region)

    return InvestigationDetailResponse(
        run_id=run.run_id,
        incident_id=run.incident_id,
        status=run.status.value,
        current_stage=run.current_stage.value,
        error_message=run.error_message,
        operational_details=OperationalDetails(
            expected_delivery_date=shipment.expected_delivery_date,
            actual_delivery_date=shipment.actual_delivery_date,
            delay_days=shipment.delay_days,
            shipping_mode=shipment.shipping_mode,
            carrier=shipment.carrier.name if shipment.carrier else None,
            shipment_id=shipment.id,
        ),
        business_context=BusinessContext(
            order_value=float(order.order_value) if order and order.order_value is not None else None,
            supplier=supplier.name if supplier else None,
            region=region,
        ),
        incident_type=intake.incident_type if intake else None,
        severity=intake.severity.value if intake else None,
        impact=intake.impact.value if intake and intake.impact else None,
        priority=intake.priority.value if intake and intake.priority else None,
        rationale=intake.rationale if intake else None,
        recommended_next_step=intake.recommended_next_step if intake else None,
        normalized_summary=intake.normalized_summary if intake else None,
        triage_assessment=intake.triage_assessment if intake else None,
        contextual_metrics=run.contextual_metrics,
        region_comparison=region_comparison,
        region_trend=region_trend,
    )


class HypothesisItemResponse(BaseModel):
    id: int
    rank: int
    statement: str
    rationale: str
    supporting_signals: str
    confidence: str
    what_would_confirm: str
    what_would_refute: str
    why_ranked_here: str
    created_at: str | None = None


class RuledOutHypothesisResponse(BaseModel):
    statement: str
    reason_ruled_out: str


class HypothesisSetDetailResponse(BaseModel):
    hypotheses: list[HypothesisItemResponse]
    ruled_out: list[RuledOutHypothesisResponse]


@app.post("/investigations/{run_id}/hypotheses", response_model=HypothesisSetDetailResponse)
def create_hypotheses(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> HypothesisSetDetailResponse:
    settings = get_settings()
    try:
        HypothesisGenerationService(
            session=session,
            model_client=GeminiHypothesisClient(settings),
        ).generate(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hypothesis generation failed: {exc}") from exc

    run = InvestigationRunRepository(session).get_by_run_id(run_id)
    hypotheses = HypothesisRepository(session).get_by_run_id(run_id)
    return HypothesisSetDetailResponse(
        hypotheses=[
            HypothesisItemResponse(
                id=h.id,
                rank=h.rank,
                statement=h.statement,
                rationale=h.rationale,
                supporting_signals=h.supporting_signals,
                confidence=h.confidence.value,
                what_would_confirm=h.what_would_confirm,
                what_would_refute=h.what_would_refute,
                why_ranked_here=h.why_ranked_here,
                created_at=h.created_at.isoformat() if h.created_at else None,
            )
            for h in hypotheses
        ],
        ruled_out=[
            RuledOutHypothesisResponse(**item) for item in (run.ruled_out_hypotheses or [])
        ],
    )


@app.get("/investigations/{run_id}/hypotheses", response_model=HypothesisSetDetailResponse)
def get_hypotheses(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> HypothesisSetDetailResponse:
    run = InvestigationRunRepository(session).get_by_run_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Investigation run not found: {run_id}")

    hypotheses = HypothesisRepository(session).get_by_run_id(run_id)
    return HypothesisSetDetailResponse(
        hypotheses=[
            HypothesisItemResponse(
                id=h.id,
                rank=h.rank,
                statement=h.statement,
                rationale=h.rationale,
                supporting_signals=h.supporting_signals,
                confidence=h.confidence.value,
                what_would_confirm=h.what_would_confirm,
                what_would_refute=h.what_would_refute,
                why_ranked_here=h.why_ranked_here,
                created_at=h.created_at.isoformat() if h.created_at else None,
            )
            for h in hypotheses
        ],
        ruled_out=[
            RuledOutHypothesisResponse(**item) for item in (run.ruled_out_hypotheses or [])
        ],
    )



# ---------------------------------------------------------------------------
# Hybrid investigation: structured ERP facts + RAG document evidence,
# grounded LLM answering with citations. Part of the same investigation
# surface as the endpoints above, not a separate app/router.
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class EvidenceCitationResponse(BaseModel):
    reference_number: int
    kind: str
    label: str
    source_table: str | None = None
    source_id: str | None = None
    category: str | None = None
    filename: str | None = None
    section: str | None = None
    relevance_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[EvidenceCitationResponse]


@app.post("/investigate/ask", response_model=AskResponse)
def investigate_ask(
    request: AskRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AskResponse:
    """Evidence-grounded question answering over BOTH the structured ERP
    database (suppliers, POs, shipments, inventory, contracts) and the
    indexed internal knowledge base (contracts, policies, correspondence,
    historical incidents). Never invents an answer -- if neither source
    has relevant evidence, says so explicitly rather than guessing."""
    from logisense.services.investigate import InvestigationService
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient, EmbeddingModelLoadError
    from logisense.services.rag.hybrid_context import HybridContextBuilder
    from logisense.services.rag.hybrid_investigation import HybridInvestigationService
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    try:
        embedding_client = BGEEmbeddingClient(settings)
    except EmbeddingModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    vector_store = ChromaVectorStore(settings)
    context_builder = HybridContextBuilder(
        session, embedding_client, vector_store,
        top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
    )
    hybrid_service = HybridInvestigationService(context_builder, settings)

    svc = InvestigationService(session, hybrid_service=hybrid_service)
    result = svc.ask(request.question)

    return AskResponse(
        question=result.question, answer=result.answer, sufficient_evidence=result.sufficient_evidence,
        citations=[
            EvidenceCitationResponse(
                reference_number=c.reference_number, kind=c.kind, label=c.label,
                source_table=c.source_table, source_id=c.source_id, category=c.category,
                filename=c.filename, section=c.section, relevance_score=c.relevance_score,
            )
            for c in result.citations
        ],
    )



# ---------------------------------------------------------------------------
# Hypothesis -> Evidence Retrieval -> Hypothesis Validation -> Grounded
# Finding. Extends the existing investigation surface (same run_id, same
# hypotheses already generated above) rather than a separate endpoint tree.
# ---------------------------------------------------------------------------

class HypothesisValidationItemResponse(BaseModel):
    hypothesis_id: int
    rank: int
    statement: str
    status: str
    confidence: int
    reasoning: str
    contradicting_evidence_summary: str
    evidence_item_count: int
    citations: list[dict]


class GroundedInvestigationResponse(BaseModel):
    run_id: str
    hypothesis_validations: list[HypothesisValidationItemResponse]
    root_cause_hypothesis_id: int | None
    confidence: int
    explanation: str
    supporting_evidence_summary: str
    contradicting_evidence_summary: str
    remaining_uncertainty: str
    evidence_gaps: str
    recommended_next_action: str
    generated_at: str | None = None


@app.post("/investigations/{run_id}/verdict", response_model=GroundedInvestigationResponse)
def create_verdict(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
    include_external: bool = False,
) -> GroundedInvestigationResponse:
    """Runs hybrid structured+RAG evidence retrieval and validation for
    every hypothesis already generated on this run, and produces the
    final grounded investigation finding. Requires hypotheses to already
    exist (POST /investigations/{run_id}/hypotheses first).

    include_external mirrors the CLI's `validate-hypotheses --include-
    external` flag -- also queries weather/maritime/news APIs where
    relevant to each hypothesis. Off by default, same as the CLI."""
    from logisense.services.evidence_validation import HypothesisEvidenceValidator
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient, EmbeddingModelLoadError
    from logisense.services.rag.bm25_index import BM25Index
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    try:
        embedding_client = BGEEmbeddingClient(settings)
    except EmbeddingModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    bm25_index = None
    try:
        bm25_index = BM25Index.load(settings.bm25_index_path)
    except FileNotFoundError:
        pass  # falls back to dense-only retrieval -- not a hard error

    validator = HypothesisEvidenceValidator(
        session, embedding_client, ChromaVectorStore(settings), settings,
        bm25_index=bm25_index, rrf_k=settings.rrf_k, enable_external_evidence=include_external,
    )
    try:
        result = validator.validate_run(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    from logisense.repositories import VerdictRepository
    persisted = VerdictRepository(session).get_by_run_id(run_id)

    return GroundedInvestigationResponse(
        run_id=result.run_id,
        hypothesis_validations=[
            HypothesisValidationItemResponse(
                hypothesis_id=r.hypothesis_id, rank=r.rank, statement=r.statement, status=r.status,
                confidence=r.confidence, reasoning=r.reasoning,
                contradicting_evidence_summary=r.contradicting_evidence_summary,
                evidence_item_count=len(r.evidence_items), citations=r.citations,
            )
            for r in result.hypothesis_results
        ],
        root_cause_hypothesis_id=result.root_cause_hypothesis_id, confidence=result.confidence,
        explanation=result.explanation, supporting_evidence_summary=result.supporting_evidence_summary,
        contradicting_evidence_summary=result.contradicting_evidence_summary,
        remaining_uncertainty=result.remaining_uncertainty, evidence_gaps=result.evidence_gaps,
        recommended_next_action=result.recommended_next_action,
        generated_at=persisted.generated_at.isoformat() if persisted else None,
    )


@app.get("/investigations/{run_id}/verdict", response_model=GroundedInvestigationResponse)
def get_verdict(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> GroundedInvestigationResponse:
    """Returns the already-persisted grounded finding for this run
    without re-running validation."""
    from logisense.repositories import HypothesisRepository, VerdictRepository

    verdict = VerdictRepository(session).get_by_run_id(run_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail=f"No verdict found for run {run_id}. POST first to generate one.")

    return GroundedInvestigationResponse(
        run_id=run_id,
        hypothesis_validations=[
            HypothesisValidationItemResponse(
                hypothesis_id=u["hypothesis_id"], rank=u["rank"], statement=u["statement"],
                status=u["validation_status"], confidence=u["confidence"], reasoning=u["reasoning"],
                contradicting_evidence_summary=u.get("contradicting_evidence_summary", ""),
                evidence_item_count=len(u.get("citations", [])), citations=u.get("citations", []),
            )
            for u in verdict.hypothesis_confidence_updates
        ],
        root_cause_hypothesis_id=verdict.root_cause_hypothesis_id, confidence=int(verdict.confidence_score),
        explanation=verdict.explanation, supporting_evidence_summary=verdict.supporting_evidence_summary,
        contradicting_evidence_summary=verdict.contradicting_evidence_summary,
        remaining_uncertainty=verdict.remaining_uncertainty, evidence_gaps=verdict.evidence_gaps,
        recommended_next_action=verdict.recommended_next_action,
        generated_at=verdict.generated_at.isoformat() if verdict.generated_at else None,
    )


# ---------------------------------------------------------------------------
# Multi-source analyze: internal DB + internal RAG + external real-time
# data (weather/maritime/news), exactly what `python -m logisense.cli
# analyze "<question>"` already does. This is a thin route onto the
# existing MultiSourceInvestigationService -- no retrieval/ranking/
# prompting logic is duplicated or reimplemented here.
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=1)


class AnalyzeCitationResponse(BaseModel):
    citation_text: str
    source_type: str
    source_name: str
    source_id: str
    content: str
    title: str | None = None
    relevance_score: float | None = None


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    source_status: dict[str, str]
    citations: list[AnalyzeCitationResponse]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AnalyzeResponse:
    """Free-text supply-chain question, answered from internal database
    facts + internal knowledge-base documents + external real-time data
    (Open-Meteo weather, GDELT events/news, AISstream.io maritime data)
    where relevant -- with explicit, traceable citations for every claim.
    Says so explicitly, rather than guessing, when no source has enough
    evidence (e.g. a question that names an entity in a format the
    entity extractor doesn't recognize, such as a plain description
    instead of a SUP-/PO-/SHIP-/CTR- id)."""
    from logisense.services.multi_source_investigation import MultiSourceInvestigationService
    from logisense.services.rag.bge_embeddings import BGEEmbeddingClient, EmbeddingModelLoadError
    from logisense.services.rag.hybrid_context import HybridContextBuilder
    from logisense.services.rag.vector_store import ChromaVectorStore

    settings = get_settings()
    try:
        embedding_client = BGEEmbeddingClient(settings)
    except EmbeddingModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    context_builder = HybridContextBuilder(
        session, embedding_client, ChromaVectorStore(settings),
        top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity,
    )
    service = MultiSourceInvestigationService(session, context_builder, settings)
    try:
        result = service.answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analyze failed: {exc}") from exc

    return AnalyzeResponse(
        question=result.question, answer=result.answer, sufficient_evidence=result.sufficient_evidence,
        source_status=result.source_status,
        citations=[
            AnalyzeCitationResponse(
                citation_text=c.citation_text, source_type=c.source_type, source_name=c.source_name,
                source_id=c.source_id, content=c.content, title=c.title, relevance_score=c.relevance_score,
            )
            for c in result.citations
        ],
    )


# ---------------------------------------------------------------------------
# Phase 4: mitigation strategy generation -- thin routes onto the existing
# MitigationGenerationService, exactly what
# `python -m logisense.cli generate-mitigations --run-id <RUN_ID>` already
# does. No mitigation business logic (category rules, validation,
# priority/urgency/confidence scoring) is duplicated here -- the service
# owns all of it and persists via MitigationRepository itself.
# ---------------------------------------------------------------------------

class MitigationStrategyResponse(BaseModel):
    id: int
    strategy_key: str
    title: str
    strategy_type: str
    description: str
    why_this_mitigation: str
    investigation_basis: list[str]
    risk_addressed: str
    implementation_steps: list[str]
    expected_impact: str
    priority: str
    urgency: str
    confidence: float
    limitations: str
    dependencies: list[str]
    success_metrics: list[str]
    is_confirmed_corrective_action: bool
    validation_status: str
    validation_notes: str | None = None
    generated_at: str | None = None


class MitigationSetResponse(BaseModel):
    run_id: str
    root_cause_status: str
    categories_identified: list[str]
    rejected_count: int
    strategies: list[MitigationStrategyResponse]


def _mitigation_strategy_response(s) -> "MitigationStrategyResponse":
    return MitigationStrategyResponse(
        id=s.id, strategy_key=s.strategy_key, title=s.title, strategy_type=s.strategy_type,
        description=s.description, why_this_mitigation=s.why_this_mitigation,
        investigation_basis=s.investigation_basis, risk_addressed=s.risk_addressed,
        implementation_steps=s.implementation_steps, expected_impact=s.expected_impact,
        priority=s.priority.value, urgency=s.urgency.value, confidence=float(s.confidence),
        limitations=s.limitations, dependencies=s.dependencies, success_metrics=s.success_metrics,
        is_confirmed_corrective_action=s.is_confirmed_corrective_action,
        validation_status=s.validation_status.value, validation_notes=s.validation_notes,
        generated_at=s.generated_at.isoformat() if s.generated_at else None,
    )


@app.post("/investigations/{run_id}/mitigations", response_model=MitigationSetResponse)
def create_mitigations(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> MitigationSetResponse:
    """Generates mitigation strategies from this run's already-persisted
    grounded finding (requires POST /investigations/{run_id}/verdict to
    have run first). Same service as the CLI's generate-mitigations."""
    from logisense.services.mitigation import MitigationGenerationService

    settings = get_settings()
    try:
        result = MitigationGenerationService(session, settings).generate(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Mitigation generation failed: {exc}") from exc

    return MitigationSetResponse(
        run_id=result.run_id, root_cause_status=result.root_cause_status,
        categories_identified=result.categories_identified, rejected_count=result.rejected_count,
        strategies=[_mitigation_strategy_response(s) for s in result.strategies],
    )


@app.get("/investigations/{run_id}/mitigations", response_model=MitigationSetResponse)
def get_mitigations(
    run_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> MitigationSetResponse:
    """Returns the already-persisted mitigation strategies for this run
    without calling the LLM again. root_cause_status/categories_identified
    are recomputed deterministically from the persisted Verdict (no Groq
    call involved) so the response shape matches the POST response."""
    from logisense.repositories import MitigationRepository
    from logisense.services.mitigation import load_investigation_summary, identify_mitigation_categories, extract_factors

    strategies = MitigationRepository(session).get_by_run_id(run_id)
    if not strategies:
        raise HTTPException(status_code=404, detail=f"No mitigation strategies found for run {run_id}. POST first to generate them.")

    try:
        summary = load_investigation_summary(session, run_id)
        factors = extract_factors(summary)
        categories = [c.key for c in identify_mitigation_categories(summary, factors)]
        root_cause_status = summary.root_cause_status
    except LookupError:
        root_cause_status = "inconclusive"
        categories = sorted({s.strategy_type for s in strategies})

    return MitigationSetResponse(
        run_id=run_id, root_cause_status=root_cause_status, categories_identified=categories,
        rejected_count=0,
        strategies=[_mitigation_strategy_response(s) for s in strategies],
    )
