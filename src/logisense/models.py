"""Canonical supply-chain schema + Phase 1 (incident intake & triage) models.

These models are intentionally dataset-agnostic. Nothing here assumes a
specific source dataset (Olist, DataCo, Walmart, ...). Any dataset gets
mapped into this canonical shape by a separate ingestion/normalization
layer (not implemented here). Optional business fields stay optional so
the schema does not force datasets to have data they don't contain.
"""
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from logisense.db import Base


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    """Make SQLAlchemy persist/compare enum .value (e.g. "pending"),
    not the Python member .name (e.g. "PENDING"). Without this,
    SQLAlchemy's default Enum type sends member names to the database,
    which does not match the lowercase values the Postgres ENUM types
    (and this app's Gemini/API contracts) actually use."""
    return [member.value for member in enum_cls]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InvestigationRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InvestigationStage(str, Enum):
    INCIDENT_INTAKE = "incident_intake"
    READY_FOR_HYPOTHESIS = "ready_for_hypothesis"
    HYPOTHESIS_GENERATED = "hypothesis_generated"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentImpact(str, Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"


class InvestigationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HypothesisConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Canonical supply-chain schema
# ---------------------------------------------------------------------------

class Location(Base):
    """Generic reusable location dimension (used for origin/destination/
    supplier/customer locations). All fields optional: a dataset may only
    provide a country, or only a city, etc."""

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    city: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(128))
    region: Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (
        UniqueConstraint("city", "state", "country", "region", name="uq_locations_identity"),
        Index("ix_locations_country_region", "country", "region"),
    )


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_product_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Supplier(Base):
    """Supplier / seller. Optional dataset concept -- a dataset without a
    supplier/seller notion simply never populates this table and Order.
    supplier_id stays NULL."""

    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_supplier_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    location: Mapped[Location | None] = relationship()


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_customer_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    location: Mapped[Location | None] = relationship()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    order_date: Mapped[date | None] = mapped_column(Date)
    order_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    quantity: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(128))

    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"))
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"))

    customer: Mapped[Customer | None] = relationship()
    product: Mapped[Product | None] = relationship()
    supplier: Mapped[Supplier | None] = relationship()

    __table_args__ = (
        CheckConstraint("quantity IS NULL OR quantity >= 0"),
        Index("ix_orders_order_date", "order_date"),
    )


class Shipment(Base):
    """Shipment/delivery record. Delay/deviation fields are derived (not
    raw dataset columns) so they work the same way regardless of which
    source columns a dataset originally used."""

    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), unique=True)
    """Nullable: a Shipment fulfills EITHER a customer Order (outbound leg,
    the original use of this table) OR a PurchaseOrder (inbound, supplier
    -> warehouse leg, added for the ERP layer). Every existing code path
    that reads shipment.order only does so via an Incident's shipment,
    and incidents are only ever created from Order-linked shipments, so
    this relaxation does not change behavior for any existing caller."""
    purchase_order_id: Mapped[int | None] = mapped_column(ForeignKey("purchase_orders.id"), unique=True)

    origin_location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    destination_location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))

    shipping_mode: Mapped[str | None] = mapped_column(String(128))
    carrier_id: Mapped[int | None] = mapped_column(ForeignKey("carriers.id"))
    freight_cost: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))

    expected_delivery_date: Mapped[date | None] = mapped_column(Date)
    actual_delivery_date: Mapped[date | None] = mapped_column(Date)
    delay_days: Mapped[int | None] = mapped_column(Integer)

    order: Mapped["Order | None"] = relationship()
    purchase_order: Mapped["PurchaseOrder | None"] = relationship()
    carrier: Mapped["Carrier | None"] = relationship()
    origin: Mapped[Location | None] = relationship(foreign_keys=[origin_location_id])
    destination: Mapped[Location | None] = relationship(foreign_keys=[destination_location_id])

    __table_args__ = (
        Index("ix_shipments_delay_days", "delay_days"),
        Index("ix_shipments_expected_actual", "expected_delivery_date", "actual_delivery_date"),
        CheckConstraint(
            "(order_id IS NOT NULL) OR (purchase_order_id IS NOT NULL)",
            name="ck_shipment_has_order_or_po",
        ),
    )


class Incident(Base):
    """A data-derived operational incident. incident_type/rationale here
    are purely factual/data-derived (e.g. 'delivery_delay', 'delay_days=12
    exceeds expected delivery date'). Interpretive severity classification
    happens in Phase 1 (IncidentIntake), not here."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shipment_id: Mapped[int] = mapped_column(ForeignKey("shipments.id"), nullable=False, unique=True)
    incident_type: Mapped[str] = mapped_column(String(128), nullable=False)
    deviation_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    shipment: Mapped[Shipment] = relationship()

    __table_args__ = (Index("ix_incidents_type", "incident_type"),)


# ---------------------------------------------------------------------------
# Phase 1: investigation runs + incident intake
# ---------------------------------------------------------------------------

class InvestigationRun(Base):
    __tablename__ = "investigation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), nullable=False)
    status: Mapped[InvestigationRunStatus] = mapped_column(
        SAEnum(InvestigationRunStatus, name="investigation_run_status", values_callable=_enum_values),
        nullable=False,
    )
    current_stage: Mapped[InvestigationStage] = mapped_column(
        SAEnum(InvestigationStage, name="investigation_stage", values_callable=_enum_values),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    contextual_metrics: Mapped[dict | None] = mapped_column(JSON)
    """Snapshot of ContextualMetricsRepository.build_profile(...).to_dict()
    at the moment this run was triaged -- i.e. exactly what Gemini saw.
    This is deterministic, SQL-derived data (not the LLM's response), so
    storing it does not violate the "no raw LLM JSON blob" rule; it lets
    the frontend and any later processing reuse the same metrics without
    recomputing them, and gives a historical record of what informed a
    given severity call even if live metrics later drift."""

    ruled_out_hypotheses: Mapped[list | None] = mapped_column(JSON)
    """Phase 2: candidate explanations Gemini considered and explicitly
    ruled out, as [{"statement": ..., "reason_ruled_out": ...}, ...].
    Kept separate from the `investigation_hypotheses` table since these
    were never surviving candidates -- they're the agent's rule-out
    reasoning, not investigation targets."""

    incident: Mapped[Incident] = relationship()

    __table_args__ = (
        Index("ix_investigation_runs_incident_started", "incident_id", "started_at"),
        Index("ix_investigation_runs_status_stage", "status", "current_stage"),
    )


class IncidentIntake(Base):
    """Phase 1 output -- the Triage Agent's assessment.

    `triage_assessment` holds the complete, verbatim, human-readable text
    Gemini returned (INCIDENT TYPE / SEVERITY / WHAT HAPPENED / WHY THIS
    SEVERITY / POTENTIAL BUSINESS IMPACT / IMPORTANT SIGNALS /
    INVESTIGATION PRIORITIES / INITIAL ASSESSMENT) -- this is the single
    source of truth for everything the UI needs to display, never
    reconstructed or paraphrased.

    incident_type/severity are additionally parsed out into typed columns
    purely so existing typed consumers (API filters, the dashboard's
    severity badge) keep working; rationale/recommended_next_step/
    normalized_summary mirror the closest-matching sections for the same
    reason. impact/priority are no longer produced by this prompt (it
    yields a business-impact narrative and an investigation checklist
    instead of discrete enum levels), so both are nullable and left NULL.

    Never stored as a raw JSON blob, and never handed off via a file --
    PostgreSQL is the sole Phase 1 -> Phase 2 handoff mechanism."""

    __tablename__ = "incident_intake"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("investigation_runs.run_id"), nullable=False, unique=True
    )
    incident_type: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[IncidentSeverity] = mapped_column(
        SAEnum(IncidentSeverity, name="incident_severity", values_callable=_enum_values), nullable=False
    )
    impact: Mapped[IncidentImpact | None] = mapped_column(
        SAEnum(IncidentImpact, name="incident_impact", values_callable=_enum_values), nullable=True
    )
    priority: Mapped[InvestigationPriority | None] = mapped_column(
        SAEnum(InvestigationPriority, name="investigation_priority", values_callable=_enum_values),
        nullable=True,
    )
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_next_step: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_summary: Mapped[str] = mapped_column(Text, nullable=False)
    triage_assessment: Mapped[str] = mapped_column(Text, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[InvestigationRun] = relationship()

    __table_args__ = (Index("ix_incident_intake_type_severity", "incident_type", "severity"),)


# ---------------------------------------------------------------------------
# Phase 2: hypothesis generation
# ---------------------------------------------------------------------------

class Hypothesis(Base):
    """A single Phase 2 candidate root-cause hypothesis for a run. Never a
    confirmed root cause -- confirmation/refutation is later phases' job
    (evidence retrieval + verification). `rank` reflects Gemini's own
    qualitative ordering of plausibility (1 = most plausible), not a
    computed probability."""

    __tablename__ = "investigation_hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("investigation_runs.run_id"), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_signals: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[HypothesisConfidence] = mapped_column(
        SAEnum(HypothesisConfidence, name="hypothesis_confidence", values_callable=_enum_values),
        nullable=False,
    )
    what_would_confirm: Mapped[str] = mapped_column(Text, nullable=False)
    what_would_refute: Mapped[str] = mapped_column(Text, nullable=False)
    why_ranked_here: Mapped[str] = mapped_column(Text, nullable=False)
    """Why this hypothesis sits at this rank relative to the others --
    explicit rule-out/comparison reasoning, not just a silent ordering."""
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[InvestigationRun] = relationship()

    __table_args__ = (
        UniqueConstraint("run_id", "rank", name="uq_hypothesis_run_rank"),
        Index("ix_investigation_hypotheses_run", "run_id"),
    )


# ---------------------------------------------------------------------------
# Phase 3: evidence retrieval & validation
# ---------------------------------------------------------------------------

class EvidenceSourceCategory(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class ProviderStatus(str, Enum):
    """What happened when a source was queried -- distinct from whether
    evidence was found. NO_EVIDENCE_FOUND is not treated as evidence
    against a hypothesis; UNAVAILABLE/ERROR must never be silently
    dropped or presented as if the source had nothing to say."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NO_EVIDENCE_FOUND = "no_evidence_found"


class EvidenceStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceSource(Base):
    __tablename__ = "evidence_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("investigation_runs.run_id"), nullable=False)
    category: Mapped[EvidenceSourceCategory] = mapped_column(
        SAEnum(EvidenceSourceCategory, name="evidence_source_category", values_callable=_enum_values),
        nullable=False,
    )
    provider_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ProviderStatus] = mapped_column(
        SAEnum(ProviderStatus, name="provider_status", values_callable=_enum_values), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    queried_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[InvestigationRun] = relationship()

    __table_args__ = (Index("ix_evidence_sources_run", "run_id"),)


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("evidence_sources.id"), nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_reference: Mapped[str | None] = mapped_column(String(512))
    relevance_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    reliability_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    evidence_metadata: Mapped[dict | None] = mapped_column(JSON)

    source: Mapped[EvidenceSource] = relationship()

    __table_args__ = (Index("ix_evidence_source", "source_id"),)


class HypothesisEvidence(Base):
    __tablename__ = "hypothesis_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(ForeignKey("investigation_hypotheses.id"), nullable=False)
    evidence_id: Mapped[int] = mapped_column(ForeignKey("evidence.id"), nullable=False)
    evidence_question: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    hypothesis: Mapped[Hypothesis] = relationship()
    evidence: Mapped[Evidence] = relationship()

    __table_args__ = (
        UniqueConstraint("hypothesis_id", "evidence_id", name="uq_hypothesis_evidence"),
        Index("ix_hypothesis_evidence_hypothesis", "hypothesis_id"),
    )


class EvidenceAssessment(Base):
    __tablename__ = "evidence_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hypothesis_evidence_id: Mapped[int] = mapped_column(
        ForeignKey("hypothesis_evidence.id"), nullable=False, unique=True
    )
    supports_hypothesis: Mapped[bool] = mapped_column(nullable=False)
    contradicts_hypothesis: Mapped[bool] = mapped_column(nullable=False)
    strength: Mapped[EvidenceStrength] = mapped_column(
        SAEnum(EvidenceStrength, name="evidence_strength", values_callable=_enum_values), nullable=False
    )
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    hypothesis_evidence: Mapped[HypothesisEvidence] = relationship()

    __table_args__ = (
        CheckConstraint(
            "NOT (supports_hypothesis AND contradicts_hypothesis)",
            name="ck_evidence_assessment_not_both",
        ),
    )


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("investigation_runs.run_id"), nullable=False, unique=True
    )
    root_cause_hypothesis_id: Mapped[int | None] = mapped_column(
        ForeignKey("investigation_hypotheses.id")
    )
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    contradicting_evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    internal_evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    external_evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    remaining_uncertainty: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_gaps: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_next_action: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis_confidence_updates: Mapped[list] = mapped_column(JSON, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[InvestigationRun] = relationship()
    root_cause_hypothesis: Mapped[Hypothesis | None] = relationship()


# ---------------------------------------------------------------------------
# Phase 4: mitigation strategy generation
#
# Consumes ONLY the persisted Phase 3 output (Verdict + Hypothesis +
# Evidence rows for this run_id) -- it never re-investigates, never
# re-queries ERP tables directly, and never generates a new root cause.
# See services/mitigation.py for the loader that reconstructs the
# investigation summary these rows are scored/validated against.
# ---------------------------------------------------------------------------

class MitigationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MitigationUrgency(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class MitigationValidationStatus(str, Enum):
    """Outcome of the deterministic validation pass (see
    services/mitigation.py: validate_strategy()) run on every strategy
    Groq proposes, before it is shown to anyone."""

    VALID = "valid"
    FLAGGED = "flagged"  # kept, but a validation concern is recorded (e.g. weak evidence match)
    REJECTED = "rejected"  # dropped -- e.g. asserted an unconfirmed root cause as fact


class MitigationStrategy(Base):
    """One Phase 4 mitigation strategy for a run. `strategy_type` is the
    deterministic category (see services/mitigation.py:
    identify_mitigation_categories()) it was generated for -- Groq fills
    in the descriptive fields, but never invents the category itself."""

    __tablename__ = "mitigation_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("investigation_runs.run_id"), nullable=False)
    strategy_key: Mapped[str] = mapped_column(String(64), nullable=False)
    """Stable identifier within this run (e.g. 'supplier-performance-1'),
    NOT a database primary key -- lets the frontend/CLI reference a
    specific strategy without depending on the numeric id."""
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    why_this_mitigation: Mapped[str] = mapped_column(Text, nullable=False)
    investigation_basis: Mapped[list] = mapped_column(JSON, nullable=False)
    risk_addressed: Mapped[str] = mapped_column(Text, nullable=False)
    implementation_steps: Mapped[list] = mapped_column(JSON, nullable=False)
    expected_impact: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[MitigationPriority] = mapped_column(
        SAEnum(MitigationPriority, name="mitigation_priority", values_callable=_enum_values), nullable=False
    )
    urgency: Mapped[MitigationUrgency] = mapped_column(
        SAEnum(MitigationUrgency, name="mitigation_urgency", values_callable=_enum_values), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    """0-100, same convention as Verdict.confidence_score -- deliberately
    NOT copied from root-cause confidence (a preventive strategy can be
    highly relevant even when the root cause is inconclusive); see
    services/mitigation.py: score_confidence()."""
    limitations: Mapped[str] = mapped_column(Text, nullable=False)
    dependencies: Mapped[list] = mapped_column(JSON, nullable=False)
    success_metrics: Mapped[list] = mapped_column(JSON, nullable=False)
    is_confirmed_corrective_action: Mapped[bool] = mapped_column(nullable=False, default=False)
    """True only when this strategy addresses the investigation's
    CONFIRMED root cause; False means it's a preventive/risk-reduction
    strategy -- the distinction Section 9 of the spec requires never be
    blurred."""
    validation_status: Mapped[MitigationValidationStatus] = mapped_column(
        SAEnum(MitigationValidationStatus, name="mitigation_validation_status", values_callable=_enum_values),
        nullable=False,
    )
    validation_notes: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[InvestigationRun] = relationship()

    __table_args__ = (
        UniqueConstraint("run_id", "strategy_key", name="uq_mitigation_run_strategy_key"),
        Index("ix_mitigation_strategies_run", "run_id"),
    )


# ---------------------------------------------------------------------------
# ERP layer -- reuses Supplier/Product/Customer/Order/Shipment/Location.
# New tables only for genuinely new concepts.
# ---------------------------------------------------------------------------

class PurchaseOrderStatus(str, Enum):
    PLACED = "placed"
    IN_PRODUCTION = "in_production"
    READY_FOR_SHIPMENT = "ready_for_shipment"
    SHIPPED = "shipped"
    DELAYED = "delayed"
    RECEIVED = "received"
    CLOSED = "closed"


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_reference: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), nullable=False)
    warehouse_id: Mapped[int | None] = mapped_column(ForeignKey("warehouses.id"))
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int | None] = mapped_column(Integer)
    order_date: Mapped[date | None] = mapped_column(Date)
    expected_ready_date: Mapped[date | None] = mapped_column(Date)
    total_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    status: Mapped[PurchaseOrderStatus] = mapped_column(
        SAEnum(PurchaseOrderStatus, name="purchase_order_status", values_callable=_enum_values),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    supplier: Mapped[Supplier] = relationship()
    warehouse: Mapped["Warehouse | None"] = relationship()
    product: Mapped[Product | None] = relationship()

    __table_args__ = (Index("ix_purchase_orders_supplier", "supplier_id"),)


class PurchaseOrderItem(Base):
    __tablename__ = "purchase_order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    purchase_order_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_cost: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    expected_quantity: Mapped[int | None] = mapped_column(Integer)
    expected_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    purchase_order: Mapped[PurchaseOrder] = relationship()
    product: Mapped[Product] = relationship()

    __table_args__ = (Index("ix_purchase_order_items_po", "purchase_order_id"),)


class Warehouse(Base):
    __tablename__ = "warehouses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    capacity: Mapped[int | None] = mapped_column(Integer)
    current_utilization: Mapped[float | None] = mapped_column(Numeric(5, 3))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    location: Mapped[Location | None] = relationship()


class Inventory(Base):
    __tablename__ = "inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouses.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    on_hand: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved: Mapped[int | None] = mapped_column(Integer)
    available: Mapped[int | None] = mapped_column(Integer)
    reorder_point: Mapped[int | None] = mapped_column(Integer)
    safety_stock: Mapped[int | None] = mapped_column(Integer)
    daily_demand: Mapped[float | None] = mapped_column(Numeric(10, 2))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    warehouse: Mapped[Warehouse] = relationship()
    product: Mapped[Product] = relationship()

    __table_args__ = (UniqueConstraint("warehouse_id", "product_id", name="uq_inventory_warehouse_product"),)


class Carrier(Base):
    __tablename__ = "carriers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_carrier_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    service_type: Mapped[str | None] = mapped_column(String(128))
    region: Mapped[str | None] = mapped_column(String(128))
    on_time_rate: Mapped[float | None] = mapped_column(Numeric(5, 3))
    avg_transit_days: Mapped[float | None] = mapped_column(Numeric(6, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ShipmentEventType(str, Enum):
    CREATED = "created"
    PICKUP = "pickup"
    DEPARTURE = "departure"
    IN_TRANSIT = "in_transit"
    CUSTOMS = "customs"
    CUSTOMS_HOLD = "customs_hold"
    CARRIER_DELAY = "carrier_delay"
    WEATHER_DELAY = "weather_delay"
    QUALITY_HOLD = "quality_hold"
    ARRIVAL = "arrival"
    DELIVERED = "delivered"
    RECEIVED = "received"
    EXCEPTION = "exception"


class ShipmentEvent(Base):
    __tablename__ = "shipment_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shipment_id: Mapped[int] = mapped_column(ForeignKey("shipments.id"), nullable=False)
    event_type: Mapped[ShipmentEventType] = mapped_column(
        SAEnum(ShipmentEventType, name="shipment_event_type", values_callable=_enum_values), nullable=False
    )
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    location_label: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    shipment: Mapped[Shipment] = relationship()
    location: Mapped[Location | None] = relationship()

    __table_args__ = (Index("ix_shipment_events_shipment", "shipment_id", "event_time"),)


class Contract(Base):
    """Supplier contract header. New model -- the earlier ERP work never
    had a distinct Contract entity; the requested chain (Supplier ->
    Contract -> Purchase Order -> ...) needs one."""

    __tablename__ = "contracts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_contract_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    agreed_lead_time: Mapped[int | None] = mapped_column(Integer)
    minimum_order_quantity: Mapped[int | None] = mapped_column(Integer)
    sla: Mapped[str | None] = mapped_column(Text)
    penalty_terms: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    supplier: Mapped[Supplier] = relationship()

    __table_args__ = (Index("ix_contracts_supplier", "supplier_id"),)


# ---------------------------------------------------------------------------
# RAG layer: document generation/ingestion, chunking, embedding-backed
# retrieval. Postgres holds provenance; the vector store (ChromaDB) holds
# the actual embeddings.
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_reference: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("source_type", "source_reference", name="uq_document_source"),
        Index("ix_documents_source", "source_type", "source_reference"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    vector_store_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship()

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        Index("ix_document_chunks_document", "document_id"),
    )
