"""Phase 4: Mitigation Strategy Generation.

    Incident Detection -> Investigation -> Hypothesis Generation ->
    Hypothesis Validation -> Evidence / Grounded Finding ->
    MITIGATION STRATEGY GENERATION -> MITIGATION VALIDATION ->
    FINAL MITIGATION RECOMMENDATIONS

This layer consumes ONLY the persisted Phase 3 output for a run_id --
the Verdict, Hypothesis, and Evidence rows `validate-hypotheses` already
wrote to the database (see load_investigation_summary() below). It never
re-investigates, never regenerates hypotheses, never determines a new
root cause, and never queries ERP tables directly: everything it knows
about the supplier/carrier/shipment/PO/inventory comes from the evidence
TEXT the Investigation layer already gathered and persisted.

    Investigation: "What happened and what can we establish?"
    Mitigation:    "Given what the investigation established, what can
                     we do to reduce the risk or impact?"

Pipeline (see MitigationGenerationService.generate()):
    1. load_investigation_summary()      -- DB rows -> InvestigationSummary
    2. extract_factors()                 -- deterministic factor extraction
    3. identify_mitigation_categories()  -- deterministic rule-based categories
    4. Groq generates descriptive strategies grounded in exactly that
       material (the LLM is never asked for a category, a priority, or a
       confidence score -- only for the descriptive fields).
    5. validate_strategy()               -- deterministic validation pass
    6. score_strategy()                  -- deterministic priority/urgency/confidence
    7. MitigationRepository persists the result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import groq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from logisense import models as m
from logisense.config import Settings
from logisense.repositories import (
    HypothesisRepository,
    InvestigationRunRepository,
    MitigationRepository,
    VerdictRepository,
)
from logisense.services.gemini_client import GroqClient, build_gemini_client

logger = logging.getLogger(__name__)

RootCauseStatus = Literal["confirmed", "inconclusive"]


# ---------------------------------------------------------------------------
# 1. Load the investigation's own output -- no re-investigation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceFact:
    category: str
    text: str
    reference: str


@dataclass(frozen=True)
class HypothesisSummary:
    hypothesis_id: int
    rank: int
    statement: str
    initial_confidence: str  # Phase 2's own low/medium/high label
    validation_status: str | None  # Phase 3's supported/partially_supported/unsupported/insufficient_evidence
    validation_confidence: int | None  # 0-100, None if this hypothesis wasn't in the verdict's updates
    reasoning: str | None
    contradicting_evidence_summary: str | None


@dataclass(frozen=True)
class InvestigationSummary:
    run_id: str
    incident_id: int
    incident_type: str
    incident_rationale: str
    hypotheses: list[HypothesisSummary]
    root_cause_hypothesis_id: int | None
    root_cause_status: RootCauseStatus
    root_cause_confidence: float
    explanation: str
    supporting_evidence_summary: str
    contradicting_evidence_summary: str
    remaining_uncertainty: str
    evidence_gaps: str
    recommended_next_action: str
    evidence: list[EvidenceFact]

    @property
    def root_cause_statement(self) -> str | None:
        for h in self.hypotheses:
            if h.hypothesis_id == self.root_cause_hypothesis_id:
                return h.statement
        return None

    @property
    def unconfirmed_hypothesis_statements(self) -> list[str]:
        """Hypotheses the investigation did NOT confirm -- mitigation
        must never present these as established fact."""
        return [h.statement for h in self.hypotheses if h.validation_status != "supported"]


# Every EvidenceItem.reference the evidence_validation.py retrieval step
# writes follows one of these fixed prefixes for internal/database facts,
# or the shared "[EXTERNAL: ...]" citation format for external evidence
# (see services/external/evidence_item.py: citation_text). Anything else
# is RAG document evidence ("{category}/{filename}[, section: ...]").
# Classifying by these prefixes reads the investigation's own output --
# it does not re-derive anything from ERP tables.
_REFERENCE_CATEGORY_PREFIXES = [
    ("supplier_performance:", "supplier_performance"),
    ("suppliers:", "supplier"),
    ("contracts:", "contract"),
    ("purchase_order_items:", "purchase_order_item"),
    ("purchase_orders:", "purchase_order"),
    ("inventory:", "inventory"),
    ("shipments:", "shipment"),
]


def _classify_evidence_reference(reference: str | None) -> str:
    ref = reference or ""
    if ref.startswith("[EXTERNAL:"):
        return "external"
    for prefix, category in _REFERENCE_CATEGORY_PREFIXES:
        if ref.startswith(prefix):
            return category
    return "document"


def load_investigation_summary(session: Session, run_id: str) -> InvestigationSummary:
    """The single entry point that turns Phase 3's persisted state into
    the structure the rest of this module works from. Raises LookupError
    if validate-hypotheses hasn't produced a grounded finding yet --
    mitigation cannot manufacture one."""
    run = InvestigationRunRepository(session).get_by_run_id(run_id)
    if run is None:
        raise LookupError(f"Investigation run not found: {run_id}")

    verdict = VerdictRepository(session).get_by_run_id(run_id)
    if verdict is None:
        raise LookupError(
            f"No grounded finding for run {run_id} -- run validate-hypotheses first. "
            "Mitigation strategies are generated from the investigation's own findings, "
            "never by re-investigating."
        )

    hypotheses = HypothesisRepository(session).get_by_run_id(run_id)
    validation_by_id = {u["hypothesis_id"]: u for u in (verdict.hypothesis_confidence_updates or [])}

    hypothesis_summaries = [
        HypothesisSummary(
            hypothesis_id=h.id, rank=h.rank, statement=h.statement,
            initial_confidence=h.confidence.value,
            validation_status=(validation_by_id.get(h.id) or {}).get("validation_status"),
            validation_confidence=(validation_by_id.get(h.id) or {}).get("confidence"),
            reasoning=(validation_by_id.get(h.id) or {}).get("reasoning"),
            contradicting_evidence_summary=(validation_by_id.get(h.id) or {}).get("contradicting_evidence_summary"),
        )
        for h in hypotheses
    ]

    # Root cause is "confirmed" ONLY when the verdict named a hypothesis
    # AND that hypothesis's own Phase 3 validation status is "supported"
    # -- never inferred, never assumed from the verdict's existence alone.
    root_cause_status: RootCauseStatus = "inconclusive"
    if verdict.root_cause_hypothesis_id is not None:
        matching = validation_by_id.get(verdict.root_cause_hypothesis_id)
        if matching and matching.get("validation_status") == "supported":
            root_cause_status = "confirmed"

    evidence_rows = MitigationRepository(session).get_evidence_for_run(run_id)
    evidence = [
        EvidenceFact(
            category=_classify_evidence_reference(row.evidence_reference),
            text=row.evidence_text, reference=row.evidence_reference or "",
        )
        for row in evidence_rows
    ]

    incident = session.get(m.Incident, run.incident_id)

    return InvestigationSummary(
        run_id=run_id, incident_id=run.incident_id,
        incident_type=incident.incident_type if incident else "unknown",
        incident_rationale=incident.rationale if incident else "",
        hypotheses=hypothesis_summaries,
        root_cause_hypothesis_id=verdict.root_cause_hypothesis_id,
        root_cause_status=root_cause_status,
        root_cause_confidence=float(verdict.confidence_score),
        explanation=verdict.explanation,
        supporting_evidence_summary=verdict.supporting_evidence_summary,
        contradicting_evidence_summary=verdict.contradicting_evidence_summary,
        remaining_uncertainty=verdict.remaining_uncertainty,
        evidence_gaps=verdict.evidence_gaps,
        recommended_next_action=verdict.recommended_next_action,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# 2. Extract deterministic factors from the investigation's own evidence text
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MitigationFactors:
    delay_days: int | None
    carrier_name: str | None
    supplier_reference: str | None
    supplier_on_time_rate: float | None
    supplier_avg_delay_days: float | None
    supplier_max_delay_days: float | None
    has_contract: bool
    penalty_terms: str | None
    inventory_available: int | None
    inventory_reorder_point: int | None
    inventory_low: bool
    has_external_evidence: bool
    has_document_evidence: bool
    root_cause_confirmed: bool
    evidence_gaps_present: bool


_SHIPMENT_RE = re.compile(r"delay_days=(-?\d+),\s*carrier=([^.]+)\.")
_SUPPLIER_PERF_RE = re.compile(r"on-time rate ([\d.]+), average delay ([\d.]+) days, max delay ([\d.]+) days")
_INVENTORY_RE = re.compile(r"available=(\d+), reorder_point=(\d+)")
_PENALTY_RE = re.compile(r"penalty terms:\s*(.+)$")


def extract_factors(summary: InvestigationSummary) -> MitigationFactors:
    """Only uses factors actually present in the investigation output --
    every field is None/False by default and is only ever set from a
    successful regex match against this run's own persisted evidence
    text (never invented, never looked up fresh)."""
    delay_days = carrier_name = None
    supplier_reference = None
    supplier_on_time_rate = supplier_avg_delay_days = supplier_max_delay_days = None
    has_contract = False
    penalty_terms = None
    inventory_available = inventory_reorder_point = None
    has_external_evidence = False
    has_document_evidence = False

    for ev in summary.evidence:
        if ev.category == "shipment":
            match = _SHIPMENT_RE.search(ev.text)
            if match:
                delay_days = int(match.group(1))
                carrier_name = match.group(2).strip()
                if carrier_name.lower() == "unknown":
                    carrier_name = None
        elif ev.category in ("supplier", "supplier_performance") and supplier_reference is None:
            if ":" in ev.reference:
                supplier_reference = ev.reference.split(":", 1)[1]
        if ev.category == "supplier_performance":
            match = _SUPPLIER_PERF_RE.search(ev.text)
            if match:
                supplier_on_time_rate = float(match.group(1))
                supplier_avg_delay_days = float(match.group(2))
                supplier_max_delay_days = float(match.group(3))
        elif ev.category == "contract":
            has_contract = True
            match = _PENALTY_RE.search(ev.text)
            if match:
                penalty_terms = match.group(1).strip()
        elif ev.category == "inventory":
            match = _INVENTORY_RE.search(ev.text)
            if match:
                # Multiple inventory rows can exist (several products/
                # warehouses) -- keep the lowest available-vs-reorder
                # margin, i.e. the most constrained one, as the factor.
                available, reorder_point = int(match.group(1)), int(match.group(2))
                if inventory_available is None or (available - reorder_point) < (inventory_available - (inventory_reorder_point or 0)):
                    inventory_available, inventory_reorder_point = available, reorder_point
        elif ev.category == "external":
            has_external_evidence = True
        elif ev.category == "document":
            has_document_evidence = True

    inventory_low = (
        inventory_available is not None and inventory_reorder_point is not None
        and inventory_available <= inventory_reorder_point
    )

    return MitigationFactors(
        delay_days=delay_days, carrier_name=carrier_name, supplier_reference=supplier_reference,
        supplier_on_time_rate=supplier_on_time_rate, supplier_avg_delay_days=supplier_avg_delay_days,
        supplier_max_delay_days=supplier_max_delay_days, has_contract=has_contract, penalty_terms=penalty_terms,
        inventory_available=inventory_available, inventory_reorder_point=inventory_reorder_point,
        inventory_low=inventory_low, has_external_evidence=has_external_evidence,
        has_document_evidence=has_document_evidence,
        root_cause_confirmed=summary.root_cause_status == "confirmed",
        evidence_gaps_present=bool(summary.evidence_gaps and summary.evidence_gaps.strip()),
    )


# ---------------------------------------------------------------------------
# 3. Deterministic mitigation-category rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MitigationCategory:
    key: str
    title_hint: str
    rationale: str
    # True only when this category corresponds to the investigation's
    # CONFIRMED root cause -- everything else is preventive/risk-reduction,
    # per the spec's Section 9 distinction. Computed deterministically,
    # never left to the LLM to decide.
    is_confirmed_corrective: bool


_SUPPLIER_KEYWORDS = re.compile(r"supplier", re.IGNORECASE)
_CARRIER_KEYWORDS = re.compile(r"carrier|transit|shipping|logistics provider", re.IGNORECASE)


def identify_mitigation_categories(summary: InvestigationSummary, factors: MitigationFactors) -> list[MitigationCategory]:
    """Rule-based only -- identifies WHICH categories apply; the
    descriptive content for each is Groq's job, grounded in the same
    factors/evidence these rules already checked."""
    root_cause_text = (summary.root_cause_statement or "") + " " + summary.explanation
    root_cause_is_supplier = bool(summary.root_cause_status == "confirmed" and _SUPPLIER_KEYWORDS.search(root_cause_text))
    root_cause_is_carrier = bool(summary.root_cause_status == "confirmed" and _CARRIER_KEYWORDS.search(root_cause_text))

    categories: list[MitigationCategory] = []

    if factors.supplier_on_time_rate is not None and factors.supplier_on_time_rate < 0.8:
        categories.append(MitigationCategory(
            key="supplier_performance", title_hint="Supplier Performance / Supplier Diversification",
            rationale=f"Supplier on-time delivery rate is {factors.supplier_on_time_rate:.0%}, below an acceptable threshold.",
            is_confirmed_corrective=root_cause_is_supplier,
        ))

    if factors.carrier_name and factors.delay_days is not None and factors.delay_days > 0:
        categories.append(MitigationCategory(
            key="carrier_contingency", title_hint="Carrier Contingency / Alternate Carrier",
            rationale=f"Shipment via {factors.carrier_name} was delayed {factors.delay_days} day(s).",
            is_confirmed_corrective=root_cause_is_carrier,
        ))

    if factors.delay_days is not None and factors.delay_days > 0:
        categories.append(MitigationCategory(
            key="delay_monitoring", title_hint="Delay Monitoring / Early Warning",
            rationale=f"A confirmed shipment delay of {factors.delay_days} day(s) occurred on this incident.",
            is_confirmed_corrective=False,
        ))

    if factors.inventory_low:
        categories.append(MitigationCategory(
            key="inventory_safety_stock", title_hint="Safety Stock / Emergency Replenishment",
            rationale=f"Available inventory ({factors.inventory_available}) is at or below the reorder point ({factors.inventory_reorder_point}).",
            is_confirmed_corrective=False,
        ))

    if factors.has_contract and factors.penalty_terms and factors.delay_days is not None and factors.delay_days > 0:
        categories.append(MitigationCategory(
            key="contract_enforcement", title_hint="Contract / SLA Enforcement",
            rationale=f"An existing contract defines penalty terms ({factors.penalty_terms}) potentially applicable to this delay.",
            is_confirmed_corrective=False,
        ))

    if factors.has_external_evidence:
        categories.append(MitigationCategory(
            key="external_disruption_contingency", title_hint="External Disruption Contingency",
            rationale="External evidence (weather/maritime/news) was retrieved and considered for this incident.",
            is_confirmed_corrective=False,
        ))

    if summary.root_cause_status == "inconclusive":
        categories.append(MitigationCategory(
            key="risk_reduction_monitoring", title_hint="Risk Reduction / Monitoring / Contingency",
            rationale="The investigation could not establish a confirmed root cause, so risk-reduction and monitoring strategies are appropriate rather than a single corrective action.",
            is_confirmed_corrective=False,
        ))

    if not categories:
        # Always produce at least one strategy -- a completed investigation
        # with nothing else actionable still warrants preventive monitoring.
        categories.append(MitigationCategory(
            key="preventive_monitoring", title_hint="Preventive Monitoring / Process Improvement",
            rationale="No specific high-risk factor was identified beyond the incident itself; general preventive monitoring is still warranted.",
            is_confirmed_corrective=False,
        ))

    return categories


# ---------------------------------------------------------------------------
# 4. Groq structured-output schema + prompt
# ---------------------------------------------------------------------------

class MitigationStrategyItem(BaseModel):
    strategy_type: str
    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=30, max_length=2000)
    why_this_mitigation: str = Field(min_length=20, max_length=1500)
    investigation_basis: list[str] = Field(min_length=1, max_length=8)
    risk_addressed: str = Field(min_length=10, max_length=500)
    implementation_steps: list[str] = Field(min_length=2, max_length=10)
    expected_impact: str = Field(min_length=10, max_length=800)
    limitations: str = Field(min_length=10, max_length=800)
    dependencies: list[str] = Field(default_factory=list, max_length=8)
    success_metrics: list[str] = Field(min_length=1, max_length=8)


class MitigationSetResponse(BaseModel):
    strategies: list[MitigationStrategyItem] = Field(min_length=1, max_length=10)


_MITIGATION_SYSTEM_INSTRUCTIONS = """You are generating mitigation strategies from an existing supply-chain investigation.

Do not perform another investigation.
Do not invent evidence.
Do not invent a root cause.
Use only the supplied investigation findings.

Generate practical, descriptive, evidence-grounded mitigation strategies -- never a one-line platitude like "Use another supplier" or "Monitor the carrier". For every strategy explain:
- what should be done
- why it is relevant
- which investigation evidence supports it (investigation_basis must quote or closely reference facts/numbers actually given below -- never a fact not present here)
- what risk it addresses
- how it can be implemented (concrete steps)
- expected impact
- limitations (explicitly say if this is a preventive/risk-reduction strategy rather than a confirmed corrective action, when the root cause is not confirmed)

Generate exactly one strategy for each mitigation category listed below, using that category's own key as "strategy_type" verbatim.
"""


def _build_mitigation_prompt(summary: InvestigationSummary, factors: MitigationFactors, categories: list[MitigationCategory]) -> str:
    hyp_lines = []
    for h in summary.hypotheses:
        status = h.validation_status or "not validated"
        hyp_lines.append(f"- [{status}, confidence={h.validation_confidence}] {h.statement}")
    hypotheses_block = "\n".join(hyp_lines) if hyp_lines else "(none)"

    evidence_lines = [f"- ({ev.category}) {ev.text}" for ev in summary.evidence]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(none)"

    categories_block = "\n".join(f"- {c.key}: {c.title_hint} -- {c.rationale}" for c in categories)

    root_cause_block = (
        f"Root cause: CONFIRMED -- {summary.root_cause_statement}"
        if summary.root_cause_status == "confirmed"
        else "Root cause: INCONCLUSIVE -- the investigation could not confirm a single root cause. "
             "Do not present any hypothesis below as an established fact; treat all proposed causes as unconfirmed."
    )

    unconfirmed = summary.unconfirmed_hypothesis_statements
    unconfirmed_block = (
        "\n".join(f"- {s}" for s in unconfirmed) if unconfirmed else "(none)"
    )

    return f"""{_MITIGATION_SYSTEM_INSTRUCTIONS}

INCIDENT: {summary.incident_type} -- {summary.incident_rationale}

{root_cause_block}
Root cause confidence: {summary.root_cause_confidence}

Hypotheses considered (with their Phase 3 validation outcome):
{hypotheses_block}

Hypotheses that were NOT confirmed -- do not state these as facts, only as considered-but-unconfirmed possibilities:
{unconfirmed_block}

Grounded finding explanation: {summary.explanation}
Supporting evidence summary: {summary.supporting_evidence_summary}
Contradicting evidence summary: {summary.contradicting_evidence_summary}
Remaining uncertainty: {summary.remaining_uncertainty}
Evidence gaps: {summary.evidence_gaps}

Investigation evidence (the ONLY facts you may cite in investigation_basis):
{evidence_block}

Mitigation categories to generate one strategy for each (use these exact strategy_type keys):
{categories_block}
"""


# ---------------------------------------------------------------------------
# 5. Deterministic validation
# ---------------------------------------------------------------------------

_ASSERTIVE_CAUSE_PHRASES = re.compile(
    r"\b(is the root cause|was the root cause|is the cause of|was the cause of|"
    r"caused the (incident|delay|shipment delay)|definitively caused|confirmed cause)\b",
    re.IGNORECASE,
)
_HEDGE_PHRASES = re.compile(
    r"\b(inconclusive|could not establish|preventive|risk[- ]reduction|potential|"
    r"may have|possible cause|unconfirmed|not confirmed|has not been confirmed)\b",
    re.IGNORECASE,
)
_EVIDENCE_TOKEN_RE = re.compile(r"[A-Za-z]{2,6}-\d+|\d+(?:\.\d+)?%?")


def _basis_is_grounded(basis_line: str, evidence_blob: str) -> bool:
    tokens = _EVIDENCE_TOKEN_RE.findall(basis_line)
    if not tokens:
        return True  # prose with no checkable identifier/number -- don't penalize
    return any(tok in evidence_blob for tok in tokens)


@dataclass(frozen=True)
class ValidationOutcome:
    status: m.MitigationValidationStatus
    notes: str | None


def validate_strategy(
    item: MitigationStrategyItem,
    summary: InvestigationSummary,
    category_keys: set[str],
    evidence_blob: str,
) -> ValidationOutcome:
    issues: list[str] = []

    # Completeness (Pydantic already enforced non-empty/min-length on every
    # field, so this only checks relevance and cross-field consistency).

    # Relevance: the category must be one this run's deterministic rules
    # actually identified -- Groq never gets to invent a category.
    if item.strategy_type not in category_keys:
        return ValidationOutcome(
            m.MitigationValidationStatus.REJECTED,
            f"strategy_type '{item.strategy_type}' is not one of the categories identified for this run: {sorted(category_keys)}",
        )

    # Root cause respect: if the investigation is inconclusive, the
    # strategy must not assert a definitive, unhedged cause.
    combined_text = f"{item.description} {item.why_this_mitigation} {item.risk_addressed}"
    if summary.root_cause_status == "inconclusive":
        asserts_cause = _ASSERTIVE_CAUSE_PHRASES.search(combined_text)
        hedges = _HEDGE_PHRASES.search(combined_text) or _HEDGE_PHRASES.search(item.limitations)
        if asserts_cause and not hedges:
            return ValidationOutcome(
                m.MitigationValidationStatus.REJECTED,
                "asserts a definitive root cause while the investigation's root cause is inconclusive",
            )
        if asserts_cause and hedges:
            issues.append("asserts a cause with some hedging language present -- kept but flagged for review")

    # Evidence grounding: how many investigation_basis lines actually
    # reference something present in this run's evidence/verdict text.
    grounded = [line for line in item.investigation_basis if _basis_is_grounded(line, evidence_blob)]
    grounding_ratio = len(grounded) / len(item.investigation_basis) if item.investigation_basis else 0.0
    if grounding_ratio == 0.0:
        return ValidationOutcome(
            m.MitigationValidationStatus.REJECTED,
            "none of the cited investigation_basis facts could be matched against this run's evidence",
        )
    if grounding_ratio < 0.5:
        issues.append(f"only {len(grounded)}/{len(item.investigation_basis)} investigation_basis lines could be matched against this run's evidence")

    if issues:
        return ValidationOutcome(m.MitigationValidationStatus.FLAGGED, "; ".join(issues))
    return ValidationOutcome(m.MitigationValidationStatus.VALID, None)


# ---------------------------------------------------------------------------
# 6. Deterministic priority / urgency / confidence scoring
# ---------------------------------------------------------------------------

def score_strategy(category: MitigationCategory, summary: InvestigationSummary, factors: MitigationFactors) -> tuple[m.MitigationPriority, m.MitigationUrgency, float]:
    """Transparent, rule-based -- never simply copies root-cause
    confidence (an inconclusive root cause does not mean a preventive
    strategy has zero value)."""
    score = 0.0

    if category.key == "supplier_performance" and factors.supplier_on_time_rate is not None:
        score += (1.0 - factors.supplier_on_time_rate) * 40.0  # weaker on-time rate -> higher score
    if category.key == "carrier_contingency" and factors.delay_days:
        score += min(factors.delay_days, 10) * 3.0
    if category.key == "delay_monitoring" and factors.delay_days:
        score += min(factors.delay_days, 10) * 2.5
    if category.key == "inventory_safety_stock" and factors.inventory_low:
        score += 30.0
    if category.key == "contract_enforcement" and factors.has_contract:
        score += 20.0
    if category.key == "external_disruption_contingency" and factors.has_external_evidence:
        score += 15.0
    if category.key in ("risk_reduction_monitoring", "preventive_monitoring") and summary.root_cause_status == "inconclusive":
        score += 20.0
    if factors.evidence_gaps_present:
        score += 5.0
    if category.is_confirmed_corrective:
        score += 25.0  # directly addresses the investigation's confirmed root cause

    score = min(score, 100.0)

    if score >= 70:
        priority, urgency = m.MitigationPriority.CRITICAL, m.MitigationUrgency.IMMEDIATE
    elif score >= 45:
        priority, urgency = m.MitigationPriority.HIGH, m.MitigationUrgency.IMMEDIATE
    elif score >= 20:
        priority, urgency = m.MitigationPriority.MEDIUM, m.MitigationUrgency.SHORT_TERM
    else:
        priority, urgency = m.MitigationPriority.LOW, m.MitigationUrgency.LONG_TERM

    # Confidence reflects evidence coverage for THIS strategy, not the
    # investigation's root-cause confidence -- deliberately decoupled
    # per the spec (a 0% root-cause confidence does not mean 0%
    # mitigation confidence).
    confidence = 45.0 + (score * 0.4)
    if category.is_confirmed_corrective:
        confidence = max(confidence, summary.root_cause_confidence)
    confidence = round(min(confidence, 97.0), 1)

    return priority, urgency, confidence


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MitigationResult:
    run_id: str
    root_cause_status: RootCauseStatus
    categories_identified: list[str]
    strategies: list[m.MitigationStrategy]
    rejected_count: int


class MitigationGenerationService:
    def __init__(self, session: Session, settings: Settings, groq_client: GroqClient | None = None) -> None:
        self.session = session
        self.settings = settings
        self._client = groq_client
        self.repo = MitigationRepository(session)

    def _groq_client(self) -> GroqClient:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set to generate mitigation strategies")
            self._client = build_gemini_client(self.settings)
        return self._client

    def generate(self, run_id: str) -> MitigationResult:
        logger.info("Loading investigation result for run %s...", run_id)
        summary = load_investigation_summary(self.session, run_id)

        logger.info("Extracting mitigation factors...")
        factors = extract_factors(summary)

        categories = identify_mitigation_categories(summary, factors)
        category_by_key = {c.key: c for c in categories}
        logger.info("Identified %d mitigation categories: %s", len(categories), [c.key for c in categories])

        logger.info("Generating mitigation strategies with Groq...")
        prompt = _build_mitigation_prompt(summary, factors, categories)
        try:
            response = self._groq_client().models.generate_content(
                model=self.settings.groq_model, contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": MitigationSetResponse},
            )
        except groq.APIError as exc:
            logger.warning("Mitigation-generation call for run %s failed after all retry attempts (HTTP %s): %s", run_id, getattr(exc, "status_code", None), exc)
            raise
        if not response.text:
            raise ValueError("Empty response from the model")
        parsed = MitigationSetResponse.model_validate_json(response.text)

        logger.info("Validating mitigation strategies...")
        evidence_blob = "\n".join(ev.text for ev in summary.evidence) + "\n" + summary.explanation + "\n" + summary.supporting_evidence_summary + "\n" + summary.contradicting_evidence_summary

        rows_to_persist: list[dict] = []
        rejected_count = 0
        used_keys: set[str] = set()
        for idx, item in enumerate(parsed.strategies, start=1):
            outcome = validate_strategy(item, summary, set(category_by_key.keys()), evidence_blob)
            if outcome.status == m.MitigationValidationStatus.REJECTED:
                rejected_count += 1
                logger.info("Rejected strategy %r: %s", item.title, outcome.notes)
                continue

            category = category_by_key.get(item.strategy_type)
            priority, urgency, confidence = (
                score_strategy(category, summary, factors) if category is not None
                else (m.MitigationPriority.LOW, m.MitigationUrgency.LONG_TERM, 40.0)
            )

            base_key = item.strategy_type.replace("_", "-")
            strategy_key = base_key
            suffix = 2
            while strategy_key in used_keys:
                strategy_key = f"{base_key}-{suffix}"
                suffix += 1
            used_keys.add(strategy_key)

            rows_to_persist.append({
                "strategy_key": strategy_key,
                "title": item.title,
                "strategy_type": item.strategy_type,
                "description": item.description,
                "why_this_mitigation": item.why_this_mitigation,
                "investigation_basis": item.investigation_basis,
                "risk_addressed": item.risk_addressed,
                "implementation_steps": item.implementation_steps,
                "expected_impact": item.expected_impact,
                "priority": priority.value,
                "urgency": urgency.value,
                "confidence": confidence,
                "limitations": item.limitations,
                "dependencies": item.dependencies,
                "success_metrics": item.success_metrics,
                "is_confirmed_corrective_action": category.is_confirmed_corrective if category else False,
                "validation_status": outcome.status.value,
                "validation_notes": outcome.notes,
            })

        persisted = self.repo.replace_strategies(run_id, rows_to_persist)
        self.session.commit()
        logger.info("Generated %d valid mitigation strategies (%d rejected).", len(persisted), rejected_count)

        return MitigationResult(
            run_id=run_id, root_cause_status=summary.root_cause_status,
            categories_identified=[c.key for c in categories],
            strategies=persisted, rejected_count=rejected_count,
        )
