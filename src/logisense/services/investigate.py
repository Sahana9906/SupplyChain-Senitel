"""Combined ERP + RAG investigation interface.

Deliberately keeps two clean, separate surfaces rather than merging them
into one opaque structure:

  ERP  -- "what is happening operationally?" (facts from the database)
  RAG  -- "what does policy/contract/history say?" (evidence-grounded,
           cited passages from the knowledge base)

A future Sentinel agent tool should be able to call either independently,
or call investigate_supplier() to get both at once for a quick overview.
Nothing here decides a root cause or recommends a mitigation -- it only
assembles operational facts and retrieves relevant organizational
knowledge for something else (the agent) to reason over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from logisense import models as m
from logisense.repositories import ERPQueryRepository
from logisense.services.rag.hybrid_investigation import HybridAnswer, HybridInvestigationService
from logisense.services.rag.qa_service import EvidenceGroundedQAService, GroundedAnswer


@dataclass(frozen=True)
class SupplierOperationalSnapshot:
    """The ERP half: what is happening operationally, as of now."""

    supplier_id: str
    supplier_name: str | None
    contract_id: str | None
    contract_sla: str | None
    open_purchase_orders: list[dict]
    recent_shipments: list[dict]


@dataclass(frozen=True)
class SupplierInvestigation:
    operational: SupplierOperationalSnapshot | None
    knowledge_answers: list[GroundedAnswer] = field(default_factory=list)


class InvestigationService:
    def __init__(
        self,
        session: Session,
        qa_service: EvidenceGroundedQAService | None = None,
        hybrid_service: HybridInvestigationService | None = None,
    ) -> None:
        self.session = session
        self.erp = ERPQueryRepository(session)
        self.qa_service = qa_service
        self.hybrid_service = hybrid_service

    # -- ERP: operational facts -----------------------------------------------

    def get_supplier_operational_snapshot(self, source_supplier_id: str) -> SupplierOperationalSnapshot | None:
        supplier = self.erp.get_supplier_by_source_id(source_supplier_id)
        if supplier is None:
            return None

        contract = self.erp.get_contract_for_supplier(supplier.id)
        purchase_orders = self.erp.get_purchase_orders_for_supplier(supplier.id, limit=10)

        open_pos = [
            {
                "po_id": po.source_reference, "status": po.status.value,
                "order_date": po.order_date.isoformat() if po.order_date else None,
                "expected_ready_date": po.expected_ready_date.isoformat() if po.expected_ready_date else None,
                "total_value": float(po.total_value) if po.total_value is not None else None,
            }
            for po in purchase_orders
        ]

        shipments = self.erp.get_shipments_for_supplier(supplier.id, limit=10)
        recent_shipments = [
            {
                "shipment_id": s.id, "po_id": s.purchase_order.source_reference if s.purchase_order else None,
                "expected_delivery_date": s.expected_delivery_date.isoformat() if s.expected_delivery_date else None,
                "actual_delivery_date": s.actual_delivery_date.isoformat() if s.actual_delivery_date else None,
                "delay_days": s.delay_days, "carrier": s.carrier.name if s.carrier else None,
            }
            for s in shipments
        ]

        return SupplierOperationalSnapshot(
            supplier_id=source_supplier_id, supplier_name=supplier.name,
            contract_id=contract.source_contract_id if contract else None,
            contract_sla=contract.sla if contract else None,
            open_purchase_orders=open_pos, recent_shipments=recent_shipments,
        )

    # -- RAG: organizational knowledge -----------------------------------------------

    def ask_knowledge_base(self, question: str) -> GroundedAnswer:
        if self.qa_service is None:
            raise ValueError("No QA service configured -- cannot query the knowledge base.")
        return self.qa_service.answer(question)

    # -- Hybrid: real structured facts + entity-filtered document -----
    # -- evidence, blended into one grounded answer with citations for --
    # -- both. This is the general-purpose entry point -- works for any --
    # -- free-text question (supplier disruptions, PO impact, contract --
    # -- obligations), not just a fixed supplier lookup. ------------------

    def ask(self, question: str) -> HybridAnswer:
        if self.hybrid_service is None:
            raise ValueError("No hybrid investigation service configured -- cannot answer hybrid questions.")
        return self.hybrid_service.answer(question)

    # -- Combined -----------------------------------------------------------

    def investigate_supplier(self, source_supplier_id: str, knowledge_questions: list[str] | None = None) -> SupplierInvestigation:
        operational = self.get_supplier_operational_snapshot(source_supplier_id)

        knowledge_answers: list[GroundedAnswer] = []
        if self.qa_service is not None:
            questions = knowledge_questions or [
                f"What are supplier {source_supplier_id}'s contractual obligations?",
                f"What escalation procedures apply if supplier {source_supplier_id} has a delayed shipment or purchase order?",
            ]
            for q in questions:
                knowledge_answers.append(self.qa_service.answer(q))

        return SupplierInvestigation(operational=operational, knowledge_answers=knowledge_answers)
