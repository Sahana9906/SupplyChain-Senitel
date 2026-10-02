"""Builds hybrid context for a question: real structured facts from the
database (suppliers, POs, shipments, inventory, contracts) PLUS
semantically retrieved, entity-filtered document evidence.

The two halves stay clearly separated all the way through -- structured
facts are labeled as database records, document evidence is labeled with
its real source (category/filename/section) -- so a citation never
confuses "the database says X" with "a document says X".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from logisense import models as m
from logisense.repositories import ERPQueryRepository
from logisense.services.rag.embeddings import EmbeddingClient, embed_query_text
from logisense.services.rag.entity_extraction import ExtractedEntities, extract_entities
from logisense.services.rag.vector_store import ChromaVectorStore, VectorMatch


@dataclass(frozen=True)
class StructuredFact:
    """One real fact pulled from the database, with enough provenance to
    cite it precisely (table + real id)."""

    label: str
    value: str
    source_table: str
    source_id: str


@dataclass(frozen=True)
class HybridContext:
    question: str
    entities: ExtractedEntities
    structured_facts: list[StructuredFact] = field(default_factory=list)
    document_matches: list[VectorMatch] = field(default_factory=list)

    @property
    def has_any_evidence(self) -> bool:
        return bool(self.structured_facts) or bool(self.document_matches)


class HybridContextBuilder:
    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
        vector_store: ChromaVectorStore,
        top_k: int = 6,
        min_similarity: float = 0.2,
    ) -> None:
        self.session = session
        self.erp = ERPQueryRepository(session)
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def build(self, question: str) -> HybridContext:
        entities = extract_entities(question)
        structured_facts = self._gather_structured_facts(entities)
        document_matches = self._gather_document_evidence(question, entities)

        return HybridContext(
            question=question, entities=entities,
            structured_facts=structured_facts, document_matches=document_matches,
        )

    # -- structured half: real DB facts, always exact-match lookups, --
    # -- never a guess -----------------------------------------------------

    def _gather_structured_facts(self, entities: ExtractedEntities) -> list[StructuredFact]:
        facts: list[StructuredFact] = []

        for source_supplier_id in entities.supplier_ids:
            supplier = self.erp.get_supplier_by_source_id(source_supplier_id)
            if supplier is None:
                continue
            facts.append(StructuredFact(
                label="Supplier", value=f"{supplier.name} ({source_supplier_id}), status: active",
                source_table="suppliers", source_id=source_supplier_id,
            ))
            contract = self.erp.get_contract_for_supplier(supplier.id)
            if contract:
                facts.append(StructuredFact(
                    label="Contract (database record)",
                    value=(
                        f"{contract.source_contract_id}: agreed lead time {contract.agreed_lead_time} days, "
                        f"MOQ {contract.minimum_order_quantity}, SLA: {contract.sla}"
                    ),
                    source_table="contracts", source_id=contract.source_contract_id,
                ))
            open_pos = self.erp.get_purchase_orders_for_supplier(supplier.id, limit=5)
            for po in open_pos:
                facts.append(StructuredFact(
                    label="Purchase Order",
                    value=(
                        f"{po.source_reference}: status={po.status.value}, "
                        f"order_date={po.order_date}, expected_ready_date={po.expected_ready_date}, "
                        f"total_value={po.total_value}"
                    ),
                    source_table="purchase_orders", source_id=po.source_reference,
                ))
            shipments = self.erp.get_shipments_for_supplier(supplier.id, limit=5)
            for s in shipments:
                facts.append(StructuredFact(
                    label="Shipment",
                    value=(
                        f"shipment for {s.purchase_order.source_reference if s.purchase_order else '?'}: "
                        f"expected={s.expected_delivery_date}, actual={s.actual_delivery_date}, "
                        f"delay_days={s.delay_days}, carrier={s.carrier.name if s.carrier else 'unknown'}"
                    ),
                    source_table="shipments", source_id=str(s.id),
                ))

        for po_ref in entities.purchase_order_ids:
            po = self.erp.get_purchase_order_by_reference(po_ref)
            if po is None:
                continue
            facts.append(StructuredFact(
                label="Purchase Order",
                value=(
                    f"{po.source_reference}: status={po.status.value}, supplier={po.supplier.source_supplier_id if po.supplier else '?'}, "
                    f"order_date={po.order_date}, expected_ready_date={po.expected_ready_date}, total_value={po.total_value}"
                ),
                source_table="purchase_orders", source_id=po.source_reference,
            ))
            shipment = self.erp.get_shipment_by_purchase_order_reference(po_ref)
            if shipment:
                facts.append(StructuredFact(
                    label="Shipment",
                    value=(
                        f"expected={shipment.expected_delivery_date}, actual={shipment.actual_delivery_date}, "
                        f"delay_days={shipment.delay_days}, carrier={shipment.carrier.name if shipment.carrier else 'unknown'}"
                    ),
                    source_table="shipments", source_id=str(shipment.id),
                ))
            # Inventory linkage: a PO's products come from its LINE
            # ITEMS (PurchaseOrderItem), not a direct product_id on the
            # PO header itself -- the loader never populates that field.
            for item in self.erp.get_purchase_order_items(po.id):
                if item.product is None:
                    continue
                inv_rows = self.erp.get_inventory_for_product_reference(item.product.source_product_id)
                for inv in inv_rows:
                    facts.append(StructuredFact(
                        label="Inventory",
                        value=(
                            f"warehouse {inv.warehouse.code}: on_hand={inv.on_hand}, available={inv.available}, "
                            f"reorder_point={inv.reorder_point}, daily_demand={inv.daily_demand}"
                        ),
                        source_table="inventory", source_id=f"{inv.warehouse.code}/{item.product.source_product_id}",
                    ))

        for sku_id in entities.sku_ids:
            inv_rows = self.erp.get_inventory_for_product_reference(sku_id)
            for inv in inv_rows:
                facts.append(StructuredFact(
                    label="Inventory",
                    value=f"warehouse {inv.warehouse.code}: on_hand={inv.on_hand}, available={inv.available}, reorder_point={inv.reorder_point}",
                    source_table="inventory", source_id=f"{inv.warehouse.code}/{sku_id}",
                ))

        return facts

    # -- document half: semantic search, entity-filtered when possible --

    def _gather_document_evidence(self, question: str, entities: ExtractedEntities) -> list[VectorMatch]:
        query_embedding = embed_query_text(self.embedding_client, question)
        if not query_embedding:
            return []

        where = None
        if entities.primary_supplier_id:
            where = {"supplier_id": entities.primary_supplier_id}

        matches = self.vector_store.query(query_embedding, top_k=self.top_k, where=where)
        # NOTE: deliberately no fallback to unfiltered search when an
        # entity filter comes up empty. A question naming a specific
        # supplier that has no matching documents (e.g. it doesn't exist
        # in the data) must surface as NO evidence for that entity, not
        # get padded out with semantically-similar but unrelated
        # documents -- that would defeat the insufficient-evidence check.

        return [mtch for mtch in matches if mtch.similarity >= self.min_similarity]
