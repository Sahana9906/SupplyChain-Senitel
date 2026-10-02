"""phase 3 evidence schema + erp layer + rag document schema

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-07

This migration is written to be safely RE-RUNNABLE from any partial
prior state -- e.g. if an earlier attempt created some tables/types and
then failed or was interrupted before finishing. Every operation checks
for existence first:

  - enum types: created via `.create(bind, checkfirst=True)`, and
    referenced in columns with `create_type=False` so op.create_table()
    never tries to create them a second time.
  - tables: `op.create_table(..., if_not_exists=True)`.
  - indexes: `op.create_index(..., if_not_exists=True)`.
  - the two add_column calls and the check constraint on `shipments`
    (which don't have a native if-not-exists option in Alembic) are
    guarded by an inspector lookup before running.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

_evidence_source_category = postgresql.ENUM(
    "internal", "external", name="evidence_source_category", create_type=False
)
_provider_status = postgresql.ENUM(
    "available", "unavailable", "error", "no_evidence_found", name="provider_status", create_type=False
)
_evidence_strength = postgresql.ENUM("low", "medium", "high", name="evidence_strength", create_type=False)
_purchase_order_status = postgresql.ENUM(
    "placed", "in_production", "ready_for_shipment", "shipped", "delayed", "received", "closed",
    name="purchase_order_status", create_type=False,
)
_shipment_event_type = postgresql.ENUM(
    "created", "pickup", "departure", "in_transit", "customs", "customs_hold",
    "carrier_delay", "weather_delay", "quality_hold", "arrival", "delivered",
    "received", "exception",
    name="shipment_event_type", create_type=False,
)

_ALL_ENUMS = [
    _evidence_source_category, _provider_status, _evidence_strength,
    _purchase_order_status, _shipment_event_type,
]


def _column_exists(bind, table_name: str, column_name: str) -> bool:
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    return any(c["name"] == column_name for c in inspector.get_columns(table_name))


def _constraint_exists(bind, table_name: str, constraint_name: str) -> bool:
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    names = {c["name"] for c in inspector.get_check_constraints(table_name)}
    return constraint_name in names


def upgrade() -> None:
    op.execute("ALTER TYPE investigation_stage ADD VALUE IF NOT EXISTS 'evidence_gathered'")
    op.execute("ALTER TYPE investigation_stage ADD VALUE IF NOT EXISTS 'verdict_generated'")

    bind = op.get_bind()
    for enum_type in _ALL_ENUMS:
        enum_type.create(bind, checkfirst=True)

    # ---- Phase 3: evidence retrieval & validation --------------------
    op.create_table(
        "evidence_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("investigation_runs.run_id"), nullable=False),
        sa.Column("category", _evidence_source_category, nullable=False),
        sa.Column("provider_name", sa.String(128), nullable=False),
        sa.Column("status", _provider_status, nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("notes", sa.Text()),
        sa.Column("queried_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_evidence_sources_run", "evidence_sources", ["run_id"], if_not_exists=True)

    op.create_table(
        "evidence",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("evidence_sources.id"), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("evidence_reference", sa.String(512)),
        sa.Column("relevance_score", sa.Numeric(4, 3)),
        sa.Column("reliability_score", sa.Numeric(4, 3)),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("evidence_metadata", sa.JSON()),
        if_not_exists=True,
    )
    op.create_index("ix_evidence_source", "evidence", ["source_id"], if_not_exists=True)

    op.create_table(
        "hypothesis_evidence",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("hypothesis_id", sa.Integer(), sa.ForeignKey("investigation_hypotheses.id"), nullable=False),
        sa.Column("evidence_id", sa.Integer(), sa.ForeignKey("evidence.id"), nullable=False),
        sa.Column("evidence_question", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("hypothesis_id", "evidence_id", name="uq_hypothesis_evidence"),
        if_not_exists=True,
    )
    op.create_index("ix_hypothesis_evidence_hypothesis", "hypothesis_evidence", ["hypothesis_id"], if_not_exists=True)

    op.create_table(
        "evidence_assessments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("hypothesis_evidence_id", sa.Integer(), sa.ForeignKey("hypothesis_evidence.id"), nullable=False, unique=True),
        sa.Column("supports_hypothesis", sa.Boolean(), nullable=False),
        sa.Column("contradicts_hypothesis", sa.Boolean(), nullable=False),
        sa.Column("strength", _evidence_strength, nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("assessed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("NOT (supports_hypothesis AND contradicts_hypothesis)", name="ck_evidence_assessment_not_both"),
        if_not_exists=True,
    )

    op.create_table(
        "verdicts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("investigation_runs.run_id"), nullable=False, unique=True),
        sa.Column("root_cause_hypothesis_id", sa.Integer(), sa.ForeignKey("investigation_hypotheses.id")),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("supporting_evidence_summary", sa.Text(), nullable=False),
        sa.Column("contradicting_evidence_summary", sa.Text(), nullable=False),
        sa.Column("internal_evidence_summary", sa.Text(), nullable=False),
        sa.Column("external_evidence_summary", sa.Text(), nullable=False),
        sa.Column("remaining_uncertainty", sa.Text(), nullable=False),
        sa.Column("evidence_gaps", sa.Text(), nullable=False),
        sa.Column("recommended_next_action", sa.Text(), nullable=False),
        sa.Column("hypothesis_confidence_updates", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )

    # ---- ERP layer ----------------------------------------------------
    op.create_table(
        "warehouses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(32), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id")),
        sa.Column("capacity", sa.Integer()),
        sa.Column("current_utilization", sa.Numeric(5, 3)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )

    op.create_table(
        "carriers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_carrier_id", sa.String(64), unique=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("service_type", sa.String(128)),
        sa.Column("region", sa.String(128)),
        sa.Column("on_time_rate", sa.Numeric(5, 3)),
        sa.Column("avg_transit_days", sa.Numeric(6, 2)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )

    op.create_table(
        "purchase_orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_reference", sa.String(64), nullable=False, unique=True),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.id"), nullable=False),
        sa.Column("warehouse_id", sa.Integer(), sa.ForeignKey("warehouses.id")),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.id")),
        sa.Column("quantity", sa.Integer()),
        sa.Column("order_date", sa.Date()),
        sa.Column("expected_ready_date", sa.Date()),
        sa.Column("total_value", sa.Numeric(14, 2)),
        sa.Column("status", _purchase_order_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_purchase_orders_supplier", "purchase_orders", ["supplier_id"], if_not_exists=True)

    op.create_table(
        "purchase_order_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("purchase_order_id", sa.Integer(), sa.ForeignKey("purchase_orders.id"), nullable=False),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.id"), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_cost", sa.Numeric(14, 2)),
        sa.Column("expected_quantity", sa.Integer()),
        sa.Column("expected_date", sa.Date()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_purchase_order_items_po", "purchase_order_items", ["purchase_order_id"], if_not_exists=True)

    op.create_table(
        "inventory",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("warehouse_id", sa.Integer(), sa.ForeignKey("warehouses.id"), nullable=False),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.id"), nullable=False),
        sa.Column("on_hand", sa.Integer(), nullable=False),
        sa.Column("reserved", sa.Integer()),
        sa.Column("available", sa.Integer()),
        sa.Column("reorder_point", sa.Integer()),
        sa.Column("safety_stock", sa.Integer()),
        sa.Column("daily_demand", sa.Numeric(10, 2)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("warehouse_id", "product_id", name="uq_inventory_warehouse_product"),
        if_not_exists=True,
    )

    op.create_table(
        "contracts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_contract_id", sa.String(64), nullable=False, unique=True),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.id"), nullable=False),
        sa.Column("effective_date", sa.Date()),
        sa.Column("expiry_date", sa.Date()),
        sa.Column("agreed_lead_time", sa.Integer()),
        sa.Column("minimum_order_quantity", sa.Integer()),
        sa.Column("sla", sa.Text()),
        sa.Column("penalty_terms", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_contracts_supplier", "contracts", ["supplier_id"], if_not_exists=True)

    # ---- Relax Shipment for the inbound (PO-fulfillment) leg ----------
    # alter_column to nullable is naturally idempotent (setting nullable=True
    # on an already-nullable column is a no-op), but add_column and the
    # check constraint are NOT idempotent in Alembic -- guard them manually.
    op.alter_column("shipments", "order_id", nullable=True)

    if not _column_exists(bind, "shipments", "purchase_order_id"):
        op.add_column("shipments", sa.Column("purchase_order_id", sa.Integer(), sa.ForeignKey("purchase_orders.id"), unique=True))
    if not _column_exists(bind, "shipments", "carrier_id"):
        op.add_column("shipments", sa.Column("carrier_id", sa.Integer(), sa.ForeignKey("carriers.id")))
    if not _constraint_exists(bind, "shipments", "ck_shipment_has_order_or_po"):
        op.create_check_constraint(
            "ck_shipment_has_order_or_po", "shipments", "(order_id IS NOT NULL) OR (purchase_order_id IS NOT NULL)"
        )

    op.create_table(
        "shipment_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("shipment_id", sa.Integer(), sa.ForeignKey("shipments.id"), nullable=False),
        sa.Column("event_type", _shipment_event_type, nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id")),
        sa.Column("location_label", sa.String(255)),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_shipment_events_shipment", "shipment_events", ["shipment_id", "event_time"], if_not_exists=True)

    # ---- RAG document schema -------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column("source_reference", sa.String(255), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("doc_metadata", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("source_type", "source_reference", name="uq_document_source"),
        if_not_exists=True,
    )
    op.create_index("ix_documents_source", "documents", ["source_type", "source_reference"], if_not_exists=True)

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.String(128)),
        sa.Column("vector_store_id", sa.String(128), unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        if_not_exists=True,
    )
    op.create_index("ix_document_chunks_document", "document_chunks", ["document_id"], if_not_exists=True)


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("shipment_events")
    op.drop_constraint("ck_shipment_has_order_or_po", "shipments", type_="check")
    op.drop_column("shipments", "carrier_id")
    op.drop_column("shipments", "purchase_order_id")
    op.alter_column("shipments", "order_id", nullable=False)
    op.drop_table("contracts")
    op.drop_table("inventory")
    op.drop_table("purchase_order_items")
    op.drop_table("purchase_orders")
    op.drop_table("carriers")
    op.drop_table("warehouses")
    op.drop_table("verdicts")
    op.drop_table("evidence_assessments")
    op.drop_table("hypothesis_evidence")
    op.drop_table("evidence")
    op.drop_table("evidence_sources")

    bind = op.get_bind()
    for enum_type in reversed(_ALL_ENUMS):
        enum_type.drop(bind, checkfirst=True)
