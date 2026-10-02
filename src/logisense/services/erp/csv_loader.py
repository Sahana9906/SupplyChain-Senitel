"""Loads company_data/*.csv into the ERP database, preserving every real
ID as the row's source_* identifier and reconstructing the full chain:

  Supplier -> Contract -> Purchase Order -> PO Items -> Shipment -> Shipment Events
  Product -> Inventory -> Warehouse -> Customer Orders

CSVs are seed data only -- this is the ONE place they get read. Nothing
downstream (investigation, RAG, API) reads a CSV file; everything reads
the database from here on.

Idempotent: safe to re-run against the same CSVs (upserts throughout).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from logisense import models as m
from logisense.repositories import (
    CarrierRepository,
    ContractRepository,
    DimensionRepository,
    InventoryRepository,
    OrderShipmentRepository,
    PurchaseOrderItemRepository,
    PurchaseOrderRepository,
    ShipmentEventRepository,
    WarehouseRepository,
)

_PO_STATUS_MAP = {
    "placed": m.PurchaseOrderStatus.PLACED, "in production": m.PurchaseOrderStatus.IN_PRODUCTION,
    "ready for shipment": m.PurchaseOrderStatus.READY_FOR_SHIPMENT, "shipped": m.PurchaseOrderStatus.SHIPPED,
    "delayed": m.PurchaseOrderStatus.DELAYED, "received": m.PurchaseOrderStatus.RECEIVED,
    "closed": m.PurchaseOrderStatus.CLOSED,
}
_EVENT_TYPE_MAP = {
    "pickup": m.ShipmentEventType.PICKUP, "departure": m.ShipmentEventType.DEPARTURE,
    "transit": m.ShipmentEventType.IN_TRANSIT, "customs": m.ShipmentEventType.CUSTOMS,
    "arrival": m.ShipmentEventType.ARRIVAL, "received": m.ShipmentEventType.RECEIVED,
}


@dataclass(frozen=True)
class LoadResult:
    suppliers: int
    products: int
    warehouses: int
    carriers: int
    customers: int
    contracts: int
    purchase_orders: int
    purchase_order_items: int
    inbound_shipments: int
    shipment_events: int
    customer_orders: int
    inventory_rows: int
    skipped_rows: dict[str, int]


def _read(data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _parse_location_string(location: str) -> tuple[str | None, str | None, str | None]:
    parts = [p.strip() for p in str(location).split(",") if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[-1]
    if len(parts) == 2:
        return parts[0], None, parts[-1]
    if len(parts) == 1:
        return parts[0], None, None
    return None, None, None


def _vectorize_dates(df: pd.DataFrame, *columns: str) -> None:
    """Converts date/datetime columns to parsed datetime64 ONCE for the
    whole column (pandas vectorized), instead of calling pd.to_datetime()
    per-row inside the loop -- pd.to_datetime() re-guesses the format on
    every single scalar call, which is the dominant cost at thousands of
    rows. Mutates df in place; safe to call on non-existent columns."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _date(value) -> "datetime.date | None":
    if pd.isna(value):
        return None
    # After _vectorize_dates, value is already a Timestamp/NaT -- no
    # re-parsing needed, just extract .date().
    return value.date() if hasattr(value, "date") else pd.Timestamp(value).date()


def _dt(value) -> datetime | None:
    if pd.isna(value):
        return None
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else pd.Timestamp(value).to_pydatetime()


def load_company_data(session: Session, data_dir: str) -> LoadResult:
    dimensions = DimensionRepository(session)
    warehouses_repo = WarehouseRepository(session)
    carriers_repo = CarrierRepository(session)
    contracts_repo = ContractRepository(session)
    purchase_orders_repo = PurchaseOrderRepository(session)
    po_items_repo = PurchaseOrderItemRepository(session)
    shipment_events_repo = ShipmentEventRepository(session)
    order_shipments = OrderShipmentRepository(session)
    inventory_repo = InventoryRepository(session)

    skipped: dict[str, int] = {}

    # -- Suppliers --------------------------------------------------------
    suppliers_df = _read(data_dir, "suppliers")
    supplier_id_map: dict[str, int] = {}
    _supplier_objs: dict[str, m.Supplier] = {}
    for _, row in suppliers_df.iterrows():
        location = dimensions.upsert_location(city=None, state=None, country=row.get("country"), region=row.get("region"), flush=False)
        session.flush()  # location needs a real id before being used as a FK below
        supplier = dimensions.upsert_supplier(source_supplier_id=row["supplier_id"], name=row.get("supplier_name"), location_id=location.id, flush=False)
        _supplier_objs[row["supplier_id"]] = supplier
    session.flush()
    supplier_id_map = {k: v.id for k, v in _supplier_objs.items()}

    # -- Products -----------------------------------------------------------
    products_df = _read(data_dir, "products")
    _product_objs: dict[str, m.Product] = {}
    for _, row in products_df.iterrows():
        product = dimensions.upsert_product(source_product_id=row["sku_id"], name=row.get("product_name"), category=row.get("category"), flush=False)
        _product_objs[row["sku_id"]] = product
    session.flush()
    product_id_map = {k: v.id for k, v in _product_objs.items()}

    # -- Warehouses -----------------------------------------------------------
    warehouses_df = _read(data_dir, "warehouses")
    _warehouse_objs: dict[str, m.Warehouse] = {}
    for _, row in warehouses_df.iterrows():
        city, state, country = _parse_location_string(row.get("location", ""))
        location = None
        if any([city, state, country, row.get("region")]):
            location = dimensions.upsert_location(city=city, state=state, country=country, region=row.get("region"), flush=False)
            session.flush()
        warehouse = warehouses_repo.upsert(
            code=row["warehouse_id"], name=row.get("warehouse_name"), location_id=location.id if location else None,
            capacity=int(row["capacity"]) if pd.notna(row.get("capacity")) else None,
            current_utilization=float(row["current_utilization"]) if pd.notna(row.get("current_utilization")) else None,
            flush=False,
        )
        _warehouse_objs[row["warehouse_id"]] = warehouse
    session.flush()
    warehouse_id_map = {k: v.id for k, v in _warehouse_objs.items()}

    # -- Carriers -----------------------------------------------------------
    carriers_df = _read(data_dir, "carriers")
    _carrier_objs: dict[str, m.Carrier] = {}
    for _, row in carriers_df.iterrows():
        carrier = carriers_repo.upsert(
            source_carrier_id=row["carrier_id"], name=row.get("carrier_name"), service_type=row.get("service_type"),
            region=row.get("region"), on_time_rate=float(row["on_time_rate"]) if pd.notna(row.get("on_time_rate")) else None,
            avg_transit_days=float(row["avg_transit_days"]) if pd.notna(row.get("avg_transit_days")) else None,
            flush=False,
        )
        _carrier_objs[row["carrier_id"]] = carrier
    session.flush()
    carrier_id_map = {k: v.id for k, v in _carrier_objs.items()}

    # -- Customers -----------------------------------------------------------
    customers_df = _read(data_dir, "customers")
    _customer_objs: dict[str, m.Customer] = {}
    for _, row in customers_df.iterrows():
        location = dimensions.upsert_location(city=None, state=None, country=None, region=row.get("region"), flush=False)
        session.flush()
        customer = dimensions.upsert_customer(source_customer_id=row["customer_id"], location_id=location.id, flush=False)
        _customer_objs[row["customer_id"]] = customer
    session.flush()
    customer_id_map = {k: v.id for k, v in _customer_objs.items()}

    # -- Contracts -----------------------------------------------------------
    contracts_df = _read(data_dir, "contracts")
    _vectorize_dates(contracts_df, "effective_date", "expiry_date")
    contract_count = 0
    for _, row in contracts_df.iterrows():
        supplier_db_id = supplier_id_map.get(row["supplier_id"])
        if supplier_db_id is None:
            skipped["contracts"] = skipped.get("contracts", 0) + 1
            continue
        contracts_repo.upsert(
            source_contract_id=row["contract_id"], supplier_id=supplier_db_id,
            effective_date=_date(row.get("effective_date")), expiry_date=_date(row.get("expiry_date")),
            agreed_lead_time=int(row["agreed_lead_time"]) if pd.notna(row.get("agreed_lead_time")) else None,
            minimum_order_quantity=int(row["minimum_order_quantity"]) if pd.notna(row.get("minimum_order_quantity")) else None,
            sla=row.get("SLA"), penalty_terms=row.get("penalty_terms"), flush=False,
        )
        contract_count += 1
    session.flush()

    # -- Purchase orders -----------------------------------------------------------
    po_df = _read(data_dir, "purchase_orders")
    _vectorize_dates(po_df, "order_date", "expected_delivery_date")
    po_id_map: dict[str, int] = {}
    _po_objs: dict[str, m.PurchaseOrder] = {}
    for _, row in po_df.iterrows():
        supplier_db_id = supplier_id_map.get(row["supplier_id"])
        if supplier_db_id is None:
            skipped["purchase_orders"] = skipped.get("purchase_orders", 0) + 1
            continue
        status = _PO_STATUS_MAP.get(str(row.get("status", "")).strip().lower(), m.PurchaseOrderStatus.PLACED)
        po = purchase_orders_repo.upsert(
            source_reference=row["po_id"], supplier_id=supplier_db_id,
            warehouse_id=warehouse_id_map.get(row.get("warehouse_id")),
            order_date=_date(row.get("order_date")), expected_ready_date=_date(row.get("expected_delivery_date")),
            total_value=float(row["total_value"]) if pd.notna(row.get("total_value")) else None,
            status=status, flush=False,
        )
        _po_objs[row["po_id"]] = po
    session.flush()
    po_id_map = {k: v.id for k, v in _po_objs.items()}

    # -- Purchase order items -----------------------------------------------------------
    po_items_df = _read(data_dir, "purchase_order_items")
    po_item_count = 0
    for _, row in po_items_df.iterrows():
        po_db_id = po_id_map.get(row["po_id"])
        product_db_id = product_id_map.get(row["sku_id"])
        if po_db_id is None or product_db_id is None:
            skipped["purchase_order_items"] = skipped.get("purchase_order_items", 0) + 1
            continue
        with session.no_autoflush:
            po_items_repo.upsert(
                purchase_order_id=po_db_id, product_id=product_db_id,
                quantity=int(row["quantity"]), unit_cost=float(row["unit_cost"]) if pd.notna(row.get("unit_cost")) else None,
                expected_quantity=int(row["expected_quantity"]) if pd.notna(row.get("expected_quantity")) else None,
                flush=False,
            )
        po_item_count += 1
    session.flush()

    # -- Inbound shipments -----------------------------------------------------------
    shipments_df = _read(data_dir, "inbound_shipments")
    _vectorize_dates(shipments_df, "promised_date", "actual_arrival")
    shipment_id_map: dict[str, int] = {}
    shipment_count = 0

    # Pre-fetch what would otherwise be a query PER ROW (N+1): existing
    # shipments keyed by purchase_order_id, and each PO's warehouse
    # location -- both computed once for the whole loop, not 4000 times.
    existing_shipments_by_po = {
        s.purchase_order_id: s
        for s in session.query(m.Shipment).filter(m.Shipment.purchase_order_id.in_(po_id_map.values())).all()
    }
    po_warehouse_location = {
        po.id: (po.warehouse.location_id if po.warehouse else None)
        for po in session.query(m.PurchaseOrder).filter(m.PurchaseOrder.id.in_(po_id_map.values())).all()
    }

    _new_shipments: list[tuple[str, m.Shipment]] = []
    for _, row in shipments_df.iterrows():
        po_db_id = po_id_map.get(row["po_id"])
        if po_db_id is None:
            skipped["inbound_shipments"] = skipped.get("inbound_shipments", 0) + 1
            continue

        promised = _date(row.get("promised_date"))
        actual = _date(row.get("actual_arrival"))
        delay_days = (actual - promised).days if (promised and actual) else None
        destination_location_id = po_warehouse_location.get(po_db_id)
        carrier_db_id = carrier_id_map.get(row.get("carrier"))

        values = dict(
            destination_location_id=destination_location_id, carrier_id=carrier_db_id,
            expected_delivery_date=promised, actual_delivery_date=actual, delay_days=delay_days,
        )
        existing = existing_shipments_by_po.get(po_db_id)
        if existing:
            for k, v in values.items():
                setattr(existing, k, v)
            shipment_id_map[row["shipment_id"]] = existing.id
        else:
            shipment = m.Shipment(purchase_order_id=po_db_id, **values)
            session.add(shipment)
            _new_shipments.append((row["shipment_id"], shipment))
        shipment_count += 1

    session.flush()  # ONE flush for all newly created shipments, not one per row
    for source_shipment_id, shipment in _new_shipments:
        shipment_id_map[source_shipment_id] = shipment.id

    # -- Shipment events -----------------------------------------------------------
    events_df = _read(data_dir, "shipment_events")
    _vectorize_dates(events_df, "timestamp")
    event_count = 0
    for _, row in events_df.iterrows():
        shipment_db_id = shipment_id_map.get(row["shipment_id"])
        if shipment_db_id is None:
            skipped["shipment_events"] = skipped.get("shipment_events", 0) + 1
            continue
        event_type = _EVENT_TYPE_MAP.get(str(row.get("event_type", "")).strip().lower(), m.ShipmentEventType.EXCEPTION)
        shipment_events_repo.add_event(
            shipment_db_id, event_type, _dt(row.get("timestamp")),
            location_label=row.get("location"), description=row.get("description"), flush=False,
        )
        event_count += 1
    session.flush()

    # -- Customer orders + items -----------------------------------------------------------
    orders_df = _read(data_dir, "customer_orders")
    order_items_df = _read(data_dir, "customer_order_items")
    _vectorize_dates(orders_df, "order_date")
    orders_by_id = orders_df.set_index("order_id")
    order_count = 0
    for idx, row in order_items_df.iterrows():
        order_id = row["order_id"]
        if order_id not in orders_by_id.index:
            skipped["customer_order_items"] = skipped.get("customer_order_items", 0) + 1
            continue
        header = orders_by_id.loc[order_id]
        customer_db_id = customer_id_map.get(header["customer_id"])
        product_db_id = product_id_map.get(row["sku_id"])
        if customer_db_id is None or product_db_id is None:
            skipped["customer_orders"] = skipped.get("customer_orders", 0) + 1
            continue
        # DataCo/Olist convention for multi-item orders: one canonical
        # Order row per (order_id, item), source_order_id encodes both.
        quantity = int(row["quantity"])
        unit_price = float(row["unit_price"])
        with session.no_autoflush:
            order_shipments.upsert_order(
                source_order_id=f"{order_id}:{idx}", order_date=_date(header.get("order_date")),
                order_value=round(quantity * unit_price, 2), quantity=quantity, status=header.get("status"),
                customer_id=customer_db_id, product_id=product_db_id, supplier_id=None, flush=False,
            )
        order_count += 1
    session.flush()

    # -- Inventory -----------------------------------------------------------
    inventory_df = _read(data_dir, "inventory")
    inventory_count = 0
    for _, row in inventory_df.iterrows():
        warehouse_db_id = warehouse_id_map.get(row["warehouse_id"])
        product_db_id = product_id_map.get(row["sku_id"])
        if warehouse_db_id is None or product_db_id is None:
            skipped["inventory"] = skipped.get("inventory", 0) + 1
            continue
        inventory_repo.upsert(
            warehouse_id=warehouse_db_id, product_id=product_db_id, on_hand=int(row["on_hand"]),
            reserved=int(row["reserved"]) if pd.notna(row.get("reserved")) else None,
            available=int(row["available"]) if pd.notna(row.get("available")) else None,
            reorder_point=int(row["reorder_point"]) if pd.notna(row.get("reorder_point")) else None,
            safety_stock=int(row["safety_stock"]) if pd.notna(row.get("safety_stock")) else None,
            daily_demand=float(row["daily_demand"]) if pd.notna(row.get("daily_demand")) else None,
            flush=False,
        )
        inventory_count += 1
    session.flush()
    session.commit()

    return LoadResult(
        suppliers=len(supplier_id_map), products=len(product_id_map), warehouses=len(warehouse_id_map),
        carriers=len(carrier_id_map), customers=len(customer_id_map), contracts=contract_count,
        purchase_orders=len(po_id_map), purchase_order_items=po_item_count, inbound_shipments=shipment_count,
        shipment_events=event_count, customer_orders=order_count, inventory_rows=inventory_count,
        skipped_rows=skipped,
    )
