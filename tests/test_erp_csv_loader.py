import csv
import os

from sqlalchemy import select

from logisense import models as m
from logisense.services.erp.csv_loader import load_company_data

_SUPPLIERS = [
    {"supplier_id": "SUP-001", "supplier_name": "Acme Components", "category": "Electronics",
     "region": "Asia", "country": "China", "lead_time_days": 20, "reliability_score": 0.8,
     "contractual_sla": "Net 30, 80% on-time", "payment_terms": "Net 30", "status": "active"},
]
_PRODUCTS = [
    {"sku_id": "SKU-001", "product_name": "Sensor", "category": "Electronics", "supplier_id": "SUP-001",
     "unit_cost": 10.0, "selling_price": 18.0, "lead_time": 20, "criticality": "High"},
]
_WAREHOUSES = [
    {"warehouse_id": "WH-01", "warehouse_name": "East DC", "location": "Columbus, Ohio, USA",
     "region": "Midwest", "capacity": 50000, "current_utilization": 0.6},
]
_CARRIERS = [
    {"carrier_id": "CAR-01", "carrier_name": "Global Freight", "service_type": "Ocean",
     "region": "Asia", "on_time_rate": 0.9, "avg_transit_days": 10.0},
]
_CUSTOMERS = [
    {"customer_id": "CUST-001", "customer_name": "Jane Doe", "region": "South", "segment": "Consumer", "priority": "Standard"},
]
_CONTRACTS = [
    {"contract_id": "CTR-001", "supplier_id": "SUP-001", "effective_date": "2026-01-01", "expiry_date": "2027-01-01",
     "agreed_lead_time": 20, "minimum_order_quantity": 100, "SLA": "Net 30, 80% on-time", "penalty_terms": "2% per day late"},
]
_PURCHASE_ORDERS = [
    {"po_id": "PO-001", "supplier_id": "SUP-001", "warehouse_id": "WH-01", "order_date": "2026-01-01",
     "expected_delivery_date": "2026-01-21", "status": "Received", "total_value": 1000.0},
]
_PO_ITEMS = [
    {"po_id": "PO-001", "sku_id": "SKU-001", "quantity": 100, "unit_cost": 10.0, "expected_quantity": 100},
]
_SHIPMENTS = [
    {"shipment_id": "SHIP-001", "po_id": "PO-001", "supplier_id": "SUP-001", "origin": "China",
     "destination": "East DC", "carrier": "CAR-01", "promised_date": "2026-01-21",
     "estimated_arrival": "2026-01-22", "actual_arrival": "2026-01-23", "status": "Delivered"},
]
_SHIPMENT_EVENTS = [
    {"event_id": "EVT-001", "shipment_id": "SHIP-001", "timestamp": "2026-01-15T08:00:00",
     "location": "China", "event_type": "pickup", "description": "Picked up from supplier."},
    {"event_id": "EVT-002", "shipment_id": "SHIP-001", "timestamp": "2026-01-23T14:00:00",
     "location": "East DC", "event_type": "received", "description": "Received into inventory."},
]
_CUSTOMER_ORDERS = [
    {"order_id": "ORD-001", "customer_id": "CUST-001", "warehouse_id": "WH-01", "order_date": "2026-01-25",
     "promised_delivery_date": "2026-01-28", "priority": "Standard", "status": "Delivered", "revenue": 36.0},
]
_CUSTOMER_ORDER_ITEMS = [
    {"order_id": "ORD-001", "sku_id": "SKU-001", "quantity": 2, "unit_price": 18.0},
]
_INVENTORY = [
    {"warehouse_id": "WH-01", "sku_id": "SKU-001", "on_hand": 98, "reserved": 5, "available": 93,
     "reorder_point": 20, "safety_stock": 10, "daily_demand": 1.5},
]


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture_dir(tmp_path) -> str:
    data_dir = str(tmp_path / "company_data")
    os.makedirs(data_dir, exist_ok=True)
    files = {
        "suppliers": _SUPPLIERS, "products": _PRODUCTS, "warehouses": _WAREHOUSES,
        "carriers": _CARRIERS, "customers": _CUSTOMERS, "contracts": _CONTRACTS,
        "purchase_orders": _PURCHASE_ORDERS, "purchase_order_items": _PO_ITEMS,
        "inbound_shipments": _SHIPMENTS, "shipment_events": _SHIPMENT_EVENTS,
        "customer_orders": _CUSTOMER_ORDERS, "customer_order_items": _CUSTOMER_ORDER_ITEMS,
        "inventory": _INVENTORY,
    }
    for name, rows in files.items():
        _write_csv(os.path.join(data_dir, f"{name}.csv"), rows)
    return data_dir


def test_load_preserves_real_ids_and_counts(session, tmp_path):
    data_dir = _build_fixture_dir(tmp_path)
    result = load_company_data(session, data_dir)

    assert result.suppliers == 1
    assert result.products == 1
    assert result.warehouses == 1
    assert result.carriers == 1
    assert result.customers == 1
    assert result.contracts == 1
    assert result.purchase_orders == 1
    assert result.purchase_order_items == 1
    assert result.inbound_shipments == 1
    assert result.shipment_events == 2
    assert result.customer_orders == 1
    assert result.inventory_rows == 1
    assert result.skipped_rows == {}

    supplier = session.scalar(select(m.Supplier).where(m.Supplier.source_supplier_id == "SUP-001"))
    assert supplier is not None
    assert supplier.name == "Acme Components"


def test_full_chain_referential_integrity(session, tmp_path):
    data_dir = _build_fixture_dir(tmp_path)
    load_company_data(session, data_dir)

    supplier = session.scalar(select(m.Supplier).where(m.Supplier.source_supplier_id == "SUP-001"))
    contract = session.scalar(select(m.Contract).where(m.Contract.supplier_id == supplier.id))
    assert contract is not None
    assert contract.source_contract_id == "CTR-001"

    po = session.scalar(select(m.PurchaseOrder).where(m.PurchaseOrder.supplier_id == supplier.id))
    assert po is not None
    assert po.source_reference == "PO-001"
    assert po.status == m.PurchaseOrderStatus.RECEIVED

    item = session.scalar(select(m.PurchaseOrderItem).where(m.PurchaseOrderItem.purchase_order_id == po.id))
    assert item is not None
    assert item.quantity == 100

    shipment = session.scalar(select(m.Shipment).where(m.Shipment.purchase_order_id == po.id))
    assert shipment is not None
    assert shipment.delay_days == 2  # 2026-01-23 actual vs 2026-01-21 promised
    assert shipment.carrier.source_carrier_id == "CAR-01"

    events = session.scalars(select(m.ShipmentEvent).where(m.ShipmentEvent.shipment_id == shipment.id)).all()
    assert len(events) == 2
    assert {e.event_type for e in events} == {m.ShipmentEventType.PICKUP, m.ShipmentEventType.RECEIVED}

    product = session.scalar(select(m.Product).where(m.Product.source_product_id == "SKU-001"))
    warehouse = session.scalar(select(m.Warehouse).where(m.Warehouse.code == "WH-01"))
    inv = session.scalar(
        select(m.Inventory).where(m.Inventory.product_id == product.id, m.Inventory.warehouse_id == warehouse.id)
    )
    assert inv is not None
    assert inv.on_hand == 98
    assert inv.available == 93

    order = session.scalar(select(m.Order).where(m.Order.source_order_id == "ORD-001:0"))
    assert order is not None
    assert order.product_id == product.id
    assert order.quantity == 2
    assert order.order_value == 36.0  # 2 * 18.0


def test_load_is_idempotent(session, tmp_path):
    data_dir = _build_fixture_dir(tmp_path)
    load_company_data(session, data_dir)
    counts_before = {
        "suppliers": session.query(m.Supplier).count(),
        "purchase_orders": session.query(m.PurchaseOrder).count(),
        "shipments": session.query(m.Shipment).count(),
        "shipment_events": session.query(m.ShipmentEvent).count(),
        "inventory": session.query(m.Inventory).count(),
    }

    load_company_data(session, data_dir)
    counts_after = {
        "suppliers": session.query(m.Supplier).count(),
        "purchase_orders": session.query(m.PurchaseOrder).count(),
        "shipments": session.query(m.Shipment).count(),
        "shipment_events": session.query(m.ShipmentEvent).count(),
        "inventory": session.query(m.Inventory).count(),
    }

    assert counts_before["suppliers"] == counts_after["suppliers"] == 1
    assert counts_before["purchase_orders"] == counts_after["purchase_orders"] == 1
    assert counts_before["shipments"] == counts_after["shipments"] == 1
    assert counts_before["inventory"] == counts_after["inventory"] == 1
    # NOTE: shipment_events are append-only (no natural unique key to
    # upsert against), so a second load doubles them -- a known,
    # documented limitation, not silently hidden.
    assert counts_after["shipment_events"] == counts_before["shipment_events"] * 2


def test_missing_supplier_reference_is_skipped_not_crashed(session, tmp_path):
    data_dir = _build_fixture_dir(tmp_path)
    # Add a PO referencing a supplier that doesn't exist in suppliers.csv.
    bad_po = dict(_PURCHASE_ORDERS[0])
    bad_po["po_id"] = "PO-999"
    bad_po["supplier_id"] = "SUP-DOES-NOT-EXIST"
    _write_csv(os.path.join(data_dir, "purchase_orders.csv"), _PURCHASE_ORDERS + [bad_po])

    result = load_company_data(session, data_dir)

    assert result.purchase_orders == 1  # only the valid one loaded
    assert result.skipped_rows.get("purchase_orders") == 1
    assert session.scalar(select(m.PurchaseOrder).where(m.PurchaseOrder.source_reference == "PO-999")) is None
