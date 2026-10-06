"""Shared builders for DocumentExtraction-shaped payloads (what a model would return)."""

from typing import Any


def extracted(value: Any, confidence: float = 0.9, **extra: Any) -> dict[str, Any]:
    return {"value": value, "confidence": confidence, "status": "extracted", **extra}


def absent(confidence: float = 0.95, **extra: Any) -> dict[str, Any]:
    return {"value": None, "confidence": confidence, "status": "absent", **extra}


def abstained(reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "value": None,
        "confidence": 0.2,
        "status": "abstained",
        "reasons": ["smudged"] if reasons is None else reasons,
    }


def row(**extra: Any) -> dict[str, Any]:
    return {
        "description": "Widget",
        "quantity": 2,
        "unit_price": "4.50",
        "amount": 9.0,
        "status": "extracted",
        "confidence": 0.8,
        "evidence": "Widget 2 4.50 9.00",
        "page": 1,
        **extra,
    }


def full_payload() -> dict[str, Any]:
    return {
        "vendor_name": extracted("ACME Corp", evidence="ACME Corp", page=1),
        "vendor_address": extracted("1 Main St"),
        "vendor_tax_id": absent(),
        "buyer_name": extracted("Jane Doe"),
        "document_number": extracted("INV-001"),
        "document_date": extracted("2024-03-05"),
        "due_date": absent(0.7, reasons=["no due date printed"]),
        "currency": extracted("USD"),
        "payment_method": abstained(),
        "line_items": [row()],
        "subtotal": extracted("9.00"),
        "tax": extracted(0.72),
        "discount": absent(),
        "total": extracted("9.72"),
    }
