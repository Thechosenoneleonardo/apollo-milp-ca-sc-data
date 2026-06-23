"""Validation helpers reserved for generated dataset artifacts."""

from __future__ import annotations

from .manifest import ManifestRecord


def validate_record(record: ManifestRecord) -> None:
    """Validate the structural invariants of one manifest record."""
    record.validate()
