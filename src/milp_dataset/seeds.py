"""Stable, independent seed derivation for dataset instances."""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, problem: str, split: str, index: int) -> int:
    """Derive a deterministic unsigned 64-bit seed from an instance identity."""
    if problem not in {"ca", "sc"}:
        raise ValueError("problem must be 'ca' or 'sc'")
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if index < 0:
        raise ValueError("index must be non-negative")

    payload = f"apollo-milp-ca-sc-data/v1|{base_seed}|{problem}|{split}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
