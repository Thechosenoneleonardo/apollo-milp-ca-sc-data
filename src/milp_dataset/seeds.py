"""Collision-free seed derivation for the configured instance identity space."""
from __future__ import annotations

_MAX_INDEX = 1_000_000
_SEGMENT = _MAX_INDEX
_PROBLEM_OFFSET = {"ca": 0, "sc": 3 * _SEGMENT}
_SPLIT_OFFSET = {"train": 0, "valid": _SEGMENT, "test": 2 * _SEGMENT}
_MAX_OFFSET = 6 * _SEGMENT

def derive_seed(base_seed: int, problem: str, split: str, index: int) -> int:
    """Return a deterministic NumPy RandomState-compatible seed with no collisions.

    The supported identity space is six problem/split segments of one million
    indices each. The base seed selects a disjoint rotation of that space.
    """
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if problem not in _PROBLEM_OFFSET:
        raise ValueError("problem must be 'ca' or 'sc'")
    if split not in _SPLIT_OFFSET:
        raise ValueError("split must be train, valid, or test")
    if not 0 <= index < _MAX_INDEX:
        raise ValueError(f"index must be in [0, {_MAX_INDEX})")
    return (base_seed % (2**32 - _MAX_OFFSET)) + _PROBLEM_OFFSET[problem] + _SPLIT_OFFSET[split] + index