"""Configuration loading and validation for the dataset protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    base_seed: int
    splits: dict[str, int]
    ca: dict[str, Any]
    sc: dict[str, Any]

    def validate(self) -> None:
        expected_splits = {"train": 240, "valid": 60, "test": 100}
        if self.splits != expected_splits:
            raise ValueError(f"splits must equal {expected_splits}, got {self.splits}")
        if self.ca.get("n_items") != 300 or self.ca.get("n_bids") != 1500:
            raise ValueError("CA configuration must use n_items=300 and n_bids=1500")
        if self.sc.get("nrows") != 3000 or self.sc.get("ncols") != 5000:
            raise ValueError("SC configuration must use nrows=3000 and ncols=5000")


def load_config(path: str | Path) -> DatasetConfig:
    """Load and validate a protocol YAML file."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")

    dataset = raw.get("dataset", {})
    config = DatasetConfig(
        base_seed=int(dataset["base_seed"]),
        splits=dict(dataset["splits"]),
        ca=dict(raw["ca"]),
        sc=dict(raw["sc"]),
    )
    config.validate()
    return config
