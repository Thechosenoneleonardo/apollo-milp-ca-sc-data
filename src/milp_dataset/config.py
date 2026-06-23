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
        if not isinstance(self.base_seed, int) or self.base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if set(self.splits) != {"train", "valid", "test"} or any(not isinstance(count, int) or count < 0 for count in self.splits.values()):
            raise ValueError("splits must contain non-negative train, valid, and test counts")
        for key in ("n_items", "n_bids"):
            if not isinstance(self.ca.get(key), int) or self.ca[key] < 1:
                raise ValueError(f"CA {key} must be a positive integer")
        for key in ("nrows", "ncols"):
            if not isinstance(self.sc.get(key), int) or self.sc[key] < 1:
                raise ValueError(f"SC {key} must be a positive integer")
        density = self.sc.get("density")
        if not isinstance(density, (int, float)) or not 0 < density <= 1:
            raise ValueError("SC density must be in (0, 1]")

def load_config(path: str | Path) -> DatasetConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    dataset = raw.get("dataset", {})
    config = DatasetConfig(int(dataset["base_seed"]), dict(dataset["splits"]), dict(raw["ca"]), dict(raw["sc"]))
    config.validate()
    return config