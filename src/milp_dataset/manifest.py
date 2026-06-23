"""Manifest record definitions and CSV serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_FIELDS = (
    "problem", "split", "index", "seed", "relative_path", "size_bytes", "sha256", "parameters"
)


@dataclass(frozen=True)
class ManifestRecord:
    problem: str
    split: str
    index: int
    seed: int
    relative_path: str
    size_bytes: int
    sha256: str
    parameters: dict[str, object]

    def validate(self) -> None:
        if self.problem not in {"ca", "sc"}:
            raise ValueError("manifest problem must be ca or sc")
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("manifest split is invalid")
        if self.index < 0 or self.seed < 0 or self.size_bytes < 0:
            raise ValueError("manifest numeric fields must be non-negative")
        if not self.relative_path.endswith(".lp.gz"):
            raise ValueError("manifest artifacts must end in .lp.gz")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    """Write validated records to a deterministic CSV manifest."""
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            record.validate()
            row = asdict(record)
            row["parameters"] = json.dumps(row["parameters"], sort_keys=True, separators=(",", ":"))
            writer.writerow(row)
