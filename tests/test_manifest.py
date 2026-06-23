import csv

import pytest

from milp_dataset.manifest import MANIFEST_FIELDS, ManifestRecord, parameters_json, write_manifest


def make_record(**changes: object) -> ManifestRecord:
    values: dict[str, object] = {
        "problem": "ca", "split": "train", "index": 0, "seed": 1,
        "relative_path": "data/ca/train/ca_train_0000.lp.gz",
        "generator": "learn2branch.combinatorial_auction", "generator_commit": "a" * 40,
        "parameters_json": parameters_json({"n_items": 3, "n_bids": 5}),
        "size_bytes": 100, "sha256": "a" * 64,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    values.update(changes)
    return ManifestRecord(**values)  # type: ignore[arg-type]


def test_write_manifest_serializes_required_fields(tmp_path) -> None:
    output = tmp_path / "manifest.csv"
    write_manifest([make_record()], output)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == MANIFEST_FIELDS
    assert rows[0]["parameters_json"] == '{"n_bids":5,"n_items":3}'


@pytest.mark.parametrize("changes", [{"relative_path": "bad.lp"}, {"sha256": "not-a-digest"}])
def test_manifest_rejects_invalid_records(changes: dict[str, object], tmp_path) -> None:
    with pytest.raises(ValueError):
        write_manifest([make_record(**changes)], tmp_path / "manifest.csv")