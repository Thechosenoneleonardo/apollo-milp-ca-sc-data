import gzip
import hashlib
from milp_dataset.manifest import ManifestRecord, parameters_json, write_manifest
from milp_dataset.validation import validate_dataset

def test_problem_filter_allows_single_problem_dataset(tmp_path):
    artifact = tmp_path / "data/ca/train/one.lp.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(gzip.compress(b"maximize\nOBJ: +1 x1\n\nsubject to\nC1: +1 x1 <= 1\n\nbinary\n x1\n", mtime=0))
    record = ManifestRecord("ca", "train", 0, 1, "data/ca/train/one.lp.gz", "test", "a" * 40, parameters_json({}), artifact.stat().st_size, hashlib.sha256(artifact.read_bytes()).hexdigest(), "2026-01-01T00:00:00+00:00")
    write_manifest([record], tmp_path / "metadata/manifest.csv")
    counts = {"train": 1, "valid": 0, "test": 0}
    assert validate_dataset(tmp_path, counts, problems={"ca"}, solver_reader=lambda path: None) == []
    assert validate_dataset(tmp_path, counts, solver_reader=lambda path: None)
    assert validate_dataset(tmp_path, {"train": 0, "valid": 0, "test": 0}, problems={"sc"}, solver_reader=lambda path: None) == []