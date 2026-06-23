from pathlib import Path

from milp_dataset.config import DatasetConfig
from milp_dataset.generation import generate_instances
from milp_dataset.manifest import read_manifest
from milp_dataset.validation import validate_dataset


class TinyGenerators:
    @staticmethod
    def generate_cauctions(*, filename, **kwargs):
        Path(filename).write_text("maximize\nOBJ: +1 x1\n\nsubject to\nC1: +1 x1 <= 1\n\nbinary\n x1\n", encoding="utf-8")

    @staticmethod
    def generate_setcover(*, filename, **kwargs):
        Path(filename).write_text("minimize\nOBJ: +1 x1\n\nsubject to\nC1: +1 x1 >= 1\n\nbinary\n x1\n", encoding="utf-8")


def test_custom_output_root_uses_posix_relative_paths(tmp_path, monkeypatch):
    import milp_dataset.generation as generation

    monkeypatch.setattr(generation, "_module", lambda root: TinyGenerators)
    monkeypatch.setattr(generation, "_commit", lambda root: "a" * 40)
    config = DatasetConfig(
        2025,
        {"train": 1, "valid": 1, "test": 1},
        {"generator": "learn2branch.combinatorial_auction", "n_items": 3, "n_bids": 5},
        {"generator": "learn2branch.set_covering", "nrows": 3, "ncols": 4, "density": 0.5, "max_coef": 3},
    )
    kwargs = dict(
        config=config, problems=["ca", "sc"], splits=["train", "valid", "test"],
        counts={"train": 1, "valid": 1, "test": 1}, output_root=tmp_path, repo_root=tmp_path,
    )
    first = generate_instances(**kwargs)
    records = read_manifest(tmp_path / "metadata" / "manifest.csv")
    assert len(records) == 6
    for record in records:
        assert not Path(record.relative_path).is_absolute()
        assert "\\" not in record.relative_path
        assert not record.relative_path.startswith(tmp_path.name + "/")
        assert (tmp_path / record.relative_path).is_file()
    assert validate_dataset(tmp_path) == []
    generate_instances(**kwargs)
    assert len(read_manifest(tmp_path / "metadata" / "manifest.csv")) == len(first)