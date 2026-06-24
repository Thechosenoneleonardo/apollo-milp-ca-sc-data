from pathlib import Path
from milp_dataset.config import DatasetConfig
from milp_dataset.generation import generate_instances, repository_root
from milp_dataset.manifest import read_manifest

def test_workers_produce_identical_artifacts(tmp_path):
    root = repository_root()
    config = DatasetConfig(2025, {"train": 2, "valid": 0, "test": 0}, {"generator":"learn2branch.combinatorial_auction","n_items":3,"n_bids":5}, {"generator":"learn2branch.set_covering","nrows":3,"ncols":4,"density":0.5,"max_coef":3})
    first = root / ".tmp" / "pytest-workers-one"
    second = root / ".tmp" / "pytest-workers-two"
    import shutil
    shutil.rmtree(first, ignore_errors=True); shutil.rmtree(second, ignore_errors=True)
    try:
        generate_instances(config, problems=["ca"], splits=["train"], counts={"train":2,"valid":0,"test":0}, output_root=first, repo_root=root, workers=1)
        generate_instances(config, problems=["ca"], splits=["train"], counts={"train":2,"valid":0,"test":0}, output_root=second, repo_root=root, workers=2)
        a=read_manifest(first/"metadata/manifest.csv"); b=read_manifest(second/"metadata/manifest.csv")
        assert [(r.seed,r.relative_path,r.sha256) for r in a] == [(r.seed,r.relative_path,r.sha256) for r in b]
        generate_instances(config, problems=["ca"], splits=["train"], counts={"train":2,"valid":0,"test":0}, output_root=second, repo_root=root, workers=2)
        assert len(read_manifest(second/"metadata/manifest.csv")) == 2
    finally:
        shutil.rmtree(first, ignore_errors=True); shutil.rmtree(second, ignore_errors=True)