import csv
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


def load_batch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "solve_ca_batch.py"
    spec = importlib.util.spec_from_file_location("solve_ca_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeFuture:
    def __init__(self, fn, task):
        self.fn = fn
        self.task = task

    def result(self):
        return self.fn(self.task)


class InlineExecutor:
    submitted = []

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, task):
        self.submitted.append(task)
        return FakeFuture(fn, task)


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_instance(root, split="test", name="ca_test_0000.lp.gz", content=b"lp"):
    path = root / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def write_result(output_root, instance, payload):
    output_dir = output_root / instance.name.removesuffix(".lp.gz")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        __import__("json").dumps(payload),
        encoding="utf-8",
    )


def test_resume_skips_valid_solution_and_keeps_one_csv_row(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    instance = make_instance(input_root)
    digest = hashlib.sha256(instance.read_bytes()).hexdigest()
    write_result(
        output_root,
        instance,
        {
            "status": "TIME_LIMIT",
            "has_feasible_solution": True,
            "objective": 10.0,
            "objective_bound": 11.0,
            "mip_gap": 0.1,
            "runtime": 2.0,
            "solution_count": 1,
            "threads": 2,
            "seed": 7,
            "time_limit": 120.0,
            "requested_mip_gap": 0.05,
            "mip_focus": 1,
            "heuristics": 0.2,
            "instance_sha256": digest,
        },
    )

    class BombExecutor:
        def __init__(self, max_workers):
            raise AssertionError("resume should not submit solved instances")

    monkeypatch.setattr(script, "EXECUTOR_CLASS", BombExecutor)

    args = [
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--resume",
    ]
    assert script.main(args) == 0
    assert script.main(args) == 0

    result_rows = rows(output_root / "results.csv")
    assert len(result_rows) == 1
    assert result_rows[0]["instance"] == "ca_test_0000"
    assert result_rows[0]["solution_count"] == "1"
    assert result_rows[0]["error"] == ""


def test_resume_reruns_sha_mismatch(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    instance = make_instance(input_root)
    write_result(output_root, instance, {"solution_count": 1, "instance_sha256": "0" * 64})

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_solve(task):
        return {
            "instance": "ca_test_0000",
            "split": "test",
            "status": "TIME_LIMIT",
            "has_feasible_solution": True,
            "objective": 1.0,
            "objective_bound": 2.0,
            "mip_gap": 0.5,
            "runtime": 1.0,
            "solution_count": 1,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "error": "",
        }

    monkeypatch.setattr(script, "solve_one_for_batch", fake_solve)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--resume",
    ]) == 0
    assert len(InlineExecutor.submitted) == 1
    assert rows(output_root / "results.csv")[0]["objective"] == "1.0"


def test_records_no_solution_and_worker_exception(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    make_instance(input_root, name="ca_test_0000.lp.gz")
    make_instance(input_root, name="ca_test_0001.lp.gz")

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_solve(task):
        instance = Path(task["instance"]).name.removesuffix(".lp.gz")
        if instance == "ca_test_0001":
            raise RuntimeError("solver exploded")
        return {
            "instance": instance,
            "split": "test",
            "status": "TIME_LIMIT",
            "has_feasible_solution": False,
            "objective": None,
            "objective_bound": 3.0,
            "mip_gap": None,
            "runtime": 4.0,
            "solution_count": 0,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "error": "",
        }

    monkeypatch.setattr(script, "solve_one_for_batch", fake_solve)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--workers", "2",
        "--threads", "1",
    ]) == 0

    result_rows = {row["instance"]: row for row in rows(output_root / "results.csv")}
    assert set(result_rows) == {"ca_test_0000", "ca_test_0001"}
    assert result_rows["ca_test_0000"]["has_feasible_solution"] == "False"
    assert result_rows["ca_test_0000"]["solution_count"] == "0"
    assert result_rows["ca_test_0000"]["error"] == ""
    assert result_rows["ca_test_0001"]["status"] == "ERROR"
    assert "RuntimeError: solver exploded" in result_rows["ca_test_0001"]["error"]


@pytest.mark.skipif(os.environ.get("RUN_GUROBI_INTEGRATION") != "1", reason="requires Gurobi license")
def test_real_gurobi_first_12_test_instances(tmp_path):
    script = load_batch_script()
    assert script.main([
        "--input-root", "data/ca",
        "--output-root", str(tmp_path / "ca_pass1"),
        "--split", "test",
        "--limit", "12",
        "--workers", "2",
        "--threads", "2",
        "--time-limit", "120",
        "--resume",
    ]) == 0


def test_instances_file_filters_deduplicates_and_sorts(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    make_instance(input_root, split="train", name="ca_train_0085.lp.gz")
    make_instance(input_root, split="test", name="ca_test_0052.lp.gz")
    make_instance(input_root, split="test", name="ca_test_0099.lp.gz")
    instances_file = tmp_path / "instances.txt"
    instances_file.write_text("ca_train_0085\nca_test_0052\nca_train_0085\n", encoding="utf-8")

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_solve(task):
        instance = Path(task["instance"]).name.removesuffix(".lp.gz")
        return {
            "instance": instance,
            "split": Path(task["instance"]).parent.name,
            "status": "TIME_LIMIT",
            "has_feasible_solution": True,
            "objective": 1.0,
            "objective_bound": 2.0,
            "mip_gap": 0.5,
            "runtime": 1.0,
            "solution_count": 1,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "improve_start_time": task["improve_start_time"],
            "improve_start_gap": task["improve_start_gap"],
            "warm_start_used": False,
            "warm_start_path": task["warm_start_path"],
            "error": "",
        }

    monkeypatch.setattr(script, "solve_one_for_batch", fake_solve)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--instances-file", str(instances_file),
    ]) == 0

    submitted = [Path(task["instance"]).name.removesuffix(".lp.gz") for task in InlineExecutor.submitted]
    assert submitted == ["ca_test_0052", "ca_train_0085"]
    result_rows = rows(output_root / "results.csv")
    assert [row["instance"] for row in result_rows] == ["ca_test_0052", "ca_train_0085"]


def test_instances_file_records_missing_instance(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    make_instance(input_root, name="ca_test_0000.lp.gz")
    instances_file = tmp_path / "instances.txt"
    instances_file.write_text("ca_missing_9999\nca_test_0000\n", encoding="utf-8")

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_solve(task):
        return {
            "instance": "ca_test_0000",
            "split": "test",
            "status": "TIME_LIMIT",
            "has_feasible_solution": True,
            "objective": 1.0,
            "objective_bound": 2.0,
            "mip_gap": 0.5,
            "runtime": 1.0,
            "solution_count": 1,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "improve_start_time": task["improve_start_time"],
            "improve_start_gap": task["improve_start_gap"],
            "warm_start_used": False,
            "warm_start_path": task["warm_start_path"],
            "error": "",
        }

    monkeypatch.setattr(script, "solve_one_for_batch", fake_solve)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--instances-file", str(instances_file),
    ]) == 0

    result_rows = {row["instance"]: row for row in rows(output_root / "results.csv")}
    assert result_rows["ca_missing_9999"]["status"] == "ERROR"
    assert "instance not found" in result_rows["ca_missing_9999"]["error"]
    assert result_rows["ca_test_0000"]["solution_count"] == "1"


def test_warm_start_root_passes_candidate_path_to_tasks(tmp_path, monkeypatch):
    script = load_batch_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "out"
    warm_root = tmp_path / "pass1"
    make_instance(input_root, name="ca_test_0000.lp.gz")
    (warm_root / "ca_test_0000").mkdir(parents=True)
    (warm_root / "ca_test_0000" / "best.sol").write_text("solution", encoding="utf-8")

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_solve(task):
        return {
            "instance": "ca_test_0000",
            "split": "test",
            "status": "TIME_LIMIT",
            "has_feasible_solution": True,
            "objective": 1.0,
            "objective_bound": 2.0,
            "mip_gap": 0.5,
            "runtime": 1.0,
            "solution_count": 1,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "improve_start_time": task["improve_start_time"],
            "improve_start_gap": task["improve_start_gap"],
            "warm_start_used": True,
            "warm_start_path": task["warm_start_path"],
            "error": "",
        }

    monkeypatch.setattr(script, "solve_one_for_batch", fake_solve)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--warm-start-root", str(warm_root),
        "--improve-start-gap", "0.20",
    ]) == 0

    expected = warm_root / "ca_test_0000" / "best.sol"
    assert InlineExecutor.submitted[0]["warm_start_path"] == str(expected)
    result = rows(output_root / "results.csv")[0]
    assert result["warm_start_used"] == "True"
    assert result["warm_start_path"] == str(expected)
    assert result["improve_start_gap"] == "0.2"
