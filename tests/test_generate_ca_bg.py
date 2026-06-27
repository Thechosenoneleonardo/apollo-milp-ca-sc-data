import csv
import gzip
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_ca_bg.py"
    spec = importlib.util.spec_from_file_location("generate_ca_bg", path)
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


class FakeVar:
    def __init__(self, obj, lb, ub, vtype):
        self.Obj = obj
        self.LB = lb
        self.UB = ub
        self.VType = vtype


class FakeConstr:
    def __init__(self, rhs, sense):
        self.RHS = rhs
        self.Sense = sense


class FakeCoo:
    row = np.array([0, 1, 1], dtype=np.int64)
    col = np.array([0, 0, 1], dtype=np.int64)
    data = np.array([1.5, -2.0, 3.0], dtype=np.float32)

    def tocoo(self):
        return self


class FakeModel:
    disposed = False

    def getVars(self):
        return [FakeVar(4.0, 0.0, 1.0, "B"), FakeVar(-1.0, 0.0, float("inf"), "C")]

    def getConstrs(self):
        return [FakeConstr(5.0, "<"), FakeConstr(2.0, "=")]

    def getA(self):
        return FakeCoo()

    def dispose(self):
        self.disposed = True


def make_instance(root, split="test", name="ca_test_0000.lp.gz"):
    path = root / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lp")
    return path


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def install_fake_gurobi(monkeypatch):
    fake = types.ModuleType("gurobipy")
    fake.last_model = None

    def read(path):
        fake.last_model = FakeModel()
        return fake.last_model

    fake.read = read
    monkeypatch.setitem(sys.modules, "gurobipy", fake)
    return fake


def test_generate_one_writes_apollo_bg_payload(tmp_path, monkeypatch):
    script = load_script()
    install_fake_gurobi(monkeypatch)
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "bg"
    instance = make_instance(input_root)

    row = script.generate_one({
        "instance": str(instance),
        "input_root": str(input_root),
        "output_root": str(output_root),
    })

    assert row["status"] == "OK"
    assert row["num_variables"] == 2
    assert row["num_constraints"] == 2
    assert row["num_edges"] == 3
    output = output_root / "test" / "ca_test_0000.pkl"
    with gzip.open(output, "rb") as stream:
        payload = pickle.load(stream)

    assert payload["instance"] == "ca_test_0000"
    assert payload["format"] == "learn2branch_bipartite_state_v1"
    assert payload["data"] == payload["bg"]
    c, e, v = payload["data"]
    assert c["names"] == ["rhs", "sense_le", "sense_eq", "sense_ge"]
    assert v["names"] == ["obj", "has_lb", "lb", "has_ub", "ub", "is_binary", "is_integer", "is_continuous"]
    assert e["names"] == ["coef"]
    assert c["values"].shape == (2, 4)
    assert v["values"].shape == (2, 8)
    assert e["indices"].shape == (2, 3)
    assert e["values"].shape == (3, 1)


def test_resume_skips_existing_valid_bg_and_keeps_one_manifest_row(tmp_path, monkeypatch):
    script = load_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "bg"
    instance = make_instance(input_root)
    payload = script.build_bg_payload(FakeModel(), instance=instance, split="test")
    script.write_bg_payload(payload, output_root / "test" / "ca_test_0000.pkl")

    class BombExecutor:
        def __init__(self, max_workers):
            raise AssertionError("resume should skip valid BG files")

    monkeypatch.setattr(script, "EXECUTOR_CLASS", BombExecutor)

    args = [
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--resume",
    ]
    assert script.main(args) == 0
    assert script.main(args) == 0

    manifest_rows = rows(output_root / "manifest.csv")
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["instance"] == "ca_test_0000"
    assert manifest_rows[0]["status"] == "OK"
    assert manifest_rows[0]["num_edges"] == "3"


def test_worker_exception_is_recorded_in_manifest(tmp_path, monkeypatch):
    script = load_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "bg"
    make_instance(input_root, name="ca_test_0000.lp.gz")
    make_instance(input_root, name="ca_test_0001.lp.gz")

    InlineExecutor.submitted = []
    monkeypatch.setattr(script, "EXECUTOR_CLASS", InlineExecutor)
    monkeypatch.setattr(script, "AS_COMPLETED", lambda futures: list(futures))

    def fake_generate(task):
        instance = Path(task["instance"])
        if instance.name == "ca_test_0001.lp.gz":
            raise RuntimeError("bad instance")
        output_path = script.output_path_for(instance, Path(task["input_root"]), Path(task["output_root"]))
        return script.row_for_success(instance, Path(task["input_root"]), output_path, (2, 3, 4))

    monkeypatch.setattr(script, "generate_one", fake_generate)

    assert script.main([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--workers", "2",
    ]) == 0

    manifest_rows = {row["instance"]: row for row in rows(output_root / "manifest.csv")}
    assert manifest_rows["ca_test_0000"]["status"] == "OK"
    assert manifest_rows["ca_test_0001"]["status"] == "ERROR"
    assert "RuntimeError: bad instance" in manifest_rows["ca_test_0001"]["error"]


def test_discover_instances_respects_split_and_limit(tmp_path):
    script = load_script()
    input_root = tmp_path / "data" / "ca"
    make_instance(input_root, split="train", name="ca_train_0000.lp.gz")
    make_instance(input_root, split="valid", name="ca_valid_0000.lp.gz")
    make_instance(input_root, split="test", name="ca_test_0000.lp.gz")
    make_instance(input_root, split="test", name="ca_test_0001.lp.gz")

    test_instances = script.discover_instances(input_root, "test", 1)
    all_instances = script.discover_instances(input_root, "all", None)

    assert [path.name for path in test_instances] == ["ca_test_0000.lp.gz"]
    assert [path.name for path in all_instances] == [
        "ca_train_0000.lp.gz",
        "ca_valid_0000.lp.gz",
        "ca_test_0000.lp.gz",
        "ca_test_0001.lp.gz",
    ]
