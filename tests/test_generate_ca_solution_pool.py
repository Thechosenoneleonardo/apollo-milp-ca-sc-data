import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_ca_solution_pool.py"
    spec = importlib.util.spec_from_file_location("generate_ca_solution_pool", path)
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


class FakeParams:
    SolutionNumber = 0


class FakeVar:
    def __init__(self, model, name, index):
        self.model = model
        self.VarName = name
        self.index = index

    @property
    def Xn(self):
        return self.model.solutions[self.model.Params.SolutionNumber][1][self.index]


class FakeModel:
    Status = 9

    def __init__(self, *, read_error=None):
        self.Params = FakeParams()
        self.Params.SolutionNumber = 0
        self.params = {}
        self.read_paths = []
        self.optimized = False
        self.disposed = False
        self.read_error = read_error
        self.solutions = [
            (8.0, [1.0, 0.0, 1.0]),
            (10.0, [0.0, 1.0, 1.0]),
            (9.0, [1.0, 0.0, 1.0]),
        ]
        self.vars = [FakeVar(self, "x0", 0), FakeVar(self, "x1", 1), FakeVar(self, "x2", 2)]

    @property
    def SolCount(self):
        return len(self.solutions)

    @property
    def PoolObjVal(self):
        return self.solutions[self.Params.SolutionNumber][0]

    def setParam(self, name, value):
        self.params[name] = value

    def read(self, path):
        self.read_paths.append(path)
        if self.read_error is not None:
            raise self.read_error

    def optimize(self):
        self.optimized = True

    def getVars(self):
        return self.vars

    def dispose(self):
        self.disposed = True


def install_fake_gurobi(monkeypatch, *, read_error=None):
    fake = types.ModuleType("gurobipy")
    fake.last_model = None

    def read(path):
        fake.last_model = FakeModel(read_error=read_error)
        return fake.last_model

    fake.read = read
    monkeypatch.setitem(sys.modules, "gurobipy", fake)
    return fake


def make_instance(root, split="test", name="ca_test_0000.lp.gz"):
    path = root / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lp")
    return path


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_generate_one_writes_sorted_deduplicated_solution_pool(tmp_path, monkeypatch):
    script = load_script()
    fake = install_fake_gurobi(monkeypatch)
    input_root = tmp_path / "data" / "ca"
    warm_root = tmp_path / "outputs" / "gurobi" / "ca_final"
    output_root = tmp_path / "pool"
    instance = make_instance(input_root)
    warm_start = warm_root / "ca_test_0000" / "best.sol"
    warm_start.parent.mkdir(parents=True)
    warm_start.write_text("solution", encoding="utf-8")

    row = script.generate_one({
        "instance": str(instance),
        "input_root": str(input_root),
        "warm_start_root": str(warm_root),
        "output_root": str(output_root),
        "threads": 2,
        "time_limit": 12.0,
        "seed": 7,
        "pool_solutions": 50,
        "pool_gap": 0.2,
        "mip_focus": 1,
        "heuristics": 0.25,
    })

    assert row["status"] == "TIME_LIMIT"
    assert row["sol_count"] == 2
    assert row["best_objective"] == 10.0
    assert row["warm_start_used"] is True
    assert fake.last_model.read_paths == [str(warm_start)]
    assert fake.last_model.params["PoolSearchMode"] == 2
    assert fake.last_model.params["PoolSolutions"] == 50
    assert fake.last_model.params["PoolGap"] == 0.2

    output = output_root / "test" / "ca_test_0000.npz"
    with np.load(output, allow_pickle=True) as data:
        assert data["format"].item() == "apollo_ca_solution_pool_v1"
        assert data["objectives"].tolist() == [10.0, 8.0]
        assert data["binary_solutions"].tolist() == [[0, 1, 1], [1, 0, 1]]
        assert data["variable_names"].tolist() == ["x0", "x1", "x2"]
        assert data["sol_count"].item() == 2
        assert data["best_objective"].item() == 10.0
        assert data["pool_gap"].item() == 0.2
        assert data["time_limit"].item() == 12.0
        assert data["seed"].item() == 7
        assert data["status"].item() == "TIME_LIMIT"
        assert data["warm_start_used"].item() is True


def test_warm_start_read_failure_continues_and_records_warning(tmp_path, monkeypatch):
    script = load_script()
    install_fake_gurobi(monkeypatch, read_error=RuntimeError("bad sol"))
    input_root = tmp_path / "data" / "ca"
    warm_root = tmp_path / "warm"
    output_root = tmp_path / "pool"
    instance = make_instance(input_root)
    warm_start = warm_root / "ca_test_0000" / "best.sol"
    warm_start.parent.mkdir(parents=True)
    warm_start.write_text("bad", encoding="utf-8")

    row = script.generate_one({
        "instance": str(instance),
        "input_root": str(input_root),
        "warm_start_root": str(warm_root),
        "output_root": str(output_root),
        "threads": 1,
        "time_limit": None,
        "seed": 0,
        "pool_solutions": 3,
        "pool_gap": 0.5,
        "mip_focus": 1,
        "heuristics": 0.25,
    })

    assert row["status"] == "TIME_LIMIT"
    assert row["warm_start_used"] is False
    with np.load(output_root / "test" / "ca_test_0000.npz", allow_pickle=True) as data:
        assert "bad sol" in data["warnings"].tolist()[0]
        assert data["warm_start_used"].item() is False


def test_resume_skips_existing_valid_pool_and_keeps_one_manifest_row(tmp_path, monkeypatch):
    script = load_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "pool"
    make_instance(input_root)
    output = output_root / "test" / "ca_test_0000.npz"
    script.write_pool_npz(
        output,
        objectives=np.asarray([3.0]),
        binary_solutions=np.asarray([[1, 0]], dtype=np.int8),
        selected_bid_indices=np.asarray([np.asarray([0], dtype=np.int32)], dtype=object),
        variable_names=np.asarray(["x0", "x1"]),
        pool_gap=0.2,
        time_limit=1.0,
        seed=0,
        status="TIME_LIMIT",
        warm_start_used=False,
        warm_start_path=None,
        warnings=[],
    )

    class BombExecutor:
        def __init__(self, max_workers):
            raise AssertionError("resume should skip valid pools")

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
    assert manifest_rows[0]["sol_count"] == "1"


def test_worker_exception_is_recorded_in_manifest(tmp_path, monkeypatch):
    script = load_script()
    input_root = tmp_path / "data" / "ca"
    output_root = tmp_path / "pool"
    warm_root = tmp_path / "warm"
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
        return script.row_for_success(
            instance,
            Path(task["input_root"]),
            output_path,
            sol_count=2,
            best_objective=5.0,
            warm_start_used=False,
            status="TIME_LIMIT",
        )

    monkeypatch.setattr(script, "generate_one", fake_generate)

    assert script.main([
        "--input-root", str(input_root),
        "--warm-start-root", str(warm_root),
        "--output-root", str(output_root),
        "--split", "test",
        "--workers", "2",
    ]) == 0

    manifest_rows = {row["instance"]: row for row in rows(output_root / "manifest.csv")}
    assert manifest_rows["ca_test_0000"]["status"] == "TIME_LIMIT"
    assert manifest_rows["ca_test_0001"]["status"] == "ERROR"
    assert "RuntimeError: bad instance" in manifest_rows["ca_test_0001"]["error"]
