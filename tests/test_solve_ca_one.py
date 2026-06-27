import csv
import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "solve_ca_one.py"
    spec = importlib.util.spec_from_file_location("solve_ca_one", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeGRB:
    OPTIMAL = 2


class FakeModel:
    Status = 9
    SolCount = 1
    ObjVal = 10.0
    ObjBound = 12.0
    MIPGap = 0.2
    Runtime = 3.5
    NodeCount = 7.0


class FakeVar:
    def __init__(self, name, value):
        self.VarName = name
        self.X = value


class FakeSolutionModel:
    def getVars(self):
        return [FakeVar("x1", 1.0), FakeVar("x2", 0.0), FakeVar("x3", 0.51)]


def test_result_payload_marks_time_limit_not_optimal():
    script = load_script()

    payload = script.result_payload(FakeModel(), FakeGRB)

    assert payload["status"] == "TIME_LIMIT"
    assert payload["optimal"] is False
    assert payload["objective"] == 10.0
    assert payload["solution_count"] == 1


def test_instance_key_strips_lp_gz_suffix():
    script = load_script()

    assert script.instance_key(Path("data/ca/test/ca_test_0000.lp.gz")) == "ca_test_0000"


def test_write_selected_bids_only_writes_selected_variables(tmp_path):
    script = load_script()
    output = tmp_path / "selected_bids.csv"

    script.write_selected_bids(FakeSolutionModel(), output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [["variable", "value"], ["x1", "1.0"], ["x3", "0.51"]]


def install_fake_gurobi(monkeypatch, *, read_error=None):
    import sys
    import types

    class FakeEnv:
        def __init__(self, empty=True):
            self.params = {}
            self.started = False
            self.disposed = False

        def setParam(self, name, value):
            self.params[name] = value

        def start(self):
            self.started = True

        def dispose(self):
            self.disposed = True

    class SolverModel:
        Status = 9
        SolCount = 0
        ObjBound = 12.0
        Runtime = 3.5
        NodeCount = 7.0

        def __init__(self):
            self.params = {}
            self.read_paths = []
            self.optimized = False
            self.disposed = False

        def setParam(self, name, value):
            self.params[name] = value

        def read(self, path):
            self.read_paths.append(path)
            if read_error is not None:
                raise read_error

        def optimize(self):
            self.optimized = True

        def write(self, path):
            Path(path).write_text("solution", encoding="utf-8")

        def getVars(self):
            return []

        def dispose(self):
            self.disposed = True

    fake_module = types.ModuleType("gurobipy")
    fake_module.GRB = FakeGRB

    def fake_read(path, env=None):
        fake_module.last_model = SolverModel()
        return fake_module.last_model

    fake_module.Env = FakeEnv
    fake_module.read = fake_read
    fake_module.last_model = None
    monkeypatch.setitem(sys.modules, "gurobipy", fake_module)
    return fake_module


def test_solve_instance_reads_existing_warm_start_and_sets_improve_start_gap(tmp_path, monkeypatch):
    script = load_script()
    fake_gurobi = install_fake_gurobi(monkeypatch)
    instance = tmp_path / "ca_test_0000.lp.gz"
    instance.write_bytes(b"lp")
    warm_start = tmp_path / "warm" / "ca_test_0000" / "best.sol"
    warm_start.parent.mkdir(parents=True)
    warm_start.write_text("solution", encoding="utf-8")

    output_dir = script.solve_instance(
        instance=instance,
        output_root=tmp_path / "out",
        threads=2,
        seed=3,
        time_limit=4.0,
        improve_start_gap=0.2,
        warm_start_path=warm_start,
    )

    payload = __import__("json").loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert fake_gurobi.last_model.read_paths == [str(warm_start)]
    assert fake_gurobi.last_model.params["ImproveStartGap"] == 0.2
    assert payload["warm_start_used"] is True
    assert payload["warm_start_path"] == str(warm_start)
    assert payload["improve_start_gap"] == 0.2


def test_solve_instance_records_missing_warm_start_without_reading(tmp_path, monkeypatch):
    script = load_script()
    fake_gurobi = install_fake_gurobi(monkeypatch)
    instance = tmp_path / "ca_test_0000.lp.gz"
    instance.write_bytes(b"lp")
    warm_start = tmp_path / "warm" / "ca_test_0000" / "best.sol"

    output_dir = script.solve_instance(
        instance=instance,
        output_root=tmp_path / "out",
        threads=2,
        seed=3,
        time_limit=None,
        warm_start_path=warm_start,
    )

    payload = __import__("json").loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert fake_gurobi.last_model.read_paths == []
    assert payload["warm_start_used"] is False
    assert payload["warm_start_path"] == str(warm_start)


def test_solve_instance_warns_and_continues_when_warm_start_read_fails(tmp_path, monkeypatch):
    script = load_script()
    install_fake_gurobi(monkeypatch, read_error=RuntimeError("bad sol"))
    instance = tmp_path / "ca_test_0000.lp.gz"
    instance.write_bytes(b"lp")
    warm_start = tmp_path / "warm" / "ca_test_0000" / "best.sol"
    warm_start.parent.mkdir(parents=True)
    warm_start.write_text("not a solution", encoding="utf-8")

    output_dir = script.solve_instance(
        instance=instance,
        output_root=tmp_path / "out",
        threads=2,
        seed=3,
        time_limit=None,
        warm_start_path=warm_start,
    )

    payload = __import__("json").loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["warm_start_used"] is False
    assert "bad sol" in payload["warnings"][0]
    assert "WARNING: failed to read warm start" in (output_dir / "gurobi.log").read_text(encoding="utf-8")
