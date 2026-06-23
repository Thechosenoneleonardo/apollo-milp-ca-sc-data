"""Dataset artifact validation with optional solver checks."""
from __future__ import annotations
import gzip
import hashlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable
from .manifest import read_manifest

SolverReader = Callable[[Path], None]

def _installed_solver_reader() -> tuple[str, SolverReader | None]:
    try:
        import gurobipy as gp
        return "gurobipy", lambda path: gp.read(str(path))
    except ImportError:
        pass
    try:
        from pyscipopt import Model
        def read(path: Path) -> None:
            model = Model()
            model.readProblem(str(path))
        return "pyscipopt", read
    except ImportError:
        return "solver", None

def _run_solver_reader(compressed_path: Path, reader: SolverReader) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".lp") as temporary:
            temp_path = Path(temporary.name)
            with gzip.open(compressed_path, "rb") as source:
                shutil.copyfileobj(source, temporary)
        reader(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

def validate_dataset(root: Path, expected_counts: dict[str, int] | None = None, *, strict_solver: bool = False, warnings: list[str] | None = None, solver_reader: SolverReader | None = None) -> list[str]:
    records = read_manifest(root / "metadata" / "manifest.csv")
    errors: list[str] = []
    notes = warnings if warnings is not None else []
    keys = [record.key for record in records]
    seeds = [record.seed for record in records]
    if len(keys) != len(set(keys)):
        errors.append("duplicate manifest instance key")
    if len(seeds) != len(set(seeds)):
        errors.append("duplicate seed")
    counts = Counter((record.problem, record.split) for record in records)
    if expected_counts is not None:
        for problem in ("ca", "sc"):
            for split, expected in expected_counts.items():
                if counts[(problem, split)] != expected:
                    errors.append(f"{problem}/{split}: expected {expected}, found {counts[(problem, split)]}")
    name, discovered = _installed_solver_reader()
    reader = solver_reader if solver_reader is not None else discovered
    if reader is None:
        notes.append(f"solver check skipped: {name} is not installed")
    for record in records:
        path = root / record.relative_path
        if not path.is_file():
            errors.append(f"missing artifact: {record.relative_path}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            errors.append(f"sha256 mismatch: {record.relative_path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                text = stream.read().lower()
        except Exception as error:
            errors.append(f"cannot decompress {record.relative_path}: {error}")
            continue
        if not (("maximize" in text or "minimize" in text) and "subject to" in text and "binary" in text):
            errors.append(f"invalid LP sections: {record.relative_path}")
        if reader is not None:
            try:
                _run_solver_reader(path, reader)
            except Exception as error:
                message = f"solver warning for {record.relative_path}: {error}"
                notes.append(message)
                if strict_solver:
                    errors.append(message)
    return errors