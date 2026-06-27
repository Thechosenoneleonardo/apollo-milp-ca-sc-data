"""Generate bipartite graph files for CA LP instances."""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from solve_ca_one import instance_key

SPLITS = ("train", "valid", "test")
MANIFEST_FIELDS = [
    "instance",
    "split",
    "num_variables",
    "num_constraints",
    "num_edges",
    "output_path",
    "status",
    "error",
]
EXECUTOR_CLASS = ProcessPoolExecutor
AS_COMPLETED = as_completed


def finite_value(value: float) -> float:
    number = float(value)
    if math.isfinite(number):
        return number
    return 0.0


def bound_features(value: float) -> tuple[float, float]:
    number = float(value)
    if math.isfinite(number):
        return 1.0, number
    return 0.0, 0.0


def discover_instances(input_root: Path, split: str, limit: int | None) -> list[Path]:
    selected_splits = SPLITS if split == "all" else (split,)
    instances: list[Path] = []
    for split_name in selected_splits:
        instances.extend(sorted((input_root / split_name).glob("*.lp.gz")))
    if limit is not None:
        return instances[:limit]
    return instances


def split_name(instance: Path, input_root: Path) -> str:
    try:
        return instance.relative_to(input_root).parts[0]
    except ValueError:
        return instance.parent.name


def output_path_for(instance: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / split_name(instance, input_root) / f"{instance_key(instance)}.pkl"


def build_bg_payload(model: Any, *, instance: Path, split: str) -> dict[str, Any]:
    variables = list(model.getVars())
    constraints = list(model.getConstrs())
    matrix = model.getA().tocoo()

    variable_values = []
    for variable in variables:
        has_lb, lb = bound_features(getattr(variable, "LB", float("-inf")))
        has_ub, ub = bound_features(getattr(variable, "UB", float("inf")))
        vtype = getattr(variable, "VType", "")
        variable_values.append([
            finite_value(getattr(variable, "Obj", 0.0)),
            has_lb,
            lb,
            has_ub,
            ub,
            1.0 if vtype == "B" else 0.0,
            1.0 if vtype == "I" else 0.0,
            1.0 if vtype == "C" else 0.0,
        ])

    constraint_values = []
    for constraint in constraints:
        sense = getattr(constraint, "Sense", "")
        constraint_values.append([
            finite_value(getattr(constraint, "RHS", 0.0)),
            1.0 if sense == "<" else 0.0,
            1.0 if sense == "=" else 0.0,
            1.0 if sense == ">" else 0.0,
        ])

    edge_indices = np.vstack([
        np.asarray(matrix.row, dtype=np.int64),
        np.asarray(matrix.col, dtype=np.int64),
    ])
    edge_values = np.asarray(matrix.data, dtype=np.float32).reshape(-1, 1)

    constraint_features = {
        "names": ["rhs", "sense_le", "sense_eq", "sense_ge"],
        "values": np.asarray(constraint_values, dtype=np.float32).reshape(len(constraints), 4),
    }
    edge_features = {
        "names": ["coef"],
        "indices": edge_indices,
        "values": edge_values,
    }
    variable_features = {
        "names": ["obj", "has_lb", "lb", "has_ub", "ub", "is_binary", "is_integer", "is_continuous"],
        "values": np.asarray(variable_values, dtype=np.float32).reshape(len(variables), 8),
    }
    bg = (constraint_features, edge_features, variable_features)
    return {
        "format": "learn2branch_bipartite_state_v1",
        "instance": instance_key(instance),
        "split": split,
        "source_path": str(instance),
        "num_variables": len(variables),
        "num_constraints": len(constraints),
        "num_edges": int(edge_values.shape[0]),
        "data": bg,
        "bg": bg,
    }


def write_bg_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temp, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, output_path)


def load_bg_payload(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("BG payload must be a dict")
    return payload


def validate_bg_payload(payload: dict[str, Any], expected_instance: str | None = None) -> tuple[int, int, int]:
    if expected_instance is not None and payload.get("instance") != expected_instance:
        raise ValueError("instance mismatch")
    bg = payload.get("data")
    if not isinstance(bg, tuple) or len(bg) != 3:
        raise ValueError("data must be a 3-tuple BG state")
    constraint_features, edge_features, variable_features = bg
    for name, features in (
        ("constraint_features", constraint_features),
        ("edge_features", edge_features),
        ("variable_features", variable_features),
    ):
        if not isinstance(features, dict):
            raise ValueError(f"{name} must be a dict")
        if "names" not in features or "values" not in features:
            raise ValueError(f"{name} missing names or values")
    if "indices" not in edge_features:
        raise ValueError("edge_features missing indices")

    c_values = np.asarray(constraint_features["values"])
    v_values = np.asarray(variable_features["values"])
    e_values = np.asarray(edge_features["values"])
    e_indices = np.asarray(edge_features["indices"])
    if c_values.ndim != 2 or v_values.ndim != 2 or e_values.ndim != 2:
        raise ValueError("feature values must be 2D arrays")
    if e_indices.ndim != 2 or e_indices.shape[0] != 2:
        raise ValueError("edge indices must have shape (2, num_edges)")
    if e_indices.shape[1] != e_values.shape[0]:
        raise ValueError("edge indices and values length mismatch")
    return int(v_values.shape[0]), int(c_values.shape[0]), int(e_values.shape[0])


def resume_valid(output_path: Path, expected_instance: str) -> tuple[int, int, int] | None:
    try:
        return validate_bg_payload(load_bg_payload(output_path), expected_instance)
    except (OSError, EOFError, pickle.PickleError, ValueError, gzip.BadGzipFile):
        return None


def row_for_success(instance: Path, input_root: Path, output_path: Path, counts: tuple[int, int, int]) -> dict[str, Any]:
    num_variables, num_constraints, num_edges = counts
    return {
        "instance": instance_key(instance),
        "split": split_name(instance, input_root),
        "num_variables": num_variables,
        "num_constraints": num_constraints,
        "num_edges": num_edges,
        "output_path": str(output_path),
        "status": "OK",
        "error": "",
    }


def row_for_error(instance: Path, input_root: Path, output_path: Path, error: BaseException | str) -> dict[str, Any]:
    return {
        "instance": instance_key(instance),
        "split": split_name(instance, input_root),
        "num_variables": "",
        "num_constraints": "",
        "num_edges": "",
        "output_path": str(output_path),
        "status": "ERROR",
        "error": str(error),
    }


def generate_one(task: dict[str, str]) -> dict[str, Any]:
    instance = Path(task["instance"])
    input_root = Path(task["input_root"])
    output_root = Path(task["output_root"])
    output_path = output_path_for(instance, input_root, output_root)
    try:
        import gurobipy as gp
    except ImportError as error:
        return row_for_error(instance, input_root, output_path, "gurobipy is required to generate BG files")

    try:
        model = gp.read(str(instance))
        try:
            payload = build_bg_payload(model, instance=instance, split=split_name(instance, input_root))
            write_bg_payload(payload, output_path)
            counts = validate_bg_payload(payload, instance_key(instance))
            return row_for_success(instance, input_root, output_path, counts)
        finally:
            try:
                model.dispose()
            except Exception:
                pass
    except BaseException as error:
        return row_for_error(instance, input_root, output_path, f"{type(error).__name__}: {error}")


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["instance"]: {field: row.get(field, "") for field in MANIFEST_FIELDS} for row in csv.DictReader(stream) if row.get("instance")}


def write_manifest(rows: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({field: rows[key].get(field, "") for field in MANIFEST_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CA bipartite graph files for Apollo training.")
    parser.add_argument("--input-root", type=Path, default=Path("data/ca"))
    parser.add_argument("--output-root", type=Path, default=Path("data/ca/bg"))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")


def generate_batch(args: argparse.Namespace) -> int:
    validate_args(args)
    instances = discover_instances(args.input_root, args.split, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    tasks: list[dict[str, str]] = []

    for instance in instances:
        output_path = output_path_for(instance, args.input_root, args.output_root)
        if args.resume:
            counts = resume_valid(output_path, instance_key(instance))
            if counts is not None:
                row = row_for_success(instance, args.input_root, output_path, counts)
                rows[row["instance"]] = row
                write_manifest(rows, manifest_path)
                continue
        tasks.append({
            "instance": str(instance),
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
        })

    if tasks:
        with EXECUTOR_CLASS(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(generate_one, task): task for task in tasks}
            for future in AS_COMPLETED(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                except BaseException as error:
                    instance = Path(task["instance"])
                    output_path = output_path_for(instance, Path(task["input_root"]), Path(task["output_root"]))
                    row = row_for_error(instance, Path(task["input_root"]), output_path, f"{type(error).__name__}: {error}")
                rows[row["instance"]] = row
                write_manifest(rows, manifest_path)

    if not manifest_path.exists():
        write_manifest(rows, manifest_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return generate_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
