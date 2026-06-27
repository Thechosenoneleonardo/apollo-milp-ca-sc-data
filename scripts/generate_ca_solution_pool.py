"""Generate CA solution-pool files with Gurobi."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from solve_ca_one import STATUS_NAMES, instance_key

SPLITS = ("train", "valid", "test")
MANIFEST_FIELDS = [
    "instance",
    "split",
    "sol_count",
    "best_objective",
    "output_path",
    "warm_start_used",
    "status",
    "error",
]
EXECUTOR_CLASS = ProcessPoolExecutor
AS_COMPLETED = as_completed


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
    return output_root / split_name(instance, input_root) / f"{instance_key(instance)}.npz"


def warm_start_path_for(instance: Path, warm_start_root: Path) -> Path:
    return warm_start_root / instance_key(instance) / "best.sol"


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def attr_or_none(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def status_name(model: Any) -> str:
    code = attr_or_none(model, "Status")
    if code is None:
        return "UNKNOWN"
    try:
        return STATUS_NAMES.get(int(code), f"UNKNOWN_{int(code)}")
    except Exception:
        return str(code)


def set_gurobi_params(
    model: Any,
    *,
    threads: int,
    seed: int,
    time_limit: float | None,
    mip_focus: int,
    heuristics: float,
    pool_solutions: int,
    pool_gap: float,
) -> None:
    model.setParam("Threads", threads)
    model.setParam("Seed", seed)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    model.setParam("MIPFocus", mip_focus)
    model.setParam("Heuristics", heuristics)
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", pool_solutions)
    model.setParam("PoolGap", pool_gap)


def read_solution_pool(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variables = list(model.getVars())
    raw_solutions: list[tuple[float, np.ndarray]] = []
    seen: set[bytes] = set()

    for k in range(int(attr_or_none(model, "SolCount") or 0)):
        model.Params.SolutionNumber = k
        objective = finite_or_nan(attr_or_none(model, "PoolObjVal"))
        binary = np.asarray([1 if float(attr_or_none(variable, "Xn") or 0.0) > 0.5 else 0 for variable in variables], dtype=np.int8)
        key = binary.tobytes()
        if key in seen:
            continue
        seen.add(key)
        raw_solutions.append((objective, binary))

    raw_solutions.sort(key=lambda item: (-math.inf if math.isnan(item[0]) else item[0]), reverse=True)
    objectives = np.asarray([item[0] for item in raw_solutions], dtype=np.float64)
    if raw_solutions:
        binary_solutions = np.vstack([item[1] for item in raw_solutions]).astype(np.int8, copy=False)
    else:
        binary_solutions = np.zeros((0, len(variables)), dtype=np.int8)
    selected_bid_indices = np.asarray([np.flatnonzero(solution).astype(np.int32) for solution in binary_solutions], dtype=object)
    return objectives, binary_solutions, selected_bid_indices


def write_pool_npz(
    output_path: Path,
    *,
    objectives: np.ndarray,
    binary_solutions: np.ndarray,
    selected_bid_indices: np.ndarray,
    variable_names: np.ndarray,
    pool_gap: float,
    time_limit: float | None,
    seed: int,
    status: str,
    warm_start_used: bool,
    warm_start_path: Path | None,
    warnings: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    best_objective = objectives[0] if len(objectives) else np.nan
    np.savez_compressed(
        temp,
        format=np.asarray("apollo_ca_solution_pool_v1"),
        objectives=objectives,
        binary_solutions=binary_solutions,
        selected_bid_indices=selected_bid_indices,
        variable_names=variable_names,
        sol_count=np.asarray(len(objectives), dtype=np.int32),
        best_objective=np.asarray(best_objective, dtype=np.float64),
        pool_gap=np.asarray(pool_gap, dtype=np.float64),
        time_limit=np.asarray(np.nan if time_limit is None else time_limit, dtype=np.float64),
        seed=np.asarray(seed, dtype=np.int64),
        status=np.asarray(status),
        warm_start_used=np.asarray(warm_start_used),
        warm_start_path=np.asarray("" if warm_start_path is None else str(warm_start_path)),
        warnings=np.asarray(warnings, dtype=object),
    )
    final_temp = temp if temp.exists() else Path(str(temp) + ".npz")
    os.replace(final_temp, output_path)


def validate_pool_npz(path: Path) -> tuple[int, float]:
    required = {
        "objectives",
        "binary_solutions",
        "selected_bid_indices",
        "variable_names",
        "sol_count",
        "best_objective",
        "pool_gap",
        "time_limit",
        "seed",
        "status",
    }
    with np.load(path, allow_pickle=True) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        objectives = np.asarray(data["objectives"])
        binary_solutions = np.asarray(data["binary_solutions"])
        variable_names = np.asarray(data["variable_names"])
        sol_count = int(np.asarray(data["sol_count"]).item())
        if objectives.ndim != 1:
            raise ValueError("objectives must be 1D")
        if binary_solutions.ndim != 2:
            raise ValueError("binary_solutions must be 2D")
        if binary_solutions.shape[0] != objectives.shape[0] or sol_count != objectives.shape[0]:
            raise ValueError("solution count mismatch")
        if binary_solutions.shape[1] != variable_names.shape[0]:
            raise ValueError("variable count mismatch")
        if sol_count <= 0:
            raise ValueError("no saved solutions")
        return sol_count, float(np.asarray(data["best_objective"]).item())


def resume_valid(path: Path) -> tuple[int, float] | None:
    try:
        return validate_pool_npz(path)
    except (OSError, ValueError, KeyError):
        return None


def row_for_success(
    instance: Path,
    input_root: Path,
    output_path: Path,
    *,
    sol_count: int,
    best_objective: float,
    warm_start_used: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "instance": instance_key(instance),
        "split": split_name(instance, input_root),
        "sol_count": sol_count,
        "best_objective": best_objective,
        "output_path": str(output_path),
        "warm_start_used": warm_start_used,
        "status": status,
        "error": "",
    }


def row_for_error(instance: Path, input_root: Path, output_path: Path, error: BaseException | str) -> dict[str, Any]:
    return {
        "instance": instance_key(instance),
        "split": split_name(instance, input_root),
        "sol_count": "",
        "best_objective": "",
        "output_path": str(output_path),
        "warm_start_used": False,
        "status": "ERROR",
        "error": str(error),
    }


def generate_one(task: dict[str, Any]) -> dict[str, Any]:
    instance = Path(task["instance"])
    input_root = Path(task["input_root"])
    output_root = Path(task["output_root"])
    warm_start_root = Path(task["warm_start_root"])
    output_path = output_path_for(instance, input_root, output_root)
    warm_start_path = warm_start_path_for(instance, warm_start_root)
    warm_start_used = False
    warnings: list[str] = []

    try:
        import gurobipy as gp
    except ImportError:
        return row_for_error(instance, input_root, output_path, "gurobipy is required to generate solution pools")

    try:
        model = gp.read(str(instance))
        try:
            set_gurobi_params(
                model,
                threads=int(task["threads"]),
                seed=int(task["seed"]),
                time_limit=task["time_limit"],
                mip_focus=int(task["mip_focus"]),
                heuristics=float(task["heuristics"]),
                pool_solutions=int(task["pool_solutions"]),
                pool_gap=float(task["pool_gap"]),
            )
            if warm_start_path.is_file():
                try:
                    model.read(str(warm_start_path))
                    warm_start_used = True
                except Exception as error:
                    warnings.append(f"failed to read warm start {warm_start_path}: {error}")

            model.optimize()
            variables = list(model.getVars())
            variable_names = np.asarray([str(getattr(variable, "VarName", f"x{i}")) for i, variable in enumerate(variables)])
            objectives, binary_solutions, selected_bid_indices = read_solution_pool(model)
            status = status_name(model)
            write_pool_npz(
                output_path,
                objectives=objectives,
                binary_solutions=binary_solutions,
                selected_bid_indices=selected_bid_indices,
                variable_names=variable_names,
                pool_gap=float(task["pool_gap"]),
                time_limit=task["time_limit"],
                seed=int(task["seed"]),
                status=status,
                warm_start_used=warm_start_used,
                warm_start_path=warm_start_path if warm_start_path.is_file() else None,
                warnings=warnings,
            )
            best_objective = float(objectives[0]) if len(objectives) else float("nan")
            return row_for_success(
                instance,
                input_root,
                output_path,
                sol_count=int(len(objectives)),
                best_objective=best_objective,
                warm_start_used=warm_start_used,
                status=status,
            )
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
    parser = argparse.ArgumentParser(description="Generate CA Gurobi solution-pool files for Apollo training.")
    parser.add_argument("--input-root", type=Path, default=Path("data/ca"))
    parser.add_argument("--warm-start-root", type=Path, default=Path("outputs/gurobi/ca_final"))
    parser.add_argument("--output-root", type=Path, default=Path("data/ca/solution_pool"))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-solutions", type=int, default=50)
    parser.add_argument("--pool-gap", type=float, default=0.20)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--heuristics", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.pool_solutions < 1:
        raise SystemExit("--pool-solutions must be positive")
    if args.pool_gap < 0:
        raise SystemExit("--pool-gap must be non-negative")


def generate_batch(args: argparse.Namespace) -> int:
    validate_args(args)
    instances = discover_instances(args.input_root, args.split, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    tasks: list[dict[str, Any]] = []

    for instance in instances:
        output_path = output_path_for(instance, args.input_root, args.output_root)
        if args.resume:
            existing = resume_valid(output_path)
            if existing is not None:
                sol_count, best_objective = existing
                row = row_for_success(
                    instance,
                    args.input_root,
                    output_path,
                    sol_count=sol_count,
                    best_objective=best_objective,
                    warm_start_used=False,
                    status="OK",
                )
                rows[row["instance"]] = row
                write_manifest(rows, manifest_path)
                continue
        tasks.append({
            "instance": str(instance),
            "input_root": str(args.input_root),
            "warm_start_root": str(args.warm_start_root),
            "output_root": str(args.output_root),
            "threads": args.threads,
            "time_limit": args.time_limit,
            "seed": args.seed,
            "pool_solutions": args.pool_solutions,
            "pool_gap": args.pool_gap,
            "mip_focus": args.mip_focus,
            "heuristics": args.heuristics,
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
