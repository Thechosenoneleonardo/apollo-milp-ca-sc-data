"""Resume-friendly batch solver for CA LP instances."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from solve_ca_one import file_sha256, instance_key, solve_instance


RESULT_FIELDS = [
    "instance",
    "split",
    "status",
    "has_feasible_solution",
    "objective",
    "objective_bound",
    "mip_gap",
    "runtime",
    "solution_count",
    "threads",
    "seed",
    "time_limit",
    "requested_mip_gap",
    "mip_focus",
    "heuristics",
    "improve_start_time",
    "improve_start_gap",
    "warm_start_used",
    "warm_start_path",
    "error",
]

SPLITS = ("train", "valid", "test")
EXECUTOR_CLASS = ProcessPoolExecutor
AS_COMPLETED = as_completed


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def read_instance_names(path: Path) -> list[str]:
    names = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            name = line.strip()
            if name:
                names.add(name)
    return sorted(names)


def index_instances(input_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(input_root.rglob("*.lp.gz")):
        index.setdefault(instance_key(path), path)
    return index


def discover_instances(
    input_root: Path,
    split: str,
    limit: int | None,
    instances_file: Path | None = None,
) -> tuple[list[Path], list[str]]:
    if instances_file is not None:
        names = read_instance_names(instances_file)
        index = index_instances(input_root)
        instances = [index[name] for name in names if name in index]
        missing = [name for name in names if name not in index]
        if limit is not None:
            instances = instances[:limit]
        return instances, missing

    selected_splits = SPLITS if split == "all" else (split,)
    instances: list[Path] = []
    for split_name in selected_splits:
        split_dir = input_root / split_name
        if not split_dir.is_dir():
            continue
        instances.extend(sorted(split_dir.glob("*.lp.gz")))
    if limit is not None:
        instances = instances[:limit]
    return instances, []


def split_name(instance: Path, input_root: Path) -> str:
    try:
        return instance.relative_to(input_root).parts[0]
    except ValueError:
        return instance.parent.name


def read_results_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            if row.get("instance"):
                rows[row["instance"]] = {field: row.get(field, "") for field in RESULT_FIELDS}
        return rows


def write_results_csv(rows: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            row = {field: rows[key].get(field, "") for field in RESULT_FIELDS}
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def row_from_payload(
    *,
    instance: Path,
    input_root: Path,
    payload: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    return {
        "instance": instance_key(instance),
        "split": split_name(instance, input_root),
        "status": payload.get("status", "ERROR" if error else ""),
        "has_feasible_solution": bool(payload.get("has_feasible_solution", False)),
        "objective": payload.get("objective"),
        "objective_bound": payload.get("objective_bound"),
        "mip_gap": payload.get("mip_gap"),
        "runtime": payload.get("runtime"),
        "solution_count": int(payload.get("solution_count") or 0),
        "threads": payload.get("threads"),
        "seed": payload.get("seed"),
        "time_limit": payload.get("time_limit"),
        "requested_mip_gap": payload.get("requested_mip_gap"),
        "mip_focus": payload.get("mip_focus"),
        "heuristics": payload.get("heuristics"),
        "improve_start_time": payload.get("improve_start_time"),
        "improve_start_gap": payload.get("improve_start_gap"),
        "warm_start_used": bool(payload.get("warm_start_used", False)),
        "warm_start_path": payload.get("warm_start_path"),
        "error": error,
    }


def missing_instance_row(name: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "instance": name,
        "split": "",
        "status": "ERROR",
        "has_feasible_solution": False,
        "objective": None,
        "objective_bound": None,
        "mip_gap": None,
        "runtime": None,
        "solution_count": 0,
        "threads": args.threads,
        "seed": args.seed,
        "time_limit": args.time_limit,
        "requested_mip_gap": args.target_gap,
        "mip_focus": args.mip_focus,
        "heuristics": args.heuristics,
        "improve_start_time": args.improve_start_time,
        "improve_start_gap": args.improve_start_gap,
        "warm_start_used": False,
        "warm_start_path": None,
        "error": f"instance not found under input root: {name}",
    }


def resume_payload(instance: Path, output_root: Path, expected_sha256: str) -> dict[str, Any] | None:
    result_path = output_root / instance_key(instance) / "result.json"
    try:
        payload = load_json(result_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if payload.get("instance_sha256") != expected_sha256:
        return None
    if int(payload.get("solution_count") or 0) <= 0:
        return None
    if not bool(payload.get("has_feasible_solution", True)):
        return None
    return payload


def warm_start_for(instance: Path, warm_start_root: Path | None) -> Path | None:
    if warm_start_root is None:
        return None
    return warm_start_root / instance_key(instance) / "best.sol"


def make_task(
    *,
    instance: Path,
    input_root: Path,
    output_root: Path,
    threads: int,
    seed: int,
    time_limit: float | None,
    mip_focus: int,
    heuristics: float,
    improve_start_time: float,
    improve_start_gap: float | None,
    target_gap: float,
    warm_start_path: Path | None,
) -> dict[str, Any]:
    return {
        "instance": str(instance),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "threads": threads,
        "seed": seed,
        "time_limit": time_limit,
        "mip_focus": mip_focus,
        "heuristics": heuristics,
        "improve_start_time": improve_start_time,
        "improve_start_gap": improve_start_gap,
        "target_gap": target_gap,
        "warm_start_path": str(warm_start_path) if warm_start_path is not None else None,
    }


def solve_one_for_batch(task: dict[str, Any]) -> dict[str, Any]:
    instance = Path(task["instance"])
    input_root = Path(task["input_root"])
    output_root = Path(task["output_root"])
    warm_start_path = Path(task["warm_start_path"]) if task.get("warm_start_path") else None
    try:
        output_dir = solve_instance(
            instance=instance,
            output_root=output_root,
            threads=int(task["threads"]),
            seed=int(task["seed"]),
            time_limit=task["time_limit"],
            mip_focus=int(task["mip_focus"]),
            heuristics=float(task["heuristics"]),
            improve_start_time=float(task["improve_start_time"]),
            improve_start_gap=task["improve_start_gap"],
            target_gap=float(task["target_gap"]),
            warm_start_path=warm_start_path,
        )
        payload = load_json(output_dir / "result.json")
        return row_from_payload(instance=instance, input_root=input_root, payload=payload)
    except BaseException as error:
        payload = {
            "status": "ERROR",
            "has_feasible_solution": False,
            "solution_count": 0,
            "threads": task["threads"],
            "seed": task["seed"],
            "time_limit": task["time_limit"],
            "requested_mip_gap": task["target_gap"],
            "mip_focus": task["mip_focus"],
            "heuristics": task["heuristics"],
            "improve_start_time": task["improve_start_time"],
            "improve_start_gap": task["improve_start_gap"],
            "warm_start_used": False,
            "warm_start_path": task.get("warm_start_path"),
        }
        return row_from_payload(
            instance=instance,
            input_root=input_root,
            payload=payload,
            error=f"{type(error).__name__}: {error}",
        )


def warn_if_oversubscribed(workers: int, threads: int) -> None:
    logical = os.cpu_count()
    if logical is not None and workers * threads > logical:
        print(
            f"warning: workers * threads = {workers * threads} exceeds logical processors = {logical}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve CA instances in a resumable Gurobi batch.")
    parser.add_argument("--input-root", type=Path, default=Path("data/ca"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gurobi/ca_pass1"))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instances-file", type=Path)
    parser.add_argument("--warm-start-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--heuristics", type=float, default=0.20)
    parser.add_argument("--improve-start-time", type=float, default=60)
    parser.add_argument("--improve-start-gap", type=float)
    parser.add_argument("--target-gap", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.instances_file is not None and not args.instances_file.is_file():
        raise SystemExit(f"--instances-file does not exist: {args.instances_file}")


def solve_batch(args: argparse.Namespace) -> int:
    validate_args(args)
    warn_if_oversubscribed(args.workers, args.threads)

    instances, missing = discover_instances(args.input_root, args.split, args.limit, args.instances_file)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.csv"
    rows = read_results_csv(results_path)
    tasks: list[dict[str, Any]] = []

    for name in missing:
        rows[name] = missing_instance_row(name, args)
        write_results_csv(rows, results_path)

    for instance in instances:
        digest = file_sha256(instance)
        warm_start_path = warm_start_for(instance, args.warm_start_root)
        if args.resume:
            payload = resume_payload(instance, args.output_root, digest)
            if payload is not None:
                row = row_from_payload(instance=instance, input_root=args.input_root, payload=payload)
                rows[row["instance"]] = row
                write_results_csv(rows, results_path)
                continue
        tasks.append(
            make_task(
                instance=instance,
                input_root=args.input_root,
                output_root=args.output_root,
                threads=args.threads,
                seed=args.seed,
                time_limit=args.time_limit,
                mip_focus=args.mip_focus,
                heuristics=args.heuristics,
                improve_start_time=args.improve_start_time,
                improve_start_gap=args.improve_start_gap,
                target_gap=args.target_gap,
                warm_start_path=warm_start_path,
            )
        )

    if tasks:
        with EXECUTOR_CLASS(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(solve_one_for_batch, task): task for task in tasks}
            for future in AS_COMPLETED(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                except BaseException as error:
                    instance = Path(task["instance"])
                    payload = {
                        "status": "ERROR",
                        "has_feasible_solution": False,
                        "solution_count": 0,
                        "threads": task["threads"],
                        "seed": task["seed"],
                        "time_limit": task["time_limit"],
                        "requested_mip_gap": task["target_gap"],
                        "mip_focus": task["mip_focus"],
                        "heuristics": task["heuristics"],
                        "improve_start_time": task["improve_start_time"],
                        "improve_start_gap": task["improve_start_gap"],
                        "warm_start_used": False,
                        "warm_start_path": task.get("warm_start_path"),
                    }
                    row = row_from_payload(
                        instance=instance,
                        input_root=args.input_root,
                        payload=payload,
                        error=f"{type(error).__name__}: {error}",
                    )
                rows[row["instance"]] = row
                write_results_csv(rows, results_path)

    if not results_path.exists():
        write_results_csv(rows, results_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return solve_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
