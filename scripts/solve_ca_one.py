"""Solve one compressed CA LP instance with Gurobi."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def instance_key(path: Path) -> str:
    name = path.name
    if name.endswith(".lp.gz"):
        return name.removesuffix(".lp.gz")
    return path.stem


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if math.isfinite(number):
        return value
    return None


def attr_or_none(model: Any, name: str) -> Any:
    try:
        return getattr(model, name)
    except Exception:
        return None


def result_payload(
    model: Any,
    grb: Any,
    *,
    requested_mip_gap: float | None = None,
    mip_focus: int | None = None,
    heuristics: float | None = None,
    improve_start_time: float | None = None,
    improve_start_gap: float | None = None,
    threads: int | None = None,
    seed: int | None = None,
    time_limit: float | None = None,
    instance_sha256: str | None = None,
    warm_start_used: bool = False,
    warm_start_path: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    status_code = int(model.Status)
    has_solution = int(attr_or_none(model, "SolCount") or 0) > 0
    return {
        "status": STATUS_NAMES.get(status_code, f"UNKNOWN_{status_code}"),
        "optimal": status_code == grb.OPTIMAL,
        "has_feasible_solution": has_solution,
        "objective": finite_or_none(attr_or_none(model, "ObjVal")) if has_solution else None,
        "objective_bound": finite_or_none(attr_or_none(model, "ObjBound")),
        "mip_gap": finite_or_none(attr_or_none(model, "MIPGap")) if has_solution else None,
        "runtime": finite_or_none(attr_or_none(model, "Runtime")),
        "node_count": finite_or_none(attr_or_none(model, "NodeCount")),
        "solution_count": int(attr_or_none(model, "SolCount") or 0),
        "requested_mip_gap": requested_mip_gap,
        "mip_focus": mip_focus,
        "heuristics": heuristics,
        "improve_start_time": improve_start_time,
        "improve_start_gap": improve_start_gap,
        "threads": threads,
        "seed": seed,
        "time_limit": time_limit,
        "instance_sha256": instance_sha256,
        "warm_start_used": warm_start_used,
        "warm_start_path": warm_start_path,
        "warnings": warnings or [],
    }


def write_selected_bids(model: Any, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variable", "value"])
        for variable in model.getVars():
            value = float(variable.X)
            if value > 0.5:
                writer.writerow([variable.VarName, value])


def append_log_warning(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"WARNING: {message}\n")


def solve_instance(
    instance: Path,
    output_root: Path,
    threads: int,
    seed: int,
    time_limit: float | None,
    mip_focus: int = 1,
    heuristics: float = 0.20,
    improve_start_time: float = 60,
    target_gap: float = 0.05,
    improve_start_gap: float | None = None,
    warm_start_path: Path | None = None,
) -> Path:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise SystemExit("gurobipy is required to solve CA instances") from error

    if not instance.is_file():
        raise SystemExit(f"instance does not exist: {instance}")
    if instance.suffix != ".gz" or not instance.name.endswith(".lp.gz"):
        raise SystemExit(f"instance must be a compressed .lp.gz file: {instance}")

    instance_digest = file_sha256(instance)
    output_dir = output_root / instance_key(instance)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "gurobi.log"
    warm_start_text = str(warm_start_path) if warm_start_path is not None else None
    warm_start_used = False
    warnings: list[str] = []

    env = gp.Env(empty=True)
    env.setParam("LogFile", str(log_path))
    env.setParam("LogToConsole", 0)
    env.start()
    model = None
    try:
        model = gp.read(str(instance), env=env)
        model.setParam("Threads", threads)
        model.setParam("Seed", seed)
        model.setParam("MIPFocus", mip_focus)
        model.setParam("Heuristics", heuristics)
        model.setParam("ImproveStartTime", improve_start_time)
        if improve_start_gap is not None:
            model.setParam("ImproveStartGap", improve_start_gap)
        model.setParam("MIPGap", target_gap)
        if time_limit is not None:
            model.setParam("TimeLimit", time_limit)

        if warm_start_path is not None and warm_start_path.is_file():
            try:
                model.read(str(warm_start_path))
                warm_start_used = True
            except Exception as error:
                message = f"failed to read warm start {warm_start_path}: {error}"
                warnings.append(message)
                append_log_warning(log_path, message)

        model.optimize()
        payload = result_payload(
            model,
            GRB,
            requested_mip_gap=target_gap,
            mip_focus=mip_focus,
            heuristics=heuristics,
            improve_start_time=improve_start_time,
            improve_start_gap=improve_start_gap,
            threads=threads,
            seed=seed,
            time_limit=time_limit,
            instance_sha256=instance_digest,
            warm_start_used=warm_start_used,
            warm_start_path=warm_start_text,
            warnings=warnings,
        )

        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if payload["solution_count"] > 0:
            model.write(str(output_dir / "best.sol"))
            write_selected_bids(model, output_dir / "selected_bids.csv")

        return output_dir
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        env.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve one CA .lp.gz instance with Gurobi.")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--heuristics", type=float, default=0.20)
    parser.add_argument("--improve-start-time", type=float, default=60)
    parser.add_argument("--improve-start-gap", type=float)
    parser.add_argument("--target-gap", type=float, default=0.05)
    parser.add_argument("--warm-start", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    solve_instance(
        instance=args.instance,
        output_root=args.output_root,
        threads=args.threads,
        seed=args.seed,
        time_limit=args.time_limit,
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        improve_start_time=args.improve_start_time,
        improve_start_gap=args.improve_start_gap,
        target_gap=args.target_gap,
        warm_start_path=args.warm_start,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
