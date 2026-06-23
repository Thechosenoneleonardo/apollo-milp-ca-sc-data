from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from .config import DatasetConfig, load_config
from .generation import generate_instances, repository_root
from .manifest import read_manifest
from .validation import validate_dataset

def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo-MILP dataset tooling")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--problem", choices=["ca", "sc", "all"], default="all")
    generate.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    for split in ("train", "valid", "test"):
        generate.add_argument(f"--{split}-count", type=int, default=0)
    generate.add_argument("--base-seed", type=int); generate.add_argument("--config", type=Path, default=Path("configs/dataset.yaml")); generate.add_argument("--output-root", type=Path, default=repository_root()); generate.add_argument("--force", action="store_true"); generate.add_argument("--no-compress", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, default=Path("configs/dataset.yaml")); validate.add_argument("--output-root", type=Path, default=repository_root()); validate.add_argument("--strict-solver", action="store_true")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", type=Path, default=Path("configs/dataset.yaml")); summarize.add_argument("--output-root", type=Path, default=repository_root())
    args = parser.parse_args()
    if args.cmd == "generate":
        config = load_config(args.config); config = DatasetConfig(args.base_seed, config.splits, config.ca, config.sc) if args.base_seed is not None else config
        counts = {split: getattr(args, f"{split}_count") for split in ("train", "valid", "test")}
        if any(count < 0 for count in counts.values()): parser.error("counts must be non-negative")
        if args.no_compress: parser.error("uncompressed LP output is disallowed by repository policy")
        records = generate_instances(config, problems=["ca", "sc"] if args.problem == "all" else [args.problem], splits=["train", "valid", "test"] if args.split == "all" else [args.split], counts=counts, output_root=args.output_root, force=args.force)
        print(f"manifest contains {len(records)} instance records"); return 0
    if args.cmd == "validate":
        warnings: list[str] = []; errors = validate_dataset(args.output_root, load_config(args.config).splits, strict_solver=args.strict_solver, warnings=warnings)
        for warning in warnings: print(warning)
        print("validation passed" if not errors else "validation failed:\n" + "\n".join(errors)); return int(bool(errors))
    records = read_manifest(args.output_root / "metadata" / "manifest.csv"); counts = Counter((record.problem, record.split) for record in records)
    for (problem, split), count in sorted(counts.items()): print(f"{problem}/{split}: {count}")
    print(f"instances: {len(records)}; bytes: {sum(record.size_bytes for record in records)}"); return 0
if __name__ == "__main__": raise SystemExit(main())