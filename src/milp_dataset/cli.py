"""Command-line entry point placeholder for future generation commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo-MILP dataset tooling")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Loaded configuration with base_seed={config.base_seed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
