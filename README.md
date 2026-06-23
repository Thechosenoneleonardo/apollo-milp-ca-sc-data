# apollo-milp-ca-sc-data

This repository provides a protocol-level reproduction of Apollo-MILP combinatorial auction (CA) and set covering (SC) data generation. It targets Python 3.14 or later.

## Data scale

Each problem has 240 training, 60 validation, and 100 test instances. CA uses 300 items and 1,500 bids. SC uses 3,000 rows and 5,000 columns. Artifacts are CPLEX LP files compressed as `.lp.gz`.

## Reproduction limitation

The paper does not publicly specify every seed or generator parameter. This repository does not claim instance-by-instance equivalence with the authors' original data. In particular, SC `density=0.05` is a reproduction assumption unless `configs/dataset.yaml` records a reliable source.

## Commands

`python -m milp_dataset generate` creates only explicitly requested per-split counts and never overwrites artifacts without `--force`. `validate` checks the manifest and LP artifacts; `summarize` reports manifest totals. No formal 800-instance dataset has been generated.