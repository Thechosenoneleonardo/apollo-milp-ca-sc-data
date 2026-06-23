# apollo-milp-ca-sc-data

This repository reproduces, at protocol level, the dataset-generation workflow used by Apollo-MILP for combinatorial auction (CA) and set covering (SC) mixed-integer linear programs.

The project targets Python 3.14 or later.

## Data scale

Each problem class will contain 240 training, 60 validation, and 100 test instances. CA targets 300 items and 1,500 bids. SC targets 3,000 rows and 5,000 columns. Generated instances will be stored as `.lp.gz` files.

## Reproduction limitation

The paper does not publicly specify every random seed or generation parameter. This repository therefore provides a protocol-level reproduction and does not claim byte-for-byte or instance-by-instance equivalence with the authors' original data.

## Next steps

The next phase will pin the learn2branch submodule, implement generator adapters, write manifests, and add dataset integrity validation. No formal dataset instances have been generated yet.
