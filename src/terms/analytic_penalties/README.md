# Analytic Penalty Manifests

This directory is the registry layer for analytic penalties implemented in
`src/terms/analytic_penalties.rs` and `src/terms/sheaf.rs`.

`mod.rs` defines `PenaltyManifest` metadata for every registered penalty:

- `KIND_TAG`: serialized descriptor kind.
- `PYTHON_WRAPPER`: Python wrapper class exposed through `gamfit`.
- `ROW_BLOCK_DIAGONAL`: whether the penalty can stay in the row-block-diagonal
  solver path.

`analytic_penalty_registry!` is the source list consumed by `build.rs` to emit
`gamfit/_penalties_manifest.py`. Add new penalties here only after the Rust
penalty type and Python wrapper contract exist.
