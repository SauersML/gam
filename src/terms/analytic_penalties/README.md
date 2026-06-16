# Analytic Penalty Manifests

This directory holds the analytic penalty primitives plus their registry layer
(`registry.rs`) and the manifest layer (`manifest.rs`); penalties are also drawn
from `src/terms/sheaf.rs`.

`manifest.rs` defines `PenaltyManifest` metadata for every registered penalty:

- `KIND_TAG`: serialized descriptor kind.
- `PYTHON_WRAPPER`: Python wrapper class exposed through `gamfit`.
- `ROW_BLOCK_DIAGONAL`: whether the penalty can stay in the row-block-diagonal
  solver path.

`analytic_penalty_registry!` is the source list consumed by `build.rs` to emit
`gamfit/_penalties_manifest.py`. Add new penalties here only after the Rust
penalty type and Python wrapper contract exist.
