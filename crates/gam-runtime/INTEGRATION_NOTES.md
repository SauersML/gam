# gam-runtime Integration Notes

- No moved file references engine-only modules such as `crate::types`,
  `crate::families`, `crate::model_types`, or `crate::linalg`.
- Existing `crate::warm_start` references are local sibling references within
  this crate and were left unchanged.
- Root `Cargo.toml`, `src/lib.rs`, and `Cargo.lock` were not edited.
