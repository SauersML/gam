# gam-data integration notes

Moved `src/inference/data.rs` to `crates/gam-data/src/lib.rs`.

External non-data `crate::` references inside the moved code:

- `crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn}` in `src/lib.rs`

No `crate::types::X` or `crate::model_types::X` references were present, so no
`gam-types` placeholder dependency was added.

References to the old engine data module were intentionally left untouched for
the central integrator rewrite from `crate::inference::data` /
`gam::inference::data` to `gam_data`:

- `src/solver/fit_orchestration/materialize/tests.rs`
- `src/families/survival/predict.rs`
- `src/solver/fit_orchestration/error.rs`
- `src/solver/fit_orchestration.rs`
- `src/families/multinomial.rs`
- `src/terms/smooth_overrides.rs`
- `src/terms/term_builder.rs`
- `crates/gam-cli/src/main.rs`
- `crates/gam-cli/src/main/dataset_io.rs`
- `crates/gam-cli/src/main/cli_errors.rs`
- `crates/gam-predict/src/input.rs`
- `crates/gam-pyffi/src/ffi_prelude.rs`
- `crates/gam-pyffi/src/manifold_and_posterior_ffi.rs`
