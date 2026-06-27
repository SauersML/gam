# #1521 Wave-2 central wiring spec (apply ALL; single writer)

Worktree /Users/user/gam-wt-carve. Apply by CONTENT match (line numbers may have drifted from the moves). Byte-identical otherwise. NO local build (CI verifies). NO commit (orchestrator commits).

## gam-solve/src/mod.rs
- REMOVE line `pub mod fit_orchestration;`
- REMOVE line `pub use fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors;`
- REMOVE line `pub mod protocol;`
- CHANGE `pub(crate) mod persistent_warm_start;` → `pub mod persistent_warm_start;`

## gam-solve/src/persistent_warm_start.rs
- Promote `pub(crate)`→`pub` on: `store_record`, `load_record`, `cache_schema_tag`, `PersistentWarmStartRecord` (+ any of its fields the drivers read). Bodies untouched.

## gam-solve/src/model_types.rs — PenaltySpec canonicalization (dedup, NOT deletion)
- Replace the entire `pub enum PenaltySpec { ... }` definition (the Block/Dense/DenseWithMean enum) with:
  `pub use gam_terms::penalty_spec::PenaltySpec;`
- Verified safe: gam-terms def is structurally identical; both use `gam_problem::CoefficientPriorMean`, `gam_terms::smooth::PenaltyStructureHint`, `gam_terms::analytic_penalties::PenaltyOp`. gam-solve already deps gam-terms. `estimate/mod.rs:36 pub use crate::model_types::{...PenaltySpec}` then propagates the canonical type. Keep the adjacent `pub use gam_problem::CoefficientPriorMean;` line.

## gam-terms/src/smooth.rs
- REMOVE `include!("smooth/design_construction.rs");`
- REMOVE `include!("smooth/spatial_optimization.rs");`
- CHANGE to `pub mod`: `input_standardization`, `shape_constraints`, `penalty_priors`, `structure_analysis` (these `mod X;` → `pub mod X;`)

## gam-terms/src/smooth/prelude.rs
- The back-edge import block consumed only by the departed drivers (`use crate::custom_family::…`, `use crate::estimate::…`, `use crate::families::family_runtime::…`) — REMOVE it. term_specs still needs only `EstimationError` → add `use gam_problem::EstimationError;`. (`UnifiedFitResult` no longer referenced after the struct cut — confirm none left; if any remain, they belong to code that also departed.)

## gam-terms/src/inference/mod.rs
- ADD `pub mod formula_dsl;`

## gam-models/src/mod.rs
- ADD `pub mod fit_orchestration;`
- ADD `pub mod protocol;`
- CONVERT the inline block `pub mod inference { pub use gam_math::probability; pub use gam_solve::quadrature; }` → `pub mod inference;`

## gam-models/src/inference/mod.rs  (NEW FILE)
```
pub use gam_math::probability;
pub use gam_solve::quadrature;
pub mod full_conformal;
pub mod generative;
pub mod model;
pub mod model_payload_builders;
pub mod predict_io;
```

## gam-models/src/bms/gpu/mod.rs
- ADD `pub mod device_pcg;` (after `pub mod row;`)

## gam-models consumer rewrites (in-crate)
- gamlss.rs: `use crate::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};` → `use crate::inference::generative::{...}`
- multinomial.rs: `use gam_solve::fit_orchestration::{` → `use crate::fit_orchestration::{`; and `gam_solve::fit_orchestration::response_column_kind` → `crate::fit_orchestration::response_column_kind`
- marginal_slope_orthogonal.rs: `use gam_terms::smooth::build_term_collection_design;` → `use crate::fit_orchestration::drivers::build_term_collection_design;`
- survival/predict.rs: `gam_terms::smooth::build_term_collection_design(` → `crate::fit_orchestration::drivers::build_term_collection_design(`

## root src/inference/mod.rs
- REMOVE the 6 lines: `pub mod formula_dsl;` `pub mod full_conformal;` `pub mod generative;` `pub mod model;` `pub mod model_payload_builders;` `pub mod predict_io;`
- ADD: `pub use gam_models::inference::{full_conformal, generative, model, model_payload_builders, predict_io};`
- ADD: `pub use gam_terms::inference::formula_dsl;`
- (Leave `pub mod model_comparison;` + the `#[cfg(test)] mod marginal_slope_predict_tests;` as-is.)

## root src/lib.rs
- `pub use solver::fit_orchestration::{…}` → `pub use gam_models::fit_orchestration::{…}` (same symbol list)
- the `gam::solver::build_analytic_penalty_registry_from_descriptors` re-export → `gam_models::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors`
- (Confirm root `gam` crate Cargo.toml deps gam-models; it should as top facade. If not, ADD it.)
