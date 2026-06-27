// Split from the original oversized module; keep included in order.
include!("smooth/prelude.rs");

mod coefficient_transforms;

mod error;

pub mod input_standardization;

pub mod shape_constraints;

pub mod penalty_priors;
pub use penalty_priors::{
    CoefficientGroupSpec, CoefficientSelector, PenaltyBlockGammaPriorMetadata,
    RealizedCoefficientGroups,
};

include!("smooth/term_specs.rs");

pub mod structure_analysis;
use self::structure_analysis::smooth_has_frozen_identifiability;
pub use self::structure_analysis::{
    SmoothStructureAnalysis, analyze_smooth_ownership, smooth_term_feature_cols,
};

// Term-collection design construction (#1521), relocated DOWN from gam-models
// `fit_orchestration/drivers/design_construction.rs`. The three re-exports are
// the entry points the staying gam-models drivers still call (via their
// `use gam_terms::smooth::*` glob): `build_term_collection_design` (public API),
// `build_term_collection_design_inner` (the joint-build variants that stay in
// gam-models), and `term_collection_has_one_sided_anchored_bspline`
// (`spatial_optimization.rs`).
mod term_design;
pub use term_design::{
    build_term_collection_design, build_term_collection_design_inner,
    term_collection_has_one_sided_anchored_bspline,
};

#[cfg(test)]
mod tests;
