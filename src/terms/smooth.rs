// Split from the original oversized module; keep included in order.
include!("smooth/prelude.rs");

mod bspline_boundary;

mod coefficient_transforms;

mod error;

mod input_standardization;

mod shape_constraints;

mod penalty_priors;
pub use penalty_priors::{
    CoefficientGroupSpec, CoefficientSelector, PenaltyBlockGammaPriorMetadata,
    RealizedCoefficientGroups,
};

include!("smooth/term_specs.rs");

mod structure_analysis;
use self::structure_analysis::smooth_has_frozen_identifiability;
pub use self::structure_analysis::{
    SmoothStructureAnalysis, analyze_smooth_ownership, smooth_term_feature_cols,
};

include!("smooth/design_construction.rs");
include!("smooth/spatial_optimization.rs");

#[cfg(test)]
mod tests;
