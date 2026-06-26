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

#[cfg(test)]
mod tests;
