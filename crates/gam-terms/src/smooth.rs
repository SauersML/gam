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

// The measure-jet term's outer-ψ plumbing, carved out of `term_specs.rs` and
// included immediately after it so every path it exposes is unchanged. It is
// one contract with one invariant — the ψ coordinate ORDER — spread over five
// functions that must agree coordinate-for-coordinate; see its own header.
include!("smooth/measure_jet_psi.rs");

/// The caller's spatial length-scale window and the single expression that
/// projects onto it (gam#2726). Re-exported into `smooth` so the ψ seed
/// constructors in `term_specs.rs` and the outer fit driver share one
/// definition instead of each spelling the projection out for themselves.
mod spatial_length_scale_window;
pub use spatial_length_scale_window::{
    project_spatial_length_scale_into_window, project_spatial_length_scales_in_spec,
    spatial_term_seed_psi,
};

/// gam#2726: the joint `[rho, psi]` seed and the scalar-rho incumbent it is
/// graded against are one point, bit-for-bit, at both projection sites.
#[cfg(test)]
mod spatial_length_scale_projection_2726_tests;

/// Exhaustive coordinate-scale laws for every smooth-basis constructor.
///
/// Kept next to the term specification rather than in an individual basis
/// implementation: wrappers and tensor products are basis trees, so only the
/// specification layer can state (and validate) the complete composed law.
mod scale_contract;
pub use scale_contract::{
    BasisCoordinateScaleAction, BasisDerivativeScaleLaw, BasisDesignScaleLaw,
    BasisNullGeometryScaleLaw, BasisPenaltyScaleLaw, BasisScaleContract, BasisScaleFamily,
    DimensionfulBasisParameter, DimensionfulParameterScale,
};

pub mod structure_analysis;
use self::structure_analysis::smooth_has_frozen_identifiability;
pub use self::structure_analysis::{
    SmoothStructureAnalysis, analyze_smooth_ownership, smooth_term_feature_cols,
};

// The advisories that read the structure `structure_analysis` computes. They
// lived in `gam-cli` with no counterpart on any other surface, so a smooth
// silently residualized against an overlapping linear term was announced to a
// CLI user and to nobody else (#2470, SPEC line 10). Only the aggregator is
// public: the per-shape collectors have no caller outside the module, and an
// unreferenced `pub` item in this workspace reads as live code.
mod structure_warnings;
pub use self::structure_warnings::collect_smooth_structure_warnings;

// Term-collection design construction (#1521), relocated DOWN from gam-models
// `fit_orchestration/drivers/design_construction.rs`. The three re-exports are
// the entry points the staying gam-models drivers still call (via their
// `use gam_terms::smooth::*` glob): `build_term_collection_design` (public API),
// `build_term_collection_design_inner` (the joint-build variants that stay in
// gam-models), and `term_collection_has_anchored_bspline`
// (`spatial_optimization.rs`).
mod term_design;
pub use term_design::{
    CollectionGaugedTerm, LocalTermRealization, RealizedCollectionGauge,
    TermCollectionDerivativeDesign,
    apply_smooth_transform_to_design,
    build_term_collection_derivative_design, build_term_collection_design,
    build_term_collection_design_inner, build_term_collection_design_with_policy,
    orthogonality_relative_residual_for_design, place_term_in_collection_gauge,
    realize_smooth_collection_gauge, smooth_intrinsic_parametric_feature_cols,
    term_collection_has_anchored_bspline, term_collection_has_nonzero_anchor,
};

// Spec→spec freezer relocated DOWN from gam-models `fit_orchestration/drivers/
// spatial_optimization.rs` (#1521). `freeze_term_collection_from_design` is the
// single canonical model-save freezer; its helper `freeze_smooth_basis_from_metadata`
// stays private. A legal gam-terms resident (pure gam-terms/gam-problem types) and
// a shared home the future family sub-crates can call without depending on gam-models.
mod design_freezing;
pub use design_freezing::freeze_term_collection_from_design;

#[cfg(test)]
mod tests;

/// gam#2716: the κ search box is derived over the configuration the
/// constant-curvature basis evaluates (`data ∪ centers` for the chart gauge,
/// `data × centers` for the antipodal fold), not over `data` alone.
#[cfg(test)]
mod constant_curvature_kappa_box_tests;
