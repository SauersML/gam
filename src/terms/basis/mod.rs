//! Spline / spatial / spherical basis construction, penalties, and the
//! hyperparameter (`ψ`) derivative machinery that the smooth terms build on.
//!
//! The module is decomposed into single-concern submodules. Shared crate
//! imports live in [`prelude.rs`](prelude.rs) and are pulled into this module's
//! namespace; every submodule re-imports them (together with the sibling
//! re-exports below) through `use super::*`. The `pub use <module>::*`
//! re-exports flatten each concern back onto the `basis::` path so external
//! callers keep referring to e.g. `basis::CenterStrategy` regardless of which
//! submodule now owns the item.

// Crate-wide imports, shared by every submodule via `use super::*`.
include!("prelude.rs");

// ---- Manifold / geometric smooth specifications and kernels ----
mod constant_curvature_smooth;
mod cyclic;
mod sphere_kernels;
mod sphere_spec;
mod sphere_spectral;

// ---- Measure-jet smooth (V0 / V∞) ----
mod measure_jet_moments;
mod measure_jet_anisotropy;
mod measure_jet_predict;
mod measure_jet_smooth;

// ---- Scalar math primitives ----
mod polylog;

// ---- Core types, evaluation, planning, and the (ψ) derivative engine ----
mod bspline_build;
mod bspline_eval;
mod center_selection;
pub mod closed_form_operator;
mod duchon_kernel_math;
mod duchon_psi_derivatives;
mod duchon_thinplate;
mod implicit_psi_derivative;
pub mod input_loc_derivatives;
mod internal;
pub mod matern_gradient;
mod matern_kernel;
mod periodic_duchon;
mod radial_jets_nd;
mod sphere_basis;
pub mod sphere_gpu;
mod spline_eval_scalar;
mod streaming_design;
mod types;
mod workspace_cache;

/// Closed-form scalar building blocks for Riesz, Matérn, and isotropic
/// hybrid Duchon kernels.
///
/// Fourier convention: f̂(ω) = ∫ e^{-iω·x} f(x) dx,
/// f(x) = (2π)^{-d} ∫ e^{iω·x} f̂(ω) dω.
///
/// Riesz kernel:  R_j^d(r) = F^{-1}{|ρ|^{-2j}}(r).
/// Matérn block:  M_ℓ^d(r; κ) = F^{-1}{(|ρ|² + κ²)^{-ℓ}}(r).
pub mod closed_form_penalty;

pub mod radial_profile;

// ---- Flat re-exports: preserve the external `basis::X` path surface ----

pub use constant_curvature_smooth::{
    ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, build_constant_curvature_basis,
    build_constant_curvature_basis_kappa_derivatives, constant_curvature_effective_length,
    constant_curvature_honest_profiled_reml_score, constant_curvature_kappa_fair_sign_score,
    constant_curvature_kernel_kappa_jets, constant_curvature_kernel_matrix,
    realized_constant_curvature_length_scale,
};

pub use measure_jet_moments::{
    MeasureJetJetStats, MeasureJetMomentTable, accumulate_moment_table, jet_sufficient_stats,
    merge_moment_tables, recenter_moment_table,
};

pub use measure_jet_predict::{
    MeasureJetExtrapolationSpectrum, measure_jet_extrapolation_variance,
};

pub use measure_jet_smooth::{
    MeasureJetBand, MeasureJetBasisSpec, MeasureJetEnergyJets, MeasureJetFrozenQuadrature,
    MeasureJetIdentifiability, build_measure_jet_basis, build_measure_jet_basis_psi_derivatives,
    measure_jet_band, measure_jet_center_masses, measure_jet_design_matrix,
    measure_jet_energy_form, measure_jet_energy_form_with_jets, measure_jet_energy_forms_per_scale,
    measure_jet_multiscale_mode, measure_jet_quadrature_nodes, measure_jet_scale_spectrum,
    measure_jet_support_curve, realized_measure_jet_length_scale,
};

pub use measure_jet_anisotropy::{
    MeasureJetAnisotropyJets, LIndex, lower_triangular_indices,
    measure_jet_anisotropy_energy_form, measure_jet_anisotropy_energy_form_with_jets,
};


pub use sphere_spec::{
    SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
};

pub use cyclic::{
    create_closure_difference_penalty_jet, create_cyclic_difference_penalty_matrix,
    create_open_difference_penalty_matrix,
};

pub(crate) use cyclic::{
    create_cyclic_bspline_basis_dense, cyclic_distance_1d, cyclic_uniform_knot_vector,
    wrap_to_period,
};

// Concern modules: flatten each onto `basis::` so external paths are unchanged.
pub use bspline_build::*;
pub use bspline_eval::*;
pub use center_selection::*;
pub use closed_form_operator::ClosedFormPenaltyOperator;
pub use duchon_kernel_math::*;
pub use duchon_psi_derivatives::*;
pub use duchon_thinplate::*;
pub use implicit_psi_derivative::*;
pub use matern_kernel::*;
pub use periodic_duchon::*;
pub use radial_jets_nd::*;
pub use sphere_basis::*;
pub use spline_eval_scalar::*;
pub use streaming_design::*;
pub use types::*;
pub use workspace_cache::*;

#[cfg(test)]
mod tests;
