// Split from the original oversized module; keep included in order.
include!("imports.rs");

mod constant_curvature_smooth;

mod cyclic;

mod measure_jet_moments;

mod measure_jet_predict;

mod measure_jet_smooth;

mod polylog;

mod sphere_kernels;

mod sphere_spec;

mod sphere_spectral;

include!("core_types_and_bspline_eval.rs");
include!("bspline_build_and_matern_penalty.rs");
include!("sphere_matern_duchon_psi_derivatives.rs");
include!("duchon_thinplate_and_periodic_splines.rs");

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

include!("tests_include.rs");
