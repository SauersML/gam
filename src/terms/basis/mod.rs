// Split from the original oversized module; keep included in order.
include!("split_parts/part_000.rs");


mod constant_curvature_smooth;

mod cyclic;

mod measure_jet_moments;

mod measure_jet_predict;

mod measure_jet_smooth;

mod polylog;

mod sphere_kernels;

mod sphere_spec;

mod sphere_spectral;

include!("split_parts/part_001.rs");
include!("split_parts/part_002.rs");
include!("split_parts/part_003.rs");
include!("split_parts/part_004.rs");


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

include!("split_parts/part_005.rs");
