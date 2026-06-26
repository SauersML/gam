//! Shared prelude for the arrow-Schur solver: external imports re-exported
//! crate-wide, the module-level tuning constants, and the matvec function-
//! pointer type aliases. Every sibling concern module pulls these in through
//! `use super::*;`, preserving the single-namespace resolution the previous
//! `include!`-based layout relied on.

pub(crate) use super::reduced_solve::ArrowSchurError;
pub(crate) use super::system::ArrowRowBlock;
pub(crate) use gam_linalg::faer_ndarray::{FaerArrayView, FaerEigh, FaerLlt};
pub(crate) use gam_linalg::triangular::{
    cholesky_solve_matrix, cholesky_solve_vector, forward_substitution_lower_matrix,
};
pub(crate) use gam_terms::analytic_penalties::{
    AnalyticPenaltyKind, AnalyticPenaltyRegistry, PenaltyTier,
};
pub(crate) use gam_terms::latent::{LatentCoordValues, LatentManifold};
pub(crate) use faer::Side;
pub(crate) use gam_runtime::warm_start::Fingerprinter;
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
pub(crate) use std::ops::Range;
pub(crate) use std::sync::Arc;

pub(crate) const DIRECT_SOLVE_MAX_K: usize = 2_000;

pub(crate) const DEFAULT_PCG_MAX_ITERATIONS: usize = 200;

pub(crate) const DEFAULT_PCG_RELATIVE_TOLERANCE: f64 = 1e-4;

/// Absolute floor on the Steihaug-CG residual stopping threshold.
///
/// The native PCG criterion is purely relative: `tol = rel_tol · ‖rhs‖`. When
/// `‖rhs‖` is tiny (degenerate / near-stationary reduced systems) this product
/// can fall below the roundoff resolution of `metric_norm` (~1e-15 for f64),
/// so the loop would "converge" on floating-point noise rather than a genuinely
/// accurate solution. Floor the threshold at 1e-14: above machine epsilon
/// (~2.2e-16) yet below any practical single-iteration residual reduction, so
/// well-scaled problems are unaffected while degenerate ones stop cleanly.
pub(crate) const PCG_ABSOLUTE_TOLERANCE_FLOOR: f64 = 1e-14;

pub(crate) const DEFAULT_TRUST_REGION_RADIUS: f64 = f64::INFINITY;

pub const DEFAULT_PROXIMAL_INITIAL_RIDGE: f64 = 1e-8;

pub(crate) const F32_UNIT_ROUNDOFF: f64 = (f32::EPSILON as f64) * 0.5;

pub(crate) const DEFAULT_MIXED_PRECISION_MAX_REFINEMENTS: usize = 6;

pub(crate) const DEFAULT_MIXED_PRECISION_CERTIFICATE_TOLERANCE: f64 = 1e-11;

pub(crate) const DEFAULT_MIXED_PRECISION_KAPPA_MARGIN: f64 = 0.5;

/// Backward-error certificate floor, expressed as a small multiple of f64 epsilon.
pub(crate) const MIXED_PRECISION_CERTIFICATE_EPSILON_MULTIPLIER: f64 = 64.0;

/// User-supplied kappa margins above this are no stricter than the unit gate.
pub(crate) const MIXED_PRECISION_KAPPA_MARGIN_CEILING: f64 = 1.0;
pub const DEFAULT_PROXIMAL_RIDGE_GROWTH: f64 = 10.0;

/// Number of geometric proximal-ridge escalations the adaptive correction
/// attempts before giving up. Raised from 16 to 22 so the ridge can climb from
/// `1e-8` to `~1e14` (`1e-8 · 10^21`): when the penalised Hessian curvature
/// along the gradient exceeds `~1e9`, the damped Newton step at ridge `1e9`
/// still overshoots, and the extra decades let the step length collapse far
/// enough to either find descent or reach the near-stationary resolution floor
/// that triggers the convergence exit. The cost of the extra attempts is paid
/// only on configs that would otherwise have failed.
pub const DEFAULT_PROXIMAL_MAX_ATTEMPTS: usize = 22;

pub(crate) const DEFAULT_ARMIJO_C1: f64 = 1e-4;

pub(crate) const DEFAULT_GRADIENT_TOLERANCE: f64 = 1e-10;

/// Relative objective resolution for the proximal-correction convergence exit.
///
/// When the best achievable change in the penalised objective across all ridge
/// attempts is within `rel_tol · (|f| + 1)` of the incumbent value, the damped
/// Newton model has reached the floating-point resolution of the objective and
/// no further productive decrease exists. `8e-12` sits a few decades above the
/// `~2.2e-16` f64 epsilon (so genuine reductions of a well-scaled objective are
/// never swallowed) yet comfortably above the accumulated rounding of the
/// `O(N·M·p)` reductions that form the objective, so a truly stationary state
/// is recognised rather than chased into a spurious failure.
pub(crate) const DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL: f64 = 8e-12;

pub(crate) const EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT: u64 = 0;

pub(crate) const ARROW_FACTOR_CACHE_HTBETA_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Matrix-free shared-block multiply for large BA/SAE Schur PCG.
///
/// The closure writes `out = H_ββ x` without the LM ridge. This is the hook
/// that lets SAE-manifold scale callers avoid materializing a dense `K × K`
/// shared block before Agarwal-style inexact Schur PCG.
pub type SharedBetaMatvec =
    Arc<dyn for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;

pub type RowHtbetaMatvec =
    Arc<dyn for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;

/// Row-local matrix-free transpose multiply `out += H_βt^(i) · v` (length `K`).
///
/// This is the adjoint of [`RowHtbetaMatvec`]: it scatters a per-row latent
/// vector `v` (length `d_i`) back into the shared β gradient, **adding** its
/// contribution to `out`. For the SAE Kronecker form this is the sparse
/// `scatter_jbeta_t` over the row's active atoms — `O(m_i · p)` per row, the
/// per-row sparse apply that replaces the `O(K)` column-probe in the GPU and
/// streaming Schur matvec.
pub type RowHtbetaTransposeMatvec =
    Arc<dyn for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;

pub type StreamingArrowRowBuilder =
    Arc<dyn Fn(usize) -> Result<ArrowRowBlock, ArrowSchurError> + Send + Sync>;

/// GPU-backed Schur matvec for CPU-driven PCG at K ≥ 5000.
///
/// The closure writes `out = S·x` where `S = H_ββ + ρ·I − Σ_i Y_i^T Y_i`
/// is the reduced shared system, with `Y_i = L_i^{-1} H_tβ^(i)` pre-computed
/// on device from the same forward kernel that Layer D uses for the dense Schur
/// build. The CPU-driven Steihaug-CG outer loop uploads `x` (K doubles),
/// receives `out` (K doubles), and handles the H_ββ contribution on the CPU side.
///
/// Constructed by `crate::gpu_kernels::arrow_schur::gpu_schur_matvec_backend` when
/// `cuda_selected()` and K ≥ 5000. The closure is `Send + Sync` so PCG callers
/// can hold it in an `Arc`.
pub type GpuSchurMatvec = Arc<dyn Fn(&Array1<f64>, &mut Array1<f64>) + Send + Sync>;

pub(crate) type MetricWeights = [f64];
