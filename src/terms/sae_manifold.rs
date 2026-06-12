//! SAE-manifold term configuration.
//!
//! This is the formal Methodspace row for the SAE-manifold term:
//!
//! ```text
//! Z_i ~= sum_k a_ik g_k(t_ik),     g_k(t) = Phi_k(t) B_k
//! ```
//!
//! Tier assignment:
//!
//! * beta: [`SaeManifoldAtom::decoder_coefficients`] (`B_k`, one block per atom).
//! * ext-coords: [`SaeAssignment`] (`logits -> a_ik` and per-atom
//!   `LatentCoordValues`). Softmax uses the identifiable reference-logit chart
//!   with `K - 1` free assignment coordinates (`0` for `K = 1`). Per-row latent coordinates are written `t`; existing
//!   kernel-shape state remains with carriers such as `SpatialLogKappaCoords`.
//! * rho: [`SaeManifoldRho`] (`lambda_sparse`, `lambda_smooth`, `alpha_kj`) plus
//!   the discrete `K` selected by the Python `compare_models` wrapper.
//!
//! The per-row local block is exactly the audit-revised shape:
//!
//! ```text
//! ext_i = (assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])
//! dim(ext_i) = assignment_dim + sum_k d_k
//! ```
//!
//! [`SaeManifoldTerm::assemble_arrow_schur`] materializes the Gauss-Newton
//! bordered Hessian in that layout and hands it to
//! [`crate::solver::arrow_schur::ArrowSchurSystem`].

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};
use std::sync::Arc;

const SAE_BYTES_PER_F64: usize = 8;
const SAE_HOST_IN_CORE_FALLBACK_BYTES: usize = 2 * 1024 * 1024 * 1024;
const SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR: usize = 3;
const SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR: usize = 5;
const SAE_CPU_L2_CACHE_BYTES: usize = 1024 * 1024;
const SAE_CHUNK_CACHE_MULTIPLE: usize = 8;
const SAE_MIN_STREAMING_CHUNK_ROWS: usize = 256;
const SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER: usize = 32;

use crate::solver::arrow_schur::{
    ArrowProximalCorrectionOptions, ArrowRowBlock, ArrowSchurError, ArrowSchurSystem,
    ArrowSolveOptions, BetaPenaltyOp, CompositePenaltyOp, DensePenaltyOp, DeviceSaePcgData,
    DeviceSaeSmoothBlock, FactoredFrameGBlock, FactoredFrameKroneckerOp,
    IdentityRightKroneckerPenaltyOp, SparseBlockKroneckerPenaltyOp, SparseGBlock,
    StreamingArrowSchur, solve_arrow_newton_step_with_proximal_correction,
    solve_streaming_reduced_beta, solve_with_lm_escalation_inner,
};
use crate::terms::analytic_penalties::{
    AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, DecoderIncoherencePenalty,
    IbpHessianDiagThirdChannels, IsometryPenalty, MechanismSparsityPenalty, NuclearNormPenalty,
    PenaltyTier, PsiSlice, WeightField, resolve_learnable_weight,
};
use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode, LatentManifold};
use crate::terms::sae_criterion_atoms::SaeCriterion;
use crate::terms::sae_optimality_certificate::{
    CriterionCertificate, DirectionalSamples, certificate_from_samples,
    deterministic_probe_direction, probe_step,
};

use crate::linalg::faer_ndarray::{
    FaerCholesky, FaerCholeskyFactor, FaerEigh, FaerSvd, fast_ab, fast_abt, fast_atb,
};
use crate::linalg::triangular::cholesky_solve_vector;
use crate::solver::arrow_schur::{
    ArrowFactorCache, ArrowRowGaugeDeflation, arrow_factor_max_pivot, arrow_factor_min_pivot,
    solve_arrow_newton_step_with_options,
};
use crate::solver::estimate::EstimationError;
use crate::solver::evidence::arrow_log_det_from_cache;
use crate::solver::outer_strategy::{
    DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterCapability, OuterEval,
    OuterEvalOrder, OuterObjective, SeedOutcome,
};
use crate::solver::structure_search::{CollapseAction, CollapseEvent};
use faer::Side;

const SAE_MANIFOLD_ARMIJO_C1: f64 = 1.0e-4;
const SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS: usize = 12;
/// Relative Cholesky-pivot floor for the analytic SAE outer-rho gradient.
///
/// The evidence value can still be honest below this threshold because it only
/// sums `log(diag(L))`. The analytic gradient is different: selected-inverse
/// traces and `ArrowFactorCache::full_inverse_apply` divide by those pivots.
/// Once `min_pivot / max_pivot` is below this floor, the gradient lane must
/// either identify a closed-form gauge orbit and stiffen only that quotient
/// direction, or reject the trial rho as numerically singular.
const SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR: f64 = 1.0e-12;
const SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR: f64 = 1.0e-8;

/// Nominal curvature-homotopy `η` step (#1007): the tracker covers `η ∈ [0, 1]`
/// in this many equal predictor-corrector waypoints when the branch is clean.
/// Five waypoints is a few corrector solves — far cheaper than the multi-seed
/// cascade it replaces — and the step is halved adaptively when the arrow-factor
/// min pivot shrinks, so a near-bifurcation stretch is resolved at finer
/// granularity without a separate knob.
const CURVATURE_WALK_INITIAL_ETA_STEP: f64 = 0.2;
/// Smallest curvature-homotopy `η` step (#1007). A pivot collapse (or corrector
/// failure) that persists at this step is a DETECTED branch bifurcation, not a
/// step-size artifact: the walk records it and defers to the seed cascade.
const CURVATURE_WALK_MIN_ETA_STEP: f64 = 1.0 / 256.0;
/// Hard ceiling on accepted corrector solves in one curvature-homotopy walk
/// (#1007). Bounds the walk's cost under repeated halving; reaching it is a
/// structural-termination signal (the branch is not cleanly trackable) that
/// defers to the cascade, never a spin.
const CURVATURE_WALK_MAX_CORRECTORS: usize = 32;

/// Relative floor on the Newton directional decrease, expressed as a tiny
/// multiple of `‖g‖·‖Δ‖`. A predicted decrease below this is at the level of
/// f64 round-off in the quadratic model and is treated as no progress (the step
/// is rejected). Scaling by the gradient/step norms makes the floor invariant
/// to the problem's overall magnitude.
const SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR: f64 = 1.0e-14;

/// Relative tolerance on the undamped Newton step norm (scaled by the iterate
/// scale) for accepting inner-solve convergence.
const SAE_MANIFOLD_INNER_STEP_REL_TOL: f64 = 1.0e-4;

/// Relative tolerance on the KKT gradient norm (scaled by the iterate scale) for
/// accepting inner-solve convergence.
const SAE_MANIFOLD_INNER_GRAD_REL_TOL: f64 = 1.0e-5;

/// Above this full-`B` β width, dense beta-penalty curvature is never
/// materialized when Grassmann frames are engaged; exact curvature is probed
/// directly in the factored coordinate space instead.
const SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM: usize = 4096;

/// Relative spectral cutoff for counting the numerical rank / nullity of a
/// symmetric penalty Gram: eigenvalues at or below `cutoff · λ_max` are treated
/// as zero. Shared by [`SaeManifoldTerm::symmetric_rank`] and
/// [`smooth_penalty_nullity`] so the two stay in lockstep.
const SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF: f64 = 1.0e-9;

/// Floor on the Levenberg-Marquardt ridge added to a per-row Hessian before
/// Cholesky, so the first attempt is always strictly positive even when the
/// caller passes a zero base ridge.
const SAE_MANIFOLD_ROW_RIDGE_FLOOR: f64 = 1.0e-12;

/// Multiplicative factor by which the LM ridge is escalated after a failed
/// Cholesky factorisation of a per-row Hessian.
const SAE_MANIFOLD_ROW_RIDGE_GROWTH: f64 = 10.0;

/// Maximum number of LM ridge-escalation attempts before declaring the per-row
/// Hessian unfactorable.
const SAE_MANIFOLD_ROW_RIDGE_MAX_ATTEMPTS: usize = 12;

#[derive(Clone, Copy, Debug, Default)]
struct SaeBetaPenaltyAssembly {
    dense_written: bool,
    deferred_factored: bool,
}

impl SaeBetaPenaltyAssembly {
    fn record_curvature(&mut self, dense_beta_curvature: bool) {
        if dense_beta_curvature {
            self.dense_written = true;
        } else {
            self.deferred_factored = true;
        }
    }
}

/// Final fitted-data explained-variance floor for the reconstruction-collapse
/// guard (#1023). This is deliberately an effectively-zero threshold: ordinary
/// under-fitting is a model-quality issue, but returning a K>=1 active SAE whose
/// fitted matrix is indistinguishable from the column mean is a structural
/// collapse and must enter the #976 CollapseEvent ledger.
const SAE_FIT_DATA_COLLAPSE_EV_FLOOR: f64 = 0.10;
const SAE_FIT_DATA_COLLAPSE_COST: f64 = 1.0e12;
const SAE_PRISTINE_SEED_EV_RETAIN_FLOOR: f64 = 0.95;
const SAE_FINAL_EV_DEGRADATION_TOL: f64 = 1.0e-3;
const SAE_SEED_DISPERSION_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeStreamingPlan {
    pub streaming: bool,
    pub chunk_size: usize,
    pub estimated_full_batch_bytes: usize,
    pub estimated_dense_schur_bytes: usize,
    pub estimated_row_cross_bytes: usize,
    pub estimated_direct_peak_bytes: usize,
    pub estimated_matrix_free_peak_bytes: usize,
    pub in_core_budget_bytes: usize,
    pub host_available_bytes: usize,
    pub direct_admitted: bool,
    pub matrix_free_admitted: bool,
}

fn sae_streaming_plan_from_budget(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
    in_core_budget_bytes: usize,
    chunk_window_bytes: usize,
    host_available_bytes: usize,
) -> SaeStreamingPlan {
    let per_row_words = total_basis
        .saturating_mul(1 + d_max)
        .saturating_add(k_atoms)
        .max(1);
    let per_row_bytes = per_row_words.saturating_mul(SAE_BYTES_PER_F64);
    let full_batch_bytes = n_obs.saturating_mul(per_row_bytes);
    let dense_schur_bytes = border_dim
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let row_block_dim = k_atoms.saturating_mul(1usize.saturating_add(d_max));
    let row_cross_bytes = n_obs
        .saturating_mul(row_block_dim)
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let direct_peak_bytes = full_batch_bytes
        .saturating_add(row_cross_bytes)
        .saturating_add(dense_schur_bytes);
    let matrix_free_peak_bytes = chunk_window_bytes
        .min(full_batch_bytes.max(per_row_bytes))
        .saturating_add(row_cross_bytes)
        .saturating_add(
            border_dim
                .saturating_mul(SAE_BYTES_PER_F64)
                .saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER),
        );
    let direct_admitted = direct_peak_bytes <= in_core_budget_bytes;
    let matrix_free_admitted = matrix_free_peak_bytes <= in_core_budget_bytes;
    let rows_per_chunk = (chunk_window_bytes / per_row_bytes).max(SAE_MIN_STREAMING_CHUNK_ROWS);
    SaeStreamingPlan {
        streaming: !direct_admitted,
        chunk_size: if direct_admitted {
            n_obs.max(1)
        } else {
            rows_per_chunk.min(n_obs).max(1)
        },
        estimated_full_batch_bytes: full_batch_bytes,
        estimated_dense_schur_bytes: dense_schur_bytes,
        estimated_row_cross_bytes: row_cross_bytes,
        estimated_direct_peak_bytes: direct_peak_bytes,
        estimated_matrix_free_peak_bytes: matrix_free_peak_bytes,
        in_core_budget_bytes,
        host_available_bytes,
        direct_admitted,
        matrix_free_admitted,
    }
}

pub fn sae_streaming_plan_for_shape(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
) -> SaeStreamingPlan {
    let (budget, chunk_window, host_available) = match crate::gpu::runtime::GpuRuntime::global() {
        Some(rt) => {
            let aggregate_budget: usize = rt
                .device_ordinals()
                .iter()
                .map(|&ord| rt.memory_budget_for(ord))
                .sum();
            let per_device_budget = aggregate_budget / rt.device_count().max(1);
            let window =
                (per_device_budget / 16).max(SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE);
            let host_available = sae_host_available_memory_bytes();
            (
                (aggregate_budget / 4).min(host_available),
                window,
                host_available,
            )
        }
        None => {
            let (budget, host_available) = sae_host_in_core_budget_bytes();
            (
                budget,
                SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                host_available,
            )
        }
    };
    sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    )
}

impl SaeStreamingPlan {
    fn admitted_or_error(self, n: usize, p: usize, k_atoms: usize) -> Result<Self, String> {
        if self.direct_admitted || self.matrix_free_admitted {
            Ok(self)
        } else {
            Err(format!(
                "SaeManifoldTerm::streaming_plan: predicted working set {} bytes exceeds budget {} bytes; shape n={n},p={p},K={k_atoms}",
                self.estimated_matrix_free_peak_bytes, self.in_core_budget_bytes
            ))
        }
    }

    fn solve_options_for_border_dim(self, border_dim: usize) -> ArrowSolveOptions {
        if self.direct_admitted {
            ArrowSolveOptions::automatic(border_dim)
        } else {
            ArrowSolveOptions::inexact_pcg()
        }
    }

    fn direct_logdet_admitted(self) -> bool {
        self.direct_admitted
    }
}

fn sae_host_available_memory_bytes() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory() as usize;
    if available == 0 {
        SAE_HOST_IN_CORE_FALLBACK_BYTES
    } else {
        available
    }
}

fn sae_host_in_core_budget_bytes() -> (usize, usize) {
    let available = sae_host_available_memory_bytes();
    let fraction = available.saturating_mul(SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR)
        / SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR;
    (fraction.max(SAE_HOST_IN_CORE_FALLBACK_BYTES), available)
}

/// Decay law for deterministic Gumbel/concrete assignment temperature.
#[derive(Debug, Clone)]
pub enum ScheduleKind {
    Geometric { rate: f64 },
    Linear { steps: usize },
    ReciprocalIter,
}

/// Outer-state temperature annealing for SAE assignment relaxations.
///
/// Annealing drives the continuous concrete/softmax assignment toward the
/// discrete argmax or IBP active-set solution while PIRLS solves smooth
/// positive-temperature subproblems. In the zero-floor limit, softmax becomes
/// argmax and the IBP-MAP sigmoid active set becomes exact; a positive
/// `tau_min` optimizes the corresponding near-discrete MAP problem.
#[derive(Debug, Clone)]
pub struct GumbelTemperatureSchedule {
    pub tau_start: f64,
    pub tau_min: f64,
    pub decay: ScheduleKind,
    pub iter_count: usize,
}

impl GumbelTemperatureSchedule {
    #[must_use = "build error must be handled"]
    pub fn new(tau_start: f64, tau_min: f64, decay: ScheduleKind) -> Result<Self, String> {
        let sched = Self {
            tau_start,
            tau_min,
            decay,
            iter_count: 0,
        };
        sched.validate()?;
        Ok(sched)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.tau_start.is_finite() && self.tau_start > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_start must be finite and positive; got {}",
                self.tau_start
            ));
        }
        if !(self.tau_min.is_finite() && self.tau_min > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min must be finite and positive; got {}",
                self.tau_min
            ));
        }
        if self.tau_min > self.tau_start {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min ({}) cannot exceed tau_start ({})",
                self.tau_min, self.tau_start
            ));
        }
        match self.decay {
            ScheduleKind::Geometric { rate } => {
                if !(rate.is_finite() && rate > 0.0 && rate < 1.0) {
                    return Err(format!(
                        "GumbelTemperatureSchedule::Geometric: rate must be in (0, 1); got {rate}"
                    ));
                }
            }
            ScheduleKind::Linear { steps } => {
                if steps == 0 {
                    return Err("GumbelTemperatureSchedule::Linear: steps must be positive".into());
                }
            }
            ScheduleKind::ReciprocalIter => {}
        }
        Ok(())
    }

    pub fn current_tau(&self, iter: usize) -> f64 {
        let raw = match self.decay {
            ScheduleKind::Geometric { rate } => self.tau_start * rate.powf(iter as f64),
            ScheduleKind::Linear { steps } => {
                if iter >= steps {
                    self.tau_min
                } else {
                    let frac = iter as f64 / steps as f64;
                    self.tau_start + frac * (self.tau_min - self.tau_start)
                }
            }
            ScheduleKind::ReciprocalIter => self.tau_start / (1.0 + iter as f64),
        };
        raw.max(self.tau_min)
    }

    pub fn step(&mut self) -> f64 {
        let tau = self.current_tau(self.iter_count);
        self.iter_count += 1;
        tau
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    Fixed,
    ExponentialSweep { values: Vec<f64> },
}

impl SearchStrategy {
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }

    #[must_use]
    pub fn sweep_values(&self) -> Option<&[f64]> {
        match self {
            Self::Fixed => None,
            Self::ExponentialSweep { values } => Some(values),
        }
    }
}

/// Basis/topology tag for one SAE manifold atom.
///
/// The evaluated basis and input-location jet live on [`SaeManifoldAtom`].
/// This enum records the user-facing topology choice so downstream diagnostics
/// and Python wrappers can round-trip whether the atom was a Duchon patch,
/// periodic curve, sphere, or a caller-supplied precomputed basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaeAtomBasisKind {
    Duchon,
    Periodic,
    Sphere,
    Torus,
    EuclideanPatch,
    Precomputed(String),
}

impl SaeAtomBasisKind {
    fn latent_manifold(&self, latent_dim: usize) -> LatentManifold {
        match self {
            // `Periodic` uses [`PeriodicHarmonicEvaluator`], whose basis
            // functions are `cos(2π·h·t), sin(2π·h·t)` — i.e. `t` is a
            // fraction of one period, not radians. The latent manifold
            // wraps modulo `period = 1.0` to match this convention.
            // Wrapping modulo `2π` instead would scramble the
            // fraction-of-period interpretation and cause #174-style
            // failures where Newton updates push `t` outside `[0, 1)` and
            // the optimiser sees a discontinuous landscape.
            Self::Periodic => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            // `Sphere` is parameterised via a (lat, lon) intrinsic chart; the
            // chart evaluator already enforces sphere geometry through its
            // cos/sin terms (in radians, multiplying lat/lon directly into
            // `sin`/`cos`), so the latent optimiser sees a 2-D product
            // manifold: lat is a bounded interval `[-π/2, π/2]` (enforced here
            // by the `Interval` retraction — its clamp + active-bound tangent
            // projection — NOT by truncating the chart jet) and lon is an `S^1`
            // angle wrapped modulo `2π`.
            // Treating it as `LatentManifold::Sphere { dim: 2 }` would
            // require ambient unit-vectors of length 2 (impossible for S^2).
            Self::Sphere => LatentManifold::Product(vec![
                LatentManifold::Interval {
                    lo: -std::f64::consts::FRAC_PI_2,
                    hi: std::f64::consts::FRAC_PI_2,
                },
                LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                },
            ]),
            // `Torus` uses [`TorusHarmonicEvaluator`], which shares the
            // fraction-of-period convention with `PeriodicHarmonicEvaluator`
            // (basis is `cos(2π·h·t)`, `sin(2π·h·t)` on each axis). Each
            // per-axis latent wraps modulo `1.0`.
            Self::Torus => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            Self::Duchon | Self::EuclideanPatch | Self::Precomputed(_) => LatentManifold::Euclidean,
        }
    }

    /// Dense candidate coordinates spanning compact latents for fixed-decoder
    /// out-of-sample projection. Unbounded/basis-linear latents return `None`
    /// because their PCA seed already lies in the convex training hull.
    fn projection_seed_grid(&self, latent_dim: usize, resolution: usize) -> Option<Array2<f64>> {
        match self {
            Self::Periodic => torus_projection_seed_grid(latent_dim, resolution),
            Self::Sphere if latent_dim == 2 => sphere_projection_seed_grid(resolution),
            Self::Sphere => None,
            Self::Torus => torus_projection_seed_grid(latent_dim, resolution),
            Self::Duchon | Self::EuclideanPatch | Self::Precomputed(_) => None,
        }
    }
}

fn sphere_projection_seed_grid(resolution: usize) -> Option<Array2<f64>> {
    use std::f64::consts::PI;
    let r = resolution.max(2);
    let mut grid = Array2::<f64>::zeros((r * r, 2));
    for i in 0..r {
        let lat = -PI / 2.0 + PI * (i as f64 + 0.5) / r as f64;
        for j in 0..r {
            let lon = -PI + 2.0 * PI * (j as f64) / r as f64;
            grid[[i * r + j, 0]] = lat;
            grid[[i * r + j, 1]] = lon;
        }
    }
    Some(grid)
}

fn torus_projection_seed_grid(latent_dim: usize, resolution: usize) -> Option<Array2<f64>> {
    if latent_dim == 0 || latent_dim >= usize::BITS as usize {
        return None;
    }
    const MAX_GRID_POINTS: usize = 4096;
    let min_points = 1usize << latent_dim;
    if min_points > MAX_GRID_POINTS {
        return None;
    }
    let requested = resolution.max(2);
    let mut per_axis = requested;
    while per_axis.saturating_pow(latent_dim as u32) > MAX_GRID_POINTS {
        per_axis -= 1;
        if per_axis < 2 {
            return None;
        }
    }
    let total: usize = (0..latent_dim).fold(1usize, |acc, _| acc.saturating_mul(per_axis));
    let mut grid = Array2::<f64>::zeros((total, latent_dim));
    let mut idx = vec![0usize; latent_dim];
    for flat in 0..total {
        for axis in 0..latent_dim {
            grid[[flat, axis]] = idx[axis] as f64 / per_axis as f64;
        }
        for axis in (0..latent_dim).rev() {
            idx[axis] += 1;
            if idx[axis] < per_axis {
                break;
            }
            idx[axis] = 0;
        }
    }
    Some(grid)
}

/// Per-axis ARD coordinate prior, evaluated as a smooth energy in the latent
/// coordinate `t` with precision `alpha = exp(log_ard)`.
///
/// On a *Euclidean* axis the prior is the usual Gaussian negative-log density
/// `½·α·t²`, with gradient `α·t` and curvature `α`.
///
/// On a *periodic* axis (a `Circle` factor of period `P`) the Euclidean `½α t²`
/// is geometrically ill-posed (it depends on the arbitrary choice of origin /
/// branch cut, so a Newton step crossing the cut makes the loss jump by
/// `½α P²` and breaks Armijo descent). We replace it with the von-Mises energy
///
/// ```text
///   V(t) = (α / κ²) · (1 − cos(κ t)),   κ = 2π / P
/// ```
///
/// which is the period-`P` periodic function whose Taylor expansion at the
/// origin is `½ α t² + O(t⁴)` — so it carries the *same* precision `α`
/// (curvature at the origin) as the Gaussian, matching the ARD interpretation,
/// but is globally smooth and continuous across the cut (`cos(κ·P)=cos 2π=1`).
/// Its derivatives are
///
/// ```text
///   V'(t)  = (α / κ) · sin(κ t)
///   V''(t) = α · cos(κ t)
/// ```
///
/// The value, gradient, and curvature returned here all come from this single
/// energy, so they are mutually FD-consistent. The *value* (`ard_value` /
/// `loss.ard`) and the *gradient* (the assembled `gt`) use the exact `V` and
/// `V'`. The curvature `V'' = α·cos(κt)` is INDEFINITE — it turns negative for
/// `|κt|` past `π/2` (a quarter period) — so it is NOT written raw into the
/// Newton/Schur `H_tt` diagonal: that would make the per-row coordinate block
/// indefinite and the Schur (and log-det) Cholesky would fail on a non-PD pivot
/// at `K ≥ 2`. The assembly accumulates the PSD majorizer `max(V'', 0)` into
/// `H_tt` instead (mirroring `add_sae_coord_penalty`'s `psd_majorizer_diag` for
/// the registry coord penalties). Majorizing the curvature of a *fixed* prior
/// only damps the Newton step; the stationary point is set by the exact gradient
/// `V'`, so it is unchanged. The Laplace `½ log|H|` is therefore evaluated on the
/// same PSD-majorized `H_tt` (a valid Cholesky requires a PD operator anyway).
///
/// `sq_equiv` is the Euclidean-equivalent `t²` such that `½·α·sq_equiv == V`,
/// i.e. `sq_equiv = 2V/α = (2/κ²)(1−cos κt)`. It is what the
/// Mackay/Fellner–Schall `α ← n / (Σ sq_equiv + tr H⁻¹)` fixed point must use so
/// that the prior energy it implies stays consistent with `ard_value`.
#[derive(Clone, Copy, Debug)]
struct ArdAxisPrior {
    value: f64,
    grad: f64,
    hess: f64,
    sq_equiv: f64,
}

impl ArdAxisPrior {
    /// Evaluate the per-axis prior at coordinate `t` with precision `alpha`.
    /// `period == None` selects the Euclidean Gaussian; `Some(p)` selects the
    /// von-Mises periodic energy with period `p`.
    fn eval(alpha: f64, t: f64, period: Option<f64>) -> Self {
        match period {
            None => Self {
                value: 0.5 * alpha * t * t,
                grad: alpha * t,
                hess: alpha,
                sq_equiv: t * t,
            },
            Some(p) => {
                let kappa = std::f64::consts::TAU / p;
                let (sin, cos) = (kappa * t).sin_cos();
                let one_minus_cos = 1.0 - cos;
                Self {
                    value: (alpha / (kappa * kappa)) * one_minus_cos,
                    grad: (alpha / kappa) * sin,
                    hess: alpha * cos,
                    sq_equiv: (2.0 / (kappa * kappa)) * one_minus_cos,
                }
            }
        }
    }
}

/// Modified Bessel function of the first kind, order zero, `I0(x)`.
///
/// Abramowitz & Stegun 9.8.1 (|x| <= 3.75) and 9.8.2 (|x| > 3.75) polynomial
/// approximations; relative error < 1.6e-7 / 1.9e-7 respectively, which is far
/// below the precision tolerance the ARD normaliser is read at. `I0` is even,
/// so only `|x|` enters. Used for the exact von-Mises precision log-partition.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.5156229
                + t2 * (3.0899424
                    + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.39894228
            + y * (0.01328592
                + y * (0.00225319
                    + y * (-0.00157565
                        + y * (0.00916281
                            + y * (-0.02057706
                                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        (ax.exp() / ax.sqrt()) * poly
    }
}

/// Modified Bessel function of the first kind, order one, `I1(x)`.
///
/// Uses the Abramowitz & Stegun approximations paired with [`bessel_i0`]. This is
/// needed only for the derivative of the periodic ARD precision normalizer
/// `log I0(η)`, whose derivative is `I1(η) / I0(η)`.
fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let value = if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        ax * (0.5
            + t2 * (0.87890594
                + t2 * (0.51498869
                    + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.39894228
            + y * (-0.03988024
                + y * (-0.00362018
                    + y * (0.00163801
                        + y * (-0.01031555
                            + y * (0.02282967
                                + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059)))))));
        (ax.exp() / ax.sqrt()) * poly
    };
    if x < 0.0 { -value } else { value }
}

pub use crate::terms::sae::assignment::*;
pub use crate::terms::sae::basis::*;
pub use crate::terms::sae::frames::*;
/// One manifold atom.
///
/// `basis_values` is `Phi_k(t_{ik})`, shape `(N, M_k)`.
/// `basis_jacobian` is `d Phi_k / d t_{ik}`, shape `(N, M_k, d_k)`.
/// `decoder_coefficients` is `B_k`, shape `(M_k, p)`.
/// `smooth_penalty` is `P_k`, shape `(M_k, M_k)`.
#[derive(Debug, Clone)]
pub struct SaeManifoldAtom {
    pub name: String,
    pub basis_kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub basis_values: Array2<f64>,
    pub basis_jacobian: Array3<f64>,
    pub decoder_coefficients: Array2<f64>,
    /// Effective (intrinsic) roughness Gram `S̃_k` that every consumer reads
    /// (smoothness value, gradient, Kronecker Hessian op, REML rank/log-det).
    ///
    /// `S̃_k` is the raw coefficient-space Gram [`Self::smooth_penalty_raw`]
    /// reparameterized by the decoder pullback metric so the roughness — and
    /// hence the topology evidence — is gauge-invariant under reparameterization
    /// of the latent coordinate `t` (issue #673). It is recomputed from the
    /// current basis Jacobian and decoder coefficients by
    /// [`Self::refresh_intrinsic_smooth_penalty`] (lagged-diffusivity: the
    /// metric weight is frozen within each inner Newton/evidence assembly and
    /// refreshed between them, so at convergence the penalty is the true
    /// arc-length roughness). The metric weight is centered (geometric mean 1),
    /// so for constant-speed atoms (the periodic sin/cos basis on `S¹`) every
    /// weight is exactly `1` and `S̃_k = S_k` — periodic atoms are untouched
    /// and no overall magnitude leaks into the penalty.
    pub smooth_penalty: Array2<f64>,
    /// Canonical raw roughness Gram `S_k` in raw coefficient/`t` space (the
    /// finite-/cyclic-difference Reinsch Gram or the Duchon RKHS Gram). Never
    /// mutated after construction; [`Self::smooth_penalty`] is derived from it
    /// each assembly via the pullback-metric reweighting.
    pub smooth_penalty_raw: Array2<f64>,
    /// Roughness operator order `r` of [`Self::smooth_penalty_raw`], recovered
    /// once at construction as its null-space dimension (an order-`r`
    /// difference / Duchon penalty annihilates the degree-`<r` polynomials, so
    /// `nullity(S) = r`). Sets the arc-length reweighting exponent
    /// `β = ½ − r` (`β = −3/2` for the standard second-derivative penalty):
    /// the metric-speed power that converts raw-`t` roughness into intrinsic
    /// arc-length roughness. `0` when the raw Gram is empty/zero (no
    /// reweighting).
    pub smooth_penalty_order: usize,
    pub basis_evaluator: Option<Arc<dyn SaeBasisEvaluator>>,
    /// Same evaluator upcast to `dyn SaeBasisSecondJet` when the
    /// implementation provides a closed-form Hessian. `None` for
    /// evaluators that only implement the base [`SaeBasisEvaluator`]
    /// trait. Installed via [`Self::with_basis_second_jet`]; the base
    /// [`Self::with_basis_evaluator`] populates only the supertrait
    /// slot. Used by [`refresh_isometry_caches_from_atom`] to install
    /// the `H` cache on isometry penalties when the second jet is
    /// analytically available.
    pub basis_second_jet: Option<Arc<dyn SaeBasisSecondJet>>,
    /// Profiled low-rank Grassmann decoder frame `U_k` (`p × r`), issue #972.
    ///
    /// `None` ⇒ the historical full-`B` path: the border carries the entire
    /// `M_k · p` decoder block and is bit-for-bit unchanged. `Some(frame)` ⇒ the
    /// decoder factors as `B_k = C_k · Uᵀ` with the `M_k · r` coordinate matrix
    /// `C_k = B_k · U` in the border and the frame `U` profiled out by streaming
    /// polar steps. [`Self::decoder_coefficients`] stays the authoritative
    /// reconstructed `B_k` (so every existing consumer is unchanged); the frame
    /// is the *representation* that shrinks the border and contributes the
    /// `r·(p − r)` Grassmann dimensions to the Laplace evidence normalizer.
    /// Activated automatically by [`Self::maybe_activate_decoder_frame`] when the
    /// decoder's effective column rank is materially below `p`; never a flag.
    pub decoder_frame: Option<GrassmannFrame>,
    /// Curvature-homotopy dial `η ∈ [0, 1]` (#1007). [`Self::refresh_basis`]
    /// scales every *curved* basis column (per
    /// [`SaeBasisEvaluator::phi_eta_split`]) by `η`, leaving the *linear*
    /// columns untouched, so `η = 0` is the Eckart-Young linear relaxation (a
    /// convex decoder problem whose global optimum [`linear_span_anchor`]
    /// certifies) and `η = 1` is the full curved basis. The certified tracker
    /// walks `η` from `0 → 1`; every other caller sees the default `1.0`, which
    /// makes [`Self::refresh_basis`] bit-for-bit identical to the un-dialed
    /// `evaluate` path (`evaluate_phi_eta` at `η = 1` returns the unscaled
    /// basis). Caller-managed atoms (no installed evaluator) ignore the dial —
    /// there is no curved/linear split without an evaluator to provide it.
    pub homotopy_eta: f64,
    /// #1019: `true` once the post-fit chart canonicalization has been
    /// applied to this atom — the latent chart is then the canonical
    /// representative of its `Diff(M)` orbit (the arc-length / unit-speed
    /// chart for `d = 1`, the minimum-isometry-defect flow chart for `d = 2`
    /// torus atoms) and the residual chart freedom is the finite isometry
    /// group of the reference manifold (rotation + reflection on `S¹`,
    /// reflection + translation on the interval, `Isom(T², flat)` on the
    /// torus). Read by the residual-gauge lowering so the certificate reports
    /// the downgrade with the `PinnedByCanonicalization` provenance. Only
    /// ever set for `latent_dim == 1` atoms and `latent_dim == 2` torus
    /// atoms; never a flag the user controls.
    pub chart_canonicalized: bool,
}

impl SaeManifoldAtom {
    #[must_use = "build error must be handled"]
    pub fn new(
        name: impl Into<String>,
        basis_kind: SaeAtomBasisKind,
        latent_dim: usize,
        basis_values: Array2<f64>,
        basis_jacobian: Array3<f64>,
        decoder_coefficients: Array2<f64>,
        smooth_penalty: Array2<f64>,
    ) -> Result<Self, String> {
        let n = basis_values.nrows();
        let m = basis_values.ncols();
        let p = decoder_coefficients.ncols();
        if basis_jacobian.dim() != (n, m, latent_dim) {
            return Err(format!(
                "SaeManifoldAtom::new: basis_jacobian must be ({n}, {m}, {latent_dim}); got {:?}",
                basis_jacobian.dim()
            ));
        }
        if decoder_coefficients.nrows() != m {
            return Err(format!(
                "SaeManifoldAtom::new: decoder rows {} must equal basis size {m}",
                decoder_coefficients.nrows()
            ));
        }
        if smooth_penalty.dim() != (m, m) {
            return Err(format!(
                "SaeManifoldAtom::new: smooth penalty must be ({m}, {m}); got {:?}",
                smooth_penalty.dim()
            ));
        }
        if p == 0 {
            return Err("SaeManifoldAtom::new: decoder output dimension must be positive".into());
        }
        // Recover the roughness operator order `r` from the raw Gram's
        // null-space dimension (`nullity(S) = r` for an order-`r` difference /
        // Duchon penalty). This pins the arc-length reweighting exponent
        // `β = ½ − r` once, so the per-assembly reweighting needs no
        // eigendecomposition in the hot loop.
        let smooth_penalty_order = smooth_penalty_nullity(&smooth_penalty)?;
        let mut atom = Self {
            name: name.into(),
            basis_kind,
            latent_dim,
            basis_values,
            decoder_coefficients,
            smooth_penalty_raw: smooth_penalty.clone(),
            smooth_penalty,
            smooth_penalty_order,
            basis_jacobian,
            basis_evaluator: None,
            basis_second_jet: None,
            decoder_frame: None,
            homotopy_eta: 1.0,
            chart_canonicalized: false,
        };
        // Seed `smooth_penalty` with the intrinsic Gram at the initial
        // decoder/coordinates so the very first assembly already reads the
        // pullback-metric-reweighted penalty.
        atom.refresh_intrinsic_smooth_penalty();
        Ok(atom)
    }

    pub fn with_basis_evaluator(mut self, evaluator: Arc<dyn SaeBasisEvaluator>) -> Self {
        self.basis_evaluator = Some(evaluator);
        self.basis_second_jet = None;
        self
    }

    /// Install an evaluator that additionally exposes a closed-form
    /// second jet. Populates both the base [`SaeBasisEvaluator`] slot
    /// (used by [`Self::refresh_basis`] and the standard evaluate path)
    /// and the [`SaeBasisSecondJet`] slot (consumed by
    /// [`refresh_isometry_caches_from_atom`] for the `H` cache).
    pub fn with_basis_second_jet(mut self, evaluator: Arc<dyn SaeBasisSecondJet>) -> Self {
        let base: Arc<dyn SaeBasisEvaluator> = evaluator.clone();
        self.basis_evaluator = Some(base);
        self.basis_second_jet = Some(evaluator);
        self
    }

    pub fn refresh_basis(&mut self, coords: ArrayView2<'_, f64>) -> Result<(), String> {
        // No installed evaluator means the caller is managing the basis
        // out-of-band (the construction-time `phi` / `jet` are authoritative).
        // The contract for that mode is documented in the constructor: the
        // caller takes responsibility for rebuilding the term after a
        // coordinate change. We must NOT fail here, because driver entry
        // points (`run_joint_fit_arrow_schur`, the inner Newton loop, …)
        // unconditionally call `refresh_basis_from_current_coords` to keep
        // the auto-refresh path correct, and that prelude has to pass through
        // unchanged for caller-managed atoms.
        let Some(evaluator) = self.basis_evaluator.as_ref() else {
            return Ok(());
        };
        // Curvature-homotopy dial (#1007): at the default `η = 1` this is the
        // un-dialed basis (`evaluate_phi_eta` returns the unscaled Φ / jet
        // bit-for-bit), so the production path is unchanged. For `η < 1` the
        // tracker scales the curved columns toward the linear relaxation; the
        // `dphi_deta` / `djet_deta` channels are discarded here (the predictor
        // forms `∂g/∂η` separately from a dedicated evaluation).
        let (phi, jet) = if self.homotopy_eta == 1.0 {
            evaluator.evaluate(coords)?
        } else {
            let evaluated = evaluator.evaluate_phi_eta(coords, self.homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        if phi.dim() != self.basis_values.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned Phi {:?}, expected {:?}",
                phi.dim(),
                self.basis_values.dim()
            ));
        }
        if jet.dim() != self.basis_jacobian.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned jet {:?}, expected {:?}",
                jet.dim(),
                self.basis_jacobian.dim()
            ));
        }
        self.basis_values = phi;
        self.basis_jacobian = jet;
        Ok(())
    }

    pub fn n_obs(&self) -> usize {
        self.basis_values.nrows()
    }

    pub fn basis_size(&self) -> usize {
        self.basis_values.ncols()
    }

    pub fn output_dim(&self) -> usize {
        self.decoder_coefficients.ncols()
    }

    /// Effective profiled frame rank `r` of this atom's decoder block in the
    /// arrow-Schur border (issue #972). `r == p` (full output dim) when no
    /// Grassmann frame is active — the historical full-`B` border width. When a
    /// frame is active the border holds only `M_k · r` coordinates.
    pub fn border_frame_rank(&self) -> usize {
        match &self.decoder_frame {
            Some(frame) => frame.rank(),
            None => self.output_dim(),
        }
    }

    /// Per-atom arrow-Schur border coefficient count: `M_k · r` when a frame is
    /// active (the factored width), else the full `M_k · p` (issue #972).
    pub fn border_coeff_count(&self) -> usize {
        self.basis_size() * self.border_frame_rank()
    }

    /// Grassmann manifold dimension `r·(p − r)` profiled OUT of the border for
    /// this atom (issue #972). `0` when no frame is active. This is the number
    /// of frame degrees of freedom that must enter the Laplace evidence
    /// dimension accounting (evidence honesty).
    pub fn frame_manifold_dimension(&self) -> usize {
        match &self.decoder_frame {
            Some(frame) => frame.manifold_dimension(),
            None => 0,
        }
    }

    /// Effective numerical column rank of the decoder `B_k` (`M_k × p`) from its
    /// singular values, with the relative cutoff [`SAE_FRAME_RANK_CUTOFF`]. This
    /// is the smallest frame rank `r` that captures `B_k`'s span up to that
    /// energy floor; the auto-activation heuristic compares it against `p`.
    pub fn decoder_numerical_rank(&self) -> Result<usize, String> {
        let p = self.output_dim();
        if p == 0 || self.basis_size() == 0 {
            return Ok(0);
        }
        let (_u, sv, _vt) = self
            .decoder_coefficients
            .svd(false, false)
            .map_err(|e| format!("SaeManifoldAtom::decoder_numerical_rank: SVD failed: {e}"))?;
        let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
        if !(max_sv > 0.0) {
            // A zero decoder has rank 0 but still needs a rank-1 frame so the
            // border carries a non-degenerate coordinate column.
            return Ok(0);
        }
        let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
        Ok(sv.iter().filter(|&&v| v > tol).count())
    }

    /// Rank that should be carried by the low-rank Grassmann decoder frame for
    /// the current decoder, or `None` when the full-`B` representation is still
    /// the intended path. This is the exact activation predicate:
    ///
    /// * `r = max(numerical_rank(B_k), 1)`;
    /// * `r <= p * (1 - SAE_FRAME_ACTIVATION_MARGIN)`;
    /// * `p - r > 0`.
    ///
    /// Because `rank(B_k) <= M_k`, a cold LSQ decoder with `p >= 896` and
    /// `M_k <= 16` always satisfies the shrink predicate (`16 << 0.75p`) unless
    /// the decoder has no output dimension or no basis columns.
    pub fn decoder_frame_activation_rank(&self) -> Result<Option<usize>, String> {
        let p = self.output_dim();
        if p == 0 || self.basis_size() == 0 {
            return Ok(None);
        }
        if p < SAE_FRAME_MIN_AUTO_OUTPUT_DIM {
            return Ok(None);
        }
        let numerical_rank = self.decoder_numerical_rank()?;
        // A degenerate all-zero decoder keeps a rank-1 frame so the coordinate
        // column is non-empty; otherwise use the numerical rank.
        let r = numerical_rank.max(1).min(p);
        // Beneficial only if the frame materially shrinks the border AND there
        // is a positive Grassmann dimension to profile out.
        let shrink_ok = (r as f64) <= (p as f64) * (1.0 - SAE_FRAME_ACTIVATION_MARGIN);
        if !shrink_ok || p.saturating_sub(r) == 0 {
            return Ok(None);
        }
        Ok(Some(r))
    }

    /// Auto-derive whether the low-rank Grassmann factorization is beneficial for
    /// this atom and, if so, activate it (issue #972) — magic-by-default, no
    /// flag. The frame is installed (decoder factored as `B_k = C_k Uᵀ`) only
    /// when the decoder's effective rank `r` shrinks the per-atom border
    /// `M_k · p → M_k · r` by at least [`SAE_FRAME_ACTIVATION_MARGIN`] AND leaves
    /// a positive Grassmann dimension (`p − r ≥ 1`). Otherwise the atom stays on
    /// the bit-for-bit full-`B` path (`decoder_frame == None`).
    ///
    /// `B_k` is unchanged numerically: the installed frame spans exactly
    /// `range(B_kᵀ)` (the column space of the decoder) up to the truncation
    /// floor, so [`Self::reconstruct_decoder_coefficients`] recovers `B_k` to
    /// machine precision when `r` equals the true rank. Returns the activated
    /// frame rank, or `None` if the full-`B` path was kept.
    pub fn maybe_activate_decoder_frame(&mut self) -> Result<Option<usize>, String> {
        let Some(r) = self.decoder_frame_activation_rank()? else {
            self.decoder_frame = None;
            return Ok(None);
        };
        let p = self.output_dim();
        // Build the canonical frame from the decoder's own column-span evidence:
        // the cross-moment `B_kᵀ B_k`-induced left subspace is exactly the top-`r`
        // right-singular subspace of `B_k`. We obtain it by polaring the rank-`r`
        // truncation of the column cross-moment `B_kᵀ · (B_k · Vr)` — equivalently
        // the top-`r` right singular vectors of `B_k`. Use the SVD of `B_k`
        // directly: `B_k = W Σ Vᵀ` (W: M×?, Vᵀ: ?×p) ⇒ frame = top-`r` rows of `Vᵀ`
        // transposed = top-`r` columns of `V` (`p × r`).
        let (_w, sv, vt_opt) = self.decoder_coefficients.svd(false, true).map_err(|e| {
            format!("SaeManifoldAtom::maybe_activate_decoder_frame: SVD failed: {e}")
        })?;
        let vt = vt_opt.ok_or_else(|| {
            "SaeManifoldAtom::maybe_activate_decoder_frame: SVD returned no right factor"
                .to_string()
        })?;
        // `vt` is `min(M,p) × p`; take its top-`r` rows as the frame columns.
        let available = vt.nrows();
        let r_eff = r.min(available);
        if r_eff == 0 || p.saturating_sub(r_eff) == 0 {
            self.decoder_frame = None;
            return Ok(None);
        }
        let mut frame = Array2::<f64>::zeros((p, r_eff));
        for col in 0..r_eff {
            for row in 0..p {
                frame[[row, col]] = vt[[col, row]];
            }
        }
        let mut gauge = Array1::<f64>::zeros(r_eff);
        for i in 0..r_eff {
            gauge[i] = sv.get(i).copied().unwrap_or(0.0);
        }
        self.decoder_frame = Some(GrassmannFrame::from_oriented(frame, gauge));
        // Project the decoder onto the activated frame so the authoritative
        // `B_k = C_k U_kᵀ` holds EXACTLY from the first factored assembly
        // (issue #972 / #977 T1). Without this, `B_k` keeps its off-frame
        // component while the factored C-block solve only moves within
        // `range(U_k)`, leaving an irreducible residual the solver cannot
        // reduce — the fit then never converges. `B ← (B U) Uᵀ` is a no-op in
        // span for a truly rank-`r` decoder (the common, beneficial case).
        let u_proj = self
            .decoder_frame
            .as_ref()
            .expect("frame just set")
            .frame()
            .to_owned();
        let c_proj = self.decoder_coefficients.dot(&u_proj);
        self.decoder_coefficients = c_proj.dot(&u_proj.t());
        Ok(Some(r_eff))
    }

    /// Deactivate the Grassmann frame, returning this atom to the full-`B`
    /// border path (issue #972). `decoder_coefficients` already holds the
    /// reconstructed `B_k`, so no numerical change occurs.
    pub fn deactivate_decoder_frame(&mut self) {
        self.decoder_frame = None;
    }

    /// Coordinate matrix `C_k = B_k · U` (`M_k × r`) that the border stores when
    /// a frame is active (issue #972). Returns `None` on the full-`B` path.
    pub fn factored_coordinates(&self) -> Result<Option<Array2<f64>>, String> {
        match &self.decoder_frame {
            Some(frame) => Ok(Some(
                frame.project_decoder(self.decoder_coefficients.view())?,
            )),
            None => Ok(None),
        }
    }

    /// Reconstruct the full decoder `B_k = C_k · Uᵀ` from a border coordinate
    /// matrix `C_k` (`M_k × r`) and the active frame (issue #972). Used when the
    /// border solver returns updated coordinates and the authoritative
    /// `decoder_coefficients` must be refreshed for the full-`B` consumers.
    pub fn reconstruct_decoder_coefficients(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let frame = self.decoder_frame.as_ref().ok_or_else(|| {
            "SaeManifoldAtom::reconstruct_decoder_coefficients: no active frame".to_string()
        })?;
        frame.reconstruct_decoder(coords)
    }

    /// Install border coordinates `C_k` (`M_k × r`) returned by the factored
    /// border solve, refreshing `decoder_coefficients = C_k · Uᵀ` so all
    /// full-`B` consumers stay consistent with the profiled frame (issue #972).
    pub fn set_factored_coordinates(&mut self, coords: ArrayView2<'_, f64>) -> Result<(), String> {
        let reconstructed = self.reconstruct_decoder_coefficients(coords)?;
        if reconstructed.dim() != self.decoder_coefficients.dim() {
            return Err(format!(
                "SaeManifoldAtom::set_factored_coordinates: reconstructed decoder {:?} \
                 must match {:?}",
                reconstructed.dim(),
                self.decoder_coefficients.dim()
            ));
        }
        self.decoder_coefficients = reconstructed;
        Ok(())
    }

    /// Closed-form streaming polar refresh of the active frame from an
    /// accumulated `p × r` cross-moment (issue #972): `U ← polar(Mcm)`, then
    /// re-project the coordinates so `B_k` is unchanged in span. The frame
    /// update happens OUTSIDE the border; the coordinate matrix is re-derived by
    /// projection onto the new frame. No-op (error) when no frame is active.
    pub fn refresh_frame_from_cross_moment(
        &mut self,
        cross_moment: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        if self.decoder_frame.is_none() {
            return Err("SaeManifoldAtom::refresh_frame_from_cross_moment: no active frame".into());
        }
        let new_frame = GrassmannFrame::polar_update(cross_moment)?;
        if new_frame.output_dim() != self.output_dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_frame_from_cross_moment: frame output dim {} \
                 must equal decoder output dim {}",
                new_frame.output_dim(),
                self.output_dim()
            ));
        }
        // Re-express the current decoder in the new frame's coordinates, then
        // reconstruct `B_k` so its in-span component is carried forward exactly
        // and the out-of-span residual (orthogonal to the refreshed span) is
        // dropped — the streaming-polar fixed point.
        let coords = new_frame.project_decoder(self.decoder_coefficients.view())?;
        self.decoder_coefficients = new_frame.reconstruct_decoder(coords.view())?;
        self.decoder_frame = Some(new_frame);
        Ok(())
    }

    /// `g_k(t_{ik}) = Phi_k(t_{ik}) B_k`.
    pub fn decoded_row(&self, row: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_row(row, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `g_k(t_{ik})` into a caller-supplied buffer of length `p`.
    /// Hot-loop variant used by the arrow-Schur assembly to avoid per-row
    /// allocations.
    pub fn fill_decoded_row(&self, row: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let phi = self.basis_values[[row, basis_col]];
            if phi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += phi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    /// `d g_k(t_{ik}) / d t_{ik,j}` for one row and latent axis.
    pub fn decoded_derivative_row(&self, row: usize, latent_axis: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_derivative_row(row, latent_axis, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `d g_k / d t_{ik,axis}` into a caller-supplied buffer of
    /// length `p`. Hot-loop variant used by the arrow-Schur assembly.
    pub fn fill_decoded_derivative_row(&self, row: usize, latent_axis: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let dphi = self.basis_jacobian[[row, basis_col, latent_axis]];
            if dphi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                out[out_col] += dphi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    /// Recompute the intrinsic (arc-length) roughness Gram
    /// [`Self::smooth_penalty`] from [`Self::smooth_penalty_raw`], the current
    /// basis Jacobian, and the current decoder coefficients (issue #673).
    ///
    /// The raw penalty `0.5·λ·tr(BᵀS B)` measures roughness per unit of the raw
    /// latent coordinate `t`, so it is *not* invariant under reparameterizing
    /// `t` — and the model evidence that ranks an atom's topology (circle vs
    /// line) inherits that gauge dependence. The decoder curve is
    /// `g(t) = Φ(t) B` and its pulled-back metric is the scalar squared speed
    /// `m(t) = ‖g'(t)‖² = ‖J(t)‖²` with `J(t) = Φ'(t) B` (the decoder
    /// Jacobian, [`Self::fill_decoded_derivative_row`]). The arc-length
    /// roughness of an order-`r` operator reweights the raw-`t` derivative
    /// energy density by `m^{½−r}` (`= m^{−3/2}` for the standard
    /// second-derivative penalty), which removes the gauge dependence.
    ///
    /// Realised as a per-coefficient symmetric congruence
    /// `S̃ = W^{½} S W^{½}`, `W = diag(w_μ)`, `w_μ = m̄_μ^{β}`, `β = ½ − r`,
    /// where `m̄_μ` is the basis-activation-weighted average squared speed
    /// localised to coefficient `μ`,
    /// `m̄_μ = (Σ_n Φ_μ(t_n)² m_n) / Σ_n Φ_μ(t_n)²`, `m_n = ‖J(t_n)‖²`. The
    /// congruence keeps `S̃` symmetric PSD with the same rank as `S` (Sylvester
    /// inertia), so the Kronecker Hessian `S̃ ⊗ I_p` and the REML
    /// `rank(S)`-Occam term are structurally unchanged; only the metric-aware
    /// log-det / quadratic value move, which is exactly the gauge correction.
    ///
    /// The metric weight is frozen at the current `B` (lagged-diffusivity /
    /// IRLS surrogate): within one inner solve the penalty stays a quadratic
    /// Gram form, and refreshing `W` between assemblies makes the *converged*
    /// penalty the true arc-length roughness. The per-coefficient weight is
    /// centered (its geometric mean is 1), so constant-speed atoms (the
    /// periodic sin/cos basis, `m̄_μ ≡ c`) get `w_μ ≡ 1` and hence `S̃ = S`
    /// exactly — periodic atoms are unaffected and no overall magnitude (which
    /// `λ` already owns) leaks into the penalty.
    ///
    /// Conservative scope: the scalar-speed reweighting is the genuine
    /// arc-length normalisation only for a 1-D latent (the circle-vs-line case
    /// the issue is about). For `latent_dim != 1`, or a degenerate (empty/zero)
    /// raw Gram, `S̃ = S` is left untouched.
    pub fn refresh_intrinsic_smooth_penalty(&mut self) {
        let m = self.basis_size();
        // No reweighting when there is no penalty operator order to invert into
        // arc length, or for higher-dim latents where the metric is a matrix
        // (det(g) volume reweighting is deferred — see scope note above).
        if m == 0 || self.smooth_penalty_order == 0 || self.latent_dim != 1 {
            self.smooth_penalty.assign(&self.smooth_penalty_raw);
            return;
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let beta = 0.5 - self.smooth_penalty_order as f64;

        // Per-sample squared speed m_n = ‖J(t_n)‖², J(t_n) = Φ'(t_n) B (axis 0,
        // the single latent axis), and the basis-activation accumulators
        // act_μ = Σ_n Φ_μ(t_n)² and num_μ = Σ_n Φ_μ(t_n)² m_n.
        let mut act = vec![0.0_f64; m];
        let mut num = vec![0.0_f64; m];
        let mut deriv = vec![0.0_f64; p];
        for row in 0..n {
            self.fill_decoded_derivative_row(row, 0, &mut deriv);
            let mut speed_sq = 0.0_f64;
            for &d in deriv.iter() {
                speed_sq += d * d;
            }
            for col in 0..m {
                let phi = self.basis_values[[row, col]];
                let w = phi * phi;
                if w == 0.0 {
                    continue;
                }
                act[col] += w;
                num[col] += w * speed_sq;
            }
        }

        // Representative squared speed per coefficient, and the geometric-mean
        // center of the finite positive speeds. Only finite positive speeds
        // enter the center so a degenerate (inf/NaN) sample cannot corrupt it.
        let mut speeds = vec![0.0_f64; m];
        let mut log_acc = 0.0_f64;
        let mut log_cnt = 0usize;
        for col in 0..m {
            let s = if act[col] > 0.0 {
                num[col] / act[col]
            } else {
                0.0
            };
            speeds[col] = s;
            if s > 0.0 && s.is_finite() {
                log_acc += s.ln();
                log_cnt += 1;
            }
        }
        let center = if log_cnt > 0 {
            (log_acc / log_cnt as f64).exp()
        } else {
            0.0
        };
        // Degenerate curve (no finite positive speed anywhere, or a non-finite
        // center): the pullback metric carries no usable scale, so leave the
        // penalty at its raw Gram — exactly `S̃ = S_raw`, matching the
        // constant-speed limit with no spurious magnitude inflation.
        if !(center > 0.0 && center.is_finite()) {
            self.smooth_penalty.assign(&self.smooth_penalty_raw);
            return;
        }

        // Reweight relative to the center so the congruence is a *scale-free*
        // shape reweighting: the geometric mean of `w_μ` is 1, so a
        // constant-speed atom (every `s_μ = center`) gives `w_μ ≡ 1` and hence
        // `S̃ = S_raw` exactly — periodic atoms are untouched and no overall
        // magnitude (which `λ` already owns) leaks in. The relative floor keeps
        // a vanishing-speed coefficient at a small fraction of the typical
        // speed rather than a singular negative power, and clamps any non-finite
        // ratio back to a finite weight.
        const RELATIVE_SPEED_FLOOR: f64 = 1.0e-6;
        const RELATIVE_SPEED_CEIL: f64 = 1.0e6;
        let mut root_w = vec![0.0_f64; m];
        for col in 0..m {
            // Normalised squared speed (ratio to the geometric-mean center),
            // clamped to `[1e-6, 1e6]` so a vanishing-/diverging-speed
            // coefficient is treated as a bounded fraction/multiple of the
            // typical speed rather than a singular negative power, and any
            // non-finite ratio (e.g. an overflowed speed) maps to the ceiling.
            // The symmetric clamp keeps every weight finite and centered near 1
            // so the REML numerical-rank eigencutoff cannot drift.
            let ratio = speeds[col] / center;
            let ratio = if ratio.is_finite() {
                ratio.clamp(RELATIVE_SPEED_FLOOR, RELATIVE_SPEED_CEIL)
            } else {
                RELATIVE_SPEED_CEIL
            };
            // w_μ = ratio^β; the congruence uses W^{½}, so store ratio^{β/2}.
            root_w[col] = ratio.powf(0.5 * beta);
        }

        // S̃ = W^{½} S_raw W^{½}: scale row i and column j by root_w.
        for i in 0..m {
            let ri = root_w[i];
            for j in 0..m {
                self.smooth_penalty[[i, j]] = ri * self.smooth_penalty_raw[[i, j]] * root_w[j];
            }
        }
    }
}

/// Null-space dimension of the symmetric PSD roughness Gram `S` — the order
/// `r` of the difference / Duchon penalty it encodes (`nullity(S) = r`, since
/// the operator annihilates exactly the degree-`<r` polynomials). Used once at
/// atom construction to fix the arc-length reweighting exponent `β = ½ − r`.
///
/// Numerical null space: eigenvalues at or below `1e-9 · max_eig` (the same
/// conventional relative spectral cutoff [`SaeManifoldTerm::symmetric_rank`]
/// uses for `S`'s rank).
fn smooth_penalty_nullity(s: &Array2<f64>) -> Result<usize, String> {
    let m = s.ncols();
    if m == 0 {
        return Ok(0);
    }
    let mut sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
        }
    }
    let (evals, _evecs) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("smooth_penalty_nullity: eigh failed: {e}"))?;
    let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if !(max_eig > 0.0) {
        // A zero (or negative-semidefinite) Gram carries no roughness; report a
        // zero operator order so the reweighting is skipped.
        return Ok(0);
    }
    let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_eig;
    Ok(evals.iter().filter(|&&v| v <= tol).count())
}

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or JumpReLU gated L1, or the
    /// learnable `log(alpha)` offset for IBP-MAP assignment.
    pub log_lambda_sparse: f64,
    /// `log(lambda_smooth)` shared by the per-atom decoder penalties.
    pub log_lambda_smooth: f64,
    /// Per-atom, per-axis `log(alpha_kj)` ARD strengths. An empty per-atom
    /// block disables native coordinate ARD for that atom.
    pub log_ard: Vec<Array1<f64>>,
}

impl SaeManifoldRho {
    #[must_use]
    pub fn new(log_lambda_sparse: f64, log_lambda_smooth: f64, log_ard: Vec<Array1<f64>>) -> Self {
        Self {
            log_lambda_sparse,
            log_lambda_smooth,
            log_ard,
        }
    }

    /// Shift every scale-coupled penalty seed by the profiled reconstruction
    /// dispersion scale. SAE's Gaussian data-fit term is in squared output
    /// units, while `lambda_sparse`, `lambda_smooth`, and ARD precisions are
    /// absolute penalty weights; adding `log(phi_seed)` makes the seeded
    /// effective stiffness `lambda / phi_seed` dimensionless.
    pub fn seed_scaled_by_dispersion(&self, dispersion: f64) -> Result<Self, String> {
        self.seed_scaled_by_dispersion_with_sparse_policy(dispersion, true)
    }

    /// Assignment-aware seed scaling. In learnable-alpha IBP mode the sparse
    /// coordinate is a dimensionless log-alpha offset, not a penalty strength, so
    /// response-dispersion scaling must skip it while still scaling smoothness and
    /// ARD precision seeds.
    pub fn seed_scaled_by_dispersion_for_assignment(
        &self,
        dispersion: f64,
        assignment_mode: AssignmentMode,
    ) -> Result<Self, String> {
        let scale_sparse = !matches!(
            assignment_mode,
            AssignmentMode::IBPMap {
                learnable_alpha: true,
                ..
            }
        );
        self.seed_scaled_by_dispersion_with_sparse_policy(dispersion, scale_sparse)
    }

    fn seed_scaled_by_dispersion_with_sparse_policy(
        &self,
        dispersion: f64,
        scale_sparse: bool,
    ) -> Result<Self, String> {
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "SaeManifoldRho::seed_scaled_by_dispersion: dispersion must be finite and \
                 positive; got {dispersion}"
            ));
        }
        let shift = dispersion.ln();
        let mut scaled = self.clone();
        if scale_sparse {
            scaled.log_lambda_sparse += shift;
        }
        scaled.log_lambda_smooth += shift;
        for atom in &mut scaled.log_ard {
            for value in atom.iter_mut() {
                *value += shift;
            }
        }
        Ok(scaled)
    }

    pub fn lambda_sparse(&self) -> f64 {
        // Clamp the log-strength into the finite-normal band before
        // exponentiating: a raw `exp(log_lambda)` overflows to `inf` for
        // `log_lambda ≳ 709`, and `inf · 0.0` / `inf / inf` then injects NaN
        // into the penalty value/grad/Hessian and poisons the solve.
        Self::stable_exp_strength(self.log_lambda_sparse)
    }

    pub fn lambda_smooth(&self) -> f64 {
        Self::stable_exp_strength(self.log_lambda_smooth)
    }

    /// Exponentiate a learnable log-strength with the exponent clamped into the
    /// finite-normal band, so the resulting strength is always a finite,
    /// strictly-positive `f64` (no overflow to `inf`, no underflow to `0.0`).
    pub(crate) fn stable_exp_strength(log_strength: f64) -> f64 {
        const MAX_LOG_STRENGTH: f64 = 700.0;
        const MIN_LOG_STRENGTH: f64 = -700.0;
        log_strength.clamp(MIN_LOG_STRENGTH, MAX_LOG_STRENGTH).exp()
    }

    /// Flatten ρ into the contiguous outer-coordinate vector the generic
    /// `OuterObjective` engine optimises over.
    ///
    /// Layout: `[log_lambda_sparse, log_lambda_smooth, <ARD>]`, where enabled
    /// ARD blocks concatenate each atom `k`'s per-axis `log_ard[k][j]` in atom
    /// order, axis `j` in `0..d_k`. Empty per-atom blocks contribute no outer
    /// coordinates. [`Self::from_flat`] is the exact inverse and reads this
    /// fixed per-atom layout from `self`.
    pub fn to_flat(&self) -> Array1<f64> {
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        let mut out = Array1::<f64>::zeros(2 + ard_len);
        out[0] = self.log_lambda_sparse;
        out[1] = self.log_lambda_smooth;
        let mut cursor = 2usize;
        for axis in &self.log_ard {
            for &v in axis.iter() {
                out[cursor] = v;
                cursor += 1;
            }
        }
        out
    }

    /// Rebuild a ρ with this ρ's per-atom ARD dimensions from a flat
    /// outer-coordinate vector produced by [`Self::to_flat`].
    ///
    /// The per-atom dims are taken from `&self` (the ARD layout is a fixed
    /// property of the term shape; the engine only moves the values). The
    /// flat vector must have length `2 + Σ_k len(log_ard[k])`.
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> SaeManifoldRho {
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        assert_eq!(
            flat.len(),
            2 + ard_len,
            "SaeManifoldRho::from_flat: flat length {} != 2 + Σ d_k = {}",
            flat.len(),
            2 + ard_len
        );
        let mut log_ard = Vec::with_capacity(self.log_ard.len());
        let mut cursor = 2usize;
        for axis in &self.log_ard {
            let d = axis.len();
            let mut block = Array1::<f64>::zeros(d);
            for (j, slot) in block.iter_mut().enumerate() {
                *slot = flat[cursor + j];
            }
            cursor += d;
            log_ard.push(block);
        }
        SaeManifoldRho {
            log_lambda_sparse: flat[0],
            log_lambda_smooth: flat[1],
            log_ard,
        }
    }
}

/// Kronecker-factored per-row beta Jacobian primitive for SAE-manifold.
///
/// The per-row beta Jacobian has exact Kronecker form
///
/// ```text
/// J_{β,i} = φ_i^T ⊗ I_p
/// ```
///
/// where `φ_i ∈ ℝ^{m_i}` (active per-row atom·basis scalar weights, the
/// `a_k * phi` products in the assembly loop) and `p` is the decoder output
/// dimension.  The four trait methods implement the four operations that the
/// Arrow-Schur solver needs without ever forming the dense `(q × K·p)` block:
///
/// * `apply_jbeta`:   `u = J_β x`   (gather along active support)
/// * `scatter_jbeta_t`: `y += J_βᵀ u`  (scatter)
/// * `apply_l`:       `w = L u`     (q × p Jacobian apply)
/// * `apply_l_t`:     `u += Lᵀ v`   (q × p Jacobian transpose-accumulate)
///
/// The inner Schur row contribution
///
/// ```text
/// S_i = J_{β,i}^T (I - L_i^T A_i^{-1} L_i) J_{β,i}
/// ```
///
/// is applied in `O(m_i p + q p + q²)` per row per PCG iteration using
/// the five-step sequence:
///
/// ```text
/// u_p        = Σ_s φ_i[s] * x_β[s, :]    // gather (apply_jbeta)
/// w_q        = L_i u_p                    // q × p apply (apply_l)
/// v_q        = A_i^{-1} w_q               // existing per-row factor
/// u_p       -= L_i^T v_q                  // q × p apply-t (apply_l_t)
/// y_β[s, :] += φ_i[s] * u_p              // scatter (scatter_jbeta_t)
/// ```
pub trait SaeKroneckerRow {
    /// `u_out[j] = Σ_s φ_i[s] * x_beta[s * p + j]` for `j in 0..p`.
    ///
    /// Gather step: projects the full `K·p` beta vector down to the `p`-dimensional
    /// decoded output space using the active per-row support weights.
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]);

    /// `y_beta[s * p + j] += φ_i[s] * u[j]` for each active `(s, j)`.
    ///
    /// Scatter step: distributes the `p`-dimensional residual back into the
    /// full `K·p` beta gradient using the active per-row support weights.
    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]);

    /// `w_out[c] = Σ_j L[c, j] * u[j]` — apply the `q × p` local Jacobian.
    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]);

    /// `u_out[j] += Σ_c L[c, j] * v[c]` — accumulate `Lᵀ v` into `u_out`.
    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]);
}

/// Per-row Kronecker data for the SAE-manifold beta Jacobian.
///
/// Each row `i` stores:
/// * `a_phi_row`: sparse support — `(beta_base_idx, scalar_weight)` pairs,
///   one entry per active `(atom, basis_col)` combination.
/// * `local_jac_row`: the `(q × p)` assignment + coordinate Jacobian `L_i`
///   (same matrix written into `block.htt` via `local_jac` in the assembler).
///
/// Together these implement `J_β = φᵀ ⊗ I_p` without materializing the dense
/// `(q × K·p)` block.  Storage is `O(m_i · q · p)` per row rather than
/// `O(q · K · p)`.
#[derive(Debug, Clone)]
pub struct SaeKroneckerRows {
    /// Decoder output dimension `p`.
    p: usize,
    /// Per-row sparse support: `a_phi[i]` is a `Vec<(beta_base_idx, weight)>`.
    a_phi: Vec<Vec<(usize, f64)>>,
    /// Per-row local Jacobian `L_i`, shape `(q_i × p)` flattened row-major.
    ///
    /// Element `(c, j)` is at `local_jac[i][c * p + j]`.
    /// For heterogeneous (active-set) systems, each row may have a different
    /// `q_i = local_jac[i].len() / p`.
    local_jac: Vec<Vec<f64>>,
}

impl SaeKroneckerRows {
    /// Build from per-row data collected during `assemble_arrow_schur`. The
    /// row count is implicit in `a_phi.len()` and `local_jac.len()`; the
    /// constructor asserts they agree so callers cannot pass mismatched rows.
    pub fn new(p: usize, a_phi: Vec<Vec<(usize, f64)>>, local_jac: Vec<Vec<f64>>) -> Self {
        assert_eq!(
            a_phi.len(),
            local_jac.len(),
            "SaeKroneckerRows: a_phi rows ({}) != local_jac rows ({})",
            a_phi.len(),
            local_jac.len(),
        );
        Self {
            p,
            a_phi,
            local_jac,
        }
    }
}

impl SaeKroneckerRow for SaeKroneckerRows {
    fn apply_jbeta(&self, row: usize, x_beta: &[f64], u_out: &mut [f64]) {
        for val in u_out.iter_mut() {
            *val = 0.0;
        }
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += phi * x_beta[beta_base + j];
            }
        }
    }

    fn scatter_jbeta_t(&self, row: usize, u: &[f64], y_beta: &mut [f64]) {
        for &(beta_base, phi) in &self.a_phi[row] {
            if phi == 0.0 {
                continue;
            }
            for j in 0..self.p {
                y_beta[beta_base + j] += phi * u[j];
            }
        }
    }

    fn apply_l(&self, row: usize, u: &[f64], w_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        // Per-row q_i = jac.len() / p (supports heterogeneous active-set layouts).
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let mut acc = 0.0_f64;
            for j in 0..self.p {
                acc += jac[c * self.p + j] * u[j];
            }
            w_out[c] = acc;
        }
    }

    fn apply_l_t(&self, row: usize, v: &[f64], u_out: &mut [f64]) {
        let jac = &self.local_jac[row];
        let q_i = jac.len() / self.p;
        for c in 0..q_i {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for j in 0..self.p {
                u_out[j] += jac[c * self.p + j] * vc;
            }
        }
    }
}

/// Loss breakdown for diagnostics and evidence ranking.
#[derive(Debug, Clone, Copy)]
pub struct SaeManifoldLoss {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub evidence_gauge_deflated_directions: usize,
}

impl SaeManifoldLoss {
    pub const fn total(&self) -> f64 {
        self.data_fit + self.assignment_sparsity + self.smoothness + self.ard
    }

    /// Laplace/REML wrappers rank larger evidence higher. This local score is
    /// the negative penalized objective, used when a full `RemlState` is not
    /// driving the term yet.
    pub const fn evidence_proxy(&self) -> f64 {
        -self.total()
    }
}

/// Componentized analytic derivative of the SAE REML criterion with respect to
/// the flat [`SaeManifoldRho`] layout.
///
/// This is intentionally only a value object for tests and derivation gates. It
/// is not wired into [`SaeManifoldOuterObjective`] capability planning until the
/// third-order logdet correction is available behind its own oracle.
#[derive(Debug, Clone)]
pub struct SaeOuterRhoGradientComponents {
    /// Direct derivative of `loss.total() + extra_penalty_energy` with respect to
    /// log-strength coordinates, excluding the Hessian logdet and Occam terms.
    pub explicit: Array1<f64>,
    /// `0.5 * tr(H^{-1} dH/d rho_j)` for the currently available penalty blocks.
    pub logdet_trace: Array1<f64>,
    /// Derivative contribution of `-occam`.
    pub occam: Array1<f64>,
    /// Reserved channel for `0.5 * tr(H^{-1} (dH/dtheta * dtheta_hat/d rho_j))`.
    pub third_order_correction: Array1<f64>,
    /// Whether `third_order_correction` is populated from analytic channels.
    pub third_order_correction_available: bool,
}

impl SaeOuterRhoGradientComponents {
    #[must_use]
    pub fn gradient_excluding_unavailable_correction(&self) -> Array1<f64> {
        &(&self.explicit + &self.logdet_trace) + &self.occam
    }

    #[must_use]
    pub fn gradient_with_available_correction(&self) -> Array1<f64> {
        // The name is a contract: callers asking for the corrected gradient
        // must not silently receive the uncorrected one. Zeros-by-omission in
        // the correction channel are exactly the objective↔gradient desync
        // class; fail loudly instead.
        assert!(
            self.third_order_correction_available,
            "gradient_with_available_correction: third-order correction channel \
             is not populated for this fit; use \
             gradient_excluding_unavailable_correction() and account for the \
             missing term explicitly"
        );
        &self.gradient_excluding_unavailable_correction() + &self.third_order_correction
    }
}

#[derive(Debug, Clone)]
pub struct SaeArrowVector {
    pub t: Array1<f64>,
    pub beta: Array1<f64>,
}

pub(crate) struct DeflatedArrowSolver<'a> {
    cache: &'a ArrowFactorCache,
    gauge_basis: Vec<Array1<f64>>,
    gauge_response_physical: Vec<Array1<f64>>,
    woodbury_factor: Option<FaerCholeskyFactor>,
    gauge_stiffness_recip: f64,
}

impl<'a> DeflatedArrowSolver<'a> {
    fn plain(cache: &'a ArrowFactorCache) -> Self {
        Self {
            cache,
            gauge_basis: Vec::new(),
            gauge_response_physical: Vec::new(),
            woodbury_factor: None,
            gauge_stiffness_recip: 0.0,
        }
    }

    fn from_orthonormal_gauges(
        cache: &'a ArrowFactorCache,
        gauge_basis: Vec<Array1<f64>>,
        stiffness: f64,
    ) -> Result<Self, String> {
        if gauge_basis.is_empty() {
            return Ok(Self::plain(cache));
        }
        if !(stiffness.is_finite() && stiffness > 0.0) {
            return Err(format!(
                "DeflatedArrowSolver: gauge stiffness must be finite and positive; got {stiffness}"
            ));
        }
        let full_len = cache.delta_t_len() + cache.k;
        let mut gauge_responses = Vec::with_capacity(gauge_basis.len());
        for gauge in &gauge_basis {
            if gauge.len() != full_len {
                return Err(format!(
                    "DeflatedArrowSolver: gauge length {} != cache full length {full_len}",
                    gauge.len()
                ));
            }
            let (sol_t, sol_beta) = cache
                .full_inverse_apply(
                    gauge.slice(s![..cache.delta_t_len()]),
                    gauge.slice(s![cache.delta_t_len()..]),
                )
                .map_err(|err| format!("DeflatedArrowSolver: gauge back-solve: {err}"))?;
            gauge_responses.push(flatten_arrow_parts(sol_t.view(), sol_beta.view()));
        }

        let rank = gauge_basis.len();
        let stiffness_recip = stiffness.recip();
        let mut gauge_metric = Array2::<f64>::zeros((rank, rank));
        let mut woodbury = Array2::<f64>::eye(rank);
        for i in 0..rank {
            woodbury[[i, i]] *= stiffness_recip;
            for j in 0..rank {
                let value = gauge_basis[i].dot(&gauge_responses[j]);
                gauge_metric[[i, j]] = value;
                woodbury[[i, j]] += value;
            }
        }
        let woodbury_factor = woodbury
            .cholesky(Side::Lower)
            .map_err(|err| format!("DeflatedArrowSolver: gauge Woodbury factor failed: {err}"))?;
        let mut gauge_response_physical = gauge_responses;
        for j in 0..rank {
            for i in 0..rank {
                let coeff = gauge_metric[[i, j]];
                for row in 0..full_len {
                    gauge_response_physical[j][row] -= coeff * gauge_basis[i][row];
                }
            }
        }
        Ok(Self {
            cache,
            gauge_basis,
            gauge_response_physical,
            woodbury_factor: Some(woodbury_factor),
            gauge_stiffness_recip: stiffness_recip,
        })
    }

    fn solve(
        &self,
        rhs_t: ArrayView1<'_, f64>,
        rhs_beta: ArrayView1<'_, f64>,
    ) -> Result<SaeArrowVector, String> {
        let (sol_t, sol_beta) = self
            .cache
            .full_inverse_apply(rhs_t, rhs_beta)
            .map_err(|err| format!("DeflatedArrowSolver: full inverse: {err}"))?;
        let Some(factor) = self.woodbury_factor.as_ref() else {
            return Ok(SaeArrowVector {
                t: sol_t,
                beta: sol_beta,
            });
        };

        let full_len = self.cache.delta_t_len() + self.cache.k;
        let mut flat = flatten_arrow_parts(sol_t.view(), sol_beta.view());
        if flat.len() != full_len {
            return Err(format!(
                "DeflatedArrowSolver: solution length {} != cache full length {full_len}",
                flat.len()
            ));
        }
        let mut gauge_coeffs = Array1::<f64>::zeros(self.gauge_basis.len());
        for (idx, gauge) in self.gauge_basis.iter().enumerate() {
            gauge_coeffs[idx] = gauge.dot(&flat);
        }
        let weights = factor.solvevec(&gauge_coeffs);
        for (gauge, &coeff) in self.gauge_basis.iter().zip(gauge_coeffs.iter()) {
            for i in 0..flat.len() {
                flat[i] -= gauge[i] * coeff;
            }
        }
        for (response, &weight) in self.gauge_response_physical.iter().zip(weights.iter()) {
            for i in 0..flat.len() {
                flat[i] -= response[i] * weight;
            }
        }
        for (gauge, &weight) in self.gauge_basis.iter().zip(weights.iter()) {
            let coeff = self.gauge_stiffness_recip * weight;
            for i in 0..flat.len() {
                flat[i] += gauge[i] * coeff;
            }
        }
        Ok(SaeArrowVector {
            t: flat.slice(s![..self.cache.delta_t_len()]).to_owned(),
            beta: flat.slice(s![self.cache.delta_t_len()..]).to_owned(),
        })
    }

    fn latent_inverse_diagonal(&self) -> Result<Array1<f64>, String> {
        if self.woodbury_factor.is_none() {
            return self
                .cache
                .latent_block_inverse_diagonal()
                .map_err(|err| format!("DeflatedArrowSolver: latent inverse diagonal: {err}"));
        }
        let total_t = self.cache.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_t);
        let rhs_beta = Array1::<f64>::zeros(self.cache.k);
        for idx in 0..total_t {
            let mut rhs_t = Array1::<f64>::zeros(total_t);
            rhs_t[idx] = 1.0;
            let solved = self.solve(rhs_t.view(), rhs_beta.view())?;
            out[idx] = solved.t[idx];
        }
        Ok(out)
    }
}

fn flatten_arrow_parts(t: ArrayView1<'_, f64>, beta: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(t.len() + beta.len());
    for i in 0..t.len() {
        out[i] = t[i];
    }
    for i in 0..beta.len() {
        out[t.len() + i] = beta[i];
    }
    out
}

fn apply_cached_arrow_hessian(
    cache: &ArrowFactorCache,
    v_t: ArrayView1<'_, f64>,
    v_beta: ArrayView1<'_, f64>,
) -> Result<SaeArrowVector, String> {
    let total_t = cache.delta_t_len();
    if v_t.len() != total_t || v_beta.len() != cache.k {
        return Err(format!(
            "apply_cached_arrow_hessian: vector shapes (t={}, beta={}) != cache shapes \
             (t={total_t}, beta={})",
            v_t.len(),
            v_beta.len(),
            cache.k
        ));
    }

    let mut out_t = Array1::<f64>::zeros(total_t);
    let mut out_beta = Array1::<f64>::zeros(cache.k);
    for row in 0..cache.n_rows() {
        let di = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let row_v = v_t.slice(s![base..base + di]);
        let factor = cache.undamped_factor(row);
        let av = cholesky_factor_apply(factor, row_v);
        for j in 0..di {
            out_t[base + j] += av[j];
        }
        if cache.k > 0 {
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) apply failed"
                ));
            }
            for j in 0..di {
                out_t[base + j] += b_vbeta[j];
            }
            if !cache.apply_htbeta_row_transpose(row, row_v, &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) apply failed"
                ));
            }
        }
    }

    if cache.k > 0 {
        let Some(schur_factor) = cache.schur_factor.as_ref() else {
            return Err(
                "apply_cached_arrow_hessian: dense Schur factor is required for gauge probing"
                    .to_string(),
            );
        };
        let schur_v = cholesky_factor_apply(schur_factor.view(), v_beta);
        for i in 0..cache.k {
            out_beta[i] += schur_v[i];
        }
        for row in 0..cache.n_rows() {
            let di = cache.row_dims[row];
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) Schur correction apply failed"
                ));
            }
            let a_inv_b_vbeta = cholesky_solve_vector(cache.undamped_factor(row), b_vbeta.view());
            if !cache.apply_htbeta_row_transpose(row, a_inv_b_vbeta.view(), &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) Schur correction apply failed"
                ));
            }
        }
    }

    Ok(SaeArrowVector {
        t: out_t,
        beta: out_beta,
    })
}

fn cholesky_factor_apply(factor: ArrayView2<'_, f64>, vector: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = factor.nrows();
    let mut lt_v = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in row..n {
            acc += factor[[col, row]] * vector[col];
        }
        lt_v[row] = acc;
    }
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in 0..=row {
            acc += factor[[row, col]] * lt_v[col];
        }
        out[row] = acc;
    }
    out
}

#[derive(Debug, Clone, Copy)]
enum SaeLocalRowVar {
    Logit { atom: usize },
    Coord { atom: usize, axis: usize },
}

#[derive(Debug, Clone)]
struct SaeBorderChannel {
    atom: usize,
    basis_col: usize,
    index: usize,
    output: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SaeRowJets {
    vars: Vec<SaeLocalRowVar>,
    first: Vec<Vec<f64>>,
    second: Vec<Vec<Vec<f64>>>,
    beta: Vec<Vec<f64>>,
    beta_deriv: Vec<Vec<Vec<f64>>>,
    beta_l_deriv: Vec<Vec<Vec<f64>>>,
}

fn sae_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Entry budget for materializing one atom's dense `(M_k·p)²` decoder
/// covariance in the fit payload. Above it (LLM-scale ambient `p`) the band
/// quantities are computed exactly from the factored frame covariance and the
/// dense export is omitted (`decoder_covariance: None`) — the python reader
/// treats it as optional. 2^24 f64 entries = 128 MiB per atom.
pub const SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES: usize = 1 << 24;

/// Posterior uncertainty of one fitted atom's manifold shape.
///
/// Produced by [`SaeManifoldTerm::assemble_shape_uncertainty`]. The covariance
/// is the φ-scaled β-block of the joint inverse Hessian (coordinates
/// marginalized out); the band is its closed-form push-forward through the
/// linear basis→ambient map `m_k(t) = Φ_k(t)·B_k`.
#[derive(Debug, Clone)]
pub struct SaeAtomShapeUncertainty {
    /// φ-scaled posterior covariance of this atom's decoder coefficients,
    /// `Cov(β_k) = φ·S_β⁻¹[block_k]`, shape `(M_k·p, M_k·p)` in the decoder's
    /// row-major `(basis, channel)` flat layout (flat index `b·p + c`).
    ///
    /// `None` when materializing it would exceed
    /// [`SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES`] (LLM-scale ambient `p`: at
    /// `(M=8, p=2048)` the dense block is 2 GiB *per atom*, at
    /// `(M=16, p=5120)` ~50 GiB). The band quantities below are still exact
    /// in that case — they are computed directly from the factored
    /// `(M_k·r_k)²` frame covariance without ever lifting it.
    pub decoder_covariance: Option<Array2<f64>>,
    /// Coordinates at which the band is evaluated, shape `(G, d_k)`.
    pub band_coords: Array2<f64>,
    /// Fitted ambient point `m_k(t) = Φ_k(t)·B_k` at each band coordinate,
    /// shape `(G, p)`.
    pub band_mean: Array2<f64>,
    /// Posterior standard deviation of each ambient channel at each band
    /// coordinate, `sqrt(Var_c(t))` with
    /// `Var_c(t) = Σ_{b1,b2} Φ[b1] Φ[b2] Cov(β_k)[(b1,c),(b2,c)]`, shape
    /// `(G, p)`.
    pub band_sd: Array2<f64>,
}

/// Posterior shape uncertainty for a whole SAE-manifold fit: one band per atom
/// plus the shared Gaussian reconstruction dispersion `φ̂` used to scale every
/// covariance. See [`SaeManifoldTerm::assemble_shape_uncertainty`].
#[derive(Debug, Clone)]
pub struct SaeShapeUncertainty {
    /// Gaussian reconstruction scale `φ̂ = RSS / residual-dof`.
    pub dispersion: f64,
    /// One entry per atom, in atom order.
    pub atoms: Vec<SaeAtomShapeUncertainty>,
}

/// Per-row active-set layout for sparse SAE assignment (any mode).
///
/// When the assignment is sparse — structurally (JumpReLU gate) or
/// effectively (softmax / IBP-MAP at large `K`, where the assignment mass
/// concentrates on a small support) — only a subset of `K` atoms are active
/// per observation.  The Arrow-Schur row block for observation `i` has dim
/// `q_active_i = |active_atoms_i| + Σ_{k ∈ active_i} d_k` rather than
/// `q = assignment_dim + Σ_k d_k`.  This struct records which atoms are active per row
/// and maps compressed block positions back to full-q positions so that
/// `apply_newton_step` can unpack the compact `delta_t` from the solve.
///
/// For JumpReLU the active set is exactly the gated support
/// (`a_{n,k} ≠ 0`), so the compact solve is identity to the dense solve.
/// For softmax / IBP-MAP the active set is the union of a top-`k_active_cap`
/// truncation and a magnitude cutoff on `a_{n,k}`; this is only enabled when
/// `K` is large enough that the dense `(m_total · p)²` data Gram would not
/// fit the host / device working-set budget, and the dropped atoms carry
/// `O(a_{n,k}²)` curvature that is negligible by construction of the cutoff.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row.
    pub active_atoms: Vec<Vec<usize>>,
    /// For row `i`, active atom `active_atoms[i][j]` has its coord block
    /// starting at compressed position `coord_starts[i][j]`.
    pub coord_starts: Vec<Vec<usize>>,
    /// Full-q coordinate offset for atom `k` (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
    /// Per-atom coordinate dimensions, indexed by atom index.
    pub coord_dims: Vec<usize>,
}

impl SaeRowLayout {
    /// JumpReLU optimization active set: atoms inside the smooth prior's
    /// machine-precision support `(logit - threshold)/tau > -36` (see
    /// [`jumprelu_in_optimization_band`]). This is intentionally wider than the
    /// hard forward gate `logit > threshold` so gated-off atoms can remain in the
    /// Newton system for value-consistent prior terms. Their forward
    /// reconstruction contribution and data-fit logit JVP remain hard-zero while
    /// `a_k = 0`.
    fn from_jumprelu(
        n: usize,
        k_atoms: usize,
        threshold: f64,
        temperature: f64,
        logits: &Array2<f64>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut per_row = Vec::with_capacity(n);
        for row in 0..n {
            let row_logits = logits.row(row);
            let active: Vec<usize> = (0..k_atoms)
                .filter(|&k| jumprelu_in_optimization_band(row_logits[k], threshold, temperature))
                .collect();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Mode-agnostic effective active set for dense-weight modes (softmax /
    /// IBP-MAP) at large `K`: keep, per row, the top-`k_active_cap` atoms by
    /// `|a_{n,k}|` whose magnitude also exceeds `cutoff`.
    ///
    /// `assignments[row]` is the dense length-`K` assignment vector `a_{n,·}`.
    /// The active set is always non-empty (the single largest-magnitude atom is
    /// retained even if below `cutoff`) so every row keeps a valid block.
    fn from_dense_weights(
        assignments: &[Array1<f64>],
        k_active_cap: usize,
        cutoff: f64,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let cap = k_active_cap.max(1);
        let mut per_row = Vec::with_capacity(assignments.len());
        for a in assignments {
            let k = a.len();
            // Rank atoms by descending |a_k|; keep those above cutoff, capped.
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&i, &j| {
                a[j].abs()
                    .partial_cmp(&a[i].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut active: Vec<usize> = idx
                .iter()
                .copied()
                .take(cap)
                .filter(|&k_idx| a[k_idx].abs() > cutoff)
                .collect();
            if active.is_empty() {
                // Retain the single largest-magnitude atom so the row block is
                // never empty (a degenerate empty block would zero the row).
                if let Some(&top) = idx.first() {
                    active.push(top);
                }
            }
            active.sort_unstable();
            per_row.push(active);
        }
        Self::from_active_atoms(per_row, coord_dims, coord_offsets_full)
    }

    /// Build from explicit per-row active-atom index lists.
    fn from_active_atoms(
        active_atoms: Vec<Vec<usize>>,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Self {
        let mut coord_starts_all = Vec::with_capacity(active_atoms.len());
        for active in &active_atoms {
            let mut starts = Vec::with_capacity(active.len());
            let mut cursor = active.len();
            for &k in active {
                starts.push(cursor);
                cursor += coord_dims[k];
            }
            coord_starts_all.push(starts);
        }
        Self {
            active_atoms,
            coord_starts: coord_starts_all,
            coord_offsets_full,
            coord_dims,
        }
    }

    /// Per-row compressed dim.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        active.len() + coord_sum
    }

    /// Expand a compact `delta_t` row slice back into full-q, zeros for inactive.
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let active = &self.active_atoms[row];
        let starts = &self.coord_starts[row];
        for (j, &k) in active.iter().enumerate() {
            out[k] = delta_t_row[j];
            let d = self.coord_dims[k];
            let full_off = self.coord_offsets_full[k];
            for axis in 0..d {
                out[full_off + axis] = delta_t_row[starts[j] + axis];
            }
        }
    }
}

/// The global-optimality verdict of the curved-dictionary incoherence
/// certificate (#1008): whether the fit's basin stationary point is certified
/// unique up to the residual gauge group, and by what margin.
///
/// The certificate is **conservative by construction**: it certifies only when
/// the conservative sufficient condition holds with positive margin, so a
/// `CertifiedGlobal` verdict can never be wrong (the phase-diagram validation
/// asserts exactly this — no certified-but-wrong cell, ever). An
/// `Uncertified` verdict is *not* a claim of non-uniqueness — it is the honest
/// "this certificate cannot decide", which is the only safe failure mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlobalOptimalityVerdict {
    /// The conservative sufficient condition holds: the basin stationary point
    /// is unique up to the certified residual gauge group. `margin` is the
    /// (positive) slack `budget − μ̂` by which the condition is met.
    CertifiedGlobal { margin: f64 },
    /// The condition is not met (or a precondition — graph-validity / SNR > 1 —
    /// fails). `margin` is the (non-positive) slack, or `f64::NEG_INFINITY` when
    /// a precondition rules certification out entirely. Multistart / homotopy is
    /// genuinely needed here.
    Uncertified { margin: f64 },
}

impl GlobalOptimalityVerdict {
    /// The signed margin `budget − μ̂` (positive ⇒ certified). A precondition
    /// failure reports `f64::NEG_INFINITY`.
    pub fn margin(&self) -> f64 {
        match self {
            Self::CertifiedGlobal { margin } | Self::Uncertified { margin } => *margin,
        }
    }

    /// Whether the fit is certified globally optimal up to the gauge group.
    pub fn is_certified(&self) -> bool {
        matches!(self, Self::CertifiedGlobal { .. })
    }
}

/// Conservative tangent-graph curvature budget: the atom image is a graph over
/// its tangent frame only while `C_KAPPA · κ̂` stays below 1 — i.e. the relative
/// second-fundamental-form curvature `κ̂` (perp curvature per unit tangent
/// scale) is below `1`. Above it the atom turns faster than its own tangent
/// extent and the linear-case perturbation argument is void, so the certificate
/// refuses to certify. A circle of radius `r` has `κ̂ = 1/r`, so this admits
/// `r > 1` (benign, well-resolved atoms) and rejects tightly-curved ones whose
/// graph approximation is uncontrolled. Raising this constant only ever shrinks
/// the certified region (withholds certification), never grants a wrong one.
pub const SAE_CERT_CURVATURE_CONSTANT: f64 = 1.0;

/// Conservative incoherence-budget constant `c0` in the sufficient condition
/// `μ̂ ≤ c0 · a_floor² · (1 − 1/SNR) · (1 − C_κ κ̂) / K`. Small (conservative):
/// shrinking the budget can only withhold certification, never grant a wrong
/// one.
pub const SAE_CERT_INCOHERENCE_BUDGET: f64 = 0.125;

/// The conservative curved-dictionary global-optimality threshold (#1008).
///
/// # The condition
///
/// Following the linear exact-recovery lineage (Spielman–Wang–Wright complete
/// case; Sun–Qu–Wright geometric analysis — in benign regimes every local min
/// is global) perturbed to curved atoms: the atom image is a graph over its
/// tangent frame with second-fundamental-form curvature `κ`, so the linear-case
/// arguments perturb when `κ·diam(chart)` is small. The competing-basin coupling
/// is the cross-atom frame incoherence `μ` amplified by co-activation; the
/// within-atom restricted strong convexity that pins each atom scales with the
/// activity floor (how reliably the atom fires) and the SNR (how far the signal
/// is above noise), and is **degraded by curvature** (the graph approximation
/// error). The certificate certifies global optimality up to the residual gauge
/// when
///
/// ```text
///   μ̂  ≤  c0 · a_floor² · (1 − 1/SNR) · (1 − C_κ · κ̂_max) / K
/// ```
///
/// subject to the preconditions `C_κ · κ̂_max < 1` (tangent-graph validity) and
/// `SNR > 1` (signal above noise). `a_floor` is the support activity floor
/// (`min_k max_i a_ik`, the same statistic the collapse guard reads), `K` the
/// atom count, `κ̂_max` the largest per-atom second-fundamental-form bound.
///
/// # Conservatism
///
/// Every constant is chosen to *shrink* the certified region relative to the
/// true (unknown) sharp threshold: `c0` is small, `C_κ` is large. A
/// `CertifiedGlobal` verdict therefore implies the sharp condition with room to
/// spare — it can never be wrong. An `Uncertified` verdict is the honest "cannot
/// decide", never a claim of non-uniqueness. The cross-validation with the
/// certified-homotopy bifurcation events (#1007) is exactly this: a bifurcation
/// (a competing basin appearing) should only ever occur where this margin is
/// non-positive.
pub fn curved_dictionary_global_optimality_verdict(
    mu_hat: f64,
    kappa_max: f64,
    activity_floor: f64,
    snr_proxy: f64,
    k_atoms: usize,
) -> GlobalOptimalityVerdict {
    // Preconditions: any non-finite input, no atoms, a curvature that voids the
    // tangent-graph perturbation, or SNR at/below the noise floor ⇒ refuse.
    if !mu_hat.is_finite()
        || !kappa_max.is_finite()
        || !activity_floor.is_finite()
        || !snr_proxy.is_finite()
        || k_atoms == 0
    {
        return GlobalOptimalityVerdict::Uncertified {
            margin: f64::NEG_INFINITY,
        };
    }
    let curvature_factor = 1.0 - SAE_CERT_CURVATURE_CONSTANT * kappa_max.max(0.0);
    let snr_factor = 1.0 - 1.0 / snr_proxy;
    if curvature_factor <= 0.0 || snr_factor <= 0.0 {
        // Tangent-graph perturbation void, or signal not above noise: the
        // linear-case argument does not apply, so certification is impossible.
        return GlobalOptimalityVerdict::Uncertified {
            margin: f64::NEG_INFINITY,
        };
    }
    let a = activity_floor.max(0.0);
    let budget =
        SAE_CERT_INCOHERENCE_BUDGET * a * a * snr_factor * curvature_factor / k_atoms as f64;
    let margin = budget - mu_hat;
    if margin > 0.0 {
        GlobalOptimalityVerdict::CertifiedGlobal { margin }
    } else {
        GlobalOptimalityVerdict::Uncertified { margin }
    }
}

/// Empirical quantities that feed the curved-dictionary incoherence theorem,
/// plus the conservative global-optimality verdict (#1008).
#[derive(Clone, Debug)]
pub struct CertificateInputs {
    /// `max_{j != k} sigma_max(U_j^T U_k)` over decoder output subspaces.
    pub mu_hat: f64,
    /// Per-atom maximum empirical second-fundamental-form norm on the fitted
    /// coordinate grid.
    pub per_atom_kappa_hat: Vec<f64>,
    /// Mean fitted gate/assignment mass per atom.
    pub per_atom_mean_activity: Vec<f64>,
    /// Largest fitted gate/assignment mass per atom.
    pub per_atom_peak_activity: Vec<f64>,
    /// Conservative dictionary activity floor, `min_k mean_i a_ik`.
    pub mean_activity_floor: f64,
    /// Support floor matching the collapse guard statistic, `min_k max_i a_ik`.
    pub peak_activity_floor: f64,
    /// `mean_i ||sum_k a_ik g_k(t_ik)||^2 / dispersion`.
    pub snr_proxy: f64,
    /// Dispersion used in [`Self::snr_proxy`].
    pub dispersion: f64,
    /// The conservative global-optimality verdict (#1008):
    /// `CertifiedGlobal { margin }` when the sufficient condition
    /// ([`curved_dictionary_global_optimality_verdict`]) holds with positive
    /// slack — the basin stationary point is unique up to the residual gauge
    /// group — else `Uncertified { margin }`. Conservative: a certified verdict
    /// is never wrong; an uncertified one is "cannot decide", not "non-unique".
    pub global_optimality: GlobalOptimalityVerdict,
    /// Human-readable summary of the quantities and verdict.
    pub note: String,
}

/// The additive post-fit diagnostics for a fitted [`SaeManifoldTerm`]: the
/// two-score per-atom lens, residual-gauge certificate, and empirical
/// incoherence/curvature certificate inputs.
///
/// Built by [`SaeManifoldTerm::fit_diagnostics_report`]. Both reports are pure
/// reads of the fitted term + its single per-row metric; nothing here feeds back
/// into any loss, criterion, penalty, or optimizer state. Under a Euclidean /
/// no-harvest provenance the lens coupling degrades to `None` and the gauge is
/// certified under Euclidean provenance — never an error, never flag-gated.
#[derive(Clone, Debug)]
pub struct SaeManifoldFitDiagnostics {
    /// Per-atom presence / behavioral coupling / discrepancy
    /// ([`crate::inference::atom_lens::atom_two_lens`]).
    pub atom_two_lens: crate::inference::atom_lens::AtomTwoLensReport,
    /// Residual-gauge certificate: which symmetry group the fit is identified up
    /// to ([`crate::sae_identifiability::residual_gauge`]).
    pub residual_gauge: crate::sae_identifiability::ResidualGaugeReport,
    /// Empirical curved-dictionary certificate inputs (#1008). Present when the
    /// caller supplies the fitted reconstruction dispersion needed for the SNR
    /// proxy; absent for legacy callers that only need the existing diagnostics.
    pub incoherence_report: Option<CertificateInputs>,
}

/// Honest trust-diagnostics payload for the Python `diagnostics` block (#1005).
///
/// This deliberately contains only quantities with exact fitted-state producers:
/// tangent spectrum/condition, assignment support, activation frequency, and the
/// basis-kind untyped flag. No topology margins, level-0 references, coherence,
/// or reconstruction proxy fields are represented here.
#[derive(Clone, Debug)]
pub struct SaeTrustDiagnostics {
    pub atom_trust: Vec<f64>,
    pub atoms: Vec<SaeAtomTrustDiagnostics>,
}

#[derive(Clone, Debug)]
pub struct SaeAtomTrustDiagnostics {
    pub trust_score: f64,
    pub sigma_min_tangent: f64,
    pub sigma_max_tangent: f64,
    pub tangent_condition_score: f64,
    pub coverage: f64,
    pub activation_frequency: f64,
    pub untyped: bool,
    pub active_token_count: usize,
}

/// Build the empirical curved-dictionary certificate quantities from a fitted
/// term and its Gaussian reconstruction dispersion.
///
/// This reports only computable theorem-side inputs. It intentionally has no
/// global-optimality verdict: the threshold function relating these inputs is
/// future theory (#1008).
pub fn dictionary_incoherence_report(term: &SaeManifoldTerm) -> Result<CertificateInputs, String> {
    let dispersion = term.certificate_dispersion.ok_or_else(|| {
        "dictionary_incoherence_report: fitted reconstruction dispersion is unavailable".to_string()
    })?;
    dictionary_incoherence_report_with_dispersion(term, dispersion)
}

/// Build the empirical curved-dictionary certificate quantities from a fitted
/// term and an explicit Gaussian reconstruction dispersion.
pub fn dictionary_incoherence_report_with_dispersion(
    term: &SaeManifoldTerm,
    dispersion: f64,
) -> Result<CertificateInputs, String> {
    if !dispersion.is_finite() || dispersion <= 0.0 {
        return Err(format!(
            "dictionary_incoherence_report: dispersion must be finite and positive, got {dispersion}"
        ));
    }
    let mu_hat = dictionary_frame_incoherence(term)?;
    let per_atom_kappa_hat = term
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, _)| atom_curvature_bound(term, atom_idx))
        .collect::<Result<Vec<_>, _>>()?;
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    let k_atoms = assignments.ncols();
    let mut per_atom_mean_activity = Vec::with_capacity(k_atoms);
    let mut per_atom_peak_activity = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut sum = 0.0_f64;
        let mut peak = 0.0_f64;
        for row in 0..n {
            let value = assignments[[row, atom_idx]];
            sum += value;
            peak = peak.max(value);
        }
        per_atom_mean_activity.push(if n > 0 { sum / n as f64 } else { 0.0 });
        per_atom_peak_activity.push(peak);
    }
    let mean_activity_floor = per_atom_mean_activity
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let peak_activity_floor = per_atom_peak_activity
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let fitted = term.fitted();
    let signal_power = if fitted.is_empty() {
        0.0
    } else {
        fitted.iter().map(|v| v * v).sum::<f64>() / fitted.len() as f64
    };
    let mean_activity_floor = if mean_activity_floor.is_finite() {
        mean_activity_floor
    } else {
        0.0
    };
    let peak_activity_floor = if peak_activity_floor.is_finite() {
        peak_activity_floor
    } else {
        0.0
    };
    let snr_proxy = signal_power / dispersion;
    // The curvature bound entering the threshold is the largest per-atom
    // second-fundamental-form norm (the worst graph-approximation error across
    // the dictionary). The support activity floor `min_k max_i a_ik` is the
    // honest "how reliably does the weakest atom fire" statistic.
    let kappa_max = per_atom_kappa_hat.iter().copied().fold(0.0_f64, f64::max);
    let global_optimality = curved_dictionary_global_optimality_verdict(
        mu_hat,
        kappa_max,
        peak_activity_floor,
        snr_proxy,
        k_atoms,
    );
    let note = match global_optimality {
        GlobalOptimalityVerdict::CertifiedGlobal { margin } => format!(
            "global optimality CERTIFIED up to the residual gauge group \
             (margin {margin:.3e}); μ̂={mu_hat:.3e}, κ̂_max={kappa_max:.3e}, \
             a_floor={peak_activity_floor:.3e}, SNR={snr_proxy:.3e}"
        ),
        GlobalOptimalityVerdict::Uncertified { margin } => format!(
            "global optimality UNCERTIFIED (margin {margin:.3e}; cannot decide — \
             multistart/homotopy genuinely needed); μ̂={mu_hat:.3e}, \
             κ̂_max={kappa_max:.3e}, a_floor={peak_activity_floor:.3e}, \
             SNR={snr_proxy:.3e}"
        ),
    };
    Ok(CertificateInputs {
        mu_hat,
        per_atom_kappa_hat,
        per_atom_mean_activity,
        per_atom_peak_activity,
        mean_activity_floor,
        peak_activity_floor,
        snr_proxy,
        dispersion,
        global_optimality,
        note,
    })
}

fn dictionary_frame_incoherence(term: &SaeManifoldTerm) -> Result<f64, String> {
    let frames = (0..term.k_atoms())
        .map(|atom_idx| certificate_output_frame(term, atom_idx))
        .collect::<Result<Vec<_>, _>>()?;
    let mut mu = 0.0_f64;
    for j in 0..frames.len() {
        for k in (j + 1)..frames.len() {
            if frames[j].ncols() == 0 || frames[k].ncols() == 0 {
                continue;
            }
            let overlap = fast_atb(&frames[j], &frames[k]);
            let (_u, s, _vt) = overlap.svd(false, false).map_err(|e| {
                format!("dictionary_frame_incoherence: SVD failed for atom pair ({j}, {k}): {e}")
            })?;
            let pair = s.iter().copied().fold(0.0_f64, f64::max);
            mu = mu.max(pair);
        }
    }
    Ok(mu)
}

fn certificate_output_frame(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Array2<f64>, String> {
    let atom = &term.atoms[atom_idx];
    if atom.decoder_frame.is_some() {
        return Ok(term.frame_output_matrix(atom_idx));
    }
    let p = atom.output_dim();
    let (_u, s, vt_opt) = atom
        .decoder_coefficients
        .svd(false, true)
        .map_err(|e| format!("certificate_output_frame: SVD failed for atom {atom_idx}: {e}"))?;
    let max_sv = s.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Ok(Array2::<f64>::zeros((p, 0)));
    }
    let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
    let rank = s.iter().filter(|&&value| value > tol).count();
    let vt = vt_opt.ok_or_else(|| {
        format!("certificate_output_frame: SVD returned no right factor for atom {atom_idx}")
    })?;
    let rank = rank.min(vt.nrows());
    let mut frame = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            frame[[row, col]] = vt[[col, row]];
        }
    }
    Ok(frame)
}

fn atom_curvature_bound(term: &SaeManifoldTerm, atom_idx: usize) -> Result<f64, String> {
    let atom = &term.atoms[atom_idx];
    let coords = term.assignment.coords[atom_idx].as_matrix();
    let second = atom
        .basis_evaluator
        .as_ref()
        .and_then(|evaluator| evaluator.second_jet_dyn(coords.view()))
        .ok_or_else(|| {
            format!(
                "atom_curvature_bound: atom {atom_idx} has no analytic second jet; cannot compute kappa_hat"
            )
        })?
        .map_err(|e| format!("atom_curvature_bound: atom {atom_idx} second jet failed: {e}"))?;
    let n = atom.n_obs();
    let m = atom.basis_size();
    let d = atom.latent_dim;
    let p = atom.output_dim();
    if second.dim() != (n, m, d, d) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} second jet shape {:?} must be ({n}, {m}, {d}, {d})",
            second.dim()
        ));
    }
    let mut max_kappa = 0.0_f64;
    let mut tangent = Array2::<f64>::zeros((p, d));
    let mut second_vec = vec![0.0_f64; p];
    for row in 0..n {
        for axis in 0..d {
            let mut col = vec![0.0_f64; p];
            atom.fill_decoded_derivative_row(row, axis, &mut col);
            for out in 0..p {
                tangent[[out, axis]] = col[out];
            }
        }
        let tangent_rank = tangent_frame_rank(tangent.view())?;
        let tangent_scale = tangent_rank.0;
        let q = tangent_rank.1;
        for axis_a in 0..d {
            for axis_b in 0..d {
                second_vec.fill(0.0);
                for basis_col in 0..m {
                    let h = second[[row, basis_col, axis_a, axis_b]];
                    if h == 0.0 {
                        continue;
                    }
                    for out in 0..p {
                        second_vec[out] += h * atom.decoder_coefficients[[basis_col, out]];
                    }
                }
                let perp_norm = projected_perp_norm(&second_vec, q.view());
                if tangent_scale > 0.0 {
                    max_kappa = max_kappa.max(perp_norm / tangent_scale);
                } else if perp_norm > 0.0 {
                    return Ok(f64::INFINITY);
                }
            }
        }
    }
    Ok(max_kappa)
}

fn tangent_frame_rank(tangent: ArrayView2<'_, f64>) -> Result<(f64, Array2<f64>), String> {
    let p = tangent.nrows();
    let d = tangent.ncols();
    if p == 0 || d == 0 {
        return Ok((0.0, Array2::<f64>::zeros((p, 0))));
    }
    let (u_opt, s, _vt) = tangent
        .to_owned()
        .svd(true, false)
        .map_err(|e| format!("tangent_frame_rank: SVD failed: {e}"))?;
    let max_sv = s.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Ok((0.0, Array2::<f64>::zeros((p, 0))));
    }
    let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
    let rank = s.iter().filter(|&&value| value > tol).count();
    let min_positive = s
        .iter()
        .copied()
        .filter(|value| *value > tol)
        .fold(f64::INFINITY, f64::min);
    let u = u_opt.ok_or_else(|| "tangent_frame_rank: SVD returned no U".to_string())?;
    let rank = rank.min(u.ncols());
    let mut q = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            q[[row, col]] = u[[row, col]];
        }
    }
    Ok((min_positive * min_positive, q))
}

fn projected_perp_norm(vector: &[f64], tangent_frame: ArrayView2<'_, f64>) -> f64 {
    let mut residual = vector.to_vec();
    for axis in 0..tangent_frame.ncols() {
        let mut coeff = 0.0_f64;
        for out in 0..tangent_frame.nrows() {
            coeff += tangent_frame[[out, axis]] * vector[out];
        }
        if coeff == 0.0 {
            continue;
        }
        for out in 0..tangent_frame.nrows() {
            residual[out] -= coeff * tangent_frame[[out, axis]];
        }
    }
    residual.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Full SAE-manifold term.
#[derive(Debug)]
pub struct SaeManifoldTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignment,
    temperature_schedule: Option<GumbelTemperatureSchedule>,
    /// Active-set row layout from the most recent `assemble_arrow_schur` call.
    /// `None` for dense modes (Softmax / IBPMap) or when not yet assembled.
    last_row_layout: Option<SaeRowLayout>,
    /// The single provenance-carrying per-row inner product (Object 2). The
    /// reconstruction likelihood whitens residuals through it and the isometry
    /// gauge's [`crate::terms::analytic_penalties::WeightField`] is constructed
    /// from the same object, so a likelihood-metric ≠ gauge-metric state is
    /// unrepresentable. `None` ⇒ Euclidean / isotropic (magic-by-default: the
    /// metric is selected by whether per-row Fisher factors were installed, not
    /// by a flag), which is bit-for-bit the historical isotropic `φ̂` path.
    row_metric: Option<crate::inference::row_metric::RowMetric>,
    /// #976 Layer-1 guard ledger for the most recent joint fit: every
    /// active-mass breach with the action taken (re-seed / terminal). Cleared
    /// at the start of each `run_joint_fit_arrow_schur`; read post-fit via
    /// [`SaeManifoldTerm::collapse_events`] and carried onto the
    /// structure-search [`crate::solver::structure_search::SearchLedger`].
    collapse_events: Vec<CollapseEvent>,
    /// Per-row **design honesty weights** (#991): Horvitz–Thompson inclusion
    /// corrections from a designed corpus subsample
    /// ([`crate::inference::row_measure::RowMeasure::designed_subsample`] /
    /// [`crate::terms::sae_corpus::designed_target`]), self-normalized to
    /// mean `1.0` over the term's rows so dispersion, dof, and the
    /// data-vs-penalty balance stay consistent at the fitted sample size while
    /// the design's selection bias is removed (oversampled loud rows are
    /// downweighted back).
    ///
    /// The weights enter the objective as a per-row scalar metric `w_i · I_p`
    /// on the reconstruction channel ONLY, realized as a `√w_i` scaling of the
    /// per-row residual, latent Jacobian, and β basis load at their single
    /// construction sites in the assembly — so the data-fit value, the t-block
    /// Gauss-Newton, the β gradient/Gram, and the cross blocks all carry
    /// exactly one factor of `w_i` and cannot desync (the same discipline as
    /// the #974 whitening seam). Per-row latent priors (assignment prior, ARD
    /// coordinate prior) are deliberately NOT weighted: included rows' latent
    /// states are genuine model components of the subsampled model,
    /// conditional on inclusion; the HT correction applies to the row
    /// *evidence* about shared structure (decoder β, ρ), not to the latent
    /// priors. `None` ⇒ the exact unweighted path, bit-for-bit.
    row_loss_weights: Option<Vec<f64>>,
    /// #972 / #977 T1: whether the MOST RECENT `assemble_arrow_schur` built the
    /// β-tier in the *factored* Grassmann-coordinate layout (border width
    /// [`Self::factored_border_dim`], the per-atom `C_k` blocks) rather than the
    /// full-`B` layout ([`Self::beta_dim`]). When `true`, the `delta_beta` the
    /// arrow solver returns is a `ΔC` (factored coordinates) that
    /// [`Self::apply_newton_step_impl`] must LIFT through each active frame
    /// (`ΔB_k = ΔC_k U_kᵀ`) before applying it to the decoder. `false` ⇒ the
    /// historical full-`B` path, where `delta_beta` is `ΔB` directly. Set in
    /// lock-step with the assembled system so the step interpretation cannot
    /// drift from the layout the system was built in.
    last_frames_active: bool,
    /// Reusable dense β-tier workspace for analytic penalty assembly. SAE
    /// immediately lowers the dense block into a `BetaPenaltyOp`, so the returned
    /// `ArrowSchurSystem` does not need to keep owning the allocation.
    border_hbb_workspace: Array2<f64>,
    /// Fitted Gaussian reconstruction dispersion used only by the empirical
    /// incoherence/curvature certificate-input report. `None` for synthetic terms
    /// or legacy internal callers that have not computed post-fit dispersion.
    certificate_dispersion: Option<f64>,
    /// Outcome of the most recent curvature-homotopy entry walk (#1007), or
    /// `None` when no walk has run (the seed cascade entry, or any consumer that
    /// never invokes the tracker). Recorded on the fit payload so the bifurcation
    /// / collapse outcome is observable — never a silent fallback. Cleared by
    /// the objective's `reset` so each seed's walk reports only its own run.
    curvature_walk_report: Option<CurvatureWalkReport>,
    /// Deflated row-gauge direction count established by the first undamped
    /// evidence factorization in the current optimization. A later change means
    /// the quotient dimension changed mid-solve, which is a structural event and
    /// must not be hidden inside the Laplace normalizer.
    expected_evidence_gauge_deflated_directions: Option<usize>,
}

impl Clone for SaeManifoldTerm {
    fn clone(&self) -> Self {
        Self {
            atoms: self.atoms.clone(),
            assignment: self.assignment.clone(),
            temperature_schedule: self.temperature_schedule.clone(),
            last_row_layout: self.last_row_layout.clone(),
            row_metric: self.row_metric.clone(),
            collapse_events: self.collapse_events.clone(),
            row_loss_weights: self.row_loss_weights.clone(),
            last_frames_active: self.last_frames_active,
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: self.certificate_dispersion,
            curvature_walk_report: self.curvature_walk_report.clone(),
            expected_evidence_gauge_deflated_directions: self
                .expected_evidence_gauge_deflated_directions,
        }
    }
}

/// Snapshot of exactly the mutable term state that an `apply_newton_step` +
/// `loss` line-search trial perturbs: per-atom decoder coefficients, the
/// `refresh_basis`-rebuilt basis evaluations (`basis_values`, `basis_jacobian`),
/// and the live intrinsic smoothness Gram read by the objective, plus the
/// assignment logits and latent coordinates.
///
/// Static fields (atom names, basis kinds, basis-evaluator `Arc`s, assignment
/// mode, temperature schedule) are *not* snapshotted: they are invariant across
/// an inner Newton line search, so the previous `self.clone()` per halving
/// re-copied them needlessly. Cloning only the line-search state keeps the
/// `O(N·M·d)` `basis_jacobian` copy off the per-halving hot path (one snapshot
/// before the search, one restore per rejected trial) instead of firing it on
/// every Armijo backtrack.
///
/// The canonical `smooth_penalty_raw` / `smooth_penalty_order` are static, but
/// the live intrinsic roughness Gram `smooth_penalty` is mutable state: it is
/// refreshed by assembly from the current decoder and basis Jacobian, and the
/// line-search objective reads it directly. Restoring it with the decoder and
/// basis caches keeps every rejected trial's baseline and nonlinear objective
/// on the same lagged-diffusivity quadratic.
#[derive(Debug)]
struct SaeManifoldMutableState {
    /// Per-atom `(basis_values, basis_jacobian, decoder_coefficients, smooth_penalty)`.
    atoms: Vec<(Array2<f64>, Array3<f64>, Array2<f64>, Array2<f64>)>,
    logits: Array2<f64>,
    coords: Vec<LatentCoordValues>,
    last_row_layout: Option<SaeRowLayout>,
}

impl SaeManifoldTerm {
    #[must_use = "build error must be handled"]
    pub fn new(atoms: Vec<SaeManifoldAtom>, assignment: SaeAssignment) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("SaeManifoldTerm::new: at least one atom required".into());
        }
        let n = atoms[0].n_obs();
        let p = atoms[0].output_dim();
        if assignment.n_obs() != n || assignment.k_atoms() != atoms.len() {
            return Err(format!(
                "SaeManifoldTerm::new: assignment shape ({}, {}) does not match atoms ({n}, {})",
                assignment.n_obs(),
                assignment.k_atoms(),
                atoms.len()
            ));
        }
        for (k, atom) in atoms.iter().enumerate() {
            if atom.n_obs() != n {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} has n_obs={} but atom 0 has {n}",
                    atom.n_obs()
                ));
            }
            if atom.output_dim() != p {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} output_dim={} but atom 0 has {p}",
                    atom.output_dim()
                ));
            }
            if atom.latent_dim != assignment.coords[k].latent_dim() {
                return Err(format!(
                    "SaeManifoldTerm::new: atom {k} latent_dim={} but assignment coord has {}",
                    atom.latent_dim,
                    assignment.coords[k].latent_dim()
                ));
            }
        }
        Ok(Self {
            atoms,
            assignment,
            temperature_schedule: None,
            last_row_layout: None,
            row_metric: None,
            collapse_events: Vec::new(),
            row_loss_weights: None,
            last_frames_active: false,
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: None,
            curvature_walk_report: None,
            expected_evidence_gauge_deflated_directions: None,
        })
    }

    /// Install the fitted reconstruction dispersion used by
    /// [`dictionary_incoherence_report`]. This is a pure diagnostic scalar and
    /// does not feed any loss, criterion, penalty, or optimizer state.
    pub fn set_certificate_dispersion(&mut self, dispersion: f64) -> Result<(), String> {
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(format!(
                "SaeManifoldTerm::set_certificate_dispersion: dispersion must be finite and positive, got {dispersion}"
            ));
        }
        self.certificate_dispersion = Some(dispersion);
        Ok(())
    }

    /// Profile the Gaussian reconstruction dispersion at the current seed
    /// state. This is the scale used to make SAE penalty seeds dimensionless
    /// before the outer rho search starts.
    pub fn seed_reconstruction_dispersion(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let fitted = self.try_fitted()?;
        if fitted.dim() != target.dim() {
            return Err(format!(
                "SaeManifoldTerm::seed_reconstruction_dispersion: fitted {:?} != target {:?}",
                fitted.dim(),
                target.dim()
            ));
        }
        let n_scalar = (target.nrows() * target.ncols()).max(1) as f64;
        let mut rss = 0.0_f64;
        for row in 0..target.nrows() {
            for col in 0..target.ncols() {
                let r = target[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        if !rss.is_finite() || rss < 0.0 {
            return Err(format!(
                "SaeManifoldTerm::seed_reconstruction_dispersion: non-finite seed RSS {rss}"
            ));
        }
        Ok((rss / n_scalar).max(SAE_SEED_DISPERSION_FLOOR))
    }

    /// Install per-row design honesty weights (#991) — the `1/π` inclusion
    /// corrections of a designed corpus subsample (see the field docs on
    /// `row_loss_weights` for exactly where they enter the objective).
    ///
    /// Weights must be finite and strictly positive, one per term row. They
    /// are self-normalized to mean `1.0` here (only the *relative* design
    /// correction matters at the fitted sample size; the absolute `n/budget`
    /// scale would silently inflate the dispersion estimate against the
    /// sample-sized dof). Weights that are identically equal after
    /// normalization (an exact full pass, or any uniform design) are stored
    /// as `None`, so the unweighted path stays bit-for-bit identical rather
    /// than "multiplied by 1.0".
    pub fn set_row_loss_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.n_obs() {
            return Err(format!(
                "SaeManifoldTerm::set_row_loss_weights: {} weights for {} rows",
                weights.len(),
                self.n_obs()
            ));
        }
        if weights.is_empty() {
            self.row_loss_weights = None;
            return Ok(());
        }
        if !weights.iter().all(|w| w.is_finite() && *w > 0.0) {
            return Err(
                "SaeManifoldTerm::set_row_loss_weights: weights must be finite and strictly \
                 positive"
                    .to_string(),
            );
        }
        let first = weights[0];
        if weights.iter().all(|w| *w == first) {
            // Uniform design (full pass, or flat measure): the normalized
            // weight is exactly 1 everywhere — take the unweighted path.
            self.row_loss_weights = None;
            return Ok(());
        }
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        self.row_loss_weights = Some(weights.into_iter().map(|w| w / mean).collect());
        Ok(())
    }

    /// The installed (mean-1 normalized) design honesty weights, `None` on the
    /// exact unweighted path.
    pub fn row_loss_weights(&self) -> Option<&[f64]> {
        self.row_loss_weights.as_deref()
    }

    /// Drop any installed per-row reconstruction weights, returning the term to
    /// the exact unweighted (full-pass) path. Used by the #997 structure-search
    /// wiring to clear the internal estimation/evaluation mask off the adopted
    /// term before the payload reconstruction is read over all rows.
    pub fn clear_row_loss_weights(&mut self) {
        self.row_loss_weights = None;
    }

    /// Install the single per-row [`RowMetric`](crate::inference::row_metric::RowMetric)
    /// that both the reconstruction likelihood and the isometry gauge read.
    /// Installing per-row output-Fisher factors here flips the provenance to
    /// `OutputFisher` *and* is the only way the gauge acquires a non-identity
    /// weight, so the two inner products cannot diverge. Passing a Euclidean
    /// metric (or never calling this) keeps the bit-identical isotropic path.
    ///
    /// The metric's row count and output dimension must match the term.
    pub fn set_row_metric(
        &mut self,
        metric: crate::inference::row_metric::RowMetric,
    ) -> Result<(), String> {
        if metric.n_rows() != self.n_obs() {
            return Err(format!(
                "SaeManifoldTerm::set_row_metric: metric has {} rows but term has {}",
                metric.n_rows(),
                self.n_obs()
            ));
        }
        if metric.p_out() != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::set_row_metric: metric output dim {} but term has {}",
                metric.p_out(),
                self.output_dim()
            ));
        }
        self.row_metric = Some(metric);
        Ok(())
    }

    /// The installed per-row metric, if any. `None` ⇒ Euclidean / isotropic.
    /// Consumed by the gauge wiring (to build the matching `WeightField`) and by
    /// Object 4 (to read the [`MetricProvenance`](crate::inference::row_metric::MetricProvenance)).
    pub fn row_metric(&self) -> Option<&crate::inference::row_metric::RowMetric> {
        self.row_metric.as_ref()
    }

    /// The per-row inner product the additive diagnostics read through: the
    /// installed [`RowMetric`](crate::inference::row_metric::RowMetric) when one
    /// was set (output-Fisher harvest present), otherwise a freshly-built
    /// Euclidean metric of the term's own `(n_obs, output_dim)` shape. Either way
    /// a metric always exists, so the diagnostics are never gated by a flag — the
    /// Euclidean fallback is the bit-identical isotropic path.
    fn diagnostic_metric(&self) -> Result<crate::inference::row_metric::RowMetric, String> {
        match self.row_metric() {
            Some(metric) => Ok(metric.clone()),
            None => {
                crate::inference::row_metric::RowMetric::euclidean(self.n_obs(), self.output_dim())
            }
        }
    }

    /// Build the additive post-fit diagnostic report for this fitted term: the
    /// two-score per-atom [`AtomTwoLensReport`](crate::inference::atom_lens::AtomTwoLensReport)
    /// (presence / behavioral coupling / discrepancy) and the residual-gauge
    /// [`ResidualGaugeReport`](crate::sae_identifiability::ResidualGaugeReport)
    /// certificate.
    ///
    /// Both reports are read through the same single metric
    /// ([`Self::diagnostic_metric`]): under a Euclidean / no-harvest provenance
    /// the lens coupling is `None` and the gauge is certified under Euclidean
    /// provenance — never an error, never gated by a flag (magic-by-default,
    /// mirroring the metric selection itself).
    ///
    /// `per_atom_ard_variances`, when supplied, is one ARD variance vector per
    /// atom (length = `latent_dim_k`), threaded into the certificate's
    /// equal-ARD-rotation detection. `None` (or a per-atom `None`) ⇒ no ARD prior
    /// on that atom. `isometry_pin_active` records whether an isometry gauge
    /// penalty was installed on the fit: `false` escalates the certificate to the
    /// `diffeomorphism-unpinned` verdict (the honest "no metric pin" statement),
    /// exactly as the certificate's own escalation flag specifies.
    ///
    /// Pure read: it never mutates the term, never touches a loss / criterion /
    /// penalty / optimizer state.
    pub fn fit_diagnostics_report(
        &self,
        per_atom_ard_variances: Option<&[Option<Array1<f64>>]>,
        isometry_pin_active: bool,
        reconstruction_dispersion: Option<f64>,
    ) -> Result<SaeManifoldFitDiagnostics, String> {
        let metric = self.diagnostic_metric()?;
        let atom_two_lens = crate::inference::atom_lens::atom_two_lens(self, &metric);

        let (certificate_model, streamed_curvature) =
            self.to_residual_gauge_model(metric, per_atom_ard_variances, isometry_pin_active)?;
        // #998: within-atom gauge families are certified on their EXACT orbits
        // in the model's own (decoder, coordinate) parameter space — compensated
        // symmetries are data-nulls by construction there, no lowering-error
        // calibration involved. This now holds whether or not an isometry pin is
        // active:
        //   * pin INACTIVE ⇒ the orbit verdict is the data residual alone (no
        //     penalty operator);
        //   * pin ACTIVE ⇒ the orbit verdict adds the isometry pin's orbit-space
        //     curvature through an [`OrbitPenaltyOperator`] lowered from the
        //     atom's second jet `Φ''` (the pullback-metric change along the orbit
        //     differentiates `J = Φ'B` through `t`). A model-class symmetry that
        //     preserves the metric stays a certified freedom; a non-isometric
        //     orbit (a basis not closed under the action) is genuinely pinned.
        // The relative-curvature fraction `cost/stiffness²` is invariant to the
        // pin strength μ (both faces scale with μ), so the operator is built at a
        // canonical unit weight. An atom whose basis exposes no analytic second
        // jet supplies no operator and falls back to the data residual — never an
        // error. Magic-by-default either way: the choice is derived from the fit,
        // never a flag.
        let views = self.atom_parameter_views();
        let ops: Vec<Option<crate::sae_identifiability::OrbitPenaltyOperator>> =
            if isometry_pin_active {
                views
                    .iter()
                    .map(|view| {
                        view.as_ref().and_then(|v| {
                            crate::sae_identifiability::isometry_orbit_penalty_operator(v, 1.0)
                        })
                    })
                    .collect()
            } else {
                (0..self.k_atoms()).map(|_| None).collect()
            };
        let residual_gauge = if isometry_pin_active {
            // The pin-active path consumes the per-row Jacobian curvature
            // directly (the certificate_model retains it under a pin), so route
            // through the non-streamed exact entry point.
            crate::sae_identifiability::residual_gauge_exact(&certificate_model, &views, &ops)?
        } else {
            let (curvature_gram, root_rows) = streamed_curvature.ok_or_else(|| {
                "fit_diagnostics_report: missing streamed residual-gauge curvature for unpinned exact path"
                    .to_string()
            })?;
            crate::sae_identifiability::residual_gauge_exact_from_curvature_gram(
                &certificate_model,
                &views,
                &ops,
                curvature_gram,
                root_rows,
            )?
        };

        Ok(SaeManifoldFitDiagnostics {
            atom_two_lens,
            residual_gauge,
            incoherence_report: match reconstruction_dispersion.or(self.certificate_dispersion) {
                Some(dispersion) => Some(dictionary_incoherence_report_with_dispersion(
                    self, dispersion,
                )?),
                None => None,
            },
        })
    }

    /// Build the trust-diagnostics producer for the Python `diagnostics` block.
    ///
    /// `assignments` is supplied by the payload assembly site so top-k projection,
    /// when requested, is reflected in coverage/frequency and in the tangent
    /// spectra. The active threshold is shared with the atom lens so all
    /// assignment-support diagnostics agree on what "active" means.
    pub fn trust_diagnostics_report(
        &self,
        assignments: ArrayView2<'_, f64>,
    ) -> Result<SaeTrustDiagnostics, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        if assignments.dim() != (n, k_atoms) {
            return Err(format!(
                "trust_diagnostics_report: assignments shape {:?} must be ({n}, {k_atoms})",
                assignments.dim()
            ));
        }
        if !assignments.iter().all(|v| v.is_finite()) {
            return Err("trust_diagnostics_report: assignments must be finite".to_string());
        }
        let metric = self.diagnostic_metric()?;
        let active_threshold = crate::inference::atom_lens::SAE_TRUST_ACTIVE_MASS_FLOOR;
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut atom_trust = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let mut active_token_count = 0usize;
            let mut activation_sum = 0.0_f64;
            for row in 0..n {
                let mass = assignments[[row, atom_idx]];
                activation_sum += mass;
                if mass > active_threshold {
                    active_token_count += 1;
                }
            }
            let coverage = if n > 0 {
                active_token_count as f64 / n as f64
            } else {
                0.0
            };
            let activation_frequency = if n > 0 {
                activation_sum / n as f64
            } else {
                0.0
            };
            let (sigma_min_tangent, sigma_max_tangent) = self
                .atom_tangent_spectrum_from_assignments(
                    atom_idx,
                    assignments,
                    &metric,
                    active_threshold,
                )?;
            let tangent_condition_score = if sigma_max_tangent > 0.0 {
                (sigma_min_tangent / sigma_max_tangent).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let trust_score = tangent_condition_score;
            atom_trust.push(trust_score);
            atoms.push(SaeAtomTrustDiagnostics {
                trust_score,
                sigma_min_tangent,
                sigma_max_tangent,
                tangent_condition_score,
                coverage,
                activation_frequency,
                untyped: matches!(atom.basis_kind, SaeAtomBasisKind::Precomputed(_)),
                active_token_count,
            });
        }
        Ok(SaeTrustDiagnostics { atom_trust, atoms })
    }

    fn atom_tangent_spectrum_from_assignments(
        &self,
        atom_idx: usize,
        assignments: ArrayView2<'_, f64>,
        metric: &crate::inference::row_metric::RowMetric,
        active_threshold: f64,
    ) -> Result<(f64, f64), String> {
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim;
        let p = self.output_dim();
        if d == 0 || p == 0 {
            return Ok((0.0, 0.0));
        }
        let mut gram = Array2::<f64>::zeros((d, d));
        let mut active_mass_sum = 0.0_f64;
        let mut jac_row = vec![0.0_f64; p * d];
        for row in 0..self.n_obs() {
            let mass = assignments[[row, atom_idx]];
            if !(mass > active_threshold) {
                continue;
            }
            active_mass_sum += mass;
            for axis in 0..d {
                let start = axis;
                let mut tangent = vec![0.0_f64; p];
                atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                for out in 0..p {
                    jac_row[out * d + start] = tangent[out];
                }
            }
            let row_pullback = metric.pullback(row, &jac_row, d);
            for axis_a in 0..d {
                for axis_b in 0..=axis_a {
                    gram[[axis_a, axis_b]] += mass * row_pullback[[axis_a, axis_b]];
                }
            }
            jac_row.fill(0.0);
        }
        if !(active_mass_sum > 0.0) {
            return Ok((0.0, 0.0));
        }
        let inv_mass = 1.0 / active_mass_sum;
        for axis_a in 0..d {
            for axis_b in 0..=axis_a {
                let value = gram[[axis_a, axis_b]] * inv_mass;
                gram[[axis_a, axis_b]] = value;
                gram[[axis_b, axis_a]] = value;
            }
        }
        let (evals, _) = gram.eigh(Side::Lower).map_err(|e| {
            format!(
                "trust_diagnostics_report: atom {atom_idx} tangent eigendecomposition failed: {e}"
            )
        })?;
        let mut sigma_min = f64::INFINITY;
        let mut sigma_max = 0.0_f64;
        for value in evals.iter().copied() {
            let clamped = value.max(0.0);
            let sigma = clamped.sqrt();
            sigma_min = sigma_min.min(sigma);
            sigma_max = sigma_max.max(sigma);
        }
        if sigma_min.is_finite() {
            Ok((sigma_min, sigma_max))
        } else {
            Ok((0.0, 0.0))
        }
    }

    /// Per-atom exact parameter-space views for the #998 certificate path:
    /// the basis values / first-derivative jet, decoder coefficients, latent
    /// coordinates, and assignment mass each atom was actually fitted with.
    /// Sphere atoms get `None` (their chart's group action is nonlinear, so
    /// the exact-orbit realisation does not apply and they stay on the frame
    /// path), as does any atom whose coordinate chart width disagrees with its
    /// latent dimension (a structurally inconsistent atom must not masquerade
    /// as exactly certified).
    fn atom_parameter_views(&self) -> Vec<Option<crate::sae_identifiability::AtomParameterView>> {
        let assignments = self.assignment.assignments();
        let n = self.n_obs();
        self.atoms
            .iter()
            .enumerate()
            .map(|(k, atom)| {
                if matches!(atom.basis_kind, SaeAtomBasisKind::Sphere) {
                    return None;
                }
                let coords = self.assignment.coords[k].as_matrix().to_owned();
                if coords.nrows() != n || coords.ncols() != atom.latent_dim {
                    return None;
                }
                let mut activations = Array1::<f64>::zeros(n);
                for row in 0..n {
                    activations[row] = assignments[[row, k]];
                }
                // Second jet Φ'' (#998): supplied when the atom's evaluator
                // exposes an analytic Hessian, so a pin-active fit can lower its
                // orbit-space isometry penalty operator (the metric-change of the
                // pullback gram differentiates Φ' through t). Absent ⇒ the orbit
                // verdict stays on the data residual / no-pin path, never an
                // error.
                let basis_second_jet = atom
                    .basis_evaluator
                    .as_ref()
                    .and_then(|evaluator| evaluator.second_jet_dyn(coords.view()))
                    .and_then(|res| res.ok());
                Some(crate::sae_identifiability::AtomParameterView {
                    basis_values: atom.basis_values.clone(),
                    basis_jacobian: atom.basis_jacobian.clone(),
                    decoder: atom.decoder_coefficients.clone(),
                    coords,
                    activations,
                    basis_second_jet,
                })
            })
            .collect()
    }

    /// Lower this fitted term into the self-contained
    /// [`FittedSaeManifold`](crate::sae_identifiability::FittedSaeManifold) the
    /// residual-gauge certificate consumes.
    ///
    /// The certificate's parameter space is the per-atom decoder **frame** — the
    /// `(output_dim, latent_dim)` image of the atom's latent axes in output space.
    /// We realise it as the active-mass-weighted mean decoder tangent
    /// `frame_k[:, a] = (Σ_n a_{nk} · ∂g_k/∂t_a(n)) / Σ_n a_{nk}` over the atom's
    /// active rows (the centroid decoder Jacobian columns the certificate docs
    /// name). The per-row pinning Jacobian block `J_n ∈ ℝ^{p × param_dim}` is the
    /// assignment-weighted per-row decoder tangent placed at each atom's frame
    /// slot: column `(k, i, a)` of `J_n` is `a_{nk} · ∂g_k/∂t_a(n)[i]` — exactly
    /// the directions the reconstruction data gives cost to, in the same metric
    /// the fit used (whitened by the certificate through `RowMetric`).
    ///
    /// The flattened frame layout matches the certificate's
    /// `vec(frame_0) ⊕ vec(frame_1) ⊕ …`, row-major within each frame
    /// (`frame_k[i, a]` at offset `atom_offset(k) + i·latent_dim_k + a`).
    fn to_residual_gauge_model(
        &self,
        metric: crate::inference::row_metric::RowMetric,
        per_atom_ard_variances: Option<&[Option<Array1<f64>>]>,
        isometry_pin_active: bool,
    ) -> Result<
        (
            crate::sae_identifiability::FittedSaeManifold,
            Option<(Array2<f64>, usize)>,
        ),
        String,
    > {
        use crate::sae_identifiability::{AtomTopology, FittedAtom, FittedSaeManifold};

        let n = self.n_obs();
        let p = self.output_dim();
        let k = self.k_atoms();
        let assignments = self.assignment.assignments();

        // Per-atom frame `(p, d)` = active-mass-weighted mean decoder tangent,
        // and the flattened-frame column offset bookkeeping for the joint
        // parameter vector (`vec(frame_0) ⊕ …`, row-major within each frame).
        let mut fitted_atoms: Vec<FittedAtom> = Vec::with_capacity(k);
        let mut atom_offsets: Vec<usize> = Vec::with_capacity(k);
        let mut atom_axis_dim: Vec<usize> = Vec::with_capacity(k);
        let mut cursor = 0usize;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let d = atom.latent_dim;
            let topology = match (&atom.basis_kind, d) {
                (SaeAtomBasisKind::Periodic, 1) | (SaeAtomBasisKind::Torus, 1) => {
                    AtomTopology::Circle
                }
                (SaeAtomBasisKind::Periodic, _) | (SaeAtomBasisKind::Torus, _) => {
                    AtomTopology::Torus { latent_dim: d }
                }
                (SaeAtomBasisKind::Sphere, _) => AtomTopology::Sphere,
                (
                    SaeAtomBasisKind::Duchon
                    | SaeAtomBasisKind::EuclideanPatch
                    | SaeAtomBasisKind::Precomputed(_),
                    _,
                ) => AtomTopology::EuclideanPatch { latent_dim: d },
            };

            let mut frame = Array2::<f64>::zeros((p, d));
            let mut active_mass = 0.0_f64;
            let mut tangent = vec![0.0_f64; p];
            for row in 0..n {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                active_mass += a_nk;
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        frame[[i, axis]] += a_nk * tangent[i];
                    }
                }
            }
            if active_mass > 0.0 {
                let inv = 1.0 / active_mass;
                frame.mapv_inplace(|v| v * inv);
            }

            // #995 lowering-error scale: mass-weighted relative dispersion of
            // the per-row tangents around the mean frame just built,
            //   Σ_n a_n Σ_ax ‖t_ax(n) − frame[:,ax]‖² / Σ_n a_n Σ_ax ‖t_ax(n)‖².
            // 0 ⇒ the frame represents every active row exactly (flat
            // decoder); → 1 ⇒ the tangent field disperses so strongly (e.g. a
            // full circle, whose tangents average out) that the mean-frame
            // compression cannot distinguish gauge motion from curvature. The
            // certificate calibrates its per-generator verdict tolerance to
            // this scale so it never claims a pin it cannot resolve.
            let mut disp_num = 0.0_f64;
            let mut disp_den = 0.0_f64;
            for row in 0..n {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        let dev = tangent[i] - frame[[i, axis]];
                        disp_num += a_nk * dev * dev;
                        disp_den += a_nk * tangent[i] * tangent[i];
                    }
                }
            }
            let lowering_error = if disp_den > 0.0 {
                (disp_num / disp_den).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let ard_variances = per_atom_ard_variances
                .and_then(|all| all.get(atom_idx))
                .and_then(|opt| opt.clone())
                .filter(|v| v.len() == d);

            fitted_atoms.push(FittedAtom {
                name: atom.name.clone(),
                topology,
                frame,
                ard_variances,
                lowering_error,
                // #1019: post-fit chart canonicalization (arc length for
                // d = 1, isometry-flow for d = 2 torus) pins the chart; the
                // certificate downgrades this atom's chart freedom to the
                // finite isometry group with PinnedByCanonicalization
                // provenance.
                chart_canonicalized: atom.chart_canonicalized
                    && (d == 1 || (d == 2 && matches!(atom.basis_kind, SaeAtomBasisKind::Torus))),
            });
            atom_offsets.push(cursor);
            atom_axis_dim.push(d);
            cursor += p * d;
        }
        let param_dim = cursor;

        // Per-row pinning Jacobian `J_n ∈ ℝ^{p × param_dim}` flattened row-major
        // (`J_n[i, c] = jacobian_rows[n][i · param_dim + c]`). Column `(k, i', a)`
        // of `J_n` is `a_{nk} · ∂g_k/∂t_a(n)[i']` placed at the atom-k frame slot
        // and read out on output coordinate `i = i'` (a frame perturbation of
        // output `i'` moves only the row's output coordinate `i'`).
        //
        // The pinned certificate still consumes the legacy row-block contract.
        // The unpinned exact path consumes only `RᵀR`, so stream each transient
        // row Jacobian through the metric whitening and discard it immediately.
        let (jacobian_rows, streamed_curvature) = if isometry_pin_active {
            let mut jacobian_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut tangent = vec![0.0_f64; p];
            for row in 0..n {
                let mut j_flat = vec![0.0_f64; p * param_dim];
                for (atom_idx, atom) in self.atoms.iter().enumerate() {
                    let a_nk = assignments[[row, atom_idx]];
                    if !(a_nk > 0.0) {
                        continue;
                    }
                    let d = atom_axis_dim[atom_idx];
                    let base = atom_offsets[atom_idx];
                    for axis in 0..d {
                        atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                        for i in 0..p {
                            // Frame coordinate `(k, i, axis)` sits at column
                            // `base + i·d + axis`; it sources output coordinate `i`.
                            j_flat[i * param_dim + base + i * d + axis] += a_nk * tangent[i];
                        }
                    }
                }
                jacobian_rows.push(j_flat);
            }
            (jacobian_rows, None)
        } else {
            let streamed = self.residual_gauge_streamed_data_curvature(
                &metric,
                &atom_offsets,
                &atom_axis_dim,
                param_dim,
            )?;
            (Vec::new(), Some(streamed))
        };

        // Isometry-penalty curvature root over the frame parameter space. When
        // the isometry gauge pin is active it gives curvature along every fitted
        // frame direction (it resists deviation of the decoder image from its
        // arc-length parameterization), so its row space is the span of the
        // per-atom frame columns: one root row per `(k, axis)` carrying that
        // atom's frame column at the atom's frame slot. Empty (`0 × param_dim`)
        // when the pin is inactive — exactly the certificate's escalation
        // condition to `diffeomorphism-unpinned`.
        let isometry_penalty_root = if isometry_pin_active && param_dim > 0 {
            let mut root_rows: Vec<Array1<f64>> = Vec::new();
            for (atom_idx, fitted) in fitted_atoms.iter().enumerate() {
                let d = atom_axis_dim[atom_idx];
                let base = atom_offsets[atom_idx];
                for axis in 0..d {
                    let mut r = Array1::<f64>::zeros(param_dim);
                    let mut any = false;
                    for i in 0..p {
                        let v = fitted.frame[[i, axis]];
                        if v != 0.0 {
                            any = true;
                        }
                        r[base + i * d + axis] = v;
                    }
                    if any {
                        root_rows.push(r);
                    }
                }
            }
            let mut root = Array2::<f64>::zeros((root_rows.len(), param_dim));
            for (ri, r) in root_rows.iter().enumerate() {
                root.row_mut(ri).assign(r);
            }
            root
        } else {
            Array2::<f64>::zeros((0, param_dim))
        };

        Ok((
            FittedSaeManifold {
                atoms: fitted_atoms,
                jacobian_rows,
                isometry_penalty_root,
                metric,
            },
            streamed_curvature,
        ))
    }

    fn residual_gauge_streamed_data_curvature(
        &self,
        metric: &crate::inference::row_metric::RowMetric,
        atom_offsets: &[usize],
        atom_axis_dim: &[usize],
        param_dim: usize,
    ) -> Result<(Array2<f64>, usize), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if metric.p_out() != p {
            return Err(format!(
                "residual_gauge_streamed_data_curvature: metric output dim {} but term has {p}",
                metric.p_out()
            ));
        }
        let rank = metric.metric_rank();
        let mut gram = Array2::<f64>::zeros((param_dim, param_dim));
        if param_dim == 0 || n == 0 || rank == 0 {
            return Ok((gram, n * rank));
        }

        let assignments = self.assignment.assignments();
        let mut tangent = vec![0.0_f64; p];
        let mut j_flat = vec![0.0_f64; p * param_dim];
        let mut root_row = Array1::<f64>::zeros(param_dim);
        for row in 0..n {
            j_flat.fill(0.0);
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_nk = assignments[[row, atom_idx]];
                if !(a_nk > 0.0) {
                    continue;
                }
                let d = atom_axis_dim[atom_idx];
                let base = atom_offsets[atom_idx];
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        j_flat[i * param_dim + base + i * d + axis] += a_nk * tangent[i];
                    }
                }
            }

            if metric.drives_gauge() {
                for r in 0..rank {
                    root_row.fill(0.0);
                    for c in 0..param_dim {
                        let mut acc = 0.0_f64;
                        for i in 0..p {
                            acc += metric.factor_entry(row, i, r) * j_flat[i * param_dim + c];
                        }
                        root_row[c] = acc;
                    }
                    let row_slice = root_row.as_slice().ok_or_else(|| {
                        "residual_gauge_streamed_data_curvature: non-contiguous root row"
                            .to_string()
                    })?;
                    Self::accumulate_residual_gauge_gram_row(&mut gram, row_slice);
                }
            } else {
                for i in 0..p {
                    let start = i * param_dim;
                    let end = start + param_dim;
                    Self::accumulate_residual_gauge_gram_row(&mut gram, &j_flat[start..end]);
                }
            }
        }

        for a in 0..param_dim {
            for b in 0..a {
                gram[[b, a]] = gram[[a, b]];
            }
        }
        Ok((gram, n * rank))
    }

    fn accumulate_residual_gauge_gram_row(gram: &mut Array2<f64>, row: &[f64]) {
        for a in 0..row.len() {
            let va = row[a];
            if va == 0.0 {
                continue;
            }
            for b in 0..=a {
                let vb = row[b];
                if vb != 0.0 {
                    gram[[a, b]] += va * vb;
                }
            }
        }
    }

    pub fn set_temperature_schedule(
        &mut self,
        sched: GumbelTemperatureSchedule,
    ) -> Result<(), String> {
        sched.validate()?;
        self.assignment
            .mode
            .set_temperature(sched.current_tau(sched.iter_count))?;
        self.temperature_schedule = Some(sched);
        Ok(())
    }

    fn advance_temperature_schedule(&mut self) -> Result<Option<f64>, String> {
        let Some(schedule) = self.temperature_schedule.as_mut() else {
            return Ok(None);
        };
        schedule.validate()?;
        let tau = schedule.step();
        self.assignment.mode.set_temperature(tau)?;
        Ok(Some(tau))
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Auto-derived in-core vs streaming plan for SAE Arrow-Schur work.
    ///
    /// This is intentionally not user-configurable: the route follows the
    /// retained full-batch working-set estimate and the currently selected GPU
    /// memory budget when CUDA is usable, otherwise a conservative host budget.
    pub fn streaming_plan(&self) -> SaeStreamingPlan {
        let n_obs = self.n_obs();
        let total_basis: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = self
            .atoms
            .iter()
            .map(|atom| atom.latent_dim)
            .max()
            .unwrap_or(0);
        let border_dim = if self.any_frame_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        sae_streaming_plan_for_shape(n_obs, total_basis, self.k_atoms(), d_max, border_dim)
    }

    /// Construction-time validation: every Psi-tier analytic penalty in the
    /// registry must be dispatchable into the SAE arrow-Schur row layout.
    ///
    /// Two invariants are enforced upfront so the dispatch loop in
    /// `add_sae_analytic_penalty_contributions` is total (no runtime
    /// "unsupported penalty" fallthrough, no per-call K-gating):
    ///
    /// 1. Every Psi-tier penalty is either in [`sae_penalty_is_row_block_supported`],
    ///    or `NuclearNorm` (which is redirected to the per-atom decoder (β) block
    ///    rather than the coord "t" row block). Assignment sparsity penalties
    ///    (`IBPAssignment`, `SoftmaxAssignmentSparsity`) are refused because the SAE
    ///    term already owns them through its built-in assignment path
    ///    (`loss.assignment_sparsity`). Penalty kinds with cross-row structure
    ///    (`TotalVariation`, `Monotonicity`, `BlockSparsity`,
    ///    `IvaeRidgeMeanGauge`, `Orthogonality`, `NestedPrefix`,
    ///    `SheafConsistency`) cannot be expressed in the SAE row-block layout
    ///    and are refused here.
    ///
    /// 2. If any Psi-tier row-block penalty is present, every atom shares
    ///    the same coord latent dim. The current registry model carries one
    ///    `latent_dim` per descriptor (the "t" latent block declares one
    ///    `d` value); per-atom dispatch with heterogeneous `d_k` would
    ///    require per-atom registry entries or per-kind in-place
    ///    reshaping. Mixed-d row-block fits are rejected with an actionable
    ///    error pointing at the configuration mismatch.
    ///
    /// The K=1 case trivially satisfies (2). Beta-tier and rho-tier
    /// penalties are not constrained here.
    fn validate_analytic_penalty_registry(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<(), String> {
        let mut row_block_penalty_present = false;
        for penalty in &registry.penalties {
            if penalty.tier() != PenaltyTier::Psi {
                continue;
            }
            if matches!(
                penalty,
                AnalyticPenaltyKind::IBPAssignment(_)
                    | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            ) {
                return Err(format!(
                    "SAE-manifold term refuses analytic penalty {:?}: assignment sparsity \
                     is owned by the built-in SAE assignment path (loss.assignment_sparsity). \
                     Registering it would double-count the objective and gradient",
                    penalty.name()
                ));
            }
            // NuclearNorm is redirected to the per-atom decoder (β) block in
            // `add_sae_beta_penalty` (it penalizes each atom's decoder matrix
            // singular spectrum, i.e. its embedding rank), so it bypasses the
            // coord "t" row-block requirement below.
            if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) {
                continue;
            }
            if !sae_penalty_is_row_block_supported(penalty) {
                return Err(format!(
                    "SAE-manifold term refuses analytic penalty {:?}: this kind \
                     has cross-row structure and cannot be expressed in the \
                     arrow-Schur row layout. Use only row-block-supported \
                     coord penalties (ARD, BlockOrthogonality, \
                     Sparsity/TopK/JumpReLU, RowPrecisionPrior, \
                     ParametricRowPrecisionPrior, ScadMcp, Isometry) on the \
                     coord latent block, or move the penalty to a non-SAE \
                     term",
                    penalty.name()
                ));
            }
            row_block_penalty_present = true;
        }
        if row_block_penalty_present {
            let mut dims = self.assignment.coords.iter().map(|c| c.latent_dim());
            if let Some(first) = dims.next() {
                if let Some(mismatch) = dims.find(|d| *d != first) {
                    return Err(format!(
                        "SAE-manifold term refuses row-block analytic penalty: \
                         atoms have heterogeneous coord latent dims (saw {first} \
                         and {mismatch}). Row-block penalties (ARD, \
                         BlockOrthogonality, ...) target the unified \"t\" \
                         latent block whose declared `d` matches one shape; \
                         per-atom dispatch with mixed `d_k` would silently \
                         truncate or expand axes. Configure all atoms with the \
                         same `atom_dim`, or split the row-block penalty into \
                         per-atom descriptors keyed to per-atom latent blocks"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn output_dim(&self) -> usize {
        self.atoms[0].output_dim()
    }

    pub fn beta_dim(&self) -> usize {
        let p = self.output_dim();
        self.atoms.iter().map(|a| a.basis_size() * p).sum()
    }

    fn take_border_hbb_workspace(&mut self, border_dim: usize) -> Array2<f64> {
        let mut workspace =
            std::mem::replace(&mut self.border_hbb_workspace, Array2::<f64>::zeros((0, 0)));
        if workspace.dim() != (border_dim, border_dim) {
            workspace = Array2::<f64>::zeros((border_dim, border_dim));
        } else {
            workspace.fill(0.0);
        }
        workspace
    }

    fn reclaim_border_hbb_workspace(&mut self, sys: &mut ArrowSchurSystem) {
        let workspace = std::mem::replace(&mut sys.hbb, Array2::<f64>::zeros((0, 0)));
        self.border_hbb_workspace = workspace;
    }

    /// Factored arrow-Schur border dimension `Σ_k M_k · r_k` (issue #972): the
    /// number of decoder coordinates the border actually carries once the
    /// low-rank Grassmann frames are profiled out. Atoms with no active frame
    /// contribute their full `M_k · p` (`r_k == p`), so on the all-full-`B` path
    /// this equals [`Self::beta_dim`]. The border Cholesky / evidence log-det
    /// scale with THIS count, not `beta_dim`.
    pub fn factored_border_dim(&self) -> usize {
        self.atoms.iter().map(|a| a.border_coeff_count()).sum()
    }

    /// Total profiled-out Grassmann manifold dimension `Σ_k r_k·(p − r_k)` across
    /// all active frames (issue #972). This is the count of decoder-frame degrees
    /// of freedom estimated OUTSIDE the border by closed-form polar steps, and it
    /// must enter the Laplace evidence dimension accounting (evidence honesty):
    /// the profiled frame is a MAP point on `∏_k Gr(r_k, p)`, contributing this
    /// many free dimensions to the model. `0` when every atom is on the full-`B`
    /// path. Threaded into [`Self::reml_occam_term`].
    pub fn grassmann_evidence_dimension(&self) -> usize {
        self.atoms
            .iter()
            .map(|a| a.frame_manifold_dimension())
            .sum()
    }

    /// True iff any atom has an active low-rank Grassmann frame (issue #972).
    pub fn frames_active(&self) -> bool {
        self.atoms.iter().any(|a| a.decoder_frame.is_some())
    }

    /// Alias of [`Self::frames_active`] (issue #972 / #977 T1): the predicate the
    /// assembly / step-lift branch on to decide whether the β-tier is built in
    /// the factored coordinate layout. Named to read as the question
    /// "is the factored path engaged?" at its call sites.
    pub fn any_frame_active(&self) -> bool {
        self.frames_active()
    }

    /// Per-atom column offsets of the *factored* border (issue #972 / #977 T1):
    /// the running prefix sum of `M_k · r_k`, one entry per atom (the same
    /// convention as [`Self::beta_offsets`]). This is the start of each atom's
    /// `C_k` block in the reduced border vector; on the all-full-`B` path it
    /// equals `beta_offsets`. Distinct from [`Self::factored_border_offsets`]
    /// only in name (both compute the identical prefix sum) — this method is the
    /// one the frame transform reads, mirroring `beta_offsets` at the call site.
    pub fn factored_beta_offsets(&self) -> Vec<usize> {
        self.factored_border_offsets()
    }

    /// Frame output matrix `U_k ∈ St(p, r_k)` for atom `k` (issue #972 / #977 T1).
    /// Returns the active frame `U_k` (`p × r_k`) when atom `k` is framed, else
    /// the identity `I_p` (the `r_k == p`, `U_k == I_p` full-`B` special case) so
    /// the projection / lift code is uniform across a mixed dictionary.
    pub fn frame_output_matrix(&self, atom_idx: usize) -> Array2<f64> {
        let atom = &self.atoms[atom_idx];
        match &atom.decoder_frame {
            Some(frame) => frame.frame().to_owned(),
            None => Array2::<f64>::eye(atom.output_dim()),
        }
    }

    /// Per-pair frame factor `W_{ij} = U_iᵀ U_j` (`r_i × r_j`) used as the output
    /// factor of the factored data β-Hessian block `G_{ij} ⊗ W_{ij}` (issue #972
    /// / #977 T1). When both atoms are framed this is the dense principal-angle
    /// cosine matrix between the two frames; for `i == j` with an orthonormal
    /// frame it is exactly `I_{r_i}`; for any un-framed atom the corresponding
    /// `U` is `I_p`, so a same-atom un-framed pair gives `I_p` (the clean full-`B`
    /// `G ⊗ I_p` collapse) and a framed/un-framed cross pair gives the rectangular
    /// `U_iᵀ` / `U_j` overlap.
    pub fn frame_cross_factor(&self, atom_i: usize, atom_j: usize) -> Array2<f64> {
        let ui = self.frame_output_matrix(atom_i);
        let uj = self.frame_output_matrix(atom_j);
        // `U_iᵀ U_j`: `(r_i × p) · (p × r_j)`. `fast_atb` forms `U_iᵀ U_j` directly.
        fast_atb(&ui, &uj)
    }

    /// Per-atom column offsets of the *factored* border (issue #972): the
    /// running prefix sum of `M_k · r_k`. The analogue of [`Self::beta_offsets`]
    /// for the reduced coordinate layout — atom `k`'s `C_k` occupies
    /// `[factored_border_offsets()[k] .. + M_k·r_k)`. On the full-`B` path this
    /// equals `beta_offsets`.
    pub fn factored_border_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            out.push(cursor);
            cursor += atom.border_coeff_count();
        }
        out
    }

    /// Assemble the factored border coordinate vector `C = [vec(C_1); …; vec(C_K)]`
    /// in row-major `C_k[m, j] → C[off_k + m·r_k + j]` layout (issue #972).
    ///
    /// This is the reduced state the arrow-Schur border carries when frames are
    /// active: its length is [`Self::factored_border_dim`] (`Σ M_k·r_k`), the
    /// border-size invariant verified by [`grassmann_assert_border_dim_invariant`].
    /// Atoms
    /// without an active frame contribute their full `vec(B_k)` (their `r_k == p`
    /// coordinates are the decoder itself), so on the all-full-`B` path this
    /// reproduces [`Self::flatten_beta`].
    pub fn flatten_factored_border(&self) -> Result<Array1<f64>, String> {
        let offsets = self.factored_border_offsets();
        let mut out = Array1::<f64>::zeros(self.factored_border_dim());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let off = offsets[atom_idx];
            let r = atom.border_frame_rank();
            let m = atom.basis_size();
            let coords = match atom.factored_coordinates()? {
                Some(c) => c,
                // Full-`B` path: the decoder itself is the coordinate matrix.
                None => atom.decoder_coefficients.clone(),
            };
            for basis_col in 0..m {
                for j in 0..r {
                    out[off + basis_col * r + j] = coords[[basis_col, j]];
                }
            }
        }
        Ok(out)
    }

    /// Scatter a factored border coordinate vector `C` (length
    /// [`Self::factored_border_dim`]) back into the per-atom decoders, refreshing
    /// each `decoder_coefficients = C_k · U_kᵀ` so the full-`B` consumers stay
    /// consistent after a factored border solve (issue #972). The inverse of
    /// [`Self::flatten_factored_border`].
    pub fn scatter_factored_border(&mut self, border: ArrayView1<'_, f64>) -> Result<(), String> {
        let expected = self.factored_border_dim();
        if border.len() != expected {
            return Err(format!(
                "SaeManifoldTerm::scatter_factored_border: border length {} must equal \
                 factored border dim {expected}",
                border.len()
            ));
        }
        let offsets = self.factored_border_offsets();
        for atom_idx in 0..self.atoms.len() {
            let off = offsets[atom_idx];
            let (r, m, has_frame) = {
                let atom = &self.atoms[atom_idx];
                (
                    atom.border_frame_rank(),
                    atom.basis_size(),
                    atom.decoder_frame.is_some(),
                )
            };
            let mut coords = Array2::<f64>::zeros((m, r));
            for basis_col in 0..m {
                for j in 0..r {
                    coords[[basis_col, j]] = border[off + basis_col * r + j];
                }
            }
            if has_frame {
                self.atoms[atom_idx].set_factored_coordinates(coords.view())?;
            } else {
                // Full-`B` path: the coordinates ARE the decoder.
                self.atoms[atom_idx].decoder_coefficients = coords;
            }
        }
        Ok(())
    }

    /// Auto-derive and install low-rank Grassmann decoder frames across all
    /// atoms (issue #972) — magic-by-default, no flag. Each atom independently
    /// activates its frame iff the factorization materially shrinks its border
    /// (see [`SaeManifoldAtom::maybe_activate_decoder_frame`]). Returns the
    /// number of atoms that activated a frame. Idempotent: re-running re-derives
    /// each frame from the current decoder.
    ///
    /// The decision keys on the *frontier* regime the issue targets: at large
    /// ambient `p` the full border `Σ M_k · p` reaches `10^7`–`10^8` and the
    /// border Cholesky dies, while the decoder's effective column rank `r` stays
    /// `≪ p`. Small-`p` atoms (where `r` cannot beat the activation margin)
    /// keep the bit-for-bit full-`B` path, so the small-model evidence is
    /// unchanged (verified by `factored_evidence_matches_full_b_at_small_p`).
    pub fn auto_activate_decoder_frames(&mut self) -> Result<usize, String> {
        let mut activated = 0usize;
        for atom in &mut self.atoms {
            let expected_rank = atom.decoder_frame_activation_rank()?;
            match (
                expected_rank,
                atom.decoder_frame.as_ref().map(GrassmannFrame::rank),
            ) {
                (Some(expected), Some(current)) if expected == current => {
                    continue;
                }
                (None, Some(_)) => {
                    atom.deactivate_decoder_frame();
                    continue;
                }
                (None, None) => {
                    continue;
                }
                (Some(_), _) => {}
            }
            if atom.maybe_activate_decoder_frame()?.is_some() {
                activated += 1;
            }
        }
        Ok(activated)
    }

    /// Reconcile decoder-frame activation before a fit entry point. The
    /// user-facing `auto_activate_decoder_frames` contract returns only newly
    /// installed frames; this helper enforces the stronger invariant the large-p
    /// solver needs: every atom whose current decoder satisfies the activation
    /// predicate has an active frame after the pass.
    fn ensure_decoder_frames_active_for_current_decoder(&mut self) -> Result<(), String> {
        self.auto_activate_decoder_frames()?;
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let expected_rank = atom.decoder_frame_activation_rank()?;
            if let Some(expected_rank) = expected_rank {
                match atom.decoder_frame.as_ref() {
                    Some(frame) if frame.rank() == expected_rank => {}
                    Some(frame) => {
                        return Err(format!(
                            "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                             atom {atom_idx} frame rank {} must equal audited rank {expected_rank}",
                            frame.rank()
                        ));
                    }
                    None => {
                        return Err(format!(
                            "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                             atom {atom_idx} has audited rank {expected_rank} but no active frame"
                        ));
                    }
                }
            } else if atom.decoder_frame.is_some() {
                return Err(format!(
                    "SaeManifoldTerm::ensure_decoder_frames_active_for_current_decoder: \
                     atom {atom_idx} kept a frame after the full-B predicate won"
                ));
            }
        }
        Ok(())
    }

    /// Closed-form streaming POLAR refresh of every ACTIVE decoder frame from the
    /// current data evidence (issue #972 / #977 T1) — the U-block of the
    /// alternating block-coordinate ascent that complements the border's
    /// C-block Newton step.
    ///
    /// For each framed atom `k` we accumulate the `p × r_k` cross-moment
    ///   `A_k = Σ_n a_{n,k} · e_{n,k} · ĉ_{n,k}ᵀ`,
    /// where `e_{n,k} = z_n − Σ_{k'≠k} a_{n,k'}·decoded_{k'}(n)` is the row's
    /// partial reconstruction residual (everything except atom `k`) and
    /// `ĉ_{n,k} = Φ_k(t_n)·C_k ∈ ℝ^{r_k}` is atom `k`'s in-span decoded
    /// coordinate. The polar factor `U_new = polar(A_k)` is the closed-form MAP
    /// frame on `Gr(r_k, p)` given the C-coordinates held fixed — the same
    /// `O(p r²)` thin SVD the issue prescribes, run OUTSIDE the border. The frame
    /// is then re-installed and the decoder re-projected onto it so the
    /// authoritative `B_k = C_k U_newᵀ` and the `(C_k, U_new)` pair stay
    /// consistent (a no-op in span for a truly rank-`r` atom). Un-framed atoms
    /// are skipped. Returns the number of frames refreshed.
    fn refresh_active_frames_from_data(
        &mut self,
        target: ArrayView2<'_, f64>,
    ) -> Result<usize, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        if n == 0 {
            return Ok(0);
        }
        // Per-row assignments and per-(row, atom) decoded outputs, computed once.
        let mut assignments = Vec::with_capacity(n);
        for row in 0..n {
            assignments.push(self.assignment.try_assignments_row(row)?);
        }
        let mut decoded = Array3::<f64>::zeros((n, k_atoms, p));
        let mut dbuf = vec![0.0_f64; p];
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut dbuf);
                for c in 0..p {
                    decoded[[row, atom_idx, c]] = dbuf[c];
                }
            }
        }
        // Full fitted reconstruction `Σ_k a_k decoded_k`, so the per-atom partial
        // residual is `e_k = (z − fitted) + a_k decoded_k` (add atom k back in).
        let mut fitted = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for atom_idx in 0..k_atoms {
                let a = assignments[row][atom_idx];
                if a == 0.0 {
                    continue;
                }
                for c in 0..p {
                    fitted[[row, c]] += a * decoded[[row, atom_idx, c]];
                }
            }
        }
        let mut refreshed = 0usize;
        for atom_idx in 0..k_atoms {
            // Only atoms with an active frame are refreshed.
            let Some(coords_c) = self.atoms[atom_idx].factored_coordinates()? else {
                continue;
            };
            let r = self.atoms[atom_idx].border_frame_rank();
            let m = self.atoms[atom_idx].basis_size();
            // Accumulate `A_k = Σ_n a_k · e_{n,k} · ĉ_{n,k}ᵀ` directly (p × r).
            let mut cross = GrassmannCrossMoment::new(p, r);
            // Build per-row p-target `a_k·e_k` and r-coord `a_k·ĉ` batched, then
            // accumulate as one outer-product sum. `accumulate` forms
            // `targetsᵀ·coords`, so scaling EITHER side by `a_k` once gives the
            // `a_k²` weight on the cross-moment that matches the C-block normal
            // equations (residual leg carries `a_k`, coordinate leg carries
            // `a_k`).
            let mut targets = Array2::<f64>::zeros((n, p));
            let mut rcoords = Array2::<f64>::zeros((n, r));
            for row in 0..n {
                let a = assignments[row][atom_idx];
                // Partial residual e_{n,k} = z_n − (fitted − a_k decoded_k).
                for c in 0..p {
                    let e = target[[row, c]] - fitted[[row, c]] + a * decoded[[row, atom_idx, c]];
                    targets[[row, c]] = a * e;
                }
                // In-span coordinate ĉ_{n,k} = Φ_k(t_n)·C_k ∈ ℝ^r.
                for j in 0..r {
                    let mut acc = 0.0_f64;
                    for basis_col in 0..m {
                        acc += self.atoms[atom_idx].basis_values[[row, basis_col]]
                            * coords_c[[basis_col, j]];
                    }
                    rcoords[[row, j]] = a * acc;
                }
            }
            cross.accumulate(targets.view(), rcoords.view())?;
            // `polar(A_k)` is well-defined only when the moment is non-trivial;
            // a zero moment (e.g. a fully collapsed atom) leaves the frame as-is.
            if cross.moment().iter().all(|&v| v == 0.0) {
                continue;
            }
            self.atoms[atom_idx].refresh_frame_from_cross_moment(cross.moment())?;
            refreshed += 1;
        }
        Ok(refreshed)
    }

    pub fn beta_offsets(&self) -> Vec<usize> {
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            out.push(cursor);
            cursor += atom.basis_size() * p;
        }
        out
    }

    /// Per-atom β column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Returns one `Range<usize>` per atom, covering that atom's decoder
    /// coefficients in the flat β vector:
    ///   `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
    ///
    /// Pass to [`ArrowSchurSystem::set_block_offsets`] so that
    /// [`crate::solver::arrow_schur::JacobiPreconditioner`] builds one dense
    /// Schur sub-block per atom instead of scalar-diagonal inversion.
    pub fn beta_block_offsets(&self) -> Arc<[std::ops::Range<usize>]> {
        let p = self.output_dim();
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            let width = atom.basis_size() * p;
            ranges.push(cursor..cursor + width);
            cursor += width;
        }
        Arc::from(ranges.into_boxed_slice())
    }

    /// Decide whether the sparse per-row active-set layout is engaged for the
    /// dense-weight assignment modes (softmax / IBP-MAP), and if so derive the
    /// per-row active-atom cap and magnitude cutoff.
    ///
    /// The decision is auto-derived from the problem size and the
    /// device/host working-set budget — never a CLI flag or kwarg. JumpReLU is
    /// not handled here (it always uses its structural gate via
    /// [`SaeRowLayout::from_jumprelu`]). The dense Gauss-Newton data Gram `G`
    /// is `(m_total × m_total)` f64; if its dense form fits the budget we keep
    /// the exact full-support solve (every atom active per row), so small-`K`
    /// problems are bit-for-bit unchanged. Above that, we cap each row to the
    /// `k_active` atoms that make the *sparse* Gram fit the same budget, with a
    /// relative magnitude cutoff that drops assignment mass contributing
    /// negligible `O(a²)` curvature.
    ///
    /// Returns `Some((k_active_cap, cutoff))` to engage sparsity, or `None` to
    /// keep the dense full-support layout.
    fn sparse_active_plan(&self) -> Option<(usize, f64)> {
        // Relative magnitude cutoff: assignment mass below this fraction of the
        // row's peak `|a_k|` enters the Gram only as `O(a²)` curvature and is
        // dropped. Chosen so dropped terms are ~1e-6 of the peak self-coupling.
        const RELATIVE_CUTOFF: f64 = 1.0e-3;

        let k_atoms = self.k_atoms();
        if k_atoms <= 1 {
            return None;
        }
        // The per-row tangent projection used for non-Euclidean atom latents
        // requires the uniform-`q` dense layout; the compact active-set path
        // skips it. So the effective (truncating) sparse plan only engages
        // when the ext-coord manifold is Euclidean. JumpReLU's structural gate
        // is exact and handled separately, so this restriction does not affect
        // it. At huge `K` on a curved manifold the streaming driver still
        // bounds memory; only the per-atom truncation is withheld.
        if !self.ext_coord_manifold().is_euclidean() {
            return None;
        }
        let p = self.output_dim();
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        // Dense data Gram footprint: (m_total · m_total) f64.
        let dense_gram_bytes = m_total
            .saturating_mul(m_total)
            .saturating_mul(SAE_BYTES_PER_F64);

        let budget = match crate::gpu::runtime::GpuRuntime::global() {
            // Allow up to one quarter of the AGGREGATE device budget for the dense
            // Gram, matching the streaming dispatcher's in-core fraction. The
            // per-atom-pair Gram blocks fan out across the whole device pool, so
            // the in-core fraction sums every ordinal's budget, not just the
            // primary's.
            Some(rt) => {
                let aggregate: usize = rt
                    .device_ordinals()
                    .iter()
                    .map(|&ord| rt.memory_budget_for(ord))
                    .sum();
                aggregate / 4
            }
            None => sae_host_in_core_budget_bytes().0,
        };
        if dense_gram_bytes <= budget {
            return None;
        }

        // Sparse Gram footprint scales with the per-row active basis count
        // `k_active · m_atom`. Solve for the largest `k_active` whose sparse
        // Gram `(k_active · m_atom)²` still fits the budget.
        let m_atom = (m_total as f64 / k_atoms as f64).max(1.0);
        let max_active_basis = ((budget as f64 / SAE_BYTES_PER_F64 as f64).sqrt() / m_atom).floor();
        let k_active_cap = (max_active_basis as usize).clamp(1, k_atoms);
        // p does not enter the Gram dimension (it is carried by the `⊗ I_p`
        // structure), but a degenerate `p == 0` term has no decoder columns.
        if p == 0 {
            return None;
        }
        Some((k_active_cap, RELATIVE_CUTOFF))
    }

    pub fn flatten_beta(&self) -> Array1<f64> {
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    out[off + basis_col * p + out_col] =
                        atom.decoder_coefficients[[basis_col, out_col]];
                }
            }
        }
        out
    }

    pub fn set_flat_beta(&mut self, beta: ArrayView1<'_, f64>) -> Result<(), String> {
        if beta.len() != self.beta_dim() {
            return Err(format!(
                "set_flat_beta: beta length {} != expected {}",
                beta.len(),
                self.beta_dim()
            ));
        }
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        for (atom_idx, atom) in self.atoms.iter_mut().enumerate() {
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            for basis_col in 0..m {
                for out_col in 0..p {
                    atom.decoder_coefficients[[basis_col, out_col]] =
                        beta[off + basis_col * p + out_col];
                }
            }
        }
        Ok(())
    }

    pub fn refit_decoder_least_squares_at_current_state(
        &mut self,
        target: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::refit_decoder_least_squares_at_current_state: target shape {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        let k_atoms = self.k_atoms();
        let offsets = self.beta_offsets();
        let m_total = self.beta_dim() / p;
        let mut design = Array2::<f64>::zeros((n, m_total));
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            for atom_idx in 0..k_atoms {
                let atom = &self.atoms[atom_idx];
                let weight = assignments[atom_idx];
                let m = atom.basis_size();
                let off = offsets[atom_idx] / p;
                for basis_col in 0..m {
                    design[[row, off + basis_col]] = weight * atom.basis_values[[row, basis_col]];
                }
            }
        }
        let beta = solve_design_least_squares(design.view(), target)?;
        if beta.dim() != (m_total, p) {
            return Err(format!(
                "SaeManifoldTerm::refit_decoder_least_squares_at_current_state: beta shape {:?} != ({m_total}, {p})",
                beta.dim()
            ));
        }
        for atom_idx in 0..k_atoms {
            let m = self.atoms[atom_idx].basis_size();
            let off = offsets[atom_idx] / p;
            for basis_col in 0..m {
                for out_col in 0..p {
                    self.atoms[atom_idx].decoder_coefficients[[basis_col, out_col]] =
                        beta[[off + basis_col, out_col]];
                }
            }
            self.atoms[atom_idx].refresh_intrinsic_smooth_penalty();
        }
        Ok(())
    }

    pub fn fitted(&self) -> Array2<f64> {
        self.try_fitted().expect("assignment logits must be finite")
    }

    pub fn try_fitted(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, p));
        // Reuse a single scratch buffer across all (row, atom) pairs instead of
        // allocating a fresh `Array1<f64>` of length p per call.
        let mut g_buf = vec![0.0_f64; p];
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?;
            for atom_idx in 0..k_atoms {
                self.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                let a_k = a[atom_idx];
                let mut out_row = out.row_mut(row);
                for out_col in 0..p {
                    out_row[out_col] += a_k * g_buf[out_col];
                }
            }
        }
        Ok(out)
    }

    pub fn loss(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<SaeManifoldLoss, String> {
        self.loss_scaled(target, rho, 1.0)
    }

    /// Penalized objective with a `penalty_scale` applied to the β-tier
    /// (decoder smoothness) penalty, mirroring
    /// [`Self::assemble_arrow_schur_scaled`]. The streaming line search sums
    /// per-chunk `loss_scaled(..., n_chunk / N)` so that the global smoothness
    /// penalty is counted exactly once across a pass while the per-row data,
    /// assignment-prior, and ARD terms sum naturally. `penalty_scale == 1.0`
    /// recovers the full-batch objective.
    pub fn loss_scaled(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        penalty_scale: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::loss_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::loss: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        let fitted = self.try_fitted()?;
        let mut data_fit = 0.0_f64;
        // The likelihood whitens through the RowMetric **only** when the metric
        // is a genuinely estimated noise model (`metric.whitens_likelihood()`,
        // i.e. `WhitenedStructured` — the #974 residual-covariance seam). For
        // Euclidean (default `None`) and for the OutputFisher *gauge* metric the
        // reconstruction data-fit stays the isotropic `0.5 * Σ r²`: a gauge /
        // output-Fisher inner product must NOT silently replace the
        // reconstruction loss with a Fisher pullback (#980). It only drives the
        // gauge (see `analytic_penalties::corrected_isometry_penalty`). The
        // producer of `WhitenedStructured` is
        // `inference::residual_factor::StructuredResidualModel::row_metric`; the
        // SAME metric whitens the assembled gradient/Hessian in
        // `assemble_arrow_schur` (the single #974 seam), so this value and that
        // gradient cannot desync. Without a whitening metric this path is
        // bit-for-bit the historical isotropic data-fit.
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut resid_row = ndarray::Array1::<f64>::zeros(target.ncols());
        // #991 design honesty weights: the reconstruction channel of row `i`
        // is weighted by `w_i` (mean-1 HT inclusion correction). The assembly
        // applies the same `w_i` via a `√w_i` scaling of the row residual /
        // Jacobian / β load at its single seam, so this value and that
        // gradient/Hessian carry the identical per-row factor. `None` ⇒ the
        // historical unweighted sum, bit-for-bit.
        let row_loss_w = self.row_loss_weights.as_deref();
        for row in 0..target.nrows() {
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for out_col in 0..target.ncols() {
                resid_row[out_col] = target[[row, out_col]] - fitted[[row, out_col]];
            }
            match self.row_metric.as_ref() {
                Some(metric) if whitens => {
                    for w in metric.whiten_residual_row(row, resid_row.view()) {
                        data_fit += 0.5 * w_row * w * w;
                    }
                }
                _ => {
                    for &r in resid_row.iter() {
                        data_fit += 0.5 * w_row * r * r;
                    }
                }
            }
        }
        let assignment_sparsity = assignment_prior_value(&self.assignment, rho);
        let smoothness = penalty_scale * self.decoder_smoothness_value(rho.lambda_smooth());
        let ard = self.ard_value(rho)?;
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            evidence_gauge_deflated_directions: 0,
        })
    }

    pub fn analytic_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
    ) -> Result<f64, ArrowSchurError> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "SaeManifoldTerm::analytic_penalty_value_total: penalty_scale must be finite \
                     and positive; got {penalty_scale}"
                ),
            });
        }
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            // Skip the registry `ARDPenalty` here for the same reason it is
            // skipped in `add_sae_analytic_penalty_contributions`: the coordinate
            // ARD energy is already counted by `loss.ard` (the von-Mises
            // `ard_value`), and the registry penalty's legacy Gaussian `½λt²` is
            // period-discontinuous. Including it would double-count the energy and
            // make this line-search objective jump across the branch cut while the
            // assembled gradient (von-Mises only, after the assembly fix) stays
            // continuous — i.e. a near-zero step would change the objective by a
            // finite amount and Armijo would wrongly reject it.
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            match tier {
                PenaltyTier::Psi => {
                    if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
                        for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                            value += penalty_scale
                                * per_atom.value(beta.slice(s![start..end]), rho_local);
                        }
                    } else {
                        if !sae_penalty_is_row_block_supported(penalty) {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "validate_analytic_penalty_registry should have refused \
                                     non-row-block Psi-tier penalty {:?} (registry layout name \
                                     {name:?})",
                                    penalty.name()
                                ),
                            });
                        }
                        for atom_idx in 0..self.k_atoms() {
                            let coord = &self.assignment.coords[atom_idx];
                            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                                let corrected_kind =
                                    self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                                value += corrected_kind.value(coord.as_flat().view(), rho_local);
                            } else if sae_coord_penalty_is_origin_anchored_magnitude(penalty) {
                                // Origin-anchored magnitude shrinkage (SCAD/MCP) is
                                // restricted to the Euclidean axes; periodic axes have
                                // no chart origin and would make this energy
                                // period-discontinuous (issue #795). This must mirror
                                // the gradient/curvature assembly in
                                // `add_sae_coord_penalty` exactly.
                                match sae_coord_penalty_euclidean_restriction(coord) {
                                    Some((_axes, compacted)) => {
                                        value += penalty.value(compacted.view(), rho_local);
                                    }
                                    None => {
                                        value += penalty.value(coord.as_flat().view(), rho_local);
                                    }
                                }
                            } else {
                                value += penalty.value(coord.as_flat().view(), rho_local);
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
                        if let Some(per_fit) = self.live_decoder_incoherence_penalty(base) {
                            value += penalty_scale * per_fit.value(beta.view(), rho_local);
                        }
                    } else if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
                        for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                            if start < end {
                                value += penalty_scale * per_atom.value(beta.view(), rho_local);
                            }
                        }
                    } else {
                        value += penalty_scale * penalty.value(beta.view(), rho_local);
                    }
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(value)
    }

    /// Energy of the decoder-block analytic penalties that have no native
    /// `SaeManifoldLoss` counterpart, evaluated at the current decoder `β` and
    /// the converged SAE state. These act on the per-atom decoder coefficient
    /// matrices: cross-atom decoder incoherence (#671), mechanism
    /// (feature-group) sparsity, and nuclear-norm embedding rank (#672). Each
    /// is injected with its live per-atom shape / co-activation before its
    /// value is taken, mirroring the assemble path.
    ///
    /// This is deliberately narrower than [`Self::analytic_penalty_value_total`]:
    /// it excludes the Psi-tier coordinate / assignment penalties (ARD,
    /// Isometry, ScadMcp, BlockOrthogonality, IBP/softmax assignment sparsity).
    /// The SAE already carries its own ARD (`loss.ard`) and assignment sparsity
    /// (`loss.assignment_sparsity`) energy, so adding the registry ARD /
    /// assignment value on top would double-count, and the gauge-only
    /// coordinate penalties are not part of the penalized deviance the
    /// REML/Laplace criterion scores. The decoder-block penalties, by contrast,
    /// are real penalized-energy terms with no `loss.*` representative: the
    /// inner solve minimizes them (they enter `gb`/`hbb`) but they were absent
    /// from the criterion scalar `v`. This restores that consistency so the
    /// ρ-sweep ranks the same objective the inner solve descends — the #671
    /// incoherence lever in particular now shapes model selection, not just the
    /// Newton step.
    ///
    /// NOTE: the coordinate-block penalties with no native `loss.*` twin
    /// (`ScadMcp`, `BlockOrthogonality`) carry the same residual inconsistency
    /// (scored in the line search via `penalized_objective_total`, absent from
    /// the REML scalar). They are left out here because they share a registry
    /// dispatch with the always-on `Isometry` gauge, whose inclusion in the
    /// topology-comparison criterion is a separate design question (#673:
    /// topology evidence is gauge-conditional). Folding the coord-tier energy in
    /// is tracked apart from this #671 decoder fix.
    pub fn analytic_decoder_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        // Resolve each penalty's rho slice exactly as `analytic_penalty_value_total`
        // does (registry-local rho at zeros), so a learnable decoder-penalty weight
        // is honoured rather than indexing into an empty view.
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, _tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match penalty {
                AnalyticPenaltyKind::DecoderIncoherence(base) => {
                    if let Some(per_fit) = self.live_decoder_incoherence_penalty(base) {
                        value += per_fit.value(beta.view(), rho_local);
                    }
                }
                AnalyticPenaltyKind::MechanismSparsity(base) => {
                    for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                        if start < end {
                            value += per_atom.value(beta.view(), rho_local);
                        }
                    }
                }
                AnalyticPenaltyKind::NuclearNorm(base) => {
                    for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                        value += per_atom.value(beta.slice(s![start..end]), rho_local);
                    }
                }
                _ => {}
            }
        }
        Ok(value)
    }

    /// Energy of the COORDINATE-tier isometry penalty(ies) at the converged
    /// SAE state. This is the per-atom `½μ Σ_n ‖J_n^T W_n J_n / gbar − g_ref‖²`
    /// summed over atoms, evaluated through `corrected_isometry_penalty` so the
    /// live decoder/coordinate caches drive the value exactly as the assemble
    /// path does. It has no `SaeManifoldLoss` twin (the loss carries only
    /// data-fit / assignment / smoothness / ARD), so the Laplace/REML criterion
    /// must add it explicitly to score the same penalized objective the inner
    /// solve descends.
    pub fn isometry_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let mut value = 0.0_f64;
        for (penalty, (rho_slice, _tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                let rho_local = rho_global.slice(s![rho_slice.clone()]);
                for atom_idx in 0..self.k_atoms() {
                    let coord = &self.assignment.coords[atom_idx];
                    let corrected_kind = self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                    value += corrected_kind.value(coord.as_flat().view(), rho_local);
                }
            }
        }
        Ok(value)
    }

    /// Extra analytic-penalty energy that has no native `SaeManifoldLoss`
    /// component but is part of the penalized objective ranked by the SAE
    /// Laplace/REML criterion.
    pub fn reml_extra_penalty_value_total(
        &self,
        registry: &AnalyticPenaltyRegistry,
    ) -> Result<f64, ArrowSchurError> {
        Ok(self.analytic_decoder_penalty_value_total(registry)?
            + self.isometry_penalty_value_total(registry)?)
    }

    pub fn penalized_objective_total(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<f64, String> {
        let mut total = self.loss_scaled(target, rho, penalty_scale)?.total();
        if let Some(analytic_registry) = registry {
            total += self
                .analytic_penalty_value_total(analytic_registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::penalized_objective_total: {err}"))?;
        }
        Ok(total)
    }

    fn decoder_smoothness_value(&self, lambda_smooth: f64) -> f64 {
        // Smoothness penalty value is `0.5·λ·Σ_oc B[:,oc]ᵀ S B[:,oc]`. Form the
        // `S·B` matrix product once per atom (O(M²·p)) and reduce against `B`
        // with a single O(M·p) Hadamard sum, instead of the previous
        // four-factor multiply-accumulate inside an `O(M²·p)` triple loop.
        // The quadratic form only sees the symmetric part of `S`, so reusing
        // the raw (un-symmetrised) `smooth_penalty` here is numerically
        // identical to the symmetrised assembly form.
        // Per-atom `S_k · B_k` products are independent across atoms, so they ride
        // the multi-GPU batched smoothness GEMM (uniform-shape groups tiled across
        // every device); `symmetrize = false` because the quadratic form only sees
        // the symmetric part of `S` regardless. Exact CPU fallback per atom.
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, false);
        let mut acc = 0.0;
        for (atom, sb) in self.atoms.iter().zip(sb_all.iter()) {
            acc += 0.5 * lambda_smooth * (&atom.decoder_coefficients * sb).sum();
        }
        acc
    }

    fn ard_value(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs();
        let mut acc = 0.0;
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            if rho.log_ard[atom_idx].is_empty() {
                continue;
            }
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            // Per-axis periodicity selects the smooth von-Mises energy on
            // wrapped (Circle) axes and the Gaussian on Euclidean axes.
            let periods = coord.effective_axis_periods();
            for axis in 0..d {
                let log_alpha = rho.log_ard[atom_idx][axis];
                // Clamp the log-precision before exponentiating: a raw
                // `exp(log_ard)` overflows to `inf` for `log_ard ≳ 709`, and the
                // `inf` precision then poisons the ARD energy / curvature with
                // `inf · 0.0 = NaN` (#742, Issue 4).
                let alpha = SaeManifoldRho::stable_exp_strength(log_alpha);
                let period = periods[axis];
                let mut energy = 0.0;
                for row in 0..n {
                    let v = coord.row(row)[axis];
                    energy += ArdAxisPrior::eval(alpha, v, period).value;
                }
                // Negative-log prior for precision alpha. The data-dependent
                // energy is the (Gaussian or von-Mises) coordinate prior; the
                // accompanying normaliser is the precision log-partition.
                //
                // Euclidean axes keep the Gaussian normaliser `-0.5 n log α`.
                // Periodic (von-Mises) axes use the EXACT von-Mises precision
                // log-partition `n[-η + log I0(η)]`, η = α/κ², κ = 2π/P, rather
                // than the Gaussian surrogate: the von-Mises partition function
                // is `2π I0(η)` (up to the κ Jacobian), so the per-observation
                // normaliser is `-η + log I0(η)` and is exact across the cut.
                match period {
                    None => {
                        acc += energy - 0.5 * (n as f64) * log_alpha;
                    }
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        let log_i0 = bessel_i0(eta).ln();
                        acc += energy + (n as f64) * (-eta + log_i0);
                    }
                }
            }
        }
        Ok(acc)
    }

    /// Assemble the enlarged `(logits, t)` row-local Arrow-Schur system.
    ///
    /// Full-batch entry point: a single chunk covering all rows, with the
    /// β-tier penalties (decoder smoothness, ARD, analytic β penalties) carrying
    /// their full strength. The streaming driver calls
    /// [`Self::assemble_arrow_schur_scaled`] directly with a `penalty_scale`
    /// equal to the minibatch fraction `n_chunk / N`, so that the sum of the
    /// per-chunk β-tier contributions over a full pass reconstructs exactly the
    /// single global β penalty (the smoothness/ARD/β terms are functions of `B`
    /// and the global coordinates, not of the chunk's rows).
    pub fn assemble_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled(target, rho, analytic_penalties, 1.0)
    }

    /// Assemble the row-local Arrow-Schur system with a `penalty_scale` applied
    /// to the β-tier (decoder smoothness, ARD prior, analytic β penalties).
    ///
    /// `penalty_scale == 1.0` recovers the full-batch assembly. The streaming
    /// driver passes the minibatch fraction `n_chunk / N` so that the β-tier
    /// reduced-Schur and gradient contributions of the chunks sum to exactly one
    /// global copy across a full pass (data-fit, assignment-prior, and per-row
    /// coord/logit analytic terms are *not* scaled — they are genuine per-row
    /// sums).
    pub fn assemble_arrow_schur_scaled(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target,
            rho,
            analytic_penalties,
            penalty_scale,
            SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM,
        )
    }

    fn assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
        dense_beta_penalty_probe_max_dim: usize,
    ) -> Result<ArrowSchurSystem, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: log_ard length {} != K {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let ard_len = rho.log_ard[atom_idx].len();
            let d = coord.latent_dim();
            if ard_len != 0 && ard_len != d {
                return Err(format!(
                    "SaeManifoldTerm::assemble_arrow_schur: log_ard atom {atom_idx} \
                     has len {ard_len}; expected 0 (disabled) or atom dim {d}"
                ));
            }
        }
        // Reparameterize each atom's roughness Gram into arc length at the
        // current decoder/coordinates (issue #673). This is the single
        // chokepoint for both the inner Newton assembly and the undamped
        // evidence factorization, so freezing the pullback-metric weight here
        // (lagged-diffusivity) keeps the smoothness value, gradient, Kronecker
        // Hessian, and REML log-det mutually consistent within each assembly
        // and makes the converged penalty — hence the topology evidence —
        // gauge-invariant. Constant-speed (periodic) atoms are unaffected.
        for atom in &mut self.atoms {
            atom.refresh_intrinsic_smooth_penalty();
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        let frame_projection = FrameProjection::new(self);
        let beta_offsets = frame_projection.beta_offsets.clone();
        let coord_offsets = self.assignment.coord_offsets();
        // β-tier decoder smoothness is a global (B-only) penalty; under a
        // minibatch pass it is scaled by the chunk fraction so the per-chunk
        // contributions sum to one global copy.
        let lambda_smooth = rho.lambda_smooth() * penalty_scale;
        let (assignment_grad, assignment_hdiag) =
            assignment_prior_grad_hdiag(&self.assignment, rho)?;

        // Decoder smoothness penalty: build one KroneckerPenaltyOp per atom
        // (structure = λ·S_k ⊗ I_p, offset = beta_offsets[k]) instead of
        // materialising the dense K×K block.  The gradient is a dense K-vector
        // accumulated into `smooth_grad_gb` and written into sys.gb after sys
        // is constructed (#296).
        let mut smooth_ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len());
        // #972 / #977 T1: retain each atom's symmetrised `λ S_k` (`M_k × M_k`) so
        // the frame transform can rebuild the smooth penalty in the factored
        // coordinate space as `λ S_k ⊗ I_{r_k}` (the `tr(C_kᵀ S_k C_k)` form,
        // using `U_kᵀU_k = I`). Unused — and not even read — on the full-`B`
        // path, so this is a zero-cost capture there.
        let mut smooth_scaled_s: Vec<Array2<f64>> = Vec::with_capacity(self.atoms.len());
        let mut smooth_grad_gb = vec![0.0_f64; beta_dim];
        // Per-atom smoothness-gradient GEMMs `½(S_k+S_kᵀ)·B_k` are independent
        // across atoms; batch them across ALL GPUs (uniform-shape tiles) and
        // scale by `lambda_smooth` below. `symmetrize = true` reproduces the
        // per-atom symmetrised `scaled_s/λ` used by the Kronecker op. Exact CPU
        // fallback per atom keeps the result bit-for-bit with the all-CPU path.
        let sym_sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sym_sb_all = batched_smooth_sb(&sym_sb_inputs, true);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            // Symmetrise and scale the smoothness penalty matrix.
            let mut scaled_s = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    let s_ij = 0.5 * (atom.smooth_penalty[[i, j]] + atom.smooth_penalty[[j, i]]);
                    scaled_s[[i, j]] = lambda_smooth * s_ij;
                }
            }
            // Gradient: g[beta_i] += (λ S_k B_k)[i, out_col]. The (m×m)·(m×p)
            // GEMM `½(S+Sᵀ)·B_k` was computed in the multi-GPU batch above; here
            // we only apply the scalar `lambda_smooth`.
            let sb = &sym_sb_all[atom_idx] * lambda_smooth;
            for out_col in 0..p {
                for i in 0..m {
                    let beta_i = off + i * p + out_col;
                    smooth_grad_gb[beta_i] += sb[[i, out_col]];
                }
            }
            // IdentityRightKroneckerPenaltyOp: factor_a = λ·S_k (m×m), factor_b = I_p.
            smooth_ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                factor_a: scaled_s.clone(),
                p,
                global_offset: off,
                k: beta_dim,
            }));
            // Retain `λ S_k` for the factored rebuild (no-op cost on full-`B`).
            smooth_scaled_s.push(scaled_s);
        }

        // Per-row active-set layout. Engaged for two regimes:
        //   * JumpReLU — structural gate plus the smooth prior's
        //     machine-precision support: atoms with
        //     `(logit - threshold)/tau > -36` enter the compact solve
        //     ([`jumprelu_in_optimization_band`]). Strictly gated-off atoms
        //     (logit ≤ threshold) carry zero assignment mass so their data-fit
        //     reconstruction contribution and data-fit logit JVP are zero, but
        //     supported atoms keep value-consistent prior gradient in the row block.
        //   * IBP-MAP at large `K` — the dense `(m_total · p)²` data
        //     Gram is infeasible, so each row is truncated to its
        //     top-`k_active` atoms above a relative magnitude cutoff
        //     ([`Self::sparse_active_plan`]). Small-`K` problems return `None`
        //     and keep the exact full-support layout.
        // The compact row block is sized `q_active = |active| + Σ_{k∈active}
        // d_k` instead of the full `q`.
        let coord_dims: Vec<usize> = self
            .assignment
            .coords
            .iter()
            .map(|c| c.latent_dim())
            .collect();
        let row_layout: Option<SaeRowLayout> = match self.assignment.mode {
            AssignmentMode::JumpReLU {
                threshold,
                temperature,
            } => Some(SaeRowLayout::from_jumprelu(
                n,
                k_atoms,
                threshold,
                temperature,
                &self.assignment.logits,
                coord_dims.clone(),
                self.assignment.coord_offsets(),
            )),
            AssignmentMode::Softmax { .. } => None,
            AssignmentMode::IBPMap { .. } => {
                match self.sparse_active_plan() {
                    Some((k_active_cap, relative_cutoff)) => {
                        // Build per-row dense assignments once to derive the
                        // active set; the row loop re-derives `assignments`
                        // (cheap softmax) and reuses these active sets.
                        let mut assignments_all = Vec::with_capacity(n);
                        for row in 0..n {
                            assignments_all.push(self.assignment.try_assignments_row(row)?);
                        }
                        // Absolute cutoff = relative_cutoff · max row peak, so a
                        // single threshold drops sub-1e-3 mass across all rows.
                        let peak = assignments_all
                            .iter()
                            .flat_map(|a| a.iter())
                            .fold(0.0_f64, |m, &v| m.max(v.abs()));
                        let cutoff = relative_cutoff * peak;
                        Some(SaeRowLayout::from_dense_weights(
                            &assignments_all,
                            k_active_cap,
                            cutoff,
                            coord_dims.clone(),
                            self.assignment.coord_offsets(),
                        ))
                    }
                    None => None,
                }
            }
        };
        // #974 likelihood-whitening seam. The single per-row decision: when the
        // installed `RowMetric` is a genuinely estimated noise model
        // (`whitens_likelihood()` — only `WhitenedStructured`), the
        // reconstruction data-fit, its t-block Gauss-Newton row block, AND the
        // β-tier data-fit gradient are all assembled through the SAME per-row
        // metric `M_n = U_n U_nᵀ = Σ_n^{-1}`. There is exactly ONE construction
        // site (the `whiten_rows` closure below), so the value the line-search
        // sums and the gradient/Hessian the Newton step solves cannot drift apart
        // (the objective↔gradient-desync cure). For Euclidean / OutputFisher /
        // no-metric the closure is the identity and every downstream loop is
        // byte-identical to the historical isotropic path.
        let whitens_likelihood = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #972 / #977 T1: engage the FACTORED Grassmann-coordinate β-tier when
        // any atom has an active decoder frame. The closed-form factorization
        // `Φᵀ(G ⊗ I_p)Φ = G ⊗ (U_iᵀU_j)` is EXACT only for the isotropic
        // likelihood; under an active whitening metric (`whitens_likelihood()`,
        // only `WhitenedStructured`) the per-row output factor would be
        // `U_iᵀ M_n U_j` and does NOT factor out of the basis Gram, so we fall
        // back to the full-`B` path there (frames + whitening is out of scope —
        // see #974). The common Euclidean / OutputFisher / no-metric case factors
        // cleanly. When `frames_engaged` is false, EVERY β-tier object below is
        // assembled bit-for-bit as the historical full-`B` path.
        let frames_engaged = self.any_frame_active() && !whitens_likelihood;
        let admission_plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        let dense_beta_curvature = admission_plan.direct_admitted
            && !(frames_engaged && beta_dim > dense_beta_penalty_probe_max_dim);
        let row_htbeta_dim = if frames_engaged {
            self.factored_border_dim()
        } else {
            beta_dim
        };
        // Build the Arrow-Schur system: heterogeneous row dims when a compact
        // layout is active, uniform `q` otherwise.
        let mut sys = if let Some(ref layout) = row_layout {
            let per_row_dims: Vec<usize> = (0..n).map(|row| layout.row_q_active(row)).collect();
            if dense_beta_curvature {
                let hbb_workspace = self.take_border_hbb_workspace(beta_dim);
                ArrowSchurSystem::new_with_per_row_dims_and_hbb_and_htbeta_cols(
                    per_row_dims,
                    beta_dim,
                    hbb_workspace,
                    row_htbeta_dim,
                )
            } else {
                self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
                ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
                    per_row_dims,
                    beta_dim,
                    row_htbeta_dim,
                )
            }
        } else if dense_beta_curvature {
            let hbb_workspace = self.take_border_hbb_workspace(beta_dim);
            ArrowSchurSystem::new_with_hbb_and_htbeta_cols(
                n,
                q,
                beta_dim,
                hbb_workspace,
                row_htbeta_dim,
            )
        } else {
            self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
            ArrowSchurSystem::new_with_empty_hbb_and_htbeta_cols(n, q, beta_dim, row_htbeta_dim)
        };
        // Apply accumulated smoothness-penalty gradients into sys.gb.
        for (i, g) in smooth_grad_gb.iter().enumerate() {
            sys.gb[i] += g;
        }
        // Hoist per-row temporaries outside the row loop: these allocations
        // previously fired N times per assembly, and each `decoded_row` /
        // `decoded_derivative_row` call inside the loop allocated its own
        // `Array1<f64>` of length p.
        let mut decoded = Array2::<f64>::zeros((k_atoms, p));
        let mut dg_buf = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // `w_dim` is the whitened output dimension: `rank` of the metric factor
        // when whitening, else `p` (identity). `error_white` is the whitened
        // residual `U_nᵀ r_n ∈ ℝ^{w_dim}` whose squared norm is `r_nᵀ M_n r_n`,
        // shared by the value path, the t-block GN, and (lifted back to p-space)
        // the β-tier gradient.
        let w_dim = match self.row_metric.as_ref() {
            Some(metric) if whitens_likelihood => metric.metric_rank(),
            _ => p,
        };
        // p-space metric-applied error `M_n r_n = U_n (U_nᵀ r_n)`, used by the
        // β-tier data-fit gradient (β lives in p-output space, so its gradient
        // contracts the residual through the full p×p metric, not the rank-space
        // whitened residual). Identity (`= error`) when not whitening.
        let mut error_white = vec![0.0_f64; w_dim];
        let mut error_metric = Array1::<f64>::zeros(p);
        // Whitened per-row Jacobian `J̃ = U_nᵀ J ∈ ℝ^{q_row × w_dim}` (row-major
        // flat) reused for the t-block htt = J̃ J̃ᵀ and gt = J̃ ẽ.
        let mut jac_white = vec![0.0_f64; q * w_dim.max(p)];
        // Data-fit Gauss-Newton β-Hessian is block-diagonal across the `p`
        // output channels and identical in each: with the flat β layout
        // `β[μ·p + oc] = B[μ, oc]` (μ enumerating (atom, basis_col)) the GN
        // outer product `Jβᵀ Jβ` couples only equal `oc`, with the same
        // `(M_total × M_total)` block `G[μ, μ'] = Σ_rows (a_k φ_k[m])(a_{k'} φ_{k'}[m'])`
        // for every channel. So `H_data = G ⊗ I_p`. The `μ` index of an `a_phi`
        // entry whose global β base is `beta_base` is `beta_base / p` (every
        // `beta_offset` and the `basis_col·p` stride are multiples of `p`).
        //
        // `G` is only non-zero on `(atom_i, atom_j)` pairs that co-occur in
        // some row's active set, so we accumulate it as a sparse map of dense
        // per-atom-pair `(m_i × m_j)` blocks keyed by `(atom_i, atom_j)` rather
        // than as a dense `(m_total × m_total)` matrix. At `K = 100K` with
        // per-row active sets of size `k_active ≪ K`, only `O(N · k_active²)`
        // pairs are ever touched, so the data Gram (and every matvec /
        // diagonal pass over it via `SparseBlockKroneckerPenaltyOp`) tracks the
        // active atoms instead of `K²`. In the dense full-support layout the
        // map degenerates to every co-occurring pair, reproducing the dense
        // Gram exactly. A `BTreeMap` key order keeps the installed op's
        // fingerprint deterministic. The `μ`-space offset of atom `k` is
        // `beta_offsets[k] / p`.
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        let mu_offsets: Vec<usize> = beta_offsets.iter().map(|&off| off / p).collect();
        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        // Stick-breaking prior for IBP-MAP depends only on (k_atoms, alpha)
        // which are constant across rows; precompute once.
        let ibp_prior_vec = match self.assignment.mode {
            AssignmentMode::IBPMap { alpha, .. } => {
                Some(ibp_stick_breaking_prior(k_atoms, alpha).to_vec())
            }
            _ => None,
        };
        let ibp_prior_slice = ibp_prior_vec.as_deref();
        // #991 design honesty weights (mean-1 HT inclusion corrections); see
        // the seam comment at the per-row residual below.
        let row_loss_w = self.row_loss_weights.as_deref();
        // Scratch buffer for per-(row, atom) decoded outputs. The full `decoded`
        // matrix retains all atoms for this row so the assignment-Jacobian
        // helper can read it.
        let mut decoded_scratch = vec![0.0_f64; p];
        // Kronecker htbeta storage for the full-B path: per-row sparse support
        // and local Jacobian. Factored rows write their C-space cross slabs
        // directly in the row loop below.
        let mut kron_a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut kron_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        // Dense full-support index `[0, k_atoms)`, used by the row loop when no
        // compact layout is engaged so the active-atom iteration is uniform.
        let all_atoms_index: Vec<usize> = (0..k_atoms).collect();
        // Per-atom per-axis periodicity, hoisted out of the row loop. Selects
        // the smooth von-Mises coordinate prior on wrapped (Circle) axes and
        // the Gaussian prior on Euclidean axes; see `ArdAxisPrior`.
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            // Reconstruction uses the row's active support: for the dense
            // full-support layout this is all atoms (exact); for a compact
            // layout the dropped atoms carry negligible `O(a)` reconstruction
            // mass and zero curvature, so excluding them keeps `fitted`,
            // `error`, and the logit-JVP cross term `(decoded[k] − fitted)`
            // mutually consistent with the curvature actually assembled.
            fitted.fill(0.0);
            let row_active_owned: Option<&[usize]> =
                row_layout.as_ref().map(|l| l.active_atoms[row].as_slice());
            match row_active_owned {
                Some(active) => {
                    for &atom_idx in active {
                        let a_k = assignments[atom_idx];
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_scratch);
                        for out_col in 0..p {
                            decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                            fitted[out_col] += a_k * decoded_scratch[out_col];
                        }
                    }
                }
                None => {
                    for atom_idx in 0..k_atoms {
                        let a_k = assignments[atom_idx];
                        self.atoms[atom_idx].fill_decoded_row(row, &mut decoded_scratch);
                        for out_col in 0..p {
                            decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                            fitted[out_col] += a_k * decoded_scratch[out_col];
                        }
                    }
                }
            }
            for out_col in 0..p {
                error[out_col] = fitted[out_col] - target[[row, out_col]];
            }
            // #991 design-honesty seam: a per-row scalar weight `w_row` on the
            // reconstruction channel is exactly the metric `w_row · I_p`, so it
            // is realized as a `√w_row` scaling of the THREE row-local data
            // quantities at their construction sites — this residual, the
            // latent Jacobian (below), and the β basis load `a·φ` (below).
            // Every downstream data object then carries exactly one factor of
            // `w_row` (gt, htt, htbeta, the β Gram `G`, and the β gradient),
            // matching the `w_row`-weighted value `loss_scaled` sums; the
            // per-row latent priors (assignment / ARD, added to `gt`/`htt`
            // further down) are deliberately unweighted — see the
            // `row_loss_weights` field docs. `None` ⇒ `sqrt_row_w == 1.0` and
            // no multiply is applied (bit-identical unweighted path).
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
            if sqrt_row_w != 1.0 {
                for out_col in 0..p {
                    error[out_col] *= sqrt_row_w;
                }
            }
            // #974 seam (step 1/2): whiten the per-row residual ONCE.
            //   * not whitening ⇒ `error_white == error` (length p) and
            //     `error_metric == error`; every downstream loop is the
            //     historical isotropic path bit-for-bit.
            //   * whitening ⇒ `error_white = U_nᵀ r_n ∈ ℝ^{w_dim}` (its squared
            //     norm is `r_nᵀ M_n r_n`, the value the data-fit sums) and
            //     `error_metric = U_n (U_nᵀ r_n) = M_n r_n ∈ ℝ^p` (the p-space
            //     metric-applied residual the β-tier gradient contracts).
            match self.row_metric.as_ref() {
                Some(metric) if whitens_likelihood => {
                    let wr = metric.whiten_residual_row(row, error.view());
                    for (slot, &v) in error_white.iter_mut().zip(wr.iter()) {
                        *slot = v;
                    }
                    let mr = metric.apply_metric_row(row, error.view());
                    for (slot, &v) in error_metric.iter_mut().zip(mr.iter()) {
                        *slot = v;
                    }
                }
                _ => {
                    for out_col in 0..p {
                        error_white[out_col] = error[out_col];
                        error_metric[out_col] = error[out_col];
                    }
                }
            }

            // Determine whether this row uses the compact active-set layout.
            //   * JumpReLU: gated atoms plus the smooth prior's
            //     machine-precision support enter.
            //   * IBP-MAP at large K: only the top-`k_active` atoms.
            //   * Otherwise (small K): the dense uniform-q layout.
            let (q_row, mut local_jac_row) = if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                let starts = &layout.coord_starts[row];
                let q_active = layout.row_q_active(row);
                let mut jac_compact = Array2::<f64>::zeros((q_active, p));
                // Logit JVP rows for active atoms only, using the per-mode
                // assignment sensitivity `da_k/dl_k` contracted into the
                // decoded / fitted-corrected output direction.
                let logits_row = self.assignment.logits.row(row);
                for (j, &k) in active.iter().enumerate() {
                    fill_active_atom_logit_jvp(
                        ActiveAtomLogitJvp {
                            mode: self.assignment.mode,
                            k,
                            logit_k: logits_row[k],
                            a_k: assignments[k],
                            decoded_k: decoded.row(k),
                            fitted: fitted.view(),
                            ibp_prior: ibp_prior_slice,
                            compact_index: j,
                        },
                        &mut jac_compact,
                    );
                }
                // Coordinate JVP rows for active atoms only.
                for (j, &k) in active.iter().enumerate() {
                    let d = self.atoms[k].latent_dim;
                    let a_k = assignments[k];
                    let coord_start = starts[j];
                    for axis in 0..d {
                        self.atoms[k].fill_decoded_derivative_row(row, axis, &mut dg_buf);
                        for out_col in 0..p {
                            jac_compact[[coord_start + axis, out_col]] = a_k * dg_buf[out_col];
                        }
                    }
                }
                (q_active, jac_compact)
            } else {
                // Fresh per-row Jacobian, structurally identical to the
                // JumpReLU branch: every (q × p) element is unconditionally
                // overwritten below (assignment-chart JVP rows + coordinate rows), so the
                // `Array2::zeros` allocation needs no separate `fill(0.0)` and
                // the populated buffer is returned by move without a clone.
                let mut jac_row = Array2::<f64>::zeros((q, p));
                fill_assignment_logit_jvp_rows(
                    self.assignment.mode,
                    self.assignment.logits.row(row),
                    assignments.view(),
                    decoded.view(),
                    fitted.view(),
                    ibp_prior_slice,
                    &mut jac_row,
                );
                // Coordinate columns for all atoms.
                for atom_idx in 0..k_atoms {
                    let d = self.atoms[atom_idx].latent_dim;
                    let off = coord_offsets[atom_idx];
                    let a_k = assignments[atom_idx];
                    for axis in 0..d {
                        self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut dg_buf);
                        for out_col in 0..p {
                            jac_row[[off + axis, out_col]] = a_k * dg_buf[out_col];
                        }
                    }
                }
                (q, jac_row)
            };

            // #991 design-honesty seam, Jacobian leg: scale the row's latent
            // Jacobian by `√w_row` BEFORE the whitening / Kronecker capture so
            // htt (= J̃J̃ᵀ), the data part of gt (= J̃ẽ, the residual already
            // carries its own √w_row), and the htbeta cross block (J paired
            // with the √w_row-scaled β load below) each carry exactly one
            // factor of `w_row`. No-op on the unweighted path.
            if sqrt_row_w != 1.0 {
                for a in 0..q_row {
                    for out_col in 0..p {
                        local_jac_row[[a, out_col]] *= sqrt_row_w;
                    }
                }
            }

            // #974 seam (step 2/2): whiten the per-row Jacobian through the SAME
            // metric the residual was whitened by. `jac_white[a*w_dim + k]` holds
            // `J̃[a, k] = Σ_out U_n[out, k] · J_n[a, out]` so the t-block
            // Gauss-Newton row block is `htt = J̃ J̃ᵀ = J_n M_n J_nᵀ` and
            // `gt = J̃ ẽ = J_nᵀ M_n r_n`. When not whitening, `w_dim == p` and the
            // whitened jac equals the raw Jacobian, so htt/gt are byte-identical
            // to the historical isotropic assembly. Because the SAME `error_white`
            // feeds both the value-path data-fit (Σ½ ẽ²) and this gradient
            // (J̃ ẽ), the objective and its t-block gradient share one whitening
            // — they cannot desync.
            if whitens_likelihood {
                if let Some(metric) = self.row_metric.as_ref() {
                    for a in 0..q_row {
                        for k in 0..w_dim {
                            let mut acc = 0.0;
                            // U_n[out, k] read through the metric's factor layout.
                            for out_col in 0..p {
                                acc += metric.factor_entry(row, out_col, k)
                                    * local_jac_row[[a, out_col]];
                            }
                            jac_white[a * w_dim + k] = acc;
                        }
                    }
                }
            } else {
                for a in 0..q_row {
                    for out_col in 0..p {
                        jac_white[a * w_dim + out_col] = local_jac_row[[a, out_col]];
                    }
                }
            }

            // Build the per-row Arrow-Schur block at the row's active dim.
            let mut block = ArrowRowBlock::new(q_row, row_htbeta_dim);
            for a in 0..q_row {
                let mut g = 0.0;
                for k in 0..w_dim {
                    g += jac_white[a * w_dim + k] * error_white[k];
                }
                block.gt[a] += g;
                for b in 0..q_row {
                    let mut h = 0.0;
                    for k in 0..w_dim {
                        h += jac_white[a * w_dim + k] * jac_white[b * w_dim + k];
                    }
                    block.htt[[a, b]] += h;
                }
            }

            // Assignment prior in logit space.
            // For compact layout: position `j` = active_atoms index.
            // For dense layout: position `atom_idx` directly.
            //
            // H-consistency note (#1006 audit). This `assignment_hdiag` is the
            // assignment penalty's EXACT `hessian_diag` (softmax-sparsity, IBP-MAP
            // empirical-π, or JumpReLU sigmoid surrogate), added RAW — unlike the ARD
            // coordinate curvature (`prior.hess.max(0.0)` below) and the
            // decoder-tier penalties (`psd_majorizer_diag`), it is NOT majorized.
            // That diagonal CAN be negative (the softmax `(1−2z)`-type logit
            // curvature, IBP `score·(1−2z)` term, and JumpReLU above its
            // threshold), so the assembled `H_tt` is
            // Gauss-Newton (PSD) + majorized ARD (PSD) + raw-assignment-prior
            // (indefinite) and is therefore NOT guaranteed PD off the optimum.
            // This refutes the "pure GN+majorizer cannot be non-PD" premise: the
            // genuine non-PD the evidence factorization reports
            // (`arrow_schur.rs` "evidence mode preserves the genuine Cholesky")
            // comes from this un-majorized term, not from any divergence between
            // the Newton-solve H and the evidence H. Both paths factor THIS one
            // assembled block (single source of truth); they differ only in ridge
            // policy — Newton conditions a non-PD block, evidence refuses to (a
            // silent ridge would shift the reported log-det). At the converged
            // optimum `H_tt` is PD, where the evidence factor is taken. Because
            // the criterion's log|H| and the Γ adjoint (`logdet_theta_adjoint`)
            // both differentiate THIS raw diagonal — `hessian_diag` and its exact
            // logit third channels `hessian_diag_logit_third_channels`,
            // #1006 — value and gradient stay on the same branch with no desync.
            let assignment_base = row * k_atoms;
            if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                for (j, &k) in active.iter().enumerate() {
                    block.gt[j] += assignment_grad[assignment_base + k];
                    block.htt[[j, j]] += assignment_hdiag[assignment_base + k];
                }
            } else {
                for free_idx in 0..assignment_dim {
                    block.gt[free_idx] += assignment_grad[assignment_base + free_idx];
                    block.htt[[free_idx, free_idx]] += assignment_hdiag[assignment_base + free_idx];
                }
            }

            // ARD on each on-atom coordinate.
            // For compact layout: only active atoms; coord positions use compact starts.
            // For dense layout: all atoms; coord positions use coord_offsets.
            if let Some(ref layout) = row_layout {
                let active = &layout.active_atoms[row];
                let starts = &layout.coord_starts[row];
                for (j, &k) in active.iter().enumerate() {
                    let coord = &self.assignment.coords[k];
                    let d = coord.latent_dim();
                    if rho.log_ard[k].is_empty() {
                        continue;
                    }
                    if rho.log_ard[k].len() != d {
                        return Err(format!(
                            "ARD rho atom {k} has len {} but atom dim is {d}",
                            rho.log_ard[k].len()
                        ));
                    }
                    let row_t = coord.row(row);
                    let periods = &ard_axis_periods[k];
                    for axis in 0..d {
                        // ARD on coords is a genuine per-row prior (each row
                        // contributes the per-axis prior energy), so it is NOT
                        // minibatch-scaled — the per-chunk row sums already
                        // reconstruct the full coordinate prior across a pass.
                        // The value (`ard_value`/`loss.ard`) and the gradient
                        // both come from the SAME `ArdAxisPrior` energy, so they
                        // stay FD-consistent on periodic axes. The exact
                        // von-Mises curvature `V'' = α·cos(κt)` is INDEFINITE —
                        // it goes negative for |t| past a quarter period — so
                        // writing it raw into the Newton/Schur `htt` diagonal
                        // makes that PSD curvature block indefinite and the Schur
                        // Cholesky (used both for the Newton step and the exact
                        // log-det) fails on a non-PD pivot. Accumulate the PSD
                        // majorizer `max(V'', 0)` instead, exactly as
                        // `add_sae_coord_penalty` does for the registry coord
                        // penalties: the positive part keeps `htt` PSD so the
                        // factorization succeeds, and majorizing the curvature of
                        // a fixed prior only damps the Newton step — it does not
                        // move the stationary point (the gradient, which sets the
                        // fixed point, stays the exact `V'`).
                        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                        let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                        block.gt[starts[j] + axis] += prior.grad;
                        block.htt[[starts[j] + axis, starts[j] + axis]] += prior.hess.max(0.0);
                    }
                }
            } else {
                for atom_idx in 0..k_atoms {
                    let coord = &self.assignment.coords[atom_idx];
                    let d = coord.latent_dim();
                    if rho.log_ard[atom_idx].is_empty() {
                        continue;
                    }
                    if rho.log_ard[atom_idx].len() != d {
                        return Err(format!(
                            "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                            rho.log_ard[atom_idx].len()
                        ));
                    }
                    let off = coord_offsets[atom_idx];
                    let row_t = coord.row(row);
                    let periods = &ard_axis_periods[atom_idx];
                    for axis in 0..d {
                        // PSD-majorize the (possibly negative) von-Mises curvature
                        // into the Newton/Schur `htt` block; see the compact-layout
                        // branch above for why `max(V'', 0)` is required to keep
                        // `htt` PD (the exact `V'' = α·cos κt` is indefinite past a
                        // quarter period and breaks the Schur/log-det Cholesky).
                        let alpha =
                            SaeManifoldRho::stable_exp_strength(rho.log_ard[atom_idx][axis]);
                        let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                        block.gt[off + axis] += prior.grad;
                        block.htt[[off + axis, off + axis]] += prior.hess.max(0.0);
                    }
                }
            }

            // Beta gradient/Hessian — Kronecker form J_β = φᵀ ⊗ I_p.
            //
            // The per-row beta Jacobian is
            //   J_β[out_col, beta_idx] = a_k · phi_k[basis_col]   if out_col == out_col(beta_idx)
            //                            0                         otherwise
            // so the data-fit Gauss-Newton beta-Hessian factors as a rank-`p`
            // sum of outer products. We pre-compute the per-(atom, basis_col)
            // scalar `a_k · phi_k` once and reuse it across the `out_col`
            // and inner `(atom_j, basis_col2)` loops.
            //
            // Full-B rows keep the matrix-free Kronecker path below. Factored
            // rows write the `q_i × Σ M_k r_k` C-space cross slab directly by
            // folding each output-channel contribution through the atom frame,
            // so no `q_i × β_dim` slab is ever materialized.
            //
            // Only the row's active atoms contribute `a_phi` support and data
            // curvature: in a compact layout (JumpReLU gate or large-K
            // top-`k_active` truncation) the inactive atoms carry zero (gated)
            // or sub-cutoff assignment mass and are excluded — this is what
            // keeps both the htbeta support and the `G` accumulation
            // `O(k_active)` rather than `O(K)`. In the dense full-support
            // layout `row_active` spans all atoms.
            let row_active: &[usize] = match row_layout {
                Some(ref layout) => layout.active_atoms[row].as_slice(),
                None => &all_atoms_index,
            };
            let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(row_active.len() * 4);
            // Per-active-atom weighted basis row `a_k · φ_k[·]`, retained so the
            // data Gram blocks can be accumulated as clean per-atom-pair outer
            // products `(a_k φ_k) (a_{k'} φ_{k'})ᵀ`.
            let mut weighted_phi: Vec<(usize, Vec<f64>)> = Vec::with_capacity(row_active.len());
            for &atom_idx in row_active {
                let atom = &self.atoms[atom_idx];
                let atom_beta_off = beta_offsets[atom_idx];
                let m = atom.basis_size();
                let a_k = assignments[atom_idx];
                let mut wphi = Vec::with_capacity(m);
                for basis_col in 0..m {
                    let phi = atom.basis_values[[row, basis_col]];
                    // #991 design-honesty seam, β leg: the `√w_row` here pairs
                    // with the `√w_row` on the residual (β gradient =
                    // `a·φ · M r` ⇒ w_row) and with itself (β Gram `G` and the
                    // htbeta Kronecker capture ⇒ w_row). `1.0` when unweighted.
                    let w = a_k * phi * sqrt_row_w;
                    a_phi.push((atom_beta_off + basis_col * p, w));
                    wphi.push(w);
                }
                weighted_phi.push((atom_idx, wphi));
            }
            // β data-fit gradient `gᵦ += J_βᵀ M_n r_n`. The β-Jacobian is
            // `J_β = φ_nᵀ ⊗ I_p`, so `J_βᵀ M_n r_n = φ_n ⊗ (M_n r_n)` —
            // contract the basis weight `a·φ` against the p-space metric-applied
            // residual `error_metric` (= `M_n r_n`), the SAME whitening the value
            // path and t-block share. When not whitening, `error_metric == error`
            // and this is byte-identical to the historical `J_βᵀ r`.
            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                if j_beta_i == 0.0 {
                    continue;
                }
                for out_col in 0..p {
                    sys.gb[beta_base_i + out_col] += j_beta_i * error_metric[out_col];
                    // No dense hbb write — the sparse `G ⊗ I_p` op installed
                    // after the loop carries the data-fit GN β-Hessian.
                }
            }
            if frames_engaged {
                for &atom_idx in row_active {
                    let atom = &self.atoms[atom_idx];
                    let m = atom.basis_size();
                    let a_k = assignments[atom_idx];
                    for basis_col in 0..m {
                        let phi = atom.basis_values[[row, basis_col]];
                        let w = a_k * phi * sqrt_row_w;
                        if w == 0.0 {
                            continue;
                        }
                        let c_base = frame_projection.border_offsets[atom_idx]
                            + basis_col * frame_projection.ranks[atom_idx];
                        for c in 0..q_row {
                            let mut hrow = block.htbeta.row_mut(c);
                            let hrow_slice = hrow.as_slice_mut().expect("htbeta row is contiguous");
                            for out_col in 0..p {
                                let value = local_jac_row[[c, out_col]] * w;
                                frame_projection.accumulate_output_project(
                                    atom_idx, c_base, out_col, value, hrow_slice,
                                );
                            }
                        }
                    }
                }
            }
            // Data-fit GN β-Hessian: accumulate the channel-independent block
            // `G[μ_i, μ_j] += (a_k φ_k)[μ_i] (a_{k'} φ_{k'})[μ_j]` into the
            // sparse per-atom-pair map (the `out_col` dimension is carried by
            // `I_p`). Only co-occurring `(atom_i, atom_j)` pairs are touched.
            for ai in 0..weighted_phi.len() {
                let (atom_i, ref wphi_i) = weighted_phi[ai];
                let m_i = wphi_i.len();
                for aj in 0..weighted_phi.len() {
                    let (atom_j, ref wphi_j) = weighted_phi[aj];
                    let m_j = wphi_j.len();
                    let blk = g_blocks
                        .entry((atom_i, atom_j))
                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                    for li in 0..m_i {
                        let wi = wphi_i[li];
                        if wi == 0.0 {
                            continue;
                        }
                        for lj in 0..m_j {
                            blk[[li, lj]] += wi * wphi_j[lj];
                        }
                    }
                }
            }
            if !frames_engaged {
                kron_a_phi.push(a_phi);
                // Flatten local_jac_row row-major into a plain Vec<f64> (q_row * p entries).
                let mut jac_flat = vec![0.0_f64; q_row * p];
                for c in 0..q_row {
                    for j in 0..p {
                        jac_flat[c * p + j] = local_jac_row[[c, j]];
                    }
                }
                kron_jac.push(jac_flat);
            }
            sys.rows[row] = block;
        }
        // Apply Riemannian geometry to the per-row row blocks (htt, gt) and
        // also to the per-row Kronecker local Jacobians stored in kron_jac.
        // When the SAE ext-coord manifold is non-Euclidean (any atom latent
        // on sphere / circle / interval), the local Jacobian rows that map
        // into the t-block tangent space must be projected via the per-row
        // tangent projector P_i.  This mirrors what
        // `apply_riemannian_latent_geometry` does to `row.htbeta`, applied
        // here to the (q × p) kron_jac so the Kronecker htbeta_matvec uses
        // the Riemannian-projected form.
        // Apply Riemannian geometry only for the dense uniform-q layout. Any
        // compact active-set layout (JumpReLU gate or large-K softmax/IBP
        // truncation) has heterogeneous q_i; the Riemannian projector path
        // requires a uniform latent dimension. The sparse plan only engages on
        // Euclidean ext-coord manifolds (see `sparse_active_plan`), so skipping
        // the projector here is correct — there is nothing to project.
        if row_layout.is_none() {
            let raw_gt_rows: Vec<Array1<f64>> = sys.rows.iter().map(|row| row.gt.clone()).collect();
            self.apply_sae_riemannian_geometry(&mut sys);
            let manifold = self.ext_coord_manifold();
            if !frames_engaged && !manifold.is_euclidean() {
                let ext = self.ext_coord_matrix();
                // Project the local Jacobian columns onto the tangent space at
                // each row's ext-coord point. Each column `j` of the row's
                // (q_row × p) Jacobian is an ambient-space vector of length
                // `q_row`; the manifold projector acts on one such column at a
                // time. Working directly on the row-major `jac_flat` storage via
                // a single reusable `col_buf` avoids the two dense (q × p) copies
                // (flatten→Array2, project, unflatten→Vec) that previously fired
                // per row. `t_buf` still holds the row's ext-coord vector.
                let mut t_buf = vec![0.0_f64; q];
                let mut col_buf = Array1::<f64>::zeros(q);
                for row_idx in 0..n {
                    let ext_row = ext.row(row_idx);
                    for (slot, &v) in t_buf.iter_mut().zip(ext_row.iter()) {
                        *slot = v;
                    }
                    let t_i = ArrayView1::from(t_buf.as_slice());
                    let raw_gt = raw_gt_rows[row_idx].view();
                    let jac_flat = &mut kron_jac[row_idx];
                    let q_row = jac_flat.len() / p;
                    for j in 0..p {
                        for c in 0..q_row {
                            col_buf[c] = jac_flat[c * p + j];
                        }
                        let projected_col = manifold.project_vector_to_gradient_tangent(
                            t_i,
                            raw_gt.slice(ndarray::s![..q_row]),
                            col_buf.slice(ndarray::s![..q_row]),
                        );
                        for c in 0..q_row {
                            jac_flat[c * p + j] = projected_col[c];
                        }
                    }
                }
            }
        }
        // Build and install the full-B Kronecker htbeta_matvec.
        //
        // `SaeKroneckerRows` holds per-row `(a_phi, local_jac)` and implements
        // the cross-block operator without ever materialising the dense
        // `(q × K·p)` slab.  The cross-block factorises as `H_tβ = L · J_β`,
        // where `J_β = φᵀ ⊗ I_p` projects a length-`K` β vector onto the
        // `p`-dimensional decoded output space (`apply_jbeta`) and `L_i` is
        // the per-row `(q_i × p)` assignment+coordinate Jacobian that lifts
        // that p-vector into the row's `q_i`-dim tangent block (`apply_l`).
        // Both factors are required: the contract of `set_row_htbeta_operator`
        // is `out.len() == d` (= `q_i`), so writing `apply_jbeta`'s p-vector
        // output directly into a length-`q_i` buffer overflows whenever
        // `p > q_i` (the common case once `p` reflects real feature width).
        // Symmetric for the transpose: `H_βt = J_βᵀ · Lᵀ`, so apply `Lᵀ`
        // first to map the q_i-vector back to p-space, then scatter through
        // the support.
        let device_rows = if frames_engaged {
            None
        } else {
            Some((kron_a_phi.clone(), kron_jac.clone()))
        };
        if !frames_engaged {
            let kron = Arc::new(SaeKroneckerRows::new(p, kron_a_phi, kron_jac));
            let kron_t = Arc::clone(&kron);
            let p_dim = p;
            sys.set_row_htbeta_operator(
                move |row_idx, x, out| {
                    // out = L_i · (J_β · x). Allocate a length-p scratch buffer
                    // for the intermediate decoded-output vector; both factors
                    // overwrite their output buffers (`apply_jbeta` zeroes
                    // before accumulating, `apply_l` writes per-row), so no
                    // pre-zeroing of `u_p`/`out` is needed.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(xs) = x.as_slice() {
                        kron.apply_jbeta(row_idx, xs, &mut u_p);
                    } else {
                        let x_vec: Vec<f64> = x.iter().copied().collect();
                        kron.apply_jbeta(row_idx, &x_vec, &mut u_p);
                    }
                    kron.apply_l(row_idx, &u_p, out_slice);
                },
                move |row_idx, v, out| {
                    // out += J_βᵀ · (Lᵀ · v). `apply_l_t` accumulates into a
                    // zero-initialised length-p buffer to produce the p-vector
                    // `Lᵀ v`; `scatter_jbeta_t` then adds φ_i[s] · u_p[j] into
                    // the length-K β accumulator at each active `(s, j)`.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(vs) = v.as_slice() {
                        kron_t.apply_l_t(row_idx, vs, &mut u_p);
                    } else {
                        let v_vec: Vec<f64> = v.iter().copied().collect();
                        kron_t.apply_l_t(row_idx, &v_vec, &mut u_p);
                    }
                    kron_t.scatter_jbeta_t(row_idx, &u_p, out_slice);
                },
            );
        }
        let mut beta_penalty_assembly = SaeBetaPenaltyAssembly::default();
        let factored_row_projection = if frames_engaged && analytic_penalties.is_some() {
            Some(&frame_projection)
        } else {
            None
        };
        if let Some(registry) = analytic_penalties {
            // Upfront validation: refuse penalty kinds the SAE row layout
            // cannot host, and refuse mixed-d row-block configurations.
            // This makes the dispatch loop below total — no runtime
            // "unsupported penalty" fallthrough, no K-gating.
            self.validate_analytic_penalty_registry(registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
            beta_penalty_assembly = self
                .add_sae_analytic_penalty_contributions(
                    &mut sys,
                    registry,
                    penalty_scale,
                    row_layout.as_ref(),
                    dense_beta_curvature,
                    factored_row_projection,
                )
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
        if frames_engaged {
            // ── #972 / #977 T1 — FACTORED β-tier transform ──────────────────
            //
            // The entire β-tier above was assembled in the full-`B` (p-wide)
            // layout: `sys.gb` is `g_B` (length `beta_dim`), `sys.hbb` carries
            // any analytic Beta-tier penalty, and `g_blocks` is the
            // FRAME-INDEPENDENT basis Gram. We now rebuild the β-tier in the
            // factored coordinate space `C` (width `factored_border_dim`), the
            // full-`B` system sandwiched by `Φ = blkdiag(I_{M_k} ⊗ U_k)`:
            //   * gradient   `g_C = Φᵀ g_B`              (per atom `(g_B U_k)`),
            //   * data H      `Φᵀ(G⊗I_p)Φ = G_{ij}⊗(U_iᵀU_j)`,
            //   * smooth      `λ S_k ⊗ I_{r_k}`          (since `U_kᵀU_k = I`),
            //   * analytic    `Φᵀ hbb Φ`                 (dense, only if written).
            // Un-framed atoms ride the `r_k = p, U_k = I_p` identity special case.
            let off_c = &frame_projection.border_offsets;
            let ranks = &frame_projection.ranks;
            let basis_sizes = &frame_projection.basis_sizes;
            let border_dim = frame_projection.border_dim();
            let gb_c = frame_projection.project_border_vec(sys.gb.view());

            // Data β-Hessian: `G_{ij} ⊗ W_{ij}` with `W_{ij} = U_iᵀU_j`. The
            // basis Gram `g_blocks` is unchanged; only the output factor is the
            // per-pair frame overlap (`I_{r_k}` within a framed atom, `I_p` for
            // un-framed).
            let mut frame_blocks: Vec<FactoredFrameGBlock> = Vec::with_capacity(g_blocks.len());
            for ((atom_i, atom_j), data) in g_blocks.into_iter() {
                if data.iter().all(|&v| v == 0.0) {
                    continue;
                }
                // `W_{ij} = U_iᵀ U_j` from the precomputed per-atom frames.
                let w = self.frame_cross_factor(atom_i, atom_j);
                frame_blocks.push(FactoredFrameGBlock {
                    atom_i,
                    atom_j,
                    g: data,
                    w,
                });
            }
            let data_op =
                FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), frame_blocks)?;

            // Smooth penalty in factored space: `λ S_k ⊗ I_{r_k}` at `off_C[k]`.
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len() + 2);
            for k in 0..self.atoms.len() {
                let r = ranks[k];
                ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                    factor_a: smooth_scaled_s[k].clone(),
                    p: r,
                    global_offset: off_c[k],
                    k: border_dim,
                }));
            }
            ops.push(Arc::new(data_op));
            // Analytic Beta-tier penalty: project the dense full-`B` `hbb` block
            // `Φᵀ hbb Φ` into the factored space. Only present when a Beta-tier
            // penalty actually wrote `hbb` (else `hbb` is all-zero and the dense
            // `(border_dim)²` op is skipped entirely, exactly as full-`B`).
            if beta_penalty_assembly.dense_written {
                let hbb_c =
                    self.project_dense_penalty_to_factored(sys.hbb.view(), &frame_projection);
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            } else if beta_penalty_assembly.deferred_factored {
                let registry =
                    analytic_penalties.expect("deferred beta curvature requires registry");
                let hbb_c = self.build_factored_beta_penalty_curvature(
                    registry,
                    penalty_scale,
                    &frame_projection,
                );
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            }

            // Re-point the system's β-tier to the factored width. The t-tier
            // (per-row `htt`, `gt`) is frame-independent and untouched; row
            // cross-block slabs were allocated and assembled directly in
            // factored coordinates, so analytic row supplements and data-fit
            // cross terms already share shape `(q_i × factored_border_dim)`.
            sys.k = border_dim;
            sys.gb = gb_c;
            self.reclaim_border_hbb_workspace(&mut sys);
            // Factored per-atom block ranges for the block-Jacobi Schur
            // preconditioner: `[off_C[k] .. off_C[k] + M_k·r_k]`.
            let mut block_ranges: Vec<std::ops::Range<usize>> =
                Vec::with_capacity(self.atoms.len());
            for k in 0..self.atoms.len() {
                let start = off_c[k];
                block_ranges.push(start..start + basis_sizes[k] * ranks[k]);
            }
            sys.set_block_offsets(Arc::from(block_ranges.into_boxed_slice()));
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: border_dim, ops }));
        } else {
            let (device_a_phi, device_local_jac) =
                device_rows.expect("full-beta SAE PCG rows are cloned before row operator install");
            // Wire per-atom β block ranges so the Jacobi preconditioner builds one
            // dense Schur sub-block per atom (block-Jacobi) instead of scalar-diagonal
            // inversion.  Each atom's decoder coefficients form a natural block:
            // `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
            sys.set_block_offsets(self.beta_block_offsets());
            // Install the composite BetaPenaltyOp (#296): smoothness contributions
            // via per-atom KroneckerPenaltyOp (avoid dense K×K materialisation), the
            // data-fit Gauss-Newton β-Hessian as the structured `G ⊗ I_p`
            // SparseBlockKroneckerPenaltyOp (block-sparse over co-occurring
            // `(atom, atom')` pairs, block-diagonal across the `p` output channels,
            // identical per channel), plus — only when a Beta-tier analytic penalty
            // was written — the dense `sys.hbb` residual contribution. When no beta
            // penalty fired, `sys.hbb` is all-zero and the dense `(K·p)²` operator
            // is skipped entirely. The sparse data op tracks only the active-atom
            // couplings, so its storage and matvec cost scale with `k_active`, not
            // `K`, at `K = 100K`.
            // Convert the per-atom-pair coupling map into `SparseGBlock`s keyed
            // by μ-space offsets. Empty blocks (no co-occurrence) are simply
            // absent from the map.
            let g_sparse_blocks: Vec<SparseGBlock> = g_blocks
                .into_iter()
                .filter_map(|((atom_i, atom_j), data)| {
                    if data.iter().all(|&v| v == 0.0) {
                        None
                    } else {
                        Some(SparseGBlock {
                            row_off: mu_offsets[atom_i],
                            col_off: mu_offsets[atom_j],
                            data,
                        })
                    }
                })
                .collect();
            let device_smooth_blocks = smooth_scaled_s
                .iter()
                .enumerate()
                .map(|(atom_idx, factor_a)| DeviceSaeSmoothBlock {
                    global_offset: beta_offsets[atom_idx],
                    factor_a: factor_a.clone(),
                })
                .collect();
            sys.set_device_sae_pcg_data(DeviceSaePcgData {
                p,
                beta_dim,
                a_phi: device_a_phi,
                local_jac: device_local_jac,
                smooth_blocks: device_smooth_blocks,
                sparse_g_blocks: g_sparse_blocks.clone(),
            });
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(SparseBlockKroneckerPenaltyOp {
                p,
                dim_a: m_total,
                k: beta_dim,
                blocks: g_sparse_blocks,
            }));
            if beta_penalty_assembly.dense_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
            self.reclaim_border_hbb_workspace(&mut sys);
        }
        if let Some(deflation) = self.row_gauge_deflation_for_layout(row_layout.as_ref()) {
            sys.set_row_gauge_deflation(deflation);
        }
        // Store the active-set layout for `apply_newton_step`.
        self.last_row_layout = row_layout;
        // Record whether `delta_beta` from this system is a factored ΔC (needs a
        // frame lift) or a full-`B` ΔB. Read by `apply_newton_step_impl`.
        self.last_frames_active = frames_engaged;
        Ok(sys)
    }

    /// Project a dense full-`B` Beta-tier penalty Hessian `hbb` (`beta_dim ×
    /// beta_dim`, the analytic `∂²P/∂B∂B` block) into the factored coordinate
    /// space `Φᵀ hbb Φ` (`border_dim × border_dim`) for the #972 / #977 T1
    /// frame transform. `Φ = blkdiag(I_{M_k} ⊗ U_k)` maps C-space → B-space, so
    /// the projected block contracts both index legs through the per-atom frames.
    ///
    /// The projection is done in two passes to stay `O(beta_dim · border_dim +
    /// border_dim²)` instead of forming the dense `Φ` explicitly: first
    /// `T = hbb · Φ` (right multiply, columns fold `U`), then `Φᵀ · T` (left
    /// multiply, rows fold `U`). Analytic Beta-tier penalties are rare and small,
    /// so this only fires when one is actually installed.
    fn project_dense_penalty_to_factored(
        &self,
        hbb: ArrayView2<'_, f64>,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        projection.project_block(hbb)
    }

    fn build_factored_beta_penalty_curvature(
        &self,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let target_beta = self.flatten_beta();
        let mut hbb_c = Array2::<f64>::zeros((projection.border_dim(), projection.border_dim()));
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                PenaltyTier::Beta => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                _ => {}
            }
        }
        hbb_c
    }

    fn add_factored_beta_penalty_curvature_for_penalty(
        &self,
        hbb_c: &mut Array2<f64>,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) {
        let p = self.output_dim();
        if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
            let Some(per_fit) = self.live_decoder_incoherence_penalty(base) else {
                return;
            };
            let beta_dim = self.beta_dim();
            let mut probe = Array1::<f64>::zeros(beta_dim);
            for k in 0..self.atoms.len() {
                for basis_col in 0..projection.basis_sizes[k] {
                    for frame_col in 0..projection.ranks[k] {
                        probe.fill(0.0);
                        projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                        let col = projection.border_offsets[k]
                            + basis_col * projection.ranks[k]
                            + frame_col;
                        let hv = per_fit.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                        projection
                            .project_border_vec(hv.view())
                            .iter()
                            .enumerate()
                            .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live mechanism-sparsity offset must match an SAE atom");
                let block_len = end - start;
                let mut local_penalty = per_atom.clone();
                local_penalty.target = PsiSlice {
                    range: 0..block_len,
                    latent_dim: Some(projection.basis_sizes[atom_idx]),
                };
                let block = target_beta.slice(s![start..end]);
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = local_penalty.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live nuclear-norm offset must match an SAE atom");
                let block = target_beta.slice(s![start..end]);
                let block_len = end - start;
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = per_atom.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        let beta_dim = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for k in 0..self.atoms.len() {
            for basis_col in 0..projection.basis_sizes[k] {
                for frame_col in 0..projection.ranks[k] {
                    probe.fill(0.0);
                    projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                    let col =
                        projection.border_offsets[k] + basis_col * projection.ranks[k] + frame_col;
                    let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                    projection
                        .project_border_vec(hv.view())
                        .iter()
                        .enumerate()
                        .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                }
            }
        }
        assert_eq!(p, self.output_dim());
    }

    fn ext_coord_matrix(&self) -> Array2<f64> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let flat = self.assignment.flatten_ext_coords();
        let mut out = Array2::<f64>::zeros((n, q));
        for row in 0..n {
            for col in 0..q {
                out[[row, col]] = flat[row * q + col];
            }
        }
        out
    }

    fn ext_coord_manifold(&self) -> LatentManifold {
        let mut parts = Vec::with_capacity(self.assignment.row_block_dim());
        for _ in 0..self.assignment.assignment_coord_dim() {
            parts.push(LatentManifold::Euclidean);
        }
        let mut any_constrained = false;
        for coord in &self.assignment.coords {
            if coord.manifold().is_euclidean() {
                for _ in 0..coord.latent_dim() {
                    parts.push(LatentManifold::Euclidean);
                }
            } else {
                any_constrained = true;
                parts.push(coord.manifold().clone());
            }
        }
        if any_constrained {
            LatentManifold::Product(parts)
        } else {
            LatentManifold::Euclidean
        }
    }

    fn apply_sae_riemannian_geometry(&self, sys: &mut ArrowSchurSystem) {
        let manifold = self.ext_coord_manifold();
        if manifold.is_euclidean() {
            return;
        }
        let ext = self.ext_coord_matrix();
        let latent =
            LatentCoordValues::from_matrix_with_manifold(ext.view(), LatentIdMode::None, manifold);
        sys.apply_riemannian_latent_geometry(&latent);
    }

    /// Numerical rank of a symmetric matrix: the count of eigenvalues
    /// exceeding `tol · max_eig`, with `tol = 1e-9` (the conventional
    /// relative spectral cutoff used elsewhere in the codebase).
    ///
    /// Used to count the penalised dimension of each atom's `smooth_penalty`
    /// `S_k` so the REML criterion's `−½·p·rank(S)·log λ_smooth` Occam term
    /// uses the *effective* penalty rank rather than the ambient basis size
    /// (a thin-plate / B-spline penalty has a non-trivial null space).
    fn symmetric_rank(s: &Array2<f64>) -> Result<usize, String> {
        let m = s.ncols();
        if m == 0 {
            return Ok(0);
        }
        // Symmetrise defensively — `smooth_penalty` is conceptually symmetric
        // but may be stored with tiny asymmetry from assembly arithmetic.
        let mut sym = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
            }
        }
        let (evals, _evecs) = sym
            .eigh(Side::Lower)
            .map_err(|e| format!("SaeManifoldTerm::symmetric_rank: eigh failed: {e}"))?;
        let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        if !(max_eig > 0.0) {
            return Ok(0);
        }
        let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_eig;
        Ok(evals.iter().filter(|&&v| v > tol).count())
    }

    /// True REML criterion for the SAE term at a FIXED ρ.
    ///
    /// Runs the inner `(t, β)` arrow-Schur Newton solve to convergence at the
    /// supplied ρ (with NO in-loop ARD update — ρ is owned by the engine),
    /// then forms the Laplace/REML cost
    ///
    /// ```text
    /// V(ρ) = ℓ_pen(t̂, β̂; ρ) + ½ log|H(t̂, β̂; ρ)|
    ///        − ½ · p · (Σ_k rank S_k) · log λ_smooth
    /// ```
    ///
    /// where `ℓ_pen = loss.total()` is the penalised objective at the inner
    /// optimum and `½ log|H|` is the Laplace normaliser. `H` is the joint
    /// `(t, β)` Hessian assembled by the arrow-Schur system; its `H_tt` block
    /// carries `α = exp(log_ard)` on its diagonal, so as α grows `½ log|H|`
    /// rises while the `−½·n·log α` already inside `loss.ard` falls — their
    /// balance IS the effective-dof term that the deleted `α = n/‖t‖²` rule
    /// dropped, which is why the criterion needs no clamp to stay finite on a
    /// collapsing axis.
    ///
    /// The final `−½·p·rank(S)·log λ_smooth` term is the smoothing-penalty
    /// normaliser `−½ log|λ S|_+` restricted to its ρ-dependent part: `S_k` is
    /// shared across all `p` decoder output channels (the `⊗ I_p` Kronecker
    /// structure), so `log|λ S|_+ = p·rank(S)·log λ + p·log|S|_+`, and the
    /// `½ p·log|S|_+` piece is ρ-independent. ALL ρ-independent additive
    /// constants (the `2π` Laplace constant, the base `½ p·log|S|_+` penalty
    /// logdet, the assignment-prior normaliser) are DROPPED here: they shift
    /// `V` by a constant and do not affect the ρ-argmin the engine seeks.
    ///
    /// Returns `(V, loss)` so the engine can both rank ρ and surface the inner
    /// loss breakdown.
    pub fn reml_criterion(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        self.reml_criterion_with_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    fn reml_criterion_with_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.streaming {
            let mut rho_fixed = rho.clone();
            let loss = self.run_joint_fit_arrow_schur(
                target,
                &mut rho_fixed,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            let extra_penalty_energy = match registry {
                Some(reg) => self
                    .reml_extra_penalty_value_total(reg)
                    .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?,
                None => 0.0,
            };
            Ok((loss.total() + extra_penalty_energy, loss))
        } else {
            let (v, loss, _cache) = self.reml_criterion_with_cache_refine_policy(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                refine_progress_extension,
            )?;
            Ok((v, loss))
        }
    }

    /// As [`Self::reml_criterion`], but also returns the converged undamped
    /// `ArrowFactorCache` so callers (the EFS fixed-point step) can read the
    /// selected-inverse traces `(H⁻¹)_tt` / `(H⁻¹)_ββ` without re-factoring.
    /// The cache is the single shared O(K³) Direct factor; both the
    /// log-determinant criterion and the Fellner-Schall ρ-step consume it.
    pub fn reml_criterion_with_cache(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        self.reml_criterion_with_cache_refine_policy(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            true,
        )
    }

    fn reml_criterion_with_cache_refine_policy(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        refine_progress_extension: bool,
    ) -> Result<(f64, SaeManifoldLoss, ArrowFactorCache), String> {
        let admission_plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !admission_plan.direct_logdet_admitted() {
            return Err(format!(
                "SaeManifoldTerm::reml_criterion_with_cache: predicted working set {} bytes exceeds budget {} bytes for dense evidence cache; shape n={},p={},K={}; cost-only streaming route is required",
                admission_plan.estimated_direct_peak_bytes,
                admission_plan.in_core_budget_bytes,
                self.n_obs(),
                self.output_dim(),
                self.k_atoms()
            ));
        }
        // 1. Run the inner (t, β) Newton solve to convergence at FIXED ρ.
        //    `run_joint_fit_arrow_schur` no longer touches ρ.
        let mut rho_fixed = rho.clone();
        let mut loss = self.run_joint_fit_arrow_schur(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;

        // 2. Drive the inner (t, β) solve to the KKT/step-converged optimum and
        //    take one final UNDAMPED factor there to obtain the joint Hessian
        //    log-determinant. We force ridge = 0 and the dense `Direct` Schur
        //    mode so `arrow_log_det_from_cache` returns the exact
        //    `log|H| = Σ_i log|H_tt^(i)| + log|Schur_β|` (it rejects damped
        //    factors and InexactPCG caches, which have no dense Schur factor).
        //    This is the same evidence convention the main GAM REML path uses.
        //    The shared `converge_inner_for_undamped_logdet` driver guarantees
        //    the per-row `H_tt^(i)` blocks are PD at the converged optimum so
        //    the undamped (`ridge = 0`) factorization succeeds — the streaming
        //    log-det path reuses the identical driver so both rank the same
        //    converged Laplace optimum and stay bit-identical.
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &options,
            refine_progress_extension,
        )?;
        self.record_evidence_gauge_deflation_count(cache.gauge_deflated_directions)?;
        loss.evidence_gauge_deflated_directions = cache.gauge_deflated_directions;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None at \
             ridge=0 Direct mode (no dense Schur factor); the joint Hessian log-det is \
             required for the Laplace normaliser"
                .to_string()
        })?;

        // 3. Smoothing-penalty Occam term `−½·Σ_k r_k·rank(S_k)·log λ_smooth`
        //    plus the profiled-frame evidence-dimension correction
        //    `+½·Σ_k r_k·(p−r_k)·log λ_smooth` (issue #972). On the full-`B` path
        //    (`r_k == p`, no frames) this is exactly the historical
        //    `½·p·(Σ rank S_k)·log λ_smooth`, so the small-model criterion is
        //    unchanged. The single seam is `reml_occam_term`, shared with the
        //    streaming path so both rank the identical Laplace dimension count.
        let occam = self.reml_occam_term(rho)?;

        // Decoder-block analytic-penalty energy (#671/#672). The inner solve
        // descended this energy (it enters `gb`/`hbb`) but it had no native
        // `loss.*` representative, so the Laplace criterion `v` was scoring a
        // different objective than the one minimized. Add the converged
        // decoder-penalty value so the ρ-sweep ranks the same penalized
        // deviance. Excludes the Psi-tier ARD/assignment penalties already
        // accounted for in `loss.total()` (see
        // `analytic_decoder_penalty_value_total`).
        // Extra analytic-penalty energy (#671/#737). Decoder-block penalties and
        // coordinate-tier isometry enter the inner solve but have no `loss.*`
        // representative, so the Laplace criterion must add them explicitly to
        // rank the same penalized deviance the Newton solve descends.
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?,
            None => 0.0,
        };

        let v = loss.total() + extra_penalty_energy + 0.5 * log_det - occam;
        Ok((v, loss, cache))
    }

    fn record_evidence_gauge_deflation_count(&mut self, count: usize) -> Result<(), String> {
        match self.expected_evidence_gauge_deflated_directions {
            Some(expected) if expected != count => Err(format!(
                "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count changed \
                 within one optimization (expected {expected}, got {count}); this is a structural \
                 quotient-dimension event, refusing to compare Laplace normalizers"
            )),
            Some(_) => Ok(()),
            None => {
                self.expected_evidence_gauge_deflated_directions = Some(count);
                Ok(())
            }
        }
    }

    fn is_undamped_evidence_row_non_pd(err: &ArrowSchurError) -> bool {
        matches!(
            err,
            ArrowSchurError::PerRowFactorFailed { reason, .. }
                if reason.contains("H_tt is non-PD at base ridge")
                    && reason.contains("evidence mode preserves the genuine Cholesky")
        )
    }

    /// Drive the inner `(t, β)` Newton solve to the KKT/step-converged optimum
    /// and return the final UNDAMPED (`ridge = 0`) joint-Hessian factor cache.
    ///
    /// The Laplace normaliser `½log|H|` is only the correct REML criterion at
    /// the inner optimum `(t̂, β̂)`, so the criterion must refine the inner state
    /// until either the KKT gradient or the undamped Newton step meets tolerance
    /// before factoring. Crucially, **at the converged optimum the per-row
    /// `H_tt^(i)` blocks are PD**, so the undamped (`ridge = 0`) factorization
    /// succeeds; an off-optimum iterate (e.g. the initial seed, or a state
    /// stopped after only `inner_max_iter` steps) can have an indefinite /
    /// rank-deficient per-row block (`p_out = 1` → rank-1 `JᵀJ`, softmax
    /// assignment-sparsity negative logit curvature) that surfaces
    /// `PerRowFactorFailed` from the undamped `factor_one_row`. Both the dense
    /// (`reml_criterion_with_cache`) and the streaming
    /// (`reml_criterion_streaming_exact`) evidence paths route through this same
    /// driver, so they converge to the identical inner state and their
    /// `ridge = 0` log-determinants stay bit-identical (#847).
    fn converge_inner_for_undamped_logdet(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        loss: &mut SaeManifoldLoss,
        options: &ArrowSolveOptions,
        refine_progress_extension: bool,
    ) -> Result<ArrowFactorCache, String> {
        // `inner_max_iter == 0` is a genuine FREEZE of the inner `(t, β)` state
        // — a verbatim warm-start reuse, not a convergence request (gam#577/#579,
        // #850). The convergence/refinement loop below MUST NOT run even one
        // Newton step in that case (the old `inner_max_iter.max(1)` floor moved
        // β off the seed), so we factor exactly once at the frozen iterate and
        // return that undamped cache without invoking the stationarity gate.
        // The caller has already run `run_joint_fit_arrow_schur(..., 0, ...)`,
        // which left the seed untouched, so `self` is at the warm-start β here.
        if inner_max_iter == 0 {
            let sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            let factored = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // The frozen-state Newton step (factored.0, factored.1) is discarded
            // — only the undamped factor cache (factored.2) is consumed for the
            // log-det / selected-inverse traces; β stays at the warm-start seed.
            return Ok(factored.2);
        }
        let mut total_inner_iter = inner_max_iter;
        let accepted_base_refine_iter = inner_max_iter.max(1).saturating_mul(16).max(64);
        let value_probe_base_refine_iter = inner_max_iter.max(1).saturating_mul(4).max(16);
        let base_refine_iter = if refine_progress_extension {
            accepted_base_refine_iter
        } else {
            value_probe_base_refine_iter
        };
        let progress_refine_iter = if refine_progress_extension {
            inner_max_iter.max(1).saturating_mul(64).max(256)
        } else {
            base_refine_iter
        };
        let mut previous_refine_grad_norm: Option<f64> = None;
        let mut saw_refine_progress = false;
        loop {
            let sys = self
                .assemble_arrow_schur(target, rho, registry)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
            // Evidence-only factorization: the Newton step (Δt, Δβ) is discarded
            // and only the factor cache is consumed — the exact undamped log-det
            // and the selected-inverse traces. As ρ sweeps to extremes (e.g. a
            // wide ARD-α sweep), H_tt is genuinely PD but can be ill-conditioned;
            // the standard Direct guard rejects that to protect Newton-step
            // accuracy, but the log-det is exact from diag(L) regardless of the
            // condition number and the traces only need the (PD) factor. So
            // tolerate the ill-conditioning rejection here (a genuine non-PD pivot
            // still errors). The cache stays undamped at ridge=0, so
            // `arrow_log_det_from_cache` remains exact.
            // The exact KKT stationarity residual is the joint gradient
            // ‖g‖ = √(Σ_i ‖g_t^(i)‖² + ‖g_β‖²), read straight off the assembled
            // system. Unlike the Newton step Δ = H⁻¹g, the gradient is
            // factorisation-independent: it is NOT amplified by an inverse, so a
            // genuinely stationary but ill-conditioned fit (tiny g, possibly large
            // Δ in a flat direction) is correctly recognised as converged. The
            // `with_ill_conditioning_tolerated` Direct factor below documents that
            // its Δ may be inaccurate in exactly those flat directions, so using Δ
            // alone as the convergence gate would falsely reject healthy fits.
            let grad_norm_sq: f64 = sys
                .rows
                .iter()
                .map(|row| row.gt.iter().map(|&v| v * v).sum::<f64>())
                .sum::<f64>()
                + sys.gb.iter().map(|&v| v * v).sum::<f64>();
            let grad_norm = grad_norm_sq.sqrt();
            let iterate_scale = self.inner_iterate_scale();
            // Relative parameter-step tolerance for Δ (well-conditioned charts)
            // and a scaled KKT-gradient tolerance. Convergence is accepted on
            // EITHER a small KKT gradient OR a small undamped Newton step: SAE
            // manifold fits contain gauge-like coordinate/decoder directions (the
            // circle's rotation gauge, decoder column-space rotations) where the
            // shared-block Hessian is near-singular, so the undamped step can stay
            // large in that flat direction even at a genuine stationary point; the
            // gradient, which is not amplified by the inverse, recognises it. With
            // the isometry Gauss-Newton block now a coherent PSD pullback (no
            // indefinite Schur pivot), the inner solve reaches true stationarity,
            // so the gradient tolerance is a standard relative KKT residual rather
            // than the 0.1.154-regression band-aid (3e-3) that masked the
            // non-convergence the indefinite curvature caused.
            let step_tolerance = SAE_MANIFOLD_INNER_STEP_REL_TOL * iterate_scale;
            let grad_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
            if !grad_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: undamped inner KKT residual is non-finite \
                     at the inner optimum (‖g‖²={grad_norm_sq}); the joint Hessian \
                     factorisation is degenerate at this ρ"
                ));
            }
            let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
                match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options) {
                    Ok(factored) => factored,
                    Err(err) if Self::is_undamped_evidence_row_non_pd(&err) => {
                        if grad_norm <= grad_tolerance {
                            return Err(format!(
                                "SaeManifoldTerm::reml_criterion: stationary undamped evidence \
                                 factorization still has a non-PD per-row H_tt block \
                                 (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}); {err}"
                            ));
                        }
                        let refine_limit = Self::refine_iteration_limit(
                            total_inner_iter,
                            base_refine_iter,
                            progress_refine_iter,
                            previous_refine_grad_norm,
                            grad_norm,
                            saw_refine_progress,
                        );
                        if total_inner_iter >= refine_limit {
                            return Err(format!(
                                "SaeManifoldTerm::reml_criterion: undamped evidence \
                                 factorization hit a non-PD per-row H_tt block before KKT \
                                 stationarity (‖g‖={grad_norm:.6e}, tol {grad_tolerance:.6e}) \
                                 and the refinement budget was exhausted after \
                                 {total_inner_iter} inner iterations; {err}"
                            ));
                        }
                        let remaining = refine_limit - total_inner_iter;
                        let refine_iter = inner_max_iter.max(1).min(remaining);
                        saw_refine_progress |=
                            Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
                        previous_refine_grad_norm = Some(grad_norm);
                        *loss = self.run_joint_fit_arrow_schur(
                            target,
                            rho_fixed,
                            registry,
                            refine_iter,
                            learning_rate,
                            ridge_ext_coord,
                            ridge_beta,
                        )?;
                        total_inner_iter += refine_iter;
                        continue;
                    }
                    Err(err) => {
                        return Err(format!("SaeManifoldTerm::reml_criterion: {err}"));
                    }
                };
            // The Laplace normaliser ½log|H| is only the correct REML criterion at
            // the inner optimum (t̂, β̂). Convergence is judged by EITHER a small
            // gradient (KKT stationarity) OR a small undamped Newton step; the
            // solve is only rejected as non-converged when BOTH are large, i.e.
            // the iterate is neither stationary nor about to move negligibly. That
            // disjunction is what keeps an ill-conditioned-but-stationary fit
            // (small g, large Δ) from being rejected while still refusing to rank
            // an off-optimum Laplace criterion that is genuinely mid-flight.
            let step_norm_sq: f64 = delta_t.iter().map(|&v| v * v).sum::<f64>()
                + delta_beta.iter().map(|&v| v * v).sum::<f64>();
            if !step_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: undamped inner residual is non-finite at \
                     the inner optimum (‖Δ‖²={step_norm_sq}, ‖g‖²={grad_norm_sq}); the joint \
                     Hessian factorisation is degenerate at this ρ"
                ));
            }
            let step_norm = step_norm_sq.sqrt();
            let quotient_step_norm_sq =
                self.quotient_newton_step_norm_sq(delta_t.view(), delta_beta.view(), step_norm_sq)?;
            let quotient_step_norm = quotient_step_norm_sq.sqrt();
            if grad_norm <= grad_tolerance || quotient_step_norm <= step_tolerance {
                return Ok(cache);
            }
            let refine_limit = Self::refine_iteration_limit(
                total_inner_iter,
                base_refine_iter,
                progress_refine_iter,
                previous_refine_grad_norm,
                grad_norm,
                saw_refine_progress,
            );
            if total_inner_iter >= refine_limit {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                     neither the KKT gradient ‖g‖={grad_norm:.6e} (tol {grad_tolerance:.6e}) nor \
                     the quotient Newton step ‖Π⊥gauge Δ‖={quotient_step_norm:.6e} \
                     (raw ‖Δ‖={step_norm:.6e}, tol {step_tolerance:.6e}) met \
                     tolerance after {total_inner_iter} inner iterations. Refusing to rank an \
                     off-optimum Laplace criterion."
                ));
            }
            let remaining = refine_limit - total_inner_iter;
            let refine_iter = inner_max_iter.max(1).min(remaining);
            saw_refine_progress |=
                Self::refine_round_made_progress(previous_refine_grad_norm, grad_norm);
            previous_refine_grad_norm = Some(grad_norm);
            *loss = self.run_joint_fit_arrow_schur(
                target,
                rho_fixed,
                registry,
                refine_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )?;
            total_inner_iter += refine_iter;
        }
    }

    fn refine_iteration_limit(
        total_inner_iter: usize,
        base_refine_iter: usize,
        progress_refine_iter: usize,
        previous_grad_norm: Option<f64>,
        grad_norm: f64,
        saw_refine_progress: bool,
    ) -> usize {
        // Flat affine-gauge valleys can keep crawling productively after the
        // historical base budget. Extend only when the measured KKT residual has
        // shown a real finite round-to-round drop; true stalls end at the base
        // work budget (#968/#1029). Value-order probes pass the base budget as
        // their progress budget, so this branch cannot make probes expensive.
        if total_inner_iter < base_refine_iter {
            return base_refine_iter;
        }
        let making_progress =
            saw_refine_progress || Self::refine_round_made_progress(previous_grad_norm, grad_norm);
        if making_progress && grad_norm.is_finite() {
            progress_refine_iter
        } else {
            base_refine_iter
        }
    }

    fn refine_round_made_progress(previous_grad_norm: Option<f64>, grad_norm: f64) -> bool {
        previous_grad_norm
            .is_some_and(|prev| prev.is_finite() && grad_norm.is_finite() && grad_norm < prev)
    }

    fn outer_gradient_arrow_solver<'a>(
        &'a self,
        cache: &'a ArrowFactorCache,
    ) -> Result<DeflatedArrowSolver<'a>, String> {
        let Err(conditioning_err) = Self::outer_gradient_conditioning_error(cache) else {
            return Ok(DeflatedArrowSolver::plain(cache));
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(conditioning_err);
        };
        if !(max_pivot.is_finite() && max_pivot > 0.0) {
            return Err(conditioning_err);
        }

        let full_len = cache.delta_t_len() + cache.k;
        let mut raw_gauges = Vec::new();
        for gauge in self.dense_step_gauge_vectors()? {
            if gauge.len() != full_len {
                continue;
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            raw_gauges.push(gauge);
        }
        if raw_gauges.is_empty() {
            return Err(conditioning_err);
        }

        let mut gauge_span: Vec<Array1<f64>> = Vec::new();
        for mut gauge in raw_gauges {
            for basis in &gauge_span {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in gauge.iter_mut() {
                *value *= inv_norm;
            }
            gauge_span.push(gauge);
        }
        if gauge_span.is_empty() {
            return Err(conditioning_err);
        }

        let span_rank = gauge_span.len();
        let mut h_span = Array2::<f64>::zeros((span_rank, span_rank));
        for col in 0..span_rank {
            let h_gauge = match apply_cached_arrow_hessian(
                cache,
                gauge_span[col].slice(s![..cache.delta_t_len()]),
                gauge_span[col].slice(s![cache.delta_t_len()..]),
            ) {
                Ok(value) => value,
                Err(_) => return Err(conditioning_err),
            };
            let h_flat = flatten_arrow_parts(h_gauge.t.view(), h_gauge.beta.view());
            for row in 0..span_rank {
                h_span[[row, col]] = gauge_span[row].dot(&h_flat);
            }
        }
        for row in 0..span_rank {
            for col in 0..row {
                let sym = 0.5 * (h_span[[row, col]] + h_span[[col, row]]);
                h_span[[row, col]] = sym;
                h_span[[col, row]] = sym;
            }
        }
        let (evals, evecs) = h_span
            .eigh(Side::Lower)
            .map_err(|_| conditioning_err.clone())?;
        let strict_gauge_floor = SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR * max_pivot;
        let fallback_gauge_floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR.sqrt() * max_pivot;
        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for eig_idx in 0..evals.len() {
            let rayleigh = evals[eig_idx];
            if !(rayleigh.is_finite() && rayleigh <= strict_gauge_floor) {
                continue;
            }
            let mut direction = Array1::<f64>::zeros(full_len);
            for basis_idx in 0..span_rank {
                let coeff = evecs[[basis_idx, eig_idx]];
                for row in 0..full_len {
                    direction[row] += coeff * gauge_span[basis_idx][row];
                }
            }
            let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
            if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for value in direction.iter_mut() {
                *value *= inv_norm;
            }
            orthonormal.push(direction);
        }
        if orthonormal.is_empty() {
            let mut best_idx = None;
            let mut best_rayleigh = f64::INFINITY;
            for eig_idx in 0..evals.len() {
                let rayleigh = evals[eig_idx];
                if rayleigh.is_finite()
                    && rayleigh < best_rayleigh
                    && rayleigh <= fallback_gauge_floor
                {
                    best_idx = Some(eig_idx);
                    best_rayleigh = rayleigh;
                }
            }
            if let Some(eig_idx) = best_idx {
                let mut direction = Array1::<f64>::zeros(full_len);
                for basis_idx in 0..span_rank {
                    let coeff = evecs[[basis_idx, eig_idx]];
                    for row in 0..full_len {
                        direction[row] += coeff * gauge_span[basis_idx][row];
                    }
                }
                let norm_sq = direction.iter().map(|v| v * v).sum::<f64>();
                if norm_sq.is_finite() && norm_sq > 1.0e-24 {
                    let inv_norm = norm_sq.sqrt().recip();
                    for value in direction.iter_mut() {
                        *value *= inv_norm;
                    }
                    orthonormal.push(direction);
                }
            }
        }
        if orthonormal.is_empty() {
            return Err(conditioning_err);
        }

        // Quotient-geometry gauge fixing: add stiffness only along the closed-form
        // gauge orbit (Faddeev-Popov style). Components orthogonal to that orbit
        // are identical to the original inverse solve, while gauge components are
        // bounded at the Hessian scale `max_pivot`.
        DeflatedArrowSolver::from_orthonormal_gauges(cache, orthonormal, max_pivot)
            .map_err(|_| conditioning_err)
    }

    fn outer_gradient_conditioning_error(cache: &ArrowFactorCache) -> Result<(), String> {
        let pivot = arrow_factor_min_pivot(cache);
        let Some(min_pivot) = pivot.min_pivot else {
            return Err(
                "analytic outer gradient undefined at this rho: joint Hessian numerically \
                 singular (no cached Cholesky pivots)"
                    .to_string(),
            );
        };
        let Some(max_pivot) = arrow_factor_max_pivot(cache) else {
            return Err(
                "analytic outer gradient undefined at this rho: joint Hessian numerically \
                 singular (no cached Cholesky pivot scale)"
                    .to_string(),
            );
        };
        let ratio = min_pivot / max_pivot;
        if min_pivot.is_finite()
            && max_pivot.is_finite()
            && max_pivot > 0.0
            && ratio.is_finite()
            && ratio >= SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR
        {
            return Ok(());
        }
        Err(format!(
            "analytic outer gradient undefined at this rho: joint Hessian numerically singular \
             (min/max pivot ratio {ratio:.3e} < floor {floor:.3e}; min pivot {min_pivot:.3e}, \
             max pivot {max_pivot:.3e})",
            floor = SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR,
        ))
    }

    /// Smoothing-penalty Occam normalizer `−½ Σ_k r_k·rank(S_k)·log λ_smooth`
    /// PLUS the profiled-frame evidence-dimension term `½ Σ_k r_k·(p−r_k)·log
    /// λ_smooth` (issue #972).
    ///
    /// On the full-`B` path every atom's frame rank `r_k == p`, so the first
    /// piece reduces to the historical `½ p·(Σ rank S_k)·log λ_smooth` and the
    /// Grassmann term is zero — bit-for-bit unchanged. When a frame is active the
    /// decoder coordinates `C_k` carry the `⊗ I_{r_k}` Kronecker structure (the
    /// smoothing penalty `S_k` now acts on `r_k` channels, not `p`), so the
    /// penalty-logdet normalizer uses `r_k·rank(S_k)`; and the `r_k·(p−r_k)`
    /// frame degrees of freedom profiled OUT of the border are counted explicitly
    /// in the Laplace dimension accounting (evidence honesty) so the criterion
    /// cannot buy a free evidence boost by hiding decoder freedom in the frame.
    fn reml_occam_term(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        let mut penalized_channel_dim = 0usize;
        for atom in &self.atoms {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            // Penalized decoder dimension: `r_k` coordinate channels carry the
            // `S_k` roughness penalty (full-`B` path ⇒ `r_k == p`).
            penalized_channel_dim += atom.border_frame_rank() * rank_s;
        }
        // Profiled Grassmann dimensions enter the Laplace evidence dimension
        // count with the OPPOSITE sign of the penalty Occam term (they are
        // free, unpenalized-by-`S` profiled directions), so `−occam` adds
        // `+½ Σ r(p−r) log λ` to the criterion `V` — the honesty correction.
        let grassmann_dim = self.grassmann_evidence_dimension();
        let occam_penalty = 0.5 * (penalized_channel_dim as f64) * rho.log_lambda_smooth;
        let frame_dim_term = 0.5 * (grassmann_dim as f64) * rho.log_lambda_smooth;
        // `V = … − occam`, so we want the net occam to SUBTRACT the penalty
        // normalizer and ADD the frame-dimension count. Returning
        // `occam_penalty − frame_dim_term` achieves that after the caller's
        // `− occam`.
        Ok(occam_penalty - frame_dim_term)
    }

    fn reml_occam_log_lambda_smooth_derivative(&self) -> Result<f64, String> {
        let mut penalized_channel_dim = 0usize;
        for atom in &self.atoms {
            let rank_s = Self::symmetric_rank(&atom.smooth_penalty)?;
            penalized_channel_dim += atom.border_frame_rank() * rank_s;
        }
        let grassmann_dim = self.grassmann_evidence_dimension();
        Ok(0.5 * ((penalized_channel_dim as f64) - (grassmann_dim as f64)))
    }

    pub fn reml_criterion_streaming_exact(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(f64, SaeManifoldLoss), String> {
        let mut rho_fixed = rho.clone();
        let mut loss = self.run_joint_fit_arrow_schur(
            target,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        // Drive the inner (t, β) state to the SAME KKT/step-converged optimum the
        // dense `reml_criterion_with_cache` reaches before factoring. At that
        // optimum the per-row `H_tt^(i)` blocks are PD, so the undamped
        // (`ridge_t = 0`) streaming factorization in `streaming_exact_arrow_log_det`
        // succeeds — without this, a state stopped after only `inner_max_iter`
        // steps can leave a rank-deficient / indefinite row block (`p_out = 1` →
        // rank-1 `JᵀJ`, softmax negative-logit curvature) that surfaces
        // `PerRowFactorFailed` at base ridge 0. Sharing the driver also keeps the
        // streaming and dense log-determinants bit-identical (#847).
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        // The dense factor cache from convergence is surplus here — the streaming
        // path recomputes the (bit-identical) log-determinant chunk-by-chunk in
        // `streaming_exact_arrow_log_det` to bound peak memory — so it is dropped.
        let converged_cache = self.converge_inner_for_undamped_logdet(
            target,
            rho,
            &mut rho_fixed,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &options,
            true,
        )?;
        drop(converged_cache);
        let log_det = self.streaming_exact_arrow_log_det(target, rho, registry)?;
        let occam = self.reml_occam_term(rho)?;
        // Extra analytic-penalty energy (#671/#737), matching the full-batch
        // `reml_criterion_with_cache` path so streaming and dense criteria rank
        // the identical penalized objective.
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::reml_criterion_streaming_exact: {err}"))?,
            None => 0.0,
        };
        Ok((
            loss.total() + extra_penalty_energy + 0.5 * log_det - occam,
            loss,
        ))
    }

    pub fn streaming_exact_arrow_log_det(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<f64, String> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det: target must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if plan.estimated_dense_schur_bytes > plan.in_core_budget_bytes {
            return Err(format!(
                "SaeManifoldTerm::streaming_exact_arrow_log_det: predicted dense reduced Schur {} bytes exceeds budget {} bytes; cost-only matrix-free route is required",
                plan.estimated_dense_schur_bytes, plan.in_core_budget_bytes
            ));
        }
        let n_total = self.n_obs();
        let chunk_size = plan.chunk_size.min(n_total.max(1));
        // #972 / #977 T1: the reduced β-Schur is over the FACTORED border when
        // frames are active (each chunk inherits the frames via
        // `materialize_chunk`, so every `chunk_schur` is `border_dim²`), matching
        // the dense path's factored log-det. Full-`B` ⇒ `border_dim == beta_dim`.
        let border_dim = if self.frames_active() {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        let mut schur_acc = Array2::<f64>::zeros((border_dim, border_dim));
        let mut log_det_tt = 0.0_f64;
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let mut start = 0usize;
        while start < n_total {
            let end = (start + chunk_size).min(n_total);
            let penalty_scale = (end - start) as f64 / n_total as f64;
            let chunk_logits = self.assignment.logits.slice(s![start..end, ..]).to_owned();
            let chunk_coords: Vec<Array2<f64>> = self
                .assignment
                .coords
                .iter()
                .map(|coord| coord.as_matrix().slice(s![start..end, ..]).to_owned())
                .collect();
            let mut chunk = self.materialize_chunk(chunk_logits, chunk_coords)?;
            // #991: chunk terms inherit the row's design honesty weight slice
            // (global mean-1 normalization preserved — NOT re-normalized per
            // chunk — so the per-chunk sums reconstruct the global weighted
            // objective exactly).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let z_chunk = target.slice(s![start..end, ..]);
            let sys = chunk
                .assemble_arrow_schur_scaled(z_chunk, rho, registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let (chunk_log_det_tt, chunk_schur) = streaming
                .reduced_schur_and_log_det_tt(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..border_dim {
                for col in 0..border_dim {
                    schur_acc[[row, col]] += chunk_schur[[row, col]];
                }
            }
            start = end;
        }
        let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur_acc, &options)
            .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
        Ok(log_det_tt + log_det_schur)
    }

    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`.
    ///
    /// On a *periodic* (Circle) axis the relevant statistic is the von-Mises
    /// energy-equivalent `Σ_i 2/α·V(t_i) = Σ_i (2/κ²)(1−cos κ t_i)` (independent
    /// of α), so that `½·α·sumsq == Σ_i V(t_i)` matches `ard_value`. This keeps
    /// the Mackay/Fellner–Schall fixed point `α ← n / (sumsq + tr H⁻¹)`
    /// consistent with the actual periodic prior energy rather than the
    /// origin-dependent raw `t²`.
    fn ard_coord_sumsq(&self) -> Vec<Array1<f64>> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            let d = coord.latent_dim();
            let periods = coord.effective_axis_periods();
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
                }
            }
            out.push(sq);
        }
        out
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`); each per-row block is laid out as
    /// `[logit scalars…, then per-active-atom coord axes…]`. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact active-set mode (JumpReLU / large-K
    ///   softmax-IBP truncation). For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = assignment_dim + Σ d_k`; atom `k`'s coord block sits at the
    ///   fixed full-row offset `coord_offsets[k]` after the assignment chart.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = cache.latent_block_inverse_diagonal()?;
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    fn ard_log_precision_explicit_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atoms but term has {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        let n = self.n_obs() as f64;
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let mut atom_out = Array1::<f64>::zeros(rho.log_ard[atom_idx].len());
            if rho.log_ard[atom_idx].is_empty() {
                out.push(atom_out);
                continue;
            }
            if rho.log_ard[atom_idx].len() != d {
                return Err(format!(
                    "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                    rho.log_ard[atom_idx].len()
                ));
            }
            let periods = coord.effective_axis_periods();
            for axis in 0..d {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom_idx][axis]);
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let t = coord.row(row)[axis];
                    energy_deriv += ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_deriv = match period {
                    None => -0.5 * n,
                    Some(p) => {
                        let kappa = std::f64::consts::TAU / p;
                        let eta = alpha / (kappa * kappa);
                        let i0 = bessel_i0(eta);
                        let i1 = bessel_i1(eta);
                        n * eta * (-1.0 + i1 / i0)
                    }
                };
                atom_out[axis] = energy_deriv + normalizer_deriv;
            }
            out.push(atom_out);
        }
        Ok(out)
    }

    fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            traces[k][axis] +=
                                0.5 * inv_diag[row_base + block_start + axis] * prior.hess.max(0.0);
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            traces[k][axis] +=
                                0.5 * inv_diag[row_base + block_start + axis] * prior.hess.max(0.0);
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Decoder smoothness penalty quadratic form `Σ_k Σ_oc B_k[:,oc]ᵀ S_k B_k[:,oc]`.
    ///
    /// This is `βᵀ (⊕_k S_k ⊗ I_p) β` — the un-scaled (λ-free) penalty energy
    /// in the flat β layout, the denominator of the λ_smooth Fellner-Schall
    /// update. `S_k` is symmetrised defensively (as the assembler does).
    fn decoder_smoothness_quadratic_form(&self) -> f64 {
        // `Σ_k Σ_oc B_k[:,oc]ᵀ ½(S_k+S_kᵀ) B_k[:,oc]` = `Σ_k <B_k, ½(S_k+S_kᵀ)·B_k>`.
        // The per-atom `½(S+Sᵀ)·B_k` GEMMs are independent, so they ride the
        // multi-GPU batched smoothness GEMM (uniform-shape tiles across every
        // device) with an exact per-atom CPU fallback.
        let sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sb_all = batched_smooth_sb(&sb_inputs, true);
        let mut acc = 0.0_f64;
        for (atom, sb) in self.atoms.iter().zip(sb_all.iter()) {
            acc += (&atom.decoder_coefficients * sb).sum();
        }
        acc
    }

    /// Effective penalized dof of the decoder smoothness penalty:
    /// `tr(S_β⁻¹ · M)` with `M = ⊕_k (λ_smooth · S_k) ⊗ I_p` embedded in the
    /// flat β layout, where `S_β⁻¹ = (H⁻¹)_ββ` is the Schur-complement inverse.
    ///
    /// Built per keystone's documented pattern on
    /// [`ArrowFactorCache::schur_inverse_apply`]:
    /// `tr(S_β⁻¹ M) = Σ_col e_colᵀ S_β⁻¹ M e_col`. Column `(k, μ, oc)` of `M`
    /// (global index `off_k + μ·p + oc`) is `λ·S_k[:,μ] ⊗ e_oc` — nonzero only
    /// at `off_k + ν·p + oc` for `ν in 0..M_k` — so we materialise just that
    /// sparse K-vector, apply `S_β⁻¹`, and read back `result[col]`. The
    /// `⊗ I_p` only couples equal `oc`, but `S_β` itself couples channels
    /// through the data-fit block, so all `p` channels are summed (no
    /// channel-block-identity shortcut). Total cost `beta_dim` Schur solves.
    fn decoder_smoothness_effective_dof(
        &self,
        cache: &ArrowFactorCache,
        lambda_smooth: f64,
    ) -> Result<f64, ArrowSchurError> {
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda_smooth * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
    }

    fn decoder_smoothness_effective_dof_with_solver(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        lambda_smooth: f64,
    ) -> Result<f64, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache's β block is the FACTORED border when frames
        // are active (`cache.k == factored_border_dim`), so the smoothness edf
        // trace `tr((H⁻¹)_ββ · M)` is taken over the same factored layout, with
        // `M = ⊕_k (λ S_k) ⊗ I_{r_k}` at the factored offsets (the `U_kᵀU_k = I`
        // collapse means the per-coordinate-channel penalty is `λ S_k`, exactly
        // as in the full-`B` `⊗ I_p` case but with `r_k` channels). On the
        // full-`B` path `frames_active` is false: `out_dim_k = p`, the offsets
        // are `beta_offsets`, and this is bit-for-bit the historical trace.
        let frames_active = self.frames_active();
        let (offsets, out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) = if frames_active {
            let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                self.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (self.beta_offsets(), Box::new(move |_k: usize| p))
        };
        let k = cache.k;
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = offsets[atom_idx];
            let r = out_dim(atom_idx);
            for mu in 0..m {
                for oc in 0..r {
                    let col = off + mu * r + oc;
                    // M[:,col] = λ · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·r+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * r + oc] = lambda_smooth * s_nu_mu;
                    }
                    let zero_t = Array1::<f64>::zeros(cache.delta_t_len());
                    let z = solver.solve(zero_t.view(), m_col.view())?.beta;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
    }

    fn assignment_log_strength_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, String> {
        let hdiag = assignment_prior_log_strength_hdiag(&self.assignment, rho)?;
        if hdiag.is_empty() {
            return Ok(0.0);
        }
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| format!("assignment_log_strength_hessian_trace: {err}"))?;
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let mut trace = 0.0_f64;
        for row in 0..self.n_obs() {
            let row_base = cache.row_offsets[row];
            let assignment_base = row * k_atoms;
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        trace += inv_diag[row_base + pos] * hdiag[assignment_base + atom];
                    }
                }
                None => {
                    for free_idx in 0..assignment_dim {
                        trace += inv_diag[row_base + free_idx] * hdiag[assignment_base + free_idx];
                    }
                }
            }
        }
        Ok(0.5 * trace)
    }

    fn border_channels_for_cache(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeBorderChannel>, String> {
        let p = self.output_dim();
        let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
        let offsets = if frames_active {
            self.factored_beta_offsets()
        } else {
            self.beta_offsets()
        };
        let mut channels = Vec::with_capacity(cache.k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let frame = if frames_active {
                self.frame_output_matrix(atom_idx)
            } else {
                Array2::<f64>::eye(p)
            };
            let r = frame.ncols();
            for basis_col in 0..m {
                for channel in 0..r {
                    let mut output = vec![0.0_f64; p];
                    for out_col in 0..p {
                        output[out_col] = frame[[out_col, channel]];
                    }
                    channels.push(SaeBorderChannel {
                        atom: atom_idx,
                        basis_col,
                        index: offsets[atom_idx] + basis_col * r + channel,
                        output,
                    });
                }
            }
        }
        if channels.len() != cache.k {
            return Err(format!(
                "border channel layout has {} entries but cache border has {}",
                channels.len(),
                cache.k
            ));
        }
        Ok(channels)
    }

    fn row_vars_for_cache_row(
        &self,
        row: usize,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<SaeLocalRowVar>, String> {
        let q_row = cache.row_dims[row];
        let mut vars: Vec<Option<SaeLocalRowVar>> = vec![None; q_row];
        match self.last_row_layout {
            Some(ref layout) => {
                for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                    vars[pos] = Some(SaeLocalRowVar::Logit { atom });
                    let start = layout.coord_starts[row][pos];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
            None => {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let coord_offsets = self.assignment.coord_offsets();
                for atom in 0..assignment_dim {
                    vars[atom] = Some(SaeLocalRowVar::Logit { atom });
                }
                for atom in 0..self.k_atoms() {
                    let start = coord_offsets[atom];
                    let d = self.assignment.coords[atom].latent_dim();
                    for axis in 0..d {
                        vars[start + axis] = Some(SaeLocalRowVar::Coord { atom, axis });
                    }
                }
            }
        }
        vars.into_iter()
            .enumerate()
            .map(|(idx, v)| {
                v.ok_or_else(|| {
                    format!("row_vars_for_cache_row: row {row} position {idx} was not mapped")
                })
            })
            .collect()
    }

    fn atom_second_jets(&self) -> Result<Vec<Array4<f64>>, String> {
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            let jet = if let Some(second) = atom.basis_second_jet.as_ref() {
                second.second_jet(coords.view())?
            } else {
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "logdet_theta_adjoint: atom '{}' has no basis evaluator for second jets",
                        atom.name
                    )
                })?;
                evaluator
                    .second_jet_dyn(coords.view())
                    .ok_or_else(|| {
                        format!(
                            "logdet_theta_adjoint: atom '{}' basis does not expose analytic second jets",
                            atom.name
                        )
                    })??
            };
            let expected = (
                atom.n_obs(),
                atom.basis_size(),
                atom.latent_dim,
                atom.latent_dim,
            );
            if jet.dim() != expected {
                return Err(format!(
                    "logdet_theta_adjoint: atom '{}' second jet shape {:?}, expected {:?}",
                    atom.name,
                    jet.dim(),
                    expected
                ));
            }
            out.push(jet);
        }
        Ok(out)
    }

    fn gate_derivatives_for_row(
        &self,
        row: usize,
        assignments: ArrayView1<'_, f64>,
        vars: &[SaeLocalRowVar],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>), String> {
        let k_atoms = self.k_atoms();
        let q = vars.len();
        let mut dz = vec![vec![0.0_f64; k_atoms]; q];
        let mut d2z = vec![vec![vec![0.0_f64; k_atoms]; q]; q];
        match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                let inv_tau = 1.0 / temperature;
                for (a_idx, var_a) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                        continue;
                    };
                    for k in 0..k_atoms {
                        let indicator = if k == j { 1.0 } else { 0.0 };
                        dz[a_idx][k] = assignments[k] * (indicator - assignments[j]) * inv_tau;
                    }
                }
                for (a_idx, var_a) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: j } = *var_a else {
                        continue;
                    };
                    for (b_idx, var_b) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: l } = *var_b else {
                            continue;
                        };
                        for k in 0..k_atoms {
                            let ikl = if k == l { 1.0 } else { 0.0 };
                            let ikj = if k == j { 1.0 } else { 0.0 };
                            let ijl = if j == l { 1.0 } else { 0.0 };
                            d2z[a_idx][b_idx][k] = assignments[k]
                                * ((ikl - assignments[l]) * (ikj - assignments[j])
                                    - assignments[j] * (ijl - assignments[l]))
                                * inv_tau
                                * inv_tau;
                        }
                    }
                }
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => {
                let prior = ibp_stick_breaking_prior(k_atoms, alpha);
                let inv_tau = 1.0 / temperature;
                for (idx, var) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *var else {
                        continue;
                    };
                    let (_z, d1, d2) =
                        sae_sigmoid_derivatives_from_value(assignments[atom], inv_tau, prior[atom]);
                    dz[idx][atom] = d1;
                    d2z[idx][idx][atom] = d2;
                }
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                let inv_tau = 1.0 / temperature;
                let logits = self.assignment.logits.row(row);
                for (idx, var) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *var else {
                        continue;
                    };
                    if logits[atom] <= threshold {
                        continue;
                    }
                    let (_z, d1, d2) =
                        sae_sigmoid_derivatives_from_value(assignments[atom], inv_tau, 1.0);
                    dz[idx][atom] = d1;
                    d2z[idx][idx][atom] = d2;
                }
            }
        }
        Ok((dz, d2z))
    }

    fn decoded_second_row(
        atom: &SaeManifoldAtom,
        second_jet: &Array4<f64>,
        row: usize,
        axis_a: usize,
        axis_b: usize,
        out: &mut [f64],
    ) {
        out.fill(0.0);
        for basis_col in 0..atom.basis_size() {
            let d2phi = second_jet[[row, basis_col, axis_a, axis_b]];
            if d2phi == 0.0 {
                continue;
            }
            for out_col in 0..atom.output_dim() {
                out[out_col] += d2phi * atom.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    fn row_jets_for_logdet(
        &self,
        row: usize,
        vars: Vec<SaeLocalRowVar>,
        assignments: ArrayView1<'_, f64>,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
    ) -> Result<SaeRowJets, String> {
        let p = self.output_dim();
        let q = vars.len();
        let k_atoms = self.k_atoms();
        let sqrt_row_w = self
            .row_loss_weights
            .as_deref()
            .map_or(1.0, |w| w[row].sqrt());
        let (dz, d2z) = self.gate_derivatives_for_row(row, assignments, &vars)?;

        let mut decoded = vec![vec![0.0_f64; p]; k_atoms];
        let mut d1: Vec<Vec<Vec<f64>>> = self
            .atoms
            .iter()
            .map(|atom| vec![vec![0.0_f64; p]; atom.latent_dim])
            .collect();
        let mut d2: Vec<Vec<Vec<Vec<f64>>>> = self
            .atoms
            .iter()
            .map(|atom| vec![vec![vec![0.0_f64; p]; atom.latent_dim]; atom.latent_dim])
            .collect();
        let mut scratch = vec![0.0_f64; p];
        for k in 0..k_atoms {
            self.atoms[k].fill_decoded_row(row, &mut decoded[k]);
            for axis in 0..self.atoms[k].latent_dim {
                self.atoms[k].fill_decoded_derivative_row(row, axis, &mut d1[k][axis]);
            }
            for axis_a in 0..self.atoms[k].latent_dim {
                for axis_b in 0..self.atoms[k].latent_dim {
                    Self::decoded_second_row(
                        &self.atoms[k],
                        &second_jets[k],
                        row,
                        axis_a,
                        axis_b,
                        &mut scratch,
                    );
                    d2[k][axis_a][axis_b].clone_from_slice(&scratch);
                }
            }
        }

        let mut first = vec![vec![0.0_f64; p]; q];
        for (idx, var) in vars.iter().enumerate() {
            match *var {
                SaeLocalRowVar::Logit { .. } => {
                    for k in 0..k_atoms {
                        let coeff = dz[idx][k] * sqrt_row_w;
                        if coeff == 0.0 {
                            continue;
                        }
                        for out_col in 0..p {
                            first[idx][out_col] += coeff * decoded[k][out_col];
                        }
                    }
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    let coeff = assignments[atom] * sqrt_row_w;
                    for out_col in 0..p {
                        first[idx][out_col] = coeff * d1[atom][axis][out_col];
                    }
                }
            }
        }

        let mut second = vec![vec![vec![0.0_f64; p]; q]; q];
        for a in 0..q {
            for b in 0..q {
                match (vars[a], vars[b]) {
                    (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Logit { .. }) => {
                        for k in 0..k_atoms {
                            let coeff = d2z[a][b][k] * sqrt_row_w;
                            if coeff == 0.0 {
                                continue;
                            }
                            for out_col in 0..p {
                                second[a][b][out_col] += coeff * decoded[k][out_col];
                            }
                        }
                    }
                    (SaeLocalRowVar::Logit { .. }, SaeLocalRowVar::Coord { atom, axis }) => {
                        let coeff = dz[a][atom] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                        }
                    }
                    (SaeLocalRowVar::Coord { atom, axis }, SaeLocalRowVar::Logit { .. }) => {
                        let coeff = dz[b][atom] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d1[atom][axis][out_col];
                        }
                    }
                    (
                        SaeLocalRowVar::Coord {
                            atom: atom_a,
                            axis: axis_a,
                        },
                        SaeLocalRowVar::Coord {
                            atom: atom_b,
                            axis: axis_b,
                        },
                    ) if atom_a == atom_b => {
                        let coeff = assignments[atom_a] * sqrt_row_w;
                        for out_col in 0..p {
                            second[a][b][out_col] = coeff * d2[atom_a][axis_a][axis_b][out_col];
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut beta = vec![vec![0.0_f64; p]; border.len()];
        let mut beta_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        let mut beta_l_deriv = vec![vec![vec![0.0_f64; p]; border.len()]; q];
        for (beta_pos, channel) in border.iter().enumerate() {
            let atom = channel.atom;
            let phi = self.atoms[atom].basis_values[[row, channel.basis_col]];
            let base = assignments[atom] * phi * sqrt_row_w;
            for out_col in 0..p {
                beta[beta_pos][out_col] = base * channel.output[out_col];
            }
            for (var_idx, var) in vars.iter().enumerate() {
                let scalar = match *var {
                    SaeLocalRowVar::Logit { .. } => dz[var_idx][atom] * phi * sqrt_row_w,
                    SaeLocalRowVar::Coord {
                        atom: coord_atom,
                        axis,
                    } if coord_atom == atom => {
                        assignments[atom]
                            * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                            * sqrt_row_w
                    }
                    _ => 0.0,
                };
                if scalar != 0.0 {
                    for out_col in 0..p {
                        beta_deriv[var_idx][beta_pos][out_col] = scalar * channel.output[out_col];
                    }
                }
                let scalar_l = match *var {
                    SaeLocalRowVar::Logit { .. } => {
                        dz[var_idx][atom]
                            * self.atoms[atom].basis_values[[row, channel.basis_col]]
                            * sqrt_row_w
                    }
                    SaeLocalRowVar::Coord {
                        atom: coord_atom,
                        axis,
                    } if coord_atom == atom => {
                        assignments[atom]
                            * self.atoms[atom].basis_jacobian[[row, channel.basis_col, axis]]
                            * sqrt_row_w
                    }
                    _ => 0.0,
                };
                if scalar_l != 0.0 {
                    for out_col in 0..p {
                        beta_l_deriv[var_idx][beta_pos][out_col] =
                            scalar_l * channel.output[out_col];
                    }
                }
            }
        }

        Ok(SaeRowJets {
            vars,
            first,
            second,
            beta,
            beta_deriv,
            beta_l_deriv,
        })
    }

    fn assignment_prior_hdiag_derivative_entry(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        diag_atom: usize,
        wrt: SaeLocalRowVar,
        ibp_channels: Option<&IbpHessianDiagThirdChannels>,
    ) -> f64 {
        let SaeLocalRowVar::Logit { atom: wrt_atom } = wrt else {
            return 0.0;
        };
        match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                let assignments = self.assignment.assignments_row(row);
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                let k_atoms = assignments.len();
                let mut l = vec![0.0_f64; k_atoms];
                let mut mean = 0.0_f64;
                for k in 0..k_atoms {
                    l[k] = assignments[k].max(1.0e-300).ln() + 1.0;
                    mean += assignments[k] * l[k];
                }
                let mut da = vec![0.0_f64; k_atoms];
                for k in 0..k_atoms {
                    let indicator = if k == wrt_atom { 1.0 } else { 0.0 };
                    da[k] = assignments[k] * (indicator - assignments[wrt_atom]) * inv_tau;
                }
                let dmean: f64 = (0..k_atoms).map(|k| da[k] * l[k]).sum();
                let k = diag_atom;
                let term = (1.0 - 2.0 * assignments[k]) * (mean - l[k]) + assignments[k] - 1.0;
                let dl_k = da[k] / assignments[k].max(1.0e-300);
                let dterm = -2.0 * da[k] * (mean - l[k])
                    + (1.0 - 2.0 * assignments[k]) * (dmean - dl_k)
                    + da[k];
                scale * (da[k] * term + assignments[k] * dterm)
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                let logit = self.assignment.logits[[row, diag_atom]];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    return 0.0;
                }
                let inv_tau = 1.0 / temperature;
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                2.0 * rho.lambda_sparse()
                    * slope
                    * slope
                    * (1.0 - 2.0 * activation)
                    * inv_tau
                    * inv_tau
                    * inv_tau
            }
            AssignmentMode::IBPMap { .. } => {
                // The assembled `htt` diagonal consumes
                // `IBPAssignmentPenalty::hessian_diag`, whose logit derivative
                // splits into a row-local direct-`z` channel and a global
                // empirical-`M_k` channel (π_k couples every row in column k).
                // This same-row primitive returns only the LOCAL direct-`z`
                // channel — and only on the matching logit (`diag_atom == w`),
                // since H_ik depends on no other row's z explicitly. The global
                // M_k channel is accumulated column-wise in
                // `logdet_theta_adjoint` (it needs the per-row selected-inverse
                // diagonals), so adding it here would double-count.
                if diag_atom != wrt_atom {
                    return 0.0;
                }
                match ibp_channels {
                    Some(ch) => ch.local_logit_third[row * ch.k_max + diag_atom],
                    None => 0.0,
                }
            }
        }
    }

    fn ard_majorized_hessian_derivative(
        &self,
        rho: &SaeManifoldRho,
        row: usize,
        atom: usize,
        axis: usize,
    ) -> f64 {
        if rho.log_ard[atom].is_empty() {
            return 0.0;
        }
        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
        let periods = self.assignment.coords[atom].effective_axis_periods();
        let t = self.assignment.coords[atom].row(row)[axis];
        let prior = ArdAxisPrior::eval(alpha, t, periods[axis]);
        if prior.hess <= 0.0 {
            return 0.0;
        }
        match periods[axis] {
            None => 0.0,
            Some(period) => {
                let kappa = std::f64::consts::TAU / period;
                -alpha * kappa * (kappa * t).sin()
            }
        }
    }

    pub fn outer_rho_gradient_ift_rhs(
        &self,
        rho: &SaeManifoldRho,
        j: usize,
        cache: &ArrowFactorCache,
    ) -> Result<SaeArrowVector, String> {
        let n_params = rho.to_flat().len();
        if j >= n_params {
            return Err(format!(
                "outer_rho_gradient_ift_rhs: coordinate {j} outside rho dim {n_params}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        if j == 0 {
            let (assignment_grad, _) = assignment_prior_grad_hdiag(&self.assignment, rho)?;
            let k_atoms = self.k_atoms();
            let assignment_dim = self.assignment.assignment_coord_dim();
            for row in 0..self.n_obs() {
                let base = cache.row_offsets[row];
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(ref layout) => {
                        for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                            t[base + pos] = assignment_grad[assignment_base + atom];
                        }
                    }
                    None => {
                        for free_idx in 0..assignment_dim {
                            t[base + free_idx] = assignment_grad[assignment_base + free_idx];
                        }
                    }
                }
            }
        } else if j == 1 {
            let lambda = rho.lambda_smooth();
            let frames_active = self.last_frames_active && cache.k == self.factored_border_dim();
            let offsets = if frames_active {
                self.factored_beta_offsets()
            } else {
                self.beta_offsets()
            };
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let m = atom.basis_size();
                let coeffs = if frames_active {
                    match &atom.decoder_frame {
                        Some(frame) => frame.project_decoder(atom.decoder_coefficients.view())?,
                        None => atom.decoder_coefficients.clone(),
                    }
                } else {
                    atom.decoder_coefficients.clone()
                };
                let r = coeffs.ncols();
                let off = offsets[atom_idx];
                for mu in 0..m {
                    for channel in 0..r {
                        let mut acc = 0.0_f64;
                        for nu in 0..m {
                            let s_sym = 0.5
                                * (atom.smooth_penalty[[mu, nu]] + atom.smooth_penalty[[nu, mu]]);
                            acc += s_sym * coeffs[[nu, channel]];
                        }
                        beta[off + mu * r + channel] = lambda * acc;
                    }
                }
            }
        } else {
            let mut cursor = 2usize;
            for atom in 0..rho.log_ard.len() {
                for axis in 0..rho.log_ard[atom].len() {
                    if cursor == j {
                        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
                        let periods = self.assignment.coords[atom].effective_axis_periods();
                        for row in 0..self.n_obs() {
                            let row_t = self.assignment.coords[atom].row(row);
                            let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                            let Some(pos) = sae_coord_penalty_offset(
                                self.last_row_layout.as_ref(),
                                self.assignment.coord_offsets()[atom] + axis,
                                row,
                                atom,
                            ) else {
                                continue;
                            };
                            t[cache.row_offsets[row] + pos] = prior.grad;
                        }
                        return Ok(SaeArrowVector { t, beta });
                    }
                    cursor += 1;
                }
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    pub(crate) fn logdet_theta_adjoint(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeArrowVector, String> {
        // Γ_a = tr(H⁻¹ ∂H/∂θ_a) over the inner variables θ (#1006). `H` here is
        // the SAME object the evidence factor builds — Gauss-Newton data
        // curvature plus the prior majorizers / `hessian_diag` diagonals the
        // Newton/Schur Cholesky factorizes — so each block's θ-derivative channel
        // is differentiated on the criterion's own branch (no value/gradient
        // desync). The IBP-MAP assignment prior is the one block whose
        // `hessian_diag` couples every row in a column through the plug-in
        // empirical mass `M_k = Σ_i z_ik`; its logit derivative therefore has a
        // row-local channel (handled inline via
        // `assignment_prior_hdiag_derivative_entry`) and a cross-row channel
        // (accumulated column-wise after the row loop, below).
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let mut beta_inv = Array2::<f64>::zeros((cache.k, cache.k));
        if cache.k > 0 {
            let rhs_t = Array1::<f64>::zeros(total_t);
            for col in 0..cache.k {
                let mut rhs_beta = Array1::<f64>::zeros(cache.k);
                rhs_beta[col] = 1.0;
                let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                    format!("logdet_theta_adjoint: beta selected inverse solve: {err}")
                })?;
                for row in 0..cache.k {
                    beta_inv[[row, col]] = solved.beta[row];
                }
            }
        }
        // Exact IBP `hessian_diag` logit third-derivative channels (#1006), built
        // once on the same penalty configuration the assembly uses. `None` for
        // non-IBP modes. The cross-row empirical-`M_k` channel needs the per-row
        // selected-inverse diagonals collected during the row loop, so it is
        // distributed column-wise afterwards.
        let ibp_channels = ibp_assignment_third_channels(&self.assignment, rho)?;
        let k_atoms = self.k_atoms();
        // Per active logit position: (row i, column k, global t-index,
        // (H⁻¹)_ik,ik) — the inputs to the IBP cross-row empirical-`M_k` channel.
        let mut ibp_logit_sites: Vec<(usize, usize, usize, f64)> = Vec::new();

        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let assignments = self.assignment.try_assignments_row(row)?;
            let jets =
                self.row_jets_for_logdet(row, vars, assignments.view(), &second_jets, &border)?;

            let mut inv_vv = Array2::<f64>::zeros((q, q));
            let mut inv_vbeta = Array2::<f64>::zeros((q, cache.k));
            for col in 0..q {
                let mut rhs_t = Array1::<f64>::zeros(total_t);
                let rhs_beta = Array1::<f64>::zeros(cache.k);
                rhs_t[base + col] = 1.0;
                let solved = solver.solve(rhs_t.view(), rhs_beta.view()).map_err(|err| {
                    format!("logdet_theta_adjoint: selected inverse solve: {err}")
                })?;
                for r in 0..q {
                    inv_vv[[r, col]] = solved.t[base + r];
                }
                for b in 0..cache.k {
                    inv_vbeta[[col, b]] = solved.beta[b];
                }
            }

            // Record each active logit's column, global t-index, and
            // selected-inverse diagonal (H⁻¹)_ik,ik for the IBP cross-row pass.
            if ibp_channels.is_some() {
                for (pos, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        ibp_logit_sites.push((row, atom, base + pos, inv_vv[[pos, pos]]));
                    }
                }
            }

            for w in 0..q {
                let mut gamma = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(&jets.second[a][w], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.second[b][w]);
                        if a == b {
                            dh += match jets.vars[a] {
                                SaeLocalRowVar::Logit { atom } => self
                                    .assignment_prior_hdiag_derivative_entry(
                                        rho,
                                        row,
                                        atom,
                                        jets.vars[w],
                                        ibp_channels.as_ref(),
                                    ),
                                SaeLocalRowVar::Coord { atom, axis } if a == w => {
                                    self.ard_majorized_hessian_derivative(rho, row, atom, axis)
                                }
                                _ => 0.0,
                            };
                        }
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.second[a][w], &jets.beta[beta_pos])
                            + sae_dot(&jets.first[a], &jets.beta_deriv[w][beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                for (beta_i, channel_i) in border.iter().enumerate() {
                    for (beta_j, channel_j) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_deriv[w][beta_i], &jets.beta[beta_j])
                            + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[w][beta_j]);
                        gamma += beta_inv[[channel_i.index, channel_j.index]] * dh;
                    }
                }
                gamma_t[base + w] = gamma;
            }

            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.first[b])
                            + sae_dot(&jets.first[a], &jets.beta_l_deriv[b][w_beta_pos]);
                        gamma += inv_vv[[b, a]] * dh;
                    }
                }
                for a in 0..q {
                    for (beta_pos, channel) in border.iter().enumerate() {
                        let dh = sae_dot(&jets.beta_l_deriv[a][w_beta_pos], &jets.beta[beta_pos]);
                        gamma += 2.0 * inv_vbeta[[a, channel.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }

        // IBP cross-row empirical-`M_k` channel of Γ (#1006). The assembled
        // diagonal H_ik consumes `hessian_diag`, whose dependence on the column
        // mass M_k = Σ_i z_ik couples every row in a column. Differentiating
        // tr(H⁻¹ ∂H/∂ℓ_wk) on that shared branch:
        //   Γ_wk += [ Σ_i (H⁻¹)_ik,ik · ∂_M H_ik ] · J_wk = C_k · J_wk,
        // where ∂_M H_ik = `m_channel[i*K+k]` and J_wk = `z_jac[w*K+k]`. The
        // row-local direct-`z` channel was already added inline above, so this
        // pass adds only the cross-row remainder (it spans `w ≠ i` and the
        // self-row M_k self-coupling, which the row-local primitive deliberately
        // omits to avoid double-counting).
        if let Some(channels) = ibp_channels.as_ref() {
            let mut col_coeff = vec![0.0_f64; k_atoms];
            for &(row, atom, _t_index, inv_diag) in &ibp_logit_sites {
                col_coeff[atom] += inv_diag * channels.m_channel[row * k_atoms + atom];
            }
            for &(row, atom, t_index, _inv_diag) in &ibp_logit_sites {
                gamma_t[t_index] += col_coeff[atom] * channels.z_jac[row * k_atoms + atom];
            }
        }

        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// Analytic SAE REML outer-ρ gradient components at the already converged
    /// inner state represented by `loss` and `cache`.
    ///
    /// The returned gradient is the assembled analytic outer derivative:
    /// explicit penalty terms, direct logdet traces, Occam terms, and the #1006
    /// implicit-state third-order correction.
    pub(crate) fn analytic_outer_rho_gradient_components(
        &self,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeOuterRhoGradientComponents, String> {
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);

        explicit[0] = assignment_prior_log_strength_derivative(&self.assignment, rho);
        logdet_trace[0] = self.assignment_log_strength_hessian_trace(rho, cache, solver)?;

        explicit[1] = loss.smoothness;
        logdet_trace[1] = 0.5
            * self
                .decoder_smoothness_effective_dof_with_solver(cache, solver, rho.lambda_smooth())
                .map_err(|err| format!("analytic_outer_rho_gradient_components: {err}"))?;
        occam[1] = -self.reml_occam_log_lambda_smooth_derivative()?;

        let ard_explicit = self.ard_log_precision_explicit_derivatives(rho)?;
        let ard_trace = self
            .ard_log_precision_hessian_trace(rho, cache, solver)
            .map_err(|err| format!("analytic_outer_rho_gradient_components: {err}"))?;
        let mut cursor = 2usize;
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                explicit[cursor] = ard_explicit[k][axis];
                logdet_trace[cursor] = ard_trace[k][axis];
                cursor += 1;
            }
        }

        let gamma = self.logdet_theta_adjoint(rho, cache, solver)?;
        for coord in 0..n_params {
            let rhs = self.outer_rho_gradient_ift_rhs(rho, coord, cache)?;
            let solved = solver.solve(rhs.t.view(), rhs.beta.view()).map_err(|err| {
                format!("analytic_outer_rho_gradient_components: full_inverse_apply: {err}")
            })?;
            let mut dot = 0.0_f64;
            for idx in 0..gamma.t.len() {
                dot += gamma.t[idx] * solved.t[idx];
            }
            for idx in 0..gamma.beta.len() {
                dot += gamma.beta[idx] * solved.beta[idx];
            }
            third_order_correction[coord] = -0.5 * dot;
        }

        Ok(SaeOuterRhoGradientComponents {
            explicit,
            logdet_trace,
            occam,
            third_order_correction,
            third_order_correction_available: true,
        })
    }

    /// Public analytic outer-ρ gradient at a converged inner state, constructing
    /// the deflated arrow solver from the supplied cache. Use this seam from
    /// integration tests and external consumers that have a converged
    /// `(loss, cache)` from [`Self::reml_criterion_with_cache`] but no access to
    /// the crate-private `DeflatedArrowSolver`.
    pub fn analytic_outer_rho_gradient_at_converged(
        &self,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<SaeOuterRhoGradientComponents, String> {
        let solver = self.outer_gradient_arrow_solver(cache)?;
        self.analytic_outer_rho_gradient_components(rho, loss, cache, &solver)
    }

    /// Compose the SAE LAML criterion as a sum of atoms (#931 SAE pilot).
    ///
    /// This is the single seam that establishes value↔gradient coherence for
    /// the SAE objective: it runs the inner solve once via
    /// [`Self::reml_criterion_with_cache`], reads the value decomposition
    /// (`loss.total() + extra_penalty_energy`, `log|H|`, `occam`) and the
    /// matching gradient channels (`SaeOuterRhoGradientComponents`) from the
    /// SAME converged cache, and hands them to [`SaeCriterion::assemble`]. The
    /// returned criterion's [`SaeCriterion::value`] and
    /// [`SaeCriterion::gradient`] are then projections of one factorization —
    /// the outer optimizer can no longer evaluate a value path and a gradient
    /// path that disagree (the #752/#748/#901 desync class). The
    /// implicit-stationarity envelope correction (#1006's Γ term) is its own
    /// named atom, so the channel the desync class keeps dropping is visible
    /// rather than a silent zero.
    pub fn criterion_as_atoms(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeCriterion, String> {
        let (_v, loss, cache) = self.reml_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "criterion_as_atoms: arrow_log_det_from_cache returned None".to_string()
        })?;
        let occam = self.reml_occam_term(rho)?;
        let extra_penalty_energy = match registry {
            Some(reg) => self
                .reml_extra_penalty_value_total(reg)
                .map_err(|err| format!("SaeManifoldTerm::criterion_as_atoms: {err}"))?,
            None => 0.0,
        };
        let data_fit_priors_value = loss.total() + extra_penalty_energy;

        let solver = self.outer_gradient_arrow_solver(&cache)?;
        let components =
            self.analytic_outer_rho_gradient_components(rho, &loss, &cache, &solver)?;
        Ok(SaeCriterion::assemble(
            data_fit_priors_value,
            log_det,
            occam,
            components.explicit,
            components.logdet_trace,
            components.occam,
            components.third_order_correction,
        ))
    }

    /// Gaussian reconstruction dispersion `φ̂`, the scale that turns the
    /// unscaled inverse-Hessian β-block `S_β⁻¹` into a posterior covariance
    /// `Cov(β) = φ̂·S_β⁻¹` — the same `Vb = φ·H⁻¹` convention the main GAM
    /// inference path uses.
    ///
    /// `RSS = Σ_{i,c} (z_{ic} − ẑ_{ic})² = 2·data_fit` (the loss stores the
    /// half-sum `½Σr²`). The residual degrees of freedom subtract the effective
    /// parameter count from the `N·p` scalar observations:
    ///   * decoder β: `beta_dim − tr(λ_smooth · S_β⁻¹ · ⊕_k S_k⊗I_p)`, the
    ///     smoothness effective-dof already assembled for the Fellner-Schall
    ///     step (penalty-shrunk directions do not cost a full parameter);
    ///   * latent coordinates: enabled ARD axes use the exact ARD-shrunk trace
    ///     `Σ_k Σ_j (n_active_k − α_{kj}·tr_{kj}(H⁻¹))`; atoms with disabled
    ///     native ARD charge the full active coordinate count because those
    ///     latent variables are estimated without an ARD precision.
    ///
    /// The coordinate term is the **exact** ARD-shrunk effective dof of the
    /// latent block: along axis `(k,j)` the MacKay/Fellner-Schall edf is
    /// `n_active_k − α_{kj}·tr_{kj}(H⁻¹)`, the well-determined-direction count
    /// after the ARD prior `α_{kj}` shrinks each coordinate. `tr_{kj}(H⁻¹)` is
    /// the same posterior-variance trace [`Self::ard_inverse_traces`] assembles
    /// for the EFS ARD step (reused here, not recomputed), so the dispersion is
    /// consistent with the precision update `α_new = n/(‖t‖²+tr(H⁻¹))`. The
    /// per-axis scalar count `n_active_k` must match the support the trace sums
    /// over: `n` for the dense full-support layout, or the number of rows where
    /// atom `k` is active for the compact active-set layout (inactive
    /// prior-dominated coordinates contribute 0 to both the trace and the
    /// count, hence 0 edf). The residual dof is floored at 1 so `φ̂` stays
    /// finite and positive.
    fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let n_scalar = (n * p) as f64;
        let rss = 2.0 * loss.data_fit;
        let smooth_edf = self
            .decoder_smoothness_effective_dof(cache, rho.lambda_smooth())
            .map_err(|e| format!("reconstruction_dispersion: smooth edf: {e}"))?;
        // #972 / #977 T1: the raw decoder-parameter count is `beta_dim` on the
        // full-`B` path, but when frames are active the estimated decoder freedom
        // is the factored border `Σ M_k·r_k` PLUS the `Σ r_k·(p−r_k)` Grassmann
        // frame degrees profiled out (both are genuinely estimated), which the
        // smoothness shrinkage `smooth_edf` (taken over the factored border) then
        // discounts. On the full-`B` path `factored_border_dim == beta_dim` and
        // `grassmann_evidence_dimension == 0`, so this is exactly `beta_dim`.
        let raw_decoder_dof = if self.frames_active() {
            (self.factored_border_dim() + self.grassmann_evidence_dimension()) as f64
        } else {
            self.beta_dim() as f64
        };
        let beta_edf = (raw_decoder_dof - smooth_edf).max(0.0);
        // Exact ARD-shrunk latent-coordinate edf, reusing the EFS trace cache.
        let traces = self
            .ard_inverse_traces(cache)
            .map_err(|e| format!("reconstruction_dispersion: ARD traces: {e}"))?;
        if rho.log_ard.len() != self.atoms.len() {
            return Err(format!(
                "reconstruction_dispersion: ρ has {} ARD atoms but term has {}",
                rho.log_ard.len(),
                self.atoms.len()
            ));
        }
        let mut coord_edf = 0.0_f64;
        for (k, atom) in self.atoms.iter().enumerate() {
            let d_k = atom.latent_dim;
            if traces[k].len() != d_k {
                return Err(format!(
                    "reconstruction_dispersion: trace shape mismatch at atom {k} \
                     (traces={}, d_k={d_k})",
                    traces[k].len()
                ));
            }
            let ard_len = rho.log_ard[k].len();
            if ard_len != 0 && ard_len != d_k {
                return Err(format!(
                    "reconstruction_dispersion: ARD shape mismatch at atom {k} \
                     (log_ard={ard_len}, d_k={d_k})"
                ));
            }
            // Scalar count matched to the trace support (see fn doc).
            let n_active_k = match self.last_row_layout {
                Some(ref layout) => layout
                    .active_atoms
                    .iter()
                    .filter(|active| active.contains(&k))
                    .count() as f64,
                None => n as f64,
            };
            if ard_len == 0 {
                coord_edf += n_active_k * d_k as f64;
                continue;
            }
            for j in 0..d_k {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][j]);
                // edf_kj ∈ [0, n_active_k]; clamp against numerical drift.
                let edf_kj = (n_active_k - alpha * traces[k][j]).clamp(0.0, n_active_k);
                coord_edf += edf_kj;
            }
        }
        let resid_dof = (n_scalar - beta_edf - coord_edf).max(1.0);
        let phi = rss / resid_dof;
        if !phi.is_finite() || phi < 0.0 {
            return Err(format!(
                "reconstruction_dispersion: non-finite/negative φ̂={phi} \
                 (RSS={rss}, resid_dof={resid_dof}, beta_edf={beta_edf}, coord_edf={coord_edf})"
            ));
        }
        Ok(phi.max(f64::MIN_POSITIVE))
    }

    /// Posterior covariance and ambient shape band for every atom — the
    /// user-facing uncertainty of the fitted manifold shapes.
    ///
    /// For atom `k` with decoder-block range `r_k` (see
    /// [`Self::beta_block_offsets`]), `Cov(β_k) = φ·S_β⁻¹[r_k, r_k]` is the
    /// φ-scaled posterior covariance of its decoder coefficients with the
    /// latent coordinates marginalized out. The ambient point at a coordinate
    /// `t` is `m_k(t) = Φ_k(t)·B_k`, *linear* in `β_k`, so its per-channel
    /// posterior variance is the closed form
    /// `Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]`
    /// — no sampling. The band is evaluated at up to [`SHAPE_BAND_MAX_POINTS`]
    /// evenly-strided of the atom's own on-atom coordinates, reusing the basis
    /// values already stored on the atom, so it reports uncertainty exactly
    /// where the data lives and needs no basis-kind-specific grid.
    ///
    /// A near-degenerate atom has a near-singular Schur block, so `Cov(β_k)` —
    /// and the band — fans out automatically: the band width is a
    /// per-coordinate visual of how well each atom is identified.
    pub fn assemble_shape_uncertainty(
        &self,
        cache: &ArrowFactorCache,
        dispersion: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache β block is the FACTORED border when frames
        // are active, so each atom's Schur inverse block is the `(M_k·r_k)`
        // coordinate covariance `Cov(vec C_k)`. We LIFT it to the full
        // `(M_k·p)` decoder covariance `Cov(vec B_k) = (I_{M_k} ⊗ U_k) Cov(vec
        // C_k)(I_{M_k} ⊗ U_k)ᵀ` (since `B_k = C_k U_kᵀ`) so the downstream band
        // code — which reads the `b·p + c` flat layout — is unchanged. On the
        // full-`B` path the block is already `(M_k·p)` and the lift is skipped.
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if frames_active {
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect::<Vec<_>>()
        } else {
            self.beta_block_offsets().to_vec()
        };
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let cov_block = cache
                .schur_inverse_block(block_ranges[k].clone())
                .map_err(|e| format!("assemble_shape_uncertainty: atom {k}: {e}"))?;
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            // Evenly-strided evaluation rows bound the band cost.
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let mut band_sd = Array2::<f64>::zeros((g, p));
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }

            let framed = frames_active && atom.decoder_frame.is_some();
            let dense_entries = (m * p).saturating_mul(m * p);
            let cov = if framed && dense_entries > SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES {
                // LLM-scale ambient `p`: the dense `(M_k·p)²` lift would be
                // gigabytes per atom and exists only to export the full
                // covariance. Compute the band variance EXACTLY from the
                // factored frame covariance instead: with `B_k = C_k·U_kᵀ`,
                //   Var_c(t) = (φ ⊗ u_c)ᵀ Cov(vec C_k) (φ ⊗ u_c)
                // which is the r×r quadratic form `u_cᵀ Y u_c` with
                //   Y = Σ_{b1,b2} φ[b1] φ[b2] Cov(C)[(b1,·),(b2,·)].
                let mut cov_c = cov_block;
                cov_c.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let basis = atom.basis_values.row(row);
                    for c in 0..p {
                        let var = frame_projection.output_variance(k, cov_c.view(), basis, c);
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                None
            } else {
                // Lift the factored `(M_k·r_k)` coordinate covariance to the
                // full `(M_k·p)` decoder covariance through this atom's frame;
                // identity (a plain scaled copy) on the un-framed full-`B` path.
                let mut cov = if framed {
                    frame_projection.lift_block(k, cov_block.view())
                } else {
                    cov_block
                };
                cov.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    // Var_c = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]; the flat
                    // decoder index is basis·p + channel (row-major (M_k, p)).
                    for c in 0..p {
                        let var = frame_projection.full_output_variance(
                            k,
                            cov.view(),
                            atom.basis_values.row(row),
                            c,
                        );
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                Some(cov)
            };
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords,
                band_mean,
                band_sd,
            });
        }
        Ok(SaeShapeUncertainty { dispersion, atoms })
    }

    fn shape_uncertainty_without_decoder_covariance(&self, dispersion: f64) -> SaeShapeUncertainty {
        let p = self.output_dim();
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords,
                band_mean,
                band_sd,
            });
        }
        SaeShapeUncertainty { dispersion, atoms }
    }

    /// Returns whether Beta-tier analytic curvature was accumulated into the
    /// dense `sys.hbb` block or deferred for exact factored-space probing.
    fn add_sae_analytic_penalty_contributions(
        &self,
        sys: &mut ArrowSchurSystem,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        row_layout: Option<&SaeRowLayout>,
        dense_beta_curvature: bool,
        factored_row_projection: Option<&FrameProjection>,
    ) -> Result<SaeBetaPenaltyAssembly, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut beta_assembly = SaeBetaPenaltyAssembly::default();
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            // The coordinate ARD prior is owned by the built-in `ArdAxisPrior`
            // path (the unconditional row-block gradient/curvature write above,
            // and `ard_value`/`loss.ard` for the energy). That path uses the
            // smooth von-Mises energy `V(t) = (α/κ²)(1−cos κt)` on periodic
            // (Circle) axes, whose value, gradient (`α/κ·sin κt`), and curvature
            // (`α·cos κt`) are mutually FD-consistent and continuous across the
            // branch cut. The registry `ARDPenalty` is the legacy Euclidean
            // Gaussian (`½λt²`, grad `λt`, curvature `λ`): adding it here would
            // (a) double-count the coordinate prior in both gradient and Newton
            // curvature, and (b) reintroduce the period-discontinuous `½λt²`
            // energy — its grad `λt` is continuous but its value jumps by
            // `½λ(t_after²−t_before²)` across the cut, so a near-zero Newton step
            // crossing the cut changes the line-search objective discontinuously
            // and Armijo rejects it. Skip it on every SAE path so the von-Mises
            // built-in is the single source of truth (matching the REML criterion,
            // which already scores only `loss.ard`).
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            match tier {
                PenaltyTier::Psi => {
                    if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) {
                        // NuclearNorm is a Psi-tier penalty but it targets each
                        // atom's decoder (β) matrix singular spectrum, not the
                        // coord "t" row block. Route it to the β tier so it
                        // shrinks each atom's embedding rank.
                        if self.add_sae_beta_penalty(
                            sys,
                            penalty,
                            beta.view(),
                            rho_local,
                            penalty_scale,
                            dense_beta_curvature,
                        ) {
                            beta_assembly.record_curvature(dense_beta_curvature);
                        }
                    } else {
                        // Every other Psi-tier penalty here is row-block
                        // supported with a coord-shape that matches each
                        // atom — `validate_analytic_penalty_registry`
                        // refused everything else upfront, so this branch
                        // is total and the K=1 vs K>=2 path is the same
                        // loop. Row-block coord penalties (ARD,
                        // BlockOrthogonality, Sparsity/TopK/JumpReLU,
                        // RowPrecisionPrior, ScadMcp, Isometry) target the
                        // "t" latent block (n_obs × d) and apply per atom
                        // — accumulate into the corresponding row offsets.
                        assert!(
                            sae_penalty_is_row_block_supported(penalty),
                            "validate_analytic_penalty_registry should have \
                             refused non-row-block Psi-tier penalty {:?} \
                             (registry layout name {name:?})",
                            penalty.name()
                        );
                        let offsets = self.assignment.coord_offsets();
                        for atom_idx in 0..self.k_atoms() {
                            let off = offsets[atom_idx];
                            let coord = &self.assignment.coords[atom_idx];
                            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                                let corrected_kind =
                                    self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                                self.add_sae_coord_penalty(
                                    sys,
                                    atom_idx,
                                    off,
                                    coord,
                                    &corrected_kind,
                                    rho_local,
                                    row_layout,
                                    factored_row_projection,
                                );
                                // The isometry penalty value depends on the
                                // decoder B as well as the latent coords, through
                                // the pullback metric `g = JᵀWJ` with the model
                                // Jacobian `J = (∂Φ/∂t)·B`. `add_sae_coord_penalty`
                                // only routes `∂P/∂t` into `gt`; the matching
                                // `∂P/∂B` must be accumulated into `gb`, or the
                                // assembled gradient disagrees with the penalized
                                // objective on the β block (value path counts the
                                // isometry energy, which moves with B).
                                if let AnalyticPenaltyKind::Isometry(corrected) = &corrected_kind {
                                    self.add_sae_isometry_beta_penalty(
                                        sys,
                                        atom_idx,
                                        coord,
                                        corrected,
                                        rho_local,
                                        dense_beta_curvature,
                                    );
                                    beta_assembly.record_curvature(dense_beta_curvature);
                                }
                            } else {
                                self.add_sae_coord_penalty(
                                    sys,
                                    atom_idx,
                                    off,
                                    coord,
                                    penalty,
                                    rho_local,
                                    row_layout,
                                    factored_row_projection,
                                );
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    // β-tier analytic penalties are global (B-only); minibatch-
                    // scaled so per-chunk sums reconstruct one global copy.
                    if self.add_sae_beta_penalty(
                        sys,
                        penalty,
                        beta.view(),
                        rho_local,
                        penalty_scale,
                        dense_beta_curvature,
                    ) {
                        beta_assembly.record_curvature(dense_beta_curvature);
                    }
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(beta_assembly)
    }

    fn corrected_isometry_penalty(
        &self,
        iso: &Arc<IsometryPenalty>,
        atom_idx: usize,
        coord: &LatentCoordValues,
    ) -> Result<AnalyticPenaltyKind, ArrowSchurError> {
        // Isometry requires per-step cache refresh from the atom's second jet
        // before value / grad_target / hvp are live. The registry-held
        // IsometryPenalty was constructed with p_out equal to the latent dim
        // from the JSON latent spec; clone it and correct p_out to the atom's
        // true decoder output dimension before refreshing caches.
        let atom = &self.atoms[atom_idx];
        let p = atom.decoder_coefficients.ncols();
        let mut corrected: IsometryPenalty = (**iso).clone();
        corrected.p_out = p;
        // Single-source-of-truth gauge metric: the isometry pullback weight is
        // taken from the SAME RowMetric the reconstruction likelihood whitens
        // through. There is no independent gauge-weight setter, so a
        // likelihood-metric ≠ gauge-metric state is unrepresentable. When the
        // term carries no RowMetric (Euclidean default) the gauge weight stays
        // Identity, matching the isotropic likelihood exactly. The metric's
        // p_out must agree with the atom's true decoder output dimension.
        if let Some(metric) = self.row_metric.as_ref() {
            // Only a metric that actually drives the gauge installs a non-identity
            // pullback weight: any non-Euclidean provenance (OutputFisher or the
            // #974 WhitenedStructured) pulls the isometry penalty back through its
            // per-row inner product. A Euclidean metric reduces the gauge to the
            // bare `J_nᵀ J_n` (Identity weight), so it is left untouched and the
            // gauge is bit-for-bit the historical isotropic pullback.
            if metric.drives_gauge() {
                if metric.p_out() == p {
                    corrected.weight = metric.to_weight_field();
                } else {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "corrected_isometry_penalty: RowMetric p_out {} disagrees with atom \
                             {} decoder output dim {p}; the gauge metric must match the likelihood \
                             metric",
                            metric.p_out(),
                            atom_idx
                        ),
                    });
                }
            }
        }
        let coords_mat = coord.as_matrix();
        let second_jet_installed =
            refresh_isometry_caches_from_atom(&corrected, atom, coords_mat.view())
                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        if !second_jet_installed {
            match atom
                .basis_evaluator
                .as_ref()
                .and_then(|e| e.second_jet_dyn(coords_mat.view()))
            {
                Some(Ok(hess)) => {
                    let n_obs = coords_mat.nrows();
                    let d = atom.latent_dim;
                    let m = atom.basis_size();
                    if hess.dim() != (n_obs, m, d, d) {
                        return Err(ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "SAE Isometry atom '{}': second_jet_dyn returned shape {:?}, \
                                 expected ({n_obs}, {m}, {d}, {d})",
                                atom.name,
                                hess.dim()
                            ),
                        });
                    }
                    let b = &atom.decoder_coefficients;
                    let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
                    for n in 0..n_obs {
                        for i in 0..p {
                            for a in 0..d {
                                for c in 0..d {
                                    let mut acc = 0.0;
                                    for mm in 0..m {
                                        acc += hess[[n, mm, a, c]] * b[[mm, i]];
                                    }
                                    jac2[[n, (i * d + a) * d + c]] = acc;
                                }
                            }
                        }
                    }
                    corrected.set_jacobian_second_cache(Some(Arc::new(jac2)));
                }
                Some(Err(reason)) => {
                    return Err(ArrowSchurError::SchurFactorFailed { reason });
                }
                None => {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "IsometryPenalty requested for SAE atom '{}' (basis kind {:?}) but \
                             this evaluator does not expose an analytic second jet; use \
                             AffineCoordinateEvaluator, SphereChartEvaluator, \
                             PeriodicHarmonicEvaluator, or TorusHarmonicEvaluator for \
                             SAE-Isometry",
                            atom.name, atom.basis_kind
                        ),
                    });
                }
            }
        }
        Ok(AnalyticPenaltyKind::Isometry(Arc::new(corrected)))
    }

    fn add_sae_logit_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty: &AnalyticPenaltyKind,
        target: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        row_layout: Option<&SaeRowLayout>,
    ) {
        let n = self.n_obs();
        let k = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let grad = penalty.grad_target(target, rho_local);
        for row in 0..n {
            if let Some(layout) = row_layout {
                for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                    sys.rows[row].gt[pos] += grad[row * k + atom];
                }
            } else {
                for free_idx in 0..assignment_dim {
                    sys.rows[row].gt[free_idx] += grad[row * k + free_idx];
                }
            }
        }
        // The ArrowSchur `htt` block is the Newton / PIRLS curvature operator and
        // must stay PSD. Nonconvex sparsifiers (log, JumpReLU) have an indefinite
        // true Hessian, so we accumulate the PSD majorizer here — never the exact
        // `hessian_diag`, which goes negative and would destroy the solve's
        // positive-definiteness. Convex penalties' majorizer equals their exact
        // Hessian, so this is exact for them.
        if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
            for row in 0..n {
                if let Some(layout) = row_layout {
                    for (pos, &atom) in layout.active_atoms[row].iter().enumerate() {
                        sys.rows[row].htt[[pos, pos]] += diag[row * k + atom];
                    }
                } else {
                    for free_idx in 0..assignment_dim {
                        sys.rows[row].htt[[free_idx, free_idx]] += diag[row * k + free_idx];
                    }
                }
            }
        }
    }

    fn add_sae_coord_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        dense_off: usize,
        coord: &LatentCoordValues,
        penalty: &AnalyticPenaltyKind,
        rho_local: ArrayView1<'_, f64>,
        row_layout: Option<&SaeRowLayout>,
        factored_row_projection: Option<&FrameProjection>,
    ) {
        let n = coord.n_obs();
        let d = coord.latent_dim();
        // Origin-anchored magnitude shrinkage (SCAD/MCP) is restricted to the
        // Euclidean axes: a periodic chart axis has no origin, and folding a
        // raw-|t| energy there is period-discontinuous and breaks the joint
        // Newton solve (issue #795). Evaluate the axis-separable penalty on the
        // Euclidean-only compacted coordinate and scatter its gradient / PSD
        // curvature back to those axis slots — periodic axes get nothing. This
        // mirrors the value accounting in `analytic_penalty_value_total` exactly,
        // so the assembled gradient stays the gradient of the line-search
        // objective.
        if sae_coord_penalty_is_origin_anchored_magnitude(penalty) {
            if let Some((euclidean_axes, compacted)) =
                sae_coord_penalty_euclidean_restriction(coord)
            {
                let de = euclidean_axes.len();
                let grad = penalty.grad_target(compacted.view(), rho_local);
                let diag = penalty.psd_majorizer_diag(compacted.view(), rho_local);
                for row in 0..n {
                    if let Some(row_off) =
                        sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                    {
                        for (j, &axis) in euclidean_axes.iter().enumerate() {
                            sys.rows[row].gt[row_off + axis] += grad[row * de + j];
                            if let Some(diag) = diag.as_ref() {
                                sys.rows[row].htt[[row_off + axis, row_off + axis]] +=
                                    diag[row * de + j];
                            }
                        }
                    }
                }
                return;
            }
        }
        let target = coord.as_flat().view();
        let grad = penalty.grad_target(target, rho_local);
        for row in 0..n {
            if let Some(row_off) = sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx) {
                for axis in 0..d {
                    sys.rows[row].gt[row_off + axis] += grad[row * d + axis];
                }
            }
        }
        if let AnalyticPenaltyKind::Isometry(corrected) = penalty {
            self.add_sae_isometry_metric_gn_blocks(
                sys,
                atom_idx,
                dense_off,
                coord,
                corrected,
                rho_local,
                row_layout,
                factored_row_projection,
            );
            return;
        }
        // `htt` is the PSD Newton / PIRLS curvature block: accumulate the PSD
        // majorizer (exact for convex penalties), not the indefinite exact
        // Hessian, for the same reason as `add_sae_logit_penalty` above.
        if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
            for row in 0..n {
                if let Some(row_off) =
                    sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                {
                    for axis in 0..d {
                        sys.rows[row].htt[[row_off + axis, row_off + axis]] += diag[row * d + axis];
                    }
                }
            }
            return;
        }
        let mut probe = Array1::<f64>::zeros(n * d);
        for axis in 0..d {
            probe.fill(0.0);
            for row in 0..n {
                probe[row * d + axis] = 1.0;
            }
            let hv = penalty.psd_majorizer_hvp(target, rho_local, probe.view());
            for row in 0..n {
                if let Some(row_off) =
                    sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                {
                    for b in 0..d {
                        sys.rows[row].htt[[row_off + b, row_off + axis]] += hv[row * d + b];
                    }
                }
            }
        }
    }

    fn add_sae_isometry_metric_gn_blocks(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        dense_off: usize,
        coord: &LatentCoordValues,
        corrected: &Arc<IsometryPenalty>,
        rho_local: ArrayView1<'_, f64>,
        row_layout: Option<&SaeRowLayout>,
        factored_row_projection: Option<&FrameProjection>,
    ) {
        let n_obs = coord.n_obs();
        let d = coord.latent_dim();
        let atom = &self.atoms[atom_idx];
        let p = atom.decoder_coefficients.ncols();
        let m = atom.basis_size();
        let Some(jac) = corrected.jacobian_cache() else {
            return;
        };
        if jac.dim() != (n_obs, p * d) {
            return;
        }
        let Some(jac2) = corrected.jacobian_second_cache() else {
            return;
        };
        if jac2.dim() != (n_obs, p * d * d) {
            return;
        }
        let beta_off = self.beta_offsets()[atom_idx];
        let beta_block = m * p;
        let jet = &atom.basis_jacobian;
        // Resolve the learnable isometry strength `scalar_weight · exp(rho)` in
        // log-space with a clamped exponent: the naive `scalar_weight *
        // rho.exp()` overflows to `inf` for `rho ≳ 709`, and the downstream
        // `inf · jacobian` / `inf · 0.0` then injects NaN into the GN curvature
        // block and β-penalty, poisoning the joint solve (#742, Issue 4).
        let mu = resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
        // A negligible (or non-finite) effective isometry weight contributes a
        // zero curvature block; writing zeros would still flip the solver onto
        // the dense-supplement Schur path (and invalidate caches) for no model
        // change. Skip entirely so `isometry_weight≈0` is bit-identical to the
        // no-isometry assembly. (`isometry_weight=0` never constructs the
        // penalty at all; this guards the ρ-sweep driving `exp(ρ)` to ~0.)
        if !(mu.is_finite() && mu > 0.0) {
            return;
        }
        // Coherence invariant for the coupled Gauss-Newton block. The isometry
        // residual `r_{ab} = (JᵀWJ − G_ref)_{ab}` yields one residual Jacobian
        // `A = [A_t | A_β]`, so `[[htt,cross],[crossᵀ,hbb]] = μ AᵀA` is PSD *as a
        // whole* and its Schur complement is PSD — but ONLY while all three
        // blocks stay that exact pullback. After assembly the latent blocks pass
        // through `apply_riemannian_latent_geometry`, which on a *curved* chart
        // rewrites `htt` with the (indefinite) Riemannian connection term and
        // column-projects the `htbeta` cross-block to `T_tM`, while the shared
        // `hbb` is left untouched. That projection breaks the `μ AᵀA` coherence:
        // the cross-block is then a nonzero coupling NOT paired with diagonals
        // from the same Jacobian, and the Schur complement
        // `hbb − Σ crossᵀ htt⁻¹ cross` can go indefinite (the #681 sphere
        // failure mode flagged in the math review).
        //
        // The decision must therefore key on whether the geometry transform is
        // the IDENTITY for this chart, NOT on `is_euclidean()`. A flat periodic
        // chart (`Circle`/`Torus`) is non-Euclidean yet transforms as the exact
        // identity — its tangent projection is the identity, it carries no
        // connection term, and it adds no normal pinning — so the coupled block
        // survives exactly and the full cross-coupling must be kept. Keying on
        // `is_euclidean()` instead wrongly dropped the cross-block for the
        // single-circle fit, leaving a block-diagonal Hessian that misses the
        // strong isometry `t`↔`B` coupling; the joint Newton step then never
        // reaches the KKT stationarity the REML criterion now requires, and the
        // arrow-Schur proximal ridge saturates at 1e15 (issue #795, a regression
        // of #681). For a genuinely curved chart (Sphere, an active Interval
        // boundary) we contribute only the PSD `htt` diagonal block and DROP the
        // cross-block: a block-diagonal `diag(μ A_tᵀA_t, μ A_βᵀA_β)` of two PSD
        // blocks is still PSD, so the Schur stays PSD by construction while the
        // gradient (which alone fixes the stationary point) is unchanged.
        let couple_cross_block = coord.manifold().preserves_isometry_cross_block_coherence();
        let mut metric_coord_jac = Array2::<f64>::zeros((d * d, d));
        let mut metric_beta_jac = Array2::<f64>::zeros((d * d, beta_block));
        let mut wrote_dense_cross = false;
        for row in 0..n_obs {
            let Some(row_off) = sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
            else {
                continue;
            };
            let Some(wj) = Self::sae_isometry_weighted_jacobian_row(corrected, &jac, row, p, d)
            else {
                return;
            };
            metric_coord_jac.fill(0.0);
            for a in 0..d {
                for b in 0..d {
                    let metric_row = a * d + b;
                    for c in 0..d {
                        let mut acc = 0.0;
                        for i in 0..p {
                            acc += jac2[[row, (i * d + a) * d + c]] * wj[[i, b]];
                            acc += wj[[i, a]] * jac2[[row, (i * d + b) * d + c]];
                        }
                        metric_coord_jac[[metric_row, c]] = acc;
                    }
                }
            }
            if couple_cross_block {
                metric_beta_jac.fill(0.0);
                for a in 0..d {
                    for b in 0..d {
                        let metric_row = a * d + b;
                        for basis_col in 0..m {
                            let jet_a = jet[[row, basis_col, a]];
                            let jet_b = jet[[row, basis_col, b]];
                            for output in 0..p {
                                metric_beta_jac[[metric_row, basis_col * p + output]] =
                                    jet_a * wj[[output, b]] + wj[[output, a]] * jet_b;
                            }
                        }
                    }
                }
            }
            for c in 0..d {
                for e in 0..d {
                    let mut acc = 0.0;
                    for metric_row in 0..(d * d) {
                        acc +=
                            metric_coord_jac[[metric_row, c]] * metric_coord_jac[[metric_row, e]];
                    }
                    sys.rows[row].htt[[row_off + c, row_off + e]] += mu * acc;
                }
                if !couple_cross_block {
                    continue;
                }
                for beta_col in 0..beta_block {
                    let mut acc = 0.0;
                    for metric_row in 0..(d * d) {
                        acc += metric_coord_jac[[metric_row, c]]
                            * metric_beta_jac[[metric_row, beta_col]];
                    }
                    if let Some(projection) = factored_row_projection {
                        let basis_col = beta_col / p;
                        let output = beta_col % p;
                        let c_base = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx];
                        let mut hrow = sys.rows[row].htbeta.row_mut(row_off + c);
                        let hrow_slice = hrow.as_slice_mut().expect("htbeta row is contiguous");
                        projection.accumulate_output_project(
                            atom_idx,
                            c_base,
                            output,
                            mu * acc,
                            hrow_slice,
                        );
                    } else {
                        sys.rows[row].htbeta[[row_off + c, beta_off + beta_col]] += mu * acc;
                    }
                    wrote_dense_cross = true;
                }
            }
        }
        if wrote_dense_cross {
            sys.activate_dense_htbeta_supplement();
        }
    }

    fn sae_isometry_weighted_jacobian_row(
        corrected: &IsometryPenalty,
        jac: &Array2<f64>,
        row: usize,
        p: usize,
        d: usize,
    ) -> Option<Array2<f64>> {
        match &corrected.weight {
            WeightField::Identity => {
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        out[[i, a]] = jac[[row, i * d + a]];
                    }
                }
                Some(out)
            }
            WeightField::Factored { u, rank, p_out } => {
                if *p_out != p || u.nrows() != jac.nrows() || u.ncols() != p * *rank {
                    return None;
                }
                let mut projected = Array2::<f64>::zeros((*rank, d));
                for weight_axis in 0..*rank {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for i in 0..p {
                            acc += u[[row, i * *rank + weight_axis]] * jac[[row, i * d + a]];
                        }
                        projected[[weight_axis, a]] = acc;
                    }
                }
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for weight_axis in 0..*rank {
                            acc += u[[row, i * *rank + weight_axis]] * projected[[weight_axis, a]];
                        }
                        out[[i, a]] = acc;
                    }
                }
                Some(out)
            }
        }
    }

    /// Accumulate the isometry penalty's decoder-block gradient `∂P/∂B` into the
    /// β-tier `gb` and its decoder-block Gauss-Newton majorizer into `hbb`. The
    /// isometry value
    ///   `P = ½ μ Σ_n ‖J_nᵀ W J_n − G_ref‖²_F`
    /// is a function of the model Jacobian `J_n[i, a] = Σ_m (∂Φ/∂t)[n, m, a]·B[m, i]`,
    /// so it depends on the decoder `B` as well as the latent coords `t`. The
    /// penalty exposes `∂P/∂J` (shape `(n_obs, p·d)`, layout `[n, i·d + a]`) via
    /// [`IsometryPenalty::grad_jacobian`]; the chain rule through
    /// `∂J[n, i·d + a]/∂B[m, i] = (∂Φ/∂t)[n, m, a]` gives
    ///   `∂P/∂B[m, i] = Σ_n Σ_a (∂P/∂J)[n, i·d + a] · (∂Φ/∂t)[n, m, a]`.
    /// Since `J` is linear in `B`, the PSD decoder curvature is the exact
    /// pullback of the J-space Gauss-Newton block:
    ///   `Σ_n jet[n,m,a] · B_GN^J[n,(i,a),(i',a')] · jet[n,m',a']`.
    /// This drops only the indefinite residual-curvature term, matching the
    /// file-wide PSD-majorizer convention for Newton / Arrow-Schur blocks.
    /// The flat β layout is `β[beta_offsets[k] + m·p + i] = B_k[m, i]`, so each
    /// atom's contribution lands in its own decoder span. The isometry penalty is
    /// unscaled at the row-block (Psi) tier — mirroring its coord-block routing
    /// and `analytic_penalty_value_total` — so no `penalty_scale` is applied here.
    fn add_sae_isometry_beta_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        coord: &LatentCoordValues,
        corrected: &Arc<IsometryPenalty>,
        rho_local: ArrayView1<'_, f64>,
        dense_beta_curvature: bool,
    ) {
        let atom = &self.atoms[atom_idx];
        let d = coord.latent_dim();
        let p = atom.decoder_coefficients.ncols();
        let m = atom.basis_size();
        let n_obs = coord.n_obs();
        let grad_jac = corrected.grad_jacobian(coord.as_flat().view(), rho_local);
        if grad_jac.dim() != (n_obs, p * d) {
            return;
        }
        let jet = &atom.basis_jacobian;
        let beta_off = self.beta_offsets()[atom_idx];
        for basis_col in 0..m {
            for i in 0..p {
                let mut acc = 0.0;
                for n in 0..n_obs {
                    for a in 0..d {
                        acc += grad_jac[[n, i * d + a]] * jet[[n, basis_col, a]];
                    }
                }
                sys.gb[beta_off + basis_col * p + i] += acc;
            }
        }
        if !dense_beta_curvature {
            return;
        }
        let Some(jac) = corrected.jacobian_cache() else {
            return;
        };
        if jac.dim() != (n_obs, p * d) {
            return;
        }
        let mut weighted_jacobian_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            let Some(wj) = Self::sae_isometry_weighted_jacobian_row(corrected, &jac, n, p, d)
            else {
                return;
            };
            weighted_jacobian_rows.push(wj);
        }
        // Resolve the learnable isometry strength `scalar_weight · exp(rho)` in
        // log-space with a clamped exponent: the naive `scalar_weight *
        // rho.exp()` overflows to `inf` for `rho ≳ 709`, and the downstream
        // `inf · jacobian` / `inf · 0.0` then injects NaN into the GN curvature
        // block and β-penalty, poisoning the joint solve (#742, Issue 4).
        let mu = resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
        let mut metric_jvp = Array2::<f64>::zeros((d, d));
        let mut jac_hvp = Array2::<f64>::zeros((p, d));
        let mut beta_hvp = Array2::<f64>::zeros((m, p));
        for probe_basis_col in 0..m {
            for probe_output in 0..p {
                beta_hvp.fill(0.0);
                for n in 0..n_obs {
                    let wj = &weighted_jacobian_rows[n];
                    metric_jvp.fill(0.0);
                    for a in 0..d {
                        let probe_jet_a = jet[[n, probe_basis_col, a]];
                        for b in 0..d {
                            metric_jvp[[a, b]] = probe_jet_a * wj[[probe_output, b]]
                                + wj[[probe_output, a]] * jet[[n, probe_basis_col, b]];
                        }
                    }
                    jac_hvp.fill(0.0);
                    for i in 0..p {
                        for c in 0..d {
                            let mut acc = 0.0;
                            for b in 0..d {
                                acc += metric_jvp[[c, b]] * wj[[i, b]];
                            }
                            for a in 0..d {
                                acc += metric_jvp[[a, c]] * wj[[i, a]];
                            }
                            jac_hvp[[i, c]] = mu * acc;
                        }
                    }
                    for basis_row in 0..m {
                        for i in 0..p {
                            let mut acc = 0.0;
                            for a in 0..d {
                                acc += jac_hvp[[i, a]] * jet[[n, basis_row, a]];
                            }
                            beta_hvp[[basis_row, i]] += acc;
                        }
                    }
                }
                let beta_col = beta_off + probe_basis_col * p + probe_output;
                for basis_row in 0..m {
                    for i in 0..p {
                        sys.hbb[[beta_off + basis_row * p + i, beta_col]] +=
                            beta_hvp[[basis_row, i]];
                    }
                }
            }
        }
    }

    fn live_decoder_incoherence_penalty(
        &self,
        base: &Arc<DecoderIncoherencePenalty>,
    ) -> Option<DecoderIncoherencePenalty> {
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            return None;
        }
        let p = self.output_dim();
        let block_sizes: Vec<usize> = self.atoms.iter().map(|atom| atom.basis_size()).collect();
        let m_total: usize = block_sizes.iter().sum();
        let gates = self.assignment.assignments();
        let n = gates.nrows();
        let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        let mut coactivation = Array2::<f64>::zeros((k_atoms, k_atoms));
        for j in 0..k_atoms {
            for k in 0..k_atoms {
                let mut s = 0.0;
                for row in 0..n {
                    s += gates[[row, j]] * gates[[row, k]];
                }
                coactivation[[j, k]] = s * inv_n;
            }
        }
        let mut per_fit: DecoderIncoherencePenalty = (**base).clone();
        per_fit.block_sizes = block_sizes;
        per_fit.p_out = p;
        per_fit.target = PsiSlice {
            range: 0..m_total * p,
            latent_dim: Some(m_total),
        };
        per_fit.coactivation = coactivation;
        Some(per_fit)
    }

    fn live_mechanism_sparsity_penalties(
        &self,
        base: &Arc<MechanismSparsityPenalty>,
    ) -> Vec<(MechanismSparsityPenalty, usize, usize)> {
        let beta_offsets = self.beta_offsets();
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.atoms.len());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let start = beta_offsets[atom_idx];
            let end = start + m * p;
            let mut per_atom: MechanismSparsityPenalty = (**base).clone();
            per_atom.target = PsiSlice {
                range: start..end,
                latent_dim: Some(m),
            };
            out.push((per_atom, start, end));
        }
        out
    }

    fn live_nuclear_norm_penalties(
        &self,
        base: &Arc<NuclearNormPenalty>,
    ) -> Vec<(NuclearNormPenalty, usize, usize)> {
        let beta_offsets = self.beta_offsets();
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.atoms.len());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let start = beta_offsets[atom_idx];
            let end = start + m * p;
            let mut per_atom: NuclearNormPenalty = (**base).clone();
            per_atom.n_eff = m;
            per_atom.target = PsiSlice {
                range: start..end,
                latent_dim: Some(p),
            };
            out.push((per_atom, start, end));
        }
        out
    }

    fn add_sae_beta_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        // MechanismSparsityPenalty is a group-lasso over a single
        // (latent_dim, p) decoder matrix and indexes its target via
        // `target.range.start + latent * p + feature`, treating its range as
        // one contiguous (M, p) block. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so for K≥2
        // (and in general for K=1, where it collapses to the same single
        // block) the penalty must operate per atom on its own
        // `[beta_offsets[k] .. beta_offsets[k+1])` slice with `latent_dim = M_k`.
        // Build a per-atom view of the penalty (cloning only the cheap
        // descriptor: range + latent_dim) and accumulate each atom's
        // contribution into the corresponding β segment. This removes the
        // K≥2 limitation (#240) at root rather than guarding it away.
        // DecoderIncoherencePenalty (#671) is a cross-atom decoder
        // column-space incoherence term restricted to co-activating atom pairs.
        // Its descriptor carries only placeholder shape/co-activation: the live
        // M_k (per-atom basis sizes), p_out, β target span, and the per-pair
        // co-activation weights `W[j,k] = mean_n gate[n,j]·gate[n,k]` are all
        // injected here from the current SAE state before the penalty's
        // gradient / PSD curvature are accumulated into the β-tier system.
        if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
            let Some(per_fit) = self.live_decoder_incoherence_penalty(base) else {
                return false;
            };
            let beta_dim = self.beta_dim();
            let grad = per_fit.grad_target(target_beta, rho_local);
            for j in 0..beta_dim {
                sys.gb[j] += penalty_scale * grad[j];
            }
            if !dense_beta_curvature {
                return true;
            }
            // `hbb` is the PSD Newton / PIRLS curvature block: probe the PSD
            // majorizer (the Gauss-Newton Hessian, which is already PSD here).
            let mut probe = Array1::<f64>::zeros(beta_dim);
            for j in 0..beta_dim {
                probe.fill(0.0);
                probe[j] = 1.0;
                let hv = per_fit.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                for i in 0..beta_dim {
                    sys.hbb[[i, j]] += penalty_scale * hv[i];
                }
            }
            return true;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            let mut any = false;
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                any |= self.add_sae_mech_sparsity_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                    dense_beta_curvature,
                );
            }
            return any;
        }
        // NuclearNormPenalty is a smoothed sum of singular values of a single
        // (n_eff, latent_dim) matrix. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so it must
        // operate per atom on that atom's own `[beta_offsets[k] .. +M_k*p)`
        // slice as an `M_k × p` matrix (`n_eff = M_k`, `latent_dim = p`). This
        // penalizes the embedding rank of each atom's decoder independently.
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            let mut any = false;
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                any |= self.add_sae_nuclear_norm_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                    dense_beta_curvature,
                );
            }
            return any;
        }
        let k = self.beta_dim();
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            sys.gb[j] += penalty_scale * grad[j];
        }
        if !dense_beta_curvature {
            return true;
        }
        // `hbb` is the PSD Newton / PIRLS curvature block for the β tier:
        // accumulate the PSD majorizer (exact for convex penalties), not the
        // indefinite exact Hessian, so the solve stays positive-definite.
        if let Some(diag) = penalty.psd_majorizer_diag(target_beta, rho_local) {
            for j in 0..k {
                sys.hbb[[j, j]] += penalty_scale * diag[j];
            }
            return true;
        }
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
        true
    }

    /// Accumulate one atom's MechanismSparsity contribution into `sys`. The
    /// `per_atom` penalty has its `target.range` set to that atom's β segment
    /// `[start, end)` and `latent_dim = M_k`, so `grad_target` / `hvp` return
    /// full-length β vectors whose nonzero support lies inside `[start, end)`.
    /// The Hessian probe only needs to sweep that segment, and its support is
    /// likewise confined to `[start, end)`, so the inner accumulation is
    /// quadratic in the atom's block size rather than the full β dimension.
    fn add_sae_mech_sparsity_atom(
        &self,
        sys: &mut ArrowSchurSystem,
        per_atom: &MechanismSparsityPenalty,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        start: usize,
        end: usize,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        let grad = per_atom.grad_target(target_beta, rho_local);
        for j in start..end {
            sys.gb[j] += penalty_scale * grad[j];
        }
        if !dense_beta_curvature {
            return true;
        }
        let k = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(k);
        for j in start..end {
            probe.fill(0.0);
            probe[j] = 1.0;
            // `hbb` is the PSD Newton / PIRLS curvature block, so probe the PSD
            // majorizer. The group-lasso Hessian `factor·(I − ŵŵᵀ)/‖w‖` is
            // already PSD, so its majorizer equals the exact Hessian (the trait
            // default delegates), but we use the majorizer name to honor the
            // curvature-block contract uniformly with the other SAE penalties.
            let hv = per_atom.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in start..end {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
        true
    }

    /// Accumulate one atom's NuclearNorm contribution into `sys`. The
    /// `per_atom` penalty has `n_eff = M_k` and `latent_dim = Some(p)`, so it
    /// treats this atom's β segment `[start, end)` as an `M_k × p` matrix and
    /// shrinks its singular spectrum (embedding rank).
    ///
    /// Unlike MechanismSparsity, `NuclearNormPenalty::grad_target` / `hvp`
    /// reshape the *entire* `target` argument they are given (they do not use
    /// `self.target.range` to slice), so the local `M_k × p` block is passed
    /// directly and the returned vectors are local (length `M_k*p`). The PSD
    /// curvature is probed via `psd_majorizer_hvp`, which for NuclearNorm has
    /// no diagonal majorizer and delegates to its analytic spectral HVP.
    fn add_sae_nuclear_norm_atom(
        &self,
        sys: &mut ArrowSchurSystem,
        per_atom: &NuclearNormPenalty,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        start: usize,
        end: usize,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        let block = target_beta.slice(s![start..end]);
        let block_len = end - start;
        let grad = per_atom.grad_target(block, rho_local);
        for local in 0..block_len {
            sys.gb[start + local] += penalty_scale * grad[local];
        }
        if !dense_beta_curvature {
            return true;
        }
        let mut probe = Array1::<f64>::zeros(block_len);
        for local in 0..block_len {
            probe.fill(0.0);
            probe[local] = 1.0;
            let hv = per_atom.psd_majorizer_hvp(block, rho_local, probe.view());
            for i in 0..block_len {
                sys.hbb[[start + i, start + local]] += penalty_scale * hv[i];
            }
        }
        true
    }

    pub fn solve_newton_step(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let sys = self
            .assemble_arrow_schur(target, rho, analytic_penalties)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // Self-heal against non-PD per-row blocks produced by PCA-seeded
        // latent coordinates on subset / out-of-sample data (#163, #175):
        // route every Newton-step solve through the Ceres-style LM ridge
        // escalation, reusing the caller-supplied Tikhonov ridges
        // (`ridge_ext_coord`, `ridge_beta`) as the base damping. No new
        // tuning knobs — just the existing proximal-correction schedule.
        let plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let options = plan.solve_options_for_border_dim(sys.k);
        solve_with_lm_escalation_inner(&sys, ridge_ext_coord, ridge_beta, &options)
            .map(|(delta_t, delta_beta, _diag)| (delta_t, delta_beta))
    }

    pub fn apply_newton_step(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, true)
    }

    pub fn apply_newton_step_external_basis_refresh(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
    ) -> Result<(), String> {
        self.apply_newton_step_impl(delta_ext_coord, delta_beta, step_size, false)
    }

    /// Capture the mutable state perturbed by an `apply_newton_step` +
    /// `loss` line-search trial, plus the row-layout state read by
    /// `apply_newton_step` when unpacking compact Newton steps. See
    /// [`SaeManifoldMutableState`].
    fn snapshot_mutable_state(&self) -> SaeManifoldMutableState {
        let atoms = self
            .atoms
            .iter()
            .map(|atom| {
                (
                    atom.basis_values.clone(),
                    atom.basis_jacobian.clone(),
                    atom.decoder_coefficients.clone(),
                    atom.smooth_penalty.clone(),
                )
            })
            .collect();
        SaeManifoldMutableState {
            atoms,
            logits: self.assignment.logits.clone(),
            coords: self.assignment.coords.clone(),
            last_row_layout: self.last_row_layout.clone(),
        }
    }

    /// Restore the mutable state captured by [`Self::snapshot_mutable_state`].
    /// Assigns into the existing arrays in place so the restore reuses the
    /// already-allocated buffers rather than reallocating per trial.
    fn restore_mutable_state(&mut self, snapshot: &SaeManifoldMutableState) {
        for (atom, (basis_values, basis_jacobian, decoder, smooth_penalty)) in
            self.atoms.iter_mut().zip(snapshot.atoms.iter())
        {
            atom.basis_values.assign(basis_values);
            atom.basis_jacobian.assign(basis_jacobian);
            atom.decoder_coefficients.assign(decoder);
            atom.smooth_penalty.assign(smooth_penalty);
        }
        self.assignment.logits.assign(&snapshot.logits);
        self.assignment.coords.clone_from(&snapshot.coords);
        self.last_row_layout.clone_from(&snapshot.last_row_layout);
    }

    fn refresh_basis_from_current_coords(&mut self) -> Result<(), String> {
        for atom_idx in 0..self.k_atoms() {
            let coords = self.assignment.coords[atom_idx].as_matrix();
            self.atoms[atom_idx].refresh_basis(coords.view())?;
        }
        Ok(())
    }

    fn canonicalize_affine_gauge_after_accept(&mut self) -> Result<(), String> {
        for atom_idx in 0..self.k_atoms() {
            if !matches!(
                self.atoms[atom_idx].basis_kind,
                SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Duchon
            ) {
                continue;
            }
            self.canonicalize_atom_affine_gauge(atom_idx)?;
        }
        Ok(())
    }

    fn canonicalize_atom_affine_gauge(&mut self, atom_idx: usize) -> Result<(), String> {
        let n = self.n_obs();
        let d = self.assignment.coords[atom_idx].latent_dim();
        if n == 0 || d == 0 {
            return Ok(());
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref() else {
            return Ok(());
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let weights = self.atom_affine_gauge_weights(atom_idx)?;
        let weight_sum: f64 = weights.iter().sum();
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return Ok(());
        }

        let mut shift = vec![0.0_f64; d];
        for row in 0..n {
            let w = weights[row];
            for axis in 0..d {
                shift[axis] += w * coords[[row, axis]];
            }
        }
        for value in &mut shift {
            *value /= weight_sum;
        }

        let mut scale = vec![1.0_f64; d];
        let mut changed = false;
        for axis in 0..d {
            let mut var = 0.0_f64;
            for row in 0..n {
                let centered = coords[[row, axis]] - shift[axis];
                var += weights[row] * centered * centered;
            }
            let rms = (var / weight_sum).sqrt();
            if rms.is_finite() && rms > 1.0e-12 {
                scale[axis] = rms;
            }
            if shift[axis].abs() > 1.0e-12 || (scale[axis] - 1.0).abs() > 1.0e-12 {
                changed = true;
            }
        }
        if !changed {
            return Ok(());
        }

        let Some(new_evaluator) = evaluator.affine_transformed_evaluator(
            &shift,
            &scale,
            self.atoms[atom_idx].basis_size(),
        )?
        else {
            return Ok(());
        };

        let mut new_coords = coords.clone();
        for row in 0..n {
            for axis in 0..d {
                new_coords[[row, axis]] = (coords[[row, axis]] - shift[axis]) / scale[axis];
            }
        }
        let (new_phi, new_jet) = if self.atoms[atom_idx].homotopy_eta == 1.0 {
            new_evaluator.evaluate(new_coords.view())?
        } else {
            let evaluated = new_evaluator
                .evaluate_phi_eta(new_coords.view(), self.atoms[atom_idx].homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        let old_phi = self.atoms[atom_idx].basis_values.clone();
        if new_phi.dim() != old_phi.dim() {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_affine_gauge: transformed basis shape {:?} != {:?}",
                new_phi.dim(),
                old_phi.dim()
            ));
        }
        let transport = solve_basis_transport(new_phi.view(), old_phi.view())?;
        let old_decoder = self.atoms[atom_idx].decoder_coefficients.clone();
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let new_decoder = fast_ab(&transport, &old_decoder);
        let old_fit = fast_ab(&old_phi, &old_decoder);
        let new_fit = fast_ab(&new_phi, &new_decoder);
        let fit_scale = old_fit
            .iter()
            .chain(new_fit.iter())
            .fold(1.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs = old_fit
            .iter()
            .zip(new_fit.iter())
            .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
        if max_abs > 1.0e-8 * fit_scale {
            return Ok(());
        }

        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = new_decoder;
        let base: Arc<dyn SaeBasisEvaluator> = new_evaluator.clone();
        atom.basis_evaluator = Some(base);
        atom.basis_second_jet = Some(new_evaluator);
        atom.smooth_penalty =
            transport_smooth_penalty_for_decoder(transport.view(), old_smooth_penalty.view())?;
        Ok(())
    }

    fn atom_affine_gauge_weights(&self, atom_idx: usize) -> Result<Array1<f64>, String> {
        let n = self.n_obs();
        let mut weights = Array1::<f64>::zeros(n);
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            let mut w = assignments[atom_idx].max(0.0);
            if let Some(row_weights) = self.row_loss_weights.as_ref() {
                w *= row_weights[row].max(0.0);
            }
            weights[row] = if w.is_finite() { w } else { 0.0 };
        }
        Ok(weights)
    }

    /// #1019 stage 1 — post-fit arc-length (unit-speed) chart canonicalization
    /// for every fitted `d = 1` atom with circle or interval topology.
    ///
    /// Image-frozen and gauge-legal: each eligible atom's latent chart is
    /// replaced by the canonical representative of its `Diff(M)` orbit — the
    /// unit-speed (arc-length) chart for `d = 1` circle/interval atoms, the
    /// minimum-isometry-defect flow chart for `d = 2` torus atoms (#1019
    /// stage 2) — and the decoder is recomposed by exact least squares so the
    /// decoded image is unchanged
    /// ([`crate::terms::sae_chart_canonicalization`]). Atoms whose basis
    /// cannot absorb the reparameterized image within the recomposition
    /// tolerance are left untouched (honest fallback, recorded by the flag
    /// staying `false`).
    ///
    /// The whole pass is additionally gated on the penalized objective: the
    /// canonical state is kept only when the same scalar the line search
    /// minimized does not increase beyond the image-invariance tolerance
    /// (the intrinsic smoothness penalty is reparameterization-invariant by
    /// design, so a genuine increase means the transport went numerically
    /// wrong and the fitted state is restored verbatim).
    ///
    /// Runs automatically from `into_fitted` after the joint fit converges,
    /// before the payload / residual-gauge certificate is assembled — never
    /// a flag (magic-by-default).
    pub fn canonicalize_charts_post_fit(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<(), String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, CanonicalChartTopology,
        };
        /// Which canonical-representative construction applies to an atom:
        /// arc length for `d = 1` (#1019 stage 1), the minimum-isometry-defect
        /// flow for `d = 2` torus atoms (#1019 stage 2).
        enum ChartPlan {
            UnitSpeed(CanonicalChartTopology),
            TorusFlow { period: f64 },
        }
        let mut eligible: Vec<(usize, ChartPlan)> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            if atom.basis_evaluator.is_none()
                || atom.homotopy_eta != 1.0
                || self.assignment.coords[atom_idx].latent_dim() != atom.latent_dim
            {
                continue;
            }
            // Same fraction-of-period convention as the latent manifold
            // wiring (`SaeAtomBasisKind::latent_manifold`): the harmonic
            // evaluators read `t` as a fraction of one period.
            let plan = match (&atom.basis_kind, atom.latent_dim) {
                (SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus, 1) => {
                    ChartPlan::UnitSpeed(CanonicalChartTopology::Circle { period: 1.0 })
                }
                (SaeAtomBasisKind::Duchon | SaeAtomBasisKind::EuclideanPatch, 1) => {
                    ChartPlan::UnitSpeed(CanonicalChartTopology::Interval)
                }
                // #1019 stage 2: d = 2 torus atoms pin to the
                // minimum-isometry-defect flow representative.
                (SaeAtomBasisKind::Torus, 2) => ChartPlan::TorusFlow { period: 1.0 },
                // d = 1 never matches Sphere; Precomputed bases carry no
                // evaluator semantics to re-express the image in; S² has no
                // global pole-free flow basis (hairy ball), so sphere charts
                // are honestly left as fitted.
                _ => continue,
            };
            eligible.push((atom_idx, plan));
        }
        if eligible.is_empty() {
            return Ok(());
        }

        let snapshot = self.snapshot_mutable_state();
        let prior_flags: Vec<bool> = self
            .atoms
            .iter()
            .map(|atom| atom.chart_canonicalized)
            .collect();
        let pre_total = self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;

        let mut any_changed = false;
        for (atom_idx, plan) in &eligible {
            let outcome = match plan {
                ChartPlan::UnitSpeed(topology) => {
                    self.canonicalize_atom_unit_speed_chart(*atom_idx, topology)
                }
                ChartPlan::TorusFlow { period } => {
                    self.canonicalize_atom_torus_flow_chart(*atom_idx, *period)
                }
            };
            match outcome {
                Ok(changed) => any_changed |= changed,
                Err(err) => {
                    self.restore_mutable_state(&snapshot);
                    for (atom, flag) in self.atoms.iter_mut().zip(prior_flags.iter()) {
                        atom.chart_canonicalized = *flag;
                    }
                    return Err(err);
                }
            }
        }
        if !any_changed {
            return Ok(());
        }

        // Keep the canonical state only when the optimized scalar is preserved
        // within the image-invariance tolerance (the data fit moved by at most
        // the certified recomposition residual; the intrinsic penalty is
        // reparameterization-invariant, transported exactly).
        let canonical_total = self.penalized_objective_total(target, rho, analytic_penalties, 1.0);
        let keep = match canonical_total {
            Ok(total) => {
                total.is_finite()
                    && total <= pre_total + CHART_RECOMPOSITION_REL_TOL * (1.0 + pre_total.abs())
            }
            Err(_) => false,
        };
        if !keep {
            self.restore_mutable_state(&snapshot);
            for (atom, flag) in self.atoms.iter_mut().zip(prior_flags.iter()) {
                atom.chart_canonicalized = *flag;
            }
        }
        Ok(())
    }

    /// Apply the arc-length reparameterization to one eligible `d = 1` atom.
    /// Returns `Ok(true)` when the atom was canonicalized, `Ok(false)` on an
    /// honest skip (degenerate chart, basis not closed under the
    /// reparameterization, or per-row image drift above tolerance).
    fn canonicalize_atom_unit_speed_chart(
        &mut self,
        atom_idx: usize,
        topology: &crate::terms::sae_chart_canonicalization::CanonicalChartTopology,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, unit_speed_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let row_coords = coords.column(0).to_owned();
        let Some(repar) = unit_speed_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            row_coords.view(),
            topology,
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates.
        let mut new_coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            new_coords[[row, 0]] = repar.new_row_coords[row];
        }
        let (new_phi, new_jet) = if self.atoms[atom_idx].homotopy_eta == 1.0 {
            evaluator.evaluate(new_coords.view())?
        } else {
            let evaluated =
                evaluator.evaluate_phi_eta(new_coords.view(), self.atoms[atom_idx].homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_unit_speed_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the grid gate certified the curve at
        // the audit nodes; this certifies it at the coordinates the fit
        // actually sits on. Same honest-fallback contract.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as the affine
        // gauge pass).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// Apply the minimum-isometry-defect flow reparameterization (#1019
    /// stage 2) to one eligible `d = 2` torus atom. Returns `Ok(true)` when
    /// the atom was canonicalized, `Ok(false)` on an honest skip (degenerate
    /// or already-canonical chart, only folded/non-improving flow candidates,
    /// basis not closed under the reparameterization, or per-row image drift
    /// above tolerance).
    fn canonicalize_atom_torus_flow_chart(
        &mut self,
        atom_idx: usize,
        period: f64,
    ) -> Result<bool, String> {
        use crate::terms::sae_chart_canonicalization::{
            CHART_RECOMPOSITION_REL_TOL, torus_isometry_flow_reparameterization,
        };
        let n = self.n_obs();
        if n == 0 {
            return Ok(false);
        }
        let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.as_ref().cloned() else {
            return Ok(false);
        };
        let coords = self.assignment.coords[atom_idx].as_matrix();
        let Some(repar) = torus_isometry_flow_reparameterization(
            evaluator.as_ref(),
            self.atoms[atom_idx].decoder_coefficients.view(),
            coords.view(),
            period,
        )?
        else {
            return Ok(false);
        };

        // Per-row basis/jet at the canonical coordinates (eligibility pins
        // `homotopy_eta == 1.0`, so the plain evaluate IS the dialed path).
        let new_coords = repar.new_row_coords.clone();
        let (new_phi, new_jet) = evaluator.evaluate(new_coords.view())?;
        if new_phi.dim() != self.atoms[atom_idx].basis_values.dim()
            || new_jet.dim() != self.atoms[atom_idx].basis_jacobian.dim()
        {
            return Err(format!(
                "SaeManifoldTerm::canonicalize_atom_torus_flow_chart: canonical basis {:?} / jet {:?} must match the fitted shapes {:?} / {:?}",
                new_phi.dim(),
                new_jet.dim(),
                self.atoms[atom_idx].basis_values.dim(),
                self.atoms[atom_idx].basis_jacobian.dim()
            ));
        }

        // Per-row image-invariance gate: the audit grid certified the image
        // at the transport nodes; this certifies it at the coordinates the
        // fit actually sits on. Same honest-fallback contract as d = 1.
        let old_fit = fast_ab(
            &self.atoms[atom_idx].basis_values,
            &self.atoms[atom_idx].decoder_coefficients,
        );
        let new_fit = fast_ab(&new_phi, &repar.new_decoder);
        let mut fit_scale = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (a, b) in old_fit.iter().zip(new_fit.iter()) {
            fit_scale = fit_scale.max(a.abs()).max(b.abs());
            max_abs = max_abs.max((a - b).abs());
        }
        if !(fit_scale.is_finite() && max_abs.is_finite()) {
            return Ok(false);
        }
        if fit_scale > 0.0 && max_abs > CHART_RECOMPOSITION_REL_TOL * fit_scale {
            return Ok(false);
        }

        // Commit: canonical coordinates, basis, decoder, and the congruence-
        // transported smoothness Gram (`B̃ᵀ S̃ B̃ = Bᵀ S B`, same as the affine
        // gauge pass and the d = 1 path).
        let old_smooth_penalty = self.atoms[atom_idx].smooth_penalty.clone();
        let flat = Array1::from_iter(new_coords.iter().copied());
        self.assignment.coords[atom_idx].set_flat(flat.view());
        let atom = &mut self.atoms[atom_idx];
        atom.basis_values = new_phi;
        atom.basis_jacobian = new_jet;
        atom.decoder_coefficients = repar.new_decoder;
        atom.smooth_penalty = transport_smooth_penalty_for_decoder(
            repar.decoder_transport.view(),
            old_smooth_penalty.view(),
        )?;
        atom.chart_canonicalized = true;
        Ok(true)
    }

    /// The iterate scale `1 + ‖(logits, coords, decoder)‖` used to make the
    /// inner KKT gradient and Newton-step tolerances relative. This is the
    /// SINGLE source of truth for that scale: `reml_criterion`'s convergence
    /// gate and `run_joint_fit_arrow_schur`'s non-descent stationarity gate
    /// must agree on it, or a point one of them calls converged is mid-flight
    /// to the other (the objective↔gradient desync class).
    fn inner_iterate_scale(&self) -> f64 {
        let mut iterate_norm_sq = 0.0_f64;
        for &v in self.assignment.logits.iter() {
            iterate_norm_sq += v * v;
        }
        for coords in &self.assignment.coords {
            let matrix = coords.as_matrix();
            for &v in matrix.iter() {
                iterate_norm_sq += v * v;
            }
        }
        for atom in &self.atoms {
            for &v in atom.decoder_coefficients.iter() {
                iterate_norm_sq += v * v;
            }
        }
        1.0 + iterate_norm_sq.sqrt()
    }

    fn quotient_newton_step_norm_sq(
        &self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        raw_step_norm_sq: f64,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        if delta_ext_coord.len() != n * q || delta_beta.len() != beta_dim {
            return Ok(raw_step_norm_sq);
        }
        let mut residual = Array1::<f64>::zeros(delta_ext_coord.len() + delta_beta.len());
        for i in 0..delta_ext_coord.len() {
            residual[i] = delta_ext_coord[i];
        }
        let beta_base = delta_ext_coord.len();
        for i in 0..delta_beta.len() {
            residual[beta_base + i] = delta_beta[i];
        }

        let mut orthonormal: Vec<Array1<f64>> = Vec::new();
        for mut gauge in self.dense_step_gauge_vectors()? {
            for basis in &orthonormal {
                let coeff = gauge.dot(basis);
                for i in 0..gauge.len() {
                    gauge[i] -= coeff * basis[i];
                }
            }
            let norm_sq = gauge.iter().map(|v| v * v).sum::<f64>();
            if norm_sq <= 1.0e-24 || !norm_sq.is_finite() {
                continue;
            }
            let inv_norm = norm_sq.sqrt().recip();
            for v in gauge.iter_mut() {
                *v *= inv_norm;
            }
            let coeff = residual.dot(&gauge);
            for i in 0..residual.len() {
                residual[i] -= coeff * gauge[i];
            }
            orthonormal.push(gauge);
        }
        let quotient = residual.iter().map(|v| v * v).sum::<f64>();
        Ok(if quotient.is_finite() {
            quotient.max(0.0).min(raw_step_norm_sq)
        } else {
            raw_step_norm_sq
        })
    }

    fn dense_step_gauge_vectors(&self) -> Result<Vec<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let coord_offsets = self.assignment.coord_offsets();
        let beta_offsets = self.beta_offsets();
        let total_len = n * q + self.beta_dim();
        let mut out = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            let d = self.assignment.coords[atom_idx].latent_dim();
            let coords = self.assignment.coords[atom_idx].as_matrix();
            match self.atoms[atom_idx].basis_kind {
                SaeAtomBasisKind::EuclideanPatch => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        for row in 0..n {
                            field[[row, axis]] = coords[[row, axis]];
                        }
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                SaeAtomBasisKind::Duchon => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        for row in 0..n {
                            field[[row, axis]] = coords[[row, axis]];
                        }
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                    for axis in 0..d {
                        let mut field = Array2::<f64>::zeros((n, d));
                        field.column_mut(axis).fill(1.0);
                        if let Some(g) = self.dense_step_gauge_vector_from_field(
                            atom_idx,
                            field.view(),
                            &coord_offsets,
                            &beta_offsets,
                            total_len,
                        )? {
                            out.push(g);
                        }
                    }
                }
                _ => {}
            }
        }
        if p == 0 {
            return Ok(Vec::new());
        }
        Ok(out)
    }

    fn row_gauge_deflation_for_layout(
        &self,
        row_layout: Option<&SaeRowLayout>,
    ) -> Option<ArrowRowGaugeDeflation> {
        let n = self.n_obs();
        let mut rows: Vec<Vec<Array1<f64>>> = Vec::with_capacity(n);
        for row in 0..n {
            let q_row = match row_layout {
                Some(layout) => layout.row_q_active(row),
                None => self.assignment.row_block_dim(),
            };
            rows.push(Vec::with_capacity(self.k_atoms().min(4)));
            match row_layout {
                Some(layout) => {
                    for (active_pos, &atom_idx) in layout.active_atoms[row].iter().enumerate() {
                        let start = layout.coord_starts[row][active_pos];
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            start,
                            q_row,
                        );
                    }
                }
                None => {
                    let coord_offsets = self.assignment.coord_offsets();
                    for atom_idx in 0..self.k_atoms() {
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            coord_offsets[atom_idx],
                            q_row,
                        );
                    }
                }
            }
        }
        if rows.iter().all(Vec::is_empty) {
            None
        } else {
            Some(ArrowRowGaugeDeflation::new(rows))
        }
    }

    fn push_atom_row_gauge_deflations(
        &self,
        row_dirs: &mut Vec<Array1<f64>>,
        row: usize,
        atom_idx: usize,
        coord_start: usize,
        q_row: usize,
    ) {
        let d = self.assignment.coords[atom_idx].latent_dim();
        let mut tangent = vec![0.0_f64; self.output_dim()];
        match self.atoms[atom_idx].basis_kind {
            SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Duchon => {
                for axis in 0..d {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut tangent);
                    if tangent.iter().map(|&v| v * v).sum::<f64>() <= 1.0e-24 {
                        continue;
                    }
                    let mut translation = Array1::<f64>::zeros(q_row);
                    translation[coord_start + axis] = 1.0;
                    row_dirs.push(translation);

                    let coord_value = self.assignment.coords[atom_idx].as_matrix()[[row, axis]];
                    let mut scale = Array1::<f64>::zeros(q_row);
                    scale[coord_start + axis] = coord_value;
                    row_dirs.push(scale);
                }
            }
            SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    self.atoms[atom_idx].fill_decoded_derivative_row(row, axis, &mut tangent);
                    if tangent.iter().map(|&v| v * v).sum::<f64>() <= 1.0e-24 {
                        continue;
                    }
                    let mut phase = Array1::<f64>::zeros(q_row);
                    phase[coord_start + axis] = 1.0;
                    row_dirs.push(phase);
                }
            }
            _ => {}
        }
    }

    fn dense_step_gauge_vector_from_field(
        &self,
        atom_idx: usize,
        field: ArrayView2<'_, f64>,
        coord_offsets: &[usize],
        beta_offsets: &[usize],
        total_len: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let atom = &self.atoms[atom_idx];
        let m = atom.basis_size();
        let d = self.assignment.coords[atom_idx].latent_dim();
        if field.dim() != (n, d) {
            return Err(format!(
                "dense_step_gauge_vector_from_field: field shape {:?} != ({n}, {d})",
                field.dim()
            ));
        }
        let mut design = Array2::<f64>::zeros((n, m));
        let mut motion = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            let a = assignments[atom_idx];
            if a == 0.0 {
                continue;
            }
            for col in 0..m {
                design[[row, col]] = a * atom.basis_values[[row, col]];
            }
            for axis in 0..d {
                let dt = field[[row, axis]];
                if dt == 0.0 {
                    continue;
                }
                for col in 0..m {
                    let w = a * dt * atom.basis_jacobian[[row, col, axis]];
                    if w == 0.0 {
                        continue;
                    }
                    for out_col in 0..p {
                        motion[[row, out_col]] += w * atom.decoder_coefficients[[col, out_col]];
                    }
                }
            }
        }
        let raw = motion.iter().map(|v| v * v).sum::<f64>();
        if raw <= f64::MIN_POSITIVE || !raw.is_finite() {
            return Ok(None);
        }
        motion.mapv_inplace(|v| -v);
        let delta_b = solve_design_least_squares(design.view(), motion.view())?;
        let mut gauge = Array1::<f64>::zeros(total_len);
        for row in 0..n {
            let row_base = row * q + coord_offsets[atom_idx];
            for axis in 0..d {
                gauge[row_base + axis] = field[[row, axis]];
            }
        }
        let beta_base = n * q + beta_offsets[atom_idx];
        for col in 0..m {
            for out_col in 0..p {
                gauge[beta_base + col * p + out_col] = delta_b[[col, out_col]];
            }
        }
        Ok(Some(gauge))
    }

    /// #976 Layer-1 guard ledger for the most recent joint fit (empty when no
    /// atom ever breached the active-mass floor). A terminal event here is the
    /// canonical death-proposal feed for the structure search.
    pub fn collapse_events(&self) -> &[CollapseEvent] {
        &self.collapse_events
    }

    /// Record an externally-observed collapse event on this term's guard ledger
    /// (#976/#997). The joint fit appends its own events during
    /// [`Self::run_joint_fit_arrow_schur`]; this lets a structure-search driver
    /// (or a streaming chunk loop reconciling per-chunk guard outcomes) feed a
    /// collapse observation back onto the term so the next
    /// [`crate::solver::structure_harvest::harvest_move_proposals`] pass sees it
    /// as a death trigger.
    pub fn record_collapse_event(&mut self, event: CollapseEvent) {
        self.collapse_events.push(event);
    }

    /// #1023 final fitted-data guard: a fit with material active atoms whose
    /// fitted matrix explains essentially none of the training variation is a
    /// structural collapse, not a quiet success. Record terminal CollapseEvents
    /// so the #976 structure-search layer and payload ledger see the outcome.
    pub fn record_fit_data_collapse_if_needed(
        &mut self,
        target: ArrayView2<'_, f64>,
        fitted: ArrayView2<'_, f64>,
        assignments: ArrayView2<'_, f64>,
        iteration: usize,
    ) -> Result<bool, String> {
        if target.dim() != fitted.dim() {
            return Err(format!(
                "SaeManifoldTerm::record_fit_data_collapse_if_needed: target {:?} != fitted {:?}",
                target.dim(),
                fitted.dim()
            ));
        }
        let (n, p) = target.dim();
        if assignments.dim() != (n, self.k_atoms()) {
            return Err(format!(
                "SaeManifoldTerm::record_fit_data_collapse_if_needed: assignments {:?} != ({}, {})",
                assignments.dim(),
                n,
                self.k_atoms()
            ));
        }
        if n == 0 || p == 0 || self.k_atoms() == 0 {
            return Ok(false);
        }

        let mut means = vec![0.0_f64; p];
        for col in 0..p {
            let mut acc = 0.0;
            for row in 0..n {
                acc += target[[row, col]];
            }
            means[col] = acc / n as f64;
        }
        let mut ssr = 0.0_f64;
        let mut sst = 0.0_f64;
        for row in 0..n {
            for col in 0..p {
                let r = target[[row, col]] - fitted[[row, col]];
                ssr += r * r;
                let centered = target[[row, col]] - means[col];
                sst += centered * centered;
            }
        }
        if !(ssr.is_finite() && sst.is_finite()) || sst <= f64::MIN_POSITIVE {
            return Ok(false);
        }
        let ev = 1.0 - ssr / sst;
        if !(ev.is_finite() && ev <= SAE_FIT_DATA_COLLAPSE_EV_FLOOR) {
            return Ok(false);
        }

        let mut collapsed_active_atom = false;
        for atom in 0..self.k_atoms() {
            let active_mass = assignments
                .column(atom)
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            if active_mass <= 1.0e-8 {
                continue;
            }
            collapsed_active_atom = true;
            let already_terminal = self
                .collapse_events
                .iter()
                .any(|e| e.atom == atom && e.action == CollapseAction::Terminal);
            if already_terminal {
                continue;
            }
            self.collapse_events.push(CollapseEvent {
                iteration,
                atom,
                max_active_mass: ev,
                floor: SAE_FIT_DATA_COLLAPSE_EV_FLOOR,
                action: CollapseAction::Terminal,
            });
        }
        Ok(collapsed_active_atom)
    }

    /// Set the curvature-homotopy dial `η ∈ [0, 1]` on every atom (#1007). At
    /// the default `η = 1` the basis is the full curved basis; `η = 0` is the
    /// linear (Eckart-Young) relaxation. The next `refresh_basis` — which every
    /// joint-fit entry point runs — installs the dialed basis, so the dial takes
    /// effect on the following corrector solve. Errors on a non-finite or
    /// out-of-range `η`.
    pub fn set_homotopy_eta(&mut self, eta: f64) -> Result<(), String> {
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(format!(
                "SaeManifoldTerm::set_homotopy_eta: η must be finite in [0, 1]; got {eta}"
            ));
        }
        for atom in &mut self.atoms {
            atom.homotopy_eta = eta;
        }
        Ok(())
    }

    /// The most recent curvature-homotopy entry walk outcome (#1007), or `None`
    /// when no walk has run on this term. Read off the fitted term so the
    /// arrival / bifurcation / collapse outcome is observable.
    pub fn curvature_walk_report(&self) -> Option<&CurvatureWalkReport> {
        self.curvature_walk_report.as_ref()
    }

    /// Record the curvature-homotopy walk outcome on the fit payload (#1007).
    pub fn set_curvature_walk_report(&mut self, report: CurvatureWalkReport) {
        self.curvature_walk_report = Some(report);
    }

    /// Per-row reconstruction residual `r_i = fitted_i − z_i` of the current
    /// `(gates, coords, decoder)` state against `target`, in the term's native
    /// `(n, p)` layout. The curvature-homotopy predictor (#1007) contracts this
    /// against `∂Φ/∂η` to form the data-fit half of `∂g_β/∂η`.
    fn reconstruction_residual(&self, target: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let fitted = self.try_fitted()?;
        if fitted.dim() != target.dim() {
            return Err(format!(
                "SaeManifoldTerm::reconstruction_residual: fitted {:?} != target {:?}",
                fitted.dim(),
                target.dim()
            ));
        }
        Ok(&fitted - &target)
    }

    /// True when the curvature-homotopy `η` dial cannot move the basis: no
    /// atom evaluator declares curved columns (caller-managed atoms have no
    /// evaluator, hence no split — equally immovable). A one-harmonic periodic
    /// bank (`M = 3`) is the canonical case: constant + fundamental are all
    /// linear columns. Combined with an all-zero isometry ramp this makes the
    /// entry walk's corrector problem η-invariant, which
    /// [`SaeManifoldOuterObjective::run_curvature_homotopy_entry_at_rho`] uses
    /// to collapse the η-grid to its first corrector.
    fn curvature_homotopy_eta_is_inert(&self) -> Result<bool, String> {
        for atom in &self.atoms {
            if let Some(evaluator) = atom.basis_evaluator.as_ref()
                && !evaluator
                    .phi_eta_split(atom.basis_size())?
                    .curved_cols
                    .is_empty()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Per-atom curved-column basis derivative `∂Φ^η/∂η` (#1007): the raw
    /// (un-dialed) basis on each evaluator's *curved* columns and zero on the
    /// linear columns and on caller-managed atoms (no evaluator → no split).
    /// This is the η-independent derivative channel, so it is exact at any
    /// current `η`.
    fn curvature_basis_eta_derivatives(&self) -> Result<Vec<Array2<f64>>, String> {
        let n = self.n_obs();
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let mut d = Array2::<f64>::zeros((n, m));
            if let Some(evaluator) = atom.basis_evaluator.as_ref() {
                let split = evaluator.phi_eta_split(m)?;
                if !split.curved_cols.is_empty() {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    let (phi_raw, _jet) = evaluator.evaluate(coords.view())?;
                    for &col in &split.curved_cols {
                        for row in 0..n {
                            d[[row, col]] = phi_raw[[row, col]];
                        }
                    }
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// Build the β-block of the curvature-homotopy predictor RHS `∂g_β/∂η`
    /// (#1007) at the current corrected state, in the flat β layout
    /// [`Self::flatten_beta`] uses (`[atom][basis_col · p + out_col]`).
    ///
    /// The data-fit β-gradient is `g_β[k,μ,c] = Σ_i a_ik Φ^η_k[i,μ] r_i[c]`, so
    /// (W = I for the Gaussian reconstruction channel)
    /// `∂g_β/∂η[k,μ,c] = Σ_i a_ik (∂Φ^η_k[i,μ]/∂η) r_i[c]`
    /// `              + Σ_i a_ik Φ^η_k[i,μ] (∂r_i[c]/∂η)`,
    /// with `∂Φ^η/∂η` the raw curved-column basis (zero on linear columns) and
    /// `∂r_i/∂η = Σ_{k'} a_ik' (∂Φ^η_{k'}[i,:]/∂η) · B_{k'}`. The smoothness and
    /// ARD penalties do not depend on `η`, so they contribute nothing. The
    /// predictor solves `Δβ = −H⁻¹ · ∂g_β/∂η · Δη` on the cached evidence factor.
    fn curvature_beta_gradient_eta_derivative(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let residual = self.reconstruction_residual(target)?;
        let dphi_deta = self.curvature_basis_eta_derivatives()?;
        // ∂fitted_i/∂η = Σ_{k'} a_ik' (dΦ_{k'}[i,:]) · B_{k'}.
        let mut dfitted = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?;
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_k = a[atom_idx];
                if a_k == 0.0 {
                    continue;
                }
                let m = atom.basis_size();
                for mu in 0..m {
                    let dphi = dphi_deta[atom_idx][[row, mu]];
                    if dphi == 0.0 {
                        continue;
                    }
                    let w = a_k * dphi;
                    for c in 0..p {
                        dfitted[[row, c]] += w * atom.decoder_coefficients[[mu, c]];
                    }
                }
            }
        }
        // ∂g_β/∂η[k,μ,c] = Σ_i a_ik (dΦ_k[i,μ] r_i[c] + Φ^η_k[i,μ] dfitted_i[c]).
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?;
            for (atom_idx, atom) in self.atoms.iter().enumerate() {
                let a_k = a[atom_idx];
                if a_k == 0.0 {
                    continue;
                }
                let m = atom.basis_size();
                let off = offsets[atom_idx];
                for mu in 0..m {
                    let dphi = dphi_deta[atom_idx][[row, mu]];
                    let phi = atom.basis_values[[row, mu]];
                    for c in 0..p {
                        out[off + mu * p + c] +=
                            a_k * (dphi * residual[[row, c]] + phi * dfitted[[row, c]]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// #976 Layer-1 guard 3: the per-atom active-mass floor, checked once per
    /// accepted outer iteration of the joint fit.
    ///
    /// The collapse statistic is each atom's MAXIMUM assignment mass over rows
    /// (see [`SAE_ATOM_ACTIVE_MASS_FLOOR`] for why max, not mean). A breach is
    /// answered with a gate-logit re-seed — once per atom per fit
    /// ([`SAE_ATOM_COLLAPSE_RESEED_BUDGET`]) — and recorded as a
    /// [`CollapseEvent`]; a breach after the budget is recorded once as
    /// terminal and otherwise left alone: at that point the collapse is the
    /// objective's (local) verdict at the current hyperparameters, and the
    /// keep-or-kill decision belongs to the evidence-gated structure search,
    /// not to an inner-loop heuristic. Observable events, never silent deaths,
    /// never fit errors.
    fn enforce_active_mass_guard(&mut self, iteration: usize) -> Result<(), String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        if n == 0 || k == 0 {
            return Ok(());
        }
        let mut max_mass = vec![0.0_f64; k];
        for row in 0..n {
            let a = self
                .assignment
                .try_assignments_row(row)
                .map_err(|e| format!("SaeManifoldTerm::enforce_active_mass_guard: {e}"))?;
            for atom in 0..k {
                if a[atom] > max_mass[atom] {
                    max_mass[atom] = a[atom];
                }
            }
        }
        for atom in 0..k {
            if max_mass[atom] >= SAE_ATOM_ACTIVE_MASS_FLOOR {
                continue;
            }
            let reseeds_used = self
                .collapse_events
                .iter()
                .filter(|e| e.atom == atom && e.action == CollapseAction::Reseeded)
                .count();
            if reseeds_used < SAE_ATOM_COLLAPSE_RESEED_BUDGET {
                self.reseed_collapsed_atom_logits(atom);
                self.collapse_events.push(CollapseEvent {
                    iteration,
                    atom,
                    max_active_mass: max_mass[atom],
                    floor: SAE_ATOM_ACTIVE_MASS_FLOOR,
                    action: CollapseAction::Reseeded,
                });
            } else {
                let already_terminal = self
                    .collapse_events
                    .iter()
                    .any(|e| e.atom == atom && e.action == CollapseAction::Terminal);
                if !already_terminal {
                    self.collapse_events.push(CollapseEvent {
                        iteration,
                        atom,
                        max_active_mass: max_mass[atom],
                        floor: SAE_ATOM_ACTIVE_MASS_FLOOR,
                        action: CollapseAction::Terminal,
                    });
                }
            }
        }
        Ok(())
    }

    /// Re-seed one collapsed atom's gate logits to the mode-appropriate
    /// neutral that restores material support — the data-fit term can then
    /// hold the atom active iff it carries signal. Latent coordinates are
    /// deliberately left untouched: gate-driven collapse kills the support,
    /// not the (still data-adjacent) coordinates, and a coordinate re-seed
    /// would discard exactly the warm state that makes the second chance
    /// cheap.
    fn reseed_collapsed_atom_logits(&mut self, atom: usize) {
        let n = self.n_obs();
        match self.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                // Tie the re-seeded atom with each row's current winner so it
                // re-enters the simplex at parity instead of inheriting a
                // saturated deficit.
                for row in 0..n {
                    let row_max = self
                        .assignment
                        .logits
                        .row(row)
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    self.assignment.logits[[row, atom]] =
                        if row_max.is_finite() { row_max } else { 0.0 };
                }
                canonicalize_softmax_logits(&mut self.assignment.logits);
            }
            AssignmentMode::IBPMap { .. } => {
                // σ(0/τ) = ½ — the gate's neutral point; the IBP prior π_k
                // still applies its geometric damping, as it should.
                for row in 0..n {
                    self.assignment.logits[[row, atom]] = 0.0;
                }
            }
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => {
                // One temperature unit above the hard gate threshold:
                // just-active, inside the smooth transition band.
                for row in 0..n {
                    self.assignment.logits[[row, atom]] = threshold + temperature;
                }
            }
        }
    }

    fn apply_newton_step_impl(
        &mut self,
        delta_ext_coord: ArrayView1<'_, f64>,
        delta_beta: ArrayView1<'_, f64>,
        step_size: f64,
        refresh_basis: bool,
    ) -> Result<(), String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: step_size must be finite and positive; got {step_size}"
            ));
        }
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        // #972 / #977 T1: when the most recent assembly built the factored
        // β-tier, `delta_beta` is a factored ΔC (length `factored_border_dim`)
        // that must be LIFTED through each active frame (`ΔB_k = ΔC_k U_kᵀ`)
        // before being applied to the p-wide decoder. Otherwise it is a plain
        // ΔB of length `beta_dim`. The expected length and the application path
        // both branch on `last_frames_active`.
        let expected_delta_len = if self.last_frames_active {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        if delta_beta.len() != expected_delta_len {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_beta length {} != expected {}",
                delta_beta.len(),
                expected_delta_len
            ));
        }

        // When last_row_layout is set (compact active-set mode — JumpReLU
        // gate or large-K IBP truncation), delta_ext_coord uses a
        // variable-stride layout where row i occupies
        // [compact_offset_i .. compact_offset_i + q_active_i].
        // We expand each row back to full-q before applying.
        if let Some(ref layout) = self.last_row_layout.clone() {
            let total_len: usize = (0..n).map(|row| layout.row_q_active(row)).sum();
            if delta_ext_coord.len() != total_len {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: compact delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    total_len
                ));
            }
            // Expand compact layout to full-q flat buffer.
            let mut full_delta = vec![0.0_f64; n * q];
            let mut compact_off = 0usize;
            for row in 0..n {
                let q_active = layout.row_q_active(row);
                // Collect compact row (handles both contiguous and strided views).
                let compact_row: Vec<f64> = delta_ext_coord
                    .slice(ndarray::s![compact_off..compact_off + q_active])
                    .iter()
                    .copied()
                    .collect();
                layout.expand_row(row, &compact_row, &mut full_delta[row * q..(row + 1) * q]);
                compact_off += q_active;
            }
            // Apply logits from expanded buffer, clamped to the #976 gate-scale
            // step cap (see SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS for the Armijo
            // consistency argument).
            let logit_step_cap =
                SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * self.assignment.mode.temperature();
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] += (step_size
                        * full_delta[row_base + atom_idx])
                        .clamp(-logit_step_cap, logit_step_cap);
                }
            }
            // Apply coords from expanded buffer.
            let coord_offsets = self.assignment.coord_offsets();
            for atom_idx in 0..k_atoms {
                let d = self.assignment.coords[atom_idx].latent_dim();
                let mut delta_coord = Array1::<f64>::zeros(n * d);
                for row in 0..n {
                    let row_base = row * q + coord_offsets[atom_idx];
                    for axis in 0..d {
                        delta_coord[row * d + axis] = step_size * full_delta[row_base + axis];
                    }
                }
                self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
                if refresh_basis {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    self.atoms[atom_idx].refresh_basis(coords.view())?;
                }
            }
        } else {
            // Dense layout: uniform q per row.
            if delta_ext_coord.len() != n * q {
                return Err(format!(
                    "SaeManifoldTerm::apply_newton_step: delta_ext_coord length {} != expected {}",
                    delta_ext_coord.len(),
                    n * q
                ));
            }
            let coord_offsets = self.assignment.coord_offsets();
            // #976 gate-scale step cap, as in the compact branch above.
            let logit_step_cap =
                SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * self.assignment.mode.temperature();
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] += (step_size
                        * delta_ext_coord[row_base + atom_idx])
                        .clamp(-logit_step_cap, logit_step_cap);
                }
            }
            for atom_idx in 0..k_atoms {
                let d = self.assignment.coords[atom_idx].latent_dim();
                let mut delta_coord = Array1::<f64>::zeros(n * d);
                for row in 0..n {
                    let row_base = row * q + coord_offsets[atom_idx];
                    for axis in 0..d {
                        delta_coord[row * d + axis] = step_size * delta_ext_coord[row_base + axis];
                    }
                }
                self.assignment.coords[atom_idx].retract_flat_delta(delta_coord.view());
                if refresh_basis {
                    let coords = self.assignment.coords[atom_idx].as_matrix();
                    self.atoms[atom_idx].refresh_basis(coords.view())?;
                }
            }
        }
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut self.assignment.logits);
        }

        let mut beta = self.flatten_beta();
        if self.last_frames_active {
            // Factored ΔC → lift to a p-wide ΔB and add `step·ΔB`. For atom `k`,
            // basis row `m`, output channel `i`:
            //   ΔB_k[m,i] = Σ_j ΔC[off_C[k] + m·r_k + j] · U_k[i,j].
            // Un-framed atoms (`U_k = I_p`, `r_k = p`) lift by identity, so a
            // mixed dictionary is handled uniformly. The decoder is then
            // refreshed below via `set_flat_beta` (the authoritative `B_k` is the
            // p-wide flatten; the active frames are re-synced from the decoder by
            // the polar refresh in the joint-fit driver).
            let delta_b = FrameProjection::new(self).lift_border_vec(delta_beta);
            for idx in 0..beta.len() {
                beta[idx] += step_size * delta_b[idx];
            }
        } else {
            for idx in 0..beta.len() {
                beta[idx] += step_size * delta_beta[idx];
            }
        }
        self.set_flat_beta(beta.view())
    }

    fn solve_fixed_decoder_row_step(
        h: ArrayView2<'_, f64>,
        g: ArrayView1<'_, f64>,
        base_ridge: f64,
    ) -> Result<Array1<f64>, String> {
        let d = h.nrows();
        if h.ncols() != d || g.len() != d {
            return Err(format!(
                "SaeManifoldTerm::solve_fixed_decoder_row_step: shape mismatch H={:?}, g={}",
                h.dim(),
                g.len()
            ));
        }
        if d == 0 {
            return Ok(Array1::<f64>::zeros(0));
        }
        let mut ridge = base_ridge.max(SAE_MANIFOLD_ROW_RIDGE_FLOOR);
        let mut last_err = String::new();
        for _ in 0..SAE_MANIFOLD_ROW_RIDGE_MAX_ATTEMPTS {
            let mut a = h.to_owned();
            for axis in 0..d {
                a[[axis, axis]] += ridge;
            }
            match sae_cholesky_solve_neg_gradient(a.view(), g) {
                Ok(delta) => return Ok(delta),
                Err(err) => {
                    last_err = err;
                    ridge *= SAE_MANIFOLD_ROW_RIDGE_GROWTH;
                }
            }
        }
        Err(format!(
            "SaeManifoldTerm::solve_fixed_decoder_row_step: row Hessian did not factor after LM escalation; last error: {last_err}"
        ))
    }

    fn fixed_decoder_step_from_rows(
        sys: &ArrowSchurSystem,
        ridge_ext_coord: f64,
    ) -> Result<Array1<f64>, String> {
        let total = sys.row_offsets[sys.rows.len()];
        let mut delta = Array1::<f64>::zeros(total);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_delta =
                Self::solve_fixed_decoder_row_step(row.htt.view(), row.gt.view(), ridge_ext_coord)?;
            let start = sys.row_offsets[row_idx];
            let end = sys.row_offsets[row_idx + 1];
            if row_delta.len() != end - start {
                return Err(format!(
                    "SaeManifoldTerm::fixed_decoder_step_from_rows: row {row_idx} delta len {} != row span {}",
                    row_delta.len(),
                    end - start
                ));
            }
            delta.slice_mut(s![start..end]).assign(&row_delta);
        }
        Ok(delta)
    }

    /// Row visitation order for the discovery/seeding pass, drawn from the
    /// per-row Fisher-mass enrichment measure (#980, role (c)).
    ///
    /// Builds [`RowMeasure::from_metric`](crate::inference::row_measure::RowMeasure::from_metric)
    /// from the term's installed [`RowMetric`] (Euclidean fallback when none is
    /// installed), draws a length-`n` systematic-resampling
    /// [`enrichment_order`](crate::inference::row_measure::RowMeasure::enrichment_order),
    /// and reduces it to a first-seen unique permutation. Behaviorally-live rows
    /// (high Fisher mass) appear earliest; any row the measure never named is
    /// appended in index order so **every** row is still visited exactly once.
    ///
    /// Under a Euclidean / no-harvest metric the measure is exactly uniform, the
    /// systematic-resampling draw is an even round-robin, and the first-seen
    /// reduction is the plain `0..n` index order — bit-for-bit today's behavior.
    ///
    /// Pure attention: the order is consumed only to decide *which row is looked
    /// at first*; each visited row runs the identical unmodified per-row
    /// objective, so this touches no loss / criterion / penalty.
    fn enrichment_visit_order(&self) -> Vec<usize> {
        let n = self.n_obs();
        // No installed metric ⇒ the measure is exactly uniform and the
        // systematic draw reduces to plain index order (documented below), so
        // skip building the Euclidean metric object entirely — this runs in
        // the seeding hot path, per seed-candidate evaluation.
        if self.row_metric.is_none() {
            return (0..n).collect();
        }
        let metric = match self.diagnostic_metric() {
            Ok(m) => m,
            // A metric build failure cannot occur for the term's own validated
            // shape, but degrade to the plain index sweep rather than propagate:
            // the order is attention-only and must never gate the seed.
            Err(_) => return (0..n).collect(),
        };
        let measure = crate::inference::row_measure::RowMeasure::from_metric(&metric);
        // Seed the deterministic systematic-resampling draw from the row count so
        // the ordering is reproducible across runs (no clock randomness).
        let drawn = measure.enrichment_order(n, n as u64);
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        for row in drawn {
            if row < n && !seen[row] {
                seen[row] = true;
                order.push(row);
            }
        }
        // Append any row the enrichment draw never named so every row is seeded.
        for (row, &was_seen) in seen.iter().enumerate() {
            if !was_seen {
                order.push(row);
            }
        }
        order
    }

    /// Globally seed every atom's per-row latent coordinate by projecting each
    /// target row onto that atom's **frozen** decoder image manifold.
    ///
    /// For a fixed decoder the exact out-of-sample encoding of row `i` against
    /// atom `k` is the projection
    /// `t*_{ik} = argmin_t ‖x_i − Φ_k(t)·B_k‖²`. That objective is non-convex
    /// on a compact latent (a trigonometric polynomial for periodic / torus
    /// atoms, a chart function on the sphere), so the cold PCA-`atan2` seed plus
    /// a handful of Newton steps frequently converges into the wrong basin and
    /// mis-routes the row — the root cause of the negative-`R²`, near-uniform
    /// assignment OOS failures. We evaluate each atom's decoder once on a dense
    /// manifold-spanning grid (provided by the atom basis kind), take the per-row
    /// global argmin as the coordinate seed, refresh the atom basis there, and
    /// let the subsequent Newton refinement polish to full precision from inside
    /// the correct basin. Because the residual-based softmax logit seed reads the
    /// freshly decoded rows, routing then follows the true per-atom projection
    /// error rather than the cold-seed error.
    ///
    /// Atoms whose basis kind exposes no projection seed grid
    /// (unbounded / basis-linear latents) are left at their incoming seed. The
    /// decoder, assignment logits, smoothness penalties and ρ are all untouched;
    /// only the latent coordinates and the basis caches that depend on them move.
    pub fn seed_coords_by_decoder_projection(
        &mut self,
        target: ArrayView2<'_, f64>,
        resolution: usize,
    ) -> Result<(), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "SaeManifoldTerm::seed_coords_by_decoder_projection: target shape {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        // ENRICHMENT (#980, role (c)): the order in which this discovery/seeding
        // pass *visits* rows is drawn from the per-row Fisher-mass sampling
        // measure when an output-Fisher harvest is present, so behaviorally-live
        // rows get attention FIRST. This is attention-only: every visited row
        // runs the identical, unmodified per-row argmin projection objective
        // below — the measure reweights *which row is looked at first*, never the
        // loss. Under a Euclidean / no-harvest metric the measure is exactly
        // uniform, so the order degrades to the plain `0..n` index sweep and the
        // result is bit-for-bit today's behavior. Because each row's seed is
        // computed independently and written exactly once, the visitation order
        // cannot change any seed value — confirming the attention-only invariant.
        let visit_order = self.enrichment_visit_order();
        for atom_idx in 0..self.k_atoms() {
            let d = self.atoms[atom_idx].latent_dim;
            let Some(grid) = self.atoms[atom_idx]
                .basis_kind
                .projection_seed_grid(d, resolution)
            else {
                continue;
            };
            let Some(evaluator) = self.atoms[atom_idx].basis_evaluator.clone() else {
                continue;
            };
            if grid.ncols() != d {
                return Err(format!(
                    "SaeManifoldTerm::seed_coords_by_decoder_projection: atom {atom_idx} grid has {} columns but latent_dim is {d}",
                    grid.ncols()
                ));
            }
            let g = grid.nrows();
            if g == 0 {
                continue;
            }
            // Decode the whole grid once: `decoded = Φ(grid) · B_k`  (g × p).
            let (phi_grid, _jet) = evaluator.evaluate(grid.view())?;
            if phi_grid.ncols() != self.atoms[atom_idx].basis_size() {
                return Err(format!(
                    "SaeManifoldTerm::seed_coords_by_decoder_projection: atom {atom_idx} grid Φ has {} columns but decoder expects {}",
                    phi_grid.ncols(),
                    self.atoms[atom_idx].basis_size()
                ));
            }
            let decoded = phi_grid.dot(&self.atoms[atom_idx].decoder_coefficients);
            // Per-row global argmin of ‖x_i − decoded_g‖² over the grid. Rows are
            // *visited* in the enrichment order (live rows first); the projection
            // objective for each row is unchanged, and each row is seeded exactly
            // once, so the order is pure attention and cannot move any seed.
            let mut seeded = Array2::<f64>::zeros((n, d));
            for &row in &visit_order {
                let mut best_idx = 0usize;
                let mut best_err = f64::INFINITY;
                for grid_idx in 0..g {
                    let mut err = 0.0_f64;
                    for col in 0..p {
                        let diff = target[[row, col]] - decoded[[grid_idx, col]];
                        err += diff * diff;
                    }
                    if err < best_err {
                        best_err = err;
                        best_idx = grid_idx;
                    }
                }
                for axis in 0..d {
                    seeded[[row, axis]] = grid[[best_idx, axis]];
                }
            }
            let flat = Array1::from_iter(seeded.iter().copied());
            self.assignment.coords[atom_idx].set_flat(flat.view());
            let coords = self.assignment.coords[atom_idx].as_matrix();
            self.atoms[atom_idx].refresh_basis(coords.view())?;
        }
        Ok(())
    }

    pub fn run_fixed_decoder_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_fixed_decoder_arrow_schur: step_size must be finite and positive; got {step_size}"
            ));
        }
        if max_iter < 1 {
            return Err(
                "SaeManifoldTerm::run_fixed_decoder_arrow_schur: max_iter must be positive".into(),
            );
        }
        let beta_zero = Array1::<f64>::zeros(self.beta_dim());
        let mut last_loss = self.loss(target, rho)?;
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            let pre_step_loss = self.loss(target, rho)?;
            let sys = self
                .assemble_arrow_schur(target, rho, analytic_penalties)
                .map_err(|err| format!("SaeManifoldTerm::run_fixed_decoder_arrow_schur: {err}"))?;
            let pre_step_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            let delta_ext_coord = Self::fixed_decoder_step_from_rows(&sys, ridge_ext_coord)?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                beta_zero.view(),
            );
            let grad_norm_sq: f64 = sys
                .rows
                .iter()
                .flat_map(|row| row.gt.iter())
                .map(|&v| v * v)
                .sum();
            let step_norm_sq: f64 = delta_ext_coord.iter().map(|&v| v * v).sum();
            let directional_decrease_floor = SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR
                * grad_norm_sq.sqrt()
                * step_norm_sq.sqrt();
            let snapshot = self.snapshot_mutable_state();
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor)
            {
                self.restore_mutable_state(&snapshot);
                last_loss = pre_step_loss;
                break;
            }

            let mut trial_step_size = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                if halving > 0 {
                    self.restore_mutable_state(&snapshot);
                }
                let trial_result = self
                    .apply_newton_step(delta_ext_coord.view(), beta_zero.view(), trial_step_size)
                    .and_then(|()| {
                        self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                    });
                if let Ok(post_step_total) = trial_result {
                    let armijo_bound = pre_step_total
                        - SAE_MANIFOLD_ARMIJO_C1 * trial_step_size * directional_decrease;
                    if post_step_total.is_finite() && post_step_total <= armijo_bound {
                        accepted_loss = Some(self.loss(target, rho)?);
                        break;
                    }
                }
                trial_step_size *= 0.5;
            }
            match accepted_loss {
                Some(loss) => last_loss = loss,
                None => {
                    self.restore_mutable_state(&snapshot);
                    last_loss = pre_step_loss;
                    break;
                }
            }
        }
        Ok(last_loss)
    }

    pub fn run_joint_fit_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur: step_size must be finite and positive; got {step_size}"
            ));
        }
        self.refresh_basis_from_current_coords()
            .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
        // #972 / #977 T1 — magic-by-default decoder-frame activation. Before the
        // outer loop, auto-derive and install the low-rank Grassmann frames
        // (each atom independently, only when the factorization materially
        // shrinks its border and leaves a positive Grassmann dimension). No
        // flag: small-`p` / full-rank atoms stay on the bit-for-bit full-`B`
        // path, so the small-model fits are unchanged; large-ambient-`p`,
        // low-decoder-rank atoms collapse their border `M_k·p → M_k·r_k` and the
        // joint solve runs in the factored coordinate space.
        self.ensure_decoder_frames_active_for_current_decoder()
            .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
        // #976 Layer-1 guard ledger is per joint fit: each inner solve gets a
        // fresh re-seed budget and reports only its own breaches.
        self.collapse_events.clear();
        // #1003 — run the active-mass guard at ENTRY (iteration 0), before the
        // pre-fit identifiability audit. A cold seed can hand the fit an atom
        // whose gates are vacuous on every row (the outer seed cascade sweeps
        // ρ states the seeding heuristics never saw); without this call the
        // audit below reports that atom as a fatal rank-0 weighted design and
        // the whole seed dies, even though the #976 guard exists precisely to
        // answer a support collapse with one observable re-seed. The guard's
        // shared per-fit budget applies unchanged: an atom re-seeded here that
        // collapses again mid-fit goes Terminal, which is the structure
        // search's signal, not the inner loop's. Genuinely degenerate bases
        // (zero rows regardless of gates) still fail the audit — correctly.
        self.enforce_active_mass_guard(0)?;
        // ── Pre-fit decoder identifiability audit ──────────────────────────
        //
        // Each decoder atom `k` contributes `η_i += a_ik · Φ_k(t_ik) · B_k`,
        // with `B_k ∈ ℝ^{M_k × p}`. The decoder Hessian for atom `k` is
        // `H_data = G_k ⊗ I_p` where `G_k = (diag(a_·k)·Φ_k)ᵀ (diag(a_·k)·Φ_k)`
        // (the diagonal `(atom_k, atom_k)` block of the sparse data Gram `G`
        // assembled in `assemble_arrow_schur`); the
        // `p` output channels share the identical `M_k × M_k` Gram, so decoder
        // identifiability is fully determined by the per-atom `(n, M_k)` design
        // `D_k = diag(a_·k)·Φ_k`. The `p`-fold output replication carries no
        // extra structural information and must NOT be materialised — doing so
        // (the former `(n·p, M_k·p)` channel-block route through the
        // cross-block flat audit) broadcast an `(n·p)`-row Jacobian into the
        // `n`-row placeholder design and panicked inside ndarray.
        //
        // We therefore run the rank check directly on each `D_k`. A
        // rank-deficient `D_k` means atom `k`'s decoder block Hessian is
        // singular (its Cholesky will fail or produce garbage steps), which is
        // surfaced as an immediate fatal error. Near-rank-deficient columns are
        // logged as INFO so callers can see which atoms are weakly identified.
        //
        // The check is performed chunk-aware through the per-atom Gram
        // accumulator: the full-batch path is the single-chunk special case.
        // `D_k`'s singular spectrum equals `√spec(G_k)` with
        // `G_k = D_kᵀ D_k`, so accumulating `G_k` over the whole design and
        // taking its eigenvalues reproduces the former pivoted-QR rank exactly
        // while never retaining an `(N × M_k)` design.
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            self.accumulate_decoder_gram(&mut grams);
            self.finalize_decoder_identifiability_audit(&grams, self.n_obs())?;
        }
        for outer_iteration in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ρ (including the ARD precisions) is owned by the outer engine
            // (`SaeManifoldOuterObjective`) and held FIXED across this inner
            // (t, β) Newton solve. The inner loop solves the joint manifold +
            // decoder system at the engine's current ρ; the engine alone
            // moves ρ by minimising the true REML criterion (see
            // `SaeManifoldTerm::reml_criterion`). The former in-loop
            // `update_ard_reml` rule (α = n / ‖t‖²) dropped the logdet /
            // effective-dof term and collapsed α on near-degenerate axes; it
            // has been removed in favour of the criterion-driven update.
            let sys = self
                .assemble_arrow_schur(target, rho, analytic_penalties)
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let plan = self
                .streaming_plan()
                .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
                .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let solve_options = plan.solve_options_for_border_dim(sys.k);
            // Inner Newton step with principled LM-style ridge escalation. The
            // PCA-seed starting state on a small batch (e.g. `predict` on a
            // strict subset of the training set) can produce a per-row
            // `H_tt + ridge_t·I` whose Cholesky has a negative pivot, or a
            // near-singular Schur complement, at the caller's nominal ridges.
            // Rather than abort, mirror the proximal-correction outer wrapper
            // and grow both ridges geometrically until the linear system
            // factors. This is the same LM-trust-region damping the convergent
            // proximal_correction path applies; we route it through the same
            // factor-failure error variants so legitimate, non-recoverable
            // errors (PCG divergence with no factor failure, adaptive-step
            // exhaustion, …) still surface immediately.
            let (delta_ext_coord, delta_beta, _diag) =
                solve_with_lm_escalation_inner(&sys, ridge_ext_coord, ridge_beta, &solve_options)
                    .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            let directional_decrease = sae_manifold_newton_directional_decrease(
                &sys,
                delta_ext_coord.view(),
                delta_beta.view(),
            );
            // Relative-scale floor on the directional decrease. When the
            // gradient is nearly orthogonal to the Newton step (ill-conditioned
            // near-convergence), `directional_decrease` collapses to O(machine
            // epsilon · ‖g‖ · ‖Δ‖). At that scale the Armijo bound
            // `pre_step_total − c1·step·directional_decrease` is numerically
            // indistinguishable from `pre_step_total`, so the line search would
            // "accept" on rounding noise. Treat that as converged and stop. The
            // norms are the natural scale of the inner product; the relative
            // constant keeps the reduction term distinguishable from rounding at
            // full step size given SAE_MANIFOLD_ARMIJO_C1 = 1e-4.
            let mut grad_norm_sq = 0.0;
            for (row_idx, row) in sys.rows.iter().enumerate() {
                let di = sys.row_dims[row_idx];
                for axis in 0..di {
                    grad_norm_sq += row.gt[axis] * row.gt[axis];
                }
            }
            for idx in 0..sys.k {
                grad_norm_sq += sys.gb[idx] * sys.gb[idx];
            }
            let mut step_norm_sq = 0.0;
            for &v in delta_ext_coord.iter() {
                step_norm_sq += v * v;
            }
            for &v in delta_beta.iter() {
                step_norm_sq += v * v;
            }
            let directional_decrease_floor = SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR
                * grad_norm_sq.sqrt()
                * step_norm_sq.sqrt();
            // Capture the exact state whose assembled gradient/Hessian produced
            // `sys`, then evaluate the Armijo baseline from that same state.
            // Assembly installs compact active-set layout in `last_row_layout`;
            // computing the baseline after that mutation prevents comparing a
            // trial at one represented state against a bound from another.
            //
            // Each rejected trial restores from this snapshot in place; the
            // static atom metadata, smoothness penalties and basis-evaluator
            // `Arc`s are never re-cloned. This replaces the per-halving full
            // `self.clone()`, whose dominant cost was copying the
            // `O(N·M·d)` `basis_jacobian` and `O(N·M)` `basis_values` on every
            // backtrack.
            let snapshot = self.snapshot_mutable_state();
            let pre_step_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            if !pre_step_total.is_finite() {
                // Pre-step state is unperturbed here; restore is a no-op but
                // keeps the invariant explicit.
                self.restore_mutable_state(&snapshot);
                break;
            }
            // A non-descent Newton direction (gᵀΔ ≤ 0 or below the rounding
            // floor) is only a STOPPING criterion when the iterate is actually
            // stationary: the floor exists for benign ill-conditioned
            // near-convergence, where `gᵀΔ` collapses to rounding noise while
            // ‖g‖ is already tiny. In degenerate multi-atom geometry (gate
            // tug-of-war, rank-deficient duchon columns) the LM solve factors
            // with a near-zero pivot, the step is dominated by that near-null
            // direction, and `gᵀΔ/(‖g‖·‖Δ‖)` collapses while ‖g‖ is HUGE —
            // breaking there silently froze the iterate and let the
            // `reml_criterion` refine loop re-measure the same point until its
            // budget died (the constant-‖g‖=1e12 signature). Gate the break on
            // genuine KKT stationarity — the SAME iterate-scaled tolerance
            // `reml_criterion` uses — and otherwise fall through to the
            // proximal-correction ridge escalation below: heavier LM damping
            // bends the step toward steepest descent, which is always a
            // descent direction for a consistent gradient.
            let descent_direction_ok = directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor;
            if !descent_direction_ok {
                let grad_tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * self.inner_iterate_scale();
                if grad_norm_sq.sqrt() <= grad_tolerance {
                    self.restore_mutable_state(&snapshot);
                    break;
                }
            }

            let mut trial_step_size = step_size;
            let mut accepted = false;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                if !descent_direction_ok {
                    // No Armijo bound is meaningful along a non-descent
                    // direction; route straight to the proximal correction.
                    break;
                }
                if halving > 0 {
                    // Reset to the pre-step state before re-applying at the
                    // halved step. The first trial starts from the pre-step
                    // state already, so the restore is only needed after a
                    // rejected trial mutated `self`.
                    self.restore_mutable_state(&snapshot);
                }
                let trial_result = self
                    .apply_newton_step(delta_ext_coord.view(), delta_beta.view(), trial_step_size)
                    .and_then(|()| {
                        self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                    });
                if let Ok(post_step_total) = trial_result {
                    let armijo_bound = pre_step_total
                        - SAE_MANIFOLD_ARMIJO_C1 * trial_step_size * directional_decrease;
                    if post_step_total.is_finite() && post_step_total <= armijo_bound {
                        accepted = true;
                        break;
                    }
                }
                trial_step_size *= 0.5;
            }
            if !accepted {
                self.restore_mutable_state(&snapshot);
                let correction = ArrowProximalCorrectionOptions {
                    initial_ridge: ridge_ext_coord
                        .max(ridge_beta)
                        .max(SAE_MANIFOLD_ROW_RIDGE_FLOOR),
                    armijo_c1: SAE_MANIFOLD_ARMIJO_C1,
                    ..ArrowProximalCorrectionOptions::default()
                };
                let accepted_step = match solve_arrow_newton_step_with_proximal_correction(
                    &sys,
                    ridge_ext_coord,
                    ridge_beta,
                    pre_step_total,
                    // `sys.k` is the actual border width — factored
                    // (`factored_border_dim`) when frames are active, else
                    // `beta_dim` — so the direct/PCG mode threshold keys on the
                    // dimension the solve actually runs at.
                    &solve_options,
                    &correction,
                    |trial_delta_t, trial_delta_beta| {
                        self.restore_mutable_state(&snapshot);
                        self.apply_newton_step(trial_delta_t, trial_delta_beta, 1.0)
                            .and_then(|()| {
                                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)
                            })
                            .unwrap_or(f64::INFINITY)
                    },
                ) {
                    Ok(step) => step,
                    Err(err) => {
                        log::debug!(
                            "run_joint_fit_arrow_schur: proximal correction errored at \
                             iteration {outer_iteration} (gᵀΔ={directional_decrease:.3e}, \
                             floor={directional_decrease_floor:.3e}, \
                             ‖g‖={:.3e}): {err}",
                            grad_norm_sq.sqrt()
                        );
                        self.restore_mutable_state(&snapshot);
                        break;
                    }
                };
                if !(accepted_step.trial_objective_value.is_finite()
                    && accepted_step.trial_objective_value < pre_step_total)
                {
                    log::debug!(
                        "run_joint_fit_arrow_schur: proximal correction made no decrease at \
                         iteration {outer_iteration} (trial={:.9e}, pre={pre_step_total:.9e}, \
                         ‖g‖={:.3e})",
                        accepted_step.trial_objective_value,
                        grad_norm_sq.sqrt()
                    );
                    self.restore_mutable_state(&snapshot);
                    break;
                }
            }
            // Affine gauge canonicalization is a representation change, but the
            // decoder smoothness term is part of the optimized objective. Keep the
            // canonicalized state only when the same scalar used by the line search
            // does not increase; otherwise REML would inspect an off-contract
            // post-accept state whose gradient was never accepted.
            let accepted_snapshot = self.snapshot_mutable_state();
            let accepted_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            self.canonicalize_affine_gauge_after_accept()?;
            let canonical_total =
                self.penalized_objective_total(target, rho, analytic_penalties, 1.0)?;
            if !(canonical_total.is_finite() && canonical_total <= accepted_total) {
                self.restore_mutable_state(&accepted_snapshot);
            }
            // #976 Layer-1 guard 3: after an accepted step (Armijo or proximal
            // — the rejection paths `break` above), check every atom's support
            // and answer breaches with a bounded re-seed or a terminal
            // CollapseEvent. Runs post-acceptance so it never perturbs a
            // line-search trial, and any re-seed is simply the next
            // iteration's starting state.
            self.enforce_active_mass_guard(outer_iteration)?;
            // #972 / #977 T1 — U-block of the alternating block-coordinate ascent.
            // After the decoder `B` has been updated by the accepted (t, ΔC) step
            // (lifted through the OLD frames in `apply_newton_step`), re-polar each
            // ACTIVE atom's frame from the refreshed data evidence and re-project
            // the decoder onto it, so the next assembly's C-block solve runs in an
            // up-to-date frame. The refresh is a closed-form `O(p r²)` thin SVD per
            // atom run OUTSIDE the border; the C-coordinates are held fixed during
            // it (the block-coordinate split). Skipped entirely when no frame is
            // active (the full-`B` path never touches this). One refresh per
            // accepted outer iteration is a sensible cadence (the issue's
            // streaming-polar fixed point).
            if self.frames_active() {
                self.refresh_active_frames_from_data(target)
                    .map_err(|err| format!("SaeManifoldTerm::run_joint_fit_arrow_schur: {err}"))?;
            }
        }
        // ρ is owned by the outer engine and unchanged here; just return the
        // converged inner loss at the fixed ρ.
        self.loss(target, rho)
    }

    /// Allocate one zero `(M_k × M_k)` Gram accumulator per atom for the
    /// chunk-aware decoder identifiability audit.
    fn empty_decoder_gram_accumulator(&self) -> Vec<Array2<f64>> {
        self.atoms
            .iter()
            .map(|atom| {
                let m = atom.basis_size();
                Array2::<f64>::zeros((m, m))
            })
            .collect()
    }

    /// Accumulate this term's (chunk's) contribution to the per-atom weighted
    /// design Gram `G_k += D_kᵀ D_k`, with `D_k = diag(a_·k)·Φ_k`.
    ///
    /// `grams[k]` must be `(M_k × M_k)`. Streaming callers invoke this once per
    /// chunk against the freshly materialized chunk term; the full-batch path
    /// invokes it once against `self`. The Gram is symmetric and channel-free
    /// (the `p`-fold output replication is carried by the `⊗ I_p` Kronecker
    /// structure, so it adds no rank information), so accumulating `Φ` weighted
    /// by the per-row assignment exactly reproduces the data-fit decoder block
    /// curvature `G_k` that `assemble_arrow_schur` installs.
    fn accumulate_decoder_gram(&self, grams: &mut [Array2<f64>]) {
        let n = self.n_obs();
        let assignments = self.assignment.assignments();
        // Each atom's Gram `G_k = Φ_kᵀ diag(a_k²) Φ_k` is an independent
        // weighted cross-product over the N rows — the canonical `xt_diag_x`
        // shape, and independent across the per-atom axis. This feeds a
        // tolerance-based identifiability RANK decision (not a fitted quantity),
        // so the device path's accumulation order is admissible.
        //
        // Spread the atoms across EVERY device via `gpu::pool::scatter_batched`;
        // each device tile computes its atoms' Grams through the size-gated
        // `try_fast_xt_diag_x` shim. Atoms whose device path declines (no
        // runtime, sub-threshold size, or backend miss) drop to the exact CPU
        // rank-1 accumulation, so the result matches the all-CPU path.
        let weights: Vec<Array1<f64>> = (0..self.atoms.len())
            .map(|atom_idx| {
                let col = assignments.column(atom_idx);
                col.mapv(|a| a * a)
            })
            .collect();

        // CPU per-atom contribution, used for fallback and as the whole path
        // when no GPU runtime is present.
        let cpu_one = |atom_idx: usize, gram: &mut Array2<f64>| {
            let atom = &self.atoms[atom_idx];
            let m = atom.basis_size();
            let assign_col = assignments.column(atom_idx);
            let mut weighted = vec![0.0_f64; m];
            for row in 0..n {
                let a_k = assign_col[row];
                if a_k == 0.0 {
                    continue;
                }
                for col in 0..m {
                    weighted[col] = a_k * atom.basis_values[[row, col]];
                }
                for i in 0..m {
                    let wi = weighted[i];
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..m {
                        gram[[i, j]] += wi * weighted[j];
                    }
                }
            }
        };

        let rt = crate::gpu::runtime::GpuRuntime::global();
        match rt {
            None => {
                for atom_idx in 0..self.atoms.len() {
                    if self.atoms[atom_idx].basis_size() == 0 {
                        continue;
                    }
                    cpu_one(atom_idx, &mut grams[atom_idx]);
                }
            }
            Some(rt) => {
                // Device tiles produce each owned atom's Gram into a side channel
                // keyed by atom index; splice them back into `grams` (with `+=`
                // accumulation) after the scatter. Atoms the device declines are
                // marked so the CPU fallback runs for exactly those.
                let mut items: Vec<usize> = (0..self.atoms.len())
                    .filter(|&i| self.atoms[i].basis_size() > 0)
                    .collect();
                let device_grams: std::sync::Mutex<Vec<(usize, Array2<f64>)>> =
                    std::sync::Mutex::new(Vec::with_capacity(items.len()));
                let declined: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
                let atoms_ref = &self.atoms;
                let weights_ref = &weights;
                let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_ordinal, slice| {
                    for &atom_idx in slice.iter() {
                        let phi = atoms_ref[atom_idx].basis_values.view();
                        let w = weights_ref[atom_idx].view();
                        match crate::gpu::linalg::try_fast_xt_diag_x(phi, w) {
                            Some(g) => device_grams
                                .lock()
                                .expect("device_grams mutex poisoned")
                                .push((atom_idx, g)),
                            None => declined
                                .lock()
                                .expect("declined mutex poisoned")
                                .push(atom_idx),
                        }
                    }
                    Some(())
                });
                match ok {
                    Some(()) => {
                        for (atom_idx, g) in device_grams
                            .into_inner()
                            .expect("device_grams mutex poisoned")
                        {
                            grams[atom_idx] += &g;
                        }
                        for atom_idx in declined.into_inner().expect("declined mutex poisoned") {
                            cpu_one(atom_idx, &mut grams[atom_idx]);
                        }
                    }
                    None => {
                        for atom_idx in 0..self.atoms.len() {
                            if self.atoms[atom_idx].basis_size() == 0 {
                                continue;
                            }
                            cpu_one(atom_idx, &mut grams[atom_idx]);
                        }
                    }
                }
            }
        }
    }

    /// Decide rank-deficiency of each accumulated decoder Gram and surface the
    /// same fatal / INFO contract as the former pivoted-QR audit.
    fn finalize_decoder_identifiability_audit(
        &self,
        grams: &[Array2<f64>],
        n_total: usize,
    ) -> Result<(), String> {
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            if m == 0 {
                continue;
            }
            let rank =
                crate::solver::identifiability_audit::rank_of_gram(&grams[atom_idx], n_total)
                    .map_err(|e| {
                        format!(
                            "SaeManifoldTerm: pre-fit decoder audit (atom '{}'): \
                         Gram eigendecomposition failed: {e}",
                            atom.name,
                        )
                    })?;
            if rank < m {
                let dropped = m - rank;
                if rank == 0 {
                    return Err(format!(
                        "SaeManifoldTerm: pre-fit identifiability audit: decoder atom '{}' has \
                         rank-0 weighted design (n={n_total}, M_k={m}); all assignment weights \
                         vanish or the basis is degenerate, so the Arrow-Schur Newton system for \
                         this block is singular",
                        atom.name,
                    ));
                }
                log::info!(
                    "[SAE-AUDIT] decoder atom '{}' weighted design is rank-deficient \
                     (rank={rank}/{m}, {dropped} weakly-identified column(s), n={n_total}); the \
                     Arrow-Schur ridge will regularise the deficient directions",
                    atom.name,
                );
            }
        }
        Ok(())
    }

    /// Materialize a row-chunk `[start, end)` of this term as a standalone
    /// `SaeManifoldTerm`, recomputing `(basis_values, basis_jacobian)` on demand
    /// from the persisted decoder + atom geometry and the caller-supplied
    /// per-chunk `(logits, coords)`.
    ///
    /// The streaming joint fit NEVER persists the `(N × M)` basis or `(N × K)`
    /// logit buffers. Instead, for each chunk the caller re-seeds the chunk's
    /// latent state (the SAE PCA seed restricted to the chunk's `Z` rows for the
    /// coordinates, and the chunk's gating logits) and this constructor rebuilds
    /// a small `n_chunk`-row term whose atoms share the global decoder
    /// coefficients (`B_k`), smoothness penalties, and basis evaluators with
    /// `self`. Each atom's basis is re-evaluated at the chunk coordinates via
    /// its `basis_evaluator`, so the chunk term is exactly the restriction of
    /// the global model to those rows.
    ///
    /// Errors if any atom lacks a basis evaluator (a streaming fit must be able
    /// to re-evaluate `Φ(t)` at the per-chunk coordinates) or if the supplied
    /// shapes disagree with the term's atom layout.
    pub fn materialize_chunk(
        &self,
        chunk_logits: Array2<f64>,
        chunk_coords: Vec<Array2<f64>>,
    ) -> Result<SaeManifoldTerm, String> {
        let k_atoms = self.k_atoms();
        if chunk_logits.ncols() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_logits has {} cols but K={k_atoms}",
                chunk_logits.ncols()
            ));
        }
        if chunk_coords.len() != k_atoms {
            return Err(format!(
                "SaeManifoldTerm::materialize_chunk: chunk_coords has {} atoms but K={k_atoms}",
                chunk_coords.len()
            ));
        }
        let n_chunk = chunk_logits.nrows();
        let mut atoms = Vec::with_capacity(k_atoms);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let coords = &chunk_coords[atom_idx];
            if coords.nrows() != n_chunk || coords.ncols() != atom.latent_dim {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom {atom_idx} coords shape {:?} != ({n_chunk}, {})",
                    coords.dim(),
                    atom.latent_dim
                ));
            }
            let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' has no basis evaluator; a \
                     streaming fit must re-evaluate Φ(t) at each chunk's coordinates",
                    atom.name
                )
            })?;
            let (phi, jet) = evaluator.evaluate(coords.view())?;
            let m = atom.basis_size();
            if phi.dim() != (n_chunk, m) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned Φ {:?}, expected ({n_chunk}, {m})",
                    atom.name,
                    phi.dim()
                ));
            }
            if jet.dim() != (n_chunk, m, atom.latent_dim) {
                return Err(format!(
                    "SaeManifoldTerm::materialize_chunk: atom '{}' evaluator returned jet {:?}, expected ({n_chunk}, {m}, {})",
                    atom.name,
                    jet.dim(),
                    atom.latent_dim
                ));
            }
            // Seed the chunk atom from the *raw* roughness Gram (not the
            // already arc-length-reweighted `smooth_penalty`), so its
            // constructor recovers the true operator order and its own
            // `refresh_intrinsic_smooth_penalty` reweights from the canonical
            // penalty on the chunk's coordinates rather than double-applying
            // the metric (issue #673).
            let mut chunk_atom = SaeManifoldAtom::new(
                atom.name.clone(),
                atom.basis_kind.clone(),
                atom.latent_dim,
                phi,
                jet,
                atom.decoder_coefficients.clone(),
                atom.smooth_penalty_raw.clone(),
            )?;
            chunk_atom.basis_evaluator = atom.basis_evaluator.clone();
            chunk_atom.basis_second_jet = atom.basis_second_jet.clone();
            // #972 / #977 T1: carry the active Grassmann frame onto the chunk
            // atom so the streaming per-chunk assembly uses the SAME factored
            // border layout as the dense path. Without this the chunk would
            // default to the full-`B` path and the streaming REML log-det would
            // be taken over a different (larger) β block than the dense one,
            // breaking the streaming↔dense log-det agreement (#847). The
            // decoder is unchanged, so the frame stays consistent with `B_k`.
            chunk_atom.decoder_frame = atom.decoder_frame.clone();
            atoms.push(chunk_atom);
        }
        // Rebuild the assignment from the chunk's logits + coords, preserving
        // each atom's latent manifold and the global assignment mode.
        let coord_values: Vec<LatentCoordValues> = chunk_coords
            .iter()
            .zip(self.assignment.coords.iter())
            .map(|(c, src)| {
                LatentCoordValues::from_matrix_with_manifold(
                    c.view(),
                    LatentIdMode::None,
                    src.manifold().clone(),
                )
            })
            .collect();
        let assignment =
            SaeAssignment::with_mode(chunk_logits, coord_values, self.assignment.mode)?;
        let mut term = SaeManifoldTerm::new(atoms, assignment)?;
        // The temperature schedule is global outer state; the chunk term is
        // assembled at the schedule's current temperature, which the caller
        // already baked into `self.assignment.mode` before materializing.
        term.temperature_schedule = self.temperature_schedule.clone();
        Ok(term)
    }

    /// Streaming / minibatch joint fit: refine the shared decoder coefficients
    /// `B_k` (and the ARD ρ axes) by sweeping the rows in chunks of
    /// `chunk_size`, accumulating the reduced Schur system over the shared β
    /// online, and NEVER materializing the `(N × M)` / `(N × K)` per-row
    /// buffers.
    ///
    /// For each outer iteration:
    ///
    /// 1. Each chunk `[start, end)` re-seeds its per-row latent state from the
    ///    chunk's `Z` slice (`chunk_init` supplies `(logits, coords)` — the SAE
    ///    PCA seed restricted to the chunk), materializes a small chunk term via
    ///    [`Self::materialize_chunk`], and assembles its Arrow-Schur system with
    ///    the β-tier penalties scaled by the chunk fraction `n_chunk / N` (so
    ///    they sum to exactly one global copy across the pass).
    /// 2. The chunk's reduced contribution `H_βt(H_tt)⁻¹H_tβ` and `H_βt(H_tt)⁻¹g_t`
    ///    are accumulated into a single global [`StreamingArrowSchur`] over β,
    ///    consuming each chunk's Kronecker `htbeta_matvec` procedurally.
    /// 3. After one full pass, the global reduced system is solved for `Δβ` with
    ///    the same LM ridge escalation as the full-batch driver, and a streaming
    ///    Armijo line search on `Δβ` accepts the step against the summed
    ///    per-chunk loss.
    /// 4. ARD ρ is refreshed online from the accumulated `Σ‖t‖²` and row count.
    ///
    /// Only the global decoder coefficients persist across chunks and outer
    /// iterations; the per-row `(logits, coords)` are re-seeded each pass and
    /// discarded. `self`'s own per-row buffers are left untouched — the fitted
    /// decoder is written back into `self`'s atoms.
    ///
    /// This is the out-of-core counterpart of [`Self::run_joint_fit_arrow_schur`]:
    /// the in-core driver holds the full `(N × M)` target and per-row state in
    /// memory, while this driver bounds peak memory to a single chunk by
    /// re-seeding `(logits, coords, Z)` through `chunk_init` on demand — the
    /// fit-side analogue of [`Self::streaming_exact_arrow_log_det`]'s chunked
    /// evidence assembly. Wired through [`Self::fit_streaming_in_memory`] for the
    /// in-memory case; a disk-backed `chunk_init` drives the LLM-scale fit.
    pub fn run_joint_fit_arrow_schur_streaming<F>(
        &mut self,
        n_total: usize,
        chunk_size: usize,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        mut chunk_init: F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        if !(step_size.is_finite() && step_size > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: step_size must be finite and positive; got {step_size}"
            ));
        }
        if chunk_size == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk_size must be positive"
                    .to_string(),
            );
        }
        if n_total == 0 {
            return Err(
                "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: n_total must be positive"
                    .to_string(),
            );
        }
        // #972 / #977 T1: magic-by-default frame activation, mirroring the dense
        // driver, so the streaming fit runs in the same factored coordinate
        // space (the chunk terms inherit the frames via `materialize_chunk`).
        self.ensure_decoder_frames_active_for_current_decoder()
            .map_err(|err| {
                format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
            })?;
        // The β-tier width the reduced-Schur accumulators are sized at: the
        // FACTORED border `Σ M_k·r_k` when frames are active (every chunk's
        // `sys.gb` / reduced Schur is in that space), else the full-`B`
        // `beta_dim`. The accepted `delta_beta` is a factored ΔC in the former
        // case and is lifted through the frames before being applied.
        let frames_engaged = self.frames_active();
        let border_dim = if frames_engaged {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };

        // ── Chunk-aware pre-fit decoder identifiability audit ───────────────
        {
            let mut grams = self.empty_decoder_gram_accumulator();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let (logits, coords, _z_chunk) = chunk_init(start, end)?;
                let chunk = self.materialize_chunk(logits, coords)?;
                chunk.accumulate_decoder_gram(&mut grams);
                start = end;
            }
            self.finalize_decoder_identifiability_audit(&grams, n_total)?;
        }

        let mut last_loss = SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
            evidence_gauge_deflated_directions: 0,
        };
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ── Pass 1: accumulate the global reduced Schur over β online. ──
            let options = ArrowSolveOptions::automatic(border_dim);
            let mut s_acc = Array2::<f64>::zeros((border_dim, border_dim));
            let mut rhs_acc = Array1::<f64>::zeros(border_dim);
            let mut gb_acc = Array1::<f64>::zeros(border_dim);
            // ρ (including the ARD precisions) is owned by the outer engine and
            // held FIXED across this streaming inner solve; the former online
            // `Σ t²` ARD accumulator + `update_ard_reml_from_sumsq` rule has
            // been removed in favour of the criterion-driven update.
            let mut pre_step_total = 0.0_f64;
            // Retain only the per-chunk row ranges so the line search can
            // re-materialize each chunk by re-invoking `chunk_init` at trial β
            // values. The chunk's `(logits, coords, Z)` are re-provided by the
            // seeder each time — never retained — so the pass stays O(Σ M_k²)
            // in memory rather than O(N · M) / O(N · K).
            let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
            let mut start = 0usize;
            while start < n_total {
                let end = (start + chunk_size).min(n_total);
                let n_chunk = end - start;
                let penalty_scale = n_chunk as f64 / n_total as f64;
                let (logits, coords, z_chunk) = chunk_init(start, end)?;
                if z_chunk.dim() != (n_chunk, self.output_dim()) {
                    return Err(format!(
                        "SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: chunk [{start}, {end}) \
                         Z slice shape {:?} != ({n_chunk}, {})",
                        z_chunk.dim(),
                        self.output_dim()
                    ));
                }
                let mut chunk = self.materialize_chunk(logits, coords)?;
                // #991: inherit the design honesty weight slice (see
                // streaming_exact_arrow_log_det for the no-renormalize rule).
                if let Some(w) = self.row_loss_weights.as_deref() {
                    chunk.row_loss_weights = Some(w[start..end].to_vec());
                }
                chunk_ranges.push((start, end));
                pre_step_total += chunk.penalized_objective_total(
                    z_chunk.view(),
                    rho,
                    analytic_penalties,
                    penalty_scale,
                )?;
                let sys = chunk
                    .assemble_arrow_schur_scaled(
                        z_chunk.view(),
                        rho,
                        analytic_penalties,
                        penalty_scale,
                    )
                    .map_err(|err| {
                        format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                    })?;
                // Accumulate the chunk's data-fit β gradient (its g_β already
                // carries the minibatch-scaled β-penalty gradient). `sys.gb` is
                // in the factored C-space when frames are engaged (the chunk
                // inherits them), matching `gb_acc`'s `border_dim` width.
                for j in 0..border_dim {
                    gb_acc[j] += sys.gb[j];
                }
                Self::accumulate_chunk_reduced_schur(
                    &sys,
                    ridge_ext_coord,
                    &options,
                    &mut s_acc,
                    &mut rhs_acc,
                )?;
                start = end;
            }
            // The summed chunk β-blocks already reconstruct the full
            // `H_ββ` (data-fit GN `G ⊗ I_p` + smoothness + analytic β); add the
            // global β ridge exactly once, and form the reduced RHS. After this
            // step `rhs_acc = Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i) − g_β` is the
            // negated Schur-reduced β gradient `−g_reduced`, so the reduced
            // system `S Δβ = rhs_acc` yields the marginal Newton step in β with
            // the per-row latent eliminated.
            for j in 0..border_dim {
                s_acc[[j, j]] += ridge_beta;
                rhs_acc[j] -= gb_acc[j];
            }
            // ── Solve the global reduced β system with LM ridge escalation. ──
            let delta_beta =
                solve_streaming_reduced_beta(&s_acc, &rhs_acc, &options).map_err(|err| {
                    format!("SaeManifoldTerm::run_joint_fit_arrow_schur_streaming: {err}")
                })?;
            // ── Streaming Armijo line search on Δβ. ──
            // The directional decrease uses the *reduced* β gradient
            // `g_reduced = −rhs_acc`, the true gradient of the β-marginal
            // objective along which the line search backtracks (the per-row
            // latent block is profiled out, not stepped, in streaming).
            let beta0 = self.flatten_beta();
            let mut directional_decrease = 0.0_f64;
            for j in 0..border_dim {
                // dd = −(g_reduced · Δβ) = −((−rhs_acc) · Δβ) = rhs_acc · Δβ.
                directional_decrease += rhs_acc[j] * delta_beta[j];
            }
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0)
            {
                // No descent direction available; ρ is engine-owned and fixed,
                // so just record the loss and stop.
                last_loss = self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                break;
            }
            // #972 / #977 T1: when frames are engaged, `delta_beta` is a factored
            // ΔC; pre-lift it ONCE to a full-`B` ΔB (`ΔB_k = ΔC_k U_kᵀ`) so the
            // per-trial β update is a plain `beta0 + step·ΔB` (the decoder lives
            // in the full p-space). Un-framed atoms lift by identity. On the
            // full-`B` path `delta_b` is just `delta_beta`.
            let delta_b: Array1<f64> = if frames_engaged {
                FrameProjection::new(self).lift_border_vec(delta_beta.view())
            } else {
                delta_beta.clone()
            };
            let mut trial_step = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                let mut trial_beta = beta0.clone();
                for j in 0..self.beta_dim() {
                    trial_beta[j] += trial_step * delta_b[j];
                }
                self.set_flat_beta(trial_beta.view())?;
                let (trial_loss, trial_total) = self.streaming_loss_and_penalized_objective_total(
                    &chunk_ranges,
                    rho,
                    analytic_penalties,
                    n_total,
                    &mut chunk_init,
                )?;
                let armijo_bound =
                    pre_step_total - SAE_MANIFOLD_ARMIJO_C1 * trial_step * directional_decrease;
                if trial_total.is_finite() && trial_total <= armijo_bound {
                    accepted_loss = Some(trial_loss);
                    break;
                }
                trial_step *= 0.5;
            }
            match accepted_loss {
                Some(loss) => {
                    last_loss = loss;
                }
                None => {
                    // Restore the pre-step β before stopping. ρ is engine-owned
                    // and held fixed across the streaming inner solve.
                    self.set_flat_beta(beta0.view())?;
                    last_loss =
                        self.streaming_loss(&chunk_ranges, rho, n_total, &mut chunk_init)?;
                    break;
                }
            }
        }
        Ok(last_loss)
    }

    /// In-memory driver for [`Self::run_joint_fit_arrow_schur_streaming`]: build
    /// the `chunk_init` seeder by slicing the resident `target`, `self.assignment`
    /// logits and `self.assignment` coords per row-range — the identical chunking
    /// [`Self::streaming_exact_arrow_log_det`] already uses for the evidence pass.
    ///
    /// This is the streaming fit's wiring for data that is already resident: it
    /// bounds the Newton solve's peak memory to one chunk (no `(N × M)` /
    /// `(N × K)` materialization) while reading from the in-core buffers. The
    /// out-of-core LLM-scale path swaps this seeder for a disk-backed loader and
    /// calls `run_joint_fit_arrow_schur_streaming` directly. `chunk_size` is the
    /// auto-derived [`Self::streaming_plan`] chunk (clamped to `n`).
    pub fn fit_streaming_in_memory(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        let n_total = self.n_obs();
        if target.dim() != (n_total, self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::fit_streaming_in_memory: target must be ({}, {}); got {:?}",
                n_total,
                self.output_dim(),
                target.dim()
            ));
        }
        let chunk_size = self.streaming_plan().chunk_size.min(n_total.max(1));
        // Snapshot the resident seed state so the per-pass re-seed is a pure
        // read (the streaming driver re-invokes the closure every line-search
        // trial and must hand back identical seeds each time).
        let seed_logits = self.assignment.logits.clone();
        let seed_coords: Vec<Array2<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.as_matrix().to_owned())
            .collect();
        // The `target` view is sliced per chunk (not cloned wholesale); the
        // driver re-invokes this seeder every line-search trial, and each call
        // returns owned per-chunk copies. Shape is validated inside
        // `run_joint_fit_arrow_schur_streaming` against `(n_chunk, output_dim)`.
        let chunk_init = move |start: usize, end: usize| {
            let logits = seed_logits.slice(s![start..end, ..]).to_owned();
            let coords: Vec<Array2<f64>> = seed_coords
                .iter()
                .map(|coord| coord.slice(s![start..end, ..]).to_owned())
                .collect();
            let z_chunk = target.slice(s![start..end, ..]).to_owned();
            Ok((logits, coords, z_chunk))
        };
        self.run_joint_fit_arrow_schur_streaming(
            n_total,
            chunk_size,
            rho,
            analytic_penalties,
            max_iter,
            step_size,
            ridge_ext_coord,
            ridge_beta,
            chunk_init,
        )
    }

    /// Accumulate one chunk system's reduced-Schur contribution into the shared
    /// `(β × β)` accumulator and reduced RHS, consuming the chunk's Kronecker
    /// `htbeta_matvec` procedurally via [`StreamingArrowSchur`].
    ///
    /// The chunk system's β-block already carries the chunk's data-fit
    /// Gauss-Newton curvature `G_chunk ⊗ I_p` (a genuine per-row sum) plus its
    /// minibatch-scaled smoothness / analytic-β penalty. So the contribution
    /// `s_acc_chunk = hbb_chunk − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` and
    /// `rhs_acc_chunk = +Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i)` sum across a full pass
    /// to `H_ββ − Σ_all_i (…)` and `Σ_all_i (…)` respectively, with the global
    /// β ridge added exactly once by the caller. No per-chunk ridge is applied.
    fn accumulate_chunk_reduced_schur(
        sys: &ArrowSchurSystem,
        ridge_ext_coord: f64,
        options: &ArrowSolveOptions,
        s_acc: &mut Array2<f64>,
        rhs_acc: &mut Array1<f64>,
    ) -> Result<(), String> {
        let k = sys.k;
        let chunk_n = sys.rows.len();
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_n.max(1));
        // `reset_accumulator(0.0)` seeds `s_acc` with the chunk's dense β-block
        // (`hbb_chunk`, including the data-fit GN block and the minibatch-scaled
        // penalty) and no ridge; `accumulate_chunk` then subtracts the per-row
        // reduction. The global β ridge is applied once by the streaming driver.
        streaming
            .reset_accumulator(0.0)
            .map_err(|e| e.to_string())?;
        streaming
            .accumulate_chunk(0, chunk_n, ridge_ext_coord, options.mode)
            .map_err(|e| e.to_string())?;
        let (contrib_s, contrib_rhs) = streaming.take_accumulators();
        for i in 0..k {
            rhs_acc[i] += contrib_rhs[i];
            for j in 0..k {
                s_acc[[i, j]] += contrib_s[[i, j]];
            }
        }
        Ok(())
    }

    /// Streaming total loss: sum of the minibatch-scaled per-chunk losses at the
    /// current β, re-materializing each chunk from a fresh re-seed via
    /// `chunk_init`. The β-penalty terms are scaled by the chunk fraction so the
    /// global smoothness penalty is counted once across the pass.
    fn streaming_loss<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        rho: &SaeManifoldRho,
        n_total: usize,
        chunk_init: &mut F,
    ) -> Result<SaeManifoldLoss, String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        let mut data_fit = 0.0_f64;
        let mut assignment_sparsity = 0.0_f64;
        let mut smoothness = 0.0_f64;
        let mut ard = 0.0_f64;
        for &(start, end) in chunk_ranges {
            let n_chunk = end - start;
            let penalty_scale = n_chunk as f64 / n_total as f64;
            let (logits, coords, z_chunk) = chunk_init(start, end)?;
            let mut chunk = self.materialize_chunk(logits, coords)?;
            // #991: inherit the design honesty weight slice (global mean-1
            // normalization preserved; see streaming_exact_arrow_log_det).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let loss = chunk.loss_scaled(z_chunk.view(), rho, penalty_scale)?;
            data_fit += loss.data_fit;
            assignment_sparsity += loss.assignment_sparsity;
            smoothness += loss.smoothness;
            ard += loss.ard;
        }
        Ok(SaeManifoldLoss {
            data_fit,
            assignment_sparsity,
            smoothness,
            ard,
            evidence_gauge_deflated_directions: 0,
        })
    }

    fn streaming_loss_and_penalized_objective_total<F>(
        &self,
        chunk_ranges: &[(usize, usize)],
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        n_total: usize,
        chunk_init: &mut F,
    ) -> Result<(SaeManifoldLoss, f64), String>
    where
        F: FnMut(usize, usize) -> Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>), String>,
    {
        let mut data_fit = 0.0_f64;
        let mut assignment_sparsity = 0.0_f64;
        let mut smoothness = 0.0_f64;
        let mut ard = 0.0_f64;
        let mut total = 0.0_f64;
        for &(start, end) in chunk_ranges {
            let n_chunk = end - start;
            let penalty_scale = n_chunk as f64 / n_total as f64;
            let (logits, coords, z_chunk) = chunk_init(start, end)?;
            let mut chunk = self.materialize_chunk(logits, coords)?;
            // #991: inherit the design honesty weight slice (global mean-1
            // normalization preserved; see streaming_exact_arrow_log_det).
            if let Some(w) = self.row_loss_weights.as_deref() {
                chunk.row_loss_weights = Some(w[start..end].to_vec());
            }
            let loss = chunk.loss_scaled(z_chunk.view(), rho, penalty_scale)?;
            data_fit += loss.data_fit;
            assignment_sparsity += loss.assignment_sparsity;
            smoothness += loss.smoothness;
            ard += loss.ard;
            total += chunk.penalized_objective_total(
                z_chunk.view(),
                rho,
                analytic_penalties,
                penalty_scale,
            )?;
        }
        Ok((
            SaeManifoldLoss {
                data_fit,
                assignment_sparsity,
                smoothness,
                ard,
                evidence_gauge_deflated_directions: 0,
            },
            total,
        ))
    }
}

fn reconstruction_explained_variance(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
) -> Option<f64> {
    if target.dim() != fitted.dim() {
        return None;
    }
    let (n, p) = target.dim();
    if n == 0 || p == 0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0;
        for row in 0..n {
            acc += target[[row, col]];
        }
        means[col] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let residual = target[[row, col]] - fitted[[row, col]];
            ssr += residual * residual;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    if ssr.is_finite() && sst.is_finite() && sst > f64::MIN_POSITIVE {
        Some(1.0 - ssr / sst)
    } else {
        None
    }
}

/// Outer REML objective for the SAE-manifold term.
///
/// Routes the SAE's smoothing hyperparameters ρ
/// (`log_lambda_sparse`, `log_lambda_smooth`, per-atom/axis `log_ard`)
/// through the *one* generic [`OuterObjective`] engine + cascade that the
/// main GAM REML path uses, instead of the SAE's deleted forked
/// `update_ard_reml` fixed-point rule. Each outer eval runs the inner
/// `(t, β)` arrow-Schur Newton solve at the engine's current ρ and returns
/// the true REML criterion (see [`SaeManifoldTerm::reml_criterion`]).
///
/// The SAE's outer coordinates ρ are all penalty-like / τ (precisions and
/// log-smoothing-strengths), so `psi_dim = 0`: there are no design-moving
/// (ψ) coordinates. No analytic outer gradient/Hessian is exposed yet
/// (task v2 wires the selected-inverse block-trace ρ-gradient), so this
/// is a cost-only objective and the engine routes it to a derivative-free /
/// finite-difference outer strategy per the planner.
pub struct SaeManifoldOuterObjective {
    term: SaeManifoldTerm,
    /// Pristine term to restore from on `reset` (multi-start baseline).
    baseline_term: SaeManifoldTerm,
    target: Array2<f64>,
    registry: Option<AnalyticPenaltyRegistry>,
    /// ρ template carrying the per-atom ARD dims; `from_flat` reads its
    /// layout. Updated to each evaluated ρ so `into_fitted` can report the
    /// last ρ the engine settled on.
    current_rho: SaeManifoldRho,
    /// Pristine ρ to restore from on `reset`.
    baseline_rho: SaeManifoldRho,
    inner_max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    /// Last inner loss breakdown observed (for `into_fitted`).
    last_loss: Option<SaeManifoldLoss>,
    /// Optional warm-start β slot. When the cache / continuation walk seeds a
    /// β, the next inner solve opens from it instead of cold.
    seeded_beta: Option<Array1<f64>>,
}

impl SaeManifoldOuterObjective {
    pub fn new(
        mut term: SaeManifoldTerm,
        target: Array2<f64>,
        registry: Option<AnalyticPenaltyRegistry>,
        init_rho: SaeManifoldRho,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Self {
        term.expected_evidence_gauge_deflated_directions = None;
        let baseline_term = term.clone();
        let baseline_rho = init_rho.clone();
        Self {
            term,
            baseline_term,
            target,
            registry,
            current_rho: init_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss: None,
            seeded_beta: None,
        }
    }

    /// Consume the objective, returning the inner-fitted term, the last ρ the
    /// engine evaluated, and the inner loss breakdown at that ρ.
    pub fn into_fitted(self) -> (SaeManifoldTerm, SaeManifoldRho, SaeManifoldLoss) {
        let Self {
            term,
            mut baseline_term,
            target,
            registry,
            current_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss,
            ..
        } = self;
        let pristine_seed_term = baseline_term.clone();
        let pristine_seed_rho = baseline_rho.clone();
        let mut fitted_rho = current_rho;
        let loss = last_loss.unwrap_or_else(|| SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
            evidence_gauge_deflated_directions: 0,
        });
        // Basin guard against the multi-atom routing-collapse failure mode
        // (#629 #630). The outer ρ cascade mutates `term` cumulatively across
        // candidate ρ evaluations and never restores it between evals, so a
        // single ill-conditioned ρ poll can drag the per-row routing off the
        // decisive seed basin (the EM routing-seed / decoder-projection start)
        // into the near-uniform saddle. The settled `term` then reports that
        // collapsed routing even though the seed basin reconstructs the planted
        // disjoint atoms far better. `baseline_term` preserves the pristine
        // seeded geometry; re-solve the inner joint fit from it at the SAME
        // settled ρ the engine selected (smoothing choice is untouched) and
        // keep whichever converged state attains the lower penalized objective.
        // For an already-routed fit the two coincide (the seed basin is the
        // optimum), so this is a no-op there and never weakens the criterion;
        // for a drifted fit it recovers the routed solution the seed reached.
        let settled_objective =
            term.penalized_objective_total(target.view(), &fitted_rho, registry.as_ref(), 1.0);
        let mut rho_seed = fitted_rho.clone();
        let seed_solve = match baseline_term.streaming_plan().admitted_or_error(
            baseline_term.n_obs(),
            baseline_term.output_dim(),
            baseline_term.k_atoms(),
        ) {
            Ok(plan)
                if plan.streaming
                    && plan.estimated_full_batch_bytes > plan.in_core_budget_bytes
                    && plan.estimated_dense_schur_bytes <= plan.in_core_budget_bytes =>
            {
                baseline_term.fit_streaming_in_memory(
                    target.view(),
                    &mut rho_seed,
                    registry.as_ref(),
                    inner_max_iter,
                    learning_rate,
                    ridge_ext_coord,
                    ridge_beta,
                )
            }
            Ok(_) => baseline_term.run_joint_fit_arrow_schur(
                target.view(),
                &mut rho_seed,
                registry.as_ref(),
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            ),
            Err(err) => Err(err),
        };
        let mut seed_won = false;
        if let (Ok(settled_total), Ok(_)) = (&settled_objective, &seed_solve) {
            let seed_total = baseline_term.penalized_objective_total(
                target.view(),
                &fitted_rho,
                registry.as_ref(),
                1.0,
            );
            if let Ok(seed_total) = seed_total {
                seed_won = seed_total.is_finite() && seed_total < *settled_total;
            }
        }
        let (mut fitted, mut fitted_loss) = if seed_won {
            let seed_loss = seed_solve.expect("seed_won implies seed_solve is Ok");
            (baseline_term, seed_loss)
        } else {
            (term, loss)
        };
        if let (Ok(seed_fit), Ok(returned_fit)) =
            (pristine_seed_term.try_fitted(), fitted.try_fitted())
            && let (Some(seed_ev), Some(returned_ev)) = (
                reconstruction_explained_variance(target.view(), seed_fit.view()),
                reconstruction_explained_variance(target.view(), returned_fit.view()),
            )
            && seed_ev >= SAE_PRISTINE_SEED_EV_RETAIN_FLOOR
            && returned_ev < SAE_PRISTINE_SEED_EV_RETAIN_FLOOR
            && returned_ev + SAE_FINAL_EV_DEGRADATION_TOL < seed_ev
            && let Ok(seed_loss) = pristine_seed_term.loss(target.view(), &pristine_seed_rho)
        {
            fitted = pristine_seed_term;
            fitted_rho = pristine_seed_rho;
            fitted_loss = seed_loss;
        }
        // #1019 — the post-fit assembly seam: canonicalize every eligible
        // atom's chart to its canonical Diff(M) representative (arc length
        // for d = 1, minimum-isometry-defect flow for d = 2 torus atoms)
        // BEFORE the fitted term is handed to the payload / residual-gauge
        // certificate. Internally objective-gated and image-frozen (the
        // fitted state is restored verbatim on any failure or tolerance
        // breach), so the fit this returns is never degraded — an error here
        // is a refused canonicalization, not a broken fit.
        if let Err(err) =
            fitted.canonicalize_charts_post_fit(target.view(), &fitted_rho, registry.as_ref())
        {
            log::debug!("into_fitted: chart canonicalization refused: {err}");
        }
        (fitted, fitted_rho, fitted_loss)
    }

    /// First-order optimality certificate for this fit (#934).
    ///
    /// At the converged outer optimum `ρ̂` this runs the self-audit the desync
    /// bug genus (#752/#748/#808/#901) was always diagnosed by hand: it draws
    /// one deterministic direction `v` from the problem fingerprint, central-
    /// differences the criterion **value path** at `ρ̂ ± h v` (with a Richardson
    /// `2h` step for the FD's own error bar), and compares against the analytic
    /// directional derivative `∇V(ρ̂)·v` from the production gradient path. The
    /// returned [`CriterionCertificate`] records whether the objective and its
    /// analytic gradient agree *here*, on this data shape, where #901-class
    /// desyncs actually manifest.
    ///
    /// The finite difference is the *audit instrument*, not an estimator: it
    /// only checks the production analytic gradient against the production value
    /// path at one point after convergence, so it is fully compatible with the
    /// exact-REML-only policy (see `sae_optimality_certificate`). Cost is four
    /// criterion value-path evaluations at the single final point.
    ///
    /// The value probes are taken on a **clone of the pristine baseline term**
    /// so the production fitted state is untouched and the value caches start
    /// cold — they must not alias the gradient path's converged warm state,
    /// since that aliasing is exactly what the certificate audits. Call before
    /// [`Self::into_fitted`].
    pub fn optimality_certificate(&mut self) -> Result<CriterionCertificate, String> {
        let rho_hat_flat = self.current_rho.to_flat();
        let dir = deterministic_probe_direction(rho_hat_flat.view());
        let h = probe_step(rho_hat_flat.view());

        // Analytic directional derivative at ρ̂, from the production gradient
        // path (same code the outer optimizer consumed). Re-forming the cache
        // here re-runs the inner solve at the settled ρ — already at its
        // optimum, so it converges immediately — and reads the exact analytic
        // outer gradient with the third-order correction included.
        let rho_hat = self.current_rho.clone();
        let (_v_hat, loss_hat, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho_hat,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        let solver = self.term.outer_gradient_arrow_solver(&cache)?;
        let components = self
            .term
            .analytic_outer_rho_gradient_components(&rho_hat, &loss_hat, &cache, &solver)?;
        let grad = components.gradient_with_available_correction();
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let analytic_directional: f64 = grad.iter().zip(dir.iter()).map(|(g, d)| g * d).sum();

        // Value-path probe on a cold clone of the pristine baseline term: the
        // value path must be exercised WITHOUT the gradient path's warm caches,
        // since aliasing the two is exactly the failure the certificate audits.
        let mut probe_term = self.baseline_term.clone();
        let value_at = |term: &mut SaeManifoldTerm, mult: f64| -> Result<f64, String> {
            let flat: Array1<f64> =
                Array1::from_shape_fn(rho_hat_flat.len(), |i| rho_hat_flat[i] + mult * h * dir[i]);
            let rho = self.baseline_rho.from_flat(flat.view());
            let (cost, _loss) = term.reml_criterion(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )?;
            Ok(cost)
        };
        let plus_h = value_at(&mut probe_term, 1.0)?;
        let minus_h = value_at(&mut probe_term, -1.0)?;
        let plus_2h = value_at(&mut probe_term, 2.0)?;
        let minus_2h = value_at(&mut probe_term, -2.0)?;

        let well_posed = plus_h.is_finite()
            && minus_h.is_finite()
            && plus_2h.is_finite()
            && minus_2h.is_finite();
        let samples = DirectionalSamples {
            plus_h,
            minus_h,
            plus_2h,
            minus_2h,
            step: h,
            grad_norm,
            analytic_directional,
            well_posed,
        };
        Ok(certificate_from_samples(&samples))
    }

    /// Posterior shape uncertainty of the fitted atoms — per-atom decoder
    /// covariance and ambient bands (see
    /// [`SaeManifoldTerm::assemble_shape_uncertainty`]).
    ///
    /// Recomputes the converged joint-Hessian Laplace factor at the settled ρ
    /// — the same undamped Direct factor the REML criterion forms at the inner
    /// optimum — and reads the per-atom covariance and bands off its cached
    /// Schur factor, scaling by the Gaussian reconstruction dispersion `φ̂`.
    /// The term is already at the optimum after the outer fit, so the inner
    /// re-solve converges immediately. Call before [`Self::into_fitted`].
    /// The most recent curvature-homotopy entry walk outcome on the live term
    /// (#1007), or `None` when no walk has run. Surfaced on the objective so the
    /// arrival / bifurcation / collapse outcome is observable without consuming
    /// the objective via [`Self::into_fitted`].
    pub fn curvature_walk_report(&self) -> Option<&CurvatureWalkReport> {
        self.term.curvature_walk_report()
    }

    pub fn decoder_shape_uncertainty(&mut self) -> Result<SaeShapeUncertainty, String> {
        let rho = self.current_rho.clone();
        let plan = self.term.streaming_plan().admitted_or_error(
            self.term.n_obs(),
            self.term.output_dim(),
            self.term.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            let loss = self.term.loss(self.target.view(), &rho)?;
            let n_scalar = (self.term.n_obs().saturating_mul(self.term.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self
                .term
                .shape_uncertainty_without_decoder_covariance(dispersion));
        }
        let (_cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        let dispersion = self.term.reconstruction_dispersion(&loss, &cache, &rho)?;
        self.term.assemble_shape_uncertainty(&cache, dispersion)
    }

    /// Certified curvature-homotopy entry walk (#1007): replace the blind
    /// multi-seed multistart with one predictor-corrector walk of the basis
    /// curvature dial `η` from the Eckart-Young anchor (`η = 0`, global by
    /// construction) to the full curved basis (`η = 1`).
    ///
    /// 1. **Anchor (`η = 0`).** The curved columns are suppressed, so the decoder
    ///    sub-problem is convex and its optimum is the Eckart-Young projection
    ///    certified by [`linear_span_anchor`]; the joint corrector lands on it. A
    ///    degenerate anchor (no recoverable linear span / a non-finite target /
    ///    a failed relaxation solve) returns `Ok(false)` — the caller falls back
    ///    to the cascade.
    /// 2. **Walk `η: 0 → 1`.** Each waypoint: a *predictor* applies the IFT step
    ///    `Δβ = −H⁻¹ · ∂g_β/∂η · Δη` on the cached evidence factor
    ///    ([`ArrowFactorCache::full_inverse_apply`], β-channel; the t / gate
    ///    blocks are re-converged by the corrector), then the *corrector* (the
    ///    damped joint Newton in `reml_criterion_with_cache`) re-converges at
    ///    `η_next`. The invariant is that the arrow factor's smallest pivot stays
    ///    at or above the safe-SPD floor `√eps · max(diag_scale, 1)`; when it
    ///    shrinks the `η` step is halved and retried from the last converged
    ///    state. A pivot collapse at the minimum step is a DETECTED bifurcation
    ///    (recorded on [`CurvatureWalkReport`], never silent) and returns
    ///    `Ok(false)`.
    /// 3. **Arrival (`η = 1`).** The term is left warm at the certified branch's
    ///    `η = 1` solution; the report is recorded and the call returns
    ///    `Ok(true)`.
    ///
    /// The direct helper walks at the construction entry ρ (`baseline_rho`);
    /// the outer seed loop uses `run_curvature_homotopy_entry_at_rho` so every
    /// generated candidate gets its own entry solve before the ρ-anneal.
    pub fn run_curvature_homotopy_entry(&mut self) -> Result<bool, String> {
        let rho = self.baseline_rho.clone();
        self.run_curvature_homotopy_entry_at_rho(&rho)
    }

    /// Certified curvature-homotopy entry walk at an explicit seed ρ. The outer
    /// seed loop calls this form so each generated candidate lands on its own
    /// fixed baseline instead of every walk reusing the construction baseline.
    pub fn run_curvature_homotopy_entry_at_rho(
        &mut self,
        rho: &SaeManifoldRho,
    ) -> Result<bool, String> {
        let rho = rho.clone();
        self.current_rho = rho.clone();
        let isometry_targets = self
            .registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        self.set_isometry_homotopy_weight(0.0, &isometry_targets);
        // Eckart-Young anchor certificate at η = 0 (output-subspace coords). A
        // degenerate anchor is the cascade's job, not the walk's.
        let anchor = match linear_span_anchor(&self.term, self.target.view()) {
            Ok(anchor) => anchor,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor degenerate ({err}); deferring to seed cascade"
                );
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };
        let anchor_residual_norm_sq = anchor.residual_norm_sq;

        // Anchor corrector at η = 0: the convex linear relaxation.
        let (_loss0, mut last_cache) = match self.solve_at_eta(&rho, 0.0, &isometry_targets) {
            Ok(pair) => pair,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor solve failed at η=0 ({err}); deferring to cascade"
                );
                self.term.set_homotopy_eta(1.0).ok();
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };

        let mut eta = 0.0_f64;
        let mut eta_step = CURVATURE_WALK_INITIAL_ETA_STEP;
        let mut eta_steps = 0usize;
        let mut step_halvings = 0usize;
        let mut total_correctors = 0usize;
        let mut bifurcation: Option<CurvatureBifurcation> = None;

        // Identity-homotopy shortcut: with no curved basis columns anywhere
        // AND an all-zero isometry ramp, `solve_at_eta` poses the SAME problem
        // at every η — the grid legs after the anchor corrector would re-solve
        // its converged state verbatim, paying a full criterion/factorization
        // rebuild each time. The anchor + first corrector carry all the value
        // (certified Eckart-Young initialization + one full solve); arrive at
        // η = 1 directly. `set_homotopy_eta(1.0)` restores the plain-evaluate
        // fast path (η == 1 skips the dialed evaluator); the isometry weights
        // are already at target because every ramp target is zero.
        if isometry_targets.iter().all(|&target| target == 0.0)
            && self.term.curvature_homotopy_eta_is_inert()?
        {
            self.term.set_homotopy_eta(1.0)?;
            eta = 1.0;
        }

        'walk: while eta < 1.0 {
            let eta_next = (eta + eta_step).min(1.0);
            let d_eta = eta_next - eta;

            // Predictor: IFT step on the cached factor warm-starts the corrector
            // (β-channel only; `w_t = 0`). Non-fatal — on any predictor failure
            // the corrector simply opens from the previous η's converged β.
            if let Ok(dg_beta) = self
                .term
                .curvature_beta_gradient_eta_derivative(self.target.view())
                && dg_beta.len() == last_cache.k
            {
                let w_t = Array1::<f64>::zeros(last_cache.delta_t_len());
                if let Ok((_u_t, u_beta)) =
                    last_cache.full_inverse_apply(w_t.view(), dg_beta.view())
                {
                    let mut beta = self.term.flatten_beta();
                    if beta.len() == u_beta.len() {
                        for (b, u) in beta.iter_mut().zip(u_beta.iter()) {
                            *b -= u * d_eta;
                        }
                        if beta.iter().all(|v| v.is_finite()) {
                            self.term.set_flat_beta(beta.view()).ok();
                        }
                    }
                }
            }

            // Corrector at η_next.
            let cache = match self.solve_at_eta(&rho, eta_next, &isometry_targets) {
                Ok((_loss, cache)) => cache,
                Err(err) => {
                    // Corrector struggled: treat like a pivot shrink — halve the
                    // η step and retry from the last converged state. A failure
                    // at the minimum step is a branch bifurcation.
                    if eta_step <= CURVATURE_WALK_MIN_ETA_STEP {
                        log::info!(
                            "[#1007] curvature corrector failed at η={eta_next:.4} at the minimum \
                             η-step ({err}); recording branch bifurcation"
                        );
                        bifurcation = Some(CurvatureBifurcation {
                            eta: eta_next,
                            min_pivot: 0.0,
                        });
                        break 'walk;
                    }
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
            };
            total_correctors += 1;

            // Pivot invariant: min pivot ≥ √eps · max(diag_scale, 1), the same
            // safe-SPD floor the inner solver uses — measured ON THE GAUGE
            // QUOTIENT. A closed-form gauge null (affine chart freedom, circle
            // rotation) is constant along the entire η-walk, so it can never
            // signal a branch bifurcation; only a NON-gauge pivot collapse can.
            // Without this discrimination the walk dies at η≈0 on any fixture
            // whose ambient dimension is small enough for the gauge directions
            // to dominate (every p=2 atlas tile), and the fit pays for the full
            // seed cascade instead. `outer_gradient_arrow_solver` succeeds iff
            // the sub-floor pivots are explained by the closed-form gauge span
            // (the same Faddeev-Popov deflation the gradient lane uses) and
            // errs honestly otherwise, which is exactly the verdict needed.
            let pivot = arrow_factor_min_pivot(&cache).min_pivot.unwrap_or(0.0);
            let diag_scale = arrow_factor_max_pivot(&cache).unwrap_or(1.0);
            let floor = f64::EPSILON.sqrt() * diag_scale.max(1.0);
            let pivot_deficit_is_gauge = !(pivot.is_finite() && pivot >= floor)
                && self.term.outer_gradient_arrow_solver(&cache).is_ok();
            if !(pivot.is_finite() && pivot >= floor) && !pivot_deficit_is_gauge {
                if eta_step > CURVATURE_WALK_MIN_ETA_STEP {
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
                log::info!(
                    "[#1007] curvature branch bifurcation at η={eta_next:.4}: min pivot \
                     {pivot:.3e} < floor {floor:.3e}; deferring to seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta: eta_next,
                    min_pivot: pivot,
                });
                break 'walk;
            }

            // Accepted waypoint: advance and gently regrow the step toward the
            // nominal cadence (a clean stretch should not stay throttled).
            eta = eta_next;
            last_cache = cache;
            eta_steps += 1;
            eta_step = (eta_step * 2.0).min(CURVATURE_WALK_INITIAL_ETA_STEP);
            if total_correctors >= CURVATURE_WALK_MAX_CORRECTORS && eta < 1.0 {
                log::info!(
                    "[#1007] curvature walk hit its corrector budget at η={eta:.4}; deferring to \
                     seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta,
                    min_pivot: pivot,
                });
                break 'walk;
            }
        }

        let arrived = bifurcation.is_none() && eta >= 1.0;
        // Leave the term at the real (η = 1) objective regardless of outcome so
        // an aborted walk hands the cascade the full basis.
        if !arrived {
            self.term.set_homotopy_eta(1.0).ok();
        }
        self.set_isometry_homotopy_weight(1.0, &isometry_targets);
        if arrived
            && let Ok(before_fit) = self.term.try_fitted()
            && let Some(before_ev) =
                reconstruction_explained_variance(self.target.view(), before_fit.view())
            && before_ev < 0.9
        {
            let snapshot = self.term.snapshot_mutable_state();
            let accepted_polish = self
                .term
                .refit_decoder_least_squares_at_current_state(self.target.view())
                .and_then(|()| {
                    self.term
                        .seed_coords_by_decoder_projection(self.target.view(), 256)
                })
                .and_then(|()| {
                    self.term
                        .refit_decoder_least_squares_at_current_state(self.target.view())
                })
                .and_then(|()| {
                    let after_fit = self.term.try_fitted()?;
                    let Some(after_ev) =
                        reconstruction_explained_variance(self.target.view(), after_fit.view())
                    else {
                        return Err(
                            "curvature-homotopy decoder LSQ polish produced no EV".to_string()
                        );
                    };
                    if after_ev > before_ev {
                        self.term.loss(self.target.view(), &rho)
                    } else {
                        Err(format!(
                            "curvature-homotopy decoder LSQ polish refused: EV {after_ev:.6} \
                             did not improve from {before_ev:.6}"
                        ))
                    }
                });
            match accepted_polish {
                Ok(loss) => self.last_loss = Some(loss),
                Err(_) => self.term.restore_mutable_state(&snapshot),
            }
        }
        let collapse_events = self.term.collapse_events().len();
        self.term.set_curvature_walk_report(CurvatureWalkReport {
            arrived,
            anchor_residual_norm_sq,
            bifurcation,
            eta_steps,
            step_halvings,
            collapse_events,
            reseeds: 0,
        });
        Ok(arrived)
    }

    /// Curvature-homotopy corrector (#1007): install the `η` dial and re-converge
    /// the joint fit at the entry ρ, returning the converged loss and the
    /// undamped evidence cache (for the predictor IFT solve + the pivot
    /// invariant). The dial is read on the next basis refresh inside the solve.
    fn solve_at_eta(
        &mut self,
        rho: &SaeManifoldRho,
        eta: f64,
        isometry_targets: &[f64],
    ) -> Result<(SaeManifoldLoss, ArrowFactorCache), String> {
        self.term.set_homotopy_eta(eta)?;
        self.set_isometry_homotopy_weight(eta, isometry_targets);
        let (_cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.last_loss = Some(loss.clone());
        Ok((loss, cache))
    }

    fn set_isometry_homotopy_weight(&mut self, eta: f64, targets: &[f64]) {
        if targets.is_empty() {
            return;
        }
        if let Some(registry) = self.registry.as_mut() {
            let eta = eta.clamp(0.0, 1.0);
            let weights: Vec<f64> = targets.iter().map(|target| eta * target).collect();
            registry.set_isometry_scalar_weights(&weights);
        }
    }

    fn add_fit_data_collapse_penalty(&mut self, cost: f64) -> Result<f64, String> {
        let fitted = self.term.try_fitted()?;
        let assignments = self.term.assignment.assignments();
        let collapsed = self.term.record_fit_data_collapse_if_needed(
            self.target.view(),
            fitted.view(),
            assignments.view(),
            self.inner_max_iter,
        )?;
        if collapsed {
            Ok(cost + SAE_FIT_DATA_COLLAPSE_COST)
        } else {
            Ok(cost)
        }
    }

    fn is_recoverable_value_probe_refusal(err: &str) -> bool {
        err.contains("inner solve did not converge at fixed ρ")
            || err.contains(
                "undamped evidence factorization hit a non-PD per-row H_tt block before KKT",
            )
    }

    /// Shared cost path: evaluate the REML criterion at `rho_flat`, updating
    /// the cached ρ / loss and (optionally) priming the inner solve from a
    /// seeded β. Returns `(cost, β̂)`.
    ///
    /// `refine_progress_extension = false` selects the value-probe refine
    /// budget (#1029). The budget cut keeps the SAME KKT/step tolerance as the
    /// full path — a successfully returned value is converged to the identical
    /// stationarity measure, so probe values and accepted-point values are
    /// always comparable; only an expensive grind-then-refuse becomes a cheap
    /// refusal (a recoverable line-search reject).
    fn evaluate_with_refine_policy(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        refine_progress_extension: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take() {
            // Warm-start the inner decoder coefficients before the solve.
            if beta.len() == self.term.beta_dim() {
                self.term.set_flat_beta(beta.view())?;
            }
        }
        let (cost, loss) = self.term.reml_criterion_with_refine_policy(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
            refine_progress_extension,
        )?;
        self.current_rho = rho;
        self.last_loss = Some(loss);
        let beta_hat = self.term.flatten_beta();
        let cost = self.add_fit_data_collapse_penalty(cost)?;
        Ok((cost, beta_hat))
    }

    /// Fellner-Schall / Mackay multiplicative fixed-point step on ρ at
    /// `rho_flat`. Runs the inner `(t, β)` solve to convergence at fixed ρ
    /// (sharing the single Direct factor with the REML criterion), then
    /// returns `(cost, additive-log-steps, β̂)`.
    ///
    /// All ρ coords are log-quantities, so the engine's additive step
    /// `rho_new = rho + step` IS the multiplicative FS update. Per coord:
    /// - ARD axis (k,j): `α_new = φ̂ n / (‖t_kj‖² + tr_kj(H⁻¹))`,
    ///   `step = ln α_new − log_ard[k][j]`. The `tr_kj(H⁻¹)` posterior
    ///   variance (from the selected-inverse latent diagonal) is exactly the
    ///   term the deleted `α=n/‖t‖²` rule dropped, so α cannot collapse on a
    ///   degenerate axis: as `‖t‖²→0`, `tr_kj(H⁻¹)→1/α` bounds the
    ///   denominator and the fixed point has a finite root.
    /// - λ_smooth: `λ_new = φ̂[p·Σ_k rank S_k − tr(S_β⁻¹ M)] / βᵀ(⊕S_k⊗I_p)β`
    ///   (Wood-Fasiolo EFS), `step = ln λ_new − log_lambda_smooth`.
    /// - λ_sparse: 0.0 — the assignment-sparsity priors (softmax entropy,
    ///   gated L1, IBP) are non-quadratic, so no Gaussian-logdet FS fixed
    ///   point exists; it stays cost-driven (the cascade still moves it via
    ///   the cost path when EFS is not the active lane for that coord).
    fn efs_step(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<EfsEval, String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term.set_flat_beta(beta.view())?;
        }
        let (cost, loss, cache) = self.term.reml_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.current_rho = rho.clone();
        let dispersion = self
            .term
            .reconstruction_dispersion(&loss, &cache, &rho)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: dispersion: {e}"))?;
        self.last_loss = Some(loss);

        let n_obs = self.term.n_obs() as f64;
        let sumsq = self.term.ard_coord_sumsq();
        let traces = self
            .term
            .ard_inverse_traces(&cache)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: ARD traces: {e}"))?;

        // Build the flat step vector in `to_flat` layout:
        // [0]=log_lambda_sparse, [1]=log_lambda_smooth, then per-atom axes.
        let n_params = rho.to_flat().len();
        let mut steps = vec![0.0_f64; n_params];

        // λ_sparse (index 0): non-quadratic prior → no FS fixed point. Step 0.
        steps[0] = 0.0;

        // λ_smooth (index 1): Wood-Fasiolo EFS multiplicative update.
        let lambda_smooth = rho.lambda_smooth();
        let p_out = self.term.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.term.atoms {
            smooth_rank_total += SaeManifoldTerm::symmetric_rank(&atom.smooth_penalty)?;
        }
        let rank_total = p_out * (smooth_rank_total as f64);
        let quad = self.term.decoder_smoothness_quadratic_form();
        let eff_dof = self
            .term
            .decoder_smoothness_effective_dof(&cache, lambda_smooth)
            .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: smooth dof: {e}"))?;
        // λ_new = φ̂ · (penalty_rank − effective_dof) / penalty_energy. The
        // dispersion factor makes the update target the dimensionless effective
        // stiffness λ/φ̂ instead of an absolute output-unit penalty weight.
        // Guard the FS ratio against a vanishing penalty energy or a non-positive
        // numerator (which can occur transiently far from the optimum) by holding
        // λ_smooth fixed (step 0) — the cost path still moves it then.
        if quad > 0.0 && rank_total - eff_dof > 0.0 && lambda_smooth > 0.0 {
            let lambda_new = dispersion * (rank_total - eff_dof) / quad;
            if lambda_new.is_finite() && lambda_new > 0.0 {
                steps[1] = lambda_new.ln() - rho.log_lambda_smooth;
            }
        }

        // ARD axes (indices 2..): Mackay fixed point with posterior variance.
        let mut cursor = 2usize;
        for (k, axis_logard) in rho.log_ard.iter().enumerate() {
            let d = axis_logard.len();
            for j in 0..d {
                let denom = sumsq[k][j] + traces[k][j];
                if denom > 0.0 {
                    let alpha_new = dispersion * n_obs / denom;
                    if alpha_new.is_finite() && alpha_new > 0.0 {
                        steps[cursor + j] = alpha_new.ln() - axis_logard[j];
                    }
                }
            }
            cursor += d;
        }

        let beta_hat = self.term.flatten_beta();
        let cost = self.add_fit_data_collapse_penalty(cost)?;
        Ok(EfsEval {
            cost,
            steps,
            beta: Some(beta_hat),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
            logdet_enclosure_gap: None,
        })
    }
}

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        let plan = self.term.streaming_plan();
        let gradient = if plan.direct_admitted {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        };
        OuterCapability {
            // The full analytic outer-ρ gradient is assembled for every
            // assignment mode, including IBP-MAP (its empirical-π third channel
            // landed exactly in `logdet_theta_adjoint`, #1006).
            gradient,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // ρ are all penalty-like / τ coordinates: precisions and
            // log-smoothing strengths. No design-moving ψ coordinates.
            psi_dim: 0,
            // SAE's penalty coordinates are scale-coupled to the profiled
            // Gaussian reconstruction dispersion. The generic fixed-point lane
            // can drive the smoothness axis to the absolute upper boundary after
            // a good low-noise seed, collapsing the decoder to the mean (#1023).
            // Keep the analytic value/gradient lane in charge so the
            // dimensionless seed is not overwritten by an absolute-unit EFS step.
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Value-only comparison path (compass polls, EFS backtracking, seed
        // validation, FD certificate probes): no gradient/Hessian is ever
        // consumed at this iterate, so it takes the cheap probe refine budget
        // (#1029). Accepted points are always re-polished through
        // `eval`/`eval_with_order(ValueAndGradient|ValueGradientHessian)`
        // before any derivative consumption, and a probe value — when one is
        // returned at all — is converged to the same KKT/step tolerance as
        // the full-budget path, so all ranked comparisons stay in one measure.
        match self.evaluate_with_refine_policy(rho.view(), false) {
            Ok((cost, _beta)) => Ok(cost),
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => Ok(f64::INFINITY),
            Err(err) => Err(EstimationError::RemlOptimizationFailed(err)),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let rho_state = self.baseline_rho.from_flat(rho.view());
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term
                .set_flat_beta(beta.view())
                .map_err(EstimationError::RemlOptimizationFailed)?;
        }
        let (cost, loss, cache) = self
            .term
            .reml_criterion_with_cache(
                self.target.view(),
                &rho_state,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let solver = self
            .term
            .outer_gradient_arrow_solver(&cache)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let components = self
            .term
            .analytic_outer_rho_gradient_components(&rho_state, &loss, &cache, &solver)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let gradient = components.gradient_with_available_correction();
        self.current_rho = rho_state;
        self.last_loss = Some(loss);
        let beta_hat = self.term.flatten_beta();
        let cost = self
            .add_fit_data_collapse_penalty(cost)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        Ok(OuterEval {
            cost,
            gradient,
            hessian: HessianResult::Unavailable,
            inner_beta_hint: Some(beta_hat),
        })
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::Value => {
                let (cost, _beta_hat) = match self.evaluate_with_refine_policy(rho.view(), false) {
                    Ok(evaluated) => evaluated,
                    Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                        return Ok(OuterEval::infeasible(rho.len()));
                    }
                    Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
                };
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            }
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        self.efs_step(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn reset(&mut self) {
        self.term = self.baseline_term.clone();
        self.current_rho = self.baseline_rho.clone();
        self.last_loss = None;
        self.seeded_beta = None;
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Contract (see src/solver/reml/continuation.rs:727-737): an empty-β
        // seed means "no warm-start available, use your own cold default" and
        // MUST be accepted as a no-op. The continuation pre-warm forwards the
        // previous eval's `inner_beta_hint`, but before the first accepted eval
        // that hint is empty (`state.last_beta` starts empty). Rejecting it
        // fatally dropped every continuation seed and forced a full cold solve
        // on every outer seed — the slowness in gam#577. Only a *populated* β
        // must match the decoder dimension.
        if beta.is_empty() {
            // NoSlot is the documented continuation reply for "no usable seed;
            // proceed cold, no log" (outer_strategy.rs:1776). The real β slot
            // gets populated on the next accepted eval, which publishes
            // `inner_beta_hint`, so steps 2+ warm-start normally.
            return Ok(SeedOutcome::NoSlot);
        }
        if beta.len() != self.term.beta_dim() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "SaeManifoldOuterObjective::seed_inner_state: β length {} != decoder dim {}",
                beta.len(),
                self.term.beta_dim()
            )));
        }
        self.seeded_beta = Some(beta.clone());
        Ok(SeedOutcome::Installed)
    }

    /// The SAE-manifold joint fit enters through the heavy-smoothing
    /// [`crate::solver::continuation_path::ContinuationPath`] WHEN there is a
    /// combinatorial inter-atom routing active-set to protect: the joint
    /// `(logits, t, β)` block has a routing component that a cold solve at ρ*
    /// can collapse — but that failure class is specifically the **K ≥ 2**
    /// routing collapse (atoms competing for assignment mass). A single-atom
    /// (`K = 1`) fit has no inter-atom routing, so the coupled ρ / τ / isometry
    /// walk has nothing to prevent and is pure overhead: the cold direct cascade
    /// solves it directly (an order of magnitude faster on tiny fixtures). Gate
    /// the walk on `K ≥ 2`. When it returns `true` every seed routes through the
    /// homotopy walk (Object 1) and the seed cascade's structural-failure
    /// handling flips from REJECT to DEMOTE-WITH-REASON so the candidate set
    /// never empties on a structural diagnosis.
    fn requires_continuation_path_entry(&self) -> bool {
        // K >= 2: routing multimodality makes blind multistart hopeless — the
        // certified walk is the entry of record. K = 1 with a curved-capable
        // chart (duchon / euclidean patch): the Eckart-Young LINEAR optimum is
        // a genuine local minimum (a straight line through an arc), and cold
        // seeds converge INTO it — the walk exists precisely to track from
        // that anchor into the curved branch, and with the gauge-quotient
        // pivot invariant it arrives in a handful of legs, replacing the
        // 12-seed cascade outright. K = 1 periodic atoms keep the cascade:
        // their circular topology is baked into the basis, so the linear
        // basin is not an attractor for them and the walk buys nothing.
        if self.term.k_atoms() >= 2 {
            return true;
        }
        self.term.atoms.iter().any(|atom| {
            matches!(
                atom.basis_kind,
                SaeAtomBasisKind::Duchon | SaeAtomBasisKind::EuclideanPatch
            )
        })
    }

    /// The SAE-manifold objective has a certified anchor (#1007): its `η = 0`
    /// Eckart-Young linear relaxation is convex with a global optimum certified
    /// by [`linear_span_anchor`]. Run the predictor-corrector `η`-walk from that
    /// anchor before blind multistart. On arrival the inner state is warm
    /// at the certified `η = 1` solution for the active seed; on a
    /// degenerate anchor or a detected bifurcation the term is left at the full
    /// basis (`η = 1`) and the documented cascade takes over — the outcome is
    /// recorded on the fit payload either way.
    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        let rho_state = self.baseline_rho.from_flat(rho.view());
        Some(
            self.run_curvature_homotopy_entry_at_rho(&rho_state)
                .map_err(EstimationError::RemlOptimizationFailed),
        )
    }
}

fn sae_manifold_newton_directional_decrease(
    sys: &ArrowSchurSystem,
    delta_ext_coord: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    // delta_ext_coord has variable-stride layout for heterogeneous systems.
    assert_eq!(delta_ext_coord.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut gradient_dot_step = 0.0;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let row_base = sys.row_offsets[row_idx];
        let di = sys.row_dims[row_idx];
        for axis in 0..di {
            gradient_dot_step += row.gt[axis] * delta_ext_coord[row_base + axis];
        }
    }
    for idx in 0..sys.k {
        gradient_dot_step += sys.gb[idx] * delta_beta[idx];
    }
    -gradient_dot_step
}

/// Per-atom decoder-smoothness GEMM `S_k · B_k`, batched across ALL GPUs.
///
/// Every atom contributes one dense product of its `(m_k × m_k)` smoothness
/// penalty `S_k` with its `(m_k × p)` decoder coefficients `B_k`. These products
/// are independent across atoms, so the per-atom axis is the natural batch /
/// device-fan-out dimension. This helper:
///
///   * groups atoms by identical `(m_k, p)` shape (the strided-batched cuBLAS
///     GEMM requires a uniform tile),
///   * for each group with ≥ 2 atoms whose aggregate flop count clears the
///     dispatch threshold, partitions the group's atoms across every available
///     device with [`crate::gpu::pool::scatter_batched`] and runs one
///     `try_fast_abt_strided_batched` per device tile (computing
///     `S_k · B_k = S_k · (B_kᵀ)ᵀ`),
///   * falls back, atom-by-atom, to the exact ndarray `S_k.dot(B_k)` whenever no
///     GPU runtime is present, the pool returns `None`, or a tile's batched GEMM
///     declines. The result is bit-for-bit identical to the all-CPU path (f64
///     throughout, same accumulation order per product).
///
/// Returns one `S_k · B_k` matrix per atom, in atom order. `symmetrize`
/// pre-symmetrises each `S_k` (the assembly path needs `½(S+Sᵀ)`); the value /
/// quadratic-form callers pass `false` since the quadratic form only sees the
/// symmetric part regardless.
fn batched_smooth_sb(
    sb_inputs: &[(ArrayView2<'_, f64>, ArrayView2<'_, f64>)],
    symmetrize: bool,
) -> Vec<Array2<f64>> {
    let n_atoms = sb_inputs.len();
    // Materialise the (optionally symmetrised) S factors once; the GPU tile and
    // the CPU fallback both read these, so a single pass keeps the two routes
    // numerically identical.
    let s_mats: Vec<Array2<f64>> = sb_inputs
        .iter()
        .map(|(s, _)| {
            if symmetrize {
                let m = s.nrows();
                let mut sym = Array2::<f64>::zeros((m, m));
                for i in 0..m {
                    for j in 0..m {
                        sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
                    }
                }
                sym
            } else {
                s.to_owned()
            }
        })
        .collect();

    // Exact CPU fallback for a single atom, reused by both the no-GPU route and
    // per-tile decline.
    let cpu_one = |idx: usize| -> Array2<f64> { s_mats[idx].dot(&sb_inputs[idx].1) };

    let rt = match crate::gpu::runtime::GpuRuntime::global() {
        Some(rt) => rt,
        None => return (0..n_atoms).map(cpu_one).collect(),
    };

    // Group atom indices by uniform (m, p) shape; only same-shape groups can ride
    // a strided-batched GEMM tile.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, (_, b)) in sb_inputs.iter().enumerate() {
        let m = s_mats[idx].nrows();
        let p = b.ncols();
        groups.entry((m, p)).or_default().push(idx);
    }

    let mut out: Vec<Option<Array2<f64>>> = (0..n_atoms).map(|_| None).collect();
    for ((m, p), members) in groups {
        // Singletons and tiny groups gain nothing from batched device launch;
        // the single-product `fast_*` shim (size-gated) already handles a large
        // lone GEMM, so route those straight through the CPU-or-shim helper.
        if members.len() < 2 || m == 0 || p == 0 {
            for &idx in &members {
                out[idx] = Some(cpu_one(idx));
            }
            continue;
        }
        // Build the per-tile batched inputs lazily inside the device closure so
        // each device only packs the atoms it owns. `items` carries the member
        // atom indices; `scatter_batched` slices it per device ordinal.
        let mut items: Vec<usize> = members.clone();
        let s_ref = &s_mats;
        // Collect per-tile results into a side channel keyed by atom index, then
        // splice them in after scatter completes (scatter's closure borrows
        // `items` immutably-per-tile and must stay `Sync`).
        let tile_results: std::sync::Mutex<Vec<(usize, Array2<f64>)>> =
            std::sync::Mutex::new(Vec::with_capacity(members.len()));
        let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_ordinal, slice| {
            if slice.is_empty() {
                return Some(());
            }
            let batch = slice.len();
            // A = stacked S_k  (batch, m, m); B = stacked B_kᵀ (batch, p, m) so
            // that `A · Bᵀ` per tile yields `S_k · B_k` (batch, m, p).
            let mut a = Array3::<f64>::zeros((batch, m, m));
            let mut bt = Array3::<f64>::zeros((batch, p, m));
            for (t, &idx) in slice.iter().enumerate() {
                let s = &s_ref[idx];
                let b = &sb_inputs[idx].1;
                for i in 0..m {
                    for j in 0..m {
                        a[[t, i, j]] = s[[i, j]];
                    }
                }
                for i in 0..p {
                    for j in 0..m {
                        bt[[t, i, j]] = b[[j, i]];
                    }
                }
            }
            let prod = crate::gpu::try_fast_abt_strided_batched(a.view(), bt.view())?;
            let mut sink = tile_results.lock().expect("tile_results mutex poisoned");
            for (t, &idx) in slice.iter().enumerate() {
                sink.push((idx, prod.slice(s![t, .., ..]).to_owned()));
            }
            Some(())
        });
        // The scatter closure has returned, so all borrows of `items`/`s_mats`/
        // `tile_results` are released; write the results back into `out`.
        match ok {
            Some(()) => {
                let sink = tile_results
                    .into_inner()
                    .expect("tile_results mutex poisoned");
                for (idx, mat) in sink {
                    out[idx] = Some(mat);
                }
                // Any member a tile silently skipped (cannot happen with the
                // contract, but keep the result total) falls back to CPU.
                for &idx in &members {
                    if out[idx].is_none() {
                        out[idx] = Some(cpu_one(idx));
                    }
                }
            }
            None => {
                for &idx in &members {
                    out[idx] = Some(cpu_one(idx));
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(idx, slot)| slot.unwrap_or_else(|| cpu_one(idx)))
        .collect()
}

/// A detected bifurcation on the curvature-homotopy branch (#1007): the arrow
/// factor's smallest Cholesky pivot collapsed below the safe-SPD tolerance at a
/// homotopy parameter `η`, so the optimal branch the tracker was following lost
/// strict positive-definiteness. Recorded on [`CurvatureWalkReport`] and never
/// silent — the walk returns control to the documented multi-seed cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureBifurcation {
    /// Homotopy parameter at which the pivot collapsed.
    pub eta: f64,
    /// The smallest arrow-factor pivot observed at `eta` (Hessian-scale, i.e.
    /// squared lower-Cholesky diagonal); below the safe-SPD floor.
    pub min_pivot: f64,
}

/// Outcome of one certified curvature-homotopy entry walk (#1007).
///
/// The tracker walks the basis curvature dial `η` from the Eckart-Young anchor
/// (`η = 0`, global by construction) to the full curved basis (`η = 1`),
/// predictor-corrector style, holding the per-pivot positivity invariant. This
/// report makes the outcome observable on the fit payload: `arrived` says the
/// walk reached `η = 1` on the certified branch; `bifurcation` records the first
/// detected pivot collapse (if any); `collapse_events` mirrors the inner active
/// -mass guard's verdict at the arrival state; `eta_steps` / `step_halvings`
/// are the walk's cost. A walk that did not arrive (degenerate anchor or a
/// recorded bifurcation) hands control back to the multi-seed cascade.
#[derive(Debug, Clone)]
pub struct CurvatureWalkReport {
    /// Whether the walk reached `η = 1` on the certified optimal branch.
    pub arrived: bool,
    /// Eckart-Young anchor residual energy at `η = 0` (the certificate the
    /// linear relaxation is solved to).
    pub anchor_residual_norm_sq: f64,
    /// First detected branch bifurcation (pivot collapse), or `None` when the
    /// pivot stayed strictly positive across the whole walk.
    pub bifurcation: Option<CurvatureBifurcation>,
    /// Number of accepted `η` waypoints (anchor → 1).
    pub eta_steps: usize,
    /// Number of `η`-step halvings forced by a shrinking min-pivot.
    pub step_halvings: usize,
    /// Number of inner active-mass collapse events recorded at the arrival
    /// state (the same `#976` guard ledger the cascade reads); a clean walk
    /// arrives with this empty.
    pub collapse_events: usize,
    /// Number of scaffold re-seeds the walk itself triggered. A certified walk
    /// from the global anchor reaches `η = 1` with zero reseeds.
    pub reseeds: usize,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAtomAnchor {
    pub gate_weight: f64,
    pub frame: GrassmannFrame,
    pub decoder_coordinates: Array2<f64>,
    pub singular_values: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAnchor {
    pub atoms: Vec<LinearSpanAtomAnchor>,
    pub reconstruction: Array2<f64>,
    pub residual_norm_sq: f64,
}

/// Curvature-homotopy linear-span anchor at `eta = 0`.
///
/// This stage-1 primitive solves the neutral-gate linear relaxation by
/// sequential Eckart-Young residual SVDs and canonicalizes every recovered output
/// subspace through the same [`GrassmannFrame`] gauge used by the #972 frame
/// machinery. It does not mutate `term` or replace the existing seed cascade.
pub fn linear_span_anchor(
    term: &SaeManifoldTerm,
    targets: ArrayView2<'_, f64>,
) -> Result<LinearSpanAnchor, String> {
    let n = term.n_obs();
    let p = term.output_dim();
    if targets.dim() != (n, p) {
        return Err(format!(
            "linear_span_anchor: targets shape {:?} != ({n}, {p})",
            targets.dim()
        ));
    }
    if term.k_atoms() == 0 {
        return Err("linear_span_anchor: term must contain at least one atom".into());
    }
    if !targets.iter().all(|v| v.is_finite()) {
        return Err("linear_span_anchor: targets must be finite".into());
    }
    let gates = neutral_gate_weights(term.assignment.mode, term.k_atoms());
    let mut residual = targets.to_owned();
    let mut reconstruction = Array2::<f64>::zeros((n, p));
    let mut atoms = Vec::with_capacity(term.k_atoms());
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let gate = gates[atom_idx];
        if !(gate.is_finite() && gate > 0.0) {
            return Err(format!(
                "linear_span_anchor: neutral gate for atom {atom_idx} must be positive finite; got {gate}"
            ));
        }
        let requested_rank = atom.basis_size().min(n).min(p);
        if requested_rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} has no recoverable linear span rank"
            ));
        }
        let weighted = residual.mapv(|v| gate * v);
        let (_u_opt, singular_values_full, vt_opt) = weighted
            .svd(false, true)
            .map_err(|err| format!("linear_span_anchor: SVD failed for atom {atom_idx}: {err}"))?;
        let vt = vt_opt.ok_or_else(|| {
            format!("linear_span_anchor: SVD returned no right factor for atom {atom_idx}")
        })?;
        let rank = requested_rank
            .min(vt.nrows())
            .min(singular_values_full.len());
        if rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} SVD returned rank zero"
            ));
        }
        let mut frame = Array2::<f64>::zeros((p, rank));
        for col in 0..rank {
            for row in 0..p {
                frame[[row, col]] = vt[[col, row]];
            }
        }
        let singular_values = singular_values_full.slice(s![..rank]).to_owned();
        let frame = GrassmannFrame::from_oriented(frame, singular_values.clone());
        let frame_matrix = frame.frame().to_owned();
        let mut coordinates = residual.dot(&frame_matrix);
        coordinates.mapv_inplace(|v| v / gate);
        let contribution = fast_abt(&coordinates, &frame_matrix).mapv(|v| gate * v);
        reconstruction += &contribution;
        residual -= &contribution;
        atoms.push(LinearSpanAtomAnchor {
            gate_weight: gate,
            frame,
            decoder_coordinates: coordinates,
            singular_values,
        });
    }
    let residual_norm_sq = residual.iter().map(|v| v * v).sum();
    Ok(LinearSpanAnchor {
        atoms,
        reconstruction,
        residual_norm_sq,
    })
}

fn sae_cholesky_solve_neg_gradient(
    h: ArrayView2<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let n = h.nrows();
    if h.ncols() != n || g.len() != n {
        return Err(format!(
            "sae_cholesky_solve_neg_gradient: shape mismatch H={:?}, g={}",
            h.dim(),
            g.len()
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(format!("non-positive Cholesky pivot at {i}: {sum}"));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("sae_cholesky_solve_neg_gradient: non-finite solution".into());
    }
    Ok(x)
}

fn solve_basis_transport(
    new_phi: ArrayView2<'_, f64>,
    old_phi: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    solve_design_least_squares(new_phi, old_phi)
}

fn transport_smooth_penalty_for_decoder(
    decoder_transport: ArrayView2<'_, f64>,
    old_smooth_penalty: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let m = decoder_transport.nrows();
    if decoder_transport.ncols() != m {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: decoder transport must be square; got {:?}",
            decoder_transport.dim()
        ));
    }
    if old_smooth_penalty.dim() != (m, m) {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: smooth penalty shape {:?} != ({m}, {m})",
            old_smooth_penalty.dim()
        ));
    }
    let transport_inverse =
        solve_design_least_squares(decoder_transport, Array2::<f64>::eye(m).view())?;
    Ok(fast_atb(
        &transport_inverse,
        &fast_ab(&old_smooth_penalty.to_owned(), &transport_inverse),
    ))
}

pub(crate) fn solve_design_least_squares(
    design: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != rhs.nrows() {
        return Err(format!(
            "solve_design_least_squares: row mismatch design={} rhs={}",
            design.nrows(),
            rhs.nrows()
        ));
    }
    let (u_opt, sigma, vt_opt) = design
        .to_owned()
        .svd(true, true)
        .map_err(|err| format!("solve_design_least_squares: SVD failed: {err}"))?;
    let u = u_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted Vt".to_string())?;
    let smax = sigma.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if !(smax.is_finite() && smax > 0.0) {
        return Err("solve_design_least_squares: design has zero numerical rank".to_string());
    }
    let cutoff = smax * f64::EPSILON * (design.nrows().max(design.ncols()) as f64);
    let coeffs = u.t().dot(&rhs);
    let mut scaled = Array2::<f64>::zeros(coeffs.dim());
    for row in 0..sigma.len() {
        if sigma[row] > cutoff {
            let inv = 1.0 / sigma[row];
            for col in 0..rhs.ncols() {
                scaled[[row, col]] = inv * coeffs[[row, col]];
            }
        }
    }
    Ok(vt.t().dot(&scaled))
}

fn sae_coord_penalty_offset(
    row_layout: Option<&SaeRowLayout>,
    dense_off: usize,
    row: usize,
    atom_idx: usize,
) -> Option<usize> {
    match row_layout {
        Some(layout) => {
            let active = &layout.active_atoms[row];
            let starts = &layout.coord_starts[row];
            active
                .iter()
                .zip(starts.iter())
                .find_map(|(&active_atom, &coord_start)| {
                    if active_atom == atom_idx {
                        Some(coord_start)
                    } else {
                        None
                    }
                })
        }
        None => Some(dense_off),
    }
}

fn sae_penalty_is_row_block_supported(penalty: &AnalyticPenaltyKind) -> bool {
    matches!(
        penalty,
        AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::RowPrecisionPrior(_)
            | AnalyticPenaltyKind::ParametricRowPrecisionPrior(_)
            | AnalyticPenaltyKind::ScadMcp(_)
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::Isometry(_)
    )
}

/// Whether a row-block coordinate penalty is an *origin-anchored, axis-separable
/// magnitude shrinkage* — its energy is `Σ_axis Σ_row f(|t|)` with a fixed zero,
/// evaluated independently per flat entry.
///
/// Such a penalty is only well-posed on a **Euclidean** chart axis, which has a
/// distinguished origin. On a **periodic** chart axis (a `Circle`/`Torus`
/// coordinate) the latent is a homogeneous angle defined only modulo its period:
/// there is no rotation-invariant "zero" to shrink toward, and the raw `|t|` is
/// *discontinuous across the retraction branch cut* (a coordinate just below the
/// period wraps to just above zero). Folding such an energy into the joint
/// Newton objective makes the line-search value jump by an `O(1)` amount under a
/// near-zero coordinate step, so Armijo rejects otherwise-valid steps and the
/// inner solve never reaches stationarity (issue #795; the same failure mode the
/// ARD prior avoids by switching to the periodic von-Mises energy on these axes).
///
/// For these penalties [`sae_coord_penalty_euclidean_restriction`] restricts the
/// energy to the Euclidean axes, where it is both meaningful and continuous;
/// periodic axes contribute nothing. The restriction is exact only because the
/// energy is axis-separable, so this matcher is deliberately narrow: non-separable
/// shrinkage (e.g. the Hoyer ℓ¹/ℓ² ratio) is excluded.
fn sae_coord_penalty_is_origin_anchored_magnitude(penalty: &AnalyticPenaltyKind) -> bool {
    matches!(penalty, AnalyticPenaltyKind::ScadMcp(_))
}

/// Restrict an origin-anchored, axis-separable coordinate shrinkage penalty (see
/// [`sae_coord_penalty_is_origin_anchored_magnitude`]) to the **Euclidean**
/// (non-periodic) axes of a latent coordinate block.
///
/// Returns `Some((euclidean_axes, compacted_target))` where `compacted_target` is
/// the row-major `(n_obs × euclidean_axes.len())` flat vector holding only the
/// Euclidean-axis coordinates, in the axis order given by `euclidean_axes`. The
/// caller evaluates the (axis-separable) penalty on this compacted target and
/// scatters its per-entry gradient / curvature back to the Euclidean axis slots,
/// leaving every periodic axis untouched (zero contribution). Because the penalty
/// is a sum of independent per-entry terms, evaluating it on the compacted target
/// is *exactly* the full energy with the periodic axes dropped — value, gradient,
/// and curvature stay mutually consistent.
///
/// Returns `None` when every axis is Euclidean: there is nothing to restrict, so
/// the caller uses the full target unchanged (zero overhead on the common path).
/// When every axis is periodic the returned `euclidean_axes` is empty and the
/// compacted target has length zero, so the penalty contributes nothing at all.
fn sae_coord_penalty_euclidean_restriction(
    coord: &LatentCoordValues,
) -> Option<(Vec<usize>, Array1<f64>)> {
    let periods = coord.effective_axis_periods();
    let d = periods.len();
    let euclidean_axes: Vec<usize> = (0..d).filter(|&axis| periods[axis].is_none()).collect();
    if euclidean_axes.len() == d {
        return None;
    }
    let n = coord.n_obs();
    let de = euclidean_axes.len();
    let flat = coord.as_flat();
    let mut compacted = Array1::<f64>::zeros(n * de);
    for row in 0..n {
        for (j, &axis) in euclidean_axes.iter().enumerate() {
            compacted[row * de + j] = flat[row * d + axis];
        }
    }
    Some((euclidean_axes, compacted))
}

/// The JSON descriptor `kind` strings for the SAE row-block analytic penalties
/// this build supports (i.e. those `sae_penalty_is_row_block_supported`
/// accepts). Co-located with that matcher so the two cannot drift. The FFI
/// `build_info` advertises this list so the Python wrapper can detect a stale
/// extension that predates a given penalty and raise a clear `NotImplementedError`
/// instead of forwarding a descriptor the binary will reject with a cryptic
/// Schur error (issue #338).
pub fn sae_row_block_penalty_kinds() -> &'static [&'static str] {
    &[
        "ard",
        "top_k_activation",
        "jumprelu",
        "sparsity",
        "row_precision_prior",
        "parametric_row_precision_prior",
        "scad_mcp",
        "block_orthogonality",
        "isometry",
    ]
}

/// Helper for padded FFI callers. Arrays use `(K, N, M_max)` and
/// `(K, N, M_max, D_max)` storage, with `basis_sizes` and `latent_dims`
/// selecting each atom's active prefix.
///
/// `evaluators`, when non-empty, must have length `K`. Each entry attaches an
/// optional [`SaeBasisEvaluator`] to the matching atom so the Rust Newton
/// loop can refresh `Phi`/`dPhi/dt` between iterations without rebuilding the
/// term from Python. An empty slice leaves every atom in snapshot-only mode.
#[must_use = "build error must be handled"]
pub fn term_from_padded_blocks_with_mode(
    n_obs: usize,
    p_out: usize,
    basis_kinds: &[SaeAtomBasisKind],
    basis_values: ArrayView3<'_, f64>,
    basis_jacobian: ArrayView4<'_, f64>,
    basis_sizes: &[usize],
    latent_dims: &[usize],
    decoder_coefficients: ArrayView3<'_, f64>,
    smooth_penalties: ArrayView3<'_, f64>,
    logits: ArrayView2<'_, f64>,
    coords: &[Array2<f64>],
    mode: AssignmentMode,
    evaluators: &[Option<Arc<dyn SaeBasisEvaluator>>],
) -> Result<SaeManifoldTerm, String> {
    let k_atoms = basis_sizes.len();
    if latent_dims.len() != k_atoms || basis_kinds.len() != k_atoms || coords.len() != k_atoms {
        return Err("term_from_padded_blocks: K-length metadata mismatch".into());
    }
    if !evaluators.is_empty() && evaluators.len() != k_atoms {
        return Err(format!(
            "term_from_padded_blocks: evaluators length {} must equal K={k_atoms} or be empty",
            evaluators.len()
        ));
    }
    if logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "term_from_padded_blocks: logits must be ({n_obs}, {k_atoms}); got {:?}",
            logits.dim()
        ));
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let m = basis_sizes[k];
        let d = latent_dims[k];
        let phi = basis_values.slice(s![k, 0..n_obs, 0..m]).to_owned();
        let jet = basis_jacobian.slice(s![k, 0..n_obs, 0..m, 0..d]).to_owned();
        let b = decoder_coefficients.slice(s![k, 0..m, 0..p_out]).to_owned();
        let s = smooth_penalties.slice(s![k, 0..m, 0..m]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!("atom_{k}"),
            basis_kinds[k].clone(),
            d,
            phi,
            jet,
            b,
            s,
        )?;
        let atom = match evaluators.get(k).and_then(|slot| slot.clone()) {
            Some(evaluator) => atom.with_basis_evaluator(evaluator),
            None => atom,
        };
        atoms.push(atom);
    }
    let manifolds = basis_kinds
        .iter()
        .zip(latent_dims.iter().copied())
        .map(|(kind, d)| kind.latent_manifold(d))
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.to_owned(),
        coords.to_vec(),
        manifolds,
        mode,
    )?;
    SaeManifoldTerm::new(atoms, assignment)
}

/// Build the per-row Jacobian `J` and Hessian `H` of the decoded output
/// `Z_n = Phi_n B` with respect to the latent coordinates `t_n` of a single
/// SAE atom and install them on the supplied [`IsometryPenalty`].
///
/// Layout follows the convention used by [`IsometryPenalty::grad_target`] and
/// friends:
///
/// * `J ∈ ℝ^{n_obs × (p · d)}`, flattened as `J[n, i*d + a]` —
///   `J[n, i, a] = ∂Z_{n,i} / ∂t_{n,a} = Σ_m dPhi[n, m, a] · B[m, i]`.
/// * `H ∈ ℝ^{n_obs × (p · d · d)}`, flattened as `H[n, (i*d + a)*d + c]` —
///   `H[n, i, a, c] = ∂J[n, i, a] / ∂t_{n, c} = Σ_m d²Phi[n, m, a, c] · B[m, i]`.
/// * `K`, an `Array3` of shape `(n_obs, p, d·d·d)` with last axis packed
///   `((a·d + c)·d + e)` — `K[n, i, a, c, e] = ∂³Z_{n,i} / ∂t_a ∂t_c ∂t_e =
///   Σ_m d³Phi[n, m, a, c, e] · B[m, i]`. Installed via the new third-jet slot
///   whenever the base evaluator's `third_jet_dyn` yields a jet AND the penalty
///   carries no `duchon_radial_source`. This is the residual-curvature source
///   for the exact isometry `hvp`.
///
/// Returns `Ok(true)` when both caches were installed (i.e. the atom was
/// built via [`SaeManifoldAtom::with_basis_second_jet`], so its
/// `basis_second_jet` slot holds a [`SaeBasisSecondJet`] implementation
/// that supplies the analytic Hessian). Returns `Ok(false)` when only the
/// base [`SaeBasisEvaluator`] is installed (no second jet available) — in
/// that case only the first-jet `jacobian_cache` is installed and the
/// penalty's `has_jacobian_second_source` check still has a chance to
/// succeed via a pre-supplied `duchon_radial_source`. Returns `Err` on
/// shape mismatches (which would indicate a buggy evaluator) or when the
/// second-jet implementation itself fails (e.g. wrong latent dimension).
///
/// This entry point takes `&IsometryPenalty` rather than `&mut` because the
/// caches are interior-mutable (see [`IsometryPenalty::refresh_caches`]).
pub fn refresh_isometry_caches_from_atom(
    penalty: &IsometryPenalty,
    atom: &SaeManifoldAtom,
    coords: ArrayView2<'_, f64>,
) -> Result<bool, String> {
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!(
            "refresh_isometry_caches_from_atom: atom {} has no basis evaluator",
            atom.name
        )
    })?;
    let (_phi, jet) = evaluator.evaluate(coords)?;

    let n_obs = coords.nrows();
    let d = atom.latent_dim;
    let m = atom.basis_size();
    let p = atom.decoder_coefficients.ncols();
    if penalty.p_out != p {
        return Err(format!(
            "refresh_isometry_caches_from_atom: penalty.p_out={} but atom.decoder.cols={p}",
            penalty.p_out
        ));
    }
    if jet.dim() != (n_obs, m, d) {
        return Err(format!(
            "refresh_isometry_caches_from_atom: evaluator first jet has shape {:?}, expected ({n_obs}, {m}, {d})",
            jet.dim()
        ));
    }

    // J[n, i*d + a] = Σ_m dPhi[n, m, a] · B[m, i].
    let b = &atom.decoder_coefficients;
    let mut jac = Array2::<f64>::zeros((n_obs, p * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                let mut acc = 0.0;
                for mm in 0..m {
                    acc += jet[[n, mm, a]] * b[[mm, i]];
                }
                jac[[n, i * d + a]] = acc;
            }
        }
    }

    // The second jet is sourced from the optional `basis_second_jet`
    // slot. The trait split (`SaeBasisEvaluator` vs `SaeBasisSecondJet`)
    // encodes "no closed-form Hessian" as trait absence: when the atom
    // was built with `with_basis_evaluator` (base trait only) the slot
    // is `None` and the `H` cache is not installed. When the atom was
    // built with `with_basis_second_jet` the slot holds the same Arc
    // upcast to the supertrait, and `second_jet` returns the analytic
    // Hessian here.
    let jac2_opt = if let Some(second_eval) = atom.basis_second_jet.as_ref() {
        let hess = second_eval.second_jet(coords)?;
        if hess.dim() != (n_obs, m, d, d) {
            return Err(format!(
                "refresh_isometry_caches_from_atom: evaluator second jet has shape {:?}, expected ({n_obs}, {m}, {d}, {d})",
                hess.dim()
            ));
        }
        let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
        for n in 0..n_obs {
            for i in 0..p {
                for a in 0..d {
                    for c in 0..d {
                        let mut acc = 0.0;
                        for mm in 0..m {
                            acc += hess[[n, mm, a, c]] * b[[mm, i]];
                        }
                        jac2[[n, (i * d + a) * d + c]] = acc;
                    }
                }
            }
        }
        Some(Arc::new(jac2))
    } else {
        None
    };

    // Third jet K[n, i, ((a·d + c)·d + e)] = Σ_m d³Phi[n, m, a, c, e] · B[m, i]
    // feeds the residual-curvature term of the exact isometry Hessian
    //   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
    //             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd}.
    // Sourced from the base evaluator's object-safe `third_jet_dyn` forwarder
    // (closed-form analytic override for every basis with an analytic Hessian:
    // sphere/circle/torus/affine/euclidean/duchon; `None` otherwise — no
    // finite-difference fallback). Installed only when the penalty
    // has no `duchon_radial_source` — a Duchon penalty already carries its own
    // analytic third source and `jacobian_third` would shadow it with this
    // cache. Always written (Some or None) so a stale K from a prior outer step
    // never survives a refresh.
    let jac3_opt = if penalty.duchon_radial_source.is_none() {
        match evaluator.third_jet_dyn(coords) {
            Some(third) => {
                let t3 = third?;
                if t3.dim() != (n_obs, m, d, d, d) {
                    return Err(format!(
                        "refresh_isometry_caches_from_atom: evaluator third jet has shape {:?}, expected ({n_obs}, {m}, {d}, {d}, {d})",
                        t3.dim()
                    ));
                }
                let mut jac3 = Array3::<f64>::zeros((n_obs, p, d * d * d));
                for n in 0..n_obs {
                    for i in 0..p {
                        for a in 0..d {
                            for c in 0..d {
                                for e in 0..d {
                                    let mut acc = 0.0;
                                    for mm in 0..m {
                                        acc += t3[[n, mm, a, c, e]] * b[[mm, i]];
                                    }
                                    jac3[[n, i, ((a * d) + c) * d + e]] = acc;
                                }
                            }
                        }
                    }
                }
                Some(Arc::new(jac3))
            }
            None => None,
        }
    } else {
        None
    };

    let installed = jac2_opt.is_some();
    penalty.refresh_caches(Some(Arc::new(jac)), jac2_opt);
    penalty.set_third_decoder_derivative(jac3_opt);
    Ok(installed)
}

/// Walk an [`AnalyticPenaltyRegistry`] and refresh every Isometry penalty
/// against the SAE atom it owns. The alignment rule is positional within each
/// `(latent_dim, p_out)` signature: the penalty's `target.latent_dim` must
/// equal the atom's `latent_dim` AND the penalty's `p_out` must equal the
/// atom's decoder column count `p`. Multi-atom configurations install one
/// isometry penalty per atom, so the *k*-th isometry penalty matching a given
/// signature is paired with the *k*-th atom matching that same signature. This
/// reduces to the unambiguous single-atom/single-penalty case wired by
/// `solver/workflow.rs`, and never collapses multiple penalties onto the first
/// matching atom (which would leave every later atom's coords un-refreshed).
///
/// Returns the number of penalties that got both caches populated (i.e. the
/// number of atoms whose `basis_second_jet` slot holds a
/// [`SaeBasisSecondJet`] implementation supplying the analytic Hessian).
pub fn refresh_isometry_caches_from_term(
    registry: &AnalyticPenaltyRegistry,
    term: &SaeManifoldTerm,
    coords_per_atom: &[Array2<f64>],
) -> Result<usize, String> {
    if coords_per_atom.len() != term.atoms.len() {
        return Err(format!(
            "refresh_isometry_caches_from_term: coords_per_atom length {} != number of atoms {}",
            coords_per_atom.len(),
            term.atoms.len()
        ));
    }
    let mut refreshed_with_second = 0usize;
    // Per-signature cursor: how many atoms matching a given (latent_dim, p_out)
    // have already been consumed by earlier isometry penalties. Pairing the
    // k-th penalty of a signature with the k-th atom of that signature gives a
    // stable one-to-one mapping for multi-atom configs.
    let mut consumed_per_signature: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for entry in registry.penalties.iter() {
        let AnalyticPenaltyKind::Isometry(p) = entry else {
            continue;
        };
        let Some(p_latent_dim) = p.target.latent_dim else {
            continue;
        };
        let signature = (p_latent_dim, p.p_out);
        let already_consumed = consumed_per_signature.entry(signature).or_insert(0);
        // Advance to the (already_consumed)-th atom matching this signature.
        let mut seen = 0usize;
        let mut paired: Option<usize> = None;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let matches = atom.latent_dim == p_latent_dim
                && atom.decoder_coefficients.ncols() == p.p_out
                && atom.basis_evaluator.is_some();
            if !matches {
                continue;
            }
            if seen == *already_consumed {
                paired = Some(atom_idx);
                break;
            }
            seen += 1;
        }
        let Some(atom_idx) = paired else {
            continue;
        };
        *already_consumed += 1;
        let atom = &term.atoms[atom_idx];
        let coords = coords_per_atom[atom_idx].view();
        if refresh_isometry_caches_from_atom(p, atom, coords)? {
            refreshed_with_second += 1;
        }
    }
    Ok(refreshed_with_second)
}

#[cfg(test)]
mod tests {
    use crate::linalg::faer_ndarray::fast_ata;

    use super::*;
    use crate::solver::arrow_schur::{
        ArrowFactorSlab, ArrowHtbetaCache, ArrowSolverMode, ArrowUndampedFactors, PcgDiagnostics,
    };
    use crate::terms::analytic_penalties::ARDPenalty;
    use crate::terms::analytic_penalties::IsometryReference;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array5, array};

    fn assert_matrix_same_bits(left: &Array2<f64>, right: &Array2<f64>) {
        assert_eq!(left.dim(), right.dim());
        for ((row, col), &value) in left.indexed_iter() {
            assert_eq!(
                value.to_bits(),
                right[[row, col]].to_bits(),
                "matrix bits differ at ({row}, {col})"
            );
        }
    }

    fn assert_tensor3_same_bits(left: &Array3<f64>, right: &Array3<f64>) {
        assert_eq!(left.dim(), right.dim());
        for ((row, col, axis), &value) in left.indexed_iter() {
            assert_eq!(
                value.to_bits(),
                right[[row, col, axis]].to_bits(),
                "tensor bits differ at ({row}, {col}, {axis})"
            );
        }
    }

    fn assert_eta_one_parity(
        evaluator: &dyn SaeBasisEvaluator,
        coords: ArrayView2<'_, f64>,
        expected_curved: usize,
    ) {
        let (phi, jet) = evaluator.evaluate(coords).expect("base evaluate");
        let eta = evaluator
            .evaluate_phi_eta(coords, 1.0)
            .expect("eta evaluate");
        assert_matrix_same_bits(&eta.phi, &phi);
        assert_tensor3_same_bits(&eta.jet, &jet);
        assert_eq!(eta.split.curved_cols.len(), expected_curved);
        for &col in &eta.split.linear_cols {
            for row in 0..phi.nrows() {
                assert_eq!(eta.dphi_deta[[row, col]], 0.0);
                for axis in 0..jet.shape()[2] {
                    assert_eq!(eta.djet_deta[[row, col, axis]], 0.0);
                }
            }
        }
        for &col in &eta.split.curved_cols {
            for row in 0..phi.nrows() {
                assert_eq!(
                    eta.dphi_deta[[row, col]].to_bits(),
                    phi[[row, col]].to_bits()
                );
                for axis in 0..jet.shape()[2] {
                    assert_eq!(
                        eta.djet_deta[[row, col, axis]].to_bits(),
                        jet[[row, col, axis]].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn phi_eta_one_reproduces_current_atom_bases_bit_for_bit() {
        let periodic_coords = array![[0.0_f64], [0.125], [0.4]];
        let periodic = PeriodicHarmonicEvaluator::new(7).unwrap();
        assert_eta_one_parity(&periodic, periodic_coords.view(), 4);

        let raw_circle_coords = array![[0.0_f64], [0.3], [1.1]];
        let raw_circle = RawPeriodicCircleEvaluator::new(1).unwrap();
        assert_eta_one_parity(&raw_circle, raw_circle_coords.view(), 0);

        let torus_coords = array![[0.0_f64, 0.2], [0.25, 0.5], [0.7, 0.9]];
        let torus = TorusHarmonicEvaluator::new(2, 2).unwrap();
        assert_eta_one_parity(&torus, torus_coords.view(), 20);

        let sphere_coords = array![[0.0_f64, 0.0], [0.3, 0.4], [-0.2, 1.1]];
        let sphere = SphereChartEvaluator;
        assert_eta_one_parity(&sphere, sphere_coords.view(), 3);

        let centers = array![
            [-1.0_f64, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.5, -0.25]
        ];
        let duchon_coords = array![[0.1_f64, 0.2], [0.4, -0.3], [-0.2, 0.7]];
        let duchon = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let (duchon_phi, _) = duchon.evaluate(duchon_coords.view()).unwrap();
        let duchon_poly = 3usize;
        assert_eta_one_parity(
            &duchon,
            duchon_coords.view(),
            duchon_phi.ncols() - duchon_poly,
        );

        let euclidean = EuclideanPatchEvaluator::new(2, 3).unwrap();
        let total_cols = crate::basis::monomial_exponents(2, 3).len();
        let linear_cols = crate::basis::monomial_exponents(2, 3)
            .iter()
            .filter(|alpha| alpha.iter().sum::<usize>() <= 1)
            .count();
        assert_eta_one_parity(&euclidean, duchon_coords.view(), total_cols - linear_cols);
    }

    /// Minimal K=1 term for direct unit tests of term-state machinery that does
    /// not depend on a real fit (e.g. the gauge-deflation count guard).
    fn trivial_k1_euclidean_term() -> SaeManifoldTerm {
        let n = 4usize;
        let p = 3usize;
        let atom = SaeManifoldAtom::new(
            "atom0",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            Array2::<f64>::ones((n, 2)),
            Array3::<f64>::zeros((n, 2, 1)),
            Array2::<f64>::zeros((2, p)),
            Array2::<f64>::eye(2),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![Array2::<f64>::zeros((n, 1))],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom], assignment).unwrap()
    }

    /// The #1037 quotient-dimension guard: the recorded count of gauge-deflated
    /// evidence directions must be CONSTANT across a single optimization. The
    /// first observation pins the expected count; a later observation that
    /// matches is a no-op; a later observation that DIFFERS is a structural
    /// quotient-dimension change and must error loudly (comparing Laplace
    /// normalizers across a changed null-space is meaningless).
    #[test]
    fn evidence_gauge_deflation_count_guard_pins_then_rejects_change() {
        let mut term = trivial_k1_euclidean_term();
        assert!(term.expected_evidence_gauge_deflated_directions.is_none());

        // First observation pins the expected count.
        term.record_evidence_gauge_deflation_count(2).unwrap();
        assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(2));

        // A matching later observation is a no-op (still Ok, count unchanged).
        term.record_evidence_gauge_deflation_count(2).unwrap();
        assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(2));

        // A DIFFERENT later observation is a structural event → loud error.
        let err = term
            .record_evidence_gauge_deflation_count(3)
            .expect_err("a changed deflation count must error");
        assert!(
            err.contains("deflation count changed"),
            "guard must report the quotient-dimension change explicitly; got: {err}"
        );
    }

    /// The identity-homotopy shortcut's structural probe: the η dial is inert
    /// iff no atom evaluator declares curved columns. Caller-managed atoms
    /// (no evaluator) and one-harmonic periodic banks (M = 3: constant +
    /// fundamental, all linear columns) are inert; an M = 7 periodic bank
    /// dials its h ≥ 2 harmonics, so the walk must run for it.
    #[test]
    fn curvature_homotopy_eta_inertness_probe_tracks_curved_columns() {
        // Caller-managed atom: no evaluator, nothing to dial.
        let term = trivial_k1_euclidean_term();
        assert!(term.curvature_homotopy_eta_is_inert().unwrap());

        // Periodic atoms whose evaluator split declares every column linear.
        let (term, _target, _rho) = small_two_atom_periodic_term();
        assert!(term.curvature_homotopy_eta_is_inert().unwrap());

        // M = 7 periodic: harmonics h ≥ 2 are η-dialed curved columns.
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(7).unwrap());
        let coords = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let atom = SaeManifoldAtom::new(
            "periodic7",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::zeros((7, 1)),
            Array2::<f64>::eye(7),
        )
        .unwrap()
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((5, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        assert!(!term.curvature_homotopy_eta_is_inert().unwrap());
    }

    #[test]
    fn linear_span_anchor_recovers_planted_two_plane_configuration() {
        let n = 4usize;
        let p = 4usize;
        let phi = Array2::<f64>::ones((n, 2));
        let jet = Array3::<f64>::zeros((n, 2, 1));
        let decoder = Array2::<f64>::zeros((2, p));
        let smooth = Array2::<f64>::eye(2);
        let atoms = vec![
            SaeManifoldAtom::new(
                "plane0",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi.clone(),
                jet.clone(),
                decoder.clone(),
                smooth.clone(),
            )
            .unwrap(),
            SaeManifoldAtom::new(
                "plane1",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                smooth,
            )
            .unwrap(),
        ];
        let coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 2)),
            coords,
            vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let target = array![
            [3.0_f64, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let anchor = linear_span_anchor(&term, target.view()).unwrap();
        assert_eq!(anchor.atoms.len(), 2);
        assert_abs_diff_eq!(anchor.residual_norm_sq, 0.0, epsilon = 1.0e-18);
        let plane0 = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let plane1 = array![[0.0_f64, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let angle0 = anchor.atoms[0]
            .frame
            .max_principal_angle(plane0.view())
            .unwrap();
        let angle1 = anchor.atoms[1]
            .frame
            .max_principal_angle(plane1.view())
            .unwrap();
        assert_abs_diff_eq!(angle0, 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(angle1, 0.0, epsilon = 1.0e-12);
    }

    fn circle_certificate_fixture(radius: f64, planes: &[(usize, usize)]) -> SaeManifoldTerm {
        let n = 16usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut atoms = Vec::with_capacity(planes.len());
        let mut coord_blocks = Vec::with_capacity(planes.len());
        for (atom_idx, &(axis_sin, axis_cos)) in planes.iter().enumerate() {
            let mut decoder = Array2::<f64>::zeros((3, p));
            decoder[[1, axis_sin]] = radius;
            decoder[[2, axis_cos]] = radius;
            let atom = SaeManifoldAtom::new(
                format!("circle_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
            coord_blocks.push(coords.clone());
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, planes.len())),
            coord_blocks,
            vec![LatentManifold::Circle { period: 1.0 }; planes.len()],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        term.set_certificate_dispersion(1.0).unwrap();
        term
    }

    #[test]
    fn dictionary_incoherence_report_orthogonal_frames_has_zero_mu_hat() {
        let term = circle_certificate_fixture(2.0, &[(0, 1), (2, 3)]);
        let report = dictionary_incoherence_report(&term).unwrap();
        assert_abs_diff_eq!(report.mu_hat, 0.0, epsilon = 1.0e-12);
        assert_eq!(report.per_atom_kappa_hat.len(), 2);
        // The report carries a verdict (no longer a "not implemented" caveat).
        // The verdict is consistent with the threshold function evaluated on the
        // report's own quantities — the report does not fabricate a verdict.
        let kappa_max = report
            .per_atom_kappa_hat
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let recomputed = curved_dictionary_global_optimality_verdict(
            report.mu_hat,
            kappa_max,
            report.peak_activity_floor,
            report.snr_proxy,
            report.per_atom_kappa_hat.len(),
        );
        assert_eq!(report.global_optimality, recomputed);
        // μ̂ = 0 (orthogonal frames) ⇒ when the preconditions hold (κ̂ < 1,
        // SNR > 1) the certificate certifies, since the budget is positive and
        // μ̂ cannot exceed it. κ̂ = 1/radius = 0.5 < 1 here, so the only gate is
        // SNR; assert certification whenever SNR clears the noise floor.
        if report.snr_proxy > 1.0 {
            assert!(
                report.global_optimality.is_certified(),
                "μ̂=0, κ̂=0.5<1, SNR>1 ⇒ must certify; got {}",
                report.note
            );
        }
    }

    #[test]
    fn dictionary_incoherence_report_coherent_frames_has_unit_mu_hat() {
        let term = circle_certificate_fixture(2.0, &[(0, 1), (0, 1)]);
        let report = dictionary_incoherence_report(&term).unwrap();
        assert_abs_diff_eq!(report.mu_hat, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn dictionary_incoherence_report_circle_kappa_matches_inverse_radius() {
        let radius = 2.5_f64;
        let mut term = circle_certificate_fixture(radius, &[(0, 1)]);
        term.set_certificate_dispersion(0.25).unwrap();
        let report = dictionary_incoherence_report(&term).unwrap();
        assert_abs_diff_eq!(
            report.per_atom_kappa_hat[0],
            1.0 / radius,
            epsilon = 1.0e-10
        );
        assert!(report.snr_proxy.is_finite() && report.snr_proxy > 0.0);
        assert_abs_diff_eq!(report.mean_activity_floor, 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(report.peak_activity_floor, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn search_strategy_exposes_fixed_and_sweep_values() {
        assert!(SearchStrategy::Fixed.is_fixed());

        let strategy = SearchStrategy::ExponentialSweep {
            values: vec![0.1, 1.0, 10.0],
        };
        assert!(!strategy.is_fixed());
        assert_eq!(strategy.sweep_values(), Some([0.1, 1.0, 10.0].as_slice()));
    }

    /// `try_assignments_row` may only pin the K==1 assignment to `1.0` for
    /// Softmax, whose single simplex coordinate is genuinely fixed. For the
    /// independent gate modes (IBP-MAP, JumpReLU) the lone logit must drive the
    /// gate; otherwise the reconstruction ignores a free parameter that the
    /// prior still penalizes (an invalid objective). Regression for the
    /// audit's K==1 special-case bug.
    #[test]
    fn k1_gate_modes_do_not_pin_assignment_to_one() {
        // IBP-MAP, K=1: σ(0/τ)·π_0 = 0.5·1 = 0.5 (not 1.0).
        let ibp = SaeAssignment::from_blocks_with_mode(
            array![[0.0]],
            vec![array![[0.0]]],
            AssignmentMode::ibp_map(1.0, 1.0, false),
        )
        .unwrap();
        assert_abs_diff_eq!(ibp.try_assignments_row(0).unwrap()[0], 0.5, epsilon = 1e-9);

        // JumpReLU, K=1, logit below threshold: hard-gated off (not 1.0).
        let jr = SaeAssignment::from_blocks_with_mode(
            array![[-1.0]],
            vec![array![[0.0]]],
            AssignmentMode::jumprelu(1.0, 0.0),
        )
        .unwrap();
        assert_abs_diff_eq!(jr.try_assignments_row(0).unwrap()[0], 0.0, epsilon = 1e-12);

        // Softmax, K=1: still pinned to 1.0 (no free simplex coordinate).
        // The softmax logits matrix carries `K = k_atoms()` columns (one per
        // atom, canonicalized so the reference column is 0), so K=1 is a single
        // zero column — not the K-1 `assignment_coord_dim` layout. The K=1 pin
        // in `try_assignments_row` keys off `k_atoms() == 1`, i.e. one column.
        let sm = SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((1, 1)),
            vec![array![[0.0]]],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        assert_abs_diff_eq!(sm.try_assignments_row(0).unwrap()[0], 1.0, epsilon = 1e-12);
    }

    /// The JumpReLU surrogate is centered at the threshold: just above the
    /// threshold the gate is ≈ σ(0) = 0.5, not the uncentered σ(threshold/τ).
    /// Below the threshold the hard gate keeps the value at exactly zero.
    /// Regression for the audit's miscentered-threshold bug.
    #[test]
    fn jumprelu_surrogate_is_centered_at_threshold() {
        let threshold = 2.0;
        let temperature = 1.0;
        let logits = array![2.0 + 1e-6, 1.0];
        let gates = jumprelu_row(logits.view(), temperature, threshold);
        // Just above threshold the centered surrogate is ≈ 0.5; the old
        // uncentered surrogate would have been σ(2.0) ≈ 0.88.
        assert_abs_diff_eq!(gates[0], 0.5, epsilon = 1e-3);
        assert!(
            gates[0] < 0.6,
            "surrogate not centered at threshold: {}",
            gates[0]
        );
        // Strictly below the threshold the gate is hard-zero.
        assert_abs_diff_eq!(gates[1], 0.0, epsilon = 1e-12);
    }

    fn periodic_basis(coords: &Array2<f64>) -> (Array2<f64>, Array3<f64>) {
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 3));
        let mut jet = Array3::<f64>::zeros((n, 3, 1));
        for row in 0..n {
            let x = coords[[row, 0]].rem_euclid(1.0);
            let angle = 2.0 * std::f64::consts::PI * x;
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = angle.sin();
            phi[[row, 2]] = angle.cos();
            jet[[row, 1, 0]] = 2.0 * std::f64::consts::PI * angle.cos();
            jet[[row, 2, 0]] = -2.0 * std::f64::consts::PI * angle.sin();
        }
        (phi, jet)
    }

    // --- Periodic/topology ARD prior smoothness + value↔grad consistency ---

    /// The periodic von-Mises ARD energy must be continuous (in value, gradient,
    /// and curvature) as the latent coordinate crosses the period cut. The old
    /// Euclidean `½α t²` jumped by `½α P²` here, breaking Armijo descent. With
    /// period `P = 1` the cut is at `t = 1 ≡ 0`: evaluating just below and just
    /// above must agree to O(eps), and the wrapped-to-`0` representative must
    /// match the unwrapped value.
    #[test]
    fn ard_axis_prior_periodic_is_continuous_across_cut() {
        let alpha = 2.3_f64;
        let period = 1.0_f64;
        let eps = 1.0e-6;
        let below = ArdAxisPrior::eval(alpha, period - eps, Some(period));
        let above = ArdAxisPrior::eval(alpha, period + eps, Some(period));
        let at_zero = ArdAxisPrior::eval(alpha, 0.0, Some(period));
        // Crossing the cut changes value/grad/hess by O(eps), NOT O(½αP²≈1.15
        // for the old Euclidean prior). value and hess are even in (t-cut) so
        // they match to O(eps²); grad is odd through 0, so it flips sign but its
        // magnitude → 0 at the cut and the jump is O(eps) (continuous).
        let cont_tol = 10.0 * alpha * eps; // O(eps) continuity bound
        assert!((below.value - above.value).abs() < cont_tol);
        assert!((below.grad - above.grad).abs() < cont_tol);
        assert!((below.hess - above.hess).abs() < cont_tol);
        // The gradient vanishes at the cut (no kink): both one-sided values are
        // O(eps), unlike the old prior whose grad was α·P ≈ 2.3 just below.
        assert!(below.grad.abs() < cont_tol);
        assert!(above.grad.abs() < cont_tol);
        // The unwrapped representative `period - eps` and the wrapped-near-0
        // representative agree: the energy is a genuine function on the circle.
        assert_abs_diff_eq!(below.value, at_zero.value, epsilon = 1.0e-9);
        // At the origin: zero energy/gradient, curvature == alpha (the ARD
        // precision interpretation is preserved).
        assert_abs_diff_eq!(at_zero.value, 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(at_zero.grad, 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(at_zero.hess, alpha, epsilon = 1.0e-12);
        // `sq_equiv` is alpha-independent so the Mackay/Fellner-Schall update is
        // a clean function of the coordinates.
        let sq_a = ArdAxisPrior::eval(1.0, 0.3, Some(period)).sq_equiv;
        let sq_b = ArdAxisPrior::eval(5.0, 0.3, Some(period)).sq_equiv;
        assert_abs_diff_eq!(sq_a, sq_b, epsilon = 1.0e-12);
        // ½·α·sq_equiv reproduces the energy (consistency with ard_value).
        let p = ArdAxisPrior::eval(alpha, 0.3, Some(period));
        assert_abs_diff_eq!(0.5 * alpha * p.sq_equiv, p.value, epsilon = 1.0e-12);
    }

    /// The per-axis prior gradient must be the exact derivative of its value, on
    /// BOTH the Euclidean (Gaussian) and periodic (von-Mises) axes. This is the
    /// d=1 value↔grad FD agreement that the line search depends on.
    #[test]
    fn ard_axis_prior_value_grad_fd_consistent() {
        let alpha = 1.7_f64;
        let h = 1.0e-6;
        for &period in &[None, Some(1.0_f64), Some(std::f64::consts::TAU)] {
            // Sample several points, including near a periodic cut.
            for &t in &[-0.37_f64, 0.02, 0.49, 0.83, 0.999, 1.4] {
                let p = ArdAxisPrior::eval(alpha, t, period);
                let vp = ArdAxisPrior::eval(alpha, t + h, period).value;
                let vm = ArdAxisPrior::eval(alpha, t - h, period).value;
                let fd_grad = (vp - vm) / (2.0 * h);
                assert_abs_diff_eq!(p.grad, fd_grad, epsilon = 1.0e-5);
                // Hessian == derivative of gradient.
                let gp = ArdAxisPrior::eval(alpha, t + h, period).grad;
                let gm = ArdAxisPrior::eval(alpha, t - h, period).grad;
                let fd_hess = (gp - gm) / (2.0 * h);
                assert_abs_diff_eq!(p.hess, fd_hess, epsilon = 1.0e-5);
            }
        }
    }

    /// The manifold → per-axis periodicity map must classify every topology's
    /// d=1 (and product) axes correctly: line=non-periodic, circle=periodic,
    /// torus=per-axis periodic, sphere chart=(non-periodic lat, periodic lon),
    /// embedded sphere=non-periodic (smooth retraction, no cut).
    #[test]
    fn axis_periods_map_each_topology() {
        assert_eq!(LatentManifold::Euclidean.axis_periods(), vec![None]);
        assert_eq!(
            LatentManifold::Circle { period: 1.0 }.axis_periods(),
            vec![Some(1.0)]
        );
        // Torus (Product of Circles), each axis periodic.
        let torus = LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ]);
        assert_eq!(torus.axis_periods(), vec![Some(1.0), Some(1.0)]);
        // Sphere lat/lon chart: lat is an Interval (non-periodic), lon a Circle.
        let sphere_chart = LatentManifold::Product(vec![
            LatentManifold::Interval { lo: -1.0, hi: 1.0 },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ]);
        assert_eq!(
            sphere_chart.axis_periods(),
            vec![None, Some(std::f64::consts::TAU)]
        );
        // Embedded sphere: smooth retraction, reported non-periodic per axis.
        assert_eq!(
            LatentManifold::Sphere { dim: 3 }.axis_periods(),
            vec![None, None, None]
        );
    }

    /// End-to-end: a periodic term's `ard_value` must be continuous as a latent
    /// coordinate is stepped across the period cut via the (wrapping)
    /// retraction. Reproduces the original non-smoothness bug at the term level:
    /// the old Euclidean prior made `loss.ard` jump by ~½α·P² when a Newton step
    /// crossed `t = 1 ≡ 0`.
    #[test]
    fn ard_value_continuous_across_periodic_cut_d1() {
        // Single periodic atom, one row sitting just below the cut at t≈1.
        let coords0 = array![[0.999_f64]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((1, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.1_f64]];
        // Large alpha makes the OLD bug's jump (~½·α·P²) enormous relative to
        // the smooth O(step) change; with the von-Mises prior it stays tiny.
        let alpha = 50.0_f64;
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);

        let ard_before = term.loss(target.view(), &rho).unwrap().ard;
        // Step the coordinate by +0.002 so it crosses the cut: 0.999 -> 1.001,
        // which the Circle retraction wraps to 0.001.
        let q = term.assignment.row_block_dim();
        let beta_dim = term.beta_dim();
        let mut delta_ext = Array1::<f64>::zeros(q);
        // coord axis is the last entry of the row block (after the logit).
        delta_ext[q - 1] = 0.002;
        let delta_beta = Array1::<f64>::zeros(beta_dim);
        term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
            .unwrap();
        let wrapped = term.assignment.coords[0].row(0)[0];
        // Confirm the step actually crossed and wrapped near 0.
        assert!(
            wrapped < 0.01,
            "coordinate should have wrapped across the cut, got {wrapped}"
        );
        let ard_after = term.loss(target.view(), &rho).unwrap().ard;
        // Smooth: a 0.002 step near the cut changes the ARD energy by a tiny
        // amount. The OLD Euclidean prior would jump by ≈ ½·α·(1² - 0²) = 25.
        assert!(
            (ard_after - ard_before).abs() < 1.0e-2,
            "periodic ARD jumped across the cut: before={ard_before}, after={ard_after}"
        );
    }

    /// The *full line-search objective* (`penalized_objective_total`) — not just
    /// the built-in `loss.ard` — must be continuous across the period cut when a
    /// registry `ARDPenalty` is present, which is the production SAE config
    /// (`ard_per_atom=True` emits `{"kind":"ard","target":"t"}`). The registry
    /// `ARDPenalty` value is the legacy Euclidean Gaussian `½λΣt²`, which jumps by
    /// ≈ ½λ·P² across the cut. Before the fix it was summed into
    /// `analytic_penalty_value_total` on top of the von-Mises `loss.ard`, so the
    /// line-search objective jumped discontinuously while the assembled gradient
    /// (also double-counting the Gaussian `λt`, but that piece is continuous)
    /// predicted only an O(step) change — a near-zero Newton step crossing the
    /// cut then raised the objective by ≈ ½λ and Armijo rejected it (BUG 1). The
    /// fix skips the registry ARD on every SAE path so the smooth von-Mises
    /// built-in is the single source of truth; the objective must now stay smooth.
    #[test]
    fn penalized_objective_continuous_across_periodic_cut_with_registry_ard() {
        let coords0 = array![[0.999_f64]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((1, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.1_f64]];
        // Large precision makes the OLD Gaussian-registry jump (≈ ½λP² = 25) huge
        // relative to the smooth O(step) von-Mises change.
        let alpha = 50.0_f64;
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);

        // Production-shaped registry: one ARD penalty on the "t" coord block.
        let coord = &term.assignment.coords[0];
        let mut registry = AnalyticPenaltyRegistry::new();
        let ard_pen = ARDPenalty::new(
            PsiSlice::full(coord.len(), Some(coord.latent_dim())),
            coord.latent_dim(),
        );
        registry.push(AnalyticPenaltyKind::Ard(Arc::new(ard_pen)));

        let obj_before = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .unwrap();
        let q = term.assignment.row_block_dim();
        let beta_dim = term.beta_dim();
        let mut delta_ext = Array1::<f64>::zeros(q);
        delta_ext[q - 1] = 0.002; // 0.999 -> 1.001, wraps to 0.001 across the cut.
        let delta_beta = Array1::<f64>::zeros(beta_dim);
        term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
            .unwrap();
        let wrapped = term.assignment.coords[0].row(0)[0];
        assert!(
            wrapped < 0.01,
            "coordinate should have wrapped across the cut, got {wrapped}"
        );
        let obj_after = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .unwrap();
        // Smooth: a 0.002 step changes the full objective by a tiny amount. The
        // OLD Gaussian-registry path jumped by ≈ ½·50·(1²−0²) = 25.
        assert!(
            (obj_after - obj_before).abs() < 1.0e-2,
            "line-search objective jumped across the cut: before={obj_before}, after={obj_after}"
        );
    }

    /// Issue #795: `gate_sparsity="scad"` emits a `ScadMcpPenalty` on the "t"
    /// coordinate block. SCAD's energy `Σ f(√(t²+ε²))` is a magnitude shrinkage
    /// with a fixed origin at `t=0`. On a **periodic** (Circle) axis the latent
    /// is an angle defined only modulo its period, so the raw `|t|` is BOTH
    /// ill-posed (no rotation-invariant origin) and *discontinuous across the
    /// retraction branch cut*: a coordinate just below the period wraps to just
    /// above zero, and `f(|t|)` jumps from the flat tail to ≈0. Folded into the
    /// line-search objective, that jump made a near-zero coordinate Newton step
    /// change the objective by an O(weight) amount, so Armijo rejected
    /// otherwise-valid steps and the inner joint solve never reached
    /// stationarity (`reml_criterion: inner solve did not converge`).
    ///
    /// The fix restricts the SCAD/MCP shrinkage to the Euclidean axes, so on a
    /// pure Circle atom it contributes nothing — the objective with the SCAD
    /// registry must equal the registry-free objective, and must stay continuous
    /// across the cut.
    #[test]
    fn scad_coord_penalty_inert_and_continuous_on_periodic_axis() {
        use crate::terms::analytic_penalties::{PenaltyConcavity, ScadMcpPenalty};

        let coords0 = array![[0.999_f64]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((1, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.1_f64]];
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0_f64]]);

        // Registry with a single SCAD shrinkage on the coordinate block, with a
        // large weight so the OLD (unrestricted) energy would dominate.
        let coord = &term.assignment.coords[0];
        let mut registry = AnalyticPenaltyRegistry::new();
        let scad = ScadMcpPenalty::new(
            PsiSlice::full(coord.len(), Some(coord.latent_dim())),
            5.0,
            coord.n_obs(),
            3.7,
            1.0e-3,
            PenaltyConcavity::Scad,
            false,
        )
        .unwrap();
        registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(scad)));

        // Inert on a pure Circle: the SCAD registry adds zero energy because the
        // sole axis is periodic.
        let with_scad = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .unwrap();
        let without = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .unwrap();
        assert!(
            (with_scad - without).abs() < 1.0e-12,
            "SCAD coord penalty must be inert on a pure periodic axis: \
             with={with_scad}, without={without}"
        );

        // Continuous across the period cut: stepping 0.999 -> 1.001 (wraps to
        // 0.001) must not jump the objective. The OLD unrestricted SCAD energy
        // jumped by ≈ weight·(|0.999| − |0.001|) ≈ 5.
        let obj_before = with_scad;
        let q = term.assignment.row_block_dim();
        let beta_dim = term.beta_dim();
        let mut delta_ext = Array1::<f64>::zeros(q);
        delta_ext[q - 1] = 0.002;
        let delta_beta = Array1::<f64>::zeros(beta_dim);
        term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
            .unwrap();
        let wrapped = term.assignment.coords[0].row(0)[0];
        assert!(
            wrapped < 0.01,
            "coordinate should have wrapped across the cut, got {wrapped}"
        );
        let obj_after = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .unwrap();
        assert!(
            (obj_after - obj_before).abs() < 1.0e-2,
            "SCAD line-search objective jumped across the periodic cut: \
             before={obj_before}, after={obj_after}"
        );
    }

    /// Guard against over-broad exclusion: on a **Euclidean** chart axis the SCAD
    /// magnitude shrinkage is well-posed (`t=0` is a genuine origin) and must
    /// remain active. `sae_coord_penalty_euclidean_restriction` returns `None`
    /// (nothing to restrict) for an all-Euclidean coord and the full-support
    /// `Some` carrier for a periodic coord, so value/gradient/curvature all see
    /// the same axis set.
    #[test]
    fn scad_coord_penalty_active_on_euclidean_axis() {
        let euclid = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![array![[0.5_f64], [-0.7], [1.3]]],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        // All axes Euclidean: no restriction (the penalty applies in full).
        assert!(
            sae_coord_penalty_euclidean_restriction(&euclid.coords[0]).is_none(),
            "Euclidean coord must not be restricted"
        );

        let circle = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![array![[0.1_f64], [0.4], [0.9]]],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        // Pure periodic axis: restricted to an empty Euclidean carrier.
        let (axes, compacted) = sae_coord_penalty_euclidean_restriction(&circle.coords[0])
            .expect("periodic coord must be restricted");
        assert!(
            axes.is_empty(),
            "circle has no Euclidean axes, got {axes:?}"
        );
        assert_eq!(compacted.len(), 0, "compacted target must be empty");
    }

    /// The von-Mises coordinate-prior curvature `V'' = α·cos(κt)` is indefinite
    /// (negative for |t| past a quarter period). Writing it raw into the
    /// Newton/Schur `htt` diagonal at K=2 made the per-row coordinate block, and
    /// hence the Schur complement, non-PD and the Cholesky failed on a negative
    /// pivot (BUG 3). The assembled `htt` diagonal on every periodic coord axis
    /// must therefore be non-negative (the `max(V'',0)` PSD majorizer), while the
    /// gradient stays the exact `V'`.
    #[test]
    fn periodic_ard_curvature_is_psd_in_assembled_htt() {
        // Two rows past the quarter period (t in (0.25, 0.75)) where cos(2πt) < 0.
        let coords0 = array![[0.40_f64], [0.60_f64]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((2, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.1_f64], [0.2_f64]];
        // Large α drives V''=α·cos(2πt) strongly negative at t=0.4,0.6
        // (cos(0.8π)≈-0.809), so a raw write would push the data-fit-only htt
        // diagonal negative.
        let alpha = 100.0_f64;
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let d = row.htt.nrows();
            for a in 0..d {
                assert!(
                    row.htt[[a, a]] >= 0.0,
                    "row {row_idx} htt diagonal[{a}]={} must be PSD (von-Mises \
                     curvature clamped to its positive part)",
                    row.htt[[a, a]]
                );
            }
        }
    }

    /// `snapshot_mutable_state` / `restore_mutable_state` (the in-place
    /// line-search save/restore that replaced the per-halving full
    /// `self.clone()`) must restore exactly the state an `apply_newton_step`
    /// trial perturbs: decoder coefficients, the `refresh_basis`-rebuilt
    /// basis evaluations, assignment logits, and latent coordinates. Pins
    /// item-1 of the SAE hot-path CPU-perf refactor.
    #[test]
    fn snapshot_restore_round_trips_mutated_state() {
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((4, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Capture pre-step state, then apply a non-trivial Newton step that
        // refreshes the basis (changing basis_values/jacobian, decoder
        // coefficients, logits, and coords).
        let snapshot = term.snapshot_mutable_state();
        let pre_basis = term.atoms[0].basis_values.clone();
        let pre_jet = term.atoms[0].basis_jacobian.clone();
        let pre_decoder = term.atoms[0].decoder_coefficients.clone();
        let pre_logits = term.assignment.logits.clone();
        let pre_coords = term.assignment.coords[0].as_matrix();

        let q = term.assignment.row_block_dim();
        let beta_dim = term.beta_dim();
        let delta_ext = Array1::<f64>::from_elem(4 * q, 0.3);
        let delta_beta = Array1::<f64>::from_elem(beta_dim, -0.4);
        term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
            .unwrap();

        // Something must actually have changed, else the test is vacuous.
        assert!(
            (&term.atoms[0].basis_values - &pre_basis)
                .mapv(f64::abs)
                .sum()
                > 1e-9
                || (&term.atoms[0].decoder_coefficients - &pre_decoder)
                    .mapv(f64::abs)
                    .sum()
                    > 1e-9,
            "apply_newton_step did not perturb the snapshotted state"
        );

        // Restore and confirm every snapshotted field matches the pre-step
        // values bit-for-bit.
        term.restore_mutable_state(&snapshot);
        assert_eq!(term.atoms[0].basis_values, pre_basis);
        assert_eq!(term.atoms[0].basis_jacobian, pre_jet);
        assert_eq!(term.atoms[0].decoder_coefficients, pre_decoder);
        assert_eq!(term.assignment.logits, pre_logits);
        assert_eq!(term.assignment.coords[0].as_matrix(), pre_coords);
    }

    #[test]
    fn ibp_path_refreshes_periodic_basis_for_two_newton_iterations() {
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.2], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((4, 1)),
            vec![coords0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.10], [0.05], [-0.15], [0.20]];
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
        let loss0 = term.loss(target.view(), &rho).unwrap().total();
        let basis0 = term.atoms[0].basis_values.clone();

        let loss = term
            .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 2, 0.05, 1.0e-3, 1.0e-3)
            .unwrap();

        assert!(loss.total().is_finite());
        assert!(loss.total() <= loss0 + 1.0e-8);
        assert!(
            term.assignment.coords[0]
                .as_flat()
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(term.assignment.assignments().iter().all(|v| v.is_finite()));
        let basis_delta = (&term.atoms[0].basis_values - &basis0).mapv(f64::abs).sum();
        assert!(basis_delta > 1.0e-10);
    }

    fn small_two_atom_periodic_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
        let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let (phi1, jet1) = periodic_basis(&coords1);
        let atom0 = SaeManifoldAtom::new(
            "periodic0",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.25], [-0.35], [0.15]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let atom1 = SaeManifoldAtom::new(
            "periodic1",
            SaeAtomBasisKind::Periodic,
            1,
            phi1,
            jet1,
            array![[-0.10], [0.20], [0.30]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let logits = array![
            [0.7, -0.2],
            [0.1, 0.4],
            [-0.3, 0.5],
            [0.6, -0.1],
            [0.2, 0.3]
        ];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0, coords1],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
        let target = array![[0.12], [-0.03], [0.08], [0.20], [-0.11]];
        let rho = SaeManifoldRho::new(
            (-0.3_f64).exp().ln(),
            0.7_f64.ln(),
            vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
        );
        (term, target, rho)
    }

    /// #976 Layer-1 guard 2: a single Newton application cannot move a gate
    /// logit by more than the gate-scale cap, however large the solver's raw
    /// delta. Softmax canonicalization shifts whole rows, so the invariant is
    /// checked on the within-row logit DIFFERENCE, which the shift preserves.
    #[test]
    fn assignment_logit_step_cap_bounds_single_iteration_gate_motion() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        let n = term.assignment.n_obs();
        let q = term.assignment.row_block_dim();
        let diff_before = term.assignment.logits[[0, 0]] - term.assignment.logits[[0, 1]];

        let mut delta = Array1::<f64>::zeros(n * q);
        // Softmax K=2 has one free logit per row at offset 0 of the row block.
        delta[0] = 1.0e6;
        let delta_beta = Array1::<f64>::zeros(term.beta_dim());
        term.apply_newton_step(delta.view(), delta_beta.view(), 1.0)
            .expect("step applies");

        let cap = SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * term.assignment.mode.temperature();
        let diff_after = term.assignment.logits[[0, 0]] - term.assignment.logits[[0, 1]];
        assert!(
            ((diff_after - diff_before) - cap).abs() < 1.0e-9,
            "a 1e6 raw logit delta must realise exactly the {cap}-cap, moved {}",
            diff_after - diff_before
        );
    }

    /// #976 Layer-1 guard 3: a gate-collapsed atom (max active mass below the
    /// floor) is re-seeded back into contention exactly once, every breach is
    /// an observable CollapseEvent, and the second collapse is recorded as
    /// terminal — once — instead of fighting the optimizer.
    #[test]
    fn active_mass_guard_reseeds_once_then_records_terminal_collapse() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        let n = term.assignment.n_obs();
        let slam = |term: &mut SaeManifoldTerm| {
            for row in 0..n {
                term.assignment.logits[[row, 0]] = 0.0;
                term.assignment.logits[[row, 1]] = -1.0e3;
            }
        };

        slam(&mut term);
        term.enforce_active_mass_guard(0).expect("guard runs");
        assert_eq!(term.collapse_events().len(), 1);
        let ev = term.collapse_events()[0];
        assert_eq!(ev.atom, 1);
        assert_eq!(ev.action, CollapseAction::Reseeded);
        assert!(ev.max_active_mass < ev.floor);

        // The re-seed restored material support (softmax parity with the
        // row winner), so a healthy follow-up check records nothing.
        let masses = term.assignment.assignments();
        let max1 = (0..n).map(|r| masses[[r, 1]]).fold(0.0_f64, f64::max);
        assert!(max1 > SAE_ATOM_ACTIVE_MASS_FLOOR);
        term.enforce_active_mass_guard(1).expect("guard runs");
        assert_eq!(term.collapse_events().len(), 1);

        // Second collapse: budget exhausted ⇒ terminal, recorded exactly once
        // across repeated checks; the logits are left to the objective.
        slam(&mut term);
        term.enforce_active_mass_guard(2).expect("guard runs");
        term.enforce_active_mass_guard(3).expect("guard runs");
        let terminals: Vec<_> = term
            .collapse_events()
            .iter()
            .filter(|e| e.action == CollapseAction::Terminal)
            .collect();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0].atom, 1);
        assert!(
            term.collapse_events().iter().all(|e| e.atom == 1),
            "the healthy atom must never be flagged"
        );
    }

    #[test]
    fn sae_rho_seed_dispersion_scaling_shifts_every_scale_coupled_axis() {
        let rho = SaeManifoldRho::new(0.7_f64.ln(), 1.3_f64.ln(), vec![array![0.2, -0.4]]);
        let dispersion = 0.05_f64 * 0.05;
        let scaled = rho
            .seed_scaled_by_dispersion_for_assignment(dispersion, AssignmentMode::softmax(1.0))
            .unwrap();
        let shift = dispersion.ln();

        assert_abs_diff_eq!(
            scaled.log_lambda_sparse,
            rho.log_lambda_sparse + shift,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            scaled.log_lambda_smooth,
            rho.log_lambda_smooth + shift,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            scaled.log_ard[0][0],
            rho.log_ard[0][0] + shift,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            scaled.log_ard[0][1],
            rho.log_ard[0][1] + shift,
            epsilon = 1.0e-14
        );

        let learnable_ibp = rho
            .seed_scaled_by_dispersion_for_assignment(
                dispersion,
                AssignmentMode::ibp_map(1.0, 1.0, true),
            )
            .unwrap();
        assert_abs_diff_eq!(
            learnable_ibp.log_lambda_sparse,
            rho.log_lambda_sparse,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            learnable_ibp.log_lambda_smooth,
            rho.log_lambda_smooth + shift,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            learnable_ibp.log_ard[0][0],
            rho.log_ard[0][0] + shift,
            epsilon = 1.0e-14
        );
    }

    #[test]
    fn fit_data_collapse_records_terminal_event_for_active_atom() {
        let coords = array![[0.0], [0.25], [0.5], [0.75]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::zeros((3, 2)),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((4, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let fitted = Array2::<f64>::zeros(target.dim());
        let assignments = Array2::<f64>::ones((4, 1));

        let recorded = term
            .record_fit_data_collapse_if_needed(target.view(), fitted.view(), assignments.view(), 7)
            .unwrap();

        assert!(recorded);
        let terminals: Vec<_> = term
            .collapse_events()
            .iter()
            .filter(|event| event.action == CollapseAction::Terminal)
            .collect();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0].atom, 0);
        assert_eq!(terminals[0].iteration, 7);
        assert!(terminals[0].max_active_mass <= SAE_FIT_DATA_COLLAPSE_EV_FLOOR);
    }

    fn deterministic_circle_noise(row: usize, col: usize) -> f64 {
        let x = (row as f64 + 1.0) * 12.9898 + (col as f64 + 1.0) * 78.233;
        (x.sin() * 43758.5453).sin()
    }

    fn planted_circle_data(n: usize, sigma: f64) -> Array2<f64> {
        let mut z = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let theta = std::f64::consts::TAU * row as f64 / n as f64;
            z[[row, 0]] = theta.cos() + sigma * deterministic_circle_noise(row, 0);
            z[[row, 1]] = theta.sin() + sigma * deterministic_circle_noise(row, 1);
        }
        z
    }

    fn global_ev(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
        let (n, p) = target.dim();
        let mut means = vec![0.0_f64; p];
        for col in 0..p {
            for row in 0..n {
                means[col] += target[[row, col]];
            }
            means[col] /= n as f64;
        }
        let mut ssr = 0.0_f64;
        let mut sst = 0.0_f64;
        for row in 0..n {
            for col in 0..p {
                let r = target[[row, col]] - fitted[[row, col]];
                ssr += r * r;
                let centered = target[[row, col]] - means[col];
                sst += centered * centered;
            }
        }
        1.0 - ssr / sst.max(1.0e-300)
    }

    #[derive(Clone, Copy)]
    enum PlantedCircleAssignmentMode {
        Softmax,
        IbpMap,
    }

    impl PlantedCircleAssignmentMode {
        fn label(self) -> &'static str {
            match self {
                Self::Softmax => "softmax",
                Self::IbpMap => "ibp_map",
            }
        }

        fn mode(self) -> AssignmentMode {
            const TAU: f64 = 1.0;
            const ALPHA: f64 = 1.0;
            match self {
                Self::Softmax => AssignmentMode::softmax(TAU),
                Self::IbpMap => AssignmentMode::ibp_map(TAU, ALPHA, false),
            }
        }

        fn seed_logit(self) -> f64 {
            const TAU: f64 = 1.0;
            match self {
                Self::Softmax => 0.0,
                Self::IbpMap => 6.0 * TAU,
            }
        }

        fn seed_gate(self) -> f64 {
            match self {
                Self::Softmax => 1.0,
                Self::IbpMap => 1.0 / (1.0 + (-6.0_f64).exp()),
            }
        }
    }

    fn planted_circle_seed_term(
        z: ArrayView2<'_, f64>,
        assignment_mode: PlantedCircleAssignmentMode,
    ) -> (SaeManifoldTerm, f64) {
        let n = z.nrows();
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let seed_coords =
            sae_pca_seed_initial_coords(z, &[SaeAtomBasisKind::Periodic], &[1]).unwrap();
        let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let seed_gate = assignment_mode.seed_gate();
        let gated_phi = &phi * seed_gate;
        let mut xtx = fast_ata(&gated_phi);
        for i in 0..xtx.nrows() {
            xtx[[i, i]] += 1.0e-10;
        }
        let xtz = fast_atb(&gated_phi, &z.to_owned());
        let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
        let seed_fitted = gated_phi.dot(&decoder);
        let mut rss = 0.0_f64;
        for row in 0..n {
            for col in 0..z.ncols() {
                let r = z[[row, col]] - seed_fitted[[row, col]];
                rss += r * r;
            }
        }
        let seed_dispersion = (rss / (n * z.ncols()) as f64).max(1.0e-12);
        let atom = SaeManifoldAtom::new(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::from_elem((n, 1), assignment_mode.seed_logit()),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            assignment_mode.mode(),
        )
        .unwrap();
        (
            SaeManifoldTerm::new(vec![atom], assignment).unwrap(),
            seed_dispersion,
        )
    }

    #[test]
    fn planted_circle_noise_scale_sweep_reaches_high_ev_with_dimensionless_rho_seed() {
        for assignment_mode in [
            PlantedCircleAssignmentMode::Softmax,
            PlantedCircleAssignmentMode::IbpMap,
        ] {
            let assignment_label = assignment_mode.label();
            for &n in &[40usize, 250usize] {
                for &sigma in &[0.02_f64, 0.05, 0.18] {
                    let z = planted_circle_data(n, sigma);
                    let (term, seed_dispersion) =
                        planted_circle_seed_term(z.view(), assignment_mode);
                    let seed_ev = global_ev(z.view(), term.fitted().view());
                    let init_rho =
                        SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
                            .seed_scaled_by_dispersion_for_assignment(
                                seed_dispersion,
                                assignment_mode.mode(),
                            )
                            .unwrap();
                    let init_rho_flat = init_rho.to_flat();
                    let n_params = init_rho_flat.len();
                    let mut objective = SaeManifoldOuterObjective::new(
                        term,
                        z.clone(),
                        None,
                        init_rho,
                        50,
                        0.04,
                        1.0e-6,
                        1.0e-6,
                    );
                    crate::solver::outer_strategy::OuterProblem::new(n_params)
                        .with_initial_rho(init_rho_flat)
                        .run(&mut objective, "SAE planted circle dimensionless seed")
                        .unwrap();
                    let (fitted_term, rho, _loss) = objective.into_fitted();
                    let fitted = fitted_term.fitted();
                    let ev = global_ev(z.view(), fitted.view());
                    assert!(
                        ev > 0.95,
                        "planted circle assignment={assignment_label} n={n} sigma={sigma} seed_ev={seed_ev:.4} seed_phi={seed_dispersion:.3e} \
                         final_rho=({:.3}, {:.3}, {:?}) EV={ev:.4} should exceed 0.95",
                        rho.log_lambda_sparse,
                        rho.log_lambda_smooth,
                        rho.log_ard
                    );
                    assert!(
                        fitted_term.collapse_events().is_empty(),
                        "healthy planted circle assignment={assignment_label} fit should not record collapse events: {:?}",
                        fitted_term.collapse_events()
                    );
                }
            }
        }
    }

    #[test]
    fn sae_value_probe_refusal_classification_is_inner_only() {
        assert!(
            SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
                "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ"
            )
        );
        assert!(
            SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
                "SaeManifoldTerm::reml_criterion: undamped evidence factorization hit a non-PD per-row H_tt block before KKT stationarity"
            )
        );
        assert!(
            !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
                "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count changed"
            )
        );
    }

    #[test]
    fn streaming_exact_reml_matches_full_batch_reml_small_sae() {
        let (term0, target, rho) = small_two_atom_periodic_term();
        let mut full = term0.clone();
        let mut streaming = term0;
        let (full_cost, full_loss, _cache) = full
            .reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
            .unwrap();
        let (stream_cost, stream_loss) = streaming
            .reml_criterion_streaming_exact(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
            .unwrap();
        assert_abs_diff_eq!(stream_cost, full_cost, epsilon = 1.0e-8);
        assert_abs_diff_eq!(stream_loss.total(), full_loss.total(), epsilon = 1.0e-8);
    }

    /// #1029 measure-consistency gate: the value-probe refine policy must
    /// rank the SAME criterion as the full accepted-point policy. The probe
    /// budget only caps refinement work — it never loosens the KKT/step
    /// tolerance — so when both policies converge from the same state, the
    /// returned criterion values must match to inner-solve tolerance. A loose
    /// probe value compared against a tight reference value is exactly the
    /// estimator/threshold measure mismatch that caused the BMS HT-subsample
    /// false-reject bug; this test pins the invariant that forbids it.
    #[test]
    fn value_probe_refine_policy_ranks_same_criterion_as_full_policy() {
        let (term0, target, rho) = small_two_atom_periodic_term();
        let mut full = term0.clone();
        let mut probe = term0;
        let (full_cost, full_loss) = full
            .reml_criterion_with_refine_policy(
                target.view(),
                &rho,
                None,
                2,
                0.25,
                1.0e-4,
                1.0e-4,
                true,
            )
            .expect("full-budget criterion must converge on the small fixture");
        let (probe_cost, probe_loss) = probe
            .reml_criterion_with_refine_policy(
                target.view(),
                &rho,
                None,
                2,
                0.25,
                1.0e-4,
                1.0e-4,
                false,
            )
            .expect("probe-budget criterion must converge on the small fixture");
        assert_abs_diff_eq!(probe_cost, full_cost, epsilon = 1.0e-8);
        assert_abs_diff_eq!(probe_loss.total(), full_loss.total(), epsilon = 1.0e-8);
    }

    /// #1029 budget-policy gate: value probes get the base refine budget and
    /// NEVER earn the progress extension (their base and progress budgets
    /// coincide), while the accepted-point path extends only on a measured
    /// round-to-round KKT-residual drop and falls back to the base budget on
    /// a stall.
    #[test]
    fn refine_iteration_limit_probe_budget_never_extends() {
        let probe_base = 16usize;
        // Probe policy: base == progress, so even perfect progress cannot
        // extend past the base work budget.
        assert_eq!(
            SaeManifoldTerm::refine_iteration_limit(
                probe_base,
                probe_base,
                probe_base,
                Some(1.0),
                0.5,
                true
            ),
            probe_base
        );
        let accepted_base = 64usize;
        let accepted_progress = 256usize;
        // Accepted-point policy: a real residual drop extends the budget…
        assert_eq!(
            SaeManifoldTerm::refine_iteration_limit(
                accepted_base,
                accepted_base,
                accepted_progress,
                Some(1.0),
                0.5,
                false
            ),
            accepted_progress
        );
        // …a stalled residual does not…
        assert_eq!(
            SaeManifoldTerm::refine_iteration_limit(
                accepted_base,
                accepted_base,
                accepted_progress,
                Some(1.0),
                1.0,
                false
            ),
            accepted_base
        );
        // …and below the base budget no extension question arises yet.
        assert_eq!(
            SaeManifoldTerm::refine_iteration_limit(
                accepted_base - 1,
                accepted_base,
                accepted_progress,
                None,
                1.0e9,
                false
            ),
            accepted_base
        );
    }

    #[test]
    fn reml_retries_refinement_after_non_pd_undamped_evidence_factor() {
        let (mut term0, target, rho) = small_two_atom_periodic_term();
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let cold_sys = term0
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let cold_factor = solve_arrow_newton_step_with_options(&cold_sys, 0.0, 0.0, &options);
        let cold_err = match cold_factor {
            Err(err) => err,
            Ok(_) => panic!("fixture must start with a non-PD undamped evidence row factor"),
        };
        assert!(
            SaeManifoldTerm::is_undamped_evidence_row_non_pd(&cold_err),
            "fixture must start with a genuine evidence-mode non-PD row factor; got {cold_err}",
        );

        let mut full = term0.clone();
        let mut streaming = term0;
        let (full_cost, full_loss, cache) = full
            .reml_criterion_with_cache(target.view(), &rho, None, 1, 0.25, 1.0e-4, 1.0e-4)
            .expect("dense REML must refine through the cold non-PD evidence factor");
        let log_det = arrow_log_det_from_cache(&cache).expect("refined cache must carry log-det");
        assert!(full_cost.is_finite());
        assert!(full_loss.total().is_finite());
        assert!(log_det.is_finite());

        let (stream_cost, stream_loss) = streaming
            .reml_criterion_streaming_exact(target.view(), &rho, None, 1, 0.25, 1.0e-4, 1.0e-4)
            .expect("streaming REML must share the dense refinement retry");
        assert_abs_diff_eq!(stream_cost, full_cost, epsilon = 1.0e-8);
        assert_abs_diff_eq!(stream_loss.total(), full_loss.total(), epsilon = 1.0e-8);
    }

    #[test]
    fn reconstruction_dispersion_uses_ard_shrunk_coordinate_edf() {
        let n = 24usize;
        let p = 2usize;
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.30, -0.10], [0.20, 0.40], [-0.35, 0.15]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = Array2::from_shape_fn((n, p), |(row, col)| {
            let x = (row as f64 + 0.5) / n as f64;
            if col == 0 {
                0.45 * (std::f64::consts::TAU * x).sin() + 0.07
            } else {
                -0.20 * (std::f64::consts::TAU * x).cos() + 0.03 * row as f64
            }
        });
        let alpha = 250.0_f64;
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
        let loss = term.loss(target.view(), &rho).unwrap();
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();

        let dispersion = term.reconstruction_dispersion(&loss, &cache, &rho).unwrap();
        let smooth_edf = term
            .decoder_smoothness_effective_dof(&cache, rho.lambda_smooth())
            .unwrap();
        let beta_edf = (term.beta_dim() as f64 - smooth_edf).max(0.0);
        let traces = term.ard_inverse_traces(&cache).unwrap();
        let coord_edf = (n as f64 - alpha * traces[0][0]).clamp(0.0, n as f64);
        let rss = 2.0 * loss.data_fit;
        let expected = rss / ((n * p) as f64 - beta_edf - coord_edf).max(1.0);
        assert_abs_diff_eq!(dispersion, expected, epsilon = 1.0e-10);

        let old_full_coordinate_edf = n as f64;
        let old_full_coordinate_dispersion =
            rss / ((n * p) as f64 - beta_edf - old_full_coordinate_edf).max(1.0);
        assert!(
            coord_edf < 0.25 * old_full_coordinate_edf,
            "test setup must put the coordinate axis in an ARD-shrunk regime; \
             coord_edf={coord_edf}, old_full_coordinate_edf={old_full_coordinate_edf}"
        );
        assert!(
            dispersion < 0.75 * old_full_coordinate_dispersion,
            "φ̂ must use the ARD-shrunk coordinate edf, not the old full \
             coordinate count: got {dispersion}, old formula {old_full_coordinate_dispersion}"
        );
    }

    #[test]
    fn streaming_plan_routes_by_memory_budget_with_identical_logdet() {
        let (term0, target, rho) = small_two_atom_periodic_term();
        let total_basis: usize = term0.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = term0
            .atoms
            .iter()
            .map(|atom| atom.latent_dim)
            .max()
            .unwrap();
        let dense_plan = sae_streaming_plan_from_budget(
            term0.n_obs(),
            total_basis,
            term0.k_atoms(),
            d_max,
            term0.beta_dim(),
            usize::MAX / 4,
            1024 * 1024,
            usize::MAX / 2,
        );
        assert!(!dense_plan.streaming);
        assert!(dense_plan.direct_admitted);
        let streaming_plan = sae_streaming_plan_from_budget(
            term0.n_obs(),
            total_basis,
            term0.k_atoms(),
            d_max,
            term0.beta_dim(),
            1,
            512,
            2,
        );
        assert!(streaming_plan.streaming);
        assert!(!streaming_plan.direct_admitted);

        let mut full = term0.clone();
        // The undamped (`ridge_t = 0`) log-det is only well-defined at the inner
        // optimum, where the per-row `H_tt^(i)` blocks are PD. At the initial
        // (non-stationary) iterate a `p_out = 1` rank-1 `JᵀJ` row block plus the
        // softmax negative-logit curvature is indefinite, so factoring there at
        // ridge 0 surfaces `PerRowFactorFailed` for BOTH the dense and streaming
        // paths. Converge the inner `(t, β)` state first (matching how
        // `reml_criterion_with_cache` reaches a PD block), then compare the
        // streaming-vs-dense log-determinants of the SAME converged system —
        // which is the routing invariant this test pins (#847).
        full.reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
            .unwrap();
        let sys = full
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let factor_result = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();
        let full_logdet = arrow_log_det_from_cache(&factor_result.2).unwrap();
        let mut streaming = StreamingArrowSchur::from_system(&sys, streaming_plan.chunk_size);
        let streaming_logdet = streaming.exact_arrow_log_det(0.0, 0.0, &options).unwrap();
        assert_abs_diff_eq!(streaming_logdet, full_logdet, epsilon = 1.0e-8);
    }

    #[test]
    fn giant_host_working_set_plan_flips_to_matrix_free_before_dense_allocation() {
        let n_obs = 128usize;
        let total_basis = 48usize;
        let k_atoms = 8usize;
        let d_max = 2usize;
        let p_out = 2048usize;
        let border_dim = total_basis * p_out;
        let budget = 60usize * 1024 * 1024 * 1024;
        let plan = sae_streaming_plan_from_budget(
            n_obs,
            total_basis,
            k_atoms,
            d_max,
            border_dim,
            budget,
            8 * 1024 * 1024,
            120usize * 1024 * 1024 * 1024,
        );

        assert_eq!(border_dim, 98_304);
        assert_eq!(
            plan.estimated_row_cross_bytes,
            n_obs * k_atoms * (1 + d_max) * border_dim * SAE_BYTES_PER_F64
        );
        assert!(plan.estimated_dense_schur_bytes > budget);
        assert!(plan.estimated_matrix_free_peak_bytes < budget);
        assert!(plan.streaming);
        assert!(!plan.direct_admitted);
        assert!(plan.matrix_free_admitted);
        assert_eq!(
            plan.solve_options_for_border_dim(border_dim).mode,
            crate::solver::arrow_schur::ArrowSolverMode::InexactPCG
        );
    }

    #[test]
    fn sparse_active_layout_work_scales_with_active_atoms_not_total_k() {
        let n = 3;
        let k_atoms = 100_000;
        let mut active_rows = Vec::with_capacity(n);
        for row in 0..n {
            active_rows.push(vec![row, 10_000 + row, 90_000 + row]);
        }
        let coord_dims = vec![1usize; k_atoms];
        let coord_offsets_full: Vec<usize> = (0..k_atoms).map(|k| k_atoms + k).collect();
        let layout = SaeRowLayout::from_active_atoms(active_rows, coord_dims, coord_offsets_full);
        for row in 0..n {
            assert_eq!(layout.active_atoms[row].len(), 3);
            assert_eq!(layout.row_q_active(row), 6);
        }
        let compact_work: usize = (0..n)
            .map(|row| {
                let q = layout.row_q_active(row);
                q * q
            })
            .sum();
        let dense_q = 2 * k_atoms;
        let dense_work = n * dense_q * dense_q;
        assert!(compact_work < dense_work / 1_000_000_000);
        assert_eq!(compact_work, n * 36);
    }

    /// Regression test for https://github.com/SauersML/gam/issues/163.
    ///
    /// `ManifoldSAE.predict(X_subset)` reseeds the latent coordinates via PCA
    /// on a possibly small batch (here: a strict subset of the training data),
    /// which can produce a per-row `H_tt + ridge_t·I` that is not
    /// positive-definite at the caller's nominal `ridge_t = 1e-6`. The fit
    /// path tolerates this via the proximal LM correction outer wrapper;
    /// previously, `run_joint_fit_arrow_schur` invoked `sys.solve(...)`
    /// directly and surfaced the per-row Cholesky failure to the caller. The
    /// fix routes recoverable factor failures through a Levenberg-Marquardt
    /// damping schedule (mirrors the `proximal_correction` outer loop),
    /// so an inner step with a degenerate Hessian no longer aborts the
    /// Newton driver.
    #[test]
    fn run_joint_fit_arrow_schur_escalates_ridge_on_non_pd_row_block() {
        // Construct a periodic atom whose row block is rank-deficient when
        // the assignment column is zero — `H_tt` is then driven entirely by
        // the smoothness penalty / external coord ridge and floats just
        // above zero. At ridge_t = 1e-6 the per-row Cholesky finds a tiny
        // negative pivot from rounding error; the escalation loop should
        // recover.
        let coords = array![[0.1], [0.4], [0.7]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            // Decoder that maps to a single output dim with small magnitude
            array![[0.05], [-0.05], [0.05]],
            // No external smoothness penalty on the decoder, so the only
            // regularization on `t` comes from `ridge_ext_coord`.
            Array2::<f64>::zeros((3, 3)),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            // Zero assignment mass → H_tt has zero data contribution.
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        // log_lambda_smooth driven low so the analytic penalty contributes
        // essentially nothing to H_tt either.
        let mut rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

        // The Python-side `predict` default. Before the fix this returned
        // `Err(... per-row H_tt^(?) Cholesky failed ... non-PD pivot ...)`;
        // afterward the escalation loop bumps ridge_t until the per-row
        // factor succeeds, and run_joint_fit_arrow_schur returns Ok.
        let result =
            term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6);
        assert!(
            result.is_ok(),
            "run_joint_fit_arrow_schur should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
        );
    }

    /// Regression test for https://github.com/SauersML/gam/issues/163 and #175.
    ///
    /// `ManifoldSAE.reconstruct(X_oos)` (and `.predict(X_subset)`) reach the
    /// Rust core via `sae_manifold_predict_oos` → `sae_manifold_fit_inner` →
    /// the same `run_joint_fit_arrow_schur` Newton driver. The driver in turn
    /// calls `solve_newton_step` for single-shot refinement; before this fix
    /// that path invoked `sys.solve(...)` directly, bypassing the LM ridge
    /// escalation and surfacing the per-row Cholesky failure to the Python
    /// caller as `"row N H_tt was non-PD at ridge_t=0.000001"`. The fix routes
    /// `solve_newton_step` through `solve_with_lm_escalation` so every entry
    /// point — including OOS predict — geometrically grows the proximal ridge
    /// from the caller's nominal `ridge_ext_coord` / `ridge_beta` until the
    /// factor succeeds.
    #[test]
    fn solve_newton_step_escalates_ridge_on_non_pd_row_block() {
        // Same degenerate-H_tt construction as the predict/reconstruct
        // reproducer: zero assignment mass + zero smoothness penalty means
        // the only mass on H_tt comes from `ridge_t·I`, and at the nominal
        // 1e-6 the Cholesky still finds a tiny negative pivot from rounding.
        let coords = array![[0.1], [0.4], [0.7]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.05], [-0.05], [0.05]],
            Array2::<f64>::zeros((3, 3)),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        let rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

        // Direct `solve_newton_step` call (the predict path's single-shot
        // refinement entry). Must Ok via LM escalation, not bubble up the
        // raw per-row factor failure.
        let result = term.solve_newton_step(target.view(), &rho, None, 1.0e-6, 1.0e-6);
        assert!(
            result.is_ok(),
            "solve_newton_step should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
        );
    }

    #[test]
    fn sae_arrow_schur_beta_quadratic_model_matches_penalized_loss_change() {
        let coords = array![[0.10], [0.35], [0.80]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.65], [-0.45], [0.25]],
            array![[3.0, 0.4, -0.2], [0.1, 2.5, 0.3], [-0.5, 0.2, 1.8]],
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20], [-0.10], [0.45]];
        let rho = SaeManifoldRho::new(0.0, 1.3_f64.ln(), vec![array![0.9_f64.ln()]]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();

        let beta0 = term.flatten_beta();
        let loss0 = term.loss(target.view(), &rho).unwrap().total();
        let mut direction = sys.gb.mapv(|v| -v);
        let direction_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(direction_norm > 1.0e-12);
        for value in direction.iter_mut() {
            *value /= direction_norm;
        }

        let epsilon = 1.0e-3;
        let delta = direction.mapv(|v| epsilon * v);
        let beta_trial = beta0 + &delta;
        term.set_flat_beta(beta_trial.view()).unwrap();
        let actual = term.loss(target.view(), &rho).unwrap().total() - loss0;

        let linear = sys.gb.dot(&delta);
        // Use penalty_op to include all H_ββ contributions (GN + smoothness)
        // rather than reading sys.hbb directly, which no longer holds the
        // smoothness term after the #296 BetaPenaltyOp migration.
        let mut hbb_delta = Array1::<f64>::zeros(delta.len());
        {
            let op = sys.effective_penalty_op();
            let d_slice = delta.as_slice().expect("delta is contiguous");
            let hd_slice = hbb_delta.as_slice_mut().expect("hbb_delta is contiguous");
            op.matvec(d_slice, hd_slice);
        }
        let quadratic = 0.5 * delta.dot(&hbb_delta);
        let predicted = linear + quadratic;
        let error = (actual - predicted).abs();
        assert!(
            error <= 1.0e-4,
            "actual={actual:.12e}, predicted={predicted:.12e}, error={error:.12e}"
        );
    }

    /// `SaeRowLayout::from_dense_weights` must keep, per row, the
    /// top-`k_active_cap` atoms above the magnitude cutoff (always at least
    /// one), with compact coord starts that reproduce the `expand_row`
    /// round-trip back to full-q positions.
    #[test]
    fn sae_row_layout_from_dense_weights_top_k_and_cutoff() {
        // 3 atoms, coord dims [2, 1, 2] ⇒ full q = 3 + 5 = 8.
        let coord_dims = vec![2usize, 1, 2];
        let coord_offsets_full = vec![3usize, 5, 6];
        let assignments = vec![
            // Row 0: weights [0.7, 0.01, 0.29]; cutoff 0.05, cap 2 ⇒ {0, 2}.
            Array1::from_vec(vec![0.7, 0.01, 0.29]),
            // Row 1: weights [0.001, 0.002, 0.0005]; all below cutoff ⇒ keep
            // single largest-magnitude atom {1}.
            Array1::from_vec(vec![0.001, 0.002, 0.0005]),
        ];
        let layout =
            SaeRowLayout::from_dense_weights(&assignments, 2, 0.05, coord_dims, coord_offsets_full);
        assert_eq!(layout.active_atoms[0], vec![0, 2]);
        assert_eq!(layout.active_atoms[1], vec![1]);
        // Row 0 compact dim = |{0,2}| + d_0 + d_2 = 2 + 2 + 2 = 6.
        assert_eq!(layout.row_q_active(0), 6);
        // Row 1 compact dim = 1 + d_1 = 1 + 1 = 2.
        assert_eq!(layout.row_q_active(1), 2);
        // expand_row round-trip for row 0: compact [logit0, logit2, t0_0,
        // t0_1, t2_0, t2_1] → full-q with zeros for inactive atom 1.
        let compact = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut full = vec![0.0_f64; 8];
        layout.expand_row(0, &compact, &mut full);
        // logits: full[0] = atom0 logit, full[2] = atom2 logit, full[1] = 0.
        assert_eq!(full[0], 1.0);
        assert_eq!(full[1], 0.0);
        assert_eq!(full[2], 2.0);
        // coords: atom0 at offset 3 (d=2), atom2 at offset 6 (d=2); atom1
        // (offset 5, d=1) is inactive ⇒ zero.
        assert_eq!(full[3], 3.0);
        assert_eq!(full[4], 4.0);
        assert_eq!(full[5], 0.0);
        assert_eq!(full[6], 5.0);
        assert_eq!(full[7], 6.0);
    }

    /// MechanismSparsityPenalty must reach the SAE arrow-Schur system's
    /// `gb` (beta-tier gradient) when its target slice is shaped to match a
    /// single-atom decoder block (M, p_out). The group lasso over rows of
    /// that (M, p_out) matrix translates to a non-zero gradient on every
    /// (basis_row, feature) entry whose corresponding decoder coefficient
    /// is non-zero, and the FFI-side `"beta"` latent block is what makes
    /// the descriptor builder see exactly that target shape.
    #[test]
    fn sae_mechsparsity_beta_block_routes_through_arrow_schur_gb() {
        let coords = array![[0.10], [0.35], [0.80]];
        let (phi, jet) = periodic_basis(&coords);
        // Decoder shape: (M=3 basis × p=4 features); flatten_beta lays out
        // [basis_col * p + feature] which is exactly the (M, p) row-major
        // shape MechSparsity targets.
        let decoder = array![
            [0.7, -0.2, 0.05, 0.4],
            [-0.5, 0.6, -0.1, 0.3],
            [0.2, 0.0, -0.4, -0.6],
        ];
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder.clone(),
            Array2::<f64>::eye(3),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Two groups partition the 4 features: {0,1} and {2,3}. Each row
        // of the decoder block has non-zero entries in both groups, so the
        // group-norm denominator is finite and every column-feature in a
        // non-trivially-loaded group sees a positive penalty gradient.
        let m = 3usize;
        let p = 4usize;
        let slice = PsiSlice::full(m * p, Some(m));
        let penalty = MechanismSparsityPenalty::new(
            slice,
            vec![vec![0, 1], vec![2, 3]],
            1.0,
            1.0e-6,
            (term.n_obs()) as f64,
            false,
        )
        .unwrap();
        let mut registry = AnalyticPenaltyRegistry::new();
        registry.push(AnalyticPenaltyKind::MechanismSparsity(Arc::new(penalty)));

        let target = array![
            [0.20, 0.10, -0.05, 0.25],
            [-0.10, 0.30, 0.15, -0.20],
            [0.45, -0.05, 0.10, 0.30],
        ];
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, Some(&registry))
            .unwrap();

        assert_eq!(sys.gb.len(), m * p, "gb should match flatten_beta length");
        let mut absmax = 0.0_f64;
        for v in sys.gb.iter().copied() {
            assert!(v.is_finite());
            if v.abs() > absmax {
                absmax = v.abs();
            }
        }
        assert!(
            absmax > 1.0e-6,
            "MechSparsity must inject a non-trivial gradient into the SAE arrow-Schur gb; absmax={absmax:.3e}"
        );
        // Closed-form check on the ISOLATED MechSparsity contribution. `sys.gb`
        // is the FULL penalized β-gradient (data-fit + decoder-smoothness +
        // MechSparsity), so comparing a raw `gb` entry to the penalty-only
        // closed form is wrong (it omits the data-fit and smoothness terms).
        // Difference two assemblies — with and without the registry — to recover
        // exactly the penalty gradient `Δgb = gb_with − gb_without`, then compare
        // that delta to `MechanismSparsityPenalty::grad_target` at (basis=1,
        // feat=0):
        //   w / sqrt(|G|) · b[1,0] / ||b[1, group={0,1}]||
        // group {0,1} has size 2 → factor sqrt(2); unit weight, tiny eps.
        let sys_no_penalty = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let beta = term.flatten_beta();
        let expected = {
            // ||b[1, {0,1}]|| ≈ sqrt(0.5² + 0.6²) = sqrt(0.61)
            let s = (0.5_f64.powi(2) + 0.6_f64.powi(2) + 1.0e-12).sqrt();
            (2.0_f64).sqrt() * (-0.5_f64) / s
        };
        let delta = sys.gb[1 * p + 0] - sys_no_penalty.gb[1 * p + 0];
        assert!(
            (delta - expected).abs() <= 1.0e-6,
            "expected MechSparsity gb contribution at (basis=1, feat=0) ≈ {expected:.6e}, \
             got Δgb={delta:.6e} (gb_with={:.6e}, gb_without={:.6e}, beta entry = {})",
            sys.gb[1 * p + 0],
            sys_no_penalty.gb[1 * p + 0],
            beta[1 * p + 0]
        );
    }

    /// Smoothed sum of singular values of an `m × p` matrix, matching
    /// `NuclearNormPenalty::value` (used by the spectrum-shrinkage assertion).
    fn smoothed_nuclear_norm(decoder: &Array2<f64>, eps: f64) -> f64 {
        let (_u, s, _vt) = decoder.clone().svd(false, false).unwrap();
        s.iter()
            .map(|sigma| (sigma * sigma + eps * eps).sqrt() - eps)
            .sum()
    }

    /// NuclearNormPenalty is a Psi-tier penalty, but inside the SAE term it is
    /// redirected to the per-atom decoder (β) block rather than the coord "t"
    /// row block (#672). This pins three things:
    ///   1. `validate_analytic_penalty_registry` does NOT refuse it (it bypasses
    ///      the row-block requirement).
    ///   2. It injects a non-trivial gradient into the arrow-Schur `gb`
    ///      (β-tier gradient) equal to the analytic spectral gradient on the
    ///      atom's `(M, p)` decoder block.
    ///   3. A gradient-descent step along `gb` shrinks the decoder block's
    ///      (smoothed) singular spectrum — the rank-shrinkage objective.
    #[test]
    fn sae_nuclear_norm_beta_block_routes_through_gb_and_shrinks_spectrum() {
        let coords = array![[0.10], [0.35], [0.80]];
        let (phi, jet) = periodic_basis(&coords);
        // Full-rank (M=3 basis × p=4 features) decoder block. flatten_beta lays
        // it out [basis_row * p + feature] = the (M, p) row-major shape the
        // nuclear-norm penalty treats as a matrix.
        let decoder = array![
            [0.9, -0.2, 0.05, 0.4],
            [-0.5, 0.7, -0.1, 0.3],
            [0.2, 0.1, -0.8, -0.6],
        ];
        let m = 3usize;
        let p = 4usize;
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder.clone(),
            Array2::<f64>::eye(3),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Base penalty: the per-atom dispatch in `add_sae_beta_penalty`
        // overrides n_eff and target, so these initial values only need to
        // construct validly. Here n_eff = p (the "beta" block declares n=p_out,
        // d=Σ M_k); the SAE term rebuilds n_eff = M_k, latent_dim = p per atom.
        let eps = 1.0e-6;
        let slice = PsiSlice::full(m * p, Some(m));
        let penalty = NuclearNormPenalty::new(slice, 1.0, p, eps, None, false).unwrap();
        let mut registry = AnalyticPenaltyRegistry::new();
        registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));

        // Must NOT be refused by the construction-time validator.
        term.validate_analytic_penalty_registry(&registry)
            .expect("NuclearNorm must be accepted (redirected to the β block)");

        let target = array![
            [0.20, 0.10, -0.05, 0.25],
            [-0.10, 0.30, 0.15, -0.20],
            [0.45, -0.05, 0.10, 0.30],
        ];
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
        let baseline = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, Some(&registry))
            .unwrap();

        assert_eq!(sys.gb.len(), m * p, "gb should match flatten_beta length");
        assert_eq!(
            baseline.gb.len(),
            m * p,
            "baseline gb should match flatten_beta length"
        );
        let mut absmax = 0.0_f64;
        let mut penalty_grad = Array1::<f64>::zeros(m * p);
        for ((dst, sys_g), baseline_g) in penalty_grad
            .iter_mut()
            .zip(sys.gb.iter())
            .zip(baseline.gb.iter())
        {
            let v = *sys_g - *baseline_g;
            assert!(v.is_finite());
            *dst = v;
            absmax = absmax.max(v.abs());
        }
        assert!(
            absmax > 1.0e-6,
            "NuclearNorm must inject a non-trivial gradient into the SAE \
             arrow-Schur gb; absmax={absmax:.3e}"
        );

        // The penalty contribution to gb must equal the analytic spectral
        // gradient of the per-atom `(M, p)` block (penalty_scale defaults to
        // 1.0 in assemble_arrow_schur). Reconstruct the reference directly.
        let per_atom = NuclearNormPenalty::new(
            PsiSlice {
                range: 0..m * p,
                latent_dim: Some(p),
            },
            1.0,
            m,
            eps,
            None,
            false,
        )
        .unwrap();
        let beta = term.flatten_beta();
        let ref_grad = per_atom.grad_target(beta.view(), Array1::<f64>::zeros(0).view());
        for j in 0..m * p {
            assert!(
                (penalty_grad[j] - ref_grad[j]).abs() <= 1.0e-9,
                "penalty gb[{j}]={:.12e} must equal analytic spectral grad {:.12e}",
                penalty_grad[j],
                ref_grad[j]
            );
        }

        // A gradient-descent step on the decoder block shrinks the smoothed
        // singular spectrum: nuclear-norm is the rank-shrinkage objective.
        let base_norm = smoothed_nuclear_norm(&decoder, eps);
        let step = 1.0e-2;
        let mut shrunk = decoder.clone();
        for ((row, feat), value) in shrunk.indexed_iter_mut() {
            *value -= step * penalty_grad[row * p + feat];
        }
        let shrunk_norm = smoothed_nuclear_norm(&shrunk, eps);
        assert!(
            shrunk_norm < base_norm,
            "a step along gb must shrink the decoder spectrum: \
             before={base_norm:.9e}, after={shrunk_norm:.9e}"
        );

        // The β curvature block must be PSD. SAE returns `hbb` to the term
        // workspace after lowering the block into the effective penalty operator,
        // so validate the operator diagonal rather than the recycled field.
        assert!(sys.hbb.is_empty());
        let mut hbb_diag = vec![0.0_f64; m * p];
        sys.effective_penalty_op().diagonal(&mut hbb_diag);
        for i in 0..m * p {
            assert!(
                hbb_diag[i] >= -1.0e-9,
                "hbb diagonal must be non-negative (PSD majorizer); hbb[{i},{i}]={:.3e}",
                hbb_diag[i]
            );
        }
    }

    #[derive(Debug)]
    struct TestPeriodicEvaluator;

    impl SaeBasisEvaluator for TestPeriodicEvaluator {
        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array4<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "TestPeriodicEvaluator::second_jet_dyn: expected latent_dim 1, got {}",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array5<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "TestPeriodicEvaluator::third_jet_dyn: expected latent_dim 1, got {}",
                    coords.ncols()
                )));
            }
            None
        }

        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            Ok(periodic_basis(&coords.to_owned()))
        }
    }

    #[derive(Debug, Clone)]
    struct SaeFdWorst {
        index: usize,
        analytic: f64,
        finite_difference: f64,
        absolute_error: f64,
        relative_error: f64,
    }

    impl SaeFdWorst {
        fn new() -> Self {
            Self {
                index: 0,
                analytic: 0.0,
                finite_difference: 0.0,
                absolute_error: 0.0,
                relative_error: 0.0,
            }
        }

        fn observe(&mut self, index: usize, analytic: f64, finite_difference: f64) {
            let absolute_error = (analytic - finite_difference).abs();
            let scale = analytic.abs().max(finite_difference.abs()).max(1.0e-9);
            let relative_error = absolute_error / scale;
            if relative_error > self.relative_error {
                self.index = index;
                self.analytic = analytic;
                self.finite_difference = finite_difference;
                self.absolute_error = absolute_error;
                self.relative_error = relative_error;
            }
        }
    }

    #[derive(Debug, Clone)]
    struct SaeFdBlockReport {
        label: String,
        base_loss: f64,
        coord: SaeFdWorst,
        decoder: SaeFdWorst,
    }

    fn sae_fd_decoder(n_basis: usize, p_out: usize) -> Array2<f64> {
        let mut decoder = Array2::<f64>::zeros((n_basis, p_out));
        for basis in 0..n_basis {
            for out_col in 0..p_out {
                let phase = 0.73 * ((basis + 1) as f64) + 1.17 * ((out_col + 1) as f64);
                decoder[[basis, out_col]] = 0.16 * phase.sin() + 0.05 * (1.9 * phase).cos();
            }
        }
        decoder
    }

    fn sae_fd_target(n_obs: usize, p_out: usize) -> Array2<f64> {
        let mut target = Array2::<f64>::zeros((n_obs, p_out));
        for row in 0..n_obs {
            for out_col in 0..p_out {
                let x = (row as f64) + 1.0;
                let y = (out_col as f64) + 1.0;
                target[[row, out_col]] =
                    0.21 * (0.31 * x + 0.47 * y).sin() - 0.13 * (0.19 * x * y).cos();
            }
        }
        target
    }

    fn sae_fd_coords(label: &str, n_obs: usize) -> Array2<f64> {
        let mut coords = Array2::<f64>::zeros((n_obs, 1));
        for row in 0..n_obs {
            let x = row as f64;
            coords[[row, 0]] = match label {
                "periodic_d1" => 0.07 + 0.043 * x + 0.004 * (1.3 * x).sin(),
                "euclidean_d1" => -0.46 + 0.048 * x + 0.006 * (1.7 * x).cos(),
                other => panic!("unknown SAE FD case label {other}"),
            };
        }
        coords
    }

    fn sae_fd_term(label: &str) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let n_obs = 20usize;
        let p_out = 3usize;
        let coords = sae_fd_coords(label, n_obs);
        let (basis_kind, phi, jet, n_basis, atom) = match label {
            "periodic_d1" => {
                let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
                let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
                let n_basis = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    "periodic_d1",
                    SaeAtomBasisKind::Periodic,
                    1,
                    phi.clone(),
                    jet.clone(),
                    sae_fd_decoder(n_basis, p_out),
                    Array2::<f64>::eye(n_basis),
                )
                .unwrap()
                .with_basis_second_jet(evaluator);
                (SaeAtomBasisKind::Periodic, phi, jet, n_basis, atom)
            }
            "euclidean_d1" => {
                let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
                let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
                let n_basis = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    "euclidean_d1",
                    SaeAtomBasisKind::EuclideanPatch,
                    1,
                    phi.clone(),
                    jet.clone(),
                    sae_fd_decoder(n_basis, p_out),
                    Array2::<f64>::eye(n_basis),
                )
                .unwrap()
                .with_basis_second_jet(evaluator);
                (SaeAtomBasisKind::EuclideanPatch, phi, jet, n_basis, atom)
            }
            other => panic!("unknown SAE FD case label {other}"),
        };
        assert_eq!(
            basis_kind.latent_manifold(1),
            atom.basis_kind.latent_manifold(1)
        );
        assert_eq!(phi.dim(), (n_obs, n_basis));
        assert_eq!(jet.dim(), (n_obs, n_basis, 1));

        let manifold = atom.basis_kind.latent_manifold(1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n_obs, 1)),
            vec![coords],
            vec![manifold],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = sae_fd_target(n_obs, p_out);
        let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), vec![array![-30.0]]);
        (term, target, rho)
    }

    fn sae_fd_refresh(term: &mut SaeManifoldTerm) {
        let coords = term.assignment.coords[0].as_matrix();
        term.atoms[0].refresh_basis(coords.view()).unwrap();
    }

    fn sae_fd_set_coord(term: &mut SaeManifoldTerm, row: usize, value: f64) {
        let mut flat = term.assignment.coords[0].as_flat().clone();
        flat[row] = value;
        term.assignment.coords[0].set_flat(flat.view());
        sae_fd_refresh(term);
    }

    fn sae_fd_total_loss(
        term: &SaeManifoldTerm,
        target: &Array2<f64>,
        rho: &SaeManifoldRho,
    ) -> f64 {
        term.loss(target.view(), rho).unwrap().total()
    }

    fn sae_fd_check_case(label: &str) -> SaeFdBlockReport {
        let epsilon = 1.0e-6;
        let (term, target, rho) = sae_fd_term(label);
        let base_loss = sae_fd_total_loss(&term, &target, &rho);
        assert!(base_loss.is_finite(), "{label}: base loss is not finite");

        let mut assembled = term.clone();
        sae_fd_refresh(&mut assembled);
        let sys = assembled
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        assert_eq!(sys.rows.len(), term.n_obs());
        assert_eq!(sys.gb.len(), term.beta_dim());
        for row in 0..term.n_obs() {
            assert_eq!(
                sys.rows[row].gt.len(),
                1,
                "{label}: K=1 softmax d=1 should expose exactly one row coordinate gradient"
            );
        }

        let mut coord = SaeFdWorst::new();
        let base_coords = term.assignment.coords[0].as_flat().clone();
        for row in 0..term.n_obs() {
            let mut plus = term.clone();
            sae_fd_set_coord(&mut plus, row, base_coords[row] + epsilon);
            let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

            let mut minus = term.clone();
            sae_fd_set_coord(&mut minus, row, base_coords[row] - epsilon);
            let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

            let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
            coord.observe(row, sys.rows[row].gt[0], finite_difference);
        }

        let mut decoder = SaeFdWorst::new();
        let beta = term.flatten_beta();
        for beta_idx in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[beta_idx] += epsilon;
            let mut plus = term.clone();
            plus.set_flat_beta(beta_plus.view()).unwrap();
            sae_fd_refresh(&mut plus);
            let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

            let mut beta_minus = beta.clone();
            beta_minus[beta_idx] -= epsilon;
            let mut minus = term.clone();
            minus.set_flat_beta(beta_minus.view()).unwrap();
            sae_fd_refresh(&mut minus);
            let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

            let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
            decoder.observe(beta_idx, sys.gb[beta_idx], finite_difference);
        }

        SaeFdBlockReport {
            label: label.to_string(),
            base_loss,
            coord,
            decoder,
        }
    }

    /// Which manifold/basis a penalty-FD case runs on.
    #[derive(Clone, Copy)]
    enum SaePenCaseKind {
        EuclideanD1,
        PeriodicD1,
        EuclideanD2,
    }

    /// Which analytic penalty a penalty-FD case exercises.
    #[derive(Clone, Copy)]
    enum SaePenKind {
        Isometry,
        Ard,
        ScadMcp,
        NuclearNorm,
        DecoderIncoherence,
    }

    /// Single-atom SAE term on the requested manifold for the penalty-FD checks.
    /// Mirrors `sae_fd_term` but exposes the analytic second jet the Isometry
    /// penalty needs and allows a chosen latent dimension.
    fn sae_pen_term(
        kind: SaePenCaseKind,
    ) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, PsiSlice) {
        let n_obs = 12usize;
        let p_out = 3usize;
        let (coords, latent_dim, atom): (Array2<f64>, usize, SaeManifoldAtom) = match kind {
            SaePenCaseKind::PeriodicD1 => {
                let mut coords = Array2::<f64>::zeros((n_obs, 1));
                for row in 0..n_obs {
                    let x = row as f64;
                    coords[[row, 0]] = 0.11 + 0.037 * x + 0.004 * (1.3 * x).sin();
                }
                let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
                let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
                let n_basis = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    "periodic_d1",
                    SaeAtomBasisKind::Periodic,
                    1,
                    phi,
                    jet,
                    sae_fd_decoder(n_basis, p_out),
                    Array2::<f64>::eye(n_basis),
                )
                .unwrap()
                .with_basis_second_jet(evaluator);
                (coords, 1, atom)
            }
            SaePenCaseKind::EuclideanD1 => {
                let mut coords = Array2::<f64>::zeros((n_obs, 1));
                for row in 0..n_obs {
                    let x = row as f64;
                    coords[[row, 0]] = -0.41 + 0.052 * x + 0.006 * (1.7 * x).cos();
                }
                let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
                let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
                let n_basis = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    "euclidean_d1",
                    SaeAtomBasisKind::EuclideanPatch,
                    1,
                    phi,
                    jet,
                    sae_fd_decoder(n_basis, p_out),
                    Array2::<f64>::eye(n_basis),
                )
                .unwrap()
                .with_basis_second_jet(evaluator);
                (coords, 1, atom)
            }
            SaePenCaseKind::EuclideanD2 => {
                let mut coords = Array2::<f64>::zeros((n_obs, 2));
                for row in 0..n_obs {
                    let x = row as f64;
                    coords[[row, 0]] = -0.33 + 0.041 * x + 0.005 * (1.1 * x).cos();
                    coords[[row, 1]] = 0.27 - 0.036 * x + 0.004 * (0.9 * x).sin();
                }
                let evaluator = Arc::new(EuclideanPatchEvaluator::new(2, 2).unwrap());
                let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
                let n_basis = phi.ncols();
                let atom = SaeManifoldAtom::new(
                    "euclidean_d2",
                    SaeAtomBasisKind::EuclideanPatch,
                    2,
                    phi,
                    jet,
                    sae_fd_decoder(n_basis, p_out),
                    Array2::<f64>::eye(n_basis),
                )
                .unwrap()
                .with_basis_second_jet(evaluator);
                (coords, 2, atom)
            }
        };
        let manifold = atom.basis_kind.latent_manifold(latent_dim);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n_obs, 1)),
            vec![coords],
            vec![manifold],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = sae_fd_target(n_obs, p_out);
        // Suppress the built-in ARD / smoothness contributions so the registered
        // analytic penalty is the only penalty beyond data-fit + assignment prior.
        let log_ard = vec![Array1::from_elem(latent_dim, -30.0_f64)];
        let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), log_ard);
        let slice = PsiSlice {
            range: 0..n_obs * latent_dim,
            latent_dim: Some(latent_dim),
        };
        (term, target, rho, slice)
    }

    /// Two-atom K=2 SAE term for the DecoderIncoherence FD check. Both atoms are
    /// d=1 euclidean patches so the β block is `[B_1 (M×p), B_2 (M×p)]`.
    fn sae_pen_term_k2() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let n_obs = 12usize;
        let p_out = 3usize;
        let mut atoms = Vec::with_capacity(2);
        let mut coord_blocks = Vec::with_capacity(2);
        for atom_idx in 0..2usize {
            let mut coords = Array2::<f64>::zeros((n_obs, 1));
            for row in 0..n_obs {
                let x = row as f64;
                coords[[row, 0]] = if atom_idx == 0 {
                    -0.41 + 0.052 * x + 0.006 * (1.7 * x).cos()
                } else {
                    0.18 + 0.039 * x + 0.005 * (1.1 * x).sin()
                };
            }
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let mut decoder = sae_fd_decoder(n_basis, p_out);
            if atom_idx == 1 {
                for basis in 0..n_basis {
                    for out_col in 0..p_out {
                        decoder[[basis, out_col]] += 0.07 * ((basis + out_col) as f64 + 1.0).cos();
                    }
                }
            }
            let atom = SaeManifoldAtom::new(
                "euclidean_d1",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            atoms.push(atom);
            coord_blocks.push(coords);
        }
        let manifold = LatentManifold::Euclidean;
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::from_elem((n_obs, 2), 0.2),
            coord_blocks,
            vec![manifold.clone(), manifold],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let target = sae_fd_target(n_obs, p_out);
        let log_ard = vec![
            Array1::from_elem(1, -30.0_f64),
            Array1::from_elem(1, -30.0_f64),
        ];
        let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), log_ard);
        (term, target, rho)
    }

    /// Registry holding exactly one analytic penalty of the requested kind,
    /// sized for `term`'s coord / β block.
    fn sae_pen_registry(
        pen: SaePenKind,
        coord_slice: &PsiSlice,
        n_obs: usize,
        latent_dim: usize,
        beta_len: usize,
        p_out: usize,
    ) -> AnalyticPenaltyRegistry {
        use crate::terms::analytic_penalties::PenaltyConcavity;
        use crate::terms::analytic_penalties::ScadMcpPenalty;
        let mut registry = AnalyticPenaltyRegistry::new();
        match pen {
            SaePenKind::Isometry => {
                let penalty = IsometryPenalty::new_euclidean(coord_slice.clone(), latent_dim);
                registry.push(AnalyticPenaltyKind::Isometry(Arc::new(penalty)));
            }
            SaePenKind::Ard => {
                let penalty = ARDPenalty::new(coord_slice.clone(), latent_dim);
                registry.push(AnalyticPenaltyKind::Ard(Arc::new(penalty)));
            }
            SaePenKind::ScadMcp => {
                let penalty = ScadMcpPenalty::new(
                    coord_slice.clone(),
                    0.5,
                    n_obs,
                    3.0,
                    1.0e-4,
                    PenaltyConcavity::Mcp,
                    false,
                )
                .unwrap();
                registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(penalty)));
            }
            SaePenKind::NuclearNorm => {
                let slice = PsiSlice {
                    range: 0..beta_len,
                    latent_dim: Some(beta_len / p_out),
                };
                let penalty =
                    NuclearNormPenalty::new(slice, 0.7, p_out, 1.0e-4, None, false).unwrap();
                registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));
            }
            SaePenKind::DecoderIncoherence => {
                let m_per = beta_len / (2 * p_out);
                let slice = PsiSlice {
                    range: 0..beta_len,
                    latent_dim: Some(beta_len / p_out),
                };
                let penalty = DecoderIncoherencePenalty::new(
                    slice,
                    vec![m_per, m_per],
                    p_out,
                    Array2::<f64>::from_elem((2, 2), 0.5),
                    0.6,
                    false,
                )
                .unwrap();
                registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(penalty)));
            }
        }
        registry
    }

    /// FD-check the assembled gradient (`gt` / `gb`) against central differences
    /// of `penalized_objective_total` with the registry's single analytic penalty
    /// ACTIVE. Softmax mode always assembles the dense uniform row layout, so atom
    /// `atom_idx`'s axis `a` for row `r` lives at `sys.rows[r].gt[off + a]` with
    /// `off = coord_offsets()[atom_idx]` (a per-atom column offset, not a row
    /// offset); the row index is the plain observation row.
    fn sae_pen_fd_check(
        label: &str,
        term: &SaeManifoldTerm,
        target: &Array2<f64>,
        rho: &SaeManifoldRho,
        registry: &AnalyticPenaltyRegistry,
    ) -> SaeFdBlockReport {
        let epsilon = 1.0e-6;
        let base_obj = term
            .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
            .unwrap();
        assert!(base_obj.is_finite(), "{label}: base objective not finite");

        let mut assembled = term.clone();
        let sys = assembled
            .assemble_arrow_schur(target.view(), rho, Some(registry))
            .unwrap();

        let mut coord = SaeFdWorst::new();
        let coord_offsets = term.assignment.coord_offsets();
        for atom_idx in 0..term.k_atoms() {
            let off = coord_offsets[atom_idx];
            let d = term.assignment.coords[atom_idx].latent_dim();
            let base_flat = term.assignment.coords[atom_idx].as_flat().clone();
            let n_atom = base_flat.len() / d;
            for row in 0..n_atom {
                for axis in 0..d {
                    let lin = row * d + axis;
                    let mut plus = term.clone();
                    let mut flat_p = base_flat.clone();
                    flat_p[lin] += epsilon;
                    plus.assignment.coords[atom_idx].set_flat(flat_p.view());
                    let coords_p = plus.assignment.coords[atom_idx].as_matrix();
                    plus.atoms[atom_idx].refresh_basis(coords_p.view()).unwrap();
                    let obj_p = plus
                        .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
                        .unwrap();

                    let mut minus = term.clone();
                    let mut flat_m = base_flat.clone();
                    flat_m[lin] -= epsilon;
                    minus.assignment.coords[atom_idx].set_flat(flat_m.view());
                    let coords_m = minus.assignment.coords[atom_idx].as_matrix();
                    minus.atoms[atom_idx]
                        .refresh_basis(coords_m.view())
                        .unwrap();
                    let obj_m = minus
                        .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
                        .unwrap();

                    let finite_difference = (obj_p - obj_m) / (2.0 * epsilon);
                    coord.observe(
                        row * d + axis,
                        sys.rows[row].gt[off + axis],
                        finite_difference,
                    );
                }
            }
        }

        let mut decoder = SaeFdWorst::new();
        let beta = term.flatten_beta();
        for beta_idx in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[beta_idx] += epsilon;
            let mut plus = term.clone();
            plus.set_flat_beta(beta_plus.view()).unwrap();
            let obj_p = plus
                .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
                .unwrap();

            let mut beta_minus = beta.clone();
            beta_minus[beta_idx] -= epsilon;
            let mut minus = term.clone();
            minus.set_flat_beta(beta_minus.view()).unwrap();
            let obj_m = minus
                .penalized_objective_total(target.view(), rho, Some(registry), 1.0)
                .unwrap();

            let finite_difference = (obj_p - obj_m) / (2.0 * epsilon);
            decoder.observe(beta_idx, sys.gb[beta_idx], finite_difference);
        }

        SaeFdBlockReport {
            label: label.to_string(),
            base_loss: base_obj,
            coord,
            decoder,
        }
    }

    /// EXACT agreement between the SAE assembled gradient and the penalized
    /// objective it claims to be the gradient of, per analytic penalty kind.
    /// Central FD of `penalized_objective_total` (penalty ACTIVE) must match the
    /// assembled coord `gt` and decoder `gb`. This pins the isometry decoder
    /// gradient (`∂P/∂B`) that the value path counts but the gradient path used
    /// to drop, alongside ARD, ScadMcp, NuclearNorm, and DecoderIncoherence.
    #[test]
    fn sae_assembled_gradient_matches_penalized_objective_central_fd() {
        let p_out = 3usize;
        let mut reports: Vec<SaeFdBlockReport> = Vec::new();

        let single_cases: &[(&str, SaePenCaseKind, SaePenKind)] = &[
            (
                "isometry_circle_d1",
                SaePenCaseKind::PeriodicD1,
                SaePenKind::Isometry,
            ),
            (
                "isometry_euclid_d2",
                SaePenCaseKind::EuclideanD2,
                SaePenKind::Isometry,
            ),
            ("ard_circle_d1", SaePenCaseKind::PeriodicD1, SaePenKind::Ard),
            (
                "scadmcp_euclid_d1",
                SaePenCaseKind::EuclideanD1,
                SaePenKind::ScadMcp,
            ),
            (
                "nuclearnorm_euclid_d1",
                SaePenCaseKind::EuclideanD1,
                SaePenKind::NuclearNorm,
            ),
        ];
        for (label, case_kind, pen_kind) in single_cases {
            let (term, target, rho, slice) = sae_pen_term(*case_kind);
            let n_obs = term.n_obs();
            let latent_dim = term.assignment.coords[0].latent_dim();
            let beta_len = term.beta_dim();
            let registry = sae_pen_registry(*pen_kind, &slice, n_obs, latent_dim, beta_len, p_out);
            term.validate_analytic_penalty_registry(&registry)
                .expect("penalty registry must validate for the SAE term");
            reports.push(sae_pen_fd_check(label, &term, &target, &rho, &registry));
        }

        {
            let (term, target, rho) = sae_pen_term_k2();
            let beta_len = term.beta_dim();
            let slice = PsiSlice {
                range: 0..beta_len,
                latent_dim: Some(beta_len / p_out),
            };
            let registry = sae_pen_registry(
                SaePenKind::DecoderIncoherence,
                &slice,
                term.n_obs(),
                1,
                beta_len,
                p_out,
            );
            term.validate_analytic_penalty_registry(&registry)
                .expect("DecoderIncoherence registry must validate for the K=2 SAE term");
            reports.push(sae_pen_fd_check(
                "decoder_incoherence_k2",
                &term,
                &target,
                &rho,
                &registry,
            ));
        }

        let relative_tolerance = 1.0e-5;
        let absolute_tolerance = 1.0e-7;
        let mut all_blocks_match = true;
        for report in &reports {
            let coord_ok = report.coord.relative_error <= relative_tolerance
                || report.coord.absolute_error <= absolute_tolerance;
            let decoder_ok = report.decoder.relative_error <= relative_tolerance
                || report.decoder.absolute_error <= absolute_tolerance;
            let metadata_ok = !report.label.is_empty() && report.base_loss.is_finite();
            all_blocks_match = all_blocks_match && metadata_ok && coord_ok && decoder_ok;
        }
        assert!(
            all_blocks_match,
            "SAE assembled gradient does not match central FD of the penalized objective: {reports:#?}"
        );
    }

    #[test]
    fn sae_reml_extra_penalty_energy_counts_live_isometry_once() {
        let p_out = 3usize;
        let (term, _target, _rho, slice) = sae_pen_term(SaePenCaseKind::PeriodicD1);
        let registry = sae_pen_registry(
            SaePenKind::Isometry,
            &slice,
            term.n_obs(),
            term.assignment.coords[0].latent_dim(),
            term.beta_dim(),
            p_out,
        );

        let isometry_energy = term
            .isometry_penalty_value_total(&registry)
            .expect("live isometry value");
        assert!(
            isometry_energy > 0.0,
            "fixture must carry nonzero isometry energy"
        );

        let decoder_energy = term
            .analytic_decoder_penalty_value_total(&registry)
            .expect("decoder penalty value");
        assert_abs_diff_eq!(decoder_energy, 0.0, epsilon = 1.0e-12);

        let extra_energy = term
            .reml_extra_penalty_value_total(&registry)
            .expect("REML extra penalty value");
        assert_abs_diff_eq!(extra_energy, isometry_energy, epsilon = 1.0e-12);
    }

    #[test]
    fn sae_d1_assembled_gradient_matches_loss_central_fd() {
        let reports = vec![
            sae_fd_check_case("euclidean_d1"),
            sae_fd_check_case("periodic_d1"),
        ];
        let relative_tolerance = 3.0e-5;
        let absolute_tolerance = 3.0e-7;
        let mut all_blocks_match = true;
        for report in &reports {
            let coord_ok = report.coord.relative_error <= relative_tolerance
                || report.coord.absolute_error <= absolute_tolerance;
            let decoder_ok = report.decoder.relative_error <= relative_tolerance
                || report.decoder.absolute_error <= absolute_tolerance;
            let metadata_ok = !report.label.is_empty() && report.base_loss.is_finite();
            all_blocks_match = all_blocks_match && metadata_ok && coord_ok && decoder_ok;
        }
        assert!(
            all_blocks_match,
            "SAE d=1 assembled gradient does not match central finite difference: {reports:#?}"
        );
    }

    fn assert_jacobian_matches_central_difference<E: SaeBasisEvaluator>(
        evaluator: &E,
        coords: Array2<f64>,
        tolerance: f64,
    ) {
        let epsilon = 1.0e-6;
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let (n_rows, n_basis) = phi.dim();
        let latent_dim = coords.ncols();
        assert_eq!(jet.dim(), (n_rows, n_basis, latent_dim));

        for row in 0..n_rows {
            for axis in 0..latent_dim {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis]] += epsilon;
                minus[[row, axis]] -= epsilon;
                let (phi_plus, plus_jet) = evaluator.evaluate(plus.view()).unwrap();
                let (phi_minus, minus_jet) = evaluator.evaluate(minus.view()).unwrap();
                assert_eq!(plus_jet.dim(), jet.dim());
                assert_eq!(minus_jet.dim(), jet.dim());

                for basis in 0..n_basis {
                    let finite_difference =
                        (phi_plus[[row, basis]] - phi_minus[[row, basis]]) / (2.0 * epsilon);
                    let analytic = jet[[row, basis, axis]];
                    let error = (analytic - finite_difference).abs();
                    assert!(
                        error <= tolerance,
                        "row={row} basis={basis} axis={axis}: analytic={analytic:.12e}, \
                         finite_difference={finite_difference:.12e}, error={error:.12e}, \
                         tolerance={tolerance:.12e}"
                    );
                }
            }
        }
    }

    #[test]
    fn sae_basis_evaluator_jacobians_match_central_differences() {
        assert_jacobian_matches_central_difference(
            &PeriodicHarmonicEvaluator::new(7).unwrap(),
            array![[-0.37], [0.0], [0.125], [0.41]],
            1.0e-6,
        );

        assert_jacobian_matches_central_difference(
            &RawPeriodicCircleEvaluator::new(3).unwrap(),
            array![[-1.2, 0.3, 2.0], [0.0, -0.4, 0.8], [2.4, 1.1, -0.7]],
            1.0e-6,
        );

        let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
        assert_jacobian_matches_central_difference(
            &SphereChartEvaluator,
            sphere_coords.clone(),
            1.0e-6,
        );
        let (sphere_phi, sphere_jet) = SphereChartEvaluator.evaluate(sphere_coords.view()).unwrap();
        assert_eq!(sphere_phi.dim(), (sphere_coords.nrows(), 7));
        assert_eq!(sphere_jet.dim(), (sphere_coords.nrows(), 7, 2));
        for row in 0..sphere_coords.nrows() {
            let lat = sphere_coords[[row, 0]];
            let lon = sphere_coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let z = slat;
            let dx_dlon = -clat * slon;
            let dy_dlon = clat * clon;
            assert_eq!(sphere_jet[[row, 3, 1]], 0.0);
            assert!((sphere_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
            assert!((sphere_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);
        }

        assert_jacobian_matches_central_difference(
            &AffineCoordinateEvaluator::new(3),
            array![[0.0, -1.0, 2.0], [3.5, 0.25, -0.75]],
            1.0e-6,
        );

        // Torus T^2 with H=3 → 49-column tensor product.
        let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
        assert_jacobian_matches_central_difference(
            &TorusHarmonicEvaluator::new(2, 3).unwrap(),
            torus_coords.clone(),
            1.0e-6,
        );
        let (torus_phi, torus_jet) = TorusHarmonicEvaluator::new(2, 3)
            .unwrap()
            .evaluate(torus_coords.view())
            .unwrap();
        assert_eq!(torus_phi.dim(), (torus_coords.nrows(), 49));
        assert_eq!(torus_jet.dim(), (torus_coords.nrows(), 49, 2));
        for row in 0..torus_coords.nrows() {
            // Column 0 = product of the two constant axis terms = 1.
            assert!((torus_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
            assert!(torus_jet[[row, 0, 0]].abs() <= 1.0e-12);
            assert!(torus_jet[[row, 0, 1]].abs() <= 1.0e-12);
        }
    }

    /// The compact-latent basis kinds must each expose a projection seed grid
    /// that spans their manifold, and the unbounded / basis-linear kinds expose none (their PCA
    /// seed already lands in the convex hull of the training coordinates).
    /// Pins the grid extents the fixed-decoder OOS seed (#628) relies on.
    #[test]
    fn projection_seed_grid_spans_each_compact_manifold() {
        use std::f64::consts::PI;

        // Periodic S¹: `resolution` phases evenly on `[0, 1)` (endpoint
        // excluded — `0` and `1` are the same point on the circle).
        let periodic = SaeAtomBasisKind::Periodic
            .projection_seed_grid(1, 16)
            .unwrap();
        assert_eq!(periodic.dim(), (16, 1));
        for i in 0..16 {
            assert_abs_diff_eq!(periodic[[i, 0]], i as f64 / 16.0, epsilon = 1e-12);
        }
        assert!(periodic.iter().all(|&t| (0.0..1.0).contains(&t)));

        // Sphere lat/lon chart: an `r × r` grid, latitude strictly interior to
        // the chart (poles are degenerate), longitude on `[-π, π)`.
        let r = 6usize;
        let sphere = SaeAtomBasisKind::Sphere.projection_seed_grid(2, r).unwrap();
        assert_eq!(sphere.dim(), (r * r, 2));
        for row in 0..r * r {
            let lat = sphere[[row, 0]];
            let lon = sphere[[row, 1]];
            assert!(
                lat > -PI / 2.0 && lat < PI / 2.0,
                "sphere seed latitude {lat} is not strictly interior to the chart"
            );
            assert!(
                (-PI..PI).contains(&lon),
                "sphere seed longitude {lon} is outside [-π, π)"
            );
        }

        // Unbounded / basis-linear latents expose no grid (default `None`).
        assert!(
            SaeAtomBasisKind::EuclideanPatch
                .projection_seed_grid(2, 64)
                .is_none(),
            "Euclidean-patch (unbounded) atoms must not expose a projection seed grid"
        );
    }

    /// The torus seed grid is the Cartesian product of per-axis `[0, 1)` phase
    /// grids, with the per-axis resolution shrunk geometrically so the *total*
    /// point count stays under a fixed cap as the latent dimension grows. Pins
    /// the cap arithmetic (`per_axis^d ≤ 4096`) the OOS seed depends on so a
    /// high-`d` torus atom never blows up the per-row global-argmin scan.
    #[test]
    fn torus_projection_seed_grid_caps_total_points() {
        // d == 1: dense, no cap (256¹ ≤ 4096).
        let g1 = SaeAtomBasisKind::Torus
            .projection_seed_grid(1, 256)
            .unwrap();
        assert_eq!(g1.dim(), (256, 1));

        // d == 3: per-axis shrunk to the largest `p` with `p³ ≤ 4096`, i.e.
        // `p = 16` ⇒ exactly 4096 points.
        let g3 = SaeAtomBasisKind::Torus
            .projection_seed_grid(3, 256)
            .unwrap();
        assert_eq!(g3.ncols(), 3);
        assert_eq!(g3.nrows(), 16 * 16 * 16);
        assert!(
            g3.nrows() <= 4096,
            "torus d=3 seed grid has {} points, over the 4096 cap",
            g3.nrows()
        );
        assert!(
            g3.iter().all(|&t| (0.0..1.0).contains(&t)),
            "every torus seed coordinate must be a phase on [0, 1)"
        );
        // Full Cartesian product: each axis takes exactly `per_axis` distinct
        // phase values.
        for axis in 0..3 {
            let mut vals: Vec<f64> = g3.column(axis).iter().copied().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();
            assert_eq!(
                vals.len(),
                16,
                "torus seed axis {axis} should take 16 distinct phases"
            );
        }

        // d == 12: the coarsest dense grid is `2^12 = 4096`, exactly the cap —
        // still emitted (per_axis floors at 2).
        let g12 = SaeAtomBasisKind::Torus
            .projection_seed_grid(12, 256)
            .unwrap();
        assert_eq!(g12.nrows(), 1usize << 12);
        assert!(g12.nrows() <= 4096);

        // d == 13: even the coarsest dense grid (`2^13 = 8192`) exceeds the
        // cap, so no on-manifold grid can satisfy it. The evaluator must return
        // `None` and let the row fall back to its PCA seed rather than allocate
        // a runaway `2^d`-row grid for the per-row global-argmin scan to walk.
        assert!(
            SaeAtomBasisKind::Torus
                .projection_seed_grid(13, 256)
                .is_none(),
            "torus d=13 seed grid (2^13 > 4096) must fall back to None, not blow up the cap"
        );
    }

    /// `seed_coords_by_decoder_projection` must replace each cold coordinate
    /// with the grid point whose frozen-decoder decode is closest to the target
    /// row, and refresh the atom basis there. Built on a decoder that maps the
    /// circle injectively into `ℝ²` (`decode(t) = (sin 2πt, cos 2πt)`) so the
    /// per-row global argmin is unambiguous. Direct Rust pin for the #628 OOS
    /// seed, complementing the Python oracle end-to-end test.
    #[test]
    fn seed_coords_by_decoder_projection_lands_on_grid_minimiser() {
        use std::f64::consts::PI;

        let resolution = 8usize;
        // Deliberately wrong cold seed for both rows.
        let init_coords = array![[0.05], [0.05]];
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
        // (basis = [1, sin, cos]) × (2 output channels): decode(t) = (sin, cos).
        let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            // `K = logits.ncols()`; a single softmax atom is one logit column
            // (the lone simplex coordinate, pinned to 1.0 in `try_assignments_row`).
            Array2::<f64>::zeros((2, 1)),
            vec![init_coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Targets sit exactly on two distinct grid phases `k / resolution`.
        let phases = [3usize, 6usize];
        let mut target = Array2::<f64>::zeros((2, 2));
        for (row, &k) in phases.iter().enumerate() {
            let t = k as f64 / resolution as f64;
            target[[row, 0]] = (2.0 * PI * t).sin();
            target[[row, 1]] = (2.0 * PI * t).cos();
        }

        term.seed_coords_by_decoder_projection(target.view(), resolution)
            .unwrap();

        // Each row was seeded onto its exact grid minimiser …
        let seeded = term.assignment.coords[0].as_matrix();
        let mut expected_coords = Array2::<f64>::zeros((2, 1));
        for (row, &k) in phases.iter().enumerate() {
            let expected = k as f64 / resolution as f64;
            assert_abs_diff_eq!(seeded[[row, 0]], expected, epsilon = 1e-12);
            expected_coords[[row, 0]] = expected;
        }
        // … and the basis cache was refreshed at the seeded coordinates.
        let (phi_expected, _) = evaluator.evaluate(expected_coords.view()).unwrap();
        assert_abs_diff_eq!(
            (&term.atoms[0].basis_values - &phi_expected)
                .mapv(f64::abs)
                .sum(),
            0.0,
            epsilon = 1e-12
        );
    }

    /// A target whose shape does not match `(n_obs, output_dim)` is a caller
    /// bug and must surface as an error rather than silently mis-seeding.
    #[test]
    fn seed_coords_by_decoder_projection_rejects_shape_mismatch() {
        let init_coords = array![[0.05], [0.05]];
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
        let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((2, 1)),
            vec![init_coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

        // Output dim is 2; pass a 3-column target.
        let bad_target = Array2::<f64>::zeros((2, 3));
        let err = term
            .seed_coords_by_decoder_projection(bad_target.view(), 8)
            .unwrap_err();
        assert!(
            err.contains("target shape"),
            "expected a target-shape error, got: {err}"
        );
    }

    /// Parity guard for the sphere chart: the shared engine
    /// [`sphere_chart_basis_jet`] is the single source of derivative truth used
    /// by both the core SAE path ([`SphereChartEvaluator::evaluate`]) and the
    /// PyFFI `sphere_chart_basis_with_jet` helper, which route through the exact
    /// same function. The basis and its jet are now the *exact* analytic ones —
    /// `C^∞` in `(lat, lon)` with no clamp and no binary `chain_lat` gate — so
    /// this pins that the jet equals the closed-form analytic derivative at
    /// interior, boundary (`|lat| = π/2`), and beyond-`π/2` latitudes alike.
    #[test]
    fn sphere_chart_basis_jet_is_single_source_of_truth() {
        // A mix of interior and former clamp-boundary / beyond-π/2 latitudes;
        // the embedding and its jet are smooth everywhere, so all rows must hit
        // the same exact analytic formulas.
        let coords = array![
            [-1.2, -2.4],                         // interior
            [0.35, 0.9],                          // interior
            [std::f64::consts::FRAC_PI_2, 0.4],   // upper boundary (former gate)
            [-std::f64::consts::FRAC_PI_2, -1.1], // lower boundary (former gate)
            [2.3, 0.7],                           // beyond +π/2
            [-3.0, 1.9],                          // beyond -π/2
        ];

        // The core evaluator adapter must be bit-identical to the shared engine
        // — they are the same code path, so any difference is a regression in
        // the thin adapter rather than a tolerance question.
        let (engine_phi, engine_jet) = sphere_chart_basis_jet(coords.view()).unwrap();
        let (adapter_phi, adapter_jet) = SphereChartEvaluator.evaluate(coords.view()).unwrap();
        assert_eq!(engine_phi, adapter_phi);
        assert_eq!(engine_jet, adapter_jet);

        for row in 0..coords.nrows() {
            // No clamp: the basis uses the raw latitude directly.
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;

            // Basis is the unit-sphere embedding evaluated at the raw latitude.
            assert!((engine_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 1]] - x).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 2]] - y).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 3]] - z).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 4]] - x * y).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 5]] - y * z).abs() <= 1.0e-12);
            assert!((engine_phi[[row, 6]] - x * z).abs() <= 1.0e-12);

            // Longitude derivatives.
            let dx_dlon = -clat * slon;
            let dy_dlon = clat * clon;
            assert!((engine_jet[[row, 1, 1]] - dx_dlon).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 2, 1]] - dy_dlon).abs() <= 1.0e-12);
            assert_eq!(engine_jet[[row, 3, 1]], 0.0);
            assert!((engine_jet[[row, 4, 1]] - (dx_dlon * y + x * dy_dlon)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);

            // Latitude derivatives are the exact analytic values at EVERY row,
            // including the former clamp boundary — no gating to zero. At the
            // upper boundary lat = +π/2 the analytic dz/dlat = cos(π/2) = 0
            // naturally (no discontinuous override), while dx/dlat, dy/dlat are
            // nonzero whenever cos(lon)/sin(lon) are.
            let dx_dlat = -slat * clon;
            let dy_dlat = -slat * slon;
            let dz_dlat = clat;
            assert!((engine_jet[[row, 1, 0]] - dx_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 2, 0]] - dy_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 3, 0]] - dz_dlat).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 4, 0]] - (dx_dlat * y + x * dy_dlat)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 5, 0]] - (dy_dlat * z + y * dz_dlat)).abs() <= 1.0e-12);
            assert!((engine_jet[[row, 6, 0]] - (dx_dlat * z + x * dz_dlat)).abs() <= 1.0e-12);
        }

        // The chart penalty diagonal is also shared with the PyFFI helper.
        assert_eq!(
            SPHERE_CHART_PENALTY_DIAGONAL,
            [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
        );
    }

    /// Regression for #619 / #618-sphere: the lat/lon sphere chart jet must
    /// equal a central finite difference of the basis to ~1e-7 *at and beyond*
    /// the former clamp boundary `lat = ±π/2`, where the old binary `chain_lat`
    /// gate discontinuously zeroed the entire latitude jet and froze the atom.
    /// Also pins continuity of the basis across `lat = π/2`.
    #[test]
    fn sphere_chart_jet_matches_fd_at_clamp_boundary() {
        // Latitudes spanning interior, exactly the former boundary, and beyond.
        let coords = array![
            [std::f64::consts::FRAC_PI_2, 0.4], // exactly +π/2 (former gate flip)
            [-std::f64::consts::FRAC_PI_2, -1.1], // exactly -π/2
            [1.45, 2.0],                        // just below +π/2
            [1.69, -0.3],                       // just above +π/2
            [2.3, 0.7],                         // well beyond +π/2
            [0.35, 0.9],                        // interior control
        ];

        let (_, jet) = sphere_chart_basis_jet(coords.view()).unwrap();
        let h = 1.0e-6;
        for row in 0..coords.nrows() {
            for axis in 0..2 {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis]] += h;
                minus[[row, axis]] -= h;
                let (phi_p, _) = sphere_chart_basis_jet(plus.view()).unwrap();
                let (phi_m, _) = sphere_chart_basis_jet(minus.view()).unwrap();
                for col in 0..7 {
                    let fd = (phi_p[[row, col]] - phi_m[[row, col]]) / (2.0 * h);
                    let an = jet[[row, col, axis]];
                    assert!(
                        (fd - an).abs() <= 1.0e-7,
                        "row {row} col {col} axis {axis}: analytic {an} vs FD {fd}"
                    );
                }
            }
        }

        // Continuity of the basis across lat = π/2: the embedding does not jump.
        let eps = 1.0e-8;
        let lon = 0.4;
        let below = array![[std::f64::consts::FRAC_PI_2 - eps, lon]];
        let above = array![[std::f64::consts::FRAC_PI_2 + eps, lon]];
        let (phi_below, _) = sphere_chart_basis_jet(below.view()).unwrap();
        let (phi_above, _) = sphere_chart_basis_jet(above.view()).unwrap();
        for col in 0..7 {
            assert!(
                (phi_below[[0, col]] - phi_above[[0, col]]).abs() <= 1.0e-6,
                "basis discontinuous across lat = π/2 at col {col}: \
                 {} vs {}",
                phi_below[[0, col]],
                phi_above[[0, col]]
            );
        }
    }

    /// Central-difference oracle for `second_jet`: differentiate the analytic
    /// first jet (which is FD-validated by the test above) coordinate-wise.
    ///
    /// The threshold is magnitude-scaled (`abs_tol + rel_tol·max(|analytic|,
    /// |fd|)`), exactly like the third-jet helper, because the central-difference
    /// truncation error of a second derivative obtained by differencing the
    /// first jet is `O(ε²/6·|f⁗|)`. For a harmonic basis `sin(ωt)` the fourth
    /// derivative is `ω⁴·φ`, so with `ε = 1e-4` and the top harmonic of the
    /// periodic/torus evaluators (`ω = 2π·3 ≈ 18.85 → ω⁴ ≈ 1.26e5`) the floor is
    /// `≈ (1e-4)²/6·1.26e5 ≈ 2e-5` — several × any flat `1e-5` absolute bound.
    /// A pure absolute bound is therefore physically wrong at the top of the
    /// frequency range; the rel_tol term tracks the `ω⁴` truncation scale (the
    /// analytic second jet itself is exact, `-ω²·φ`). The FD step is 1e-4 (the
    /// sweet spot before f64 cancellation dominates a centered difference of an
    /// `O(1)` Jacobian).
    fn assert_second_jet_matches_central_difference<E: SaeBasisSecondJet>(
        evaluator: &E,
        coords: Array2<f64>,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<(), String> {
        let epsilon = 1.0e-4;
        let second = evaluator.second_jet(coords.view())?;
        let (_phi, jet) = evaluator.evaluate(coords.view())?;
        let (n_rows, n_basis, latent_dim, latent_dim_b) = second.dim();
        assert_eq!(latent_dim, latent_dim_b);
        assert_eq!((n_rows, n_basis, latent_dim), jet.dim());
        for row in 0..n_rows {
            for axis_c in 0..latent_dim {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis_c]] += epsilon;
                minus[[row, axis_c]] -= epsilon;
                let (_, jet_plus) = evaluator.evaluate(plus.view()).unwrap();
                let (_, jet_minus) = evaluator.evaluate(minus.view()).unwrap();
                for basis in 0..n_basis {
                    for axis_a in 0..latent_dim {
                        let fd = (jet_plus[[row, basis, axis_a]] - jet_minus[[row, basis, axis_a]])
                            / (2.0 * epsilon);
                        let analytic = second[[row, basis, axis_a, axis_c]];
                        let error = (analytic - fd).abs();
                        let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                        assert!(
                            error <= threshold,
                            "row={row} basis={basis} axis_a={axis_a} axis_c={axis_c}: \
                             analytic={analytic:.12e}, fd={fd:.12e}, error={error:.12e}, \
                             threshold={threshold:.12e}"
                        );
                    }
                }
            }
        }
        // Hessian symmetry in (axis_a, axis_c).
        for row in 0..n_rows {
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    for axis_b in 0..latent_dim {
                        let h_ab = second[[row, basis, axis_a, axis_b]];
                        let h_ba = second[[row, basis, axis_b, axis_a]];
                        assert!(
                            (h_ab - h_ba).abs() <= 1.0e-12,
                            "second_jet not symmetric: row={row} basis={basis} \
                             ({axis_a},{axis_b})={h_ab:.6e} vs ({axis_b},{axis_a})={h_ba:.6e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// The analytic third jet `T[n,m,a,c,e] = ∂³Φ_m/∂t_a∂t_c∂t_e` must equal the
    /// central difference of the analytic (already FD-validated) second jet along
    /// the trailing axis, and be fully symmetric across its three trailing axes.
    /// This validates the closed-form `K` providers added for the exact isometry
    /// Hessian (#458) against an independent numerical derivative — the third-jet
    /// analogue of `assert_second_jet_matches_central_difference`. A
    /// magnitude-scaled tolerance is used because the harmonic third derivatives
    /// scale like `freq³` (≈ thousands for the higher harmonics), so a pure
    /// absolute bound would be meaningless at the top of the range.
    fn assert_third_jet_matches_central_difference<E: SaeBasisThirdJet>(
        evaluator: &E,
        coords: Array2<f64>,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<(), String> {
        let epsilon = 1.0e-4;
        let third = evaluator.third_jet(coords.view())?;
        let second = evaluator.second_jet(coords.view())?;
        let (n_rows, n_basis, latent_dim, ld_b, ld_c) = third.dim();
        assert_eq!(latent_dim, ld_b);
        assert_eq!(latent_dim, ld_c);
        assert_eq!((n_rows, n_basis, latent_dim, latent_dim), second.dim());
        for row in 0..n_rows {
            for axis_e in 0..latent_dim {
                let mut plus = coords.clone();
                let mut minus = coords.clone();
                plus[[row, axis_e]] += epsilon;
                minus[[row, axis_e]] -= epsilon;
                let second_plus = evaluator.second_jet(plus.view())?;
                let second_minus = evaluator.second_jet(minus.view())?;
                for basis in 0..n_basis {
                    for axis_a in 0..latent_dim {
                        for axis_c in 0..latent_dim {
                            let fd = (second_plus[[row, basis, axis_a, axis_c]]
                                - second_minus[[row, basis, axis_a, axis_c]])
                                / (2.0 * epsilon);
                            let analytic = third[[row, basis, axis_a, axis_c, axis_e]];
                            let error = (analytic - fd).abs();
                            let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                            assert!(
                                error <= threshold,
                                "row={row} basis={basis} a={axis_a} c={axis_c} e={axis_e}: \
                                 analytic={analytic:.12e}, fd={fd:.12e}, error={error:.6e}, \
                                 threshold={threshold:.6e}"
                            );
                        }
                    }
                }
            }
        }
        // Full symmetry across the three trailing derivative axes (mixed partials
        // commute), so every permutation of `(a, c, e)` must agree.
        for row in 0..n_rows {
            for basis in 0..n_basis {
                for a in 0..latent_dim {
                    for b in 0..latent_dim {
                        for c in 0..latent_dim {
                            let reference = third[[row, basis, a, b, c]];
                            for perm in [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]] {
                                let permuted = third[[row, basis, perm[0], perm[1], perm[2]]];
                                assert!(
                                    (reference - permuted).abs() <= 1.0e-10,
                                    "third_jet not symmetric: row={row} basis={basis} \
                                     ({a},{b},{c})={reference:.6e} vs ({},{},{})={permuted:.6e}",
                                    perm[0],
                                    perm[1],
                                    perm[2]
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn isometry_periodic_second_jet_matches_fd() -> Result<(), String> {
        // Magnitude-scaled tolerance: the top harmonic (ω = 2π·3) drives a
        // O(ε²·ω⁴) ≈ 2e-5 central-difference truncation floor, far above any flat
        // 1e-5 absolute bound; rel_tol = 1e-5 tracks the ω⁴ scale (analytic exact).
        assert_second_jet_matches_central_difference(
            &PeriodicHarmonicEvaluator::new(7).unwrap(),
            array![[-0.37], [0.0], [0.125], [0.41]],
            1.0e-6,
            1.0e-5,
        )?;
        Ok(())
    }

    #[test]
    fn isometry_sphere_second_jet_matches_fd() -> Result<(), String> {
        // Stay inside the interior `(-π/2, π/2)` for lat so the chain factor
        // is active — that is where the Hessian carries information.
        let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
        assert_second_jet_matches_central_difference(
            &SphereChartEvaluator,
            sphere_coords,
            1.0e-6,
            1.0e-5,
        )?;
        Ok(())
    }

    #[test]
    fn isometry_torus_second_jet_matches_fd() -> Result<(), String> {
        let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
        let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
        assert!(evaluator.basis_size() > 0);
        // Same ω⁴ truncation floor as the periodic case (top harmonic ω = 2π·3).
        assert_second_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
        Ok(())
    }

    #[test]
    fn isometry_periodic_third_jet_matches_fd() -> Result<(), String> {
        assert_third_jet_matches_central_difference(
            &PeriodicHarmonicEvaluator::new(7).unwrap(),
            array![[-0.37], [0.0], [0.125], [0.41]],
            1.0e-6,
            1.0e-5,
        )?;
        Ok(())
    }

    #[test]
    fn isometry_sphere_third_jet_matches_fd() -> Result<(), String> {
        // Interior of `(-π/2, π/2)` for lat so the chart chain factor is active —
        // that is where the third-order curvature term carries information.
        let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
        assert_third_jet_matches_central_difference(
            &SphereChartEvaluator,
            sphere_coords,
            1.0e-6,
            1.0e-5,
        )?;
        Ok(())
    }

    #[test]
    fn isometry_torus_third_jet_matches_fd() -> Result<(), String> {
        let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
        let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
        assert!(evaluator.basis_size() > 0);
        assert_third_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
        Ok(())
    }

    #[test]
    fn isometry_affine_third_jet_is_trivial_zero() -> Result<(), String> {
        let evaluator = AffineCoordinateEvaluator { latent_dim: 3 };
        let coords = array![[0.2, -0.3, 0.7], [1.1, 0.0, -0.4]];
        let third = evaluator.third_jet(coords.view())?;
        assert_eq!(third.dim(), (coords.nrows(), 4, 3, 3, 3));
        assert!(
            third.iter().all(|x| *x == 0.0),
            "affine third jet must vanish identically, got {third:?}"
        );
        Ok(())
    }

    #[test]
    fn isometry_euclidean_patch_third_jet_matches_fd() -> Result<(), String> {
        let evaluator = EuclideanPatchEvaluator::new(2, 4)?;
        let coords = array![[0.2, -0.3], [0.7, 0.4], [-0.5, 0.9]];
        assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-6, 1.0e-5)?;
        Ok(())
    }

    /// Issue #247: the Duchon coordinate evaluator must return a forward design
    /// and a derivative jet with *matching column counts* — the original bug
    /// was a radial-only design paired with a radial+polynomial jet (or vice
    /// versa), which the consumer rejected as a "design/jet column mismatch".
    #[test]
    fn duchon_coordinate_evaluator_phi_and_jet_share_column_count() {
        for (d, centers) in [
            (1usize, array![[-1.0], [-0.4], [0.1], [0.6], [1.2], [1.9]]),
            (
                2usize,
                array![
                    [-1.0, -0.8],
                    [-0.3, 0.4],
                    [0.2, -0.5],
                    [0.7, 0.9],
                    [1.1, -0.2],
                    [1.6, 0.6],
                ],
            ),
        ] {
            let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
            let coords = match d {
                1 => array![[-0.5], [0.0], [0.3], [0.8]],
                _ => array![[-0.5, 0.2], [0.0, -0.3], [0.3, 0.7], [0.8, -0.1]],
            };
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            assert_eq!(
                phi.ncols(),
                jet.shape()[1],
                "Duchon d={d}: Phi has {} columns but jet has {}",
                phi.ncols(),
                jet.shape()[1]
            );
            assert_eq!(jet.shape()[0], coords.nrows());
            assert_eq!(jet.shape()[2], d);
        }
    }

    /// The Duchon evaluator's analytic first jet must equal the finite
    /// difference of its own forward design — i.e. `dPhi/dt` is the true
    /// derivative of `Phi(t)`, with no stray amplification/column mismatch.
    #[test]
    fn duchon_coordinate_evaluator_jacobian_matches_fd() {
        let centers = array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ];
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        // Keep probe points away from any center so the radial kernel is smooth.
        let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
        assert_jacobian_matches_central_difference(&evaluator, coords, 1.0e-4);
    }

    /// The Duchon evaluator's analytic second jet must match the FD of its
    /// (FD-validated) first jet.
    #[test]
    fn duchon_coordinate_evaluator_second_jet_matches_fd() -> Result<(), String> {
        let centers = array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ];
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
        assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
        Ok(())
    }

    /// The Duchon evaluator's closed-form analytic third jet (radial
    /// third-derivative kernel block + monomial nullspace block) must match the
    /// FD of its (FD-validated) second jet, validating the closed form that
    /// replaced the forbidden finite-difference `third_jet_dyn` default.
    #[test]
    fn duchon_coordinate_evaluator_third_jet_matches_fd() -> Result<(), String> {
        let centers = array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ];
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
        assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
        Ok(())
    }

    /// The Euclidean tangent-patch evaluator's monomial design and its
    /// first/second jets must be mutually consistent under finite differences.
    #[test]
    fn euclidean_patch_evaluator_jets_match_fd() -> Result<(), String> {
        let evaluator = EuclideanPatchEvaluator::new(2, 2).unwrap();
        let coords = array![[0.0, -1.0], [3.5, 0.25], [-0.75, 1.2], [0.4, 0.9]];
        assert_jacobian_matches_central_difference(&evaluator, coords.clone(), 1.0e-6);
        assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-5, 1.0e-5)?;
        // The degree-2 patch in d=2 has columns {1, x, y, x², xy, y²}.
        let (phi, _jet) = evaluator.evaluate(array![[0.0, 0.0]].view())?;
        assert_eq!(phi.ncols(), 6);
        Ok(())
    }

    #[test]
    fn euclidean_affine_gauge_canonicalization_preserves_reconstruction() -> Result<(), String> {
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
        let canonical = array![[-1.0_f64], [-0.35], [0.1], [0.65], [1.2]];
        let mut coords = canonical.clone();
        for row in 0..coords.nrows() {
            coords[[row, 0]] = 2.75 + 4.0 * canonical[[row, 0]];
        }
        let (phi, jet) = evaluator.evaluate(coords.view())?;
        let decoder = array![[0.25, -0.4], [1.2, 0.3], [-0.15, 0.5]];
        let atom = SaeManifoldAtom::new(
            "euclidean_patch",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(evaluator.basis_size()),
        )?
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((coords.nrows(), 1)),
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )?;
        let mut term = SaeManifoldTerm::new(vec![atom], assignment)?;
        let before = term.fitted();

        term.canonicalize_affine_gauge_after_accept()?;

        let after = term.fitted();
        let max_abs = before
            .iter()
            .zip(after.iter())
            .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            max_abs <= 1.0e-10,
            "canonicalization changed reconstruction by {max_abs:.3e}"
        );
        let live = term.assignment.coords[0].as_matrix();
        let mean = live.column(0).sum() / live.nrows() as f64;
        let rms = (live.column(0).iter().map(|v| v * v).sum::<f64>() / live.nrows() as f64).sqrt();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(rms, 1.0, epsilon = 1.0e-12);
        Ok(())
    }

    #[test]
    fn quotient_step_norm_removes_pure_euclidean_affine_gauge() -> Result<(), String> {
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
        let coords = array![[-1.0_f64], [-0.4], [0.2], [0.8], [1.3]];
        let (phi, jet) = evaluator.evaluate(coords.view())?;
        let decoder = array![[0.1, -0.2], [1.0, 0.4], [0.25, -0.3]];
        let atom = SaeManifoldAtom::new(
            "euclidean_patch",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(evaluator.basis_size()),
        )?
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((coords.nrows(), 1)),
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )?;
        let term = SaeManifoldTerm::new(vec![atom], assignment)?;
        let gauges = term.dense_step_gauge_vectors()?;
        assert!(
            gauges.len() >= 2,
            "expected translation and scale gauge generators"
        );
        let n_coord = term.n_obs() * term.assignment.row_block_dim();
        let gauge = &gauges[1];
        let delta_t = gauge.slice(s![..n_coord]);
        let delta_beta = gauge.slice(s![n_coord..]);
        let raw = gauge.iter().map(|v| v * v).sum::<f64>();

        let quotient = term.quotient_newton_step_norm_sq(delta_t, delta_beta, raw)?;

        assert!(
            quotient <= raw.max(1.0) * 1.0e-20,
            "pure affine gauge step left quotient norm squared {quotient:.3e} from raw {raw:.3e}"
        );
        Ok(())
    }

    /// Torus T^2 fit on synthetic data with a known two-frequency signal.
    /// Drives a single torus atom through the [`SaeManifoldTerm`] Newton loop
    /// and checks that the in-sample reconstruction R² clears 0.5.
    #[test]
    fn sae_torus_atom_recovers_two_frequency_synthetic() {
        let n = 96usize;
        let p = 4usize;
        let h = 3usize;
        let d = 2usize;
        let evaluator = TorusHarmonicEvaluator::new(d, h).unwrap();
        let m = evaluator.basis_size();
        // True coords on T^2 (phase in [0, 1)).
        let mut true_coords = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            true_coords[[i, 0]] = ((i as f64) * 0.137).rem_euclid(1.0);
            true_coords[[i, 1]] = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
        }
        // Synthetic target: a low-frequency periodic signal on T^2 mixed
        // linearly into a p-dim ambient.
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t1 = 2.0 * std::f64::consts::PI * true_coords[[i, 0]];
            let t2 = 2.0 * std::f64::consts::PI * true_coords[[i, 1]];
            z[[i, 0]] = t1.sin() + 0.3 * t2.cos();
            z[[i, 1]] = t1.cos() + 0.2 * (t1 + t2).sin();
            z[[i, 2]] = t2.sin();
            z[[i, 3]] = 0.5 * (t1 - t2).cos();
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
        // Initialise from the true coords (this test exercises basis correctness
        // and decoder fit, not coordinate identification on T^2).
        let (phi0, jet0) = evaluator.evaluate(true_coords.view()).unwrap();
        // Penalty: identity-on-non-constant + tiny floor on constant.
        let mut penalty = Array2::<f64>::eye(m);
        penalty *= 1.0e-4;
        let atom = SaeManifoldAtom::new(
            "torus_atom",
            SaeAtomBasisKind::Torus,
            d,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            penalty,
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TorusHarmonicEvaluator::new(d, h).unwrap()));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![true_coords],
            vec![LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ])],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        // ARD log-precision is per-axis (length == atom latent dim), not a
        // single scalar — see `SaeManifoldRho::to_flat` / `from_flat` and
        // the validation in `negative_log_ard_prior` (`ARD rho atom k has
        // len ... but atom dim is d`).
        let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(d)]);
        let ridge = 1.0e-6;
        for _ in 0..10 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
                .unwrap();
            if !loss.total().is_finite() {
                break;
            }
        }
        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut sse = 0.0_f64;
        for ((row, col), v) in fitted.indexed_iter() {
            let r = v - z[[row, col]];
            sse += r * r;
        }
        let r2 = 1.0 - sse / sst.max(1.0e-12);
        assert!(
            r2 >= 0.5,
            "torus atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
        );
    }

    /// Sphere S² fit on a synthetic spherical signal. Drives a single sphere
    /// atom through the [`SaeManifoldTerm`] Newton loop and checks in-sample
    /// R² ≥ 0.5.
    #[test]
    fn sae_sphere_atom_recovers_synthetic_signal() {
        let n = 96usize;
        let p = 3usize;
        let d = 2usize;
        // True (lat, lon) coords.
        let mut true_coords = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            true_coords[[i, 0]] = -0.5 + 1.0 * t; // lat in [-0.5, 0.5]
            true_coords[[i, 1]] = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * t;
        }
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let lat = true_coords[[i, 0]];
            let lon = true_coords[[i, 1]];
            let x = lat.cos() * lon.cos();
            let y = lat.cos() * lon.sin();
            let zc = lat.sin();
            z[[i, 0]] = x;
            z[[i, 1]] = y;
            z[[i, 2]] = zc;
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
        let (phi0, jet0) = SphereChartEvaluator.evaluate(true_coords.view()).unwrap();
        let m = phi0.ncols();
        let mut penalty = Array2::<f64>::eye(m);
        penalty *= 1.0e-4;
        let atom = SaeManifoldAtom::new(
            "sphere_atom",
            SaeAtomBasisKind::Sphere,
            d,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            penalty,
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(SphereChartEvaluator));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![true_coords],
            vec![LatentManifold::Product(vec![
                LatentManifold::Interval {
                    lo: -std::f64::consts::FRAC_PI_2,
                    hi: std::f64::consts::FRAC_PI_2,
                },
                LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                },
            ])],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        // The sphere atom's coordinate is a dim-2 product manifold (lat × lon),
        // so per-axis ARD must carry one log-precision per axis (`atom dim = 2`).
        // A length-1 block would be indexed out of bounds at `axis == 1` in the
        // per-axis assembly loop and is rejected by the per-axis ARD contract.
        let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(2)]);
        let ridge = 1.0e-6;
        for _ in 0..10 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
                .unwrap();
            if !loss.total().is_finite() {
                break;
            }
        }
        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut sse = 0.0_f64;
        for ((row, col), v) in fitted.indexed_iter() {
            let r = v - z[[row, col]];
            sse += r * r;
        }
        let r2 = 1.0 - sse / sst.max(1.0e-12);
        assert!(
            r2 >= 0.5,
            "sphere atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
        );
    }

    /// Mirror of the Python `test_sae_manifold_softmax_dispatch` shape: drive a
    /// single periodic atom on a 1-harmonic synthetic target with 10 Newton
    /// steps end-to-end in Rust and check that the multi-step loop achieves
    /// in-sample R² ≥ 0.95.
    #[test]
    fn sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2() {
        let n = 64usize;
        let m = 3usize;
        let p = 1usize;

        let true_t: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * true_t[i];
            z[[i, 0]] = 0.7 * angle.sin() + 0.3 * angle.cos();
        }
        let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();

        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut coords0_data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            // Phase-shifted initialization so the optimizer must do real work.
            coords0_data[[i, 0]] = (true_t[i] + 0.25).rem_euclid(1.0);
        }
        let (phi0, jet0) = evaluator.evaluate(coords0_data.view()).unwrap();

        let atom = SaeManifoldAtom::new(
            "periodic_atom",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            Array2::<f64>::zeros((m, p)),
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));

        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords0_data],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(0.5),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);

        let max_iter = 10usize;
        let learning_rate = 1.0;
        let ridge = 1.0e-6;
        let mut prev_total = f64::INFINITY;
        for _ in 0..max_iter {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, learning_rate, ridge, ridge)
                .unwrap();
            let total = loss.total();
            if !total.is_finite() {
                break;
            }
            let denom = prev_total.abs().max(1.0e-12);
            let rel = (prev_total - total).abs() / denom;
            prev_total = total;
            if rel < 1.0e-6 {
                break;
            }
        }

        let fitted = term.fitted();
        assert_eq!(fitted.dim(), (n, p));
        let mut ssr = 0.0;
        for i in 0..n {
            let r = z[[i, 0]] - fitted[[i, 0]];
            ssr += r * r;
        }
        let r2 = 1.0 - ssr / sst.max(1.0e-12);
        assert!(
            r2 >= 0.95,
            "10-step in-sample R² = {r2:.4} (ssr={ssr:.6}, sst={sst:.6}) should be >= 0.95"
        );
    }

    /// Regression test for issue #177: softmax assignment used to bail out of
    /// the row-block Hessian assembly with "softmax assignment hessian diag
    /// unavailable". The penalty now exposes the analytic diagonal extracted
    /// from its row-dense HVP, so the joint-fit driver completes one step.
    #[test]
    fn softmax_assignment_hessian_diag_is_available_for_k2() {
        let n = 4usize;
        let k = 2usize;
        let logits =
            Array2::<f64>::from_shape_fn((n, k), |(i, j)| 0.1 * (i as f64) - 0.2 * (j as f64));
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            manifolds,
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
            .expect("softmax assignment Hessian diagonal must be available");
        assert_eq!(grad.len(), n * k);
        assert_eq!(diag.len(), n * k);
        assert!(grad.iter().all(|v| v.is_finite()));
        assert!(diag.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sae_registry_refuses_assignment_sparsity_penalties() {
        let n = 3usize;
        let k = 2usize;
        let logits = Array2::<f64>::zeros((n, k));
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            manifolds,
            AssignmentMode::softmax(0.7),
        )
        .expect("valid assignment");
        let atoms: Vec<SaeManifoldAtom> = (0..k)
            .map(|atom_idx| {
                SaeManifoldAtom::new(
                    format!("periodic_{atom_idx}"),
                    SaeAtomBasisKind::Periodic,
                    1,
                    Array2::<f64>::ones((n, 1)),
                    Array3::<f64>::zeros((n, 1, 1)),
                    Array2::<f64>::zeros((1, 1)),
                    Array2::<f64>::eye(1),
                )
                .expect("valid atom")
            })
            .collect();
        let term = SaeManifoldTerm::new(atoms, assignment).expect("valid SAE term");

        let mut softmax_registry = AnalyticPenaltyRegistry::new();
        softmax_registry.push(AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
            crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(k, 0.7),
        )));
        let softmax_err = term
            .validate_analytic_penalty_registry(&softmax_registry)
            .expect_err("SAE registry must reject softmax assignment sparsity");
        assert!(softmax_err.contains("assignment sparsity"));

        let mut ibp_registry = AnalyticPenaltyRegistry::new();
        ibp_registry.push(AnalyticPenaltyKind::IBPAssignment(Arc::new(
            crate::terms::analytic_penalties::IBPAssignmentPenalty::new(k, 1.2, 0.7, false),
        )));
        let ibp_err = term
            .validate_analytic_penalty_registry(&ibp_registry)
            .expect_err("SAE registry must reject IBP assignment sparsity");
        assert!(ibp_err.contains("assignment sparsity"));
    }

    #[test]
    fn ibp_fixed_alpha_assignment_value_matches_logit_gradient_fd() {
        let n = 4usize;
        let k = 3usize;
        let logits = Array2::<f64>::from_shape_vec(
            (n, k),
            vec![
                -0.4, 0.2, 0.7, 0.1, -0.3, 0.5, 0.8, -0.1, -0.6, 0.3, 0.6, -0.2,
            ],
        )
        .expect("valid IBP logit grid");
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            manifolds,
            AssignmentMode::ibp_map(0.9, 1.4, false),
        )
        .expect("valid IBP assignment");
        let rho = SaeManifoldRho::new(0.23_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, _) =
            assignment_prior_grad_hdiag(&assignment, &rho).expect("IBP assignment gradient");
        let idx = 5usize;
        let step = 1.0e-6_f64;
        let mut plus = assignment.clone();
        plus.logits[[idx / k, idx % k]] += step;
        let mut minus = assignment.clone();
        minus.logits[[idx / k, idx % k]] -= step;
        let fd = (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho))
            / (2.0 * step);

        assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-7);
    }

    #[test]
    fn jumprelu_assignment_value_matches_logit_gradient_fd() {
        let n = 4usize;
        let k = 2usize;
        let temperature = 0.35_f64;
        let threshold = 0.1_f64;
        let logits = Array2::<f64>::from_shape_vec(
            (n, k),
            vec![-13.0, -0.2, 0.0, 0.05, 0.15, 0.4, 0.9, 1.5],
        )
        .expect("valid JumpReLU logit grid");
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            manifolds,
            AssignmentMode::jumprelu(temperature, threshold),
        )
        .expect("valid JumpReLU assignment");
        let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, _) =
            assignment_prior_grad_hdiag(&assignment, &rho).expect("JumpReLU assignment gradient");
        let idx = 4usize;
        let step = 1.0e-6_f64;
        let mut plus = assignment.clone();
        plus.logits[[idx / k, idx % k]] += step;
        let mut minus = assignment.clone();
        minus.logits[[idx / k, idx % k]] -= step;
        let fd = (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho))
            / (2.0 * step);

        assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-8);
    }

    #[test]
    fn jumprelu_assignment_prior_hessian_diag_is_exact_over_logit_sweep() {
        let n = 6usize;
        let k = 2usize;
        let temperature = 0.35_f64;
        let threshold = 0.1_f64;
        let logits = Array2::<f64>::from_shape_vec(
            (n, k),
            vec![
                -2.0, -0.2, 0.0, 0.05, 0.1, 0.15, 0.4, 0.9, 1.5, 2.5, 4.0, 6.0,
            ],
        )
        .expect("valid logit grid");
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords,
            manifolds,
            AssignmentMode::jumprelu(temperature, threshold),
        )
        .expect("valid JumpReLU assignment");
        let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
        let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
            .expect("JumpReLU assignment prior hessian diag");
        let inv_tau = 1.0 / temperature;
        let inv_tau2 = inv_tau * inv_tau;
        let sparsity_strength = rho.log_lambda_sparse.exp();

        assert_eq!(grad.len(), n * k);
        assert_eq!(diag.len(), n * k);
        let mut saw_negative = false;
        for (idx, &entry) in diag.iter().enumerate() {
            let logit = logits[[idx / k, idx % k]];
            // Expected = exact second derivative of the threshold-centered
            // surrogate σ((l−θ)/τ), using the same machine-precision support as
            // the value and gradient paths.
            let expected = if jumprelu_in_optimization_band(logit, threshold, temperature) {
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2
            } else {
                0.0
            };
            assert!(
                entry.is_finite(),
                "JumpReLU hessian_diag must be finite at index {idx}"
            );
            saw_negative |= entry < 0.0;
            assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
        }
        assert!(
            saw_negative,
            "exact JumpReLU hessian_diag must go negative above the threshold"
        );
    }

    /// Regression test for issue #174: K>=2 periodic atoms with zero-init
    /// decoder used to collapse to A≈0 because the assignment prior was the
    /// only term with non-zero gradient at iter 0. The pyffi entry point now
    /// seeds decoder coefficients via a joint LSQ projection of Z onto
    /// [a_init · Phi_k]. This test exercises that exact seeding strategy
    /// in pure Rust and verifies the joint Newton fit reaches positive R²
    /// on a clean K=2 periodic torus signal, mirroring the failing
    /// reproducer in #174.
    #[test]
    fn ibp_map_k2_periodic_torus_recovers_signal_with_lsq_init() {
        use crate::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
        use faer::Side as FaerSide;

        let n = 200usize;
        let p = 8usize;
        let k = 2usize;
        let m = 5usize; // 1 (constant) + 2 harmonics * 2 (sin/cos) = 5

        // Build a synthetic K=2 torus signal Z = [cos th1, sin th1, cos th2, sin th2] @ mix
        // with two latent angles. Deterministic seed via index arithmetic.
        let mut theta = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            theta[[i, 0]] = ((i as f64) * 0.07) % 1.0;
            theta[[i, 1]] = ((i as f64) * 0.13 + 0.31) % 1.0;
        }
        let mut raw = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let a1 = 2.0 * std::f64::consts::PI * theta[[i, 0]];
            let a2 = 2.0 * std::f64::consts::PI * theta[[i, 1]];
            raw[[i, 0]] = a1.cos();
            raw[[i, 1]] = a1.sin();
            raw[[i, 2]] = a2.cos();
            raw[[i, 3]] = a2.sin();
        }
        // Deterministic 4x8 mixing matrix.
        let mix = Array2::<f64>::from_shape_fn((4, p), |(i, j)| {
            ((i as f64 + 1.0) * 0.37 + (j as f64) * 0.21).sin()
        });
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let mut acc = 0.0;
                for r in 0..4 {
                    acc += raw[[i, r]] * mix[[r, j]];
                }
                z[[i, j]] = acc;
            }
        }
        // Centre Z so R² is well-defined relative to mean.
        let mut col_mean = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += z[[i, j]];
            }
            col_mean[j] = acc / n as f64;
        }
        for i in 0..n {
            for j in 0..p {
                z[[i, j]] -= col_mean[j];
            }
        }

        // Atom coordinates: use the (shifted) true angles so the periodic
        // basis aligns with the signal — the test isolates the decoder-init
        // collapse, not coordinate recovery.
        let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
        for i in 0..n {
            coords_k[0][[i, 0]] = (theta[[i, 0]] + 0.05).rem_euclid(1.0);
            coords_k[1][[i, 0]] = (theta[[i, 1]] + 0.07).rem_euclid(1.0);
        }
        // Periodic basis (constant + 2 harmonics → M=5) for each atom.
        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut phi_k = Vec::with_capacity(k);
        let mut jet_k = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
            phi_k.push(phi);
            jet_k.push(jet);
        }

        // LSQ seed: joint design X = [0.5 * Phi_1 | 0.5 * Phi_2] (IBP-MAP
        // logit 0 gives sigmoid(0/tau) = 0.5 for both atoms), solve normal
        // equations with a small ridge.
        let m_total = k * m;
        let mut x = Array2::<f64>::zeros((n, m_total));
        for atom_idx in 0..k {
            for i in 0..n {
                for col in 0..m {
                    x[[i, atom_idx * m + col]] = 0.5 * phi_k[atom_idx][[i, col]];
                }
            }
        }
        let mut xtx = fast_ata(&x);
        let mut trace = 0.0_f64;
        for i in 0..m_total {
            trace += xtx[[i, i]];
        }
        let jitter = (trace / m_total as f64).max(1.0) * 1.0e-8;
        for i in 0..m_total {
            xtx[[i, i]] += jitter;
        }
        let xtz = fast_atb(&x, &z);
        let b_joint = xtx
            .cholesky(FaerSide::Lower)
            .expect("LSQ Cholesky")
            .solve_mat(&xtz);

        let mut atoms = Vec::with_capacity(k);
        for atom_idx in 0..k {
            let mut b = Array2::<f64>::zeros((m, p));
            for col in 0..m {
                for j in 0..p {
                    b[[col, j]] = b_joint[[atom_idx * m + col, j]];
                }
            }
            let atom = SaeManifoldAtom::new(
                format!("torus_atom_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi_k[atom_idx].clone(),
                jet_k[atom_idx].clone(),
                b,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
            atoms.push(atom);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, k)),
            coords_k,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::ibp_map(0.7, 1.0, false),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        // `lambda_sparse` is the IBP assignment-sparsity prior weight (now wired
        // through `assignment_prior_grad_hdiag`'s IBP branch, #853). The
        // Beta-Bernoulli BCE energy toward the self-referential empirical active
        // fraction has its global minimum at the all-off gate, so at the old
        // full weight (`log_lambda_sparse = 0 → λ = 1`) it overwhelmed the
        // truth-seeded data fit and collapsed the assignment off both atoms. A
        // moderate prior weight keeps the sparsity pressure honest while letting
        // the LSQ-seeded reconstruction hold both real atoms active — the
        // realistic operating point this recovery test pins.
        let mut rho = SaeManifoldRho::new((0.02_f64).ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);

        let mut prev_total = f64::INFINITY;
        for _ in 0..30 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
                .unwrap();
            let total = loss.total();
            if !total.is_finite() {
                break;
            }
            let denom = prev_total.abs().max(1.0e-12);
            let rel = (prev_total - total).abs() / denom;
            prev_total = total;
            if rel < 1.0e-6 {
                break;
            }
        }

        let fitted = term.fitted();
        let mut ssr = 0.0;
        let mut sst = 0.0;
        for i in 0..n {
            for j in 0..p {
                let r = z[[i, j]] - fitted[[i, j]];
                ssr += r * r;
                sst += z[[i, j]] * z[[i, j]];
            }
        }
        let r2 = 1.0 - ssr / sst.max(1.0e-12);
        assert!(
            r2 > 0.5,
            "K=2 periodic torus IBP-MAP R² = {r2:.4} (ssr={ssr:.4}, sst={sst:.4}) should be > 0.5 with LSQ-seeded decoder"
        );
        // Also confirm at least one atom remains active (assignment did not
        // collapse to ~0) — the active mass averaged over rows must exceed
        // a non-trivial threshold.
        let assignments = term.assignment.assignments();
        let mean_active: f64 = assignments.iter().copied().sum::<f64>() / (n as f64);
        assert!(
            mean_active > 0.2,
            "mean active mass across rows = {mean_active:.4} should exceed 0.2; assignment did not collapse"
        );
    }

    /// Regression test for issue #174 + #177 combined: softmax assignment
    /// with K=2 periodic atoms should not crash and should reduce loss.
    #[test]
    fn softmax_k2_periodic_completes_joint_fit_step() {
        let n = 64usize;
        let p = 4usize;
        let k = 2usize;
        let m = 3usize;

        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            z[[i, 0]] = a.sin();
            z[[i, 1]] = a.cos();
            z[[i, 2]] = (2.0 * a).sin();
            z[[i, 3]] = (2.0 * a).cos();
        }

        let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
        let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
        for i in 0..n {
            coords_k[0][[i, 0]] = (i as f64) / (n as f64);
            coords_k[1][[i, 0]] = ((i as f64) * 2.0 / (n as f64)).rem_euclid(1.0);
        }
        let mut atoms = Vec::new();
        for atom_idx in 0..k {
            let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
            // Non-trivial decoder init (simulate LSQ seeding) so the data-fit
            // signal is non-zero at iter 0.
            let b = Array2::<f64>::from_shape_fn((m, p), |(i, j)| {
                0.1 * ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
            });
            let atom = SaeManifoldAtom::new(
                format!("a_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                b,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
            atoms.push(atom);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, k)),
            coords_k,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);

        // First step must succeed (previously bailed with hessian-diag error).
        let loss0 = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .expect("softmax K=2 must complete first joint-fit step");
        assert!(loss0.total().is_finite());
        let loss1 = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .expect("softmax K=2 must complete second joint-fit step");
        assert!(loss1.total().is_finite());
    }

    /// End-to-end Isometry wiring oracle.
    ///
    /// Build a SAE atom around an evaluator whose `second_jet` is now
    /// implemented (periodic / sphere / torus), construct an
    /// [`IsometryPenalty`] with matching `latent_dim` and `p_out`, refresh
    /// the caches via [`refresh_isometry_caches_from_atom`], and check that
    ///
    ///   * `IsometryPenalty.value(target, rho)` is strictly positive (the
    ///     decoder we feed in is not orthonormal so the pullback metric is
    ///     not the identity, and the Euclidean reference picks up the gap).
    ///   * `IsometryPenalty.grad_target(target, rho)` is non-zero on at
    ///     least one latent-coordinate component.
    ///   * The analytic gradient matches a finite-difference oracle of
    ///     `value()` w.r.t. `target` (a single coord), where each FD probe
    ///     drives a fresh cache refresh — this is exactly the chain of
    ///     calls the SAE outer loop will make.
    ///
    /// The FD oracle re-uses the existing [`refresh_isometry_caches_from_atom`]
    /// helper for both the analytic side and the FD side, so any layout
    /// mismatch between `J`/`H` would show up as a tolerance failure rather
    /// than a silently zero gradient.
    fn assert_isometry_wiring_matches_fd(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: Array2<f64>,
    ) {
        let n_obs = coords.nrows();
        let latent_dim = coords.ncols();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let p: usize = 3;
        // A deterministic non-orthonormal decoder: deterministic LCG-ish
        // floats keep the test reproducible without needing rand.
        let mut decoder = Array2::<f64>::zeros((m, p));
        for i in 0..m {
            for j in 0..p {
                let x = (i as f64) * 0.371 + (j as f64) * 0.193 + 0.5;
                decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
            }
        }
        let smooth = Array2::<f64>::eye(m);
        let atom = SaeManifoldAtom::new(
            "iso_wire_test",
            SaeAtomBasisKind::Periodic,
            latent_dim,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            smooth,
        )
        .unwrap()
        .with_basis_second_jet(evaluator);

        let target_slice = PsiSlice::full(n_obs * latent_dim, Some(latent_dim));
        let penalty = IsometryPenalty::new_euclidean(target_slice, p);
        let rho = Array1::<f64>::zeros(1);

        // Without a refresh, the safe default is zero and the gradient is
        // all zeros. Confirm the precondition so the post-refresh contrast
        // is meaningful.
        let target_flat: Array1<f64> = coords.iter().copied().collect();
        let v0 = penalty.value(target_flat.view(), rho.view());
        assert_eq!(v0, IsometryPenalty::DEFAULT_VALUE_ON_MISSING_CACHE);
        let g0 = penalty.grad_target(target_flat.view(), rho.view());
        assert!(
            g0.iter().all(|x| *x == 0.0),
            "grad_target without cache must be all zeros, got {g0:?}"
        );

        // Refresh and re-evaluate.
        let installed_second =
            refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        assert!(
            installed_second,
            "evaluator must implement second_jet for this oracle to run"
        );

        let value = penalty.value(target_flat.view(), rho.view());
        assert!(
            value > 1.0e-6,
            "expected non-trivial isometry loss after cache refresh, got {value}"
        );
        let grad = penalty.grad_target(target_flat.view(), rho.view());
        assert_eq!(grad.len(), target_flat.len());
        let max_abs = grad.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        assert!(
            max_abs > 1.0e-6,
            "expected non-zero isometry gradient on at least one component, max |grad|={max_abs}"
        );

        // FD check: bump one coord, refresh, compare value(t±h e_j) against
        // analytic grad[j]. Pick coord (row 0, axis 0).
        let h_fd = 1.0e-5;
        let probe_idx = 0usize; // (row=0, axis=0) flattens to 0.
        let mut coords_plus = coords.clone();
        coords_plus[[0, 0]] += h_fd;
        let mut coords_minus = coords.clone();
        coords_minus[[0, 0]] -= h_fd;

        refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
        let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
        let v_plus = penalty.value(target_plus.view(), rho.view());

        refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
        let target_minus: Array1<f64> = coords_minus.iter().copied().collect();
        let v_minus = penalty.value(target_minus.view(), rho.view());

        // Reinstall the base caches before reading grad at the base point.
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        let grad_base = penalty.grad_target(target_flat.view(), rho.view());

        let fd = (v_plus - v_minus) / (2.0 * h_fd);
        let analytic = grad_base[probe_idx];
        // Both `value` and `grad_target` use the cached `J` (and `grad_target`
        // also the cached `H`). With finite differencing the cache itself,
        // the analytic-vs-FD agreement bounds the entire pipeline (J build,
        // H build, accessor read, pullback metric, gradient assembly) to
        // O(h²) error. Tolerance 1e-3 leaves headroom for the per-evaluator
        // characteristic magnitude.
        assert!(
            (analytic - fd).abs() <= 1.0e-3 + 1.0e-4 * analytic.abs().max(fd.abs()),
            "isometry grad/FD mismatch at coord 0: analytic={analytic:.6e}, fd={fd:.6e}"
        );
    }

    #[test]
    fn isometry_wiring_periodic_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
            array![[0.12], [0.37], [0.58], [0.81]],
        );
    }

    #[test]
    fn isometry_wiring_sphere_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(SphereChartEvaluator),
            array![[-0.5, 0.3], [0.2, -1.1], [0.7, 0.9]],
        );
    }

    #[test]
    fn isometry_wiring_torus_matches_fd() {
        assert_isometry_wiring_matches_fd(
            Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
            array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
        );
    }

    fn deterministic_decoder(n_basis: usize, p_out: usize, seed: f64) -> Array2<f64> {
        Array2::<f64>::from_shape_fn((n_basis, p_out), |(i, j)| {
            let x = seed + 0.371 * (i as f64) - 0.193 * (j as f64) + 0.047 * ((i * j + 1) as f64);
            0.8 * x.sin() + 0.35 * (1.7 * x).cos()
        })
    }

    fn build_isometry_atom_for_evaluator(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        kind: SaeAtomBasisKind,
        coords: &Array2<f64>,
        p_out: usize,
        seed: f64,
    ) -> (SaeManifoldAtom, IsometryPenalty, Array1<f64>) {
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let decoder = deterministic_decoder(m, p_out, seed);
        let atom = SaeManifoldAtom::new(
            "exact_hvp_atom",
            kind,
            coords.ncols(),
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_second_jet(evaluator);
        let target_flat: Array1<f64> = coords.iter().copied().collect();
        let penalty = IsometryPenalty::new_euclidean(
            PsiSlice::full(target_flat.len(), Some(coords.ncols())),
            p_out,
        );
        (atom, penalty, target_flat)
    }

    fn assert_exact_isometry_hvp_matches_grad_fd(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        kind: SaeAtomBasisKind,
        coords: Array2<f64>,
        p_out: usize,
        direction: Array2<f64>,
    ) {
        let (atom, penalty, target_flat) =
            build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 0.91);
        let rho = array![0.0_f64];
        let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        assert!(
            installed,
            "second-jet cache must be installed for exact HVP test"
        );
        assert!(
            penalty.third_decoder_derivative().is_some(),
            "non-Duchon exact HVP requires a live refreshed third-decoder-jet cache"
        );
        let v: Array1<f64> = direction.iter().copied().collect();
        let exact = penalty.hvp(target_flat.view(), rho.view(), v.view());
        assert!(
            exact.iter().any(|x| x.abs() > 1.0e-7),
            "exact isometry HVP should be nonzero after K refresh; got {exact:?}"
        );

        let eps = 1.0e-6;
        let coords_plus = &coords + &(direction.mapv(|x| eps * x));
        let coords_minus = &coords - &(direction.mapv(|x| eps * x));
        let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
        let target_minus: Array1<f64> = coords_minus.iter().copied().collect();

        refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
        let grad_plus = penalty.grad_target(target_plus.view(), rho.view());
        refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
        let grad_minus = penalty.grad_target(target_minus.view(), rho.view());
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();

        let fd = (&grad_plus - &grad_minus).mapv(|x| x / (2.0 * eps));
        for i in 0..exact.len() {
            let err = (exact[i] - fd[i]).abs();
            let tol = 2.0e-4 + 3.0e-5 * exact[i].abs().max(fd[i].abs());
            assert!(
                err <= tol,
                "exact isometry HVP/grad-FD mismatch at flat index {i}: exact={:.12e}, fd={:.12e}, err={:.6e}, tol={:.6e}",
                exact[i],
                fd[i],
                err,
                tol
            );
        }
    }

    fn assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        kind: SaeAtomBasisKind,
        coords: Array2<f64>,
        p_out: usize,
        direction: Array2<f64>,
    ) {
        let (atom, penalty, target_flat) =
            build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 1.37);
        let rho = array![0.0_f64];
        let d = coords.ncols();

        // Build the reference metric from the EXACT SAME cache the exact HVP
        // differences against (#857). The exact HVP computes its residual
        // `diff = g/gbar − g_ref` where `g = penalty.pullback_metric(d)` is read
        // from `penalty`'s own Jacobian cache, and skips the third-jet `K` term
        // only when `diff == 0.0` (a bit-exact float compare). Previously `g_ref` was
        // built from a SEPARATE `scratch` penalty's cache, so a last-ULP
        // difference between the two independent refreshes left `diff` ~1e-16
        // rather than exactly 0; multiplied by the large third decoder jet
        // (`K ~ ω³`) for the torus/sphere bases, that leaked past the 1e-10
        // exact-equality bound. Refreshing `penalty` once and seeding the
        // UserSupplied reference from the normalized `penalty.pullback_metric(d)`
        // makes `g_ref` the identical array `g/gbar` is recomputed from, so the
        // residual is bit-zero and the K term is genuinely skipped — leaving
        // exactly the GN term. `with_reference` moves the penalty by value and
        // preserves every cache slot, so the J/J2/K caches read by the HVP are
        // unchanged.
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        let mut g_ref = penalty
            .pullback_metric(d)
            .expect("pullback metric is available after the cache refresh");
        let mut trace_sum = 0.0_f64;
        for row in 0..g_ref.nrows() {
            for axis in 0..d {
                trace_sum += g_ref[[row, axis * d + axis]];
            }
        }
        let normalizer = trace_sum / (g_ref.nrows() * d) as f64;
        for value in g_ref.iter_mut() {
            *value /= normalizer;
        }
        let penalty = penalty.with_reference(IsometryReference::UserSupplied(Arc::new(g_ref)));
        assert!(
            penalty.third_decoder_derivative().is_some(),
            "zero-residual exact/GN test must still carry the real refreshed K cache"
        );
        let v: Array1<f64> = direction.iter().copied().collect();
        let exact = penalty.hvp(target_flat.view(), rho.view(), v.view());
        let gn = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), v.view());
        assert!(
            gn.iter().any(|x| x.abs() > 1.0e-8),
            "GN block should be nonzero so exact/GN equality is not vacuous"
        );
        for i in 0..exact.len() {
            assert_abs_diff_eq!(exact[i], gn[i], epsilon = 1.0e-10);
        }
    }

    #[test]
    fn isometry_exact_hvp_sphere_matches_grad_fd_and_uses_refreshed_k() {
        assert_exact_isometry_hvp_matches_grad_fd(
            Arc::new(SphereChartEvaluator),
            SaeAtomBasisKind::Sphere,
            array![[-0.61, 0.23], [-0.18, -1.07], [0.42, 0.81], [0.73, -0.39]],
            4,
            array![[0.31, -0.27], [-0.18, 0.22], [0.14, 0.19], [-0.25, -0.11]],
        );
    }

    #[test]
    fn isometry_exact_hvp_torus_matches_grad_fd_and_uses_refreshed_k() {
        assert_exact_isometry_hvp_matches_grad_fd(
            Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
            SaeAtomBasisKind::Torus,
            array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
            3,
            array![[0.21, -0.16], [-0.24, 0.18], [0.13, 0.27]],
        );
    }

    #[test]
    fn isometry_exact_hvp_sphere_and_torus_collapse_to_gn_at_zero_residual() {
        assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
            Arc::new(SphereChartEvaluator),
            SaeAtomBasisKind::Sphere,
            array![[-0.52, 0.17], [-0.11, -0.93], [0.39, 0.74]],
            4,
            array![[0.17, -0.21], [-0.13, 0.08], [0.22, 0.19]],
        );
        assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
            Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
            SaeAtomBasisKind::Torus,
            array![[0.19, 0.31], [0.57, 0.73], [0.84, 0.12]],
            3,
            array![[0.11, -0.14], [-0.20, 0.07], [0.16, 0.23]],
        );
    }

    /// #457 root-cause regression: for every **non-Duchon** SAE basis the
    /// isometry penalty's *exact* `hvp` returns the zero vector (no third jet
    /// `K` cache outside the radial-Duchon source), so the Arrow-Schur coord
    /// curvature block — which routes through `psd_majorizer_hvp` — would carry
    /// **no isometry contribution at all**, and the pole fit diverges. The fix
    /// is the PSD Gauss-Newton majorizer override, which needs only the first
    /// and second decoder jets that `refresh_isometry_caches_from_atom`
    /// installs for any basis with an analytic second jet.
    ///
    /// This drives the real cache-refresh path with the sphere / circle /
    /// torus evaluators against the **Euclidean** reference (so the residual
    /// `g − I` is genuinely nonzero — the live production condition, unlike the
    /// zero-residual collapse test), then asserts the curvature operator the
    /// inner solve actually consumes is:
    ///   * genuinely **nonzero** (the bug was a silent zero block),
    ///   * **symmetric**, and
    ///   * **positive-semidefinite** (`vᵀB v ≥ 0`),
    /// pinning the exact seam #457 is about, end-to-end from the evaluator.
    fn assert_isometry_psd_majorizer_live_after_atom_refresh(
        evaluator: Arc<dyn SaeBasisSecondJet>,
        kind: SaeAtomBasisKind,
        coords: Array2<f64>,
        p_out: usize,
        probes: &[Array2<f64>],
    ) {
        let (atom, penalty, target_flat) =
            build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 0.53);
        let rho = array![0.0_f64];

        // Before any refresh the safe default is the zero block: confirm the
        // precondition so the post-refresh contrast is the genuine fix, not a
        // coincidence of a probe direction.
        let n = target_flat.len();
        let unit0 = {
            let mut e = Array1::<f64>::zeros(n);
            e[0] = 1.0;
            e
        };
        let pre = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), unit0.view());
        assert!(
            pre.iter().all(|x| *x == 0.0),
            "psd_majorizer_hvp without a cache must be the zero block; got {pre:?}"
        );

        let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        assert!(
            installed,
            "second-jet cache must install for the PSD-majorizer liveness test"
        );

        // The Euclidean reference makes g/gbar − I nonzero on this non-orthonormal
        // decoder; verify the residual is real so the curvature seam is the
        // production one (and not vacuously the zero-residual case).
        let d = coords.ncols();
        let g = penalty
            .pullback_metric(d)
            .expect("pullback metric available after refresh");
        let mut trace_sum = 0.0_f64;
        for row in 0..g.nrows() {
            for axis in 0..d {
                trace_sum += g[[row, axis * d + axis]];
            }
        }
        let normalizer = trace_sum / (g.nrows() * d) as f64;
        let mut residual_mass = 0.0_f64;
        for row in 0..g.nrows() {
            for a in 0..d {
                for b in 0..d {
                    // Euclidean reference is the identity metric I_d.
                    let g_ref = if a == b { 1.0 } else { 0.0 };
                    residual_mass += (g[[row, a * d + b]] / normalizer - g_ref).abs();
                }
            }
        }
        assert!(
            residual_mass > 1.0e-3,
            "Euclidean-reference residual must be nonzero for a real curvature test; \
             got residual mass {residual_mass:.3e}"
        );

        // Assemble the dense majorizer column-by-column via unit probes.
        let mut bmat = Array2::<f64>::zeros((n, n));
        for k in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[k] = 1.0;
            let col = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), e.view());
            for r in 0..n {
                bmat[[r, k]] = col[r];
            }
        }

        // Nonzero: the bug was a silent all-zero curvature block.
        let max_abs = bmat.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        assert!(
            max_abs > 1.0e-6,
            "isometry GN majorizer must be nonzero for a non-Duchon basis after refresh; \
             max |B| = {max_abs:.3e}"
        );

        // Symmetry: B = Σ_n (∂g/∂t)ᵀ(∂g/∂t) is symmetric by construction.
        for r in 0..n {
            for c in 0..n {
                assert_abs_diff_eq!(bmat[[r, c]], bmat[[c, r]], epsilon = 1.0e-10);
            }
        }

        // PSD: vᵀ B v ≥ 0 over a spread of probe directions.
        for probe in probes {
            let v: Array1<f64> = probe.iter().copied().collect();
            assert_eq!(v.len(), n, "probe must match the flattened target length");
            let bv = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), v.view());
            let quad = v.dot(&bv);
            assert!(
                quad >= -1.0e-9,
                "isometry GN majorizer must be PSD; got vᵀBv = {quad:.3e}"
            );
        }
    }

    #[test]
    fn isometry_psd_majorizer_live_after_sphere_refresh() {
        assert_isometry_psd_majorizer_live_after_atom_refresh(
            Arc::new(SphereChartEvaluator),
            SaeAtomBasisKind::Sphere,
            array![[-0.61, 0.23], [-0.18, -1.07], [0.42, 0.81]],
            4,
            &[
                array![[0.31, -0.27], [-0.18, 0.22], [0.14, 0.19]],
                array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                array![[-2.3, 0.6], [-0.1, 1.4], [0.8, -1.7]],
            ],
        );
    }

    #[test]
    fn isometry_psd_majorizer_live_after_circle_refresh() {
        assert_isometry_psd_majorizer_live_after_atom_refresh(
            Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
            SaeAtomBasisKind::Periodic,
            array![[0.12], [0.37], [0.58], [0.81]],
            3,
            &[
                array![[0.4], [-1.1], [0.7], [0.3]],
                array![[1.0], [1.0], [1.0], [1.0]],
                array![[-2.3], [0.6], [-0.1], [1.4]],
            ],
        );
    }

    #[test]
    fn isometry_psd_majorizer_live_after_torus_refresh() {
        assert_isometry_psd_majorizer_live_after_atom_refresh(
            Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
            SaeAtomBasisKind::Torus,
            array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
            3,
            &[
                array![[0.21, -0.16], [-0.24, 0.18], [0.13, 0.27]],
                array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                array![[-1.2, 0.5], [0.3, -0.9], [0.7, 0.2]],
            ],
        );
    }

    /// Multi-atom isometry pairing regression.
    ///
    /// Two SAE atoms share the same `(latent_dim, p_out)` signature but live
    /// on different coordinate blocks. The registry holds one isometry penalty
    /// per atom. The previous `find()` first-match logic paired *both*
    /// penalties to atom 0, so atom 1's coords were never installed into the
    /// second penalty's Jacobian cache — silently mislabeling the second
    /// atom's geometry as the first's. The positional pairing must instead
    /// refresh penalty `i` against atom `i`.
    ///
    /// We pin this by computing, independently, the Jacobian cache each atom
    /// would produce in isolation, then asserting that after
    /// `refresh_isometry_caches_from_term` the two registry penalties carry
    /// *distinct* caches matching their *own* atoms.
    #[test]
    fn refresh_isometry_caches_pairs_each_penalty_to_its_own_atom() {
        let latent_dim = 1usize;
        let p_out = 3usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());

        // Distinct coords per atom so the cached Jacobians must differ.
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let coords1 = array![[0.13], [0.41], [0.62], [0.91]];

        let build_atom = |name: &str, coords: &Array2<f64>, seed: f64| {
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let m = phi.ncols();
            let mut decoder = Array2::<f64>::zeros((m, p_out));
            for i in 0..m {
                for j in 0..p_out {
                    let x = (i as f64) * 0.371 + (j as f64) * 0.193 + seed;
                    decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
                }
            }
            let smooth = Array2::<f64>::eye(m);
            SaeManifoldAtom::new(
                name,
                SaeAtomBasisKind::Periodic,
                latent_dim,
                phi,
                jet,
                decoder,
                smooth,
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone() as Arc<dyn SaeBasisSecondJet>)
        };

        let atom0 = build_atom("atom0", &coords0, 0.5);
        let atom1 = build_atom("atom1", &coords1, 1.7);

        // Independent ground-truth caches: refresh a standalone penalty
        // against each atom in isolation.
        let slice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
        let control0 = IsometryPenalty::new_euclidean(slice0, p_out);
        refresh_isometry_caches_from_atom(&control0, &atom0, coords0.view()).unwrap();
        let expected0 = control0
            .jacobian_cache()
            .expect("control penalty 0 must have a Jacobian cache");

        let slice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
        let control1 = IsometryPenalty::new_euclidean(slice1, p_out);
        refresh_isometry_caches_from_atom(&control1, &atom1, coords1.view()).unwrap();
        let expected1 = control1
            .jacobian_cache()
            .expect("control penalty 1 must have a Jacobian cache");

        // The two atoms genuinely differ, else the test is vacuous.
        assert_ne!(
            *expected0, *expected1,
            "atom 0 and atom 1 must produce distinct Jacobian caches"
        );

        // Build the term and a registry with one isometry penalty per atom.
        let logits = Array2::<f64>::zeros((coords0.nrows(), 2));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

        let mut registry = AnalyticPenaltyRegistry::new();
        let pslice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
        let pslice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(pslice0, p_out),
        )));
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(pslice1, p_out),
        )));

        let coords_per_atom = vec![coords0.clone(), coords1.clone()];
        let refreshed =
            refresh_isometry_caches_from_term(&registry, &term, &coords_per_atom).unwrap();
        assert_eq!(refreshed, 2, "both penalties should install second caches");

        let cache0 = match &registry.penalties[0] {
            AnalyticPenaltyKind::Isometry(p) => p
                .jacobian_cache()
                .expect("penalty 0 cache must be populated"),
            _ => panic!("expected isometry penalty at index 0"),
        };
        let cache1 = match &registry.penalties[1] {
            AnalyticPenaltyKind::Isometry(p) => p
                .jacobian_cache()
                .expect("penalty 1 cache must be populated"),
            _ => panic!("expected isometry penalty at index 1"),
        };

        // Penalty i must carry atom i's cache — not both atom 0's.
        assert_eq!(
            *cache0, *expected0,
            "penalty 0 must be refreshed against atom 0"
        );
        assert_eq!(
            *cache1, *expected1,
            "penalty 1 must be refreshed against atom 1 (regression: old find() paired it to atom 0)"
        );
        assert_ne!(
            *cache0, *cache1,
            "the two penalties must not collapse onto the same atom"
        );
    }

    /// Build a minimal single-atom periodic SAE outer objective for the
    /// warm-start contract tests (gam#577 / gam#579).
    fn warmstart_test_objective() -> SaeManifoldOuterObjective {
        let coords = array![[0.10], [0.35], [0.62], [0.88]];
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            // Decoder mapping the 3 basis fns to a single output channel.
            array![[0.30], [-0.20], [0.15]],
            // Mild ridge-like smoothness penalty so the inner solve is PD.
            Array2::<f64>::eye(3),
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode(
            // Nonzero assignment mass so H_tt carries genuine data curvature.
            array![[0.9_f64], [0.8], [0.7], [0.6]],
            vec![coords],
            AssignmentMode::softmax(0.7),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
        SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
    }

    fn near_singular_outer_gradient_cache() -> ArrowFactorCache {
        ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![array![[1.0_f64, 0.0], [0.0, 1.0e-7]]]),
            htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
            schur_factor: Some(array![[1.0_f64]]),
            solver_mode: ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
            d: 2,
            row_dims: Arc::from(vec![2usize].into_boxed_slice()),
            row_offsets: Arc::from(vec![0usize, 2usize].into_boxed_slice()),
            k: 1,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
        }
    }

    fn diagonal_latent_cache(diagonal: &[f64]) -> ArrowFactorCache {
        let dim = diagonal.len();
        let mut factor = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            factor[[i, i]] = diagonal[i].sqrt();
        }
        ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![factor]),
            htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
            schur_factor: None,
            solver_mode: ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
            d: dim,
            row_dims: Arc::from(vec![dim].into_boxed_slice()),
            row_offsets: Arc::from(vec![0usize, dim].into_boxed_slice()),
            k: 0,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
        }
    }

    #[test]
    fn outer_gradient_solver_rejects_near_singular_cache_without_matching_gauge() {
        let cache = near_singular_outer_gradient_cache();
        let obj = warmstart_test_objective();
        let err = match obj.term.outer_gradient_arrow_solver(&cache) {
            Err(err) => err,
            Ok(..) => panic!("near-singular evidence factor without a matching gauge must reject"),
        };
        assert!(
            err.contains("analytic outer gradient undefined at this rho"),
            "guard error must name the undefined analytic-gradient condition; got: {err}"
        );
        assert!(
            err.contains("min/max pivot ratio") && err.contains("floor"),
            "guard error must report the pivot ratio and floor; got: {err}"
        );
    }

    #[test]
    fn deflated_solver_matches_plain_solve_when_no_gauge_is_installed() {
        let cache = diagonal_latent_cache(&[2.0_f64, 5.0, 7.0]);
        let solver = DeflatedArrowSolver::plain(&cache);
        let rhs_t = array![4.0_f64, 10.0, -14.0];
        let rhs_beta = Array1::<f64>::zeros(0);
        let (plain_t, plain_beta) = cache
            .full_inverse_apply(rhs_t.view(), rhs_beta.view())
            .expect("plain cache solve");
        let solved = solver
            .solve(rhs_t.view(), rhs_beta.view())
            .expect("adapter solve");
        assert_eq!(solved.t.len(), plain_t.len());
        for idx in 0..plain_t.len() {
            assert_abs_diff_eq!(solved.t[idx], plain_t[idx], epsilon = 0.0);
        }
        assert_eq!(solved.beta.len(), plain_beta.len());
        for idx in 0..plain_beta.len() {
            assert_abs_diff_eq!(solved.beta[idx], plain_beta[idx], epsilon = 0.0);
        }
    }

    #[test]
    fn deflated_solver_matches_dense_quotient_pseudoinverse_on_near_null_fixture() {
        let cache = diagonal_latent_cache(&[2.0_f64, 1.0e-14]);
        let gauge = array![0.0_f64, 1.0];
        let solver = DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge], 2.0)
            .expect("deflated solver");
        let rhs_beta = Array1::<f64>::zeros(0);

        let physical_rhs = array![4.0_f64, 0.0];
        let solved = solver
            .solve(physical_rhs.view(), rhs_beta.view())
            .expect("physical solve");
        let oracle = array![2.0_f64, 0.0];
        for idx in 0..oracle.len() {
            assert_abs_diff_eq!(solved.t[idx], oracle[idx], epsilon = 1.0e-12);
        }

        let gauge_rhs = array![0.0_f64, 1.0];
        let plain = cache
            .full_inverse_apply(gauge_rhs.view(), rhs_beta.view())
            .expect("plain gauge solve")
            .0;
        let stiffened = solver
            .solve(gauge_rhs.view(), rhs_beta.view())
            .expect("stiffened gauge solve")
            .t;
        assert!(plain[1] > 1.0e13, "plain near-null solve must be huge");
        assert_abs_diff_eq!(stiffened[1], 0.5, epsilon = 1.0e-12);
    }

    /// gam#577 / gam#579 root cause: the continuation pre-warm forwards an
    /// EMPTY β before the first accepted eval (`state.last_beta` starts
    /// empty). The seed hook must treat that as the documented "no warm-start
    /// available, proceed cold" no-op (`SeedOutcome::NoSlot`) rather than
    /// erroring on `β length 0 != decoder dim` — the error dropped EVERY
    /// continuation seed and forced a full cold solve on every outer seed.
    #[test]
    fn seed_inner_state_accepts_empty_beta_as_noslot() {
        let mut obj = warmstart_test_objective();
        let empty: Array1<f64> = Array1::zeros(0);
        let outcome = obj
            .seed_inner_state(&empty)
            .expect("empty-β seed must be accepted as a no-op, not rejected (gam#577/#579)");
        assert!(
            matches!(outcome, SeedOutcome::NoSlot),
            "empty-β seed must report NoSlot (proceed cold); got {outcome:?}"
        );
    }

    /// A populated β whose length matches the decoder dimension must be
    /// INSTALLED and then GENUINELY REUSED by the next inner solve — this is
    /// the warm-start the continuation walk relies on for the big speedup
    /// (gam#577 / gam#579). We verify reuse behaviorally: seed a known β, run
    /// one eval with zero inner Newton iterations (so the solve cannot move
    /// β off the seed), and confirm the published `inner_beta_hint` is exactly
    /// the seeded β. A cold start would have published the term's pristine β
    /// instead.
    #[test]
    fn seed_inner_state_installs_and_reuses_matching_beta() {
        let mut obj = warmstart_test_objective();
        let dim = obj.term.beta_dim();
        // A distinctive seed that differs from the term's pristine decoder.
        let pristine = obj.term.flatten_beta();
        let seed: Array1<f64> =
            Array1::from_shape_fn(dim, |i| pristine[i] + 0.5 + 0.01 * (i as f64));
        assert!(
            (&seed - &pristine).iter().any(|d| d.abs() > 1e-6),
            "seed must differ from the pristine β for the reuse check to be meaningful"
        );

        let outcome = obj
            .seed_inner_state(&seed)
            .expect("a length-matching β must install");
        assert!(
            matches!(outcome, SeedOutcome::Installed),
            "matching β must report Installed; got {outcome:?}"
        );

        // Freeze the inner solve at zero Newton iterations: β cannot move off
        // the warm-start, so the published hint must equal the seed exactly.
        obj.inner_max_iter = 0;
        let rho_flat = obj.baseline_rho.to_flat();
        let eval = OuterObjective::eval(&mut obj, &rho_flat)
            .expect("eval at the warm-started β must succeed");
        let hint = eval
            .inner_beta_hint
            .expect("the SAE objective must publish inner_beta_hint for continuation reuse");
        assert_eq!(
            hint.len(),
            dim,
            "published hint must have decoder dimension"
        );
        for (i, (&h, &s)) in hint.iter().zip(seed.iter()).enumerate() {
            assert!(
                (h - s).abs() < 1e-12,
                "warm-started β must be reused verbatim by the inner solve at coord {i}: \
                 hint {h} != seed {s} (gam#577/#579)"
            );
        }
    }

    /// The seed contract is only relaxed for the EMPTY sentinel. A populated
    /// β whose length disagrees with the decoder dimension is a genuine
    /// layout bug and must still surface a typed error rather than being
    /// silently dropped.
    #[test]
    fn seed_inner_state_rejects_wrong_length_populated_beta() {
        let mut obj = warmstart_test_objective();
        let dim = obj.term.beta_dim();
        let wrong: Array1<f64> = Array1::zeros(dim + 1);
        let err = obj
            .seed_inner_state(&wrong)
            .expect_err("a populated β of the wrong length must be rejected");
        match err {
            EstimationError::RemlOptimizationFailed(msg) => {
                assert!(
                    msg.contains("decoder dim"),
                    "error must name the decoder-dim mismatch; got: {msg}"
                );
            }
            other => panic!("expected RemlOptimizationFailed, got {other:?}"),
        }
    }

    /// Build a non-periodic 1-D atom with a genuine order-2 finite-difference
    /// roughness Gram, a non-constant-speed decoder, and explicit
    /// `(basis_values, basis_jacobian)` so the intrinsic reweighting in
    /// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`] is exercised
    /// directly. A localized (near-diagonal) basis makes each coefficient's
    /// representative speed the speed at its own sample.
    fn intrinsic_test_atom(jacobian_scale: f64) -> SaeManifoldAtom {
        let m = 5usize;
        let n = m;
        let p = 1usize;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        let mut decoder = Array2::<f64>::zeros((m, p));
        for mu in 0..m {
            // Localized basis: Φ_μ(t_n) ≈ δ_{nμ}.
            phi[[mu, mu]] = 1.0;
            // Per-sample basis derivative (axis 0) grows with μ — a
            // non-constant-speed curve — scaled by `jacobian_scale` to emulate
            // a global linear reparameterization t -> t / jacobian_scale.
            jet[[mu, mu, 0]] = jacobian_scale * (1.0 + mu as f64);
            decoder[[mu, 0]] = 1.0;
        }
        let s_raw = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        SaeManifoldAtom::new(
            "intrinsic-1d",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            s_raw,
        )
        .unwrap()
    }

    /// The roughness operator order is recovered from the raw Gram's null
    /// space: an order-2 difference penalty annihilates the affine functions,
    /// so `nullity = 2` and the arc-length exponent is `β = ½ − 2 = −3/2`.
    #[test]
    fn intrinsic_penalty_recovers_order_two_from_nullity() {
        let atom = intrinsic_test_atom(1.0);
        assert_eq!(atom.smooth_penalty_order, 2);
    }

    #[test]
    fn line_search_snapshot_restores_intrinsic_smooth_penalty() {
        let atom = intrinsic_test_atom(1.0);
        let n = atom.n_obs();
        let logits = Array2::<f64>::zeros((n, 1));
        let coords = vec![Array2::<f64>::zeros((n, 1))];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let original = term.atoms[0].smooth_penalty.clone();
        let snapshot = term.snapshot_mutable_state();

        term.atoms[0].decoder_coefficients[[0, 0]] *= 3.0;
        term.atoms[0].refresh_intrinsic_smooth_penalty();
        let changed = (&term.atoms[0].smooth_penalty - &original)
            .mapv(f64::abs)
            .sum();
        assert!(
            changed > 1e-6,
            "test setup must perturb the live intrinsic smoothness Gram"
        );

        term.restore_mutable_state(&snapshot);
        let restored = (&term.atoms[0].smooth_penalty - &original)
            .mapv(f64::abs)
            .sum();
        assert!(
            restored < 1e-12,
            "line-search restore left a stale intrinsic smoothness Gram: {restored}"
        );
    }

    /// Gauge invariance (issue #673): a global reparameterization of the latent
    /// coordinate scales every per-sample speed by a common factor, which
    /// cancels in the centered reweighting — so the intrinsic Gram `S̃` (and
    /// hence the topology evidence `tr(BᵀS̃B)`) is identical across the two
    /// reparameterizations, even though the basis Jacobian (the metric) differs.
    #[test]
    fn intrinsic_penalty_is_invariant_to_speed_rescaling() {
        let a1 = intrinsic_test_atom(1.0);
        let a2 = intrinsic_test_atom(7.5);
        // Same raw Gram and decoder; only the basis Jacobian (speed) differs.
        assert_abs_diff_eq!(
            (&a1.smooth_penalty_raw - &a2.smooth_penalty_raw)
                .mapv(f64::abs)
                .sum(),
            0.0,
            epsilon = 1e-12
        );
        // The intrinsic (reweighted) Gram is identical despite the 7.5x speed
        // rescale: the centered ratios are invariant to a global speed factor.
        let diff = (&a1.smooth_penalty - &a2.smooth_penalty)
            .mapv(f64::abs)
            .sum();
        assert!(
            diff < 1e-9,
            "intrinsic Gram changed under a global speed rescale (gauge leak): {diff}"
        );
    }

    fn affine_canonicalization_test_term() -> SaeManifoldTerm {
        let n = 80usize;
        let p = 2usize;
        let evaluator = EuclideanPatchEvaluator::new(1, 2).unwrap();
        let mut coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            coords[[row, 0]] = -4.0 + 12.0 * row as f64 / (n as f64 - 1.0);
        }
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[0, 0]] = 0.8;
        decoder[[1, 0]] = -0.4;
        decoder[[2, 0]] = 0.15;
        decoder[[0, 1]] = -0.2;
        decoder[[1, 1]] = 0.9;
        decoder[[2, 1]] = -0.08;
        let smooth_penalty = crate::basis::create_difference_penalty_matrix(3, 2, None).unwrap();
        let atom = SaeManifoldAtom::new(
            "affine-canonicalization",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            smooth_penalty,
        )
        .unwrap()
        .with_basis_second_jet(Arc::new(evaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom], assignment).unwrap()
    }

    #[test]
    fn affine_canonicalization_transports_live_penalty_instead_of_recomputing() {
        let mut term = affine_canonicalization_test_term();
        let before = term.decoder_smoothness_quadratic_form();
        let old_smooth_penalty = term.atoms[0].smooth_penalty.clone();
        let old_decoder = term.atoms[0].decoder_coefficients.clone();

        term.canonicalize_atom_affine_gauge(0).unwrap();
        let after = term.decoder_smoothness_quadratic_form();
        let invariant_gap = (after - before).abs() / before.abs().max(1.0);
        assert!(
            invariant_gap < 1.0e-9,
            "canonicalization changed fixed-rho smoothness energy: before={before:.12e}, after={after:.12e}"
        );

        let mut recomputed_atom = term.atoms[0].clone();
        recomputed_atom.refresh_intrinsic_smooth_penalty();
        let recomputed_term = SaeManifoldTerm::new(
            vec![recomputed_atom],
            SaeAssignment::from_blocks_with_mode_and_manifolds(
                Array2::<f64>::zeros((term.n_obs(), 1)),
                vec![term.assignment.coords[0].as_matrix()],
                vec![LatentManifold::Euclidean],
                AssignmentMode::softmax(1.0),
            )
            .unwrap(),
        )
        .unwrap();
        let recomputed = recomputed_term.decoder_smoothness_quadratic_form();
        let recompute_jump = (recomputed - before).abs() / before.abs().max(1.0);
        assert!(
            recompute_jump > 1.0e-2,
            "test fixture failed to expose the intrinsic recompute energy jump: before={before:.12e}, recomputed={recomputed:.12e}"
        );

        let transport =
            solve_basis_transport(term.atoms[0].basis_values.view(), old_smooth_penalty.view())
                .expect_err("shape mismatch must reject invalid transport solve");
        assert!(
            transport.contains("row mismatch") || transport.contains("SVD failed"),
            "unexpected transport-shape diagnostic: {transport}"
        );
        let roundtrip = transport_smooth_penalty_for_decoder(
            solve_design_least_squares(
                term.atoms[0].decoder_coefficients.view(),
                old_decoder.view(),
            )
            .unwrap_or_else(|err| panic!("decoder transport fixture became singular: {err}"))
            .view(),
            old_smooth_penalty.view(),
        );
        assert!(
            roundtrip.is_err(),
            "non-square decoder transport must not be accepted as a penalty congruence"
        );
    }

    /// Non-constant speed genuinely reshapes the penalty: the intrinsic Gram
    /// must differ from the raw Gram when the decoder curve is not
    /// constant-speed, otherwise the reweighting is a no-op and the gauge fix
    /// would be vacuous. The congruence preserves symmetry.
    #[test]
    fn intrinsic_penalty_differs_from_raw_under_varying_speed() {
        let atom = intrinsic_test_atom(1.0);
        let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
            .mapv(f64::abs)
            .sum();
        assert!(
            diff > 1e-6,
            "intrinsic reweighting was a no-op on a non-constant-speed curve: {diff}"
        );
        for i in 0..atom.basis_size() {
            for j in 0..atom.basis_size() {
                assert_abs_diff_eq!(
                    atom.smooth_penalty[[i, j]],
                    atom.smooth_penalty[[j, i]],
                    epsilon = 1e-12
                );
            }
        }
    }

    /// Constant-speed atoms are untouched: when every sample shares one speed
    /// (the periodic sin/cos limit), the centered weights are all `1`, so
    /// `S̃ = S_raw` exactly and the topology comparison among constant-speed
    /// atoms is unaffected.
    #[test]
    fn intrinsic_penalty_leaves_constant_speed_atom_unchanged() {
        let m = 6usize;
        let n = m;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        let mut decoder = Array2::<f64>::zeros((m, 1));
        for mu in 0..m {
            phi[[mu, mu]] = 1.0;
            // Identical derivative magnitude at every sample => constant speed.
            jet[[mu, mu, 0]] = 2.0;
            decoder[[mu, 0]] = 1.0;
        }
        let s_raw = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        let atom = SaeManifoldAtom::new(
            "constant-speed",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            s_raw,
        )
        .unwrap();
        let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
            .mapv(f64::abs)
            .sum();
        assert!(
            diff < 1e-9,
            "constant-speed atom's penalty was reweighted (should be identity): {diff}"
        );
    }

    #[test]
    fn pca_seed_handles_huge_equal_finite_columns_without_mean_overflow() {
        let z = array![[1.0e308_f64, 1.0e308], [1.0e308, 1.0e308]];
        let coords =
            sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1]).unwrap();
        assert_eq!(coords.dim(), (1, 2, 1));
        assert!(
            coords.iter().all(|value| value.is_finite()),
            "huge finite equal columns must not overflow the PCA seed mean: {coords:?}"
        );
    }

    #[test]
    fn pca_seed_rejects_huge_finite_span_that_overflows_centering() {
        let z = array![[1.0e308_f64, 0.0], [-1.0e308, 0.0]];
        let err = sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
            .expect_err("opposite huge finite values exceed f64 centering range");
        assert!(
            err.contains("centered Z is non-finite") || err.contains("SVD failed"),
            "unexpected PCA seed error: {err}"
        );
    }

    // ---- Issue #972: low-rank Grassmann decoder frame verification ----

    /// `polar(M) = W Vᵀ` is exactly column-orthonormal and equals `M` when `M`
    /// is already orthonormal (idempotence of the polar projection on the
    /// Stiefel manifold), and recovers the planted span of a low-rank decoder.
    #[test]
    fn planted_low_rank_frame_recovered_by_polar() {
        let p = 12usize;
        let r = 3usize;
        let n = 200usize;
        // Planted orthonormal frame: first `r` canonical axes (any rotation
        // would do; canonical axes make the angle assertion transparent).
        let mut planted = Array2::<f64>::zeros((p, r));
        for j in 0..r {
            planted[[j, j]] = 1.0;
        }
        // Latent coords drive targets onto the planted span: targets = coords·plantedᵀ.
        let mut coords = Array2::<f64>::zeros((n, r));
        for i in 0..n {
            for j in 0..r {
                // Deterministic, index-keyed pseudo-data (no clock RNG).
                let x = ((i * 7 + j * 13 + 1) % 97) as f64 / 97.0 - 0.5;
                coords[[i, j]] = x;
            }
        }
        let targets = fast_abt(&coords, &planted);
        let angle =
            grassmann_recover_planted_span_angle(targets.view(), coords.view(), planted.view())
                .expect("span recovery");
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);

        // Polar of an already-orthonormal frame is itself (up to canonical sign).
        let frame = GrassmannFrame::polar_update(planted.view()).expect("polar");
        let recovered_angle = frame
            .max_principal_angle(planted.view())
            .expect("principal angle");
        assert_abs_diff_eq!(recovered_angle, 0.0, epsilon = 1.0e-9);
        // Orthonormality: UᵀU = I_r.
        let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
        for i in 0..r {
            for j in 0..r {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
            }
        }
    }

    /// Build a low-rank decoder atom (`p` large, true column rank `r ≪ p`) and
    /// verify the auto-activation installs a frame, the factored border holds
    /// exactly `Σ M_k·r_k`, and reconstruction recovers `B_k` to machine
    /// precision.
    #[test]
    fn factored_border_dim_invariant_and_reconstruction() {
        let m = 6usize;
        let p = 16usize;
        let r = 2usize;
        // B = C0 · Frameᵀ with a planted rank-`r` column span.
        let mut frame = Array2::<f64>::zeros((p, r));
        frame[[0, 0]] = 1.0;
        frame[[1, 1]] = 1.0;
        let mut c0 = Array2::<f64>::zeros((m, r));
        for mu in 0..m {
            c0[[mu, 0]] = 1.0 + mu as f64;
            c0[[mu, 1]] = 0.5 * mu as f64 - 1.0;
        }
        let decoder = fast_abt(&c0, &frame);
        let mut phi = Array2::<f64>::zeros((m, m));
        let mut jet = Array3::<f64>::zeros((m, m, 1));
        for mu in 0..m {
            phi[[mu, mu]] = 1.0;
            jet[[mu, mu, 0]] = 1.0;
        }
        let s_raw = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        let mut atom = SaeManifoldAtom::new(
            "lowrank",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder.clone(),
            s_raw,
        )
        .unwrap();
        let activated = atom.maybe_activate_decoder_frame().expect("activate");
        assert_eq!(
            activated,
            Some(r),
            "rank-{r} decoder should profile to r={r}"
        );
        assert_eq!(atom.border_frame_rank(), r);
        assert_eq!(atom.frame_manifold_dimension(), r * (p - r));

        // Reconstruction recovers B_k to machine precision.
        let coords = atom.factored_coordinates().unwrap().expect("coords");
        assert_eq!(coords.dim(), (m, r));
        let reconstructed = atom
            .reconstruct_decoder_coefficients(coords.view())
            .unwrap();
        for mu in 0..m {
            for j in 0..p {
                assert_abs_diff_eq!(reconstructed[[mu, j]], decoder[[mu, j]], epsilon = 1.0e-9);
            }
        }

        let term = SaeManifoldTerm::new(
            vec![atom],
            SaeAssignment::from_blocks_with_mode(
                Array2::<f64>::zeros((m, 1)),
                vec![Array2::<f64>::zeros((m, 1))],
                AssignmentMode::softmax(0.7),
            )
            .unwrap(),
        )
        .unwrap();
        // Border-size invariant: factored border == Σ M_k·r_k.
        grassmann_assert_border_dim_invariant(&term).expect("border invariant");
        assert_eq!(term.factored_border_dim(), m * r);
        assert_eq!(term.grassmann_evidence_dimension(), r * (p - r));
        // Round-trip flatten/scatter of the factored border preserves B_k.
        let mut term = term;
        let border = term.flatten_factored_border().unwrap();
        assert_eq!(border.len(), m * r);
        let saved = term.atoms[0].decoder_coefficients.clone();
        term.scatter_factored_border(border.view()).unwrap();
        for mu in 0..m {
            for j in 0..p {
                assert_abs_diff_eq!(
                    term.atoms[0].decoder_coefficients[[mu, j]],
                    saved[[mu, j]],
                    epsilon = 1.0e-9
                );
            }
        }
    }

    #[test]
    fn factored_beta_penalty_probing_matches_projected_dense_curvature() {
        let k_atoms = 2usize;
        let m = 4usize;
        let p = 24usize;
        let r = 2usize;
        let n_obs = 5usize;
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut coord_blocks = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            let mut frame = Array2::<f64>::zeros((p, r));
            frame[[atom_idx * r, 0]] = 1.0;
            frame[[atom_idx * r + 1, 1]] = 1.0;
            let mut coords = Array2::<f64>::zeros((n_obs, 1));
            for row in 0..n_obs {
                coords[[row, 0]] = row as f64;
            }
            let mut phi = Array2::<f64>::zeros((n_obs, m));
            let mut jet = Array3::<f64>::zeros((n_obs, m, 1));
            for row in 0..n_obs {
                for basis_col in 0..m {
                    let x = (row + 1) as f64 * (basis_col + 1) as f64;
                    phi[[row, basis_col]] = 0.05 * x + if row == basis_col { 1.0 } else { 0.0 };
                    jet[[row, basis_col, 0]] = 0.01 * x;
                }
            }
            let mut c = Array2::<f64>::zeros((m, r));
            for basis_col in 0..m {
                c[[basis_col, 0]] = 0.3 + 0.07 * (basis_col + atom_idx) as f64;
                c[[basis_col, 1]] = -0.2 + 0.05 * (basis_col * 2 + atom_idx) as f64;
            }
            let decoder = fast_abt(&c, &frame);
            let mut atom = SaeManifoldAtom::new(
                "factored_probe",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap();
            atom.maybe_activate_decoder_frame()
                .expect("frame activation")
                .expect("rank-2 atom should activate a frame");
            atoms.push(atom);
            coord_blocks.push(coords);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::from_elem((n_obs, k_atoms), 0.25),
            coord_blocks,
            vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        assert!(term.frames_active());
        assert_eq!(term.factored_border_dim(), k_atoms * m * r);

        let beta_len = term.beta_dim();
        let mut registry = AnalyticPenaltyRegistry::new();
        let nuclear = NuclearNormPenalty::new(
            PsiSlice {
                range: 0..beta_len,
                latent_dim: Some(beta_len / p),
            },
            0.7,
            p,
            1.0e-4,
            None,
            false,
        )
        .unwrap();
        registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(nuclear)));
        let incoherence = DecoderIncoherencePenalty::new(
            PsiSlice {
                range: 0..beta_len,
                latent_dim: Some(beta_len / p),
            },
            vec![m, m],
            p,
            Array2::<f64>::from_elem((k_atoms, k_atoms), 0.5),
            0.6,
            false,
        )
        .unwrap();
        registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(
            incoherence,
        )));

        let mut dense_sys = ArrowSchurSystem::new(0, 0, beta_len);
        let dense_assembly = term
            .add_sae_analytic_penalty_contributions(
                &mut dense_sys,
                &registry,
                1.0,
                None,
                true,
                None,
            )
            .unwrap();
        assert!(dense_assembly.dense_written);
        assert!(!dense_assembly.deferred_factored);

        let projection = FrameProjection::new(&term);
        let border_dim = term.factored_border_dim();
        let projected = term.project_dense_penalty_to_factored(dense_sys.hbb.view(), &projection);
        let direct = term.build_factored_beta_penalty_curvature(&registry, 1.0, &projection);
        for row in 0..border_dim {
            for col in 0..border_dim {
                assert_abs_diff_eq!(direct[[row, col]], projected[[row, col]], epsilon = 1.0e-10);
            }
        }

        let mut deferred_term = term.clone();
        let rho = SaeManifoldRho::new(
            0.0,
            -20.0,
            vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
        );
        let target = Array2::<f64>::zeros((n_obs, p));
        let sys = deferred_term
            .assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
                target.view(),
                &rho,
                Some(&registry),
                1.0,
                1,
            )
            .unwrap();
        assert_eq!(sys.k, border_dim);
        assert!(sys.hbb.is_empty());
    }

    fn materialize_row_htbeta_for_test(sys: &ArrowSchurSystem, row_idx: usize) -> Array2<f64> {
        let di = sys.row_dims[row_idx];
        let k = sys.k;
        let row = &sys.rows[row_idx];
        let use_dense = sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none();
        let mut out = if use_dense && row.htbeta.dim() == (di, k) {
            row.htbeta.clone()
        } else {
            Array2::<f64>::zeros((di, k))
        };
        if let Some(op) = sys.htbeta_matvec.as_ref() {
            let mut basis = Array1::<f64>::zeros(k);
            let mut col = Array1::<f64>::zeros(di);
            for beta_col in 0..k {
                basis.fill(0.0);
                basis[beta_col] = 1.0;
                col.fill(0.0);
                op(row_idx, basis.view(), &mut col);
                for row_col in 0..di {
                    out[[row_col, beta_col]] += col[row_col];
                }
            }
        }
        out
    }

    fn project_row_htbeta_to_factored_for_test(
        term: &SaeManifoldTerm,
        htbeta_b: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        FrameProjection::new(term).project_rows(htbeta_b)
    }

    fn low_rank_factored_htbeta_term(
        k_atoms: usize,
        m: usize,
        p: usize,
        frame_rank: usize,
        latent_dim: usize,
        n_obs: usize,
    ) -> SaeManifoldTerm {
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut coord_blocks = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            let coords = Array2::from_shape_fn((n_obs, latent_dim), |(row, axis)| {
                let phase = (row + 1) as f64 * (axis + 2) as f64 + 0.37 * (atom_idx + 1) as f64;
                0.2 * phase.sin() + 0.1 * (0.17 * phase).cos()
            });
            let mut phi = Array2::<f64>::zeros((n_obs, m));
            let mut jet = Array3::<f64>::zeros((n_obs, m, latent_dim));
            for row in 0..n_obs {
                for basis_col in 0..m {
                    let base = (row + 1) as f64 * (basis_col + 1) as f64;
                    phi[[row, basis_col]] = if basis_col == 0 { 1.0 } else { 0.0 }
                        + 0.01 * (base + 3.0 * atom_idx as f64).sin();
                    for axis in 0..latent_dim {
                        jet[[row, basis_col, axis]] =
                            0.005 * ((base * (axis + 1) as f64) + atom_idx as f64).cos();
                    }
                }
            }
            let mut frame = Array2::<f64>::zeros((p, frame_rank));
            for frame_col in 0..frame_rank {
                frame[[(atom_idx * frame_rank + frame_col) % p, frame_col]] = 1.0;
            }
            let coords_c = Array2::from_shape_fn((m, frame_rank), |(basis_col, frame_col)| {
                0.2 + 0.03 * (basis_col + 2 * frame_col + atom_idx) as f64
            });
            let decoder = coords_c.dot(&frame.t());
            let mut atom = SaeManifoldAtom::new(
                "factored_htbeta_shape",
                SaeAtomBasisKind::EuclideanPatch,
                latent_dim,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap();
            atom.maybe_activate_decoder_frame()
                .expect("frame activation")
                .expect("low-rank atom should activate a frame");
            atoms.push(atom);
            coord_blocks.push(coords);
        }
        let logits = Array2::<f64>::from_shape_fn((n_obs, k_atoms), |(row, atom)| {
            0.03 * ((row + 1) as f64 * (atom + 2) as f64).sin()
        });
        let manifolds =
            vec![LatentManifold::Product(vec![LatentManifold::Euclidean; latent_dim]); k_atoms];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            manifolds,
            AssignmentMode::softmax(0.9),
        )
        .unwrap();
        SaeManifoldTerm::new(atoms, assignment).unwrap()
    }

    fn factored_htbeta_rho(k_atoms: usize, latent_dim: usize) -> SaeManifoldRho {
        SaeManifoldRho::new(0.0, -0.2, vec![Array1::<f64>::zeros(latent_dim); k_atoms])
    }

    #[test]
    fn factored_row_htbeta_native_solve_matches_full_b_then_project() {
        let k_atoms = 2usize;
        let m = 4usize;
        let p = 24usize;
        let r = 2usize;
        let n_obs = 5usize;
        let mut atoms = Vec::with_capacity(k_atoms);
        let mut coord_blocks = Vec::with_capacity(k_atoms);
        for atom_idx in 0..k_atoms {
            let mut frame = Array2::<f64>::zeros((p, r));
            frame[[atom_idx * r, 0]] = 1.0;
            frame[[atom_idx * r + 1, 1]] = 1.0;
            let coords = Array2::from_shape_fn((n_obs, 1), |(row, _)| 0.1 * (row + 1) as f64);
            let mut phi = Array2::<f64>::zeros((n_obs, m));
            let mut jet = Array3::<f64>::zeros((n_obs, m, 1));
            for row in 0..n_obs {
                for basis_col in 0..m {
                    let x = (row + 1) as f64 * (basis_col + 1) as f64;
                    phi[[row, basis_col]] = 0.03 * x + if row % m == basis_col { 1.0 } else { 0.0 };
                    jet[[row, basis_col, 0]] = 0.02 * x;
                }
            }
            let c = Array2::from_shape_fn((m, r), |(basis_col, frame_col)| {
                0.2 + 0.04 * (basis_col + 2 * frame_col + atom_idx) as f64
            });
            let decoder = fast_abt(&c, &frame);
            let mut atom = SaeManifoldAtom::new(
                "factored_row_native",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap();
            atom.maybe_activate_decoder_frame()
                .expect("frame activation")
                .expect("rank-2 atom should activate a frame");
            atoms.push(atom);
            coord_blocks.push(coords);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::from_shape_fn((n_obs, k_atoms), |(row, atom)| {
                0.15 * (row + 1) as f64 - 0.07 * atom as f64
            }),
            coord_blocks,
            vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
            AssignmentMode::softmax(0.9),
        )
        .unwrap();
        let mut factored_term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        assert!(factored_term.frames_active());
        let border_dim = factored_term.factored_border_dim();
        assert!(border_dim < factored_term.beta_dim());

        let mut full_term = factored_term.clone();
        for atom in &mut full_term.atoms {
            atom.deactivate_decoder_frame();
        }
        let rho = SaeManifoldRho::new(
            0.0,
            -0.2,
            vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
        );
        let target = Array2::<f64>::from_shape_fn((n_obs, p), |(row, col)| {
            0.01 * (row + 1) as f64 - 0.002 * (col + 1) as f64
        });

        let native_sys = factored_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        assert_eq!(native_sys.k, border_dim);
        assert!(native_sys.htbeta_matvec.is_none());
        assert!(native_sys.htbeta_transpose_matvec.is_none());
        for row in &native_sys.rows {
            assert_eq!(row.htbeta.ncols(), border_dim);
        }

        let full_sys = full_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let mut projected_sys = factored_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        projected_sys.htbeta_matvec = None;
        projected_sys.htbeta_transpose_matvec = None;
        projected_sys.htbeta_dense_supplement = false;
        for row_idx in 0..n_obs {
            let htbeta_b = materialize_row_htbeta_for_test(&full_sys, row_idx);
            projected_sys.rows[row_idx].htbeta =
                project_row_htbeta_to_factored_for_test(&factored_term, htbeta_b.view());
        }
        projected_sys.refresh_row_hessian_fingerprint();

        let ridge_t = 5.0e-1;
        let (native_dt, native_db, _) = native_sys.solve(ridge_t, 1.0e-8).unwrap();
        let (projected_dt, projected_db, _) = projected_sys.solve(ridge_t, 1.0e-8).unwrap();

        assert_eq!(native_dt.len(), projected_dt.len());
        assert_eq!(native_db.len(), projected_db.len());
        for idx in 0..native_dt.len() {
            assert_abs_diff_eq!(native_dt[idx], projected_dt[idx], epsilon = 1.0e-10);
        }
        for idx in 0..native_db.len() {
            assert_abs_diff_eq!(native_db[idx], projected_db[idx], epsilon = 1.0e-10);
        }
    }

    #[test]
    fn factored_row_htbeta_d2_matches_dense_full_b_then_project() {
        let k_atoms = 3usize;
        let m = 5usize;
        let p = 32usize;
        let frame_rank = 2usize;
        let latent_dim = 2usize;
        let n_obs = 6usize;
        let mut factored_term =
            low_rank_factored_htbeta_term(k_atoms, m, p, frame_rank, latent_dim, n_obs);
        assert!(factored_term.frames_active());
        assert_eq!(
            factored_term.factored_border_dim(),
            k_atoms * m * frame_rank
        );
        assert!(factored_term.factored_border_dim() < factored_term.beta_dim());

        let mut full_term = factored_term.clone();
        for atom in &mut full_term.atoms {
            atom.deactivate_decoder_frame();
        }
        let rho = factored_htbeta_rho(k_atoms, latent_dim);
        let target = Array2::<f64>::from_shape_fn((n_obs, p), |(row, col)| {
            0.01 * (row + 1) as f64 - 0.002 * (col + 1) as f64
        });

        let native_sys = factored_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let full_sys = full_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        let mut projected_sys = factored_term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();
        projected_sys.htbeta_matvec = None;
        projected_sys.htbeta_transpose_matvec = None;
        projected_sys.htbeta_dense_supplement = false;
        for row_idx in 0..n_obs {
            let htbeta_b = materialize_row_htbeta_for_test(&full_sys, row_idx);
            projected_sys.rows[row_idx].htbeta =
                project_row_htbeta_to_factored_for_test(&factored_term, htbeta_b.view());
        }
        projected_sys.refresh_row_hessian_fingerprint();

        let ridge_t = 5.0e-1;
        let (native_dt, native_db, _) = native_sys.solve(ridge_t, 1.0e-8).unwrap();
        let (projected_dt, projected_db, _) = projected_sys.solve(ridge_t, 1.0e-8).unwrap();
        assert_eq!(native_dt.len(), projected_dt.len());
        assert_eq!(native_db.len(), projected_db.len());
        for idx in 0..native_dt.len() {
            assert_abs_diff_eq!(native_dt[idx], projected_dt[idx], epsilon = 1.0e-10);
        }
        for idx in 0..native_db.len() {
            assert_abs_diff_eq!(native_db[idx], projected_db[idx], epsilon = 1.0e-10);
        }
    }

    #[test]
    fn qwen_shape_d2_factored_htbeta_assembly_stays_below_8gib() {
        const K_ATOMS: usize = 8;
        const M: usize = 10;
        const P: usize = 2048;
        const FRAME_RANK: usize = 2;
        const LATENT_DIM: usize = 2;
        const N_OBS: usize = 2000;
        const EIGHT_GIB: usize = 8 * 1024 * 1024 * 1024;

        let mut term = low_rank_factored_htbeta_term(K_ATOMS, M, P, FRAME_RANK, LATENT_DIM, N_OBS);
        assert!(term.frames_active());
        assert_eq!(term.beta_dim(), K_ATOMS * M * P);
        assert_eq!(term.factored_border_dim(), K_ATOMS * M * FRAME_RANK);
        assert!(term.factored_border_dim() < term.beta_dim());

        let rho = factored_htbeta_rho(K_ATOMS, LATENT_DIM);
        let target = Array2::<f64>::from_shape_fn((N_OBS, P), |(row, col)| {
            1.0e-4 * ((row + 1) as f64 * (col + 3) as f64).sin()
        });
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap();

        assert_eq!(sys.k, term.factored_border_dim());
        assert!(sys.htbeta_matvec.is_none());
        assert!(sys.htbeta_transpose_matvec.is_none());
        let actual_row_dim = sys.row_dims[0];
        assert!(actual_row_dim > 0);
        assert!(sys.row_dims.iter().all(|&dim| dim == actual_row_dim));
        for row in &sys.rows {
            assert_eq!(row.htbeta.ncols(), term.factored_border_dim());
            assert_eq!(row.htbeta.nrows(), actual_row_dim);
        }

        let htbeta_bytes: usize = sys
            .rows
            .iter()
            .map(|row| row.htbeta.len() * std::mem::size_of::<f64>())
            .sum();
        let assembled_dense_bytes = htbeta_bytes
            + sys.hbb.len() * std::mem::size_of::<f64>()
            + sys.gb.len() * std::mem::size_of::<f64>();
        let old_full_b_htbeta_bytes = N_OBS
            .saturating_mul(actual_row_dim)
            .saturating_mul(term.beta_dim())
            .saturating_mul(std::mem::size_of::<f64>());

        assert!(
            old_full_b_htbeta_bytes > EIGHT_GIB,
            "test shape must reproduce the old p-wide H_tbeta memory wall"
        );
        assert!(
            assembled_dense_bytes < EIGHT_GIB,
            "qwen-shaped factored assembly stored {assembled_dense_bytes} bytes, \
             exceeding the 8 GiB gate"
        );
    }

    /// A full-rank small-`p` decoder must NOT activate a frame: the factored
    /// border equals the full `M_k·p`, the Grassmann evidence dimension is `0`,
    /// and the Occam normalizer is bit-for-bit the historical
    /// `½·p·rank(S)·log λ` — the small-`p` evidence-equality contract.
    #[test]
    fn factored_evidence_matches_full_b_at_small_p() {
        let m = 5usize;
        let p = 2usize;
        // Full-rank decoder (rank 2 == p): no border saving, frame must stay off.
        let mut decoder = Array2::<f64>::zeros((m, p));
        for mu in 0..m {
            decoder[[mu, 0]] = 1.0 + mu as f64;
            decoder[[mu, 1]] = (mu as f64) - 2.0;
        }
        let mut phi = Array2::<f64>::zeros((m, m));
        let mut jet = Array3::<f64>::zeros((m, m, 1));
        for mu in 0..m {
            phi[[mu, mu]] = 1.0;
            jet[[mu, mu, 0]] = 1.0;
        }
        let s_raw = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        let mut atom = SaeManifoldAtom::new(
            "fullrank",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            s_raw,
        )
        .unwrap();
        let activated = atom.maybe_activate_decoder_frame().expect("activate");
        assert_eq!(
            activated, None,
            "full-rank small-p must stay on full-B path"
        );
        assert!(atom.decoder_frame.is_none());
        assert_eq!(atom.border_frame_rank(), p);
        assert_eq!(atom.frame_manifold_dimension(), 0);

        let mut term = SaeManifoldTerm::new(
            vec![atom],
            SaeAssignment::from_blocks_with_mode(
                Array2::<f64>::zeros((m, 1)),
                vec![Array2::<f64>::zeros((m, 1))],
                AssignmentMode::softmax(0.7),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(!term.frames_active());
        assert_eq!(term.factored_border_dim(), term.beta_dim());
        assert_eq!(term.grassmann_evidence_dimension(), 0);
        let activated_n = term.auto_activate_decoder_frames().expect("auto");
        assert_eq!(activated_n, 0, "small-p auto-activation must be a no-op");

        // Occam normalizer equals the historical ½·p·rank(S)·log λ exactly.
        let rho = SaeManifoldRho::new(0.0, 0.37, vec![array![0.0_f64]]);
        let occam = term.reml_occam_term(&rho).expect("occam");
        let rank_s = SaeManifoldTerm::symmetric_rank(&term.atoms[0].smooth_penalty).unwrap();
        let expected = 0.5 * (p as f64) * (rank_s as f64) * rho.log_lambda_smooth;
        assert_abs_diff_eq!(occam, expected, epsilon = 1.0e-12);
    }

    /// Streaming polar refresh from an accumulated cross-moment re-orients the
    /// frame toward the cross-moment span and keeps `B_k`'s in-span component
    /// while staying column-orthonormal (the closed-form streaming step).
    #[test]
    fn streaming_polar_refresh_reorients_frame() {
        let m = 4usize;
        let p = 12usize;
        let r = 2usize;
        let mut frame0 = Array2::<f64>::zeros((p, r));
        frame0[[0, 0]] = 1.0;
        frame0[[1, 1]] = 1.0;
        let mut c0 = Array2::<f64>::zeros((m, r));
        for mu in 0..m {
            c0[[mu, 0]] = 1.0 + mu as f64;
            c0[[mu, 1]] = 0.5 - mu as f64;
        }
        let decoder = fast_abt(&c0, &frame0);
        let mut phi = Array2::<f64>::zeros((m, m));
        let mut jet = Array3::<f64>::zeros((m, m, 1));
        for mu in 0..m {
            phi[[mu, mu]] = 1.0;
            jet[[mu, mu, 0]] = 1.0;
        }
        let s_raw = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        let mut atom = SaeManifoldAtom::new(
            "stream",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            s_raw,
        )
        .unwrap();
        atom.maybe_activate_decoder_frame().expect("activate");
        // New cross-moment pointing at axes {2,3}: refreshed frame must span them.
        let mut cross = Array2::<f64>::zeros((p, r));
        cross[[2, 0]] = 3.0;
        cross[[3, 1]] = 2.0;
        atom.refresh_frame_from_cross_moment(cross.view())
            .expect("refresh");
        let frame = atom.decoder_frame.as_ref().expect("frame");
        // Frame stays orthonormal.
        let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
        for i in 0..r {
            for j in 0..r {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
            }
        }
        // Refreshed span aligns with the cross-moment axes {2,3} (angle ~0).
        let mut target_span = Array2::<f64>::zeros((p, r));
        target_span[[2, 0]] = 1.0;
        target_span[[3, 1]] = 1.0;
        let angle = frame
            .max_principal_angle(target_span.view())
            .expect("angle");
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);
    }

    #[test]
    fn small_p_zero_decoder_stays_full_b() {
        let m = 3usize;
        let p = 8usize;
        let mut phi = Array2::<f64>::zeros((m, m));
        let mut jet = Array3::<f64>::zeros((m, m, 1));
        for row in 0..m {
            phi[[row, row]] = 1.0;
            jet[[row, row, 0]] = 1.0;
        }
        let smooth_penalty = crate::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
        let mut atom = SaeManifoldAtom::new(
            "small-p-zero",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            Array2::<f64>::zeros((m, p)),
            smooth_penalty,
        )
        .unwrap();

        assert_eq!(atom.decoder_frame_activation_rank().unwrap(), None);
        assert_eq!(atom.maybe_activate_decoder_frame().unwrap(), None);
        assert_eq!(atom.border_frame_rank(), p);
    }

    fn gamma_fd_tiny_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let n = 10usize;
        let p = 3usize;
        let k_atoms = 2usize;
        let m = 3usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
        let mut logits = Array2::<f64>::zeros((n, k_atoms));
        let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
        let weights = [
            [
                [0.10, -0.05, 0.03],
                [0.35, -0.20, 0.12],
                [-0.16, 0.18, 0.08],
            ],
            [
                [-0.08, 0.04, 0.06],
                [0.22, 0.10, -0.18],
                [0.11, -0.24, 0.15],
            ],
        ];
        let mut target = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let phase = (row as f64 + 0.35) / n as f64;
            coords[0][[row, 0]] = phase;
            coords[1][[row, 0]] = (phase + 0.21).fract();
            logits[[row, 0]] = if row % 2 == 0 { 0.8 } else { -0.6 };
            let assignments = softmax_row(logits.row(row), 0.9);
            for atom in 0..k_atoms {
                let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
                let basis = [1.0, theta.sin(), theta.cos()];
                for out_col in 0..p {
                    for basis_col in 0..m {
                        target[[row, out_col]] += assignments[atom]
                            * basis[basis_col]
                            * weights[atom][basis_col][out_col];
                    }
                }
            }
        }
        let mut atoms = Vec::with_capacity(k_atoms);
        for atom in 0..k_atoms {
            let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
            let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
                weights[atom][basis_col][out_col]
            });
            atoms.push(
                SaeManifoldAtom::new(
                    format!("gamma_{atom}"),
                    SaeAtomBasisKind::Periodic,
                    1,
                    phi,
                    jet,
                    decoder,
                    Array2::<f64>::eye(m),
                )
                .unwrap()
                .with_basis_second_jet(evaluator.clone()),
            );
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
            AssignmentMode::softmax(0.9),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(
            -6.0,
            -6.0,
            vec![Array1::from_vec(vec![-6.0]), Array1::from_vec(vec![-6.0])],
        );
        (term, target, rho)
    }

    fn fixed_state_logdet(
        mut term: SaeManifoldTerm,
        target: &Array2<f64>,
        rho: &SaeManifoldRho,
    ) -> f64 {
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
            .expect("fixed-state cache");
        let (tt, beta) = cache.arrow_log_det();
        tt + beta.expect("dense Schur logdet")
    }

    #[test]
    fn sae_logdet_theta_adjoint_matches_dense_fd_on_tiny_fixture() {
        let (mut term, target, rho) = gamma_fd_tiny_fixture();
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged cache");
        let solver = DeflatedArrowSolver::plain(&cache);
        let gamma = term
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("Gamma");
        let h = 1.0e-5;
        let probes = [
            (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
            (3usize, 1usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        ];
        for (row, local_pos, var) in probes {
            let mut plus = term.clone();
            let mut minus = term.clone();
            match var {
                SaeLocalRowVar::Logit { atom } => {
                    plus.assignment.logits[[row, atom]] += h;
                    minus.assignment.logits[[row, atom]] -= h;
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                    let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                    let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                    flat_p[idx] += h;
                    flat_m[idx] -= h;
                    plus.assignment.coords[atom].set_flat(flat_p.view());
                    minus.assignment.coords[atom].set_flat(flat_m.view());
                }
            }
            let fd = (fixed_state_logdet(plus, &target, &rho)
                - fixed_state_logdet(minus, &target, &rho))
                / (2.0 * h);
            let analytic = gamma.t[cache.row_offsets[row] + local_pos];
            let tol = 2.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
            assert!(
                (fd - analytic).abs() <= tol,
                "Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
            );
        }
    }

    #[test]
    fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map() {
        // The #1006 empirical-π third channel: under IBP-MAP, pi_k(M_k) couples
        // every row of column k, so perturbing one logit shifts EVERY row's
        // assembled `htt` diagonal in that column. `fixed_state_logdet` rebuilds
        // H at the perturbed state, so a single-logit FD captures both the
        // row-local direct-z channel and the global cross-row M_k channel that
        // `logdet_theta_adjoint` accumulates column-wise. lambda_sparse is the
        // active prior weight (fixed alpha), so the channel is genuinely live.
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
        rho.log_lambda_sparse = -1.0;
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged cache");
        let solver = DeflatedArrowSolver::plain(&cache);
        let gamma = term
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("Gamma");
        let h = 1.0e-5;
        // Probe both atoms across distinct rows so the cross-row coupling
        // (different rows sharing a column) is exercised on both columns.
        let probes = [
            (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
            (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
            (7usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        ];
        for (row, local_pos, var) in probes {
            let mut plus = term.clone();
            let mut minus = term.clone();
            match var {
                SaeLocalRowVar::Logit { atom } => {
                    plus.assignment.logits[[row, atom]] += h;
                    minus.assignment.logits[[row, atom]] -= h;
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                    let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                    let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                    flat_p[idx] += h;
                    flat_m[idx] -= h;
                    plus.assignment.coords[atom].set_flat(flat_p.view());
                    minus.assignment.coords[atom].set_flat(flat_m.view());
                }
            }
            let fd = (fixed_state_logdet(plus, &target, &rho)
                - fixed_state_logdet(minus, &target, &rho))
                / (2.0 * h);
            let analytic = gamma.t[cache.row_offsets[row] + local_pos];
            let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
            assert!(
                (fd - analytic).abs() <= tol,
                "IBP Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
            );
        }
    }

    /// #932 follow-up (the issue-comment cache-seam ask): the SAE row
    /// jet-program oracle driven directly from a CONVERGED production
    /// `ArrowFactorCache`, not a mirrored test layout.
    ///
    /// For every row of the converged tiny fixture, the production
    /// `row_jets_for_logdet` channels — the exact `first`/`second` tensors the
    /// #1006 `logdet_theta_adjoint` contracts — are rebuilt as a
    /// [`SaeReconstructionRowProgram`] from the SAME production inputs (the
    /// term's basis value/jacobian tensors, `atom_second_jets`, decoder
    /// blocks, gate logits/assignments, and the cache's own
    /// `row_vars_for_cache_row` primary layout) and compared column by
    /// column. The hand path sums sparse cross terms per (logit, coord)
    /// variable pair; the tower derives them by Leibniz from one expression —
    /// independent arithmetic, so agreement is a correctness proof of the
    /// production packing on a real converged state. The `weighted` arm
    /// exercises the #977 `set_row_loss_weights` √w seam, which scales every
    /// production channel by `sqrt(w_row)`.
    #[test]
    fn sae_row_jet_program_matches_production_row_jets_on_converged_cache() {
        use crate::terms::sae_row_jet_program::{
            AtomRowBasisJet, RowGate, SaeReconstructionRowProgram,
        };

        // Tiny-fixture row arity: softmax gauges the last logit as the fixed
        // reference (assignment_coord_dim = k_atoms − 1 = 1 free logit), plus
        // 2 atoms × 1 latent coord.
        const K: usize = 3;
        for weighted in [false, true] {
            let (mut term, target, rho) = gamma_fd_tiny_fixture();
            if weighted {
                let weights: Vec<f64> = (0..term.n_obs())
                    .map(|row| 0.5 + 0.17 * row as f64)
                    .collect();
                term.set_row_loss_weights(weights)
                    .expect("set row loss weights");
            }
            let (_value, _loss, cache) = term
                .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
                .expect("converged cache");
            let second_jets = term.atom_second_jets().expect("second jets");
            let border = term
                .border_channels_for_cache(&cache)
                .expect("border channels");
            let AssignmentMode::Softmax { temperature, .. } = term.assignment.mode else {
                panic!("gamma fixture is softmax-gated");
            };
            let inv_tau = 1.0 / temperature;
            let p = term.output_dim();
            let k_atoms = term.k_atoms();

            for row in 0..term.n_obs() {
                let vars = term.row_vars_for_cache_row(row, &cache).expect("row vars");
                assert_eq!(
                    vars.len(),
                    K,
                    "tiny fixture rows carry 1 free softmax logit + 2 coords"
                );
                let assignments = term
                    .assignment
                    .try_assignments_row(row)
                    .expect("assignments row");
                let jets = term
                    .row_jets_for_logdet(
                        row,
                        vars.clone(),
                        assignments.view(),
                        &second_jets,
                        &border,
                    )
                    .expect("production row jets");

                // Primary layout exactly as the cache rows it: slot positions
                // come from the production `row_vars_for_cache_row`, not a
                // re-derived convention.
                let mut logit_slot = vec![None; k_atoms];
                let mut coord_slot: Vec<Vec<usize>> = term
                    .atoms
                    .iter()
                    .map(|atom| vec![usize::MAX; atom.latent_dim])
                    .collect();
                for (pos, var) in vars.iter().enumerate() {
                    match *var {
                        SaeLocalRowVar::Logit { atom } => logit_slot[atom] = Some(pos),
                        SaeLocalRowVar::Coord { atom, axis } => coord_slot[atom][axis] = pos,
                    }
                }

                // Per-atom basis jets straight from the production tensors the
                // hand path consumes: basis_values / basis_jacobian /
                // atom_second_jets / decoder_coefficients.
                let atoms: Vec<AtomRowBasisJet> = term
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(k, atom)| {
                        let m = atom.basis_size();
                        let d = atom.latent_dim;
                        AtomRowBasisJet {
                            phi: (0..m).map(|b| atom.basis_values[[row, b]]).collect(),
                            d_phi: (0..m)
                                .map(|b| {
                                    (0..d)
                                        .map(|axis| atom.basis_jacobian[[row, b, axis]])
                                        .collect()
                                })
                                .collect(),
                            d2_phi: (0..m)
                                .map(|b| {
                                    (0..d)
                                        .map(|aa| {
                                            (0..d)
                                                .map(|bb| second_jets[k][[row, b, aa, bb]])
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect(),
                            decoder: (0..m)
                                .map(|b| {
                                    (0..p).map(|c| atom.decoder_coefficients[[b, c]]).collect()
                                })
                                .collect(),
                            latent_dim: d,
                        }
                    })
                    .collect();

                let prog = SaeReconstructionRowProgram {
                    atoms,
                    gate_value: assignments.to_vec(),
                    logits: term.assignment.logits.row(row).to_vec(),
                    gate_shift: vec![0.0; k_atoms],
                    gate: RowGate::Softmax { inv_tau },
                    logit_slot,
                    coord_slot,
                    n_primaries: K,
                };
                // The production channels carry the √w row-loss weight (#977
                // single seam); the program is the unweighted reconstruction.
                let sqrt_row_w = term
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |w| w[row].sqrt());
                if weighted {
                    assert!(
                        (sqrt_row_w - 1.0).abs() > 1e-6,
                        "weighted arm must exercise a non-unit √w (row {row}, √w={sqrt_row_w})"
                    );
                }

                for out_col in 0..p {
                    let tower = prog.reconstruction_column::<K>(out_col);
                    let g_floor = (0..K)
                        .map(|a| jets.first[a][out_col].abs())
                        .fold(1e-12_f64, f64::max);
                    let h_floor = (0..K)
                        .flat_map(|a| (0..K).map(move |b| (a, b)))
                        .map(|(a, b)| jets.second[a][b][out_col].abs())
                        .fold(1e-12_f64, f64::max);
                    for a in 0..K {
                        let want = sqrt_row_w * tower.g[a];
                        assert!(
                            (jets.first[a][out_col] - want).abs() <= 1e-9 * g_floor,
                            "weighted={weighted} row {row} col {out_col} first[{a}]: \
                             production {} vs tower {}",
                            jets.first[a][out_col],
                            want
                        );
                        for b in 0..K {
                            let want2 = sqrt_row_w * tower.h[a][b];
                            assert!(
                                (jets.second[a][b][out_col] - want2).abs() <= 1e-9 * h_floor,
                                "weighted={weighted} row {row} col {out_col} \
                                 second[{a}][{b}]: production {} vs tower {}",
                                jets.second[a][b][out_col],
                                want2
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ibp_map_outer_objective_advertises_analytic_gradient() {
        // The IBP-MAP empirical-π third channel (including the cross-row M_k
        // coupling) is now assembled exactly in `logdet_theta_adjoint` (#1006),
        // so the outer objective advertises an analytic gradient like every
        // other assignment mode.
        let (mut term, target, rho) = gamma_fd_tiny_fixture();
        term.assignment.mode = AssignmentMode::ibp_map(0.9, 1.0, false);

        let obj = SaeManifoldOuterObjective::new(term, target, None, rho, 5, 0.4, 1.0e-6, 1.0e-6);
        assert_eq!(obj.capability().gradient, Derivative::Analytic);
    }
}

/// PCA-based seed for SAE atom latent coordinates. Centers `z`, takes its SVD,
/// and projects onto leading principal components to initialize each atom's
/// chart according to its [`SaeAtomBasisKind`]: periodic atoms read a `[0, 1)`
/// phase off the top-2 PCs (remaining axes min-max normalized to
/// `[-0.5, 0.5]`), sphere atoms read `(lat, lon)` off the unit-normalized top-3
/// PCs, torus axes read a `[0, 1)` phase off disjoint PC pairs, and
/// Euclidean/other atoms take score-scaled, min-max-normalized PC projections.
/// Returns a padded
/// `(K_atoms, n_obs, d_max)` coordinate array.
pub fn sae_pca_seed_initial_coords(
    z: ArrayView2<'_, f64>,
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
) -> Result<Array3<f64>, String> {
    let k_atoms = basis_kinds.len();
    let (n_obs, _p_out) = z.dim();
    let d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, n_obs, d_max));
    if n_obs == 0 || z.ncols() == 0 {
        return Ok(out);
    }
    // Reject non-finite input up front so a clean error surfaces here rather
    // than a silent non-finite seed (or an opaque SVD failure) downstream.
    for ((row, col), &value) in z.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_pca_seed: Z must be finite; Z[{row}, {col}] = {value}"
            ));
        }
    }
    // Accumulate the column mean with Welford's running update
    // `mean += (x − mean) / count` instead of a plain running sum. The plain
    // sum overflows to `±inf` for huge finite columns (e.g. two rows of
    // `1e308` sum to `2e308 = inf`), which poisons the centered matrix and the
    // SVD. Welford's update keeps the accumulator bounded by the column's data
    // range, so the mean is finite whenever the inputs are.
    let mut col_means = Array1::<f64>::zeros(z.ncols());
    for col in 0..z.ncols() {
        let mut mean = 0.0_f64;
        for (count, row) in (0..n_obs).enumerate() {
            let x = z[[row, col]];
            mean += (x - mean) / (count as f64 + 1.0);
        }
        col_means[col] = mean;
    }
    let mut centered = z.to_owned();
    for row in 0..n_obs {
        for col in 0..z.ncols() {
            centered[[row, col]] -= col_means[col];
        }
    }
    // Centering can still overflow if the data span itself is non-finite
    // (e.g. `+1e308` and `−1e308` in one column give a finite mean but an
    // `inf` deviation). Surface that as a clean error rather than feeding a
    // non-finite matrix to the SVD.
    for ((row, col), &value) in centered.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_pca_seed: centered Z is non-finite at [{row}, {col}] \
                 (data span exceeds f64 range); rescale Z before seeding"
            ));
        }
    }
    let (u_opt, s_vals, vt_opt) = centered
        .svd(true, true)
        .map_err(|err| format!("sae_pca_seed: SVD failed: {err:?}"))?;
    let u = u_opt.ok_or_else(|| "sae_pca_seed: SVD returned no U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "sae_pca_seed: SVD returned no Vt".to_string())?;
    let vt_rows = vt.nrows();
    let u_cols = u.ncols();
    let two_pi = std::f64::consts::TAU;
    for atom_idx in 0..k_atoms {
        let d = atom_dim[atom_idx];
        if d == 0 {
            continue;
        }
        match &basis_kinds[atom_idx] {
            SaeAtomBasisKind::Periodic => {
                if vt_rows >= 2 {
                    // Diversify the per-atom circle seed (issue #671). The
                    // previous scheme shared PC0 as the first phase axis for
                    // *every* atom, so all periodic atoms read off nearly the
                    // same phase coordinate, producing near-duplicate basis
                    // designs and a severely ill-conditioned joint decoder LSQ
                    // seed. Give each atom a disjoint pair of principal
                    // components `(PC_{2k}, PC_{2k+1})` when the spectrum is
                    // wide enough, wrapping around only when atoms outnumber the
                    // available PC pairs. This keeps distinct atoms' seed
                    // coordinates decorrelated so the decoder seed stays
                    // well-conditioned and the cross-atom Gram starts small.
                    let pc_pairs = vt_rows / 2;
                    let (pc1_row, pc2_row) = if pc_pairs >= 1 {
                        let pair = if pc_pairs > 0 { atom_idx % pc_pairs } else { 0 };
                        (2 * pair, 2 * pair + 1)
                    } else {
                        (0, 1)
                    };
                    let pc1 = vt.row(pc1_row.min(vt_rows - 1));
                    let pc2 = vt.row(pc2_row.min(vt_rows - 1));
                    for row in 0..n_obs {
                        let mut a = 0.0_f64;
                        let mut b = 0.0_f64;
                        for col in 0..centered.ncols() {
                            a += centered[[row, col]] * pc1[col];
                            b += centered[[row, col]] * pc2[col];
                        }
                        let phase = b.atan2(a) / two_pi;
                        out[[atom_idx, row, 0]] = phase - phase.floor();
                    }
                }
                for axis in 1..d {
                    if axis >= vt_rows {
                        break;
                    }
                    let pc = vt.row(axis);
                    let mut proj = Array1::<f64>::zeros(n_obs);
                    for row in 0..n_obs {
                        let mut acc = 0.0_f64;
                        for col in 0..centered.ncols() {
                            acc += centered[[row, col]] * pc[col];
                        }
                        proj[row] = acc;
                    }
                    let (min_v, max_v) = proj
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let span = max_v - min_v;
                    if span > 0.0 {
                        for row in 0..n_obs {
                            out[[atom_idx, row, axis]] = (proj[row] - min_v) / span - 0.5;
                        }
                    }
                }
            }
            SaeAtomBasisKind::Sphere => {
                // Seed the sphere chart from the top-3 PCs: drop the centred
                // response onto (pc0, pc1, pc2), unit-normalise, and read off
                // (lat, lon). This places every row on the chart with
                // `lat ∈ (-π/2, π/2)` and `lon ∈ (-π, π]`.
                let n_pc = vt_rows.min(3);
                if n_pc == 0 {
                    continue;
                }
                let pcs: Vec<_> = (0..n_pc).map(|i| vt.row(i)).collect();
                for row in 0..n_obs {
                    let mut amb = [0.0_f64; 3];
                    for (i, pc) in pcs.iter().enumerate() {
                        let mut acc = 0.0_f64;
                        for col in 0..centered.ncols() {
                            acc += centered[[row, col]] * pc[col];
                        }
                        amb[i] = acc;
                    }
                    let norm = (amb[0] * amb[0] + amb[1] * amb[1] + amb[2] * amb[2]).sqrt();
                    let (x, y, z) = if norm > 0.0 {
                        (amb[0] / norm, amb[1] / norm, amb[2] / norm)
                    } else {
                        (1.0, 0.0, 0.0)
                    };
                    let lat = z.clamp(-1.0, 1.0).asin();
                    let lon = y.atan2(x);
                    if d >= 1 {
                        out[[atom_idx, row, 0]] = lat;
                    }
                    if d >= 2 {
                        out[[atom_idx, row, 1]] = lon;
                    }
                }
            }
            SaeAtomBasisKind::Torus => {
                // Seed each torus axis from a disjoint pair of PCs: axis `a`
                // uses (pc_{2a}, pc_{2a+1}) projected onto the centred
                // response and read off as `atan2`, normalised to `[0, 1)`.
                for axis in 0..d {
                    let pc_a_idx = 2 * axis;
                    let pc_b_idx = 2 * axis + 1;
                    if pc_b_idx >= vt_rows {
                        break;
                    }
                    let pc_a = vt.row(pc_a_idx);
                    let pc_b = vt.row(pc_b_idx);
                    for row in 0..n_obs {
                        let mut a = 0.0_f64;
                        let mut b = 0.0_f64;
                        for col in 0..centered.ncols() {
                            a += centered[[row, col]] * pc_a[col];
                            b += centered[[row, col]] * pc_b[col];
                        }
                        // atan2 ∈ (-π, π]; map to phase ∈ [0, 1).
                        let phase = b.atan2(a) / two_pi;
                        let wrapped = phase - phase.floor();
                        out[[atom_idx, row, axis]] = wrapped;
                    }
                }
            }
            _ => {
                let k_cols = d.min(u_cols).min(s_vals.len());
                let mut tmp = Array2::<f64>::zeros((n_obs, d));
                for col in 0..k_cols {
                    let s_col = s_vals[col];
                    for row in 0..n_obs {
                        tmp[[row, col]] = u[[row, col]] * s_col;
                    }
                }
                for col in 0..d {
                    let mut min_v = f64::INFINITY;
                    let mut max_v = f64::NEG_INFINITY;
                    for row in 0..n_obs {
                        let v = tmp[[row, col]];
                        if v < min_v {
                            min_v = v;
                        }
                        if v > max_v {
                            max_v = v;
                        }
                    }
                    let span = max_v - min_v;
                    if span > 0.0 {
                        for row in 0..n_obs {
                            out[[atom_idx, row, col]] = (tmp[[row, col]] - min_v) / span - 0.5;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod inner_contract_probe_tests {
    use super::*;
    use crate::terms::{AssignmentMode, LatentManifold, SaeAssignment};
    use std::sync::Arc;

    fn euclidean_line_contract_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let n = 150usize;
        let p = 8usize;
        let mut coords = Array2::<f64>::zeros((n, 1));
        let mut z = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let u = -1.0 + 2.0 * row as f64 / (n as f64 - 1.0);
            coords[[row, 0]] = 2.5 + 3.0 * u;
            for col in 0..p {
                let linear_loading = 0.35 + 0.07 * col as f64;
                let offset = 0.08 * ((col % 3) as f64 - 1.0);
                let phase = (row * (col + 3)) as f64;
                let noise = 0.04 * (phase.sin() + 0.5 * (0.37 * phase).cos());
                z[[row, col]] = offset + linear_loading * u + noise;
            }
        }

        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("evaluator"));
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis");
        let m = phi.ncols();
        let smooth_penalty =
            crate::basis::create_difference_penalty_matrix(m, 2, None).expect("penalty");
        let atom = SaeManifoldAtom::new(
            "contract-line",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            Array2::<f64>::zeros((m, p)),
            smooth_penalty,
        )
        .expect("atom")
        .with_basis_second_jet(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment");
        let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
        let rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1)]);
        (term, z, rho)
    }

    fn assert_contract_close(label: &str, analytic: f64, finite_difference: f64) {
        let rel = (analytic - finite_difference).abs()
            / finite_difference.abs().max(analytic.abs()).max(1.0e-12);
        assert!(
            rel < 1.0e-5,
            "{label}: analytic={analytic:.12e} fd={finite_difference:.12e} rel={rel:.3e}"
        );
    }

    #[test]
    fn euclidean_line_decoder_gradient_matches_penalized_objective_fd() {
        let (mut term, z, mut rho) = euclidean_line_contract_fixture();
        let ridge = 1.0e-6;
        for step in 0..6 {
            let loss = term
                .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
                .unwrap_or_else(|err| panic!("warm step {step} failed: {err}"));
            assert!(
                loss.total().is_finite(),
                "warm step {step} loss is non-finite"
            );
        }

        let sys_coord = term
            .assemble_arrow_schur(z.view(), &rho, None)
            .expect("coord assemble");
        assert_eq!(
            sys_coord.k,
            term.beta_dim(),
            "p=8 contract fixture must stay on full-B coordinates"
        );
        assert!(
            !term.frames_active(),
            "p=8 contract fixture must not activate a frame"
        );

        let h = 1.0e-6;
        for row in [3usize, 75, 140] {
            let analytic = sys_coord.rows[row].gt[0];
            let base_coord = term.assignment.coords[0].as_matrix()[[row, 0]];

            let mut plus_coords = term.assignment.coords[0].as_matrix();
            plus_coords[[row, 0]] = base_coord + h;
            let plus_flat = Array1::from_iter(plus_coords.iter().copied());
            term.assignment.coords[0].set_flat(plus_flat.view());
            term.refresh_basis_from_current_coords()
                .expect("plus refresh");
            let f_plus = term
                .penalized_objective_total(z.view(), &rho, None, 1.0)
                .expect("coord f+");

            let mut minus_coords = term.assignment.coords[0].as_matrix();
            minus_coords[[row, 0]] = base_coord - h;
            let minus_flat = Array1::from_iter(minus_coords.iter().copied());
            term.assignment.coords[0].set_flat(minus_flat.view());
            term.refresh_basis_from_current_coords()
                .expect("minus refresh");
            let f_minus = term
                .penalized_objective_total(z.view(), &rho, None, 1.0)
                .expect("coord f-");

            let mut restored_coords = term.assignment.coords[0].as_matrix();
            restored_coords[[row, 0]] = base_coord;
            let restored_flat = Array1::from_iter(restored_coords.iter().copied());
            term.assignment.coords[0].set_flat(restored_flat.view());
            term.refresh_basis_from_current_coords()
                .expect("restore refresh");

            let fd = (f_plus - f_minus) / (2.0 * h);
            assert_contract_close(&format!("CONTRACT coord row {row}"), analytic, fd);
        }

        let sys_decoder = term
            .assemble_arrow_schur(z.view(), &rho, None)
            .expect("decoder assemble");
        assert_eq!(sys_decoder.k, term.beta_dim());
        let p = term.output_dim();
        for (basis_col, out_col) in [(0usize, 0usize), (1, 3), (2, 7)] {
            let beta_idx = basis_col * p + out_col;
            let analytic = sys_decoder.gb[beta_idx];
            let base = term.atoms[0].decoder_coefficients[[basis_col, out_col]];

            term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base + h;
            let f_plus = term
                .penalized_objective_total(z.view(), &rho, None, 1.0)
                .expect("decoder f+");
            term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base - h;
            let f_minus = term
                .penalized_objective_total(z.view(), &rho, None, 1.0)
                .expect("decoder f-");
            term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base;

            let fd = (f_plus - f_minus) / (2.0 * h);
            assert_contract_close(
                &format!("CONTRACT decoder ({basis_col},{out_col})"),
                analytic,
                fd,
            );
        }
    }
}
