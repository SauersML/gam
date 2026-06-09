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

use ndarray::{
    Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s,
};
use std::sync::Arc;

use crate::solver::arrow_schur::{
    ArrowProximalCorrectionOptions, ArrowRowBlock, ArrowSchurError, ArrowSchurSystem,
    ArrowSolveOptions, BetaPenaltyOp, CompositePenaltyOp, DensePenaltyOp, KroneckerPenaltyOp,
    SparseBlockKroneckerPenaltyOp, SparseGBlock, StreamingArrowSchur,
    solve_arrow_newton_step_with_proximal_correction, solve_streaming_reduced_beta,
};
use crate::terms::analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    DecoderIncoherencePenalty, IBPAssignmentPenalty, IsometryPenalty, MechanismSparsityPenalty,
    NuclearNormPenalty, PenaltyTier, PsiSlice, SoftmaxAssignmentSparsityPenalty, WeightField,
    resolve_learnable_weight,
};
use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode, LatentManifold};

use crate::linalg::faer_ndarray::{FaerEigh, FaerSvd};
use crate::solver::arrow_schur::{ArrowFactorCache, solve_arrow_newton_step_with_options};
use crate::solver::estimate::EstimationError;
use crate::solver::evidence::arrow_log_det_from_cache;
use crate::solver::outer_strategy::{
    DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterCapability, OuterEval,
    OuterObjective, SeedOutcome,
};
use faer::Side;

const SAE_MANIFOLD_ARMIJO_C1: f64 = 1.0e-4;
const SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS: usize = 12;

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

/// Reactivation band width (in units of the JumpReLU temperature `τ`) below the
/// hard gate threshold. The forward gate value is hard-zero strictly below
/// `threshold`, but an atom whose logit lies within `threshold − MARGIN·τ` is
/// still admitted to the compact Newton active set for sparsity-prior support.
/// Below the band the shifted-sigmoid derivative `σ'((l−θ)/τ)` is vanishingly
/// small, so the band captures essentially all of the prior-gradient mass that
/// could act on a gated atom (at `MARGIN = 4`, `σ((l−θ)/τ) < σ(−4) ≈ 0.018` at
/// the band edge). Without the band the gate is an absorbing pruning rule, not a
/// learnable gate.
const JUMPRELU_REACTIVATION_MARGIN: f64 = 4.0;

/// Shared band predicate for JumpReLU optimization inclusion. An atom is kept
/// optimizable (compact-layout inclusion and prior-gradient support) when its
/// logit is above the reactivation band's lower edge `threshold − MARGIN·τ`.
/// This is strictly weaker than the hard forward gate `logit > threshold`,
/// which still governs data-fit reconstruction and its logit JVP.
#[inline]
fn jumprelu_in_optimization_band(logit: f64, threshold: f64, temperature: f64) -> bool {
    logit > threshold - JUMPRELU_REACTIVATION_MARGIN * temperature
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeStreamingPlan {
    pub streaming: bool,
    pub chunk_size: usize,
    pub estimated_full_batch_bytes: usize,
    pub in_core_budget_bytes: usize,
}

fn sae_streaming_plan_from_budget(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    in_core_budget_bytes: usize,
    chunk_window_bytes: usize,
) -> SaeStreamingPlan {
    const BYTES_PER_F64: usize = 8;
    const MIN_CHUNK_ROWS: usize = 256;
    let per_row_words = total_basis
        .saturating_mul(1 + d_max)
        .saturating_add(k_atoms)
        .max(1);
    let per_row_bytes = per_row_words.saturating_mul(BYTES_PER_F64);
    let full_batch_bytes = n_obs.saturating_mul(per_row_bytes);
    if full_batch_bytes <= in_core_budget_bytes {
        return SaeStreamingPlan {
            streaming: false,
            chunk_size: n_obs.max(1),
            estimated_full_batch_bytes: full_batch_bytes,
            in_core_budget_bytes,
        };
    }
    let rows_per_chunk = (chunk_window_bytes / per_row_bytes).max(MIN_CHUNK_ROWS);
    SaeStreamingPlan {
        streaming: true,
        chunk_size: rows_per_chunk.min(n_obs).max(1),
        estimated_full_batch_bytes: full_batch_bytes,
        in_core_budget_bytes,
    }
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

pub trait SaeBasisEvaluator: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String>;

    /// Object-safe forwarder to [`SaeBasisSecondJet::second_jet`] for callers
    /// holding `&dyn SaeBasisEvaluator` / `Arc<dyn SaeBasisEvaluator>`.
    ///
    /// Implementations return `Some(result)` only when an analytic second jet
    /// exists for this evaluator. Returning `None` is an explicit capability
    /// declaration, not a default sentinel hidden in the trait.
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>>;

    /// Object-safe forwarder to the basis third jet
    /// `T[n, m, a, c, e] = ∂³Φ_m / ∂t_a ∂t_c ∂t_e`, for callers holding
    /// `&dyn SaeBasisEvaluator` / `Arc<dyn SaeBasisSecondJet>`. The exact
    /// isometry Hessian (`IsometryPenalty::hvp`) needs the *decoder* third jet
    /// `K = Σ_m T[..,m,..]·B[m,:]` for its residual·curvature term; without it
    /// that exact Hessian silently drops the residual and collapses to
    /// Gauss-Newton (issue #458).
    ///
    /// Implementations return `Some(result)` only when an analytic third jet
    /// exists for this evaluator. Evaluators without one return `None`
    /// explicitly; there is no finite-difference fallback.
    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>>;
}

/// Bases that expose an analytic second jet
/// `H[n, m, a, c] = ∂²Phi_k[n, m] / (∂t_{n,a} ∂t_{n,c})`,
/// shape `(n_rows, n_basis, latent_dim, latent_dim)`.
///
/// Implemented only by evaluators with a closed-form Hessian (periodic
/// harmonic, sphere chart, torus). Callers that need an analytic
/// `∂J/∂t` require this bound; evaluators without it must use a
/// derivative-free fallback. Replaces the previous `Option<Array4<f64>>`
/// return on the base trait so the "no second jet" case is encoded by
/// trait absence rather than a sentinel `None`, and shape mismatches
/// surface as descriptive errors instead of silently collapsing to
/// `None`.
pub trait SaeBasisSecondJet: SaeBasisEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String>;
}

/// Bases that expose an analytic third jet
/// `T[n, m, a, c, e] = ∂³Φ_m[n] / (∂t_{n,a} ∂t_{n,c} ∂t_{n,e})`,
/// shape `(n_rows, n_basis, latent_dim, latent_dim, latent_dim)`.
///
/// The exact isometry Hessian (`IsometryPenalty::hvp`) needs the third decoder
/// jet `K = ∂³φ/∂t³ = Σ_m T[..,m,..] · B[m, :]` for its residual·curvature term
/// `B_{ab,cd} = K_{a,cd}ᵀ W J_b + H_{a,c}ᵀ W H_{b,d} + H_{a,d}ᵀ W H_{b,c}
/// + J_aᵀ W K_{b,cd}`. Bases that supply a closed-form `H` (the
/// [`SaeBasisSecondJet`] super-bound) but not `K` leave that exact Hessian
/// silently dropping the residual term; this trait closes that gap for every
/// analytic basis: the curved bases (sphere chart, periodic harmonic, torus
/// harmonic), the Euclidean monomial patch, the trivially-zero affine basis,
/// and the Duchon basis (radial third-derivative kernel block + monomial
/// nullspace block, both in closed form). The full third jet is symmetric in
/// its three trailing axes.
pub trait SaeBasisThirdJet: SaeBasisSecondJet {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String>;
}

/// Periodic harmonic basis evaluator for a single-dimensional circle latent.
///
/// Produces `M = 2*num_harmonics + 1` basis functions
/// `[1, sin(2π·1·t), cos(2π·1·t), …, sin(2π·H·t), cos(2π·H·t)]` where
/// `H = (M − 1) / 2`. The latent must have `latent_dim == 1`.
#[derive(Debug, Clone)]
pub struct PeriodicHarmonicEvaluator {
    pub num_basis: usize,
}

impl PeriodicHarmonicEvaluator {
    pub fn new(num_basis: usize) -> Result<Self, String> {
        if num_basis == 0 || num_basis % 2 == 0 {
            return Err(format!(
                "PeriodicHarmonicEvaluator requires odd num_basis >= 1; got {num_basis}"
            ));
        }
        Ok(Self { num_basis })
    }
}

impl SaeBasisEvaluator for PeriodicHarmonicEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = 1.0;
            for h in 1..=num_harmonics {
                let angle = two_pi * (h as f64) * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * h - 1;
                let c_idx = 2 * h;
                phi[[row, s_idx]] = s;
                phi[[row, c_idx]] = c;
                jet[[row, s_idx, 0]] = two_pi * (h as f64) * c;
                jet[[row, c_idx, 0]] = -two_pi * (h as f64) * s;
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for PeriodicHarmonicEvaluator {
    /// Second derivative of the 1D Fourier basis on the unit circle.
    ///
    /// For `Phi = [1, sin(2π h t), cos(2π h t), ...]` we have
    /// `Phi'' = [0, -(2π h)² sin(...), -(2π h)² cos(...), ...]`, i.e.
    /// the second derivative is `-(2π h)² · phi(t)` on each harmonic pair.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator::second_jet: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut h = Array4::<f64>::zeros((n, m, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            for k in 1..=num_harmonics {
                let freq = two_pi * (k as f64);
                let freq2 = freq * freq;
                let angle = freq * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * k - 1;
                let c_idx = 2 * k;
                h[[row, s_idx, 0, 0]] = -freq2 * s;
                h[[row, c_idx, 0, 0]] = -freq2 * c;
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for PeriodicHarmonicEvaluator {
    /// Third derivative of the 1-D Fourier basis on the unit circle.
    ///
    /// For `Phi = [1, sin(2π h t), cos(2π h t), …]` the chain of derivatives is
    /// `sin → ωc → −ω²s → −ω³c` and `cos → −ωs → −ω²c → ω³s`, so the third
    /// derivative is `[0, −(2π h)³ cos(…), +(2π h)³ sin(…), …]`.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let n = coords.nrows();
        let d = coords.ncols();
        if d != 1 {
            return Err(format!(
                "PeriodicHarmonicEvaluator::third_jet: expected latent_dim == 1, got {d}"
            ));
        }
        let m = self.num_basis;
        let num_harmonics = (m - 1) / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut t3 = Array5::<f64>::zeros((n, m, 1, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            for k in 1..=num_harmonics {
                let freq = two_pi * (k as f64);
                let freq3 = freq * freq * freq;
                let angle = freq * t;
                let s = angle.sin();
                let c = angle.cos();
                let s_idx = 2 * k - 1;
                let c_idx = 2 * k;
                t3[[row, s_idx, 0, 0, 0]] = -freq3 * c;
                t3[[row, c_idx, 0, 0, 0]] = freq3 * s;
            }
        }
        Ok(t3)
    }
}

/// Raw-angle periodic evaluator for the minimal SAE-manifold front-end.
///
/// The basis is exactly `[cos(t), sin(t)]` with `t` measured in radians. If
/// the latent coordinate has more than one axis, the first axis carries the
/// circle phase and the remaining axes are left available to the optimizer but
/// do not enter this basis.
#[derive(Debug, Clone)]
pub struct RawPeriodicCircleEvaluator {
    pub latent_dim: usize,
}

impl RawPeriodicCircleEvaluator {
    pub fn new(latent_dim: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("RawPeriodicCircleEvaluator requires latent_dim >= 1".to_string());
        }
        Ok(Self { latent_dim })
    }
}

impl SaeBasisEvaluator for RawPeriodicCircleEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        if coords.ncols() != self.latent_dim {
            return Some(Err(format!(
                "RawPeriodicCircleEvaluator::second_jet_dyn: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            )));
        }
        None
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        if coords.ncols() != self.latent_dim {
            return Some(Err(format!(
                "RawPeriodicCircleEvaluator::third_jet_dyn: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            )));
        }
        None
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "RawPeriodicCircleEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, self.latent_dim));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = t.cos();
            phi[[row, 1]] = t.sin();
            jet[[row, 0, 0]] = -t.sin();
            jet[[row, 1, 0]] = t.cos();
        }
        Ok((phi, jet))
    }
}

/// Diagonal of the chart-local seven-column sphere basis penalty.
///
/// The columns are `[1, x, y, z, xy, yz, xz]`; the constant column carries a
/// numerically-negligible ridge (`1e-8`) so the penalty stays positive
/// definite, the three linear columns are penalized at unit weight, and the
/// three bilinear columns at weight `4` (their second-order angular content).
/// This is the single source of truth for the chart penalty shared between the
/// core SAE path and the PyFFI `sphere_chart_basis_with_jet` helper.
pub const SPHERE_CHART_PENALTY_DIAGONAL: [f64; 7] = [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0];

/// Shared single source of truth for the chart-local sphere basis and its
/// analytic first-derivative (lat/lon) jet.
///
/// `coords` is an `(N, 2)` array of latitude/longitude pairs in radians. The
/// returned `phi` has shape `(N, 7)` with columns `[1, x, y, z, xy, yz, xz]`
/// for the unit-sphere embedding `x = cos(lat)cos(lon)`, `y = cos(lat)sin(lon)`,
/// `z = sin(lat)`; the returned `jet` has shape `(N, 7, 2)` with the last axis
/// indexing `[∂/∂lat, ∂/∂lon]`.
///
/// The map and its jet are everywhere `C^∞` in `(lat, lon)`: every column is a
/// polynomial in `cos`/`sin` of the two coordinates, and `cos`/`sin` are entire,
/// so the exact analytic derivatives `∂x/∂lat = -sin(lat)cos(lon)`, … are
/// globally smooth. Latitude is therefore **not** clamped and the latitude
/// derivatives are **not** gated here.
///
/// The physical `lat ∈ [-π/2, π/2]` box that pins a canonical latitude range is
/// enforced where it belongs — in the latent retraction / tangent projection
/// ([`crate::terms::latent_coord::LatentManifold::Interval`]), which clamps the
/// coordinate after each step and zeroes only the *outward-normal* component of
/// the tangent velocity at an active bound (a correct KKT projection). The old
/// binary `chain_lat` gate instead zeroed the *entire* latitude jet at the
/// boundary, making the basis nonsmooth there: an atom whose latitude reached
/// `±π/2` saw a zero latitude gradient and froze, even for the tangential
/// (in-box) direction along which the loss does decrease. Computing the exact
/// jet here and letting the retraction handle the bound restores a smooth
/// objective and the correct boundary behaviour. Both the core path
/// ([`SphereChartEvaluator`]) and the PyFFI helper route through this function.
pub fn sphere_chart_basis_jet(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    if coords.ncols() != 2 {
        return Err(format!(
            "sphere_chart_basis_jet expects latent_dim == 2, got {}",
            coords.ncols()
        ));
    }
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 7));
    let mut jet = Array3::<f64>::zeros((n, 7, 2));
    for row in 0..n {
        let lat = coords[[row, 0]];
        let lon = coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();
        let x = clat * clon;
        let y = clat * slon;
        let z = slat;
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = x;
        phi[[row, 2]] = y;
        phi[[row, 3]] = z;
        phi[[row, 4]] = x * y;
        phi[[row, 5]] = y * z;
        phi[[row, 6]] = x * z;

        let dx_dlat = -slat * clon;
        let dx_dlon = -clat * slon;
        let dy_dlat = -slat * slon;
        let dy_dlon = clat * clon;
        let dz_dlat = clat;
        jet[[row, 1, 0]] = dx_dlat;
        jet[[row, 1, 1]] = dx_dlon;
        jet[[row, 2, 0]] = dy_dlat;
        jet[[row, 2, 1]] = dy_dlon;
        jet[[row, 3, 0]] = dz_dlat;
        jet[[row, 4, 0]] = dx_dlat * y + x * dy_dlat;
        jet[[row, 4, 1]] = dx_dlon * y + x * dy_dlon;
        jet[[row, 5, 0]] = dy_dlat * z + y * dz_dlat;
        jet[[row, 5, 1]] = dy_dlon * z;
        jet[[row, 6, 0]] = dx_dlat * z + x * dz_dlat;
        jet[[row, 6, 1]] = dx_dlon * z;
    }
    Ok((phi, jet))
}

/// Lat/lon sphere chart evaluator used by the Rust-owned minimal SAE path.
#[derive(Debug, Clone)]
pub struct SphereChartEvaluator;

impl SaeBasisEvaluator for SphereChartEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        sphere_chart_basis_jet(coords)
    }
}

impl SaeBasisSecondJet for SphereChartEvaluator {
    /// Analytic Hessian of the 7-column lat/lon sphere chart basis.
    ///
    /// With `x = cos(lat) cos(lon)`, `y = cos(lat) sin(lon)`, `z = sin(lat)`
    /// the non-trivial second derivatives are
    ///
    /// ```text
    /// x_{lat,lat} = -x,     x_{lon,lon} = -x,     x_{lat,lon} = sin(lat)·sin(lon)
    /// y_{lat,lat} = -y,     y_{lon,lon} = -y,     y_{lat,lon} = -sin(lat)·cos(lon)
    /// z_{lat,lat} = -z,     z_{lon,lon} =  0,     z_{lat,lon} =  0
    /// ```
    ///
    /// Bilinear basis entries `xy, yz, xz` follow the product rule
    /// `(fg)_{αβ} = f_{αβ} g + f_α g_β + f_β g_α + f g_{αβ}`. The map is `C^∞`
    /// in `(lat, lon)`, so the Hessian is the exact analytic one with no clamp
    /// or boundary gating; the `lat ∈ [-π/2, π/2]` box is enforced by the
    /// retraction, not by truncating derivatives (see [`sphere_chart_basis_jet`]).
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator::second_jet expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut h = Array4::<f64>::zeros((n, 7, 2, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            let x = clat * clon;
            let y = clat * slon;
            let z = slat;
            let dx = [-slat * clon, -clat * slon];
            let dy = [-slat * slon, clat * clon];
            let dz = [clat, 0.0];
            let hx = [[-x, slat * slon], [slat * slon, -x]];
            let hy = [[-y, -slat * clon], [-slat * clon, -y]];
            let hz = [[-z, 0.0], [0.0, 0.0]];
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 1, axis_a, axis_b]] = hx[axis_a][axis_b];
                    h[[row, 2, axis_a, axis_b]] = hy[axis_a][axis_b];
                    h[[row, 3, axis_a, axis_b]] = hz[axis_a][axis_b];
                }
            }
            let pair = |hf: [[f64; 2]; 2],
                        df: [f64; 2],
                        f: f64,
                        hg: [[f64; 2]; 2],
                        dg: [f64; 2],
                        g: f64|
             -> [[f64; 2]; 2] {
                let mut out = [[0.0; 2]; 2];
                for axis_a in 0..2 {
                    for axis_b in 0..2 {
                        out[axis_a][axis_b] = hf[axis_a][axis_b] * g
                            + df[axis_a] * dg[axis_b]
                            + df[axis_b] * dg[axis_a]
                            + f * hg[axis_a][axis_b];
                    }
                }
                out
            };
            let hxy = pair(hx, dx, x, hy, dy, y);
            let hyz = pair(hy, dy, y, hz, dz, z);
            let hxz = pair(hx, dx, x, hz, dz, z);
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    h[[row, 4, axis_a, axis_b]] = hxy[axis_a][axis_b];
                    h[[row, 5, axis_a, axis_b]] = hyz[axis_a][axis_b];
                    h[[row, 6, axis_a, axis_b]] = hxz[axis_a][axis_b];
                }
            }
        }
        Ok(h)
    }
}

impl SaeBasisThirdJet for SphereChartEvaluator {
    /// Third derivative of the 7-column lat/lon sphere chart basis
    /// `[1, x, y, z, xy, yz, xz]`.
    ///
    /// Each Cartesian coordinate is *separable* in (lat, lon):
    /// `x = cos(lat) cos(lon)`, `y = cos(lat) sin(lon)`, `z = sin(lat)·1`. A
    /// separable coordinate's mixed derivative is the product of the per-axis
    /// derivative of the right order, so it is fully described by two
    /// length-4 derivative tables (orders 0..3) — one per axis. The map is
    /// `C^∞` in `(lat, lon)`; the tables are the exact analytic derivatives
    /// with no clamp or boundary gating (the `lat ∈ [-π/2, π/2]` box is
    /// enforced by the retraction, see [`sphere_chart_basis_jet`]).
    ///
    /// The bilinear columns `xy, yz, xz` are products of two separable
    /// coordinates; their third derivative is the symmetric triple-Leibniz sum
    /// over the `2³` ways to route the three derivative operators to the two
    /// factors. This is the order-3 generalization of the `pair` Leibniz used
    /// in [`SaeBasisSecondJet::second_jet`], so the two stay structurally
    /// identical and a finite difference of `second_jet` pins it.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "SphereChartEvaluator::third_jet expects latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let mut t3 = Array5::<f64>::zeros((n, 7, 2, 2, 2));
        // Derivative of a separable coordinate along axes `ax` (each 0 = lat,
        // 1 = lon): product of the lat table at order `#lat` and the lon table
        // at order `#lon`.
        let single = |lat: &[f64; 4], lon: &[f64; 4], ax: [usize; 3]| -> f64 {
            let n_lat = ax.iter().filter(|&&q| q == 0).count();
            lat[n_lat] * lon[3 - n_lat]
        };
        // Third derivative of a product of two separable coordinates: sum over
        // all 2³ routings of the three operators to factor f vs g (Leibniz).
        let product = |f_lat: &[f64; 4],
                       f_lon: &[f64; 4],
                       g_lat: &[f64; 4],
                       g_lon: &[f64; 4],
                       ax: [usize; 3]|
         -> f64 {
            let mut acc = 0.0;
            for mask in 0u8..8 {
                let (mut f_lat_n, mut f_lon_n, mut g_lat_n, mut g_lon_n) = (0, 0, 0, 0);
                for (i, &axis) in ax.iter().enumerate() {
                    let to_f = (mask >> i) & 1 == 1;
                    match (to_f, axis == 0) {
                        (true, true) => f_lat_n += 1,
                        (true, false) => f_lon_n += 1,
                        (false, true) => g_lat_n += 1,
                        (false, false) => g_lon_n += 1,
                    }
                }
                acc += f_lat[f_lat_n] * f_lon[f_lon_n] * g_lat[g_lat_n] * g_lon[g_lon_n];
            }
            acc
        };
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let slat = lat.sin();
            let clon = lon.cos();
            let slon = lon.sin();
            // Per-axis derivative tables, orders 0..3 (exact analytic, no clamp).
            let cos_lat = [clat, -slat, -clat, slat];
            let sin_lat = [slat, clat, -slat, -clat];
            let cos_lon = [clon, -slon, -clon, slon];
            let sin_lon = [slon, clon, -slon, -clon];
            let const_lon = [1.0, 0.0, 0.0, 0.0];
            // x = cos(lat)cos(lon), y = cos(lat)sin(lon), z = sin(lat).
            let (x_lat, x_lon) = (&cos_lat, &cos_lon);
            let (y_lat, y_lon) = (&cos_lat, &sin_lon);
            let (z_lat, z_lon) = (&sin_lat, &const_lon);
            for axis_a in 0..2 {
                for axis_b in 0..2 {
                    for axis_c in 0..2 {
                        let ax = [axis_a, axis_b, axis_c];
                        t3[[row, 1, axis_a, axis_b, axis_c]] = single(x_lat, x_lon, ax);
                        t3[[row, 2, axis_a, axis_b, axis_c]] = single(y_lat, y_lon, ax);
                        t3[[row, 3, axis_a, axis_b, axis_c]] = single(z_lat, z_lon, ax);
                        t3[[row, 4, axis_a, axis_b, axis_c]] =
                            product(x_lat, x_lon, y_lat, y_lon, ax);
                        t3[[row, 5, axis_a, axis_b, axis_c]] =
                            product(y_lat, y_lon, z_lat, z_lon, ax);
                        t3[[row, 6, axis_a, axis_b, axis_c]] =
                            product(x_lat, x_lon, z_lat, z_lon, ax);
                    }
                }
            }
        }
        Ok(t3)
    }
}

/// Tensor-product periodic harmonic evaluator for a `d`-dimensional torus
/// `T^d = (S^1)^d`. The basis is the tensor product over each axis of the
/// 1-D circle basis
/// `[1, cos(2π·1·t), sin(2π·1·t), …, cos(2π·H·t), sin(2π·H·t)]`
/// (each axis contributes `2H+1` factors, so the total basis size is
/// `(2H+1)^d`). The latent coords are angular phases in `[0, 1)` (consistent
/// with the periodic 1-D atoms).
#[derive(Debug, Clone)]
pub struct TorusHarmonicEvaluator {
    pub latent_dim: usize,
    pub num_harmonics: usize,
}

impl TorusHarmonicEvaluator {
    pub fn new(latent_dim: usize, num_harmonics: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("TorusHarmonicEvaluator requires latent_dim >= 1".to_string());
        }
        if num_harmonics == 0 {
            return Err("TorusHarmonicEvaluator requires num_harmonics >= 1".to_string());
        }
        Ok(Self {
            latent_dim,
            num_harmonics,
        })
    }

    pub fn axis_basis_size(&self) -> usize {
        2 * self.num_harmonics + 1
    }

    pub fn basis_size(&self) -> usize {
        // (2H+1)^d — computed iteratively to surface overflow.
        let axis_m = self.axis_basis_size();
        let mut total: usize = 1;
        for _ in 0..self.latent_dim {
            total = total
                .checked_mul(axis_m)
                .expect("TorusHarmonicEvaluator: basis size overflowed usize");
        }
        total
    }
}

impl SaeBasisEvaluator for TorusHarmonicEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator: expected latent_dim {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, d));
        // Per-axis evaluation buffer: phi_axis[axis][col] and dphi_axis[axis][col].
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                for h in 1..=h_max {
                    let freq = two_pi * (h as f64);
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * h - 1;
                    let c_idx = 2 * h;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                }
            }
            // Enumerate the Cartesian product of per-axis indices in
            // lexicographic order (axis 0 is the slowest).
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                let mut val = 1.0_f64;
                for axis in 0..d {
                    val *= phi_axis[axis][idx[axis]];
                }
                phi[[row, flat]] = val;
                // ∂/∂coords[row, axis_target] = product over axes, replacing
                // phi_axis[axis_target] with its derivative.
                for axis_target in 0..d {
                    let mut deriv = 1.0_f64;
                    for axis in 0..d {
                        deriv *= if axis == axis_target {
                            dphi_axis[axis][idx[axis]]
                        } else {
                            phi_axis[axis][idx[axis]]
                        };
                    }
                    jet[[row, flat, axis_target]] = deriv;
                }
                // Increment lexicographic index (last axis fastest).
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for TorusHarmonicEvaluator {
    /// Hessian of the tensor-product torus basis.
    ///
    /// Each basis function factors as `Φ_flat = Π_axis f_axis(t_axis)`, so
    ///
    /// * `∂² Φ / ∂t_a ∂t_b = (Π_{k ∉ {a, b}} f_k) · f_a'(t_a) · f_b'(t_b)`
    ///   when `a ≠ b`,
    /// * `∂² Φ / ∂t_a²    = (Π_{k ≠ a} f_k) · f_a''(t_a)` on the diagonal.
    ///
    /// Per-axis the basis is `[1, sin(2π h t), cos(2π h t), …]`, so
    /// `f_axis''(t) = -(2π h)² · f_axis(t)` on the harmonic columns and 0 on
    /// the constant column.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator::second_jet expects latent_dim == {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        let mut phi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut dphi_axis = vec![vec![0.0_f64; axis_m]; d];
        let mut d2phi_axis = vec![vec![0.0_f64; axis_m]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                phi_axis[axis][0] = 1.0;
                dphi_axis[axis][0] = 0.0;
                d2phi_axis[axis][0] = 0.0;
                for k in 1..=h_max {
                    let freq = two_pi * (k as f64);
                    let freq2 = freq * freq;
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * k - 1;
                    let c_idx = 2 * k;
                    phi_axis[axis][s_idx] = s;
                    phi_axis[axis][c_idx] = c;
                    dphi_axis[axis][s_idx] = freq * c;
                    dphi_axis[axis][c_idx] = -freq * s;
                    d2phi_axis[axis][s_idx] = -freq2 * s;
                    d2phi_axis[axis][c_idx] = -freq2 * c;
                }
            }
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        let mut prod = 1.0_f64;
                        for axis in 0..d {
                            let factor = if axis == axis_a && axis == axis_b {
                                d2phi_axis[axis][idx[axis]]
                            } else if axis == axis_a || axis == axis_b {
                                dphi_axis[axis][idx[axis]]
                            } else {
                                phi_axis[axis][idx[axis]]
                            };
                            prod *= factor;
                        }
                        hess[[row, flat, axis_a, axis_b]] = prod;
                    }
                }
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok(hess)
    }
}

impl SaeBasisThirdJet for TorusHarmonicEvaluator {
    /// Third derivative of the tensor-product torus basis.
    ///
    /// Each basis function factors as `Φ_flat = Π_axis f_axis(t_axis)`, so its
    /// third derivative `∂³Φ / ∂t_a ∂t_b ∂t_c` is the product, over every
    /// axis, of `f_axis` differentiated as many times as that axis appears in
    /// `{a, b, c}` (0..3). Per axis the basis is `[1, sin(2π h t),
    /// cos(2π h t), …]`, whose order-3 derivative is `[0, −(2π h)³ cos(…),
    /// +(2π h)³ sin(…), …]`. This is the order-3 sibling of
    /// [`SaeBasisSecondJet::second_jet`].
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        let d = self.latent_dim;
        if coords.ncols() != d {
            return Err(format!(
                "TorusHarmonicEvaluator::third_jet expects latent_dim == {d}, got {}",
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let axis_m = self.axis_basis_size();
        let m = self.basis_size();
        let h_max = self.num_harmonics;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut t3 = Array5::<f64>::zeros((n, m, d, d, d));
        // Per-axis derivative tables indexed [axis][order 0..3][column].
        let mut deriv_axis = vec![vec![vec![0.0_f64; axis_m]; 4]; d];
        for row in 0..n {
            for axis in 0..d {
                let t = coords[[row, axis]];
                for order in 0..4 {
                    deriv_axis[axis][order][0] = 0.0;
                }
                deriv_axis[axis][0][0] = 1.0;
                for k in 1..=h_max {
                    let freq = two_pi * (k as f64);
                    let freq2 = freq * freq;
                    let freq3 = freq2 * freq;
                    let angle = freq * t;
                    let s = angle.sin();
                    let c = angle.cos();
                    let s_idx = 2 * k - 1;
                    let c_idx = 2 * k;
                    deriv_axis[axis][0][s_idx] = s;
                    deriv_axis[axis][0][c_idx] = c;
                    deriv_axis[axis][1][s_idx] = freq * c;
                    deriv_axis[axis][1][c_idx] = -freq * s;
                    deriv_axis[axis][2][s_idx] = -freq2 * s;
                    deriv_axis[axis][2][c_idx] = -freq2 * c;
                    deriv_axis[axis][3][s_idx] = -freq3 * c;
                    deriv_axis[axis][3][c_idx] = freq3 * s;
                }
            }
            let mut idx = vec![0usize; d];
            for flat in 0..m {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        for axis_c in 0..d {
                            let mut prod = 1.0_f64;
                            for axis in 0..d {
                                let order = (axis == axis_a) as usize
                                    + (axis == axis_b) as usize
                                    + (axis == axis_c) as usize;
                                prod *= deriv_axis[axis][order][idx[axis]];
                            }
                            t3[[row, flat, axis_a, axis_b, axis_c]] = prod;
                        }
                    }
                }
                for axis in (0..d).rev() {
                    idx[axis] += 1;
                    if idx[axis] < axis_m {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
        Ok(t3)
    }
}

/// Affine Euclidean/Duchon fallback for the minimal fit entrypoint.
#[derive(Debug, Clone)]
pub struct AffineCoordinateEvaluator {
    pub latent_dim: usize,
}

impl AffineCoordinateEvaluator {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim }
    }
}

impl SaeBasisEvaluator for AffineCoordinateEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let mut phi = Array2::<f64>::zeros((n, m));
        let mut jet = Array3::<f64>::zeros((n, m, self.latent_dim));
        phi.column_mut(0).fill(1.0);
        for row in 0..n {
            for axis in 0..self.latent_dim {
                phi[[row, axis + 1]] = coords[[row, axis]];
                jet[[row, axis + 1, axis]] = 1.0;
            }
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for AffineCoordinateEvaluator {
    /// Second derivative of the affine basis `[1, t_1, ..., t_d]`.
    ///
    /// Every basis function is at most linear in `t`, so all second derivatives
    /// are identically zero. Returns the all-zeros tensor of shape
    /// `(n_obs, d+1, d, d)`.
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let d = self.latent_dim;
        Ok(Array4::<f64>::zeros((n, m, d, d)))
    }
}

impl SaeBasisThirdJet for AffineCoordinateEvaluator {
    /// Third derivative of the affine basis `[1, t_1, …, t_d]`. Every column is
    /// at most linear, so all third derivatives vanish identically.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "AffineCoordinateEvaluator::third_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let n = coords.nrows();
        let m = self.latent_dim + 1;
        let d = self.latent_dim;
        Ok(Array5::<f64>::zeros((n, m, d, d, d)))
    }
}

/// Scale-free Duchon atom evaluator for the SAE-manifold Newton loop.
///
/// Recomputes the radial+polynomial design `Φ(t)` and its first/second
/// input-location jets at arbitrary latent coordinates against a fixed set of
/// `centers` and Duchon null-space `order`. The column layout — the
/// kernel block `Φ_radial(t)·Z` followed by the polynomial block `P(t)`,
/// both carrying the same scalar kernel amplification `α` — matches
/// [`crate::basis::build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform). The
/// forward design and the jet are produced from a single core entry point
/// ([`crate::basis::duchon_sae_atom_basis_with_jet`]) so they always agree on
/// column count and scaling — the exact contract issue #247 pinned.
#[derive(Debug, Clone)]
pub struct DuchonCoordinateEvaluator {
    pub centers: Array2<f64>,
    pub order: crate::basis::DuchonNullspaceOrder,
}

impl DuchonCoordinateEvaluator {
    /// Build from the atom's centers and Duchon `m` (`m = 1` → constant
    /// null space, `m = 2` → constant+linear, `m = k+1` → degree-`k`).
    pub fn new(centers: Array2<f64>, m: usize) -> Result<Self, String> {
        if centers.ncols() == 0 {
            return Err("DuchonCoordinateEvaluator: centers must have at least one column".into());
        }
        if m == 0 {
            return Err("DuchonCoordinateEvaluator: Duchon m must be at least 1".into());
        }
        let order = match m {
            1 => crate::basis::DuchonNullspaceOrder::Zero,
            2 => crate::basis::DuchonNullspaceOrder::Linear,
            other => crate::basis::DuchonNullspaceOrder::Degree(other - 1),
        };
        Ok(Self { centers, order })
    }
}

impl SaeBasisEvaluator for DuchonCoordinateEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_basis_with_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

impl SaeBasisSecondJet for DuchonCoordinateEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator::second_jet: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_second_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

impl SaeBasisThirdJet for DuchonCoordinateEvaluator {
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.centers.ncols() {
            return Err(format!(
                "DuchonCoordinateEvaluator::third_jet: expected latent_dim {}, got {}",
                self.centers.ncols(),
                coords.ncols()
            ));
        }
        crate::basis::duchon_sae_atom_third_jet(coords, self.centers.view(), self.order)
            .map_err(|err| err.to_string())
    }
}

/// Flat Euclidean tangent-patch evaluator for the SAE-manifold Newton loop.
///
/// The basis is the set of monomials of total degree ≤ `max_degree` in the
/// atom's latent coordinates (a zero-curvature polynomial expansion, distinct
/// from the thin-plate Duchon kernel). It recomputes the monomial design and
/// its first/second derivatives at arbitrary coordinates, so the inner Newton
/// latent update stays consistent with the deployed design.
#[derive(Debug, Clone)]
pub struct EuclideanPatchEvaluator {
    pub latent_dim: usize,
    pub max_degree: usize,
}

impl EuclideanPatchEvaluator {
    pub fn new(latent_dim: usize, max_degree: usize) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("EuclideanPatchEvaluator: latent_dim must be positive".into());
        }
        Ok(Self {
            latent_dim,
            max_degree,
        })
    }

    fn order(&self) -> crate::basis::DuchonNullspaceOrder {
        match self.max_degree {
            0 => crate::basis::DuchonNullspaceOrder::Zero,
            1 => crate::basis::DuchonNullspaceOrder::Linear,
            k => crate::basis::DuchonNullspaceOrder::Degree(k),
        }
    }
}

impl SaeBasisEvaluator for EuclideanPatchEvaluator {
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        Some(<Self as SaeBasisSecondJet>::second_jet(self, coords))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        Some(<Self as SaeBasisThirdJet>::third_jet(self, coords))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let mut phi = Array2::<f64>::zeros((n, m));
        for (col, alpha) in exponents.iter().enumerate() {
            for row in 0..n {
                let mut value = 1.0_f64;
                for (axis, &exp) in alpha.iter().enumerate() {
                    if exp != 0 {
                        value *= coords[[row, axis]].powi(exp as i32);
                    }
                }
                phi[[row, col]] = value;
            }
        }
        let jet = crate::basis::duchon_polynomial_first_derivative_nd(coords, self.order());
        if jet.shape() != [n, m, self.latent_dim] {
            return Err(format!(
                "EuclideanPatchEvaluator: monomial jet shape {:?} disagrees with ({n}, {m}, {})",
                jet.shape(),
                self.latent_dim
            ));
        }
        Ok((phi, jet))
    }
}

impl SaeBasisSecondJet for EuclideanPatchEvaluator {
    fn second_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array4<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::second_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let d = self.latent_dim;
        let mut hess = Array4::<f64>::zeros((n, m, d, d));
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..d {
                if alpha[a] == 0 {
                    continue;
                }
                for c in 0..d {
                    if a != c && alpha[c] == 0 {
                        continue;
                    }
                    let lead = if a == c {
                        (alpha[a] as f64) * (alpha[a].saturating_sub(1) as f64)
                    } else {
                        (alpha[a] as f64) * (alpha[c] as f64)
                    };
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n {
                        let mut value = lead;
                        for axis in 0..d {
                            let mut exp = alpha[axis];
                            if axis == a {
                                exp = exp.saturating_sub(1);
                            }
                            if axis == c {
                                exp = exp.saturating_sub(1);
                            }
                            if exp != 0 {
                                value *= coords[[row, axis]].powi(exp as i32);
                            }
                        }
                        hess[[row, col, a, c]] = value;
                    }
                }
            }
        }
        Ok(hess)
    }
}

impl SaeBasisThirdJet for EuclideanPatchEvaluator {
    /// Third derivative of the monomial basis `Φ_α = Π_axis t_axis^{α_axis}`.
    ///
    /// Differentiating axis `j` a total of `k_j` times (where `k_j` is how
    /// often axis `j` appears in `{a, b, c}`) contracts that factor to
    /// `falling(α_j, k_j) · t_j^{α_j − k_j}`, with `falling(α, k) = α(α−1)…
    /// (α−k+1)` and the term vanishing whenever `α_j < k_j`.
    fn third_jet(&self, coords: ArrayView2<'_, f64>) -> Result<Array5<f64>, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "EuclideanPatchEvaluator::third_jet: expected latent_dim {}, got {}",
                self.latent_dim,
                coords.ncols()
            ));
        }
        let exponents = crate::basis::monomial_exponents(self.latent_dim, self.max_degree);
        let n = coords.nrows();
        let m = exponents.len();
        let d = self.latent_dim;
        let mut t3 = Array5::<f64>::zeros((n, m, d, d, d));
        let falling = |alpha: usize, k: usize| -> f64 {
            let mut acc = 1.0_f64;
            for j in 0..k {
                acc *= (alpha as f64) - (j as f64);
            }
            acc
        };
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..d {
                if alpha[a] == 0 {
                    continue;
                }
                for b in 0..d {
                    for c in 0..d {
                        // Per-axis differentiation order in this (a, b, c) cell.
                        let mut order = vec![0usize; d];
                        order[a] += 1;
                        order[b] += 1;
                        order[c] += 1;
                        if (0..d).any(|axis| order[axis] > alpha[axis]) {
                            continue;
                        }
                        let mut lead = 1.0_f64;
                        for axis in 0..d {
                            lead *= falling(alpha[axis], order[axis]);
                        }
                        if lead == 0.0 {
                            continue;
                        }
                        for row in 0..n {
                            let mut value = lead;
                            for axis in 0..d {
                                let exp = alpha[axis] - order[axis];
                                if exp != 0 {
                                    value *= coords[[row, axis]].powi(exp as i32);
                                }
                            }
                            t3[[row, col, a, b, c]] = value;
                        }
                    }
                }
            }
        }
        Ok(t3)
    }
}

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
        let (phi, jet) = evaluator.evaluate(coords)?;
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

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone, Copy)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic concrete relaxation of a truncated IBP active set.
    IBPMap {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Hard-thresholded bounded gate: each atom is off (gate = 0) when its logit
    /// is at or below `threshold`, and on with a threshold-centered shifted
    /// sigmoid `σ((logit − threshold) / temperature) ∈ [0.5, 1)` above it. This
    /// is NOT literal JumpReLU `z·1[z>θ]` — the gate carries no magnitude; it is
    /// a member of the gate family (softmax simplex / IBP sigmoid / this hard
    /// gate) and stays bounded in [0, 1]. Reconstruction magnitude lives entirely
    /// in the decoder curve `g_k(t) = φ(t)ᵀ B_k`. The discontinuity at `threshold`
    /// (0 → 0.5) is the intended "jump".
    JumpReLU { temperature: f64, threshold: f64 },
}

impl AssignmentMode {
    #[must_use]
    pub fn softmax(temperature: f64) -> Self {
        Self::Softmax {
            temperature,
            sparsity: 1.0,
        }
    }

    #[must_use]
    pub fn ibp_map(temperature: f64, alpha: f64, learnable_alpha: bool) -> Self {
        Self::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        }
    }

    #[must_use]
    pub fn jumprelu(temperature: f64, threshold: f64) -> Self {
        Self::JumpReLU {
            temperature,
            threshold,
        }
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::JumpReLU { temperature, .. } => temperature,
        }
    }

    fn set_temperature(&mut self, new_temperature: f64) -> Result<(), String> {
        if !(new_temperature.is_finite() && new_temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {new_temperature}"
            ));
        }
        match self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::JumpReLU { temperature, .. } => {
                *temperature = new_temperature;
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        let temperature = self.temperature();
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {temperature}"
            ));
        }
        match *self {
            AssignmentMode::Softmax { sparsity, .. } => {
                if !(sparsity.is_finite() && sparsity > 0.0) {
                    return Err(format!(
                        "AssignmentMode::Softmax: sparsity must be finite and positive; got {sparsity}"
                    ));
                }
            }
            AssignmentMode::IBPMap { alpha, .. } => {
                if !(alpha.is_finite() && alpha > 0.0) {
                    return Err(format!(
                        "AssignmentMode::IBPMap: alpha must be finite and positive; got {alpha}"
                    ));
                }
            }
            AssignmentMode::JumpReLU { threshold, .. } => {
                if !threshold.is_finite() {
                    return Err(format!(
                        "AssignmentMode::JumpReLU: threshold must be finite; got {threshold}"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Per-row latent assignment state.
///
/// The stored assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent IBP-MAP sigmoid active indicators,
/// or JumpReLU gates. Softmax logits are canonicalized to the reference chart
/// `logits[K - 1] = 0`, so the row-local Newton coordinates contain only the
/// first `K - 1` logits (`0` coordinates for `K = 1`). Gate-style modes keep
/// all `K` logits as identifiable scalar parameters. `coords[k]` holds
/// `t_{.,k}` for atom `k`.
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
}

impl SaeAssignment {
    #[must_use = "build error must be handled"]
    pub fn new(
        logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        temperature: f64,
    ) -> Result<Self, String> {
        Self::with_mode(logits, coords, AssignmentMode::softmax(temperature))
    }

    #[must_use = "build error must be handled"]
    pub fn with_mode(
        mut logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        mode.validate()?;
        let n = logits.nrows();
        let k = logits.ncols();
        if coords.len() != k {
            return Err(format!(
                "SaeAssignment::new: coords length {} must equal K={k}",
                coords.len()
            ));
        }
        for (atom, coord) in coords.iter().enumerate() {
            if coord.n_obs() != n {
                return Err(format!(
                    "SaeAssignment::new: coord atom {atom} has n_obs={} but logits has {n}",
                    coord.n_obs()
                ));
            }
        }
        for row in 0..n {
            validate_finite_logits(logits.row(row), row)?;
        }
        if matches!(mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut logits);
        }
        Ok(Self {
            logits,
            coords,
            mode,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.logits.nrows()
    }

    pub fn k_atoms(&self) -> usize {
        self.logits.ncols()
    }

    pub fn total_coord_dim(&self) -> usize {
        self.coords.iter().map(|c| c.latent_dim()).sum()
    }

    pub fn assignment_coord_dim(&self) -> usize {
        match self.mode {
            AssignmentMode::Softmax { .. } => self.k_atoms().saturating_sub(1),
            AssignmentMode::IBPMap { .. } | AssignmentMode::JumpReLU { .. } => self.k_atoms(),
        }
    }

    pub fn row_block_dim(&self) -> usize {
        self.assignment_coord_dim() + self.total_coord_dim()
    }

    pub fn coord_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = self.assignment_coord_dim();
        for coord in &self.coords {
            out.push(cursor);
            cursor += coord.latent_dim();
        }
        out
    }

    pub fn assignments(&self) -> Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.assignments_row(row);
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        out
    }

    pub fn try_assignments(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row(row)?;
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        Ok(out)
    }

    pub fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        validate_finite_logits(self.logits.row(row), row)?;
        // Only Softmax collapses to a fixed assignment at K==1: its
        // assignment_coord_dim is K-1 = 0, so there is no free logit. IBPMap and
        // JumpReLU keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                Ok(softmax_row(self.logits.row(row), temperature))
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => Ok(ibp_map_row(self.logits.row(row), temperature, alpha)),
            AssignmentMode::JumpReLU {
                temperature,
                threshold,
            } => Ok(jumprelu_row(self.logits.row(row), temperature, threshold)),
        }
    }

    /// Flatten extension coordinates in row-major SAE layout:
    /// `(assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])` for every row.
    /// Softmax contributes the first `K - 1` reference logits and omits the
    /// fixed reference logit; gate-style assignment modes contribute all `K`
    /// logits.
    pub fn flatten_ext_coords(&self) -> Array1<f64> {
        let n = self.n_obs();
        let q = self.row_block_dim();
        let k = self.k_atoms();
        let assignment_dim = self.assignment_coord_dim();
        let offsets = self.coord_offsets();
        let mut out = Array1::<f64>::zeros(n * q);
        for row in 0..n {
            let base = row * q;
            for atom in 0..assignment_dim {
                out[base + atom] = self.logits[[row, atom]];
            }
            for atom in 0..k {
                let d = self.coords[atom].latent_dim();
                let t_row = self.coords[atom].row(row);
                for axis in 0..d {
                    out[base + offsets[atom] + axis] = t_row[axis];
                }
            }
        }
        out
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_no_gauge(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        temperature: f64,
    ) -> Result<Self, String> {
        let coords = coord_blocks
            .iter()
            .map(|c| LatentCoordValues::from_matrix(c.view(), LatentIdMode::None))
            .collect();
        Self::new(logits, coords, temperature)
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        let coords = coord_blocks
            .iter()
            .map(|c| LatentCoordValues::from_matrix(c.view(), LatentIdMode::None))
            .collect();
        Self::with_mode(logits, coords, mode)
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode_and_manifolds(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        manifolds: Vec<LatentManifold>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        if coord_blocks.len() != manifolds.len() {
            return Err(format!(
                "SaeAssignment::from_blocks_with_mode_and_manifolds: coord block length {} != manifold length {}",
                coord_blocks.len(),
                manifolds.len()
            ));
        }
        let coords = coord_blocks
            .iter()
            .zip(manifolds)
            .map(|(c, manifold)| {
                LatentCoordValues::from_matrix_with_manifold(c.view(), LatentIdMode::None, manifold)
            })
            .collect();
        Self::with_mode(logits, coords, mode)
    }
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

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

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
    pub decoder_covariance: Array2<f64>,
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
    /// JumpReLU optimization active set: atoms within the reactivation band
    /// `logit > threshold − MARGIN·τ` (see [`jumprelu_in_optimization_band`]).
    /// This is intentionally wider than the hard forward gate `logit > threshold`
    /// so that gated-off atoms near the boundary remain in the Newton system for
    /// value-consistent prior terms. Their forward reconstruction contribution
    /// and data-fit logit JVP remain hard-zero while `a_k = 0`.
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

/// Full SAE-manifold term.
#[derive(Debug, Clone)]
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
        })
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
        const HOST_IN_CORE_BYTES: usize = 2 * 1024 * 1024 * 1024;
        const CPU_L2_CACHE_BYTES: usize = 1024 * 1024;
        const CHUNK_CACHE_MULTIPLE: usize = 8;
        let n_obs = self.n_obs();
        let total_basis: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = self
            .atoms
            .iter()
            .map(|atom| atom.latent_dim)
            .max()
            .unwrap_or(0);
        let (budget, chunk_window) = match crate::gpu::runtime::GpuRuntime::global() {
            Some(rt) => {
                // Aggregate the working-set budget over EVERY device: per-atom
                // batched smoothness/Gram work fans out across the whole pool, so
                // the in-core ceiling is the sum of all ordinals' budgets rather
                // than only the primary device's. The per-tile chunk window still
                // sizes against a single device's slice (one ordinal at a time
                // holds a tile in core), so it uses the per-device budget.
                let aggregate_budget: usize = rt
                    .device_ordinals()
                    .iter()
                    .map(|&ord| rt.memory_budget_for(ord))
                    .sum();
                let per_device_budget = aggregate_budget / rt.device_count().max(1);
                let window =
                    (per_device_budget / 16).max(CPU_L2_CACHE_BYTES * CHUNK_CACHE_MULTIPLE);
                (aggregate_budget / 4, window)
            }
            None => (
                HOST_IN_CORE_BYTES,
                CPU_L2_CACHE_BYTES * CHUNK_CACHE_MULTIPLE,
            ),
        };
        sae_streaming_plan_from_budget(
            n_obs,
            total_basis,
            self.k_atoms(),
            d_max,
            budget,
            chunk_window,
        )
    }

    /// Construction-time validation: every Psi-tier analytic penalty in the
    /// registry must be dispatchable into the SAE arrow-Schur row layout.
    ///
    /// Two invariants are enforced upfront so the dispatch loop in
    /// `add_sae_analytic_penalty_contributions` is total (no runtime
    /// "unsupported penalty" fallthrough, no per-call K-gating):
    ///
    /// 1. Every Psi-tier penalty is either a logit-target penalty
    ///    (`IBPAssignment`, `SoftmaxAssignmentSparsity`) or in
    ///    [`sae_penalty_is_row_block_supported`], or `NuclearNorm` (which is
    ///    redirected to the per-atom decoder (β) block rather than the coord
    ///    "t" row block). Penalty kinds with cross-row structure
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
            let is_logit = matches!(
                penalty,
                AnalyticPenaltyKind::IBPAssignment(_)
                    | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            );
            if is_logit {
                continue;
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
        const BYTES_PER_F64: usize = 8;
        // Host in-core ceiling for the dense Gram on a CPU-only build, mirroring
        // the streaming dispatcher's host budget (`sae_streaming_plan`).
        const HOST_GRAM_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
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
            .saturating_mul(BYTES_PER_F64);

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
            None => HOST_GRAM_BYTES,
        };
        if dense_gram_bytes <= budget {
            return None;
        }

        // Sparse Gram footprint scales with the per-row active basis count
        // `k_active · m_atom`. Solve for the largest `k_active` whose sparse
        // Gram `(k_active · m_atom)²` still fits the budget.
        let m_atom = (m_total as f64 / k_atoms as f64).max(1.0);
        let max_active_basis = ((budget as f64 / BYTES_PER_F64 as f64).sqrt() / m_atom).floor();
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
        // gauge (see `analytic_penalties::corrected_isometry_penalty`). With no
        // producer of `WhitenedStructured` at the SAE surface today, this path is
        // bit-for-bit the historical isotropic data-fit and value/gradient stay
        // trivially in sync (no whitening happens).
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut resid_row = ndarray::Array1::<f64>::zeros(target.ncols());
        for row in 0..target.nrows() {
            for out_col in 0..target.ncols() {
                resid_row[out_col] = target[[row, out_col]] - fitted[[row, out_col]];
            }
            match self.row_metric.as_ref() {
                Some(metric) if whitens => {
                    for w in metric.whiten_residual_row(row, resid_row.view()) {
                        data_fit += 0.5 * w * w;
                    }
                }
                _ => {
                    for &r in resid_row.iter() {
                        data_fit += 0.5 * r * r;
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
        let logits_flat = flat_logits(self.assignment.logits.view());
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
                    if matches!(
                        penalty,
                        AnalyticPenaltyKind::IBPAssignment(_)
                            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
                    ) {
                        value += penalty.value(logits_flat.view(), rho_local);
                    } else if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
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
    /// SAE state. This is the per-atom `½μ Σ_n ‖J_n^T W_n J_n − g_ref‖²`
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
        let beta_offsets = self.beta_offsets();
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
            // KroneckerPenaltyOp: factor_a = λ·S_k (m×m), factor_b = I_p (p×p).
            let identity_p = Array2::<f64>::eye(p);
            smooth_ops.push(Arc::new(KroneckerPenaltyOp {
                factor_a: scaled_s,
                factor_b: identity_p,
                global_offset: off,
                k: beta_dim,
            }));
        }

        // Per-row active-set layout. Engaged for two regimes:
        //   * JumpReLU — structural gate plus a reactivation band: atoms with
        //     `logit > threshold − MARGIN·τ` enter the compact solve
        //     ([`jumprelu_in_optimization_band`]). Strictly gated-off atoms
        //     (logit ≤ threshold) carry zero assignment mass so their data-fit
        //     reconstruction contribution and data-fit logit JVP are zero, but
        //     band atoms keep value-consistent prior gradient in the row block.
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
        // Build the Arrow-Schur system: heterogeneous row dims when a compact
        // layout is active, uniform `q` otherwise.
        let mut sys = if let Some(ref layout) = row_layout {
            let per_row_dims: Vec<usize> = (0..n).map(|row| layout.row_q_active(row)).collect();
            ArrowSchurSystem::new_with_per_row_dims(per_row_dims, beta_dim)
        } else {
            ArrowSchurSystem::new(n, q, beta_dim)
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
        // Scratch buffer for per-(row, atom) decoded outputs. The full `decoded`
        // matrix retains all atoms for this row so the assignment-Jacobian
        // helper can read it.
        let mut decoded_scratch = vec![0.0_f64; p];
        // Kronecker htbeta storage: per-row sparse support and local Jacobian.
        // These replace the O(q · K · p) dense htbeta write with O(m_i · q · p)
        // storage; the Arrow-Schur solver accesses them via htbeta_matvec.
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

            // Determine whether this row uses the compact active-set layout.
            //   * JumpReLU: gated atoms plus the reactivation band
            //     (logit > threshold − MARGIN·τ) enter.
            //   * IBP-MAP at large K: only the top-`k_active` atoms.
            //   * Otherwise (small K): the dense uniform-q layout.
            let (q_row, local_jac_row) = if let Some(ref layout) = row_layout {
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

            // Build the per-row Arrow-Schur block at the row's active dim.
            let mut block = ArrowRowBlock::new(q_row, beta_dim);
            for a in 0..q_row {
                let mut g = 0.0;
                for out_col in 0..p {
                    g += local_jac_row[[a, out_col]] * error[out_col];
                }
                block.gt[a] += g;
                for b in 0..q_row {
                    let mut h = 0.0;
                    for out_col in 0..p {
                        h += local_jac_row[[a, out_col]] * local_jac_row[[b, out_col]];
                    }
                    block.htt[[a, b]] += h;
                }
            }

            // Assignment prior in logit space.
            // For compact layout: position `j` = active_atoms index.
            // For dense layout: position `atom_idx` directly.
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
            // The dense (q × K·p) htbeta block is NOT written here.  Instead,
            // `a_phi` and `local_jac_row` are captured per-row into `SaeKroneckerRows`
            // and installed as `sys.htbeta_matvec` after the row loop.  All
            // Arrow-Schur inner paths (schur_matvec, reduced_rhs_beta,
            // build_dense_schur_*, JacobiPreconditioner) route through
            // `sys_htbeta_apply_row` / `sys_htbeta_accumulate_transpose` which
            // already prefer `htbeta_matvec` over the dense slab.
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
                    let w = a_k * phi;
                    a_phi.push((atom_beta_off + basis_col * p, w));
                    wphi.push(w);
                }
                weighted_phi.push((atom_idx, wphi));
            }
            // β data-fit gradient `gᵦ += J_βᵀ error`.
            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                if j_beta_i == 0.0 {
                    continue;
                }
                for out_col in 0..p {
                    sys.gb[beta_base_i + out_col] += j_beta_i * error[out_col];
                    // No htbeta write — the Kronecker matvec handles this.
                    // No dense hbb write — the sparse `G ⊗ I_p` op installed
                    // after the loop carries the data-fit GN β-Hessian.
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
            // Save per-row Kronecker data for htbeta_matvec construction.
            kron_a_phi.push(a_phi);
            // Flatten local_jac_row row-major into a plain Vec<f64> (q_row * p entries).
            let mut jac_flat = vec![0.0_f64; q_row * p];
            for c in 0..q_row {
                for j in 0..p {
                    jac_flat[c * p + j] = local_jac_row[[c, j]];
                }
            }
            kron_jac.push(jac_flat);
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
            if !manifold.is_euclidean() {
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
        // Build and install the Kronecker htbeta_matvec.
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
        {
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
        let mut beta_penalty_written = false;
        if let Some(registry) = analytic_penalties {
            // Upfront validation: refuse penalty kinds the SAE row layout
            // cannot host, and refuse mixed-d row-block configurations.
            // This makes the dispatch loop below total — no runtime
            // "unsupported penalty" fallthrough, no K-gating.
            self.validate_analytic_penalty_registry(registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
            beta_penalty_written = self
                .add_sae_analytic_penalty_contributions(
                    &mut sys,
                    registry,
                    penalty_scale,
                    row_layout.as_ref(),
                )
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
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
        {
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
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(SparseBlockKroneckerPenaltyOp {
                p,
                dim_a: m_total,
                k: beta_dim,
                blocks: g_sparse_blocks,
            }));
            if beta_penalty_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
        }
        // Store the active-set layout for `apply_newton_step`.
        self.last_row_layout = row_layout;
        Ok(sys)
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
        if self.streaming_plan().streaming {
            self.reml_criterion_streaming_exact(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
        } else {
            let (v, loss, _cache) = self.reml_criterion_with_cache(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
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
        )?;
        let log_det = arrow_log_det_from_cache(&cache).ok_or_else(|| {
            "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None at \
             ridge=0 Direct mode (no dense Schur factor); the joint Hessian log-det is \
             required for the Laplace normaliser"
                .to_string()
        })?;

        // 3. Smoothing-penalty Occam term: −½·p·(Σ_k rank S_k)·log λ_smooth.
        let p_out = self.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.atoms {
            smooth_rank_total += Self::symmetric_rank(&atom.smooth_penalty)?;
        }
        let occam = 0.5 * p_out * (smooth_rank_total as f64) * rho.log_lambda_smooth;

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
        let max_refine_iter = inner_max_iter.max(1).saturating_mul(16).max(64);
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
            let (delta_t, delta_beta, cache): (Array1<f64>, Array1<f64>, ArrowFactorCache) =
                solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, options)
                    .map_err(|err| format!("SaeManifoldTerm::reml_criterion: {err}"))?;
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
            if !step_norm_sq.is_finite() || !grad_norm_sq.is_finite() {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: undamped inner residual is non-finite at \
                     the inner optimum (‖Δ‖²={step_norm_sq}, ‖g‖²={grad_norm_sq}); the joint \
                     Hessian factorisation is degenerate at this ρ"
                ));
            }
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
            let step_norm = step_norm_sq.sqrt();
            let grad_norm = grad_norm_sq.sqrt();
            let iterate_scale = 1.0 + iterate_norm_sq.sqrt();
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
            if grad_norm <= grad_tolerance || step_norm <= step_tolerance {
                return Ok(cache);
            }
            if total_inner_iter >= max_refine_iter {
                return Err(format!(
                    "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ; \
                     neither the KKT gradient ‖g‖={grad_norm:.6e} (tol {grad_tolerance:.6e}) nor \
                     the undamped Newton step ‖Δ‖={step_norm:.6e} (tol {step_tolerance:.6e}) met \
                     tolerance after {total_inner_iter} inner iterations. Refusing to rank an \
                     off-optimum Laplace criterion."
                ));
            }
            let remaining = max_refine_iter - total_inner_iter;
            let refine_iter = inner_max_iter.max(1).min(remaining);
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

    fn reml_occam_term(&self, rho: &SaeManifoldRho) -> Result<f64, String> {
        let p_out = self.output_dim() as f64;
        let mut smooth_rank_total = 0usize;
        for atom in &self.atoms {
            smooth_rank_total += Self::symmetric_rank(&atom.smooth_penalty)?;
        }
        Ok(0.5 * p_out * (smooth_rank_total as f64) * rho.log_lambda_smooth)
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
        let n_total = self.n_obs();
        let chunk_size = self.streaming_plan().chunk_size.min(n_total.max(1));
        let beta_dim = self.beta_dim();
        let mut schur_acc = Array2::<f64>::zeros((beta_dim, beta_dim));
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
            let z_chunk = target.slice(s![start..end, ..]);
            let sys = chunk
                .assemble_arrow_schur_scaled(z_chunk, rho, registry, penalty_scale)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len().max(1));
            let (chunk_log_det_tt, chunk_schur) = streaming
                .reduced_schur_and_log_det_tt(0.0, 0.0, &options)
                .map_err(|err| format!("SaeManifoldTerm::streaming_exact_arrow_log_det: {err}"))?;
            log_det_tt += chunk_log_det_tt;
            for row in 0..beta_dim {
                for col in 0..beta_dim {
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
        let beta_offsets = self.beta_offsets();
        let k = cache.k;
        let mut trace = 0.0_f64;
        let mut m_col = Array1::<f64>::zeros(k);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let s = &atom.smooth_penalty;
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            for mu in 0..m {
                for oc in 0..p {
                    let col = off + mu * p + oc;
                    // M[:,col] = λ · S_k[:,mu] ⊗ e_oc (nonzero at off+ν·p+oc).
                    m_col.fill(0.0);
                    for nu in 0..m {
                        let s_nu_mu = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                        m_col[off + nu * p + oc] = lambda_smooth * s_nu_mu;
                    }
                    let z = cache.schur_inverse_apply(m_col.view())?;
                    trace += z[col];
                }
            }
        }
        Ok(trace)
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
        let beta_edf = (self.beta_dim() as f64 - smooth_edf).max(0.0);
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
        let blocks = self.beta_block_offsets();
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let mut cov = cache
                .schur_inverse_block(blocks[k].clone())
                .map_err(|e| format!("assemble_shape_uncertainty: atom {k}: {e}"))?;
            cov.mapv_inplace(|v| v * dispersion);

            let m = atom.basis_size();
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            // Evenly-strided evaluation rows bound the band cost; the full
            // covariance above is exact and lets callers evaluate any grid.
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
                // Var_c = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]; the flat
                // decoder index is basis·p + channel (row-major (M_k, p)).
                for c in 0..p {
                    let mut var = 0.0_f64;
                    for b1 in 0..m {
                        let phi1 = atom.basis_values[[row, b1]];
                        if phi1 == 0.0 {
                            continue;
                        }
                        let i1 = b1 * p + c;
                        for b2 in 0..m {
                            let phi2 = atom.basis_values[[row, b2]];
                            if phi2 == 0.0 {
                                continue;
                            }
                            var += phi1 * phi2 * cov[[i1, b2 * p + c]];
                        }
                    }
                    band_sd[[gi, c]] = var.max(0.0).sqrt();
                }
            }
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords,
                band_mean,
                band_sd,
            });
        }
        Ok(SaeShapeUncertainty { dispersion, atoms })
    }

    /// Returns `true` when a Beta-tier analytic penalty was accumulated into
    /// the dense `sys.hbb` block (so the caller knows to wrap it in a
    /// `DensePenaltyOp`); `false` leaves `sys.hbb` all-zero and lets the
    /// caller skip the dense `(K·p)²` operator entirely.
    fn add_sae_analytic_penalty_contributions(
        &self,
        sys: &mut ArrowSchurSystem,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        row_layout: Option<&SaeRowLayout>,
    ) -> Result<bool, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let logits_flat = flat_logits(self.assignment.logits.view());
        let beta = self.flatten_beta();
        let mut beta_penalty_written = false;
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
                    if matches!(
                        penalty,
                        AnalyticPenaltyKind::IBPAssignment(_)
                            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
                    ) {
                        self.add_sae_logit_penalty(
                            sys,
                            penalty,
                            logits_flat.view(),
                            rho_local,
                            row_layout,
                        );
                    } else if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) {
                        // NuclearNorm is a Psi-tier penalty but it targets each
                        // atom's decoder (β) matrix singular spectrum, not the
                        // coord "t" row block. Route it to the β tier so it
                        // shrinks each atom's embedding rank.
                        self.add_sae_beta_penalty(
                            sys,
                            penalty,
                            beta.view(),
                            rho_local,
                            penalty_scale,
                        );
                        beta_penalty_written = true;
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
                                        sys, atom_idx, coord, corrected, rho_local,
                                    );
                                    beta_penalty_written = true;
                                }
                            } else {
                                self.add_sae_coord_penalty(
                                    sys, atom_idx, off, coord, penalty, rho_local, row_layout,
                                );
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    // β-tier analytic penalties are global (B-only); minibatch-
                    // scaled so per-chunk sums reconstruct one global copy.
                    self.add_sae_beta_penalty(sys, penalty, beta.view(), rho_local, penalty_scale);
                    beta_penalty_written = true;
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(beta_penalty_written)
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
                sys, atom_idx, dense_off, coord, corrected, rho_local, row_layout,
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
                    sys.rows[row].htbeta[[row_off + c, beta_off + beta_col]] += mu * acc;
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
    ) {
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
                return;
            };
            let beta_dim = self.beta_dim();
            let grad = per_fit.grad_target(target_beta, rho_local);
            for j in 0..beta_dim {
                sys.gb[j] += penalty_scale * grad[j];
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
            return;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                self.add_sae_mech_sparsity_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                );
            }
            return;
        }
        // NuclearNormPenalty is a smoothed sum of singular values of a single
        // (n_eff, latent_dim) matrix. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so it must
        // operate per atom on that atom's own `[beta_offsets[k] .. +M_k*p)`
        // slice as an `M_k × p` matrix (`n_eff = M_k`, `latent_dim = p`). This
        // penalizes the embedding rank of each atom's decoder independently.
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                self.add_sae_nuclear_norm_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                );
            }
            return;
        }
        let k = self.beta_dim();
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            sys.gb[j] += penalty_scale * grad[j];
        }
        // `hbb` is the PSD Newton / PIRLS curvature block for the β tier:
        // accumulate the PSD majorizer (exact for convex penalties), not the
        // indefinite exact Hessian, so the solve stays positive-definite.
        if let Some(diag) = penalty.psd_majorizer_diag(target_beta, rho_local) {
            for j in 0..k {
                sys.hbb[[j, j]] += penalty_scale * diag[j];
            }
            return;
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
    ) {
        let grad = per_atom.grad_target(target_beta, rho_local);
        for j in start..end {
            sys.gb[j] += penalty_scale * grad[j];
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
    ) {
        let block = target_beta.slice(s![start..end]);
        let block_len = end - start;
        let grad = per_atom.grad_target(block, rho_local);
        for local in 0..block_len {
            sys.gb[start + local] += penalty_scale * grad[local];
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
        sys.solve_with_lm_escalation(ridge_ext_coord, ridge_beta)
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
        if delta_beta.len() != self.beta_dim() {
            return Err(format!(
                "SaeManifoldTerm::apply_newton_step: delta_beta length {} != expected {}",
                delta_beta.len(),
                self.beta_dim()
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
            // Apply logits from expanded buffer.
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] +=
                        step_size * full_delta[row_base + atom_idx];
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
            for row in 0..n {
                let row_base = row * q;
                for atom_idx in 0..assignment_dim {
                    self.assignment.logits[[row, atom_idx]] +=
                        step_size * delta_ext_coord[row_base + atom_idx];
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
        for idx in 0..beta.len() {
            beta[idx] += step_size * delta_beta[idx];
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
            // Per-row global argmin of ‖x_i − decoded_g‖² over the grid.
            let mut seeded = Array2::<f64>::zeros((n, d));
            for row in 0..n {
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
        for _ in 0..max_iter {
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
            let (delta_ext_coord, delta_beta, _diag) = sys
                .solve_with_lm_escalation(ridge_ext_coord, ridge_beta)
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
            if !(pre_step_total.is_finite()
                && directional_decrease.is_finite()
                && directional_decrease > 0.0
                && directional_decrease > directional_decrease_floor)
            {
                // Pre-step state is unperturbed here; restore is a no-op but
                // keeps the invariant explicit.
                self.restore_mutable_state(&snapshot);
                break;
            }

            let mut trial_step_size = step_size;
            let mut accepted = false;
            for halving in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
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
                    &ArrowSolveOptions::automatic(self.beta_dim()),
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
                    Err(_err) => {
                        self.restore_mutable_state(&snapshot);
                        break;
                    }
                };
                if !(accepted_step.trial_objective_value.is_finite()
                    && accepted_step.trial_objective_value < pre_step_total)
                {
                    self.restore_mutable_state(&snapshot);
                    break;
                }
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
        let beta_dim = self.beta_dim();

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
        };
        for _ in 0..max_iter {
            self.advance_temperature_schedule()?;
            // ── Pass 1: accumulate the global reduced Schur over β online. ──
            let options = ArrowSolveOptions::automatic(beta_dim);
            let mut s_acc = Array2::<f64>::zeros((beta_dim, beta_dim));
            let mut rhs_acc = Array1::<f64>::zeros(beta_dim);
            let mut gb_acc = Array1::<f64>::zeros(beta_dim);
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
                // carries the minibatch-scaled β-penalty gradient).
                for j in 0..beta_dim {
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
            for j in 0..beta_dim {
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
            for j in 0..beta_dim {
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
            let mut trial_step = step_size;
            let mut accepted_loss: Option<SaeManifoldLoss> = None;
            for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
                let mut trial_beta = beta0.clone();
                for j in 0..beta_dim {
                    trial_beta[j] += trial_step * delta_beta[j];
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
            let chunk = self.materialize_chunk(logits, coords)?;
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
            let chunk = self.materialize_chunk(logits, coords)?;
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
            },
            total,
        ))
    }

    pub fn run_single_external_basis_refresh_step_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeManifoldLoss, String> {
        self.advance_temperature_schedule()?;
        // ρ is owned by the outer engine and held fixed across this single
        // external-basis-refresh Newton step; no in-loop ARD update.
        let pre_step_loss = self.loss(target, rho)?;
        let (delta_ext_coord, delta_beta) = self
            .solve_newton_step(target, rho, analytic_penalties, ridge_ext_coord, ridge_beta)
            .map_err(|err| {
                format!(
                    "SaeManifoldTerm::run_single_external_basis_refresh_step_arrow_schur: {err}"
                )
            })?;
        self.apply_newton_step_external_basis_refresh(
            delta_ext_coord.view(),
            delta_beta.view(),
            step_size,
        )?;
        Ok(pre_step_loss)
    }

    /// Build the analytic-penalty descriptors that correspond to the current
    /// SAE term. This is the bridge into `analytic_penalties.rs` for callers
    /// that want to register the same ρ axes with a REML driver.
    pub fn analytic_penalty_descriptors(&self) -> (AnalyticPenaltyKind, Vec<ARDPenalty>) {
        let assignment = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
                    SoftmaxAssignmentSparsityPenalty::new(self.k_atoms(), temperature),
                ))
            }
            AssignmentMode::IBPMap {
                temperature,
                alpha,
                learnable_alpha,
            } => {
                let penalty =
                    IBPAssignmentPenalty::new(self.k_atoms(), alpha, temperature, learnable_alpha);
                let penalty = match self.temperature_schedule.clone() {
                    Some(schedule) => penalty.with_temperature_schedule(schedule),
                    None => penalty,
                };
                AnalyticPenaltyKind::IBPAssignment(Arc::new(penalty))
            }
            AssignmentMode::JumpReLU { .. } => {
                // SAFETY: `analytic_penalty_descriptors` is only called for
                // assignment modes that have a corresponding REML descriptor
                // (Softmax, IBPMap). JumpReLU is handled by the built-in
                // gated-L1 assignment prior and never reaches this bridge —
                // callers must dispatch on `self.assignment.mode` first. The
                // panic guards against a future caller forgetting to do so.
                panic!(
                    "JumpReLU assignment mode uses the built-in gated L1 assignment prior and has no AnalyticPenaltyKind descriptor"
                )
            }
        };
        let mut ard = Vec::with_capacity(self.k_atoms());
        for coord in &self.assignment.coords {
            ard.push(ARDPenalty::new(
                PsiSlice::full(coord.len(), Some(coord.latent_dim())),
                coord.latent_dim(),
            ));
        }
        (assignment, ard)
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
        term: SaeManifoldTerm,
        target: Array2<f64>,
        registry: Option<AnalyticPenaltyRegistry>,
        init_rho: SaeManifoldRho,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Self {
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
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss,
            ..
        } = self;
        let loss = last_loss.unwrap_or_else(|| SaeManifoldLoss {
            data_fit: 0.0,
            assignment_sparsity: 0.0,
            smoothness: 0.0,
            ard: 0.0,
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
            term.penalized_objective_total(target.view(), &current_rho, registry.as_ref(), 1.0);
        let mut rho_seed = current_rho.clone();
        let seed_solve = baseline_term.run_joint_fit_arrow_schur(
            target.view(),
            &mut rho_seed,
            registry.as_ref(),
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        );
        if let (Ok(settled_total), Ok(seed_loss)) = (settled_objective, seed_solve) {
            let seed_total = baseline_term.penalized_objective_total(
                target.view(),
                &current_rho,
                registry.as_ref(),
                1.0,
            );
            if let Ok(seed_total) = seed_total {
                if seed_total.is_finite() && seed_total < settled_total {
                    return (baseline_term, current_rho, seed_loss);
                }
            }
        }
        (term, current_rho, loss)
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
    pub fn decoder_shape_uncertainty(&mut self) -> Result<SaeShapeUncertainty, String> {
        let rho = self.current_rho.clone();
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

    /// Shared cost path: evaluate the REML criterion at `rho_flat`, updating
    /// the cached ρ / loss and (optionally) priming the inner solve from a
    /// seeded β. Returns `(cost, β̂)`.
    fn evaluate(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), String> {
        let rho = self.baseline_rho.from_flat(rho_flat);
        if let Some(beta) = self.seeded_beta.take() {
            // Warm-start the inner decoder coefficients before the solve.
            if beta.len() == self.term.beta_dim() {
                self.term.set_flat_beta(beta.view())?;
            }
        }
        let (cost, loss) = self.term.reml_criterion(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.current_rho = rho;
        self.last_loss = Some(loss);
        let beta_hat = self.term.flatten_beta();
        Ok((cost, beta_hat))
    }

    /// Fellner-Schall / Mackay multiplicative fixed-point step on ρ at
    /// `rho_flat`. Runs the inner `(t, β)` solve to convergence at fixed ρ
    /// (sharing the single Direct factor with the REML criterion), then
    /// returns `(cost, additive-log-steps, β̂)`.
    ///
    /// All ρ coords are log-quantities, so the engine's additive step
    /// `rho_new = rho + step` IS the multiplicative FS update. Per coord:
    /// - ARD axis (k,j): `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`,
    ///   `step = ln α_new − log_ard[k][j]`. The `tr_kj(H⁻¹)` posterior
    ///   variance (from the selected-inverse latent diagonal) is exactly the
    ///   term the deleted `α=n/‖t‖²` rule dropped, so α cannot collapse on a
    ///   degenerate axis: as `‖t‖²→0`, `tr_kj(H⁻¹)→1/α` bounds the
    ///   denominator and the fixed point has a finite root.
    /// - λ_smooth: `λ_new = [p·Σ_k rank S_k − tr(S_β⁻¹ M)] / βᵀ(⊕S_k⊗I_p)β`
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
        // λ_new = (penalty_rank − effective_dof) / penalty_energy. The
        // numerator is the unpenalised-direction count; guard the FS ratio
        // against a vanishing penalty energy or a non-positive numerator
        // (which can occur transiently far from the optimum) by holding
        // λ_smooth fixed (step 0) — the cost path still moves it then.
        if quad > 0.0 && rank_total - eff_dof > 0.0 && lambda_smooth > 0.0 {
            let lambda_new = (rank_total - eff_dof) / quad;
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
                    let alpha_new = n_obs / denom;
                    if alpha_new.is_finite() && alpha_new > 0.0 {
                        steps[cursor + j] = alpha_new.ln() - axis_logard[j];
                    }
                }
            }
            cursor += d;
        }

        let beta_hat = self.term.flatten_beta();
        Ok(EfsEval {
            cost,
            steps,
            beta: Some(beta_hat),
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
        })
    }
}

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // ρ are all penalty-like / τ coordinates: precisions and
            // log-smoothing strengths. No design-moving ψ coordinates.
            psi_dim: 0,
            // EFS fixed-point lane is the right driver for these penalty-like
            // coords: the multiplicative Fellner-Schall/Mackay step is O(1)
            // selected-inverse trace per outer iter, vs the cost-only path's
            // O(K³) dense Schur per cost eval × many derivative-free evals —
            // intractable at biobank K. `eval_efs` implements it.
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluate(rho.view())
            .map(|(cost, _beta)| cost)
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let n_params = self.baseline_rho.to_flat().len();
        let (cost, beta_hat) = self
            .evaluate(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)?;
        Ok(OuterEval {
            cost,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: Some(beta_hat),
        })
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

fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut out = Array1::<f64>::zeros(k);
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

fn validate_finite_logits(logits: ArrayView1<'_, f64>, row: usize) -> Result<(), String> {
    for (col, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "SaeAssignment: non-finite assignment logit at row {row}, atom {col}: {v}"
            ));
        }
    }
    Ok(())
}

fn canonicalize_softmax_logits(logits: &mut Array2<f64>) {
    let k = logits.ncols();
    if k == 0 {
        return;
    }
    if k == 1 {
        logits.fill(0.0);
        return;
    }
    for row in 0..logits.nrows() {
        let reference = logits[[row, k - 1]];
        for col in 0..k - 1 {
            logits[[row, col]] -= reference;
        }
        logits[[row, k - 1]] = 0.0;
    }
}

/// Deterministic ordered geometric-shrinkage MAP weights
/// `π_k = (α/(α+1))^k` for k = 0, .., K-1, with the first atom intentionally
/// left unshrunk (`π_0 = 1`, the always-available base atom). This is NOT a
/// sampled or variational Indian-Buffet-Process posterior: it is a fixed,
/// deterministic per-atom shrinkage schedule that biases assignment mass to
/// decay geometrically with atom index even when logits are tied. `α` is a
/// shrinkage rate (larger `α` ⇒ slower decay), not an IBP concentration in the
/// sampling sense. The geometric form coincides with the prior means of a
/// Beta(α, 1) stick-breaking construction, which is the motivation for the
/// schedule, but no sticks are drawn here.
fn ibp_stick_breaking_prior(k_atoms: usize, alpha: f64) -> Array1<f64> {
    // Accumulate the geometric schedule `π_k = ratio^k` in LOG space so the
    // prior stays a finite *soft* weight even for large `K`. The naive product
    // `acc *= ratio` underflows to exact `0.0` once `ratio^k < f64::MIN_POSITIVE`
    // (e.g. `(0.1/1.1)^320`), which would turn the soft shrinkage prior into a
    // HARD mask: such atoms would receive zero assignment AND zero logit
    // gradient (the gradient is multiplied by `π_k`), so they could never
    // reactivate. Working in log-space and flooring the exponentiated weight at
    // the smallest positive normal keeps every atom's gradient path alive while
    // preserving the geometric ordering.
    let mut out = Array1::<f64>::zeros(k_atoms);
    let log_ratio = (alpha / (alpha + 1.0)).ln();
    for k in 0..k_atoms {
        let log_pi = (k as f64) * log_ratio;
        out[k] = log_pi.exp().max(f64::MIN_POSITIVE);
    }
    out
}

/// IBP-MAP row activations: per-atom sigmoid likelihood times the truncated
/// stick-breaking prior mass. With tied logits the prior dominates and yields
/// strictly decreasing activations in atom index.
fn ibp_map_row(logits: ArrayView1<'_, f64>, temperature: f64, alpha: f64) -> Array1<f64> {
    let prior = ibp_stick_breaking_prior(logits.len(), alpha);
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = crate::linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
    }
    out
}

/// IBP-MAP concrete relaxation activations together with the diagonal Jacobian
/// `∂z_k/∂l_k`, for the torch autograd `Function` to consume so that torch's
/// IBP-Gumbel forward applies the same stick-breaking prior `π_k` and
/// temperature scaling as the Rust closed-form path
/// (`SaeAssignment::try_assignments_row` → [`ibp_map_row`]).
///
/// With `z_k = σ(l_k/τ) · π_k` the per-atom derivative is
/// `∂z_k/∂l_k = σ(l_k/τ) (1 − σ(l_k/τ)) · π_k / τ`. The map is diagonal in `k`
/// (each activation depends only on its own logit), so the Jacobian is returned
/// as the per-atom diagonal vector.
#[must_use]
pub fn ibp_map_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let prior = ibp_stick_breaking_prior(logits.len(), alpha);
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = crate::linalg::utils::stable_logistic(logits[i] * inv_tau);
        value[i] = sig * prior[i];
        grad[i] = sig * (1.0 - sig) * inv_tau * prior[i];
    }
    (value, grad)
}

fn jumprelu_row(logits: ArrayView1<'_, f64>, temperature: f64, threshold: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        // Hard gate: strictly zero below threshold (the intended "jump"). Above
        // threshold the surrogate is centered at the threshold so the gate is
        // most informative exactly at the boundary it switches on:
        // σ((l−θ)/τ) ∈ [0.5, 1). Magnitude lives in the decoder, so the gate
        // stays bounded in [0, 1] by design.
        if logits[i] > threshold {
            out[i] = crate::linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
        }
    }
    out
}

struct ActiveAtomLogitJvp<'a> {
    mode: AssignmentMode,
    k: usize,
    logit_k: f64,
    a_k: f64,
    decoded_k: ArrayView1<'a, f64>,
    fitted: ArrayView1<'a, f64>,
    ibp_prior: Option<&'a [f64]>,
    compact_index: usize,
}

/// Fill the single compact logit-JVP row for active atom `k`, using the
/// per-mode assignment sensitivity `da_k/dl_k` contracted into the decoded /
/// fitted-corrected output direction. This is the active-set analogue of
/// [`fill_assignment_logit_jvp_rows`]: it reproduces that function's diagonal
/// logit row exactly for the atom `k`, but writes into a compact position of a
/// heterogeneous-`q` row block instead of the dense full-`K` Jacobian. `fitted`
/// is the row's *active-set* reconstruction so the softmax cross term
/// `(decoded_k − fitted)` is consistent with the curvature the compact block
/// carries.
fn fill_active_atom_logit_jvp(input: ActiveAtomLogitJvp<'_>, jac_compact: &mut Array2<f64>) {
    let ActiveAtomLogitJvp {
        mode,
        k,
        logit_k,
        a_k,
        decoded_k,
        fitted,
        ibp_prior,
        compact_index,
    } = input;
    let p = fitted.len();
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            // da_k/dl_k contracted: a_k (decoded_k − fitted) / τ.
            let inv_tau = 1.0 / temperature;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] =
                    a_k * (decoded_k[out_col] - fitted[out_col]) * inv_tau;
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // z_k = σ(l_k/τ)·π_k ⇒ dz_k/dl_k = a_k(π_k − a_k)/(π_k τ) · π_k form
            // (matches `fill_assignment_logit_jvp_rows`).
            let inv_tau = 1.0 / temperature;
            let prior =
                ibp_prior.expect("fill_active_atom_logit_jvp: IBPMap requires precomputed prior");
            let pi_k = prior[k];
            let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
            let dz = sig * (1.0 - sig) * inv_tau * pi_k;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = dz * decoded_k[out_col];
            }
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // The data-fit Jacobian follows the hard forward gate. Below the
            // threshold the reconstruction contribution is exactly zero, so the
            // data-fit logit derivative must also be zero. Band-only atoms stay
            // in the compact row for prior terms, not phantom reconstruction
            // slope.
            if logit_k <= threshold {
                return;
            }
            let inv_tau = 1.0 / temperature;
            let activation = crate::linalg::utils::stable_logistic((logit_k - threshold) * inv_tau);
            let da = activation * (1.0 - activation) * inv_tau;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = da * decoded_k[out_col];
            }
        }
    }
}

fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    ibp_prior: Option<&[f64]>,
    local_jac: &mut Array2<f64>,
) {
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            if assignments.len() == 1 {
                return;
            }
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row. The dense row layout uses
            // the reference-logit chart, so only columns `0..K-1` are free;
            // the final reference logit is fixed at zero and has no row.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() - 1 {
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = assignments[logit_col]
                        * (decoded[[logit_col, out_col]] - fitted[out_col])
                        * inv_tau;
                }
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // Truncated-IBP concrete relaxation: z_k = σ(l_k/τ) · π_k where
            // π_k is the stick-breaking prior. Thus
            // dz_k/dl_k = σ(l/τ)(1-σ(l/τ))/τ · π_k = a_k(π_k - a_k)/(π_k τ).
            let inv_tau = 1.0 / temperature;
            let prior = ibp_prior
                .expect("fill_assignment_logit_jvp_rows: IBPMap requires precomputed prior");
            for logit_col in 0..assignments.len() {
                let pi_k = prior[logit_col];
                let a_k = assignments[logit_col];
                let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
                let dz = sig * (1.0 - sig) * inv_tau * pi_k;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = dz * decoded[[logit_col, out_col]];
                }
            }
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Data-fit sensitivity follows the hard forward gate: rows at or
            // below the threshold have zero reconstruction value and therefore
            // zero data-fit logit derivative. The reactivation band is a
            // compact-layout/prior support rule, not a data-fit STE.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if logits[logit_col] <= threshold {
                    continue;
                }
                let activation = crate::linalg::utils::stable_logistic(
                    (logits[logit_col] - threshold) * inv_tau,
                );
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
    }
}

fn flat_logits(logits: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for row in 0..logits.nrows() {
        let start = row * logits.ncols();
        for col in 0..logits.ncols() {
            out[start + col] = logits[[row, col]];
        }
    }
    out
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

fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return 0.0;
    }
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                Array1::zeros(0)
            };
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Sparsity penalty uses the same threshold-centered surrogate and
            // reactivation-band support as its gradient. This makes band
            // prior gradients part of the objective evaluated by line search,
            // while data-fit reconstruction remains hard-gated by
            // `jumprelu_row`.
            let sparsity_strength = rho.lambda_sparse();
            let mut acc = 0.0;
            for &logit in target.iter() {
                if jumprelu_in_optimization_band(logit, threshold, temperature) {
                    acc += crate::linalg::utils::stable_logistic((logit - threshold) / temperature);
                }
            }
            sparsity_strength * acc
        }
    }
}

fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut grad = Array1::<f64>::zeros(target.len());
    let mut diag = Array1::<f64>::zeros(target.len());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok((grad, diag));
    }
    let (sparsity_grad, sparsity_diag) = match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            // Scale the IBP assignment-sparsity prior by `lambda_sparse`, exactly
            // like the Softmax and JumpReLU branches do (Softmax folds it into the
            // penalty's rho coordinate, JumpReLU multiplies `sparsity_strength`).
            // Previously the IBP penalty used its hardcoded `weight = 1.0` and the
            // `rho.log_lambda_sparse` coordinate never reached it (the rho_view was
            // empty for the common `learnable_alpha = false` config), so the prior
            // ran at full strength with no way to dial it down — and its
            // Beta-Bernoulli BCE energy `−mass·ln π_k − (n−mass)·ln(1−π_k)` toward
            // the self-referential empirical active fraction `π_k` has its global
            // minimum at the all-off gate, so at full weight it over-shrank the
            // assignment off both atoms even with a truth-seeded decoder (#853).
            // Routing `lambda_sparse` into the penalty weight makes the prior a
            // genuine, user-controllable lever balanced against the data fit.
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            // When `alpha` is learnable, `log_lambda_sparse` already modulates
            // it through `resolved_alpha(rho)`, so the weight stays 1.0 to avoid
            // double-counting that coordinate. Only when `alpha` is fixed (so the
            // sparse coordinate would otherwise be ignored entirely) does
            // `lambda_sparse` become the prior's weight lever.
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                penalty.weight = rho.lambda_sparse();
                Array1::zeros(0)
            };
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "IBP assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Gradient of the sparsity value's threshold-centered surrogate
            // σ((l−θ)/τ). Support extends through the reactivation band
            // (logit > θ − MARGIN·τ) so a gated-off atom near the boundary keeps
            // prior gradient. Data-fit JVP support is narrower and follows the
            // hard forward gate.
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                g[idx] = sparsity_strength * slope * inv_tau;
                d[idx] = sparsity_strength * slope * slope * inv_tau2;
            }
            (g, d)
        }
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    Ok((grad, diag))
}

fn sae_penalty_is_row_block_supported(penalty: &AnalyticPenaltyKind) -> bool {
    matches!(
        penalty,
        AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::IBPAssignment(_)
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
        "softmax_assignment_sparsity",
        "ibp_assignment",
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
    use super::*;
    use crate::terms::analytic_penalties::IsometryReference;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
            usize::MAX / 4,
            1024 * 1024,
        );
        assert!(!dense_plan.streaming);
        let streaming_plan = sae_streaming_plan_from_budget(
            term0.n_obs(),
            total_basis,
            term0.k_atoms(),
            d_max,
            1,
            512,
        );
        assert!(streaming_plan.streaming);

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

        // The β curvature block (hbb) must be PSD (it is the Newton/PIRLS
        // curvature operator). Symmetric and non-negative diagonal at minimum.
        for i in 0..m * p {
            assert!(
                sys.hbb[[i, i]] >= -1.0e-9,
                "hbb diagonal must be non-negative (PSD majorizer); hbb[{i},{i}]={:.3e}",
                sys.hbb[[i, i]]
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
            all_blocks_match = all_blocks_match && coord_ok && decoder_ok;
            let coord_status = if coord_ok { "MATCH" } else { "MISMATCH" };
            let decoder_status = if decoder_ok { "MATCH" } else { "MISMATCH" };
            eprintln!(
                "{label}: base={base:.12e}; coord_gt={coord_status} max_rel={coord_rel:.6e} \
                 max_abs={coord_abs:.6e} worst_row={coord_idx} analytic={coord_an:.12e} \
                 fd={coord_fd:.12e}; decoder_gb={decoder_status} max_rel={decoder_rel:.6e} \
                 max_abs={decoder_abs:.6e} worst_beta={decoder_idx} analytic={decoder_an:.12e} \
                 fd={decoder_fd:.12e}",
                label = report.label,
                base = report.base_loss,
                coord_rel = report.coord.relative_error,
                coord_abs = report.coord.absolute_error,
                coord_idx = report.coord.index,
                coord_an = report.coord.analytic,
                coord_fd = report.coord.finite_difference,
                decoder_rel = report.decoder.relative_error,
                decoder_abs = report.decoder.absolute_error,
                decoder_idx = report.decoder.index,
                decoder_an = report.decoder.analytic,
                decoder_fd = report.decoder.finite_difference,
            );
        }
        assert!(
            all_blocks_match,
            "SAE assembled gradient does not match central FD of the penalized objective"
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
            all_blocks_match = all_blocks_match && coord_ok && decoder_ok;
            let coord_status = if coord_ok { "MATCH" } else { "MISMATCH" };
            let decoder_status = if decoder_ok { "MATCH" } else { "MISMATCH" };
            let line = format!(
                "{label}: base={base:.12e}; coord_gt={coord_status} max_rel={coord_rel:.6e} \
                 max_abs={coord_abs:.6e} worst_row={coord_idx} analytic={coord_an:.12e} \
                 fd={coord_fd:.12e}; decoder_gb={decoder_status} max_rel={decoder_rel:.6e} \
                 max_abs={decoder_abs:.6e} worst_beta={decoder_idx} analytic={decoder_an:.12e} \
                 fd={decoder_fd:.12e}",
                label = report.label,
                base = report.base_loss,
                coord_rel = report.coord.relative_error,
                coord_abs = report.coord.absolute_error,
                coord_idx = report.coord.index,
                coord_an = report.coord.analytic,
                coord_fd = report.coord.finite_difference,
                decoder_rel = report.decoder.relative_error,
                decoder_abs = report.decoder.absolute_error,
                decoder_idx = report.decoder.index,
                decoder_an = report.decoder.analytic,
                decoder_fd = report.decoder.finite_difference,
            );
            eprintln!("{line}");
        }
        assert!(
            all_blocks_match,
            "SAE d=1 assembled gradient does not match central finite difference"
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
    fn jumprelu_assignment_prior_hessian_diag_is_psd_over_logit_sweep() {
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
        for (idx, &entry) in diag.iter().enumerate() {
            let logit = logits[[idx / k, idx % k]];
            // Expected = JumpReLU gated majorizer with the threshold-centered
            // surrogate σ((l−θ)/τ), supported through the reactivation band
            // (logit > θ − MARGIN·τ) so gated-off atoms near the boundary keep
            // prior gradient. Softmax identifiability is handled by its
            // reference-logit chart, not by adding curvature to gate logits.
            let sparsity = if jumprelu_in_optimization_band(logit, threshold, temperature) {
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                sparsity_strength * slope * slope * inv_tau2
            } else {
                0.0
            };
            let expected = sparsity;
            assert!(
                entry.is_finite() && entry >= 0.0,
                "JumpReLU gated hessian_diag majorizer must be finite and non-negative at index \
                 {idx}; entry={entry}"
            );
            assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
        }
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
        // `diff = g − g_ref` where `g = penalty.pullback_metric(d)` is read from
        // `penalty`'s own Jacobian cache, and skips the third-jet `K` term only
        // when `diff == 0.0` (a bit-exact float compare). Previously `g_ref` was
        // built from a SEPARATE `scratch` penalty's cache, so a last-ULP
        // difference between the two independent refreshes left `diff` ~1e-16
        // rather than exactly 0; multiplied by the large third decoder jet
        // (`K ~ ω³`) for the torus/sphere bases, that leaked past the 1e-10
        // exact-equality bound. Refreshing `penalty` once and seeding the
        // UserSupplied reference from `penalty.pullback_metric(d)` makes
        // `g_ref` the identical array `g` is recomputed from, so the residual is
        // bit-zero and the K term is genuinely skipped — leaving exactly the GN
        // term. `with_reference` moves the penalty by value and preserves every
        // cache slot, so the J/J2/K caches read by the HVP are unchanged.
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
        let g_ref = penalty
            .pullback_metric(d)
            .expect("pullback metric is available after the cache refresh");
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

        // The Euclidean reference makes g − I nonzero on this non-orthonormal
        // decoder; verify the residual is real so the curvature seam is the
        // production one (and not vacuously the zero-residual case).
        let d = coords.ncols();
        let g = penalty
            .pullback_metric(d)
            .expect("pullback metric available after refresh");
        let mut residual_mass = 0.0_f64;
        for row in 0..g.nrows() {
            for a in 0..d {
                for b in 0..d {
                    // Euclidean reference is the identity metric I_d.
                    let g_ref = if a == b { 1.0 } else { 0.0 };
                    residual_mass += (g[[row, a * d + b]] - g_ref).abs();
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
