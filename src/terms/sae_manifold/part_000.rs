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
    DeviceSaeSmoothBlock, FactoredFrameGBlock, FactoredFrameKroneckerOp, IbpCrossRowSource,
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

/// Relative spectral cutoff below which a penalised decoder β-curvature
/// eigenvalue (`G_k + λ_smooth·S_k`) is treated as a genuine flat direction of
/// the joint inner Hessian — the rank-deficient-decoder null quotiented out of
/// the inner convergence measure and deflated in the outer gradient (#1051).
/// Matches the `1e-9` relative rank cutoff used across the codebase.
const SAE_DECODER_BETA_NULL_RELATIVE_FLOOR: f64 = 1.0e-9;


/// Largest decoder (`β`) block dimension for which the outer-gradient
/// conditioning path may additionally probe the β coordinate basis for a
/// near-null subspace of the joint Hessian (issue #1051, #1095).
///
/// The closed-form gauge orbit ([`SaeManifoldTerm::dense_step_gauge_vectors`])
/// only covers the *chart* reparametrisation freedom (constant + linear
/// coordinate fields). It does NOT cover a **rank-deficient decoder design** —
/// e.g. a euclidean-1D atom fit to a straight line in a `p = 2` ambient leaves
/// the decoder column space rank-1, so one decoder direction is unidentified by
/// the data and the joint Hessian acquires a near-null direction that lives in
/// the β block, not the gauge orbit. That direction is exactly a Faddeev-Popov
/// gauge of the *same* kind (a flat direction of the evidence quotient), so it
/// is deflated identically — but only after the β basis is admitted as a
/// deflation candidate. The dense `k×k` Rayleigh eigendecomposition that
/// resolves it is `O(k³)`, so it is gated to moderate β blocks; large-`p`
/// LLM-scale fits keep the pure gauge-orbit path untouched (they reach low
/// decoder rank through the Grassmann frame, which reduces the border width
/// from `M·p` to `M·r` where `r ≪ p`, so `k ≤ M·r` is always small). PCA-
/// reduced fits (p ≈ 32–128) with the Grassmann frame active can have
/// `k = M·r` up to ~512 (e.g. m=8 basis fns, p=32, r=8 → k=64, but for
/// m=16 → k=128, m=32 → k=256); 512 covers all typical small-atom PCA cases
/// while keeping the O(k³) cost ≈ 0.13B ops — negligible next to the solve.
const SAE_OUTER_GRADIENT_BETA_NULL_PROBE_MAX_DIM: usize = 512;


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


/// Row count at or above which the fused SAE reconstruction data-fit
/// (`loss_scaled`) fans its per-row decode + residual reduction out over
/// rayon. Below this the single-threaded fused pass is cheaper than the
/// fan-out; matched in spirit to the arrow-Schur `SCHUR_MATVEC_PARALLEL_ROW_MIN`
/// gate so short batches inside an outer fan-out stay sequential (#1017).
const SAE_LOSS_PARALLEL_ROW_MIN: usize = 64;

/// Relative tolerance on the undamped Newton step norm (scaled by the iterate
/// scale) for accepting inner-solve convergence.
const SAE_MANIFOLD_INNER_STEP_REL_TOL: f64 = 1.0e-4;


/// Relative tolerance on the KKT gradient norm (scaled by the iterate scale) for
/// accepting inner-solve convergence.
const SAE_MANIFOLD_INNER_GRAD_REL_TOL: f64 = 1.0e-5;


/// Relative per-refine-round penalised-objective decrease below which the inner
/// solve is treated as having reached its numerical fixed point (#1051). On an
/// ill-conditioned penalised bilinear fit the KKT gradient and undamped step
/// stay above tolerance while the objective stops moving; this `√εmach`-scale
/// floor recognises that stalled iterate as the converged inner optimum instead
/// of grinding the refine budget to the `1e12` infeasible sentinel.
const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL: f64 = 1.0e-8;


/// Fraction of the total since-entry objective reduction below which a refine
/// round's contribution is treated as cosmetic flat-valley crawl (#1051), so the
/// inner solve is accepted as numerically converged. At `1e-4` the inner fit has
/// captured ≥ 99.99% of the achievable penalised-objective reduction before the
/// criterion is ranked — far past the point where further crawl can change the
/// Laplace evidence, yet strict enough that a materially-improving fit refines on.
const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_FRACTION: f64 = 1.0e-4;


/// Minimum completed refine rounds before the objective-stagnation fixed point
/// may be accepted (#1051). Enough rounds to establish a meaningful
/// total-improvement baseline for the fraction test, but far below the full
/// refine budget — terminating the ill-conditioned crawl early is the goal.
const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS: usize = 3;


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
    /// Hyperbolic (Poincaré-ball) tangent patch at unit curvature `c = −1`.
    ///
    /// Shares the monomial decoder design of [`Self::EuclideanPatch`] — the
    /// latent coordinate `t` is read as a tangent vector at the ball origin
    /// (the wrapped / tangent parameterisation) and the decoder is the same
    /// polynomial-in-`t` expansion — but its smoothness penalty is the
    /// conformal-reweighted Dirichlet energy of the Poincaré metric
    /// (`refresh_intrinsic_smooth_penalty` measures wiggle in *hyperbolic*
    /// arc length via the `λ(p)` conformal factor). This makes an atom whose
    /// feature density grows toward the ball boundary (exponential-volume /
    /// tree-leaf hierarchy) the regime where it differs from the flat patch.
    Poincare,
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
            // Poincaré tangent patch: the latent `t` is a tangent vector at the
            // ball origin, optimised in the unconstrained tangent chart (the
            // hyperbolic geometry enters through the penalty, not a constrained
            // retraction), so it shares the Euclidean latent manifold.
            Self::Duchon
            | Self::EuclideanPatch
            | Self::Poincare
            | Self::Precomputed(_) => LatentManifold::Euclidean,
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
            // The tangent latent of a Poincaré patch lies in the convex hull of
            // its PCA seed exactly like the Euclidean patch, so no compact
            // projection grid is needed.
            Self::Duchon | Self::EuclideanPatch | Self::Poincare | Self::Precomputed(_) => None,
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


/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.2 polynomial for the
/// *exponentially-scaled* `I0`: `√x · e^{−x} · I0(x) ≈ poly(3.75/x)`. Factoring
/// the `e^{x}/√x` envelope out lets the log-partition and the `I1/I0` ratio be
/// computed without ever materialising `e^{x}` (which overflows to `+inf` for
/// `x ≳ 709`, see [`bessel_i0_log_and_ratio`]).
fn bessel_i0_scaled_poly(ax: f64) -> f64 {
    let y = 3.75 / ax;
    0.39894228
        + y * (0.01328592
            + y * (0.00225319
                + y * (-0.00157565
                    + y * (0.00916281
                        + y * (-0.02057706
                            + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))))
}


/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.4 polynomial for the
/// *exponentially-scaled* `I1`: `√x · e^{−x} · I1(x) ≈ poly(3.75/x)`. Pairs with
/// [`bessel_i0_scaled_poly`] so their shared `e^{x}/√x` envelope cancels exactly
/// in the `I1/I0` ratio.
fn bessel_i1_scaled_poly(ax: f64) -> f64 {
    let y = 3.75 / ax;
    0.39894228
        + y * (-0.03988024
            + y * (-0.00362018
                + y * (0.00163801
                    + y * (-0.01031555
                        + y * (0.02282967
                            + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059)))))))
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
        (ax.exp() / ax.sqrt()) * bessel_i0_scaled_poly(ax)
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
        (ax.exp() / ax.sqrt()) * bessel_i1_scaled_poly(ax)
    };
    if x < 0.0 { -value } else { value }
}


/// Overflow-free `(log I0(η), I1(η)/I0(η))` for `η >= 0`, the only two Bessel
/// quantities the von-Mises ARD precision normaliser and its ρ-gradient need.
///
/// The naive `bessel_i0(η).ln()` and `bessel_i1(η)/bessel_i0(η)` both route
/// through `e^{η}/√η`, which overflows to `+inf` once `η ≳ 709`. Two `+inf`s
/// then divide to `NaN`, poisoning the very first outer ρ-gradient on
/// large-norm / ill-conditioned checkpoints (issue #1113: a dispersion-inflated
/// ARD seed pushes `η = α/κ²` past the overflow threshold at iter 0). For a
/// periodic circle atom (`κ = 2π`) this fires for any seed precision
/// `α ≳ 2.8e4`, well inside the reachable seed range.
///
/// We never form `e^{η}`. For the small branch (`η < 3.75`) the A&S series are
/// finite, so we evaluate them directly. For the large branch the shared
/// `e^{η}/√η` envelope cancels in the *log* (`log I0 = η − ½ ln η + ln poly`)
/// and in the *ratio* (`I1/I0 = poly₁/poly₀`), so both are computed from the
/// bounded scaled polynomials alone — exact for non-degenerate η and finite for
/// every finite η.
fn bessel_i0_log_and_ratio(eta: f64) -> (f64, f64) {
    let ax = eta.abs();
    if ax < 3.75 {
        let i0 = bessel_i0(ax);
        let i1 = bessel_i1(ax);
        (i0.ln(), i1 / i0)
    } else {
        let poly0 = bessel_i0_scaled_poly(ax);
        let poly1 = bessel_i1_scaled_poly(ax);
        let log_i0 = ax - 0.5 * ax.ln() + poly0.ln();
        let ratio = poly1 / poly0;
        (log_i0, ratio)
    }
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
        // Poincaré tangent patch: measure the decoded speed per unit of
        // *hyperbolic* latent length rather than flat tangent length. A unit
        // step in the tangent coordinate `t` covers hyperbolic distance
        // `λ(p(t))` (the conformal factor at the ball point `p = exp₀(t)`), so
        // the arc-length speed is `‖J‖ / λ` and the squared speed picks up a
        // `1/λ²` factor. For the monomial patch (`d = 1`) the tangent coordinate
        // is the linear monomial column (`Φ = [1, t, …]`, so column 1 is `t`).
        let hyperbolic = matches!(self.basis_kind, SaeAtomBasisKind::Poincare);
        let linear_col = if hyperbolic && m >= 2 { Some(1usize) } else { None };
        for row in 0..n {
            self.fill_decoded_derivative_row(row, 0, &mut deriv);
            let mut speed_sq = 0.0_f64;
            for &d in deriv.iter() {
                speed_sq += d * d;
            }
            if let Some(col) = linear_col {
                let t = self.basis_values[[row, col]];
                // p = exp₀(t) at unit curvature c = −1: ‖p‖ = tanh(|t|), and
                // λ(p) = 2 / (1 − ‖p‖²) = 2 / (1 − tanh²|t|) = 2·cosh²(t).
                // speed_sq ← speed_sq / λ².  (cosh is even, so the sign of t
                // does not matter.)
                let lambda = 2.0 * t.cosh() * t.cosh();
                if lambda.is_finite() && lambda > 0.0 {
                    speed_sq /= lambda * lambda;
                }
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
    /// Per-atom Riesz-debiased smooth-functional inference and Bartlett smooth
    /// significance (#1097 / #1103), one entry per fitted atom in atom order.
    /// Each entry's `functionals` / `smooth_significance` are `Some` only when
    /// the atom's inner-decoder smooth was harvested at fit time (the caller ran
    /// [`SaeManifoldTerm::set_atom_inner_fits`] and the inner penalized Hessian
    /// was SPD on a non-empty active set); otherwise they degrade to `None`.
    pub atom_inference: Vec<crate::sae_identifiability::AtomInferenceReport>,
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
    atom_curvature_bound_with_decoder(
        atom,
        atom_idx,
        second.view(),
        atom.decoder_coefficients.view(),
    )
}

/// The sup-norm extrinsic-curvature bound `atom_curvature_bound` as an explicit
/// function of the decoder coefficient matrix `decoder` (shape `(M_k, p)`) and
/// the precomputed second jet, so the #1099 delta-method gradient `∂κ/∂β` can be
/// formed by finite-differencing it in the captured channel's coefficients
/// without mutating the term. With `decoder = atom.decoder_coefficients` this is
/// exactly `atom_curvature_bound`.
fn atom_curvature_bound_with_decoder(
    atom: &SaeManifoldAtom,
    atom_idx: usize,
    second: ArrayView4<'_, f64>,
    decoder: ArrayView2<'_, f64>,
) -> Result<f64, String> {
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
    if decoder.dim() != (m, p) {
        return Err(format!(
            "atom_curvature_bound: atom {atom_idx} decoder shape {:?} must be ({m}, {p})",
            decoder.dim()
        ));
    }
    let mut max_kappa = 0.0_f64;
    let mut tangent = Array2::<f64>::zeros((p, d));
    let mut second_vec = vec![0.0_f64; p];
    for row in 0..n {
        // Tangent J(t) = Φ'(t) B on this row, formed from the explicit decoder.
        tangent.fill(0.0);
        for basis_col in 0..m {
            for axis in 0..d {
                let dphi = atom.basis_jacobian[[row, basis_col, axis]];
                if dphi == 0.0 {
                    continue;
                }
                for out in 0..p {
                    tangent[[out, axis]] += dphi * decoder[[basis_col, out]];
                }
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
                        second_vec[out] += h * decoder[[basis_col, out]];
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
    /// #1026: the load-bearing curved-vs-linear hybrid-split verdict, computed
    /// once in [`Self::canonicalize_charts_post_fit`] after the joint fit
    /// converges. Each eligible `d = 1` atom's fitted curved image is adjudicated
    /// against its straight (linear special-case) sub-model on the common
    /// rank-aware Laplace evidence scale. `None` until the post-fit pass runs (or
    /// when no atom is eligible). Surfaced in the Python model output so a user
    /// sees which atoms genuinely earn their curvature and which collapse to the
    /// linear tail. Read via [`Self::hybrid_split_report`].
    hybrid_split_report: Option<crate::terms::sae::hybrid_split::SaeHybridSplitReport>,
    /// Per-atom inner-decoder-smooth byproducts harvested post-fit (#1097 /
    /// #1103), one entry per atom in [`Self::atoms`] order. Each is the fixed
    /// fitted snapshot the residual-gauge certificate's three post-PIRLS atom
    /// inference reports consume
    /// ([`crate::sae_identifiability::AtomInnerFit`]). `None` until
    /// [`Self::set_atom_inner_fits`] runs (it needs the reconstruction target
    /// `Z`, available only at the post-fit harness seam where the dispersion is
    /// also profiled); a per-atom `None` means that atom had no active rows or a
    /// degenerate inner design. Read by [`Self::to_residual_gauge_model`], which
    /// attaches each onto its [`crate::sae_identifiability::FittedAtom`].
    atom_inner_fits: Option<Vec<Option<crate::sae_identifiability::AtomInnerFit>>>,
    /// #1117 deep fix: per-atom data-null **range reduction** projectors
    /// `Π_k = N_k N_kᵀ` for any decoder atom whose bare data Gram `G_k` is
    /// rank-deficient at the spectral cutoff, one entry per [`Self::atoms`].
    /// `Some(Π_k)` ⇒ atom `k` has a dead column subspace that is deflated
    /// (unit-stiffness `⊗ I_p`) in the β-tier penalty operator and projected out
    /// of the converged decoder; `None` ⇒ the full-rank atom keeps the historical
    /// full-`B` β-tier bit-for-bit. Refreshed once per inner fit by
    /// [`Self::data_null_decoder_projectors`] (held FIXED across the inner Newton
    /// iterations so the deflated operator the solve descends and the undamped
    /// log-det rank the SAME quotient), and read by `assemble_arrow_schur`. Empty
    /// (`vec![]`) outside an inner fit, which assembly reads as "no deflation".
    decoder_data_null_projectors: Vec<Option<Array2<f64>>>,
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
            hybrid_split_report: self.hybrid_split_report.clone(),
            atom_inner_fits: self.atom_inner_fits.clone(),
            decoder_data_null_projectors: self.decoder_data_null_projectors.clone(),
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
