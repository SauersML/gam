use gam_terms::construction::ReparamResult;
use crate::estimate::EstimationError;
use gam_linalg::matrix::{
    DesignMatrix, PsdWeightsView, ReparamOperator, SignedWeightsView, SymmetricMatrix,
};
use crate::active_set::ConstraintKktDiagnostics;
use gam_problem::{Coefficients, GlmLikelihoodSpec, InverseLink, LinearPredictor, RidgePassport};
use gam_problem::LinearInequalityConstraints;
use ndarray::{ArcArray1, Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{compute_observed_hessian_curvature_arrays, computeworkingweight_derivatives_from_eta};

/// Whether the solve operates in sparse-native or dense-transformed coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirlsLinearSolvePath {
    DenseTransformed,
    SparseNative,
}

/// Coordinate frame for the PIRLS inner iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirlsCoordinateFrame {
    TransformedQs,
    OriginalSparseNative,
}

/// Firth bias-reduction diagnostics at convergence.
#[derive(Debug, Clone, Default)]
pub enum FirthDiagnostics {
    #[default]
    Inactive,
    Active {
        jeffreys_logdet: f64,
        hat_diag: Array1<f64>,
    },
}

impl FirthDiagnostics {
    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        match self {
            Self::Inactive => None,
            Self::Active {
                jeffreys_logdet, ..
            } => Some(*jeffreys_logdet),
        }
    }
}

/// Which information matrix the penalized Hessian carries at the current
/// PIRLS iterate.
///
/// Canonical links (logit-Binomial, log-Poisson) have W_obs == W_Fisher, so
/// the two choices coincide. Non-canonical links (probit, cloglog, mixture,
/// flexible, Gamma-log, ...) need observed information W_obs = W_Fisher -
/// (y - mu) * B for the outer REML/Laplace log|H| and trace terms to be
/// exact; Fisher weights alone yield a PQL-type surrogate. We fall back to
/// `Fisher` only when the observed-information Hessian fails the
/// positive-definiteness check, since the inner Newton step must be SPD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HessianCurvatureKind {
    /// Expected (Fisher) information: W_Fisher = h'^2 / (phi * V(mu)).
    /// Used as the inner iteration matrix when observed curvature fails (non-SPD).
    Fisher,
    /// Observed information: W_obs = W_Fisher - (y - mu) * B.
    /// Required for the outer REML log|H| and trace terms (exact Laplace).
    Observed,
}

/// The exported Laplace curvature kind used for the outer REML criterion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportedLaplaceCurvature {
    ObservedExact,
    ExpectedInformationSurrogate,
    InvalidObservedCurvature {
        min_eigenvalue: f64,
        pd_tolerance: f64,
        gradient_norm: f64,
    },
}

/// Working state at a PIRLS iterate: gradient, Hessian, deviance, etc.
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: LinearPredictor,
    pub gradient: Array1<f64>,
    pub hessian: gam_linalg::matrix::SymmetricMatrix,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub penalty_term: f64,
    pub firth: FirthDiagnostics,
    // Ridge added to ensure positive definiteness of the penalized Hessian.
    // `penalty_term` stores the full quadratic form contribution
    // ridge * ||beta||^2. The optimization objective uses
    // 0.5 * (deviance + penalty_term), so this corresponds to
    // 0.5 * ridge * ||beta||^2 on the log-likelihood scale.
    pub ridge_used: f64,
    pub hessian_curvature: HessianCurvatureKind,
    // Natural scale of the penalized gradient, used to form a scale-invariant
    // KKT certificate.  Equal to ||X'(weighted_residual)||_2 + ||S*beta||_2
    // (+ ridge*||beta||_2 when a stabilizing ridge is active).  Under
    // stochastic noise the score component scales as O(sqrt(n)), so an
    // absolute ||g||_2 < tol test rejects fits whose normalized stationarity
    // residual is already negligible. Convergence uses ||g||_2 / (1 + this).
    pub gradient_natural_scale: f64,
}

impl WorkingState {
    /// Value minimized by PIRLS for this fully evaluated state.
    #[inline]
    pub fn penalized_objective(&self) -> f64 {
        0.5 * (self.deviance + self.penalty_term)
    }

    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }

    /// Scale-invariant relative gradient residual.
    ///
    /// Returns ||g||_2 / (1 + ||score||_2 + ||S*beta||_2 + ridge*||beta||_2).
    /// `g_norm` is the projected/constrained stationarity residual in the
    /// current PIRLS basis; the denominator is the natural magnitude of the
    /// penalized gradient and is invariant under uniform rescaling of the
    /// objective.
    #[inline]
    pub fn relative_gradient_norm(&self, g_norm: f64) -> f64 {
        g_norm / (1.0 + self.gradient_natural_scale)
    }

    /// Dimension-based scale `√n · max(1, √p)` for the structural KKT bound.
    ///
    /// Under standardized columns, the score `Xᵀ(μ − y)` has components of
    /// order O(√n), so the absolute test ‖g‖ < τ becomes systematically too
    /// tight at large n. Multiplying τ by this scale restores the advertised
    /// per-observation meaning.
    #[inline]
    pub(crate) fn kkt_dimension_scale(&self) -> f64 {
        let n = self.eta.len().max(1) as f64;
        let p = (self.gradient.len() as f64).max(1.0);
        n.sqrt() * p.sqrt()
    }

    /// Strict KKT acceptance: `g_norm` certifies stationarity under EITHER
    /// scale-invariant criterion (dimension-based or data-driven natural-scale).
    ///
    /// Both certificates are invariant under uniform rescaling of the objective
    /// `F → c·F` (in the limit where the natural scale dominates the additive
    /// `1` floor). Acceptance under either is sufficient because:
    ///   - the natural-scale bound is tighter when the data are well-scaled
    ///     (it tracks actual gradient component magnitudes);
    ///   - the dimension bound is tighter when the design matrix has unusual
    ///     scaling (so the natural scale is dominated by a single component).
    #[inline]
    pub fn certifies_kkt(&self, g_norm: f64, tol: f64) -> bool {
        g_norm < tol * self.kkt_dimension_scale() || self.relative_gradient_norm(g_norm) < tol
    }

    /// Near-stationary band (10× the strict KKT tolerance) under EITHER
    /// scale-invariant criterion. Used as a "good-enough" plateau check
    /// that classifies a fit as `StalledAtValidMinimum` rather than as a
    /// hard non-convergence. The band is `10 · tol` without a
    /// floor — a caller asking for `tol = 1e-12` gets a 1e-11 band, not
    /// the 1e-5 the old `tol.max(1e-6) * 10` formula silently widened it
    /// to. The 1e-6 floor was masking real convergence regressions
    /// (e.g. `constant_prior_mean_centers_penalty`'s LM-ridge induced
    /// 2.5e-8 bias visible only when the user asked for sub-1e-6
    /// precision).
    #[inline]
    pub fn near_stationary_kkt(&self, g_norm: f64, tol: f64) -> bool {
        let near_tol = tol * 10.0;
        g_norm <= near_tol * self.kkt_dimension_scale()
            || self.relative_gradient_norm(g_norm) <= near_tol
    }
}

/// Numerically stable Euclidean norm of an `Array1<f64>`.
///
/// Used to assemble the penalized-gradient natural scale at every
/// `WorkingState` construction site (main GAM, identity-link short circuit,
/// survival, test mocks). Centralizing here avoids drift between sites and
/// makes the convergence certificate's denominator a single source of truth.
///
/// One pass, no allocation, O(p). At p≈10⁴ the cost is ≪ the O(np²) PIRLS
/// inner work, so this is free in any setting where it matters.
#[inline]
pub fn array1_l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Adaptive KKT tolerance parameters for the inner PIRLS convergence test.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveKktTolerance {
    pub eta: f64,
    pub floor: f64,
    pub ceiling: f64,
    pub outer_grad_norm: f64,
}

/// Per-iteration PIRLS diagnostic info reported to the callback.
#[derive(Clone, Debug)]
pub struct WorkingModelIterationInfo {
    pub iteration: usize,
    pub deviance: f64,
    pub gradient_norm: f64,
    pub step_size: f64,
    pub step_halving: usize,
}

/// Result of the inner `runworking_model_pirls` loop.
#[derive(Clone)]
pub struct WorkingModelPirlsResult {
    pub beta: Coefficients,
    pub state: WorkingState,
    pub status: PirlsStatus,
    pub iterations: usize,
    pub lastgradient_norm: f64,
    pub last_deviance_change: f64,
    pub last_step_size: f64,
    pub last_step_halving: usize,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<ConstraintKktDiagnostics>,
    /// Levenberg-Marquardt damping coefficient at the last accepted
    /// inner iter. Used by the REML runtime to seed the next PIRLS call
    /// at the same outer fit, avoiding 4-6 iters of damping rediscovery
    /// when the geometry calls for `λ_LM > 1e-6`.
    pub final_lm_lambda: f64,
    /// Gain ratio (`actual_reduction / predicted_reduction`) at the
    /// last accepted inner iter. `None` when no step was accepted
    /// (rejection-exhausted, MaxIterationsReached without acceptance).
    /// Programmatic counterpart to the per-iter `[PIRLS lm-trajectory]`
    /// log line's `accept_rho` field — the log is grep-only, this
    /// field is queryable by the outer schedule and convergence guard.
    /// Values near 1.0 indicate the quadratic model is faithful;
    /// values much smaller indicate the LM model is over-stating
    /// predicted reduction and the inner Newton may benefit from
    /// shorter steps.
    pub final_accept_rho: Option<f64>,
    /// Minimum penalized deviance (`state.deviance + state.penalty_term`)
    /// observed across all iterations whose state was computed during the
    /// inner P-IRLS loop. Penalized deviance is monotonically decreasing
    /// along any descent path the inner solver takes, so this minimum is a
    /// principled seed-screening proxy that remains meaningful even when the
    /// solver hit its iteration cap before reaching the mode. `f64::INFINITY`
    /// when no state was ever computed (paths that synthesize a result
    /// without iterating, e.g. zero-iteration warm-only paths).
    pub min_penalized_deviance: f64,
    pub exported_laplace_curvature: ExportedLaplaceCurvature,
}

/// The status of the P-IRLS convergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PirlsStatus {
    /// Converged successfully within tolerance.
    Converged,
    /// Reached maximum iterations but the gradient and Hessian indicate a valid minimum.
    StalledAtValidMinimum,
    /// Reached maximum iterations without converging.
    MaxIterationsReached,
    /// Levenberg-Marquardt step search exhausted its retry budget (damping λ
    /// reached its ceiling, attempts counter expired, or λ went non-finite)
    /// before the projected gradient entered the near-stationary band. Distinct
    /// from `MaxIterationsReached`, which means the outer iteration counter
    /// itself ran out — that exhaustion is a "looped 100×, made progress each
    /// time but never converged" signal, while this one is a "no acceptable
    /// step direction even after damping" signal pointing at curvature trouble
    /// or saturated likelihoods.
    LmStepSearchExhausted,
    /// Fitting process became unstable, likely due to perfect separation.
    Unstable,
}

impl PirlsStatus {
    /// Whether the inner loop concluded without producing a usable mode.
    /// Both the iteration-cap and LM-exhausted exits should be treated the
    /// same by callers that just want to know "did we get a valid solution?".
    #[inline]
    pub const fn is_failed_max_iterations(self) -> bool {
        matches!(
            self,
            PirlsStatus::MaxIterationsReached | PirlsStatus::LmStepSearchExhausted
        )
    }

    /// Short human-readable label for reports and diagnostics. Stable text
    /// (not the `Debug` rendering) so report output does not silently change if
    /// the variant identifiers are ever renamed.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            PirlsStatus::Converged => "Converged",
            PirlsStatus::StalledAtValidMinimum => "Stalled at valid minimum",
            PirlsStatus::MaxIterationsReached => "Max iterations reached",
            PirlsStatus::LmStepSearchExhausted => "LM step search exhausted",
            PirlsStatus::Unstable => "Unstable (possible separation)",
        }
    }

    /// Whether this status represents a clean convergence to the mode. Only
    /// `Converged` qualifies; every other state carries a caveat a reader
    /// should see flagged.
    #[inline]
    pub const fn is_converged(self) -> bool {
        matches!(self, PirlsStatus::Converged)
    }

    /// Whether the inner mode is a **certified valid minimum** — one carrying a
    /// quantitative KKT stationarity certificate — that is acceptable for
    /// minting a fit. Two statuses qualify:
    ///
    /// * `Converged` — the strict certificate (both the step-size floor and the
    ///   projected-gradient KKT band were met).
    /// * `StalledAtValidMinimum` — the iteration counter (or the LM step search)
    ///   was exhausted, but the accepted iterate is still a genuine minimum: it
    ///   is assigned *only* when [`PirlsState::near_stationary_kkt`] holds (the
    ///   projected gradient is inside the 10×-widened KKT band) and the deviance
    ///   plateaued or the step collapsed to its floor. It is not a bare label —
    ///   it is the honest "valid minimum, but the strict step tolerance was not
    ///   reached before the cap" terminal state, and every REML consumer (the
    ///   Tweedie-φ / Gamma-shape freezing gates, the GPU dispatch wire) already
    ///   treats it as equivalent to `Converged`.
    ///
    /// Every other terminal status (`MaxIterationsReached`,
    /// `LmStepSearchExhausted`, `Unstable`) reports an honest non-convergence
    /// with no valid-minimum certificate and does **not** qualify.
    ///
    /// This is the fit-minting gate shared by the standard/GAMLSS
    /// `UnifiedFitResult` constructor and the negative-binomial joint (θ, ρ)
    /// optimizer. It is deliberately distinct from [`Self::is_converged`], the
    /// *strict* certificate (only `Converged`) required by survival fits — which
    /// keep their own pre-gate that excludes `StalledAtValidMinimum` — and by
    /// the outer REML driver's per-iteration progress checks.
    #[inline]
    pub const fn certifies_valid_minimum(self) -> bool {
        matches!(
            self,
            PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
        )
    }
}

/// Holds the result of a converged P-IRLS inner loop for a fixed rho.
///
/// # Basis of Returned Tensors
///
/// **IMPORTANT:** All vector and matrix outputs in this struct (`beta_transformed`,
/// `penalized_hessian_transformed`) are in the **stable, transformed basis**
/// that was computed for the given set of smoothing parameters.
///
/// To obtain coefficients in the original, interpretable basis, the caller must
/// back-transform them using the `qs` matrix from the `reparam_result` field:
/// `beta_original = reparam_result.qs.dot(&beta_transformed)`
///
/// # Fields
///
/// * `beta_transformed`: The estimated coefficient vector in the STABLE, TRANSFORMED basis.
/// * `penalized_hessian_transformed`: The penalized Hessian matrix at convergence
///   (`X'W_H X + S_λ`, with `W_H` equal to Fisher or observed curvature,
///   depending on the accepted PIRLS step) in the STABLE, TRANSFORMED basis.
/// * `deviance`: The final deviance value. This is family-specific:
///    - Gaussian identity: weighted residual sum of squares.
///    - Binomial families: binomial deviance.
///    - Poisson log: Poisson deviance.
///    - Gamma log: Gamma unit deviance scaled by the fitted Gamma shape.
/// * `finalweights`: The final Hessian-side working weights at convergence.
/// * `solveweights`: The final score-side Fisher weights used in
///   `X'W(z-eta) - S beta`.
/// * `reparam_result`: Contains the transformation matrix (`qs`) and other reparameterization data.
///
/// # Point Estimate: Posterior Mode (MAP)
///
/// The coefficients returned by PIRLS are the **posterior mode** (Maximum A Posteriori estimate),
/// not the posterior mean. For risk predictions, the posterior mean is theoretically preferable
/// mode ≈ mean and it doesn't matter. For asymmetric posteriors (rare events, boundary effects),
/// the mean would give more accurate calibrated probabilities. To obtain the posterior mean,
/// one would need MCMC sampling from the posterior and average f(patient, β) over samples.
#[derive(Clone)]
pub struct PirlsResult {
    pub likelihood: GlmLikelihoodSpec,
    // Coefficients and Hessian are now in the STABLE, TRANSFORMED basis
    pub beta_transformed: Coefficients,
    pub penalized_hessian_transformed: SymmetricMatrix,
    // Single stabilized Hessian for consistent cost/gradient computation
    pub stabilizedhessian_transformed: SymmetricMatrix,
    /// Canonical ridge metadata passport consumed by outer objective/gradient code.
    pub ridge_passport: RidgePassport,
    // Ridge added to make the stabilized Hessian positive definite. When > 0,
    // `stable_penalty_term` includes ridge_used * ||beta||^2 (which contributes
    // 0.5 * ridge_used * ||beta||^2 in -0.5 * (deviance + stable_penalty_term)).
    // Backward-compatible mirror of `ridge_passport.delta`.
    pub ridge_used: f64,

    // The unpenalized deviance, calculated from mu and y
    pub deviance: f64,

    // Effective degrees of freedom at the solution
    pub edf: f64,

    // The penalty term, calculated stably within P-IRLS.
    // This is beta_transformed' * S_transformed * beta_transformed, plus
    // ridge_used * ||beta||^2 when stabilization is active so that the
    // penalized deviance matches the stabilized Hessian.
    pub stable_penalty_term: f64,

    /// Firth diagnostics in the converged PIRLS state.
    pub firth: FirthDiagnostics,

    // Diagonal weights defining the Hessian surface returned to outer REML/LAML.
    //
    // For canonical links Fisher = Observed identically. For non-canonical links,
    // PIRLS always recomputes observed weights at the accepted β̂ in a
    // post-convergence finalization step (see "Post-convergence Laplace curvature
    // finalization"), so `finalweights` carries the *observed-information* diagonal
    // whenever the model supports it — even if the inner LM loop ended on Fisher
    // due to a fallback. Exact label of what these represent is in
    // `exported_laplace_curvature`; do not infer the kind from `hessian_curvature`
    // (which records what the inner loop's last accepted step happened to use).
    // #1868: the length-`n` row fields are `ArcArray1` (reference-counted
    // ndarray, O(1) clone) so the n-free κ-trial skip path can SHARE the
    // once-built frozen row bundle across every trial instead of
    // re-materialising these placeholders per callback. On the exact path they
    // are built owned and moved into the shared representation via
    // `.into_shared()` (O(1) — no element copy). `ArcArray1` is an `ArrayBase`,
    // so reads (indexing, iteration, `.dot`, `&a - &b`, `.len`, `.view`) work
    // unchanged; only sites needing an owned `Array1`/`&Array1` take
    // `.to_owned()`/`.view()`.
    pub finalweights: ArcArray1<f64>,
    // Additional PIRLS state captured at the accepted step to support
    // cost/gradient consistency in the outer optimization
    pub final_offset: ArcArray1<f64>,
    pub final_eta: ArcArray1<f64>,
    pub finalmu: ArcArray1<f64>,
    /// Score-side Fisher weights used in `X'W(z-eta) - S beta`.
    pub solveweights: ArcArray1<f64>,
    pub solveworking_response: ArcArray1<f64>,
    pub solvemu: ArcArray1<f64>,
    pub solve_dmu_deta: ArcArray1<f64>,
    pub solve_d2mu_deta2: ArcArray1<f64>,
    pub solve_d3mu_deta3: ArcArray1<f64>,
    /// First eta-derivative of the diagonal Hessian curvature W_H(eta):
    /// c_i := dW_i/deta_i at the accepted PIRLS solution.
    ///
    /// This carries 3rd-order likelihood information used in exact dH/dρ
    /// terms for outer LAML derivatives.
    pub solve_c_array: ArcArray1<f64>,
    /// Second eta-derivative of the diagonal Hessian curvature W_H(eta):
    /// d_i := d²W_i/deta_i² at the accepted PIRLS solution.
    ///
    /// This carries 4th-order likelihood information used in exact d²H/dρ²
    /// terms for the outer LAML Hessian.
    pub solve_d_array: ArcArray1<f64>,
    /// True when `solve_c_array` / `solve_d_array` are placeholders rather
    /// than supported likelihood derivatives.
    pub derivatives_unsupported: bool,

    // Keep all other fields as they are
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
    pub lastgradient_norm: f64,
    /// Natural scale of the penalized gradient at the accepted PIRLS state,
    /// equal to ‖Xᵀ(weighted residual)‖₂ + ‖Sβ‖₂ (+ ridge·‖β‖₂ when active).
    /// Mirrors `WorkingState::gradient_natural_scale` so that callers reading
    /// `PirlsResult` directly (e.g. seed-screening cost augmentation) can form
    /// the scale-invariant residual r_g = ‖g‖ / (1 + this) without rebuilding
    /// the score and penalty norms.
    pub gradient_natural_scale: f64,
    /// Penalized inner KKT residual `r = ∇_β L_pen(β̂) = Sβ̂ − ∇ℓ(β̂) (+ridge·β̂)`
    /// at the accepted P-IRLS iterate, in the STABLE/TRANSFORMED coefficient
    /// basis (the same frame as `beta_transformed` and the transformed penalized
    /// Hessian). This is the exact vector whose L2 norm `lastgradient_norm`
    /// records (see `WorkingState::gradient`, assembled as `Xᵀ(η−z)·w + Sβ`,
    /// which equals `Sβ − ∇ℓ` because `Xᵀ(η−z)·w = −∇ℓ`). Storing the vector —
    /// not just its norm — lets the outer REML/LAML evaluator engage the
    /// inner-KKT envelope correction `Ṽ = V − ½·rᵀH⁻¹r` on the flexible-link
    /// (SAS/mixture) path, where the outer optimizer accepts β̂ at a first-order
    /// inner cap (`outer_inner_cap`) short of exact stationarity. The correction
    /// and its θ-gradient vanish as `r → 0`, so a fully-converged fit is
    /// unchanged. See [`crate::model_types::ProjectedKktResidual`].
    pub penalized_gradient_transformed: Array1<f64>,
    pub last_deviance_change: f64,
    pub last_step_halving: usize,
    pub hessian_curvature: HessianCurvatureKind,
    pub exported_laplace_curvature: ExportedLaplaceCurvature,
    /// Levenberg-Marquardt damping coefficient at the converged inner
    /// iter. Cached by the REML runtime so the next PIRLS call in the
    /// same outer optimization can seed `λ_LM` to this value instead
    /// of cold-starting at `1e-6`. Mirrors `WorkingModelPirlsResult::final_lm_lambda`.
    pub final_lm_lambda: f64,
    /// Gain ratio of the last accepted LM step inside this PIRLS solve,
    /// `None` when no step was accepted (e.g. zero-iteration synthesis,
    /// rejection-exhausted, MaxIterations without acceptance). Mirrors
    /// `WorkingModelPirlsResult::final_accept_rho`. Programmatic
    /// counterpart to the per-iter `[PIRLS lm-trajectory]` log line's
    /// `accept_rho` field, queryable by outer consumers (cap schedule,
    /// convergence guard) for inner-Newton model-fidelity decisions.
    pub final_accept_rho: Option<f64>,
    /// Optional KKT diagnostics when inequality constraints were active.
    pub constraint_kkt: Option<ConstraintKktDiagnostics>,
    /// Linear inequality system enforced in transformed PIRLS coordinates:
    /// `A * beta_transformed >= b`.
    pub linear_constraints_transformed: Option<LinearInequalityConstraints>,

    // Pass through the entire reparameterization result for use in the gradient
    pub reparam_result: ReparamResult,
    // Cached X·Qs for this PIRLS result (transformed design matrix)
    pub x_transformed: DesignMatrix,
    pub coordinate_frame: PirlsCoordinateFrame,
    /// True when this fixed-rho inner solve completed on a GPU path.
    pub used_device: bool,
    /// True when this result was compacted for REML LRU storage and needs
    /// cold artifacts (for example `x_transformed`) rehydrated before exact
    /// bundle construction.
    pub cache_compacted: bool,
    /// Minimum penalized deviance observed across the inner P-IRLS loop.
    /// Mirrors `WorkingModelPirlsResult::min_penalized_deviance`. Used as the
    /// seed-screening ranking proxy: penalized deviance descends monotonically
    /// along any inner descent path, so the per-seed minimum tells the outer
    /// cascade "how good a fit this rho's neighbourhood can support" even
    /// when the inner solver was capped before reaching the mode.
    pub min_penalized_deviance: f64,
}

impl PirlsResult {
    /// Export the stabilized transformed Hessian as an exact dense matrix for
    /// downstream solve paths that require explicit Hessians.
    ///
    /// The returned matrix is the convergence Hessian already used by PIRLS and
    /// REML (`X'W_HX + S_λ`, plus the explicit stabilization ridge when active).
    /// Sparse-native fits are materialized from their assembled sparse Hessian;
    /// no numerical Hessian approximation or compatibility fallback is used.
    pub fn dense_stabilizedhessian_transformed(
        &self,
        context: &str,
    ) -> Result<Array2<f64>, EstimationError> {
        self.stabilizedhessian_transformed
            .try_to_dense_exact(context)
            .map_err(EstimationError::InvalidInput)
    }

    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }

    /// Typed view of the Hessian-side working weight diagonal stored on this
    /// result, sign-honest. `finalweights` carries the observed-information
    /// diagonal whenever the model supports it (see `exported_laplace_curvature`),
    /// and observed weights `W_obs = W_F - (y - μ) · B` can be negative for
    /// non-canonical links. Consumers feeding this into the asymmetric
    /// `X_iᵀ W X_j` path, `weighted_crossprod_dense_rows`, or
    /// `xt_diag_x_signed_op` must use this typed view rather than borrowing
    /// the raw `Array1<f64>` so the function-boundary type contract from
    /// `linalg/matrix.rs` is construction-enforced.
    #[inline]
    pub fn final_weights_signed(&self) -> SignedWeightsView<'_> {
        SignedWeightsView::new(self.finalweights.view())
    }

    /// Typed view of the score-side Fisher weights `W_F = h'²/(φ V(μ)) ≥ 0`
    /// stored on this result, PSD-by-construction. Used by PSD-Gram kernels
    /// (`dense_xtwx_view`, `sparse_csr_weighted_xtwx_*`, `xt_diag_x_psd_op`)
    /// without a runtime sign scan; the PSD obligation is discharged
    /// algebraically by the Fisher formula at the construction site in
    /// `solver/pirls/mod.rs`. New callers that need the same diagonal under
    /// a sign-honest API should route through `as_signed()` on the returned
    /// view rather than reconstructing from the raw array.
    #[inline]
    pub fn solve_weights_psd(&self) -> PsdWeightsView<'_> {
        PsdWeightsView::from_view_unchecked(self.solveweights.view())
    }

    /// Scale-invariant relative gradient residual at the accepted PIRLS state.
    ///
    /// Returns ‖g‖ / (1 + ‖score‖ + ‖Sβ‖ + ridge·‖β‖). Numerator is
    /// `lastgradient_norm`; denominator is `1 + gradient_natural_scale`.
    /// This is the "r_g" used by seed-screening cost augmentation.
    #[inline]
    pub fn relative_gradient_norm(&self) -> f64 {
        self.lastgradient_norm / (1.0 + self.gradient_natural_scale)
    }

    pub(crate) fn compact_for_reml_cache(&self) -> Self {
        Self {
            likelihood: self.likelihood.clone(),
            beta_transformed: self.beta_transformed.clone(),
            penalized_hessian_transformed: self.penalized_hessian_transformed.clone(),
            stabilizedhessian_transformed: self.stabilizedhessian_transformed.clone(),
            ridge_passport: self.ridge_passport,
            ridge_used: self.ridge_used,
            deviance: self.deviance,
            edf: self.edf,
            stable_penalty_term: self.stable_penalty_term,
            firth: self.firth.clone(),
            finalweights: ArcArray1::zeros(0),
            final_offset: ArcArray1::zeros(0),
            final_eta: self.final_eta.clone(),
            finalmu: ArcArray1::zeros(0),
            solveweights: self.solveweights.clone(),
            solveworking_response: self.solveworking_response.clone(),
            solvemu: self.solvemu.clone(),
            solve_dmu_deta: ArcArray1::zeros(0),
            solve_d2mu_deta2: ArcArray1::zeros(0),
            solve_d3mu_deta3: ArcArray1::zeros(0),
            solve_c_array: self.solve_c_array.clone(),
            solve_d_array: self.solve_d_array.clone(),
            derivatives_unsupported: self.derivatives_unsupported,
            status: self.status,
            iteration: self.iteration,
            max_abs_eta: self.max_abs_eta,
            lastgradient_norm: self.lastgradient_norm,
            gradient_natural_scale: self.gradient_natural_scale,
            // Length-p vector; carried across compaction/rehydration so the
            // inner-KKT envelope correction survives an LRU round-trip without
            // rebuilding the score from the (dropped) transformed design.
            penalized_gradient_transformed: self.penalized_gradient_transformed.clone(),
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            exported_laplace_curvature: self.exported_laplace_curvature.clone(),
            final_lm_lambda: self.final_lm_lambda,
            final_accept_rho: self.final_accept_rho,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((0, 0)),
            )),
            coordinate_frame: self.coordinate_frame,
            used_device: self.used_device,
            cache_compacted: true,
            min_penalized_deviance: self.min_penalized_deviance,
        }
    }

    pub(crate) fn rehydrate_after_reml_cache(
        &self,
        x_original: &DesignMatrix,
        y: ArrayView1<'_, f64>,
        priorweights: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        inverse_link: &InverseLink,
    ) -> Result<Self, EstimationError> {
        if !self.cache_compacted {
            return Ok(self.clone());
        }

        // #1868: cold LRU rehydration path — materialise the compacted rows from
        // the frozen link/derivatives and re-wrap into the shared `ArcArray1`
        // fields (`.into()`, O(1) once owned).
        let final_eta_owned = self.final_eta.to_owned();
        let (score_c_array, score_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                &self.likelihood,
                inverse_link,
                &final_eta_owned,
                priorweights,
            )?;
        let (finalweights, solve_c_array, solve_d_array): (
            ArcArray1<f64>,
            ArcArray1<f64>,
            ArcArray1<f64>,
        ) = if self.hessian_curvature == HessianCurvatureKind::Observed {
            let (fw, sc, sd) = compute_observed_hessian_curvature_arrays(
                &self.likelihood,
                inverse_link,
                &final_eta_owned,
                y,
                &self.solveweights.to_owned(),
                priorweights,
            )?;
            (fw.into(), sc.into(), sd.into())
        } else {
            (
                self.solveweights.clone(),
                score_c_array.clone().into(),
                score_d_array.clone().into(),
            )
        };
        // Lazy rehydration: wrap in ReparamOperator instead of materializing X·Qs.
        let qs_arc = Arc::new(self.reparam_result.qs.clone());
        Ok(Self {
            likelihood: self.likelihood.clone(),
            beta_transformed: self.beta_transformed.clone(),
            penalized_hessian_transformed: self.penalized_hessian_transformed.clone(),
            stabilizedhessian_transformed: self.stabilizedhessian_transformed.clone(),
            ridge_passport: self.ridge_passport,
            ridge_used: self.ridge_used,
            used_device: self.used_device,
            deviance: self.deviance,
            edf: self.edf,
            stable_penalty_term: self.stable_penalty_term,
            firth: self.firth.clone(),
            finalweights,
            final_offset: offset.to_owned().into(),
            final_eta: self.final_eta.clone(),
            finalmu: self.solvemu.clone(),
            solveweights: self.solveweights.clone(),
            solveworking_response: self.solveworking_response.clone(),
            solvemu: self.solvemu.clone(),
            solve_dmu_deta: solve_dmu_deta.into(),
            solve_d2mu_deta2: solve_d2mu_deta2.into(),
            solve_d3mu_deta3: solve_d3mu_deta3.into(),
            solve_c_array,
            solve_d_array,
            derivatives_unsupported: self.derivatives_unsupported,
            status: self.status,
            iteration: self.iteration,
            max_abs_eta: self.max_abs_eta,
            lastgradient_norm: self.lastgradient_norm,
            gradient_natural_scale: self.gradient_natural_scale,
            // Length-p vector; carried across compaction/rehydration so the
            // inner-KKT envelope correction survives an LRU round-trip without
            // rebuilding the score from the (dropped) transformed design.
            penalized_gradient_transformed: self.penalized_gradient_transformed.clone(),
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            exported_laplace_curvature: self.exported_laplace_curvature.clone(),
            final_lm_lambda: self.final_lm_lambda,
            final_accept_rho: self.final_accept_rho,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
                ReparamOperator::new(x_original.clone(), qs_arc),
            ))),
            coordinate_frame: self.coordinate_frame,
            cache_compacted: false,
            min_penalized_deviance: self.min_penalized_deviance,
        })
    }
}
