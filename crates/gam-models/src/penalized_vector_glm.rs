//! Generic penalized vector-response GLM Newton solver (fixed λ).
//!
//! This is the shared scaffold extracted from
//! [`crate::multinomial::fit_penalized_multinomial`] (dense softmax
//! Fisher block) and
//! [`crate::binomial_multi::fit_penalized_binomial_multi`]
//! (row-diagonal independent-binomial Fisher block). Both families fit a
//! penalized vector-response GLM with a shared design `X ∈ ℝ^{N×P}` and a
//! shared penalty `S ∈ ℝ^{P×P}` replicated per output, differing **only** in
//! the per-row Fisher-block algebra and the likelihood/residual. Everything
//! else — input validation, penalized objective / gradient / Hessian assembly,
//! damped Newton with backtracking, the relative-step convergence test, and the
//! final penalized-objective / deviance tally — is written once here.
//!
//! # Fit problem
//!
//! With `β = [β_0; β_1; …; β_{M-1}]` stacked in output-major order
//! (`β_a ∈ ℝ^P` is the coefficient block for output `a`), minimise the
//! penalized negative log-likelihood
//!
//! ```text
//!   F(β) = − log L(β) + ½ Σ_{a=0}^{M-1} λ_a · β_aᵀ S β_a
//! ```
//!
//! where `log L` and its η-derivatives are supplied by the family's
//! [`VectorLikelihood`] adapter and `λ_a` is a per-output smoothing parameter
//! scaling the shared penalty `S`. The active linear predictor is
//! `η_{n,a} = (X β_a)_n`, shape `(N, M)`.
//!
//! # Newton step
//!
//! Each iteration assembles the coupled penalized Hessian and gradient in
//! output-major coefficient ordering `flat[a·P + i] = β[i, a]` (matching
//! [`gam_solve::pirls::dense_block_xtwx`]):
//!
//! ```text
//!   H[a·P + i, b·P + j] = Σ_n W_{n,a,b} · X[n,i] · X[n,j]   (+ δ_{ab} λ_a S[i,j])
//!   g[a·P + i]          = Σ_n r_{n,a} · X[n,i]              (+ λ_a (S β_a)[i])
//! ```
//!
//! with the per-row Fisher block `W_{n,·,·} = −∂² log L / ∂η ∂η` (the family's
//! [`VectorLikelihood::hess_block`], or a caller override) and the residual
//! `r_{n,a} = −∂ log L / ∂η_a` (`−`[`VectorLikelihood::grad_eta`]). The step
//! `δ = − H^{-1} g` is solved through faer's symmetric-PD-with-fallback
//! factorisation under an adaptive Levenberg–Marquardt ridge: when a
//! rank-deficient block (collinear / quasi-separated columns under a small
//! per-output λ) makes the Bunch–Kaufman fallback back-substitute through
//! near-zero pivots into a non-finite δ, a diagonal ridge `τ·I` — scaled by the
//! Hessian's largest diagonal so it is curvature-scale invariant — is added and
//! the system re-solved, escalating τ geometrically until δ is finite. The
//! step is then accepted by a backtracking line search on `F` (full step first,
//! halve up to 8 times). Because the line search validates against the
//! *unridged* objective `F`, the ridge never biases the converged β̂ (at the
//! optimum the gradient vanishes and δ → 0 for any τ). Convergence is the
//! relative coefficient step `‖δ‖ / (1 + ‖β‖) ≤ tol`.
//!
//! # Fisher-block override
//!
//! When `fisher_w_override` is `Some`, each Newton step uses the supplied
//! per-row `(N, M, M)` curvature block in place of the analytic
//! [`VectorLikelihood::hess_block`]; the gradient/residual path stays analytic
//! (issue #349). The two families differ in what they accept off the diagonal:
//! multinomial admits a full dense block, while independent-binomial columns
//! only consume the per-output diagonal (a non-zero cross term cannot be
//! represented by the separable columns). That family-specific precondition is
//! enforced by the adapter before it constructs the override view; the engine
//! consumes whatever block it is given.

use gam_linalg::faer_ndarray::{FaerArrayView, array2_to_matmut, factorize_symmetricwith_fallback};
use crate::vector_response::VectorLikelihood;
use crate::model_types::EstimationError;
use gam_solve::pirls::dense_block_xtwx;
use faer::Side;
use opt::{
    BacktrackConfig, RidgeSchedule, backtracking_line_search, escalate_ridge,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use std::convert::Infallible;

/// Base Levenberg–Marquardt ridge as a fraction of the penalized Hessian's
/// largest diagonal entry (so it is invariant to the problem's overall
/// curvature scale). At ~1e-10 of the dominant curvature it is negligible
/// relative to identified-direction curvature — it never biases the identified
/// optimum (at β̂ the unridged gradient still vanishes there) — yet large
/// enough to lift an exactly rank-deficient null direction off zero so the
/// Bunch–Kaufman fallback yields a finite, descent Newton step (gam#856).
const BASE_RIDGE_FRACTION_OF_MAX_DIAG: f64 = 1.0e-10;

/// Geometric ridge-escalation budget for a single Newton step. 30 doublings
/// span ~9 orders of magnitude over the base ridge, which covers any
/// conditioning a finite-curvature softmax/binomial block can present.
const MAX_RIDGE_ESCALATIONS: usize = 30;

/// Backtracking budget for the damped-Newton line search: full step first, then
/// halve up to this many times if the penalized objective fails to decrease.
const MAX_BACKTRACKS: usize = 8;

/// Per-step line-search contraction factor (halving).
const LINE_SEARCH_SHRINK: f64 = 0.5;

/// Slack on the "objective decreased" acceptance test, absorbing floating-point
/// round-off so a step that is flat to machine precision is not rejected.
const OBJECTIVE_DECREASE_SLACK: f64 = 1.0e-12;

/// First-order optimality gate (gam#856) as a fraction of `1 + max_diag`: the
/// unridged penalized gradient norm must fall below this curvature-scaled
/// threshold before convergence is declared, certifying stationarity on the
/// identified subspace rather than a premature step-norm stall.
const OPTIMALITY_GRAD_FRACTION: f64 = 1.0e-6;

/// Class-space metric of the replicated smoothing penalty (#1587).
///
/// * `Diagonal` — the historical `diag_a(λ_a) ⊗ S`: each active output's
///   coefficient block is penalised independently. Correct for genuinely
///   independent outputs (independent-binomial columns), but for a *softmax*
///   multinomial it penalises the reference-anchored log-odds contrasts
///   `η_a = log(p_a/p_ref)`, so the fit is NOT invariant to the arbitrary
///   reference-class choice (#1587).
/// * `Centered` — the reference-symmetric `λ · ((I_{M} − J_{M}/K) ⊗ S)` with a
///   single shared `λ` (= `lambdas[0]`; the caller must pass uniform `lambdas`)
///   and `K = M + 1`. This is exactly the symmetric CLR penalty
///   `Σ_{k=0}^{K-1} β̃_kᵀ S β̃_k` (with `Σ_k β̃_k = 0`) written in the active-class
///   (ALR) gauge — invariant to which class is the baseline (the multinomial
///   analogue of #1549's `G^{1/2}` Aitchison whitening). Couples the class
///   blocks via the `−(λ/K)·S` off-diagonals; the engine already factors a
///   class-coupled Hessian (the softmax Fisher block is dense), so this is a
///   penalty-assembly change only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassPenaltyMetric {
    /// Independent per-output penalty `diag_a(λ_a) ⊗ S` (historical default).
    #[default]
    Diagonal,
    /// Reference-symmetric centered penalty `λ·((I − J/K) ⊗ S)`, `K = M + 1`.
    Centered,
}

/// Inputs to [`fit_penalized_vector_glm`].
///
/// `M` (the number of active outputs / linear-predictor columns) is taken from
/// `lambdas.len()`; the engine validates it against the design and override
/// shapes. The response `y` is passed verbatim to the [`VectorLikelihood`]
/// adapter, which owns its own `(N, ·)` shape contract (binomial columns use
/// `K = M`; multinomial one-hot uses `K = M + 1`), so the engine does not
/// constrain its column count beyond `y.nrows() == N`.
pub struct PenalizedVectorGlmInputs<'a> {
    /// Design matrix `X ∈ ℝ^{N×P}` (one row per observation, shared across
    /// every output column).
    pub design: ArrayView2<'a, f64>,
    /// Response `Y ∈ ℝ^{N×·}`, interpreted by the [`VectorLikelihood`].
    pub y: ArrayView2<'a, f64>,
    /// Shared smoothing penalty `S ∈ ℝ^{P×P}` (symmetric, PSD).
    pub penalty: ArrayView2<'a, f64>,
    /// Per-output smoothing parameter `λ_a`, length `M`.
    pub lambdas: ArrayView1<'a, f64>,
    /// Optional per-row Fisher-block override, shape `(N, M, M)`. When `Some`,
    /// it replaces the analytic [`VectorLikelihood::hess_block`] as the Newton
    /// curvature; the gradient/residual path stays analytic (issue #349). The
    /// adapter is responsible for any family-specific structural precondition
    /// on the block (e.g. zero off-diagonals for independent columns).
    pub fisher_w_override: Option<ArrayView3<'a, f64>>,
    /// Maximum Newton iterations; recommend 50.
    pub max_iter: usize,
    /// Relative-step convergence tolerance; recommend 1e-7.
    pub tol: f64,
    /// Class-space metric of the replicated penalty (#1587). `Diagonal`
    /// preserves the historical independent-per-output penalty; `Centered`
    /// selects the reference-symmetric softmax penalty (requires uniform
    /// `lambdas`). See [`ClassPenaltyMetric`].
    pub class_penalty_metric: ClassPenaltyMetric,
}

/// Outputs of [`fit_penalized_vector_glm`].
pub struct PenalizedVectorGlmOutputs {
    /// Coefficient matrix, shape `(P, M)` (column `a` is `β_a`).
    pub coefficients: Array2<f64>,
    /// Final active linear predictor `η = X β̂`, shape `(N, M)`. The adapter
    /// turns this into fitted probabilities via its own inverse link.
    pub eta: Array2<f64>,
    /// Number of Newton iterations executed (including the final step that
    /// satisfied the tolerance).
    pub iterations: usize,
    /// `true` if the relative-step test was satisfied before `max_iter`.
    pub converged: bool,
    /// Unpenalized log-likelihood `log L(β̂)`.
    pub log_likelihood: f64,
    /// Penalty term `½ Σ_a λ_a · β̂_aᵀ S β̂_a` at the returned `β̂`.
    pub penalty_term: f64,
    /// Joint Laplace posterior coefficient covariance `H⁻¹` at the converged
    /// `β̂`, shape `(P·M)×(P·M)` (#1101). `H = block(XᵀWX) + diag_a(λ_a)⊗S` is
    /// the penalized Hessian the Newton loop already assembles and factors at
    /// every step, discarding the factor; here it is re-assembled once at the
    /// mode and inverted (solve against the identity through the same symmetric
    /// factorization used for the Newton step). Block-ordered to match the
    /// stacked coefficient vector `θ[a·P + i] = β̂[i, a]`, i.e.
    /// `β = [β_0; …; β_{M-1}]`. This is the covariance the predict / inference
    /// surface uses for delta-method standard errors and prediction intervals.
    pub coefficient_covariance: Array2<f64>,
}

/// Quadratic form `½ β_aᵀ S β_a` accumulated across outputs with per-output
/// weight `λ_a`. Shared by the objective evaluator and the final tally.
fn weighted_penalty_sum(
    beta: &Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    lambdas: ArrayView1<'_, f64>,
    metric: ClassPenaltyMetric,
) -> f64 {
    let (p, m) = beta.dim();
    match metric {
        ClassPenaltyMetric::Diagonal => {
            let mut pen = 0.0_f64;
            for a in 0..m {
                let la = lambdas[a];
                if la == 0.0 {
                    continue;
                }
                let beta_col = beta.column(a);
                let mut quad = 0.0_f64;
                for i in 0..p {
                    let mut s_beta_i = 0.0_f64;
                    for j in 0..p {
                        s_beta_i += penalty[[i, j]] * beta_col[j];
                    }
                    quad += beta_col[i] * s_beta_i;
                }
                pen += 0.5 * la * quad;
            }
            pen
        }
        // Centered (#1587): ½·λ·[ Σ_a β_aᵀSβ_a − (1/K)·gᵀSg ], g = Σ_a β_a,
        // K = M + 1. Equals the symmetric CLR penalty Σ_k β̃_kᵀSβ̃_k (Σβ̃=0) in
        // the active-class gauge — reference-invariant. Shared λ = lambdas[0].
        ClassPenaltyMetric::Centered => {
            if m == 0 {
                return 0.0;
            }
            let lam = lambdas[0];
            if lam == 0.0 {
                return 0.0;
            }
            let k = (m + 1) as f64;
            // g = Σ_a β_a (the active-class coefficient sum, a p-vector).
            let mut g = vec![0.0_f64; p];
            for a in 0..m {
                let col = beta.column(a);
                for i in 0..p {
                    g[i] += col[i];
                }
            }
            // Σ_a β_aᵀSβ_a.
            let mut sum_quad = 0.0_f64;
            for a in 0..m {
                let col = beta.column(a);
                for i in 0..p {
                    let mut s_beta_i = 0.0_f64;
                    for j in 0..p {
                        s_beta_i += penalty[[i, j]] * col[j];
                    }
                    sum_quad += col[i] * s_beta_i;
                }
            }
            // gᵀSg.
            let mut g_quad = 0.0_f64;
            for i in 0..p {
                let mut s_g_i = 0.0_f64;
                for j in 0..p {
                    s_g_i += penalty[[i, j]] * g[j];
                }
                g_quad += g[i] * s_g_i;
            }
            0.5 * lam * (sum_quad - g_quad / k)
        }
    }
}

/// Invert the symmetric penalized Hessian `H` to the joint Laplace covariance
/// `Σ = H⁻¹` by solving `H·Σ = I` through the shared symmetric factorization
/// (#1101). `dim` is the flat block dimension `P·M`; `context` prefixes any
/// diagnostic. A curvature-scaled Tikhonov ridge `τ·I` — floored at
/// [`BASE_RIDGE_FRACTION_OF_MAX_DIAG`]·max_diag and escalated geometrically up
/// to [`MAX_RIDGE_ESCALATIONS`] times — is added ONLY when the raw factor/solve
/// is non-finite (a rank-deficient null direction), exactly mirroring the
/// Newton step's ridge so the covariance is always finite; at full rank the
/// ridge is never engaged and `Σ` is the exact `H⁻¹`. The returned matrix is
/// symmetrized `(Σ + Σᵀ)/2` to null round-off asymmetry from the back-solve.
fn invert_symmetric_penalized_hessian(
    hessian: &Array2<f64>,
    dim: usize,
    context: &str,
) -> Result<Array2<f64>, EstimationError> {
    let max_diag = (0..dim).fold(0.0_f64, |acc, idx| acc.max(hessian[[idx, idx]].abs()));
    let base_ridge = if max_diag.is_finite() && max_diag > 0.0 {
        max_diag * BASE_RIDGE_FRACTION_OF_MAX_DIAG
    } else {
        BASE_RIDGE_FRACTION_OF_MAX_DIAG
    };
    // `last_failure` distinguishes the two exhaustion modes so their distinct
    // terminal errors survive the migration: `Some((ridge, err))` when the
    // final attempt died in the factorization, `None` when it factored but the
    // back-solve stayed non-finite.
    let mut last_failure: Option<(f64, String)> = None;
    let mut try_ridge = |ridge: f64| -> Option<Array2<f64>> {
        let mut ridged = hessian.clone();
        if ridge > 0.0 {
            for idx in 0..dim {
                ridged[[idx, idx]] += ridge;
            }
        }
        let factor = match factorize_symmetricwith_fallback(
            FaerArrayView::new(&ridged).as_ref(),
            Side::Lower,
        ) {
            Ok(factor) => factor,
            Err(err) => {
                last_failure = Some((ridge, err.to_string()));
                return None;
            }
        };
        // Solve H·Σ = I: identity RHS, back-solved in place to yield Σ = H⁻¹.
        let mut rhs = Array2::<f64>::eye(dim);
        {
            let rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view);
        }
        if !rhs.iter().all(|v| v.is_finite()) {
            last_failure = None;
            return None;
        }
        // Symmetrize to remove round-off asymmetry from the back-solve.
        let mut cov = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                cov[[i, j]] = 0.5 * (rhs[[i, j]] + rhs[[j, i]]);
            }
        }
        Some(cov)
    };
    // Bare (unridged) attempt first — at full rank the ridge is never engaged —
    // then the geometric escalation from `base_ridge` with the doubling growth
    // this site has always used.
    if let Some(cov) = try_ridge(0.0) {
        return Ok(cov);
    }
    match escalate_ridge(
        RidgeSchedule {
            initial: base_ridge,
            growth: 2.0,
            max_escalations: MAX_RIDGE_ESCALATIONS,
        },
        &mut try_ridge,
    ) {
        Ok(success) => Ok(success.value),
        Err(_) => match last_failure {
            Some((ridge, err)) => Err(EstimationError::InvalidInput(format!(
                "{context}: covariance factorization failed even with ridge \
                 {ridge:.3e}: {err}"
            ))),
            None => Err(EstimationError::InvalidInput(format!(
                "{context}: covariance solve remained non-finite after {} ridge escalations \
                 (max_diag={max_diag:.3e})",
                MAX_RIDGE_ESCALATIONS,
            ))),
        },
    }
}

/// Fit a penalized vector-response GLM at fixed `λ` via damped Newton.
///
/// The `likelihood` adapter supplies the per-row Fisher block, the residual
/// gradient, and the log-likelihood; the engine owns the entire optimisation
/// scaffold. See the module docs for the optimisation problem, the
/// output-major coefficient ordering, and the convergence semantics.
///
/// `context` is woven into every diagnostic message so each family keeps its
/// own error prefix (e.g. `"fit_penalized_multinomial"`).
pub fn fit_penalized_vector_glm<L: VectorLikelihood>(
    inputs: PenalizedVectorGlmInputs<'_>,
    likelihood: &L,
    context: &str,
) -> Result<PenalizedVectorGlmOutputs, EstimationError> {
    let PenalizedVectorGlmInputs {
        design,
        y,
        penalty,
        lambdas,
        fisher_w_override,
        max_iter,
        tol,
        class_penalty_metric,
    } = inputs;

    // ────────────────────────────── shape checks ──────────────────────────
    let n_obs = design.nrows();
    let p = design.ncols();
    if n_obs == 0 || p == 0 {
        crate::bail_invalid_estim!("{context}: design must be nonempty (got {n_obs}x{p})");
    }
    let m = lambdas.len();
    if m == 0 {
        crate::bail_invalid_estim!("{context}: need at least one active output (got M=0)");
    }
    if y.nrows() != n_obs {
        crate::bail_invalid_estim!("{context}: y rows {} ≠ design rows {n_obs}", y.nrows());
    }
    if penalty.dim() != (p, p) {
        crate::bail_invalid_estim!(
            "{context}: penalty shape {:?} ≠ (P, P) = ({p}, {p})",
            penalty.dim()
        );
    }
    for (i, &v) in lambdas.iter().enumerate() {
        if !(v.is_finite() && v >= 0.0) {
            crate::bail_invalid_estim!("{context}: lambdas[{i}] must be finite and ≥ 0 (got {v})");
        }
    }
    if let Some(fw) = fisher_w_override.as_ref() {
        if fw.dim() != (n_obs, m, m) {
            crate::bail_invalid_estim!(
                "{context}: fisher_w_override shape {:?} ≠ (N, M, M) = ({n_obs}, {m}, {m})",
                fw.dim()
            );
        }
    }
    for ((i, j), &v) in design.indexed_iter() {
        if !v.is_finite() {
            crate::bail_invalid_estim!("{context}: design[{i},{j}] must be finite (got {v})");
        }
    }

    // ────────────────────────── Newton iteration ──────────────────────────
    // β stored as (P, M) column-major-per-output; flat index uses output-major
    // ordering `flat[a · P + i] = β[i, a]` to align with `dense_block_xtwx`.
    let mut beta = Array2::<f64>::zeros((p, m));
    let mut eta = Array2::<f64>::zeros((n_obs, m));
    // Reused η scratch for the line-search objective probes (see
    // `evaluate_objective`): overwritten in full on every call, so it carries
    // no state between calls and hoisting it out of the backtracking loop is a
    // pure heap-allocation removal with no effect on the computed objective.
    let mut eta_objective_scratch = Array2::<f64>::zeros((n_obs, m));
    let beta_flat_dim = p * m;
    // Reused penalized-gradient buffer: each Newton iteration writes every entry
    // `grad_flat[a·p + i] = Xᵀr` (direct assignment over all a∈0..m, i∈0..p)
    // before adding the penalty term and before any read, so it carries no state
    // across iterations and hoisting it out of the Newton loop is a pure
    // heap-allocation removal with no effect on the computed gradient.
    let mut grad_flat = Array1::<f64>::zeros(beta_flat_dim);

    let mut iterations = 0usize;
    let mut converged = false;
    let mut last_objective = f64::INFINITY;

    // η = X · β for the current β, reused by the analytic Fisher / gradient.
    let recompute_eta = |beta: &Array2<f64>, eta: &mut Array2<f64>| {
        for a in 0..m {
            let beta_col = beta.column(a);
            for row in 0..n_obs {
                let mut eta_val = 0.0_f64;
                for i in 0..p {
                    eta_val += design[[row, i]] * beta_col[i];
                }
                eta[[row, a]] = eta_val;
            }
        }
    };

    // Penalized objective F(β) = − log L(X β) + ½ Σ_a λ_a β_aᵀ S β_a.
    // The caller supplies a reused `(n_obs, m)` scratch for η = X·β so the
    // backtracking line search (which calls this up to `MAX_BACKTRACKS + 1`
    // times per Newton iteration) does not heap-allocate a fresh η buffer on
    // every probe. The scratch is overwritten in full by `recompute_eta` before
    // it is read, so reusing it is bit-for-bit identical to the prior
    // allocate-fresh body: `recompute_eta` runs the SAME `Σ_i design·β` loop in
    // the SAME order this closure used inline.
    let evaluate_objective = |beta_trial: &Array2<f64>, eta_scratch: &mut Array2<f64>| -> f64 {
        recompute_eta(beta_trial, eta_scratch);
        let ll = likelihood.log_lik(eta_scratch.view(), y);
        let pen = weighted_penalty_sum(beta_trial, penalty, lambdas, class_penalty_metric);
        -ll + pen
    };

    for iter in 0..max_iter {
        iterations = iter + 1;

        recompute_eta(&beta, &mut eta);

        // Per-row dense Fisher block W_{n,a,b} = −∂² log L / ∂η_a ∂η_b: either
        // the caller-supplied curvature override (issue #349 escape-hatch —
        // curvature only) or the analytic [`VectorLikelihood::hess_block`]. The
        // residual r_{n,a} = −∂ log L / ∂η_a stays analytic in both cases.
        let analytic_fisher = fisher_w_override
            .as_ref()
            .map_or_else(|| Some(likelihood.hess_block(eta.view(), y)), |_| None);
        let fisher_blocks = match fisher_w_override.as_ref() {
            Some(fw) => *fw,
            None => analytic_fisher
                .as_ref()
                .expect("analytic Fisher computed when no override")
                .view(),
        };
        let residual = likelihood.grad_eta(eta.view(), y).mapv(|v| -v);

        // Penalized Hessian: H = block(XᵀWX) + diag_a(λ_a S).
        let mut hessian = dense_block_xtwx(design, fisher_blocks, None)?;
        if hessian.nrows() != beta_flat_dim || hessian.ncols() != beta_flat_dim {
            crate::bail_invalid_estim!(
                "{context}: assembled Hessian shape {:?} ≠ ({beta_flat_dim}, {beta_flat_dim})",
                hessian.dim()
            );
        }
        match class_penalty_metric {
            ClassPenaltyMetric::Diagonal => {
                for a in 0..m {
                    let la = lambdas[a];
                    if la == 0.0 {
                        continue;
                    }
                    let base = a * p;
                    for i in 0..p {
                        for j in 0..p {
                            hessian[[base + i, base + j]] += la * penalty[[i, j]];
                        }
                    }
                }
            }
            // Centered (#1587): H_{ab} += λ·(δ_ab − 1/K)·S, K = M+1, shared
            // λ = lambdas[0] — couples every class pair via the −(λ/K)·S
            // off-diagonals. Reference-invariant softmax penalty.
            ClassPenaltyMetric::Centered if m > 0 && lambdas[0] != 0.0 => {
                let lam = lambdas[0];
                let inv_k = 1.0 / ((m + 1) as f64);
                for a in 0..m {
                    for b in 0..m {
                        let coef = lam * (if a == b { 1.0 } else { 0.0 } - inv_k);
                        let (ba, bb) = (a * p, b * p);
                        for i in 0..p {
                            for j in 0..p {
                                hessian[[ba + i, bb + j]] += coef * penalty[[i, j]];
                            }
                        }
                    }
                }
            }
            ClassPenaltyMetric::Centered => {}
        }

        // Penalized gradient: g_a = Xᵀ r_{·,a} + (penalty gradient). For the
        // Diagonal metric that is `λ_a S β_a`; for Centered it is the
        // reference-symmetric `λ S (β_a − β̄)`, β̄ = (1/K) Σ_b β_b (#1587).
        // Written into the reused `grad_flat` buffer; the loop below assigns
        // every entry (`=`) before the penalty `+=`, so no re-zeroing is needed.
        for a in 0..m {
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n_obs {
                    acc += design[[row, i]] * residual[[row, a]];
                }
                grad_flat[a * p + i] = acc;
            }
        }
        match class_penalty_metric {
            ClassPenaltyMetric::Diagonal => {
                for a in 0..m {
                    let la = lambdas[a];
                    if la == 0.0 {
                        continue;
                    }
                    let beta_col = beta.column(a);
                    for i in 0..p {
                        let mut s_beta_i = 0.0_f64;
                        for j in 0..p {
                            s_beta_i += penalty[[i, j]] * beta_col[j];
                        }
                        grad_flat[a * p + i] += la * s_beta_i;
                    }
                }
            }
            ClassPenaltyMetric::Centered if m > 0 && lambdas[0] != 0.0 => {
                let lam = lambdas[0];
                let inv_k = 1.0 / ((m + 1) as f64);
                // β̄ = (1/K) Σ_b β_b.
                let mut bbar = vec![0.0_f64; p];
                for b in 0..m {
                    let col = beta.column(b);
                    for i in 0..p {
                        bbar[i] += col[i];
                    }
                }
                for v in bbar.iter_mut() {
                    *v *= inv_k;
                }
                for a in 0..m {
                    let beta_col = beta.column(a);
                    for i in 0..p {
                        // (S (β_a − β̄))_i.
                        let mut s_centered_i = 0.0_f64;
                        for j in 0..p {
                            s_centered_i += penalty[[i, j]] * (beta_col[j] - bbar[j]);
                        }
                        grad_flat[a * p + i] += lam * s_centered_i;
                    }
                }
            }
            ClassPenaltyMetric::Centered => {}
        }

        // δ = − H^{-1} · grad, solved through an adaptive Levenberg–Marquardt
        // ridge. The penalized Hessian `H = block(XᵀWX) + diag_a(λ_a S)` can be
        // rank-deficient — a multinomial class block with quasi-separated /
        // collinear columns and a small per-class λ leaves `XᵀW_aX + λ_a S`
        // singular. faer's symmetric fallback chain ends at Bunch–Kaufman
        // (LBLᵀ), which factorizes indefinite/singular matrices "successfully"
        // and then back-substitutes through near-zero pivots, yielding a
        // non-finite δ. Rather than aborting the whole fit on one bad block, we
        // add a small ridge `τ·I` (Levenberg style) to the diagonal and
        // re-factorize, escalating τ geometrically until the step is finite.
        //
        // The base ridge is scaled by the Hessian's largest diagonal entry so
        // it is invariant to the problem's overall curvature scale: a tiny
        // nudge relative to the dominant curvature, large enough to lift the
        // null directions off zero. A finite δ from the ridged system is a
        // descent direction for the *unridged* penalized objective `F`
        // (ridging only shrinks the step toward the gradient direction), and
        // the backtracking line search below validates it against `F` itself,
        // so the ridge never biases the converged β̂ — at the optimum the
        // gradient vanishes and the step → 0 regardless of τ.
        let max_diag =
            (0..beta_flat_dim).fold(0.0_f64, |acc, idx| acc.max(hessian[[idx, idx]].abs()));
        // The ridge floors at `base_ridge` (not 0) for every solve. An exactly
        // rank-deficient block (e.g. duplicate / collinear design columns under
        // a near-zero λ) leaves `H = block(XᵀWX) + diag_a(λ_a S)` singular along
        // a null direction. faer's Bunch–Kaufman fallback factorizes a singular
        // matrix "successfully" and back-substitutes through the zero pivot to a
        // *finite but arbitrary* component in the null space, so the resulting
        // Newton direction is not a descent direction in the identified
        // subspace — the line search then shrinks α toward 0 and the step-norm
        // test declares a false convergence at a point where the unridged
        // penalized gradient on identified directions is still large (gam#856).
        // A minimal Tikhonov ridge `base_ridge·I` resolves the null direction to
        // its minimum-norm representative, giving a true descent direction.
        let base_ridge = if max_diag.is_finite() && max_diag > 0.0 {
            max_diag * BASE_RIDGE_FRACTION_OF_MAX_DIAG
        } else {
            BASE_RIDGE_FRACTION_OF_MAX_DIAG
        };
        // A genuine factorization failure (not just a singular pivot) is
        // remembered so exhaustion can surface its distinct terminal error;
        // singular pivots back-substituted to ±inf/NaN just escalate.
        let mut last_factor_err: Option<(f64, String)> = None;
        let delta = match escalate_ridge(
            RidgeSchedule {
                initial: base_ridge,
                growth: 2.0,
                max_escalations: MAX_RIDGE_ESCALATIONS + 1,
            },
            |ridge| {
                let mut ridged = hessian.clone();
                for idx in 0..beta_flat_dim {
                    ridged[[idx, idx]] += ridge;
                }
                let factor = match factorize_symmetricwith_fallback(
                    FaerArrayView::new(&ridged).as_ref(),
                    Side::Lower,
                ) {
                    Ok(factor) => factor,
                    Err(err) => {
                        last_factor_err = Some((ridge, err.to_string()));
                        return None;
                    }
                };
                last_factor_err = None;
                let mut rhs = Array2::<f64>::zeros((beta_flat_dim, 1));
                for i in 0..beta_flat_dim {
                    rhs[[i, 0]] = -grad_flat[i];
                }
                {
                    let rhs_view = array2_to_matmut(&mut rhs);
                    factor.solve_in_place(rhs_view);
                }
                (0..beta_flat_dim)
                    .all(|i| rhs[[i, 0]].is_finite())
                    .then(|| Array1::from_iter((0..beta_flat_dim).map(|i| rhs[[i, 0]])))
            },
        ) {
            Ok(success) => success.value,
            Err(exhausted) => {
                if let Some((ridge, err)) = last_factor_err {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: Hessian factorization failed at iter {iter} \
                         even with ridge {ridge:.3e}: {err}"
                    )));
                }
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: Newton step remained non-finite at iter {iter} after {} ridge \
                     escalations up to {:.3e}; the penalized Hessian is pathologically \
                     rank-deficient (grad_norm={:.3e}, max_diag={max_diag:.3e})",
                    MAX_RIDGE_ESCALATIONS,
                    exhausted.next_ridge,
                    grad_flat.iter().map(|v| v * v).sum::<f64>().sqrt(),
                )));
            }
        };

        // Damped acceptance: full step first, halve up to `MAX_BACKTRACKS` times
        // if the penalized negative log-likelihood fails to decrease. The first
        // iteration seeds `last_objective` from the initial β.
        let proposed_beta = |alpha: f64| -> Array2<f64> {
            let mut out = beta.clone();
            for a in 0..m {
                for i in 0..p {
                    out[[i, a]] += alpha * delta[a * p + i];
                }
            }
            out
        };
        if iter == 0 {
            last_objective = evaluate_objective(&beta, &mut eta_objective_scratch);
            if !last_objective.is_finite() {
                crate::bail_invalid_estim!("{context}: non-finite objective at β = 0");
            }
        }
        // The trial closure writes each candidate into `accepted_beta`/
        // `new_objective` directly: on acceptance they hold the accepted trial,
        // and on exhaustion they hold the LAST (rejected) trial — this site has
        // always proceeded with that final halved step rather than falling back.
        let mut accepted_beta = Array2::<f64>::zeros(beta.raw_dim());
        let mut new_objective = f64::NAN;
        match backtracking_line_search::<_, Infallible>(
            BacktrackConfig {
                contraction: LINE_SEARCH_SHRINK,
                max_steps: MAX_BACKTRACKS + 1,
                ..BacktrackConfig::default()
            },
            |alpha| {
                accepted_beta = proposed_beta(alpha);
                new_objective = evaluate_objective(&accepted_beta, &mut eta_objective_scratch);
                Ok(Some((new_objective, ())))
            },
            |_alpha, f| f.is_finite() && f <= last_objective + OBJECTIVE_DECREASE_SLACK,
        ) {
            Ok(_) => {}
            Err(never) => match never {},
        }

        let mut step_norm_sq = 0.0_f64;
        let mut beta_norm_sq = 0.0_f64;
        for a in 0..m {
            for i in 0..p {
                let d = accepted_beta[[i, a]] - beta[[i, a]];
                step_norm_sq += d * d;
                let v = accepted_beta[[i, a]];
                beta_norm_sq += v * v;
            }
        }

        beta = accepted_beta;
        last_objective = new_objective;

        let step_norm = step_norm_sq.sqrt();
        let beta_norm = beta_norm_sq.sqrt();
        // First-order optimality gate (gam#856): the step-norm test alone can
        // fire prematurely when a backtracking line search has shrunk α on a
        // poor direction, leaving a point that is NOT stationary. `grad_flat`
        // is the unridged penalized gradient ∇F(β) at the pre-step β; with a
        // small step it is ≈ ∇F at the accepted β. Its norm reflects only
        // identified directions (it is exactly zero along an unidentified null
        // direction such as a duplicate-column e₁−e₂ split), so requiring it to
        // be small certifies first-order optimality on the identified subspace
        // without penalizing legitimate non-identifiability. Scale the gate by
        // the data magnitude so it is invariant to problem scale.
        let grad_norm = grad_flat.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Curvature-scaled optimality threshold: `max_diag` is the dominant
        // penalized-Hessian diagonal entry, so `OPTIMALITY_GRAD_FRACTION·max_diag`
        // is a tiny gradient relative to the problem's curvature scale and is
        // reached by a few quadratically-converging Newton steps on this smooth,
        // bounded softmax/binomial likelihood.
        let grad_optimal = grad_norm <= OPTIMALITY_GRAD_FRACTION * (1.0 + max_diag);
        if step_norm <= tol * (1.0 + beta_norm) && grad_optimal {
            converged = true;
            break;
        }
    }

    // ──────────────────────────── post-process ────────────────────────────
    recompute_eta(&beta, &mut eta);
    let log_likelihood = likelihood.log_lik(eta.view(), y);
    let penalty_term = weighted_penalty_sum(&beta, penalty, lambdas, class_penalty_metric);

    // Joint Laplace covariance `H⁻¹` at the converged mode (#1101). Re-assemble
    // the penalized Hessian `H = block(XᵀWX) + penalty` at β̂ — the SAME algebra
    // the Newton loop runs each iteration — and invert it by solving `H·Σ = I`
    // through the shared symmetric factorization. The Newton loop discarded its
    // per-step factor; this recomputes the factor once at the mode where the
    // curvature is the correct posterior precision. A tiny curvature-scaled
    // ridge is added only when the raw factorization / solve is non-finite
    // (rank-deficient null direction), mirroring the Newton step's ridge logic,
    // so the covariance is always finite; at full rank the ridge is never used.
    let analytic_fisher_final = fisher_w_override
        .as_ref()
        .map_or_else(|| Some(likelihood.hess_block(eta.view(), y)), |_| None);
    let fisher_blocks_final = match fisher_w_override.as_ref() {
        Some(fw) => *fw,
        None => analytic_fisher_final
            .as_ref()
            .expect("analytic Fisher computed when no override")
            .view(),
    };
    let mut hessian_final = dense_block_xtwx(design, fisher_blocks_final, None)?;
    match class_penalty_metric {
        ClassPenaltyMetric::Diagonal => {
            for a in 0..m {
                let la = lambdas[a];
                if la == 0.0 {
                    continue;
                }
                let base = a * p;
                for i in 0..p {
                    for j in 0..p {
                        hessian_final[[base + i, base + j]] += la * penalty[[i, j]];
                    }
                }
            }
        }
        ClassPenaltyMetric::Centered if m > 0 && lambdas[0] != 0.0 => {
            let lam = lambdas[0];
            let inv_k = 1.0 / ((m + 1) as f64);
            for a in 0..m {
                for b in 0..m {
                    let coef = lam * (if a == b { 1.0 } else { 0.0 } - inv_k);
                    let (ba, bb) = (a * p, b * p);
                    for i in 0..p {
                        for j in 0..p {
                            hessian_final[[ba + i, bb + j]] += coef * penalty[[i, j]];
                        }
                    }
                }
            }
        }
        ClassPenaltyMetric::Centered => {}
    }
    let coefficient_covariance =
        invert_symmetric_penalized_hessian(&hessian_final, beta_flat_dim, context)?;

    Ok(PenalizedVectorGlmOutputs {
        coefficients: beta,
        eta,
        iterations,
        converged,
        log_likelihood,
        penalty_term,
        coefficient_covariance,
    })
}

#[cfg(test)]
mod parity_tests {
    //! Parity tests for the shared scaffold across both Fisher-block families
    //! (issue #409). The engine is exercised through the two public adapters —
    //! [`crate::binomial_multi::fit_penalized_binomial_multi`]
    //! (row-diagonal block) and
    //! [`crate::multinomial::fit_penalized_multinomial`] (dense
    //! softmax block) — and we assert, with un-weakened bounds, that:
    //!
    //!   1. each fit hits the first-order optimality condition `∇F(β̂) = 0`,
    //!      verified by a central finite difference of the penalized objective
    //!      (the engine never sees this gradient, so this is an independent
    //!      check that the shared Newton scaffold converged correctly);
    //!   2. the reported fitted probabilities are consistent with `β̂` and the
    //!      reported deviance equals `−2 · log L(β̂)`;
    //!   3. for the binomial family, the `K`-column joint solve reproduces a
    //!      from-scratch single-column penalized logistic Newton solve column
    //!      for column (the row-diagonal block must decouple exactly).

    use super::{ClassPenaltyMetric, weighted_penalty_sum};
    use crate::binomial_multi::{BinomialMultiFitInputs, fit_penalized_binomial_multi};
    use crate::multinomial::{MultinomialFitInputs, fit_penalized_multinomial};
    use ndarray::{Array1, Array2};

    /// #1587: the `Centered` class-penalty metric is invariant to the arbitrary
    /// reference-class choice. Penalizing the `K−1` ALR contrasts under ANY of
    /// the `K` baselines yields the same value (the symmetric CLR penalty
    /// `Σ_k β̃_kᵀSβ̃_k`), whereas the historical `Diagonal` metric does not — that
    /// non-invariance is exactly the #1587 defect. Pure-algebra check on the
    /// penalty form (no fit), so it pins the engine foundation the production
    /// wiring (REML per-term λ re-key) will build on.
    #[test]
    fn centered_penalty_is_reference_class_invariant_1587() {
        // K = 3 classes, p = 2 coefficients; symmetric PSD penalty S.
        let s = ndarray::array![[2.0_f64, 0.5], [0.5, 1.0]];
        // A CLR (sum-to-zero) coefficient set: β̃_0 + β̃_1 + β̃_2 = 0.
        let bt = [[1.0_f64, 0.5], [-0.3, 0.2], [-0.7, -0.7]];
        for j in 0..2 {
            let colsum: f64 = (0..3).map(|k| bt[k][j]).sum();
            assert!(colsum.abs() < 1e-12, "test CLR set must sum to zero");
        }
        // Direct symmetric penalty Σ_k β̃_kᵀ S β̃_k.
        let mut symmetric = 0.0_f64;
        for k in 0..3 {
            for i in 0..2 {
                for j in 0..2 {
                    symmetric += bt[k][i] * s[[i, j]] * bt[k][j];
                }
            }
        }
        let lambdas = Array1::from(vec![1.0_f64, 1.0]);
        let mut centered_vals = Vec::new();
        let mut diagonal_vals = Vec::new();
        // For each reference class r, the two ALR contrasts are β̃_a − β̃_r (a≠r).
        for r in 0..3 {
            let others: Vec<usize> = (0..3).filter(|&k| k != r).collect();
            let mut beta = Array2::<f64>::zeros((2, 2));
            for (a, &o) in others.iter().enumerate() {
                for i in 0..2 {
                    beta[[i, a]] = bt[o][i] - bt[r][i];
                }
            }
            let c =
                weighted_penalty_sum(&beta, s.view(), lambdas.view(), ClassPenaltyMetric::Centered);
            let d =
                weighted_penalty_sum(&beta, s.view(), lambdas.view(), ClassPenaltyMetric::Diagonal);
            assert!(
                (c - 0.5 * symmetric).abs() < 1e-12,
                "ref {r}: Centered penalty {c} must equal ½·symmetric {}",
                0.5 * symmetric
            );
            centered_vals.push(c);
            diagonal_vals.push(d);
        }
        let cspread = centered_vals.iter().cloned().fold(f64::MIN, f64::max)
            - centered_vals.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            cspread < 1e-12,
            "Centered must be reference-invariant; got {centered_vals:?}"
        );
        let dspread = diagonal_vals.iter().cloned().fold(f64::MIN, f64::max)
            - diagonal_vals.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            dspread > 1e-6,
            "Diagonal is the non-invariant #1587 path; references must disagree, got {diagonal_vals:?}"
        );
    }

    fn sigmoid(eta: f64) -> f64 {
        if eta >= 0.0 {
            1.0 / (1.0 + (-eta).exp())
        } else {
            let e = eta.exp();
            e / (1.0 + e)
        }
    }

    /// Softmax with implicit reference column (η_ref = 0) over `M` active η.
    fn softmax_ref(eta_active: &[f64]) -> Vec<f64> {
        let m = eta_active.len();
        let mut out = vec![0.0_f64; m + 1];
        let mut max_eta = 0.0_f64;
        for &v in eta_active {
            if v > max_eta {
                max_eta = v;
            }
        }
        let baseline = (-max_eta).exp();
        let mut denom = baseline;
        for (idx, &v) in eta_active.iter().enumerate() {
            let e = (v - max_eta).exp();
            out[idx] = e;
            denom += e;
        }
        for v in out.iter_mut().take(m) {
            *v /= denom;
        }
        out[m] = baseline / denom;
        out
    }

    /// Penalized negative log-likelihood for the independent-binomial family at
    /// a candidate coefficient matrix `β ∈ ℝ^{P×K}`, computed directly from the
    /// definition (no engine internals).
    fn binomial_objective(
        design: &Array2<f64>,
        y: &Array2<f64>,
        penalty: &Array2<f64>,
        lambdas: &Array1<f64>,
        beta: &Array2<f64>,
    ) -> f64 {
        let (n, p) = design.dim();
        let k = y.ncols();
        let mut ll = 0.0_f64;
        for row in 0..n {
            for a in 0..k {
                let mut eta = 0.0_f64;
                for i in 0..p {
                    eta += design[[row, i]] * beta[[i, a]];
                }
                let mu = sigmoid(eta).clamp(1.0e-12, 1.0 - 1.0e-12);
                let yv = y[[row, a]];
                ll += yv * mu.ln() + (1.0 - yv) * (1.0 - mu).ln();
            }
        }
        let mut pen = 0.0_f64;
        for a in 0..k {
            let la = lambdas[a];
            for i in 0..p {
                let mut sbi = 0.0_f64;
                for j in 0..p {
                    sbi += penalty[[i, j]] * beta[[j, a]];
                }
                pen += 0.5 * la * beta[[i, a]] * sbi;
            }
        }
        -ll + pen
    }

    /// Penalized negative log-likelihood for the multinomial family at a
    /// candidate active-class coefficient matrix `β ∈ ℝ^{P×(K-1)}`.
    fn multinomial_objective(
        design: &Array2<f64>,
        y_one_hot: &Array2<f64>,
        penalty: &Array2<f64>,
        lambdas: &Array1<f64>,
        beta: &Array2<f64>,
    ) -> f64 {
        let (n, p) = design.dim();
        let k = y_one_hot.ncols();
        let m = k - 1;
        let mut ll = 0.0_f64;
        let mut eta_active = vec![0.0_f64; m];
        for row in 0..n {
            for a in 0..m {
                let mut eta = 0.0_f64;
                for i in 0..p {
                    eta += design[[row, i]] * beta[[i, a]];
                }
                eta_active[a] = eta;
            }
            let probs = softmax_ref(&eta_active);
            for c in 0..k {
                let yc = y_one_hot[[row, c]];
                if yc != 0.0 {
                    ll += yc * probs[c].max(1.0e-300).ln();
                }
            }
        }
        let mut pen = 0.0_f64;
        for a in 0..m {
            let la = lambdas[a];
            for i in 0..p {
                let mut sbi = 0.0_f64;
                for j in 0..p {
                    sbi += penalty[[i, j]] * beta[[j, a]];
                }
                pen += 0.5 * la * beta[[i, a]] * sbi;
            }
        }
        -ll + pen
    }

    /// Central finite-difference gradient of an objective over every entry of a
    /// `(P, C)` coefficient matrix. The optimum must drive every component to
    /// ~0; we assert the max |component| against an un-weakened bound.
    fn fd_grad<F: Fn(&Array2<f64>) -> f64>(beta: &Array2<f64>, f: F) -> f64 {
        let (p, c) = beta.dim();
        let h = 1.0e-6;
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for a in 0..c {
                let mut up = beta.clone();
                let mut dn = beta.clone();
                up[[i, a]] += h;
                dn[[i, a]] -= h;
                let g = (f(&up) - f(&dn)) / (2.0 * h);
                max_abs = max_abs.max(g.abs());
            }
        }
        max_abs
    }

    fn binomial_fixture() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 40;
        let p = 3;
        let k = 3;
        let design = Array2::<f64>::from_shape_fn((n, p), |(i, j)| match j {
            0 => 1.0,
            1 => ((i + 1) as f64 * 0.37).sin(),
            _ => ((i + 1) as f64 * 0.11).cos(),
        });
        let y = Array2::<f64>::from_shape_fn((n, k), |(i, a)| {
            // Deterministic but non-degenerate {0,1} labels per column.
            if ((i * 7 + a * 13 + 3) % 5) < 3 {
                1.0
            } else {
                0.0
            }
        });
        let penalty = Array2::<f64>::eye(p);
        let lambdas = Array1::from(vec![0.3_f64, 1.2, 2.5]);
        (design, y, penalty, lambdas)
    }

    fn multinomial_fixture() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 45;
        let p = 3;
        let k = 4;
        let design = Array2::<f64>::from_shape_fn((n, p), |(i, j)| match j {
            0 => 1.0,
            1 => ((i + 2) as f64 * 0.29).sin(),
            _ => ((i + 2) as f64 * 0.17).cos(),
        });
        let mut y = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            y[[i, (i * 3 + 1) % k]] = 1.0;
        }
        let penalty = Array2::<f64>::eye(p);
        let lambdas = Array1::from(vec![0.5_f64, 1.0, 2.0]);
        (design, y, penalty, lambdas)
    }

    #[test]
    fn binomial_engine_hits_optimum_and_is_self_consistent() {
        let (design, y, penalty, lambdas) = binomial_fixture();
        let fit = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 100,
            tol: 1.0e-12,
        })
        .expect("binomial fit must succeed");
        assert!(fit.converged, "binomial fit must converge");

        // First-order optimality: ∇F(β̂) = 0 (engine never used this gradient).
        let g = fd_grad(&fit.coefficients, |b| {
            binomial_objective(&design, &y, &penalty, &lambdas, b)
        });
        assert!(
            g < 1.0e-6,
            "binomial penalized gradient at β̂ must vanish (max |∂F| = {g})"
        );

        // Fitted probabilities reproduce σ(X β̂) and deviance = −2 log L.
        let (n, p) = design.dim();
        let k = y.ncols();
        let mut log_lik = 0.0_f64;
        for row in 0..n {
            for a in 0..k {
                let mut eta = 0.0_f64;
                for i in 0..p {
                    eta += design[[row, i]] * fit.coefficients[[i, a]];
                }
                let mu = sigmoid(eta);
                assert!(
                    (fit.fitted_probabilities[[row, a]] - mu).abs() < 1.0e-10,
                    "fitted probability must equal σ(X β̂)"
                );
                let muc = mu.clamp(1.0e-12, 1.0 - 1.0e-12);
                let yv = y[[row, a]];
                log_lik += yv * muc.ln() + (1.0 - yv) * (1.0 - muc).ln();
            }
        }
        assert!(
            (fit.deviance - (-2.0 * log_lik)).abs() < 1.0e-9,
            "deviance must equal −2 log L"
        );
    }

    #[test]
    fn binomial_joint_solve_decouples_into_single_column_solves() {
        // Parity: the row-diagonal Fisher block means the K-column joint solve
        // must reproduce, column for column, an independent single-column
        // penalized logistic Newton solve. This is the defining property the
        // shared engine preserves for the independent-binomial family.
        let (design, y, penalty, lambdas) = binomial_fixture();
        let joint = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 100,
            tol: 1.0e-12,
        })
        .expect("joint fit must succeed");

        let k = y.ncols();
        for a in 0..k {
            // Single-column problem: one binomial response, one λ.
            let y_col = y.column(a).to_owned().insert_axis(ndarray::Axis(1));
            let lam = Array1::from(vec![lambdas[a]]);
            let single = fit_penalized_binomial_multi(BinomialMultiFitInputs {
                design: design.view(),
                y: y_col.view(),
                penalty: penalty.view(),
                lambdas: lam.view(),
                row_weights: None,
                fisher_w_override: None,
                max_iter: 100,
                tol: 1.0e-12,
            })
            .expect("single-column fit must succeed");
            for i in 0..design.ncols() {
                let dj = joint.coefficients[[i, a]];
                let ds = single.coefficients[[i, 0]];
                assert!(
                    (dj - ds).abs() < 1.0e-8,
                    "joint column {a} coef {i} ({dj}) must match single-column solve ({ds})"
                );
            }
        }
    }

    #[test]
    fn multinomial_engine_hits_optimum_and_is_self_consistent() {
        let (design, y, penalty, lambdas) = multinomial_fixture();
        let fit = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 100,
            tol: 1.0e-12,
        })
        .expect("multinomial fit must succeed");
        assert!(fit.converged, "multinomial fit must converge");

        // First-order optimality: ∇F(β̂) = 0.
        let g = fd_grad(&fit.coefficients_active, |b| {
            multinomial_objective(&design, &y, &penalty, &lambdas, b)
        });
        assert!(
            g < 1.0e-6,
            "multinomial penalized gradient at β̂ must vanish (max |∂F| = {g})"
        );

        // Fitted probabilities are a valid simplex per row and reproduce the
        // softmax of X β̂; deviance = −2 log L.
        let (n, p) = design.dim();
        let k = y.ncols();
        let m = k - 1;
        let mut log_lik = 0.0_f64;
        let mut eta_active = vec![0.0_f64; m];
        for row in 0..n {
            for a in 0..m {
                let mut eta = 0.0_f64;
                for i in 0..p {
                    eta += design[[row, i]] * fit.coefficients_active[[i, a]];
                }
                eta_active[a] = eta;
            }
            let probs = softmax_ref(&eta_active);
            let mut row_sum = 0.0_f64;
            for c in 0..k {
                assert!(
                    (fit.fitted_probabilities[[row, c]] - probs[c]).abs() < 1.0e-10,
                    "fitted probability must equal softmax(X β̂)"
                );
                row_sum += fit.fitted_probabilities[[row, c]];
                let yc = y[[row, c]];
                if yc != 0.0 {
                    log_lik += yc * probs[c].max(1.0e-300).ln();
                }
            }
            assert!(
                (row_sum - 1.0).abs() < 1.0e-10,
                "fitted probabilities must sum to 1 per row"
            );
        }
        assert!(
            (fit.deviance - (-2.0 * log_lik)).abs() < 1.0e-9,
            "deviance must equal −2 log L"
        );
    }

    #[test]
    fn multinomial_rank_deficient_block_recovers_via_ridge_not_crash() {
        // Issue #557: a rank-deficient class block under a tiny per-class λ used
        // to make faer's Bunch–Kaufman fallback back-substitute through near-zero
        // pivots into a non-finite Newton step δ, and the solver aborted with
        // "Newton step is non-finite". The adaptive Levenberg–Marquardt ridge
        // must instead lift the null direction off zero, keep δ finite, and let
        // the backtracking line search converge to the penalized optimum.
        //
        // Construct an exactly rank-deficient design: column 2 is a perfect
        // duplicate of column 1, so XᵀWX is singular along (e₁ − e₂) for every
        // class, and we drive the corresponding λ to a tiny value so the penalty
        // cannot regularize that null direction. A non-robust solver crashes
        // here; the ridge path must produce a finite, self-consistent fit.
        let n = 50;
        let p = 4;
        let k = 4;
        let design = Array2::<f64>::from_shape_fn((n, p), |(i, j)| match j {
            0 => 1.0,
            1 => ((i + 1) as f64 * 0.23).sin(),
            2 => ((i + 1) as f64 * 0.23).sin(), // exact duplicate of column 1
            _ => ((i + 1) as f64 * 0.19).cos(),
        });
        let mut y = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            y[[i, (i * 5 + 2) % k]] = 1.0;
        }
        // Penalty touches only the smooth-ish columns 1..p; columns 0/1/2 share
        // the collinearity, and a near-zero λ leaves the (e₁ − e₂) null direction
        // unregularized — exactly the rank-deficient regime that triggered #557.
        let mut penalty = Array2::<f64>::zeros((p, p));
        penalty[[3, 3]] = 1.0;
        let lambdas = Array1::from(vec![1.0e-10_f64, 1.0e-10, 1.0e-10]);

        let fit = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 200,
            tol: 1.0e-10,
        })
        .expect("rank-deficient multinomial fit must NOT crash (#557): the ridge path recovers it");

        // Every coefficient and fitted probability must be finite (no inf/NaN
        // leaked from the near-singular solve).
        for &c in fit.coefficients_active.iter() {
            assert!(c.is_finite(), "coefficient must be finite, got {c}");
        }
        for &pr in fit.fitted_probabilities.iter() {
            assert!(
                pr.is_finite() && (-1.0e-9..=1.0 + 1.0e-9).contains(&pr),
                "fitted probability must be a finite simplex entry, got {pr}"
            );
        }
        // Rows must remain on the simplex.
        let (nn, kk) = fit.fitted_probabilities.dim();
        for row in 0..nn {
            let s: f64 = (0..kk).map(|c| fit.fitted_probabilities[[row, c]]).sum();
            assert!(
                (s - 1.0).abs() < 1.0e-9,
                "row {row} probabilities must sum to 1, got {s}"
            );
        }

        // The recovered fit must satisfy first-order optimality of the penalized
        // objective along every NON-NULL coordinate. The (e₁ − e₂) null
        // direction is unidentified (the ridge picks the minimum-norm split
        // between the duplicate columns), so the gradient is exactly zero along
        // every identified direction; a central finite difference of F over the
        // full coefficient matrix is dominated by the identified part and must be
        // small. We assert the penalized objective gradient is near-zero — the
        // ridge biases the step but never the optimum (at β̂ the unridged
        // gradient vanishes for any τ).
        let g = fd_grad(&fit.coefficients_active, |b| {
            multinomial_objective(&design, &y, &penalty, &lambdas, b)
        });
        assert!(
            g < 1.0e-4,
            "penalized objective gradient at the ridge-recovered β̂ must (near-)vanish \
             along identified directions (max |∂F| = {g})"
        );
    }
}
