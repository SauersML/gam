//! Penalized multinomial-logit (softmax) GLM driver — fixed-λ inner solve.
//!
//! This is the principled vector-response companion to the scalar PIRLS path:
//! the inner-loop Newton solver for a multi-class GAM at fixed smoothing
//! parameters λ, using the canonical multinomial-logit likelihood
//! ([`MultinomialLogitLikelihood`]) and the existing dense block-Fisher
//! assembly in [`crate::solver::pirls::dense_block_xtwx`] /
//! [`crate::solver::pirls::dense_block_xtwy`].
//!
//! # What this module does
//!
//! Solve, for the reference-coded multinomial-logit GAM with `K` classes and
//! design matrix `X ∈ ℝ^{N×P}`,
//!
//! ```text
//!     β̂ = argmin_β { − log L(β) + ½ Σ_{a=0}^{K-2} λ_a · β_a^T S β_a }
//! ```
//!
//! where `β = [β_0; β_1; …; β_{K-2}]` is the stacked coefficient vector in
//! output-major order (`β_a ∈ ℝ^P` is the coefficient block for class `a`),
//! `S ∈ ℝ^{P×P}` is the smoothing penalty matrix (shared across classes,
//! replicated as `I_{K-1} ⊗ S` over the full parameter space), and `λ_a` is
//! a per-class smoothing parameter.
//!
//! The likelihood uses class `K - 1` as the reference (`η_{K-1} ≡ 0`), so the
//! softmax gauge is fixed at the η level and no additional sum-to-zero
//! projection is required.
//!
//! # Layering
//!
//! * **Fixed-λ inner solve** — [`fit_penalized_multinomial`] is the canonical
//!   coefficient-space Newton solver at *given* smoothing parameters `λ`,
//!   built on the shared [`crate::families::penalized_vector_glm`] engine.
//!
//! * **REML / LAML smoothing-parameter selection** — [`fit_penalized_multinomial_formula`]
//!   routes through [`crate::families::custom_family::fit_custom_family_with_rho_prior`]
//!   so the per-active-class `λ_a` are selected by the outer REML/LAML loop;
//!   the caller's `init_lambda` is only a warm-start seed. The multinomial
//!   [`crate::families::multinomial_reml::MultinomialFamily`] `CustomFamily`
//!   impl calls the fixed-λ math above as its inner solve at each ρ trial and
//!   supplies the dense per-row Hessian block for the outer trace terms.
//!
//! * **Formula → design integration** — `build_formula_design_for_multinomial`
//!   parses the Wilkinson formula and assembles `X` and the per-term `S`
//!   blocks; the `fit_multinomial_formula_pyfunc` FFI shim wires the Python
//!   `gamfit.fit(..., family='multinomial')` entry straight to this path.
//!
//! # Convergence
//!
//! The damped-Newton-with-backtracking scaffold lives once in the shared
//! [`crate::families::penalized_vector_glm`] engine: at each iteration the
//! assembled penalized Hessian `H + I_{K-1} ⊗ (λ_a S)` is factored via faer's
//! symmetric-PD-with-fallback path, the full Newton step `δ = −H^{-1} ∇F` is
//! computed, and accepted with step halving if the objective fails to decrease
//! (up to a small backtracking budget). The convergence test is the relative
//! coefficient step norm `‖δ‖ / (1 + ‖β‖) ≤ tol`, matching the existing pyffi
//! reference path. This module is the softmax adapter over that engine: it
//! supplies the dense `(K-1)×(K-1)` Fisher block, the residual, and the
//! log-likelihood through [`MultinomialLogitLikelihood`], and owns the
//! class-count / simplex preconditions. The independent-binomial sibling
//! [`crate::families::binomial_multi`] is the same engine with a row-diagonal
//! Fisher block instead.

use crate::families::custom_family::{
    BlockwiseFitOptions, ParameterBlockState, PenaltyMatrix, fit_custom_family_with_rho_prior,
};
use crate::families::multinomial_reml::MultinomialFamily;
use crate::families::penalized_vector_glm::{PenalizedVectorGlmInputs, fit_penalized_vector_glm};
use crate::families::vector_response::{MultinomialLogitLikelihood, validate_multinomial_simplex};
use crate::inference::data::EncodedDataset;
use crate::inference::formula_dsl::parse_formula;
use crate::inference::model::ColumnKindTag;
use crate::model_types::EstimationError;
use crate::resource::ProblemHints;
use crate::solver::fit_orchestration::{
    FitConfig, build_termspec_with_geometry_and_overrides, resolved_resource_policy,
};
use crate::terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use crate::terms::term_builder::resolve_role_col;
use crate::types::ResponseColumnKind;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Solver-only numerical stabilization floor for the formula-driven
/// multinomial REML inner solve (gam#747).
///
/// Installed with [`RidgePolicy::solver_only`](crate::types::RidgePolicy::solver_only)
/// so it stabilizes the inner joint-Newton **linear solve** but never enters
/// the REML objective, the penalty log-determinant, or the Laplace Hessian.
///
/// What it does: the multinomial smoothing penalties are rank-deficient by
/// design (each smooth carries an unpenalized polynomial null space) and the
/// formula may add a fully unpenalized parametric term (`x3` / `body_mass`). On
/// near-separable hard labels the softmax curvature is ill-conditioned along
/// those directions, so the bare Newton step `H⁻¹∇` is huge. Lifting the
/// smallest Hessian eigenvalue to `δ` bounds the step (`‖(H+δI)⁻¹∇‖ ≤ ‖∇‖/δ`),
/// keeping the screening iterates finite without poisoning the softmax with
/// `inf − inf = NaN`.
///
/// What it deliberately does NOT do: it adds no `½·δ·‖β‖²` term to the
/// objective and no `δ`-shift to the REML log-determinant. The earlier
/// `explicit_stabilization_pospart` policy folded both into the criterion,
/// which made `1e-4` a fixed-λ Gaussian prior that shrank every identified
/// coefficient off the MLE and biased smoothing-parameter selection — a value
/// that had to be tuned *between* under-stabilization (NaN seeds) and
/// over-shrinkage (lost VGAM match). As a solver-only floor that tradeoff is
/// gone: the over-shrinkage failure mode cannot occur (nothing is shrunk), the
/// optimized objective is the true penalized REML criterion, and the floor
/// only has to be large enough to keep the linear algebra finite.
///
/// The separation defect (#753) is no longer this floor's job. If the
/// multinomial MLE is genuinely at infinity for an unpenalized/null-space
/// direction (complete/quasi-complete separation), no solver floor makes that
/// direction's estimate finite. The formula REML path arms the full-span
/// Jeffreys/Firth correction CONDITIONALLY — only on separation evidence (see
/// [`multinomial_formula_separation_evidence`] and the two-attempt logic in
/// [`fit_penalized_multinomial_formula`]) — so an interior, well-identified fit
/// optimizes the unbiased penalized-REML criterion with no Firth shrinkage
/// toward the uniform simplex, while a (quasi-)separated geometry gets the
/// proper prior that is the only thing able to bound its penalty-null
/// directions (#715 real-data arm). The bare fixed-λ inner driver
/// [`fit_penalized_multinomial`] (no outer REML, no Jeffreys term) surfaces the
/// explicit `MultinomialSeparationDetected` diagnostic for the path that has no
/// proper prior to lean on.
const MULTINOMIAL_FORMULA_RIDGE_FLOOR: f64 = 1.0e-4;

/// Inner joint-Newton KKT tolerance for the multinomial formula path.
///
/// The softmax Fisher weight `W = diag(p) − ppᵀ` collapses on saturated rows,
/// so near-separable fits (penguins, #715) reach the OBJECTIVE's f64 noise
/// floor before the default `inner_tol = 1e-6` KKT target: measured on the
/// penguins arm (standardized columns), the trust region collapses to 1e-12
/// with per-attempt objective changes of ~+2e-9 on |obj| ≈ 1e2 (≈ 1e-11
/// relative — pure rounding) while the KKT residual plateaus at 2.8e-5–9.4e-5
/// against a scaled tolerance of ~1.9e-5. Demanding a residual below the
/// floating-point noise floor is certifiable-never: every eval is rejected by
/// the stall guard and the whole fit fails. `1e-5` certifies the measured
/// plateaus while still resolving β to ~1e-6 in the relevant metric — the
/// LAML criterion consumes β̂ with error O(residual²/curvature), far below
/// any quantity the outer ρ-search can read.
const MULTINOMIAL_FORMULA_INNER_TOL: f64 = 1.0e-5;

/// Formula-adapter penalty calibration for multinomial softmax REML.
///
/// The term builder's normalized penalties are calibrated on single-response
/// Gaussian-style score curvature. A reference-coded softmax class block sees
/// per-row active-class Fisher diagonal `p_a(1-p_a)` plus negative cross-class
/// coupling. At the neutral simplex (`p_k = 1/K`) the active diagonal is
/// `(K-1)/K²`, so the binary-logit calibration is `2·(K-1)/K² = 1/2` and the
/// three-class calibration is `4/9` rather than the historical hard-coded
/// `1/2`. Making the scale a function of `K` keeps the physical smoothness
/// prior tied to the likelihood curvature instead of over-penalizing every
/// class as the simplex gains categories.
fn multinomial_formula_penalty_scale(n_classes: usize) -> f64 {
    let k = n_classes.max(2) as f64;
    2.0 * (k - 1.0) / (k * k)
}

/// Largest smoothing-parameter dimension where exact dense outer curvature is
/// still worth paying for multinomial formula fits.
///
/// `D = (K - 1) * n_penalties`. Medium-size loaded models use exact curvature
/// so the optimizer does not wander into over-smoothed lambda caps on
/// near-boundary softmax surfaces. The threshold was originally calibrated at
/// `D <= 6` when each `s()` term carried ONE penalty; the double-penalty
/// migration (wiggliness + null-space shrinkage per term, mgcv `select=TRUE`
/// semantics) doubled `D` for the SAME models, silently flipping the
/// reference formula fits (2 smooths, K = 3: old `D = 4`, now `D = 8`) onto
/// the gradient-only route — where the #715 quality arm showed every
/// wiggliness ρ driven onto the ±10 box bound (smooths collapsed toward their
/// polynomial null space, truth-RMSE behind VGAM). `12 = 2 × 6` preserves the
/// original classification boundary under the doubled penalty count while
/// keeping the four-smooth penguin species quality fixture on the exact ARC
/// path: that model is `D = 16`, and first-order BFGS can cycle along the
/// near-separable lambda-to-zero ridge until the wall-clock budget expires
/// (#1082). ARC observes the same exact curvature and can halt through the
/// bound-aware cost-stall guard once the REML surface stops making useful
/// progress.
const MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM: usize = 16;

fn multinomial_formula_use_outer_hessian(total_rho_dim: usize) -> bool {
    total_rho_dim <= MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM
}

/// Logit magnitude beyond which fitted probabilities are saturated at ordinary
/// double precision diagnostic scale. The bare fixed-λ driver has no outer REML
/// state and still uses this threshold to reject a non-converged saturated
/// iterate as a separation artifact. The formula REML path does not use this as
/// a Firth trigger: with smoothing parameters selected, a finite saturated
/// surface can be the valid near-separated optimum that should be scored
/// directly.
const MULTINOMIAL_SEPARATION_ETA_THRESHOLD: f64 = 25.0;

/// Calibrated convergence tolerance for the OUTER REML/LAML smoothing-parameter
/// search on the formula multinomial path. Matches the primary GLM REML outer
/// (`solver::fit_orchestration::materialize` uses `tol = 1e-7`, mirrored by the
/// `LOG_LAMBDA_TOL` / `KKT_TOL_*` constants across the REML stack): tight enough
/// that the selected λ reaches the genuine REML optimum (the recovered
/// probability surface matches the mature reference), loose enough that the
/// optimizer does not grind surface-irrelevant ρ digits down to the inner KKT
/// scale (the #1082 wall-clock overrun). The caller's `tol` is floored at this
/// value for the OUTER loop, while it continues to drive the INNER joint-Newton
/// KKT target unchanged.
const MULTINOMIAL_OUTER_REML_TOL: f64 = 1e-7;

/// The first multinomial formula solve is a separation probe: it is accepted
/// when the unbiased REML criterion converges to a finite interior iterate.
/// Near-separable data such as the penguin fixture otherwise spend the caller's
/// full outer budget on an iterate that is discarded before the Firth/Jeffreys
/// refit. Keep enough iterations for ordinary interior fits to certify quickly,
/// but hand slow/non-interior probes to the proper-prior refit promptly.
const MULTINOMIAL_UNBIASED_PROBE_OUTER_MAX_ITER: usize = 20;

/// Flexible lower λ floor for a WELL-SUPPORTED class on the formula path:
/// smoothing parameters below this level are effectively at the zero-penalty
/// boundary, so the unbiased REML optimizer is held inside this box bound rather
/// than accepting a boundary-overfit surface or switching to Firth bias on
/// finite data. This is the floor in the large-support limit.
const MULTINOMIAL_FORMULA_MIN_LAMBDA: f64 = 2.0e-4;

/// Strong lower λ floor in the SPARSE-class limit, and the support scale at
/// which the floor begins to rise. With fewer rows in a class the softmax Fisher
/// information `JᵀWJ` restricted to that class is smaller, so a boundary-hugging
/// smooth calibrates worse on held-out data and wants more shrinkage. These two
/// constants are the empirically-calibrated ENDPOINTS of a continuous
/// information-scaled floor (see [`multinomial_formula_min_lambda`]); they are
/// not a hard count threshold.
const MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_COUNT: f64 = 50.0;
const MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_LAMBDA: f64 = 1.0e-3;

/// Continuous, information-scaled lower λ floor for the formula path.
///
/// The floor scales like the inverse of the minority-class support
/// (`floor ∝ 1/information ∝ 1/count`), NOT a discontinuous count threshold: a
/// class with 49 vs 50 rows must not see a 5× jump in its penalty floor. The
/// form `base · max(1, c0/c)`, clamped to `[base, sparse]`:
///   * reduces EXACTLY to `base` for well-supported classes (`c ≥ c0`);
///   * reduces EXACTLY to `sparse` for very sparse classes
///     (`c ≤ c0·base/sparse`, here `c ≤ 10`);
///   * interpolates monotonically and continuously between them in the middle.
/// It anchors on the two empirically-calibrated endpoints while removing the
/// cliff at `c = c0`. Fixtures whose smallest class has `c ≥ 50` (e.g. penguins)
/// are unaffected — they sit in the `base` regime exactly as before.
fn multinomial_formula_min_lambda(y_one_hot: ArrayView2<'_, f64>) -> f64 {
    let min_class_count = (0..y_one_hot.ncols())
        .map(|class| y_one_hot.column(class).sum())
        .fold(f64::INFINITY, f64::min);
    if !min_class_count.is_finite() || min_class_count <= 0.0 {
        return MULTINOMIAL_FORMULA_MIN_LAMBDA;
    }
    let information_scale =
        (MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_COUNT / min_class_count).max(1.0);
    (MULTINOMIAL_FORMULA_MIN_LAMBDA * information_scale).clamp(
        MULTINOMIAL_FORMULA_MIN_LAMBDA,
        MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_LAMBDA,
    )
}

fn max_abs_eta_location(eta: ArrayView2<'_, f64>) -> (f64, usize, usize) {
    let mut best = (0.0_f64, 0usize, 0usize);
    for ((row, active_class), &value) in eta.indexed_iter() {
        let abs = value.abs();
        if abs > best.0 {
            best = (abs, row, active_class);
        }
    }
    best
}

/// Separation gate for the REML/LAML **formula** path.
///
/// Unlike the bare fixed-λ driver [`fit_penalized_multinomial`] (which has no
/// outer REML state and so must reject a saturated, non-converged iterate as a
/// separation artifact at the [`MULTINOMIAL_SEPARATION_ETA_THRESHOLD`] logit
/// magnitude), the formula path can return a finite saturated mode after the
/// coupled outer optimizer has selected smoothing parameters. A `|η| >= 25`
/// gate is therefore wrong here: the penguins arm can legitimately have large
/// fitted logits while still producing finite probabilities and a usable REML
/// mode.
///
/// Only a genuinely NON-FINITE `η` (a NaN/Inf blow-up in the inner linear
/// algebra) is a real formula-path failure. A finite, even saturated, `η` is
/// accepted so the truth-recovery / match-or-beat bars are evaluated against the
/// actual fitted surface instead of an adapter diagnostic.
fn multinomial_formula_separation_diagnostic(
    inner_cycles: usize,
    outer_iterations: usize,
    block_states: &[ParameterBlockState],
) -> Option<EstimationError> {
    let mut nonfinite: Option<(f64, usize, usize)> = None;
    for (active_class, state) in block_states.iter().enumerate() {
        for (row, &value) in state.eta.iter().enumerate() {
            if !value.is_finite() {
                nonfinite = Some((value, row, active_class));
                break;
            }
        }
        if nonfinite.is_some() {
            break;
        }
    }
    nonfinite.map(|(value, row_index, active_class_index)| {
        EstimationError::MultinomialSeparationDetected {
            iteration: inner_cycles.max(outer_iterations),
            max_abs_eta: value.abs(),
            active_class_index,
            row_index,
        }
    })
}

/// Separation EVIDENCE gate for the conditional Firth/Jeffreys engagement on
/// the formula REML path (#715 / #753).
///
/// The structural mathematics (#715 issue thread): for any coefficient
/// direction `v` with `S v = 0` (a penalty-null direction — intercept, a
/// smooth's polynomial null component, an unpenalized parametric term), the
/// penalized joint Hessian satisfies `(H + S_λ) v = H v` for EVERY smoothing
/// parameter ρ. When the data (quasi-)separate, the softmax Fisher weight
/// `W = diag(p) − p pᵀ → 0` on the saturated rows, so `H v = JᵀWJ v → 0` along
/// the penalty-null directions those rows support: `(H + S_λ) v ≈ 0` for every
/// ρ — NO λ can repair it, the inner Newton can never certify a KKT point
/// there, and every outer REML startup seed is rejected (the penguins
/// real-data arm). The only principled cure is a PROPER prior on that
/// quotient-null subspace — the Jeffreys/Firth term `Φ = ½ log|ZᵀHZ|`, whose
/// Gauss–Newton curvature supplies the missing `O(1)` bound.
///
/// But the Firth prior is not free on interior data: unconditionally armed, it
/// shrinks fitted class probabilities toward the uniform simplex `1/K`
/// (an `O(1/n)` pull that the synthetic match-or-beat arm of #715 measured as
/// a real truth-RMSE loss vs the unbiased criterion). So the formula path
/// engages it ONLY on separation evidence, mirroring the #753 "diagnose, then
/// arm" split:
///
/// * a NON-FINITE logit — the inner linear algebra blew up along an unbounded
///   direction.
///
/// Returns `Some(description)` naming the witnessing logit when evidence is
/// found, `None` for a finite fit (which is then accepted as-is, with zero
/// Firth bias). A FAILED unbiased solve (`Err` from the rho-prior driver, e.g.
/// "no startup seed passed") is the second evidence form and is handled
/// directly at the call site in [`fit_penalized_multinomial_formula`].
fn multinomial_formula_separation_evidence(block_states: &[ParameterBlockState]) -> Option<String> {
    for (active_class, state) in block_states.iter().enumerate() {
        for (row, &value) in state.eta.iter().enumerate() {
            if !value.is_finite() {
                return Some(format!(
                    "non-finite logit eta[row {row}, active class {active_class}] = {value}"
                ));
            }
        }
    }
    None
}

/// Extra evidence used only for a NON-CONVERGED capped unbiased probe.
///
/// A converged finite saturated formula fit is still a valid optimum and must be
/// scored without Firth bias. A capped probe that failed to converge while it
/// already carries separation-scale logits is different: spending the full
/// unbiased outer budget on the same lambda-to-zero surface is the #1082
/// timeout. Route that case straight to the proper-prior refit.
fn multinomial_formula_unresolved_probe_separation_evidence(
    block_states: &[ParameterBlockState],
) -> Option<String> {
    if let Some(evidence) = multinomial_formula_separation_evidence(block_states) {
        return Some(evidence);
    }

    let mut best = (0.0_f64, 0usize, 0usize);
    for (active_class, state) in block_states.iter().enumerate() {
        for (row, &value) in state.eta.iter().enumerate() {
            let abs = value.abs();
            if abs > best.0 {
                best = (abs, row, active_class);
            }
        }
    }
    if best.0 >= MULTINOMIAL_SEPARATION_ETA_THRESHOLD {
        Some(format!(
            "separation-scale finite logit |eta[row {}, active class {}]| = {:.3e} \
             after capped unbiased probe",
            best.1, best.2, best.0
        ))
    } else {
        None
    }
}

/// Inputs to [`fit_penalized_multinomial`].
///
/// The penalty matrix `S` is shared across classes; per-class smoothing
/// parameters `lambdas` (length `K - 1`) scale `S` independently for each
/// active class. The full block-replicated penalty is `diag_a(λ_a) ⊗ S`,
/// which is exactly what [`crate::solver::arrow_schur::KroneckerPenaltyOp`]
/// expresses in matrix-free form when this driver is later lifted into the
/// arrow-Schur loop.
#[derive(Debug, Clone)]
pub struct MultinomialFitInputs<'a> {
    /// Design matrix `X ∈ ℝ^{N×P}` (one row per observation).
    pub design: ArrayView2<'a, f64>,
    /// Categorical response `Y ∈ ℝ^{N×K}`. Each row must be a point on the
    /// probability simplex (`y_c ≥ 0`, `Σ_c y_c = 1`): a one-hot indicator for
    /// hard classification, or a label-smoothed probability vector. Rows whose
    /// mass departs from 1 are rejected — the softmax residual gradient and
    /// Fisher block are the derivatives of `Σ_c y_c log p_c` only under the
    /// simplex constraint (see `validate_multinomial_simplex`).
    pub y_one_hot: ArrayView2<'a, f64>,
    /// Shared smoothing penalty `S ∈ ℝ^{P×P}` (symmetric, PSD).
    pub penalty: ArrayView2<'a, f64>,
    /// Per-active-class smoothing parameter `λ_a` (length `K - 1`).
    pub lambdas: ArrayView1<'a, f64>,
    /// Optional per-row weights (length `N`); `None` ⇒ uniform 1.0.
    pub row_weights: Option<ArrayView1<'a, f64>>,
    /// Optional per-row Fisher-block override, shape `(N, K-1, K-1)` in the
    /// active-class gauge (the reference class `K-1` is dropped). When `Some`,
    /// each Newton step uses this block as the curvature `W` in place of the
    /// analytic softmax Fisher `w_n (δ_ab p_a − p_a p_b)`; the gradient/residual
    /// path stays analytic, so this is a curvature-only override (the
    /// research escape-hatch for latent multinomial fits, issue #349). Each
    /// per-row block must be symmetric, PSD, and finite — preconditions the
    /// FFI boundary discharges before constructing this view.
    pub fisher_w_override: Option<ArrayView3<'a, f64>>,
    /// Maximum Newton iterations; recommend 50.
    pub max_iter: usize,
    /// Relative-step convergence tolerance; recommend 1e-7.
    pub tol: f64,
}

/// Outputs of [`fit_penalized_multinomial`].
#[derive(Debug, Clone)]
pub struct MultinomialFitOutputs {
    /// Active-class coefficient block, shape `(P, K-1)` (column `a` is `β_a`).
    /// The reference class `K - 1` has `β_{K-1} ≡ 0` by construction and is
    /// not stored.
    pub coefficients_active: Array2<f64>,
    /// Fitted probabilities, shape `(N, K)`.
    pub fitted_probabilities: Array2<f64>,
    /// Number of Newton iterations executed (including the final step that
    /// satisfied the tolerance).
    pub iterations: usize,
    /// `true` if the relative-step test was satisfied; `false` if the
    /// solver exhausted `max_iter`. (A non-converged solve is still
    /// returned; the caller decides whether to escalate.)
    pub converged: bool,
    /// Penalized negative log-likelihood at the returned `β̂`:
    /// `−log L(β̂) + ½ Σ_a λ_a · β̂_a^T S β̂_a`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)` for diagnostic reporting.
    pub deviance: f64,
}

/// Fit a penalized multinomial-logit GAM at fixed `λ`.
///
/// See the module docs for the optimization problem and conventions. This
/// function is the canonical inner solve: the outer REML/LAML loop, when
/// added, calls this at each `ρ = log λ` trial.
pub fn fit_penalized_multinomial(
    inputs: MultinomialFitInputs<'_>,
) -> Result<MultinomialFitOutputs, EstimationError> {
    let MultinomialFitInputs {
        design,
        y_one_hot,
        penalty,
        lambdas,
        row_weights,
        fisher_w_override,
        max_iter,
        tol,
    } = inputs;

    // ──────────────────────── family-specific validation ───────────────────
    // The shared engine re-validates the geometry common to every vector-GLM
    // (nonempty design, penalty shape, λ finiteness/non-negativity, override
    // `(N, M, M)` shape, finite design). The multinomial family owns the
    // class-count contract (`K ≥ 2`, λ length `K − 1`), the per-row simplex
    // precondition under which the softmax residual/Fisher are the exact
    // derivatives of `Σ_c y_c log p_c`, and the row-weight check the likelihood
    // adapter consumes.
    let n_obs = design.nrows();
    let (y_rows, k) = y_one_hot.dim();
    if y_rows != n_obs {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: y rows {y_rows} ≠ design rows {n_obs}"
        );
    }
    if k < 2 {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: need at least 2 classes (got K={k})"
        );
    }
    let m = k - 1;
    if lambdas.len() != m {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: lambdas length {} ≠ K-1 = {m}",
            lambdas.len()
        );
    }
    if let Some(fw) = fisher_w_override.as_ref() {
        if fw.dim() != (n_obs, m, m) {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: fisher_w_override shape {:?} ≠ (N, K-1, K-1) = ({n_obs}, {m}, {m})",
                fw.dim()
            );
        }
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n_obs {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: row_weights length {} ≠ N = {n_obs}",
                w.len()
            );
        }
        for (i, &v) in w.iter().enumerate() {
            if !(v.is_finite() && v >= 0.0) {
                crate::bail_invalid_estim!(
                    "fit_penalized_multinomial: row_weights[{i}] must be finite and ≥ 0 (got {v})"
                );
            }
        }
    }
    validate_multinomial_simplex(y_one_hot, "fit_penalized_multinomial")?;

    // ────────────────────────── likelihood construction ───────────────────
    let mut likelihood = MultinomialLogitLikelihood::with_classes(k)?;
    if let Some(w) = row_weights.as_ref() {
        likelihood = likelihood.with_row_weights(w.to_owned())?;
    }

    // ─────────────────── shared penalized vector-GLM solve ─────────────────
    // The softmax Fisher block is dense across the `M = K − 1` active classes;
    // the engine assembles the coupled `(P·M)×(P·M)` penalized Hessian, runs
    // the damped Newton loop, and returns the converged `β̂` and `η = X β̂`.
    let fit = fit_penalized_vector_glm(
        PenalizedVectorGlmInputs {
            design,
            y: y_one_hot,
            penalty,
            lambdas,
            fisher_w_override,
            max_iter,
            tol,
        },
        &likelihood,
        "fit_penalized_multinomial",
    )?;

    let (max_abs_eta, row_index, active_class_index) = max_abs_eta_location(fit.eta.view());
    if !fit.converged && max_abs_eta >= MULTINOMIAL_SEPARATION_ETA_THRESHOLD {
        return Err(EstimationError::MultinomialSeparationDetected {
            iteration: fit.iterations,
            max_abs_eta,
            active_class_index,
            row_index,
        });
    }

    let fitted_probabilities = likelihood.probabilities(fit.eta.view());

    Ok(MultinomialFitOutputs {
        coefficients_active: fit.coefficients,
        fitted_probabilities,
        iterations: fit.iterations,
        converged: fit.converged,
        penalized_neg_log_likelihood: -fit.log_likelihood + fit.penalty_term,
        deviance: -2.0 * fit.log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// Formula-driven multinomial pipeline
// ---------------------------------------------------------------------------
//
// Slice A of the multinomial integration: a single public entry that takes
// a parsed `EncodedDataset`, a Wilkinson-style formula, and a uniform initial
// smoothing parameter, then runs the full
//
//     parse → termspec → design (X, S blocks) → one-hot Y → REML λ-selection
//
// pipeline. `fit_penalized_multinomial_formula` drives the outer REML/LAML
// loop (via the custom-family path) to select an independent λ per (class,
// term); `init_lambda` (default 1.0) is only the warm-start seed for every
// block. The reference class is the last level of the categorical response
// column as recorded in the dataset schema.

/// Saved-model payload for a multinomial fit driven by a Wilkinson formula.
///
/// This is what the FFI returns to Python. It carries everything the Python
/// `MultinomialModel.predict` path needs to evaluate `softmax(X_new · β)` on
/// fresh data using the *training* basis / penalty structure (no refit on
/// predict, no re-derivation of class levels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialSavedModel {
    /// The training formula, verbatim. Stored so Python's `summary()` and
    /// any round-trip persistence path can echo what was fit.
    pub formula: String,
    /// Names of the *training* response levels in canonical order. The last
    /// entry is the reference class (η = 0); the first `K - 1` carry the
    /// active linear-predictor blocks. Class permutations are forbidden:
    /// this list is fixed at fit time and predictions emit columns in the
    /// same order.
    pub class_levels: Vec<String>,
    /// Index of the reference class within `class_levels` — currently always
    /// `class_levels.len() - 1`, exposed as a field so future "user-pinned
    /// reference" gauges (e.g. `family='multinomial', reference='setosa'`)
    /// can land without changing the on-disk shape.
    pub reference_class_index: usize,
    /// Resolved term-collection spec used to build `X` at fit time. Replayed
    /// on predict via [`crate::terms::smooth::build_term_collection_design`].
    pub resolved_termspec: TermCollectionSpec,
    /// Active-class coefficient block, shape `(P, K-1)`. Column `a` is the
    /// coefficient vector for class `class_levels[a]`. Stored flat in
    /// row-major order to keep the serde payload self-describing.
    pub coefficients_flat: Vec<f64>,
    /// `P` — coefficient count per active class. Matches the column count of
    /// the design matrix the saved `resolved_termspec` produces.
    pub p_per_class: usize,
    /// Number of active classes (`K - 1`).
    pub n_active_classes: usize,
    /// Original training column headers, in dataset-column order. Needed at
    /// predict time so the FFI can align a fresh `Dataset` to the training
    /// schema before evaluating the basis.
    pub training_headers: Vec<String>,
    /// REML/LAML-selected smoothing parameters, one per `(active class, smooth
    /// term)`, flattened in block-major order: all of class 0's per-term λ,
    /// then class 1's, and so on. Per-term penalties (#561) mean each active
    /// class block selects an *independent* λ for every smooth term, so this
    /// vector has length `Σ_a (#terms in class a)` = `(K − 1) · #terms`. Use
    /// [`MultinomialSavedModel::lambdas_per_block`] to segment it by class. An
    /// unpenalized model (no smooth terms) yields an empty vector.
    pub lambdas: Vec<f64>,
    /// Number of smoothing parameters (smooth terms) in each active class
    /// block, parallel to `class_levels[0..K-1]`. Segments the flat `lambdas`
    /// vector: class `a`'s λ are `lambdas[Σ_{b<a} lambdas_per_block[b] ..][..
    /// lambdas_per_block[a]]`. Every entry is identical in the shared-design
    /// architecture (all classes share the same term structure), but it is
    /// stored explicitly so consumers never have to assume that.
    pub lambdas_per_block: Vec<usize>,
    /// Newton iterations executed; recorded for the summary report.
    pub iterations: usize,
    /// `true` if the inner Newton solver hit the relative-step tolerance.
    pub converged: bool,
    /// Penalized negative log-likelihood at the returned `β̂`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)`.
    pub deviance: f64,
    /// Per-active-class effective degrees of freedom (hat-matrix trace),
    /// length `K - 1`. Populated when the REML driver reports an
    /// inference block; falls back to `None` for the legacy fixed-λ path.
    #[serde(default)]
    pub edf_per_class: Option<Vec<f64>>,
    /// Joint posterior coefficient covariance `H⁻¹` (#1101), block-ordered to
    /// match the stacked active-class coefficient vector `β = [β_0; …; β_{K-2}]`
    /// (class `a`'s `P` coefficients occupy rows/cols `a·P .. (a+1)·P`). This is
    /// the Laplace covariance the REML driver already computes from the factored
    /// penalized Hessian; storing it gives the predict path delta-method
    /// per-class probability standard errors and the summary its Wald
    /// smooth-term tests. Flattened row-major over the `(P·M)×(P·M)` matrix.
    /// `None` for a model fitted before covariance was surfaced.
    #[serde(default)]
    pub coefficient_covariance_flat: Option<Vec<f64>>,
    /// Joint coefficient-space influence matrix `F = H⁻¹ X'WX` (#1101),
    /// block-ordered identically to [`Self::coefficient_covariance_flat`].
    /// Its per-term diagonal block trace is the term's effective degrees of
    /// freedom and its `tr(F_jj)²/tr(F_jj²)` the Wood reference d.f., feeding
    /// the rank-truncated Wald smooth-term test in `summary()`. Flattened
    /// row-major over the `(P·M)×(P·M)` matrix. `None` when unavailable.
    #[serde(default)]
    pub coefficient_influence_flat: Option<Vec<f64>>,
    /// Per-(active class, smooth term) coefficient column range and unpenalized
    /// nullspace dimension within the `P`-wide class block (#1101). Parallel to
    /// the smooth terms the design produced; replicated across classes by the
    /// shared-design architecture. Drives the Wald smooth-term table in
    /// `summary()`. Empty for a wholly parametric (no-smooth) model.
    #[serde(default)]
    pub smooth_term_spans: Vec<MultinomialSmoothTermSpan>,
}

/// One smooth term's coefficient span within a class block, plus its
/// unpenalized nullspace dimension and a display label (#1101). The Wald
/// smooth-significance test in `summary()` slices the joint covariance /
/// influence at `a·P + col_start .. a·P + col_end` for active class `a`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialSmoothTermSpan {
    /// Human-readable term label (the smooth's formula token), for the table.
    pub label: String,
    /// Start column of the term within the per-class `P`-wide coefficient block.
    pub col_start: usize,
    /// End column (exclusive) of the term within the per-class block.
    pub col_end: usize,
    /// Leading unpenalized (polynomial nullspace) dimension within the term.
    pub nullspace_dim: usize,
}

impl MultinomialSavedModel {
    /// Active-class coefficient block as an `(P, K-1)` `ndarray` view.
    pub fn coefficients_active(&self) -> Array2<f64> {
        Array2::from_shape_vec(
            (self.p_per_class, self.n_active_classes),
            self.coefficients_flat.clone(),
        )
        .expect(
            "MultinomialSavedModel.coefficients_flat length must equal p_per_class * n_active_classes",
        )
    }

    /// Evaluate `softmax(X · β)` at fresh data rows. `X_new` must have
    /// `self.p_per_class` columns (i.e. it was built from the same
    /// `resolved_termspec` as fit time). Returns an `(N_new, K)` matrix
    /// with rows summing to 1; column order matches `self.class_levels`.
    pub fn predict_probabilities(&self, x_new: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_new = x_new.nrows();
        let p = self.p_per_class;
        let m = self.n_active_classes;
        let k = m + 1;
        assert_eq!(
            x_new.ncols(),
            p,
            "MultinomialSavedModel.predict_probabilities: X has {} cols, expected {p}",
            x_new.ncols()
        );
        let beta = self.coefficients_active();
        let mut probs = Array2::<f64>::zeros((n_new, k));
        let mut eta_active = vec![0.0_f64; m];
        let mut row_probs = vec![0.0_f64; k];
        for row in 0..n_new {
            for a in 0..m {
                let mut v = 0.0_f64;
                for i in 0..p {
                    v += x_new[[row, i]] * beta[[i, a]];
                }
                eta_active[a] = v;
            }
            MultinomialLogitLikelihood::softmax_with_baseline(&eta_active, &mut row_probs);
            for c in 0..k {
                probs[[row, c]] = row_probs[c];
            }
        }
        probs
    }

    /// Reconstruct the joint posterior covariance `H⁻¹` as a `(P·M)×(P·M)`
    /// `ndarray`, block-ordered to match the stacked coefficient vector
    /// `θ[a·P + i] = β[i, a]` (#1101). `None` when the model was fitted before
    /// covariance was surfaced (legacy payload).
    pub fn coefficient_covariance(&self) -> Option<Array2<f64>> {
        let d = self.p_per_class.checked_mul(self.n_active_classes)?;
        let flat = self.coefficient_covariance_flat.as_ref()?;
        Array2::from_shape_vec((d, d), flat.clone()).ok()
    }

    /// Reconstruct the joint influence matrix `F = H⁻¹ X'WX` as a
    /// `(P·M)×(P·M)` `ndarray`, block-ordered like
    /// [`Self::coefficient_covariance`] (#1101). `None` when unavailable.
    pub fn coefficient_influence(&self) -> Option<Array2<f64>> {
        let d = self.p_per_class.checked_mul(self.n_active_classes)?;
        let flat = self.coefficient_influence_flat.as_ref()?;
        Array2::from_shape_vec((d, d), flat.clone()).ok()
    }

    /// Evaluate `softmax(X·β)` AND its delta-method per-class probability
    /// standard error at fresh data rows (#1101).
    ///
    /// For active classes `b ∈ 0..M` the softmax Jacobian is
    /// `∂p_c/∂η_b = p_c (δ_{cb} − p_b)`, and `∂η_b/∂β[i,a] = X[i]·δ_{ab}`, so the
    /// gradient of class-`c` probability w.r.t. the block-ordered coefficient
    /// vector is `g_c[a·P + i] = X[i]·p_c (δ_{ca} − p_a)` (active `a`; the
    /// reference class `M` contributes `p_c(0 − p_a)` via every active block).
    /// The delta-method variance is `Var(p_c) = g_cᵀ Σ g_c` with `Σ = H⁻¹` the
    /// joint posterior covariance, and `SE(p_c) = √Var(p_c)`. Returns
    /// `(probs (N,K), prob_se (N,K))`; `prob_se` is `None` when no covariance is
    /// stored. The simplex `[0,1]` clamp is applied by the interval consumer, not
    /// here (the SE itself is unclamped).
    pub fn predict_probabilities_with_se(
        &self,
        x_new: ArrayView2<'_, f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        let probs = self.predict_probabilities(x_new);
        let Some(cov) = self.coefficient_covariance() else {
            return (probs, None);
        };
        let n_new = x_new.nrows();
        let p = self.p_per_class;
        let m = self.n_active_classes;
        let k = m + 1;
        let d = p * m;
        let mut prob_se = Array2::<f64>::zeros((n_new, k));
        let mut grad = vec![0.0_f64; d];
        for row in 0..n_new {
            let prow = probs.row(row);
            for c in 0..k {
                let pc = prow[c];
                // g_c[a·P + i] = X[i] · p_c · (δ_{ca} − p_a), a active.
                for a in 0..m {
                    let pa = prow[a];
                    let factor = pc * (if c == a { 1.0 - pa } else { -pa });
                    let base = a * p;
                    for i in 0..p {
                        grad[base + i] = x_new[[row, i]] * factor;
                    }
                }
                // Var = gᵀ Σ g.
                let mut var = 0.0_f64;
                for r in 0..d {
                    let gr = grad[r];
                    if gr == 0.0 {
                        continue;
                    }
                    let mut acc = 0.0_f64;
                    for s in 0..d {
                        acc += cov[[r, s]] * grad[s];
                    }
                    var += gr * acc;
                }
                prob_se[[row, c]] = var.max(0.0).sqrt();
            }
        }
        (probs, Some(prob_se))
    }

    /// Wood (2013) rank-truncated Wald smooth-significance test per
    /// `(active class, smooth term)` (#1101), reusing the exact scalar-summary
    /// kernel [`crate::inference::smooth_test::wood_smooth_test`]. For active
    /// class `a` and term span `[c0, c1)` within the class block, the global
    /// coefficient range is `a·P + c0 .. a·P + c1`; the joint covariance and
    /// influence are sliced there. The term EDF is the influence-block trace
    /// `tr(F_jj)` (when present) and the reference d.f. uses `tr(F_jj)²/tr(F_jj²)`,
    /// exactly as the scalar path. The multinomial softmax is a known-dispersion
    /// family, so the χ²_{ref_df} branch applies. Returns one row per
    /// `(class label, term label, edf, ref_df, statistic, p_value)`; empty when
    /// no covariance/smooth terms are available.
    pub fn smooth_significance(&self) -> Vec<MultinomialSmoothSignificance> {
        let mut out = Vec::new();
        let p = self.p_per_class;
        let m = self.n_active_classes;
        let Some(cov) = self.coefficient_covariance() else {
            return out;
        };
        if self.smooth_term_spans.is_empty() {
            return out;
        }
        let beta = self.coefficients_active();
        // Block-ordered θ = [β_0; …; β_{M-1}], θ[a·P + i] = β[i, a].
        let d = p * m;
        let mut theta = Array1::<f64>::zeros(d);
        for a in 0..m {
            for i in 0..p {
                theta[a * p + i] = beta[[i, a]];
            }
        }
        let influence = self.coefficient_influence();
        for a in 0..m {
            let class_label = self
                .class_levels
                .get(a)
                .cloned()
                .unwrap_or_else(|| format!("class{a}"));
            let base = a * p;
            for span in &self.smooth_term_spans {
                if span.col_end > p {
                    continue;
                }
                let start = base + span.col_start;
                let end = base + span.col_end;
                // Term EDF = tr(F_jj); without an influence matrix fall back to
                // the block coefficient count (full-rank Wald on the span).
                let block_len = (span.col_end - span.col_start) as f64;
                let edf = influence
                    .as_ref()
                    .map(|f| (start..end).map(|i| f[[i, i]]).sum::<f64>())
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(block_len);
                let result = crate::inference::smooth_test::wood_smooth_test(
                    crate::inference::smooth_test::SmoothTestInput {
                        beta: theta.view(),
                        covariance: &cov,
                        influence_matrix: influence.as_ref(),
                        coeff_range: start..end,
                        edf,
                        nullspace_dim: span.nullspace_dim,
                        residual_df: f64::INFINITY,
                        scale: crate::inference::smooth_test::SmoothTestScale::Known,
                    },
                );
                if let Some(res) = result {
                    out.push(MultinomialSmoothSignificance {
                        class_label: class_label.clone(),
                        term_label: span.label.clone(),
                        edf,
                        ref_df: res.ref_df,
                        statistic: res.statistic,
                        p_value: res.p_value,
                    });
                }
            }
        }
        out
    }
}

/// One row of the multinomial smooth-significance table (#1101): the Wood
/// rank-truncated Wald test for one `(active class, smooth term)` pair.
#[derive(Debug, Clone)]
pub struct MultinomialSmoothSignificance {
    pub class_label: String,
    pub term_label: String,
    pub edf: f64,
    pub ref_df: f64,
    pub statistic: f64,
    pub p_value: f64,
}

/// One-hot-encode the categorical response column and return both the
/// encoding and the captured level names. The level order matches the order
/// recorded in the dataset schema, which is the canonical (lexicographically
/// sorted) factor order produced by inferred-schema construction (#1319) — so
/// it is a deterministic function of the label *set*, independent of training
/// row order (no silent class permutation under a row shuffle), and matches the
/// R `factor()` / pandas `Categorical` convention.
fn one_hot_categorical_response(
    data: &EncodedDataset,
    y_col: usize,
    response_name: &str,
) -> Result<(Array2<f64>, Vec<String>), EstimationError> {
    let levels: Vec<String> = data
        .schema
        .columns
        .get(y_col)
        .map(|sc| sc.levels.clone())
        .unwrap_or_default();
    if levels.len() < 2 {
        crate::bail_invalid_estim!(
            "multinomial response '{response_name}' must have at least 2 categorical levels (got {})",
            levels.len()
        );
    }
    let n = data.values.nrows();
    let k = levels.len();
    let mut y_one_hot = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        let encoded = data.values[[row, y_col]];
        if !encoded.is_finite() {
            crate::bail_invalid_estim!(
                "multinomial response '{response_name}' row {row} is non-finite ({encoded})"
            );
        }
        let class_idx = encoded.round() as i64;
        if class_idx < 0 || (class_idx as usize) >= k {
            crate::bail_invalid_estim!(
                "multinomial response '{response_name}' row {row} encoded as {encoded} \
                 is outside the level range 0..{k}"
            );
        }
        y_one_hot[[row, class_idx as usize]] = 1.0;
    }
    Ok((y_one_hot, levels))
}

/// Build `(TermCollectionSpec, TermCollectionDesign)` from a formula against
/// a categorical-response dataset. Mirrors the early scaffolding inside
/// `materialize_standard` (response role resolution, geometry-aware spec
/// build) without touching the scalar-family resolution path — multinomial
/// owns its own response kind check.
fn build_formula_design_for_multinomial(
    formula: &str,
    data: &EncodedDataset,
    config: &FitConfig,
) -> Result<
    (
        TermCollectionSpec,
        TermCollectionDesign,
        usize,
        String,
        ResponseColumnKind,
    ),
    EstimationError,
> {
    let parsed = parse_formula(formula).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "multinomial fit: failed to parse formula {formula:?}: {err}"
        ))
    })?;
    let col_map = data.column_map();
    let y_col = resolve_role_col(&col_map, &parsed.response, "response")
        .map_err(|err| EstimationError::InvalidInput(format!("multinomial fit: {err}")))?;
    let y_kind = crate::solver::fit_orchestration::response_column_kind(data, y_col);
    let policy = resolved_resource_policy(config, data, ProblemHints::default());
    let mut inference_notes: Vec<String> = Vec::new();
    let spec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        &col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )
    .map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build termspec: {err}"))
    })?;
    let design = build_term_collection_design(data.values.view(), &spec).map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build design: {err}"))
    })?;
    Ok((spec, design, y_col, parsed.response, y_kind))
}

fn scale_multinomial_formula_penalty(penalty: PenaltyMatrix, scale: f64) -> PenaltyMatrix {
    match penalty {
        PenaltyMatrix::Dense(matrix) => PenaltyMatrix::Dense(matrix.mapv(|v| v * scale)),
        PenaltyMatrix::KroneckerFactored { left, right } => PenaltyMatrix::KroneckerFactored {
            left: left.mapv(|v| v * scale),
            right,
        },
        PenaltyMatrix::Blockwise {
            local,
            col_range,
            total_dim,
        } => PenaltyMatrix::Blockwise {
            local: local.mapv(|v| v * scale),
            col_range,
            total_dim,
        },
        PenaltyMatrix::Labeled { label, inner } => PenaltyMatrix::Labeled {
            label,
            inner: Box::new(scale_multinomial_formula_penalty(*inner, scale)),
        },
        PenaltyMatrix::Fixed { log_lambda, inner } => PenaltyMatrix::Fixed {
            log_lambda,
            inner: Box::new(scale_multinomial_formula_penalty(*inner, scale)),
        },
    }
}

/// Build a warm-started copy of `blocks` whose per-block `initial_log_lambdas`
/// are seeded from a previously-selected flat `log_lambdas` vector (#1082).
///
/// The flat `log_lambdas` returned by [`fit_custom_family_with_rho_prior`]
/// concatenates each block's penalty log-λ in block order — the same order
/// `build_block_specs()` emits the blocks and the same per-block penalty order
/// the spec carries — so it splits back across blocks by each block's penalty
/// count. Warm-starting the OUTER ρ-search from a prior iterate changes only the
/// optimizer's starting point, never the penalized objective or its optimum, so
/// the converged fit is identical; it just resumes near the prior iterate
/// instead of restarting from the cold `init_lambda` seed.
///
/// Returns `None` (caller falls back to the cold blocks) if the flat vector does
/// not have exactly one entry per penalty across all blocks, or carries a
/// non-finite value — i.e. anything that would make the seed unsafe.
fn warm_start_blocks_from_log_lambdas(
    blocks: &[crate::custom_family::ParameterBlockSpec],
    log_lambdas: &[f64],
) -> Option<Vec<crate::custom_family::ParameterBlockSpec>> {
    let total: usize = blocks.iter().map(|b| b.initial_log_lambdas.len()).sum();
    if total == 0 || log_lambdas.len() != total {
        return None;
    }
    if log_lambdas.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut warm = blocks.to_vec();
    let mut offset = 0usize;
    for block in warm.iter_mut() {
        let k = block.initial_log_lambdas.len();
        for slot in 0..k {
            block.initial_log_lambdas[slot] = log_lambdas[offset + slot];
        }
        offset += k;
    }
    Some(warm)
}

/// Top-level formula-driven multinomial fit.
///
/// Routes through [`fit_custom_family_with_rho_prior`] so the per-active-class
/// smoothing parameters `λ_a` (one per class block, shared-penalty
/// architecture) are selected by the outer REML/LAML loop rather than pinned
/// by the caller. `init_lambda` survives as a warm-start hint that seeds
/// every block's `initial_log_lambdas`. `max_iter` / `tol` drive the OUTER
/// REML/LAML smoothing-parameter search (`outer_max_iter` / `outer_tol`); the
/// inner joint-Newton solve runs on the framework's principled production cycle
/// budget at the default KKT tolerance so an ill-conditioned, LM-damped
/// near-simplex-boundary solve can certify a stationary point instead of being
/// declared non-converged after only `max_iter` cycles (#715).
///
/// The Jeffreys/Firth proper prior is engaged CONDITIONALLY: attempt 1 runs
/// the unbiased penalized-REML criterion; only on separation evidence (a failed
/// solve or a non-finite logit; see [`multinomial_formula_separation_evidence`])
/// is the fit re-solved once with the full-span Firth prior armed, which bounds
/// the penalty-null directions no smoothing parameter can (`S v = 0` ⇒
/// `(H + S_λ) v = H v → 0` when the softmax likelihood has no finite mode).
///
/// The categorical response column is recognised via the dataset schema
/// (`ColumnKindTag::Categorical`); reference class = last level. Returns a
/// [`MultinomialSavedModel`] that can be serialised to bytes for the Python
/// wrapper or used in-process for `predict_probabilities`.
pub fn fit_penalized_multinomial_formula(
    data: &EncodedDataset,
    formula: &str,
    config: &FitConfig,
    init_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<MultinomialSavedModel, EstimationError> {
    if !(init_lambda.is_finite() && init_lambda > 0.0) {
        crate::bail_invalid_estim!(
            "multinomial fit: init_lambda must be finite and > 0 (got {init_lambda})"
        );
    }
    let (raw_spec, design, y_col, response_name, y_kind) =
        build_formula_design_for_multinomial(formula, data, config)?;
    // Freeze the data-derived basis state (B-spline knot vectors, by-factor
    // level sets, spatial centers, joint-null rotations, residualization
    // charts) from the fit design back onto the spec. The raw geometry spec
    // records only *which* columns and *what kind* of basis each smooth uses;
    // the actual column count and basis evaluation depend on quantities the
    // builder derives from the training data (knot placement, the distinct
    // by-factor levels, etc.). Saving the raw spec made predict re-derive those
    // from the (smaller, differently-distributed) predict frame, so the rebuilt
    // design had a different column count than the fitted one — the panic
    // "predict design has 42 cols, saved model expects 191" for an `s(x,
    // by=group)` smooth-by-factor model. Every other family's persistence path
    // freezes the spec the same way (see `freeze_term_collection_from_design`
    // call sites in `main_parts`); multinomial was the lone exception.
    let spec = freeze_term_collection_from_design(&raw_spec, &design)?;
    let class_levels = match y_kind {
        ResponseColumnKind::Categorical { levels } => levels,
        ResponseColumnKind::Binary => vec!["0".to_string(), "1".to_string()],
        ResponseColumnKind::Numeric => {
            crate::bail_invalid_estim!(
                "multinomial fit: response '{response_name}' is numeric, not categorical; \
                 use family='gaussian'/'binomial'/... or convert the column to a categorical type"
            );
        }
    };
    if data.column_kinds.get(y_col) == Some(&ColumnKindTag::Binary) {
        // Promote to a 2-level categorical for the multinomial driver; the
        // caller explicitly asked for multinomial, so we route through the
        // K-1 = 1 active-class softmax (equivalent math to logistic).
    } else if data.column_kinds.get(y_col) != Some(&ColumnKindTag::Categorical) {
        crate::bail_invalid_estim!(
            "multinomial fit: response '{response_name}' must be a categorical column \
             (got column kind {:?})",
            data.column_kinds.get(y_col)
        );
    }
    let (y_one_hot, _) = one_hot_categorical_response(data, y_col, &response_name)?;
    // Build the global X dense (the design is a DesignMatrix abstraction).
    let mut x_dense = design
        .design
        .try_to_dense_by_chunks("multinomial fit design")
        .map_err(EstimationError::InvalidInput)?;

    // ── #715 real-data conditioning: standardize unpenalized parametric
    // columns. Raw-unit linear covariates (penguins `body_mass_g` ~ 4e3 grams)
    // inflate the joint Newton information by the squared column scale (a κ(H)
    // multiplier of ~s² ≈ 1e7 against the intercept), which is what turns the
    // near-separable LM-damped inner solve into a geometric grind that
    // exhausts its cycle budgets — the adapter-level face of "all REML startup
    // seeds rejected". Because these columns are UNPENALIZED (parametric terms
    // carry no default ridge, #749), the affine reparameterization
    // `x_j ↦ (x_j − m_j)/s_j` is EXACT for the whole criterion: the optimized
    // REML/LAML objective, the fitted η, the selected λ, and the separation
    // diagnostics are all invariant — only the conditioning of `H` changes.
    // Fitted coefficients are mapped back to raw units at repack below, so the
    // saved model and the (raw-design) predict path are untouched. Penalized
    // columns are left alone (a penalty makes the rescaling non-equivalent),
    // and nothing is touched when explicit coefficient bounds/constraints
    // exist (those are stated in raw units).
    let parametric_standardization: Vec<(usize, f64, f64)> =
        if design.coefficient_lower_bounds.is_some() || design.linear_constraints.is_some() {
            Vec::new()
        } else {
            let p_total = x_dense.ncols();
            let mut penalized = vec![false; p_total];
            for bp in &design.penalties {
                for col in bp.col_range.clone() {
                    if col < p_total {
                        penalized[col] = true;
                    }
                }
            }
            let has_intercept = !design.intercept_range.is_empty();
            let n_rows = x_dense.nrows().max(1) as f64;
            let mut standardized = Vec::new();
            for (_, range) in &design.linear_ranges {
                for col in range.clone() {
                    if col >= p_total || penalized[col] {
                        continue;
                    }
                    let column = x_dense.column(col);
                    let mean = column.sum() / n_rows;
                    let var = column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n_rows;
                    let scale = var.sqrt();
                    // Skip near-constant or degenerate columns: no conditioning to
                    // be gained and the back-map would divide by ~0.
                    if !(scale.is_finite() && scale > 1e-8 * (mean.abs() + 1.0)) {
                        continue;
                    }
                    // Centering shifts mass onto the intercept; without one the
                    // shift is not representable, so scale only.
                    let center = if has_intercept { mean } else { 0.0 };
                    for v in x_dense.column_mut(col).iter_mut() {
                        *v = (*v - center) / scale;
                    }
                    standardized.push((col, center, scale));
                }
            }
            standardized
        };
    // Preserve the per-smooth-term penalty block structure (#561): each smooth
    // term `t` contributes its own `P × P` penalty component (`Blockwise` with
    // `total_dim = P`, the term's local `S_t` embedded at its `col_range`), and
    // every active class block receives the FULL list. The outer REML/LAML loop
    // then selects an independent smoothing parameter λ_{a,t} per (class, term),
    // matching mgcv/VGAM. Pre-summing the terms into one fused `S` (the prior
    // behaviour) forced a single λ per class that scales `Σ_t S_t`, so one
    // shared λ had to over-smooth a rough term while under-smoothing a smooth
    // one — biasing any multi-term class-probability surface.
    let k = y_one_hot.ncols();
    let m = k - 1;
    let n_obs = y_one_hot.nrows();
    let penalty_scale = multinomial_formula_penalty_scale(k);
    let per_term_penalties: Vec<PenaltyMatrix> = design
        .penalties_as_penalty_matrix()
        .into_iter()
        .map(|penalty| scale_multinomial_formula_penalty(penalty, penalty_scale))
        .collect();
    let per_term_nullspace_dims = design.nullspace_dims.clone();

    // ── Custom-family driven REML/LAML path ───────────────────────────────
    // Each active class becomes one ParameterBlockSpec, all sharing X and the
    // per-term penalty list. `initial_log_lambdas` is seeded from the caller's
    // `init_lambda` (one entry per term).
    let design_arc = Arc::new(x_dense);
    let penalties_arc = Arc::new(per_term_penalties);
    let nullspace_dims_arc = Arc::new(per_term_nullspace_dims);
    let weights = Array1::<f64>::ones(n_obs);
    // First attempt runs the UNBIASED penalized-REML criterion (no Firth
    // shrinkage toward the uniform simplex); the Jeffreys/Firth proper prior is
    // armed conditionally below, only on separation evidence (#715/#753 — see
    // `multinomial_formula_separation_evidence`).
    let family = MultinomialFamily::new(
        y_one_hot.clone(),
        weights,
        k,
        design_arc.clone(),
        penalties_arc.clone(),
        nullspace_dims_arc.clone(),
    )
    .map_err(EstimationError::InvalidInput)?
    .with_joint_jeffreys_term(false);
    let mut blocks = family.build_block_specs();
    let log_init = init_lambda.ln();
    for spec_block in blocks.iter_mut() {
        for v in spec_block.initial_log_lambdas.iter_mut() {
            *v = log_init;
        }
    }

    // ── Outer-derivative policy: dimension-gated exact curvature ────────────
    // The total smoothing-parameter dimension is `D = (K−1) · n_terms`.
    // Medium-D formula fits need exact curvature to keep lambda selection away
    // from over-smoothed caps, while smooth-by-factor `D = 8` models still avoid
    // the O(D²) dense Hessian path.
    let total_rho_dim = m.saturating_mul(penalties_arc.len());
    let use_outer_hessian = multinomial_formula_use_outer_hessian(total_rho_dim);

    // ── Inner-vs-outer control split (#715 non-convergence root cause) ────────
    // The legacy `max_iter` / `tol` parameters are the *outer* REML/LAML
    // smoothing-parameter optimization controls — "how hard to search λ". The
    // earlier wiring routed them straight into `inner_max_cycles` / `inner_tol`,
    // capping the joint-Newton inner solve at `max_iter` (=50 in the quality
    // suite) cycles with a `tol`-tight (=1e-8) KKT target. That is the #715
    // hang: near the simplex boundary the softmax Fisher weight
    // `W = diag(p) − p pᵀ` collapses, so `H = JᵀWJ + S_λ` is full-rank but
    // ILL-CONDITIONED. The self-vanishing Levenberg–Marquardt damping
    // (`levenberg_on_ill_conditioning()`) that keeps the inner solve from
    // oscillating on those near-singular modes makes it converge only
    // GEOMETRICALLY (linearly), not quadratically. Reaching a 1e-8 relative KKT
    // residual under geometric descent needs FAR more than 50 cycles, so the
    // inner returned `converged = false` on every outer ρ-evaluation; with the
    // exact-Hessian outer optimizer on `FallbackPolicy::Disabled` that rejects
    // every ρ-step — each rejected eval still paying a near-full 50-cycle inner
    // solve plus the O(D²) pairwise outer-Hessian directional work — so the
    // outer never certifies and the fit runs unbounded (the observed >8-minute
    // non-termination). The certificate cannot be reached, not merely slow.
    //
    // Fix: give the INNER joint-Newton the framework's principled production
    // budget (`DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES` cycles at the default
    // `inner_tol`), which exists precisely so an ill-conditioned LM-damped solve
    // can certify a stationary KKT point instead of being declared non-converged
    // prematurely — and the KKT/objective certificates still exit in a handful
    // of cycles on the well-conditioned interior fits, so this is free there.
    // The caller's `max_iter` / `tol` become the OUTER controls they were always
    // meant to be (smoothing-parameter search depth / accuracy). The inner KKT
    // target is kept no tighter than the outer accuracy can consume — and no
    // tighter than the softmax objective's f64 noise floor on near-separable
    // fits (see `MULTINOMIAL_FORMULA_INNER_TOL`).
    let outer_max_iter = max_iter.max(1);
    // The OUTER REML/LAML smoothing-parameter search must converge to a
    // well-calibrated ρ-gradient tolerance, NOT to the caller's (typically very
    // tight) INNER KKT tolerance. The #715 control-split repurposed the caller's
    // `tol` as the outer control, but feeding an inner-scale `tol = 1e-8`
    // straight into `outer_tol` makes REML grind dozens of extra exact-gradient
    // outer iterations (each an O(D·p³) Laplace-derivative assembly over the full
    // P·M joint design) to squeeze ρ digits that no longer move the fitted
    // surface — the smooth-by-factor 269s wall-clock overrun (#1082).
    //
    // The right target is the framework's CALIBRATED REML convergence tolerance,
    // `MULTINOMIAL_OUTER_REML_TOL = 1e-7` — the same value the primary GLM REML
    // outer uses (`solver::fit_orchestration::materialize` `tol: 1e-7`, mirrored by the
    // `LOG_LAMBDA_TOL`/`KKT_TOL_*` constants across the REML stack). At 1e-7 the
    // λ-search reaches the genuine REML optimum (so the recovered probability
    // surface matches the mature reference), but it does NOT chase the last
    // surface-irrelevant ρ digits down to 1e-8. The earlier 1e-5 floor (the
    // generic `BlockwiseFitOptions` default) was too LOOSE: the optimizer halted
    // in a low-curvature region with λ still well above its optimum, UNDER-fitting
    // the smooth-by-factor surface (truth-RMSE 0.164 vs VGAM's 0.061). So the
    // outer tolerance is floored at the calibrated REML tol — never tighter than
    // it (perf), never looser (accuracy) — while the caller's `tol` continues to
    // drive the INNER joint-Newton KKT target (`inner_tol` below), where its
    // precision actually matters.
    let outer_tol = if tol.is_finite() && tol > 0.0 {
        tol.max(MULTINOMIAL_OUTER_REML_TOL)
    } else {
        MULTINOMIAL_OUTER_REML_TOL
    };
    // #1082 root cause: the outer convergence test derives BOTH the absolute
    // projected-gradient floor (`max(outer_tol, n·1e-9)`) AND the relative-cost
    // stop (`rel_cost = outer_tol`) from the single `outer_tol`. The accuracy of
    // the smooth-by-factor surface is governed by the ABSOLUTE floor reaching the
    // n-scaled REML resolution `n·1e-9` (≈ 1.8e-6 at n = 1800) — that is why the
    // earlier 1e-5 floor UNDER-fit (its absolute floor was pinned at 1e-5, well
    // above the genuine optimum's gradient) and why 1e-7 recovered accuracy (it
    // unpins the floor down to the n-scaled 1.8e-6). But tightening `outer_tol`
    // to 1e-7 ALSO tightened the rel-cost stop to 1e-7, which on this family's
    // dead-flat REML ridge NEVER trips — so the optimizer no longer converges and
    // grinds all the way to `outer_max_iter`, each surplus step an O(D·p³) Laplace-
    // derivative assembly over the 382-dim joint design (the >600s wall-clock
    // overrun; tightening tol REINTRODUCED the crawl the 1e-5 floor had removed).
    //
    // The two requirements live on two different criteria, so they must be set
    // independently. Keep `outer_tol = 1e-7` (drives the accurate absolute floor)
    // but FLOOR the relative-cost stop at the framework default 1e-5 (the loose,
    // fast value that resolves the cost-decrease plateau without chasing the flat
    // tail). The absolute n·1e-9 floor still gates final λ accuracy; the rel-cost
    // stop just lets the optimizer DECLARE convergence on the flat ridge instead
    // of crawling to the iteration cap.
    let outer_rel_cost_tol = Some(BlockwiseFitOptions::default().outer_tol);
    let inner_tol = MULTINOMIAL_FORMULA_INNER_TOL.max(tol.max(0.0));

    let options = BlockwiseFitOptions {
        inner_max_cycles: crate::custom_family::DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
        inner_tol,
        outer_max_iter,
        outer_tol,
        outer_rel_cost_tol,
        rho_lower_bound: multinomial_formula_min_lambda(y_one_hot.view()).ln(),
        ridge_floor: MULTINOMIAL_FORMULA_RIDGE_FLOOR,
        // #747: the stabilization floor is SOLVER-ONLY — it keeps the inner
        // joint-Newton linear solve finite during screening (bounding the step
        // `(H+δI)⁻¹∇` away from a near-separable, rank-deficient curvature) but
        // is excluded from the REML objective, the penalty log-determinant, and
        // the Laplace Hessian. The earlier default (`explicit_stabilization_pospart`)
        // folded `½·δ·‖β‖²` and a `δ`-shift of the log-determinant into the
        // criterion, shrinking every identified coefficient off the MLE and
        // perturbing smoothing-parameter selection — a fixed-λ prior masking
        // separation, not a numerical stabilizer. With the floor solver-only the
        // optimized objective is the true penalized REML criterion (value tracks
        // its analytic gradient), and the smooth directions remain governed
        // solely by their own REML-selected `λ`.
        ridge_policy: crate::types::RidgePolicy::solver_only(),
        use_outer_hessian,
        // #715 real-data arm ("canonical-gauge null direction rejects all REML
        // seeds"): skip the multi-seed outer screening cascade and let the
        // pinned `init_lambda` ρ flow straight to the outer optimizer.
        //
        // The multinomial family declares `levenberg_on_ill_conditioning() ->
        // true`: near the simplex boundary (the near-separable penguins regime)
        // the softmax Fisher weight `W = diag(p) − p pᵀ → 0`, so the joint
        // information `H = JᵀWJ + S_λ` can become full-rank but
        // ILL-CONDITIONED. The self-vanishing LM damping that keeps the inner
        // joint-Newton from oscillating on those near-singular modes converges
        // only GEOMETRICALLY. The default screening policy ranks candidate seeds
        // with a 2-cycle inner cap (`outer_seed_config`); under geometric
        // LM-damped descent two cycles never reach a finite, meaningful proxy
        // objective, so EVERY capped seed can collapse to non-finite cost and
        // the cascade escalates to ×4, ×16, then an UNCAPPED full inner solve
        // PER SEED on the near-singular Hessian. That is the adapter-level face
        // of "all REML startup seeds rejected" and the multi-minute timeout.
        //
        // The pinned seed is already principled here: `init_lambda` gives every
        // (class, term) ρ a sensible moderate warm start, and the per-term
        // effective-df-floor upper bounds (`effective_df_floor_rho_upper_bounds`,
        // #715 arm (a)) keep any λ from collapsing the smooth onto its polynomial
        // null space. So the outer ARC/BFGS optimizer performs the real REML ρ
        // search from this seed; screening only adds the cascade cost and, on the
        // near-separable arm, the rejection stall.
        screen_initial_rho: false,
        // #1101: compute the joint Laplace posterior covariance `H⁻¹` (and the
        // influence matrix `F = H⁻¹ X'WX`) at the converged mode so the saved
        // model can surface delta-method per-class probability standard errors
        // and Wald smooth-term p-values. The driver factorizes the penalized
        // Hessian during the inner solve regardless; this only asks it to keep
        // and invert the factor instead of discarding it.
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    // ── Conditional Firth/Jeffreys engagement (#715 arm (b) / #753) ──────────
    // Attempt 1: the unbiased criterion (Jeffreys disarmed above). If the
    // returned mode is converged, finite, and interior, it is the exact penalized-REML
    // optimum with zero Firth bias — accept it (this is the synthetic-arm /
    // interior-data path, #715 arm (a)). If the solve FAILS (e.g. the
    // (quasi-)separated penguins geometry where `(H + S_λ)v ≈ 0` along
    // penalty-null directions for EVERY ρ rejects every REML startup seed) or
    // returns a non-finite artifact, that is direct separation evidence:
    // re-solve once with the full-span Jeffreys/Firth proper prior armed, which
    // supplies the O(1) curvature on the quotient-null subspace that smoothing
    // parameters mathematically cannot (`Sv = 0` ⇒ λ never touches `v`). The
    // Firth refit is the accepted result only when the unbiased formula solve
    // failed, did not converge on its full budget, or blew up; finite
    // formula-path logits can be large on valid near-separated optima and
    // should not be shrunk toward the uniform simplex once the unbiased outer
    // solve has actually certified.
    let mut unbiased_probe_options = options.clone();
    unbiased_probe_options.outer_max_iter = unbiased_probe_options
        .outer_max_iter
        .min(MULTINOMIAL_UNBIASED_PROBE_OUTER_MAX_ITER);
    // The FINAL accepted Firth/Jeffreys refit runs to the caller's full outer
    // budget: it is the result we ship, so it must reach the genuine REML
    // optimum, not a truncated iterate. The near-separable penguin refit that
    // motivated #1082's wall-clock concern is now halted honestly at its true
    // bound optimum by the KKT-stationary-at-bound guard
    // (`CostStallGuard`, #1082 / 64711ed82) and the Newton-decrement residual
    // certificate (363af9b56 / 2c9580b1f): on separable data the outer ARC
    // certifies and stops early on its own, so no artificial iteration cap is
    // needed to land in budget. On non-separable data (e.g. the
    // `vgam_smooth_by_factor` double-penalty arm) the refit needs the caller's
    // full budget to converge, which a `.min(20)` cap would cut off — accepting
    // a non-converged fit, which is dishonest. So the refit keeps `options`
    // unchanged. Only the discarded unbiased separation probe above is capped.
    let firth_refit_options = &options;

    let run_firth_refit = |evidence: String| {
        let firth_family = family.clone().with_joint_jeffreys_term(true);
        fit_custom_family_with_rho_prior(
            &firth_family,
            &blocks,
            firth_refit_options,
            crate::types::RhoPrior::Flat,
        )
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "multinomial REML: Firth/Jeffreys-armed refit (separation evidence: \
                 {evidence}) failed: {err}"
            ))
        })
    };

    // #1082: the capped unbiased probe and the (separable-path) Firth decision
    // are driven by separation scans over the full P×M logit block. The previous
    // match recomputed `multinomial_formula_separation_evidence` /
    // `..._unresolved_probe_separation_evidence` in BOTH the match guard AND the
    // arm body — three to four full logit walks per fit, paid on the hot
    // near-separable penguin path where this branch fires every iterate. Run the
    // probe once, evaluate each scan once into a binding, and branch on the
    // precomputed results. Behaviour is identical (same scans, same order of
    // precedence: converged-interior, unresolved-probe-separation,
    // no-separation-needs-full-solve, otherwise-Firth); only the duplicate
    // O(n·classes) scans are removed.
    let probe_attempt = fit_custom_family_with_rho_prior(
        &family,
        &blocks,
        &unbiased_probe_options,
        crate::types::RhoPrior::Flat,
    );
    let fit = match probe_attempt {
        Ok(probe_fit) => {
            let separation = multinomial_formula_separation_evidence(&probe_fit.block_states);
            if probe_fit.outer_converged && separation.is_none() {
                // Interior, converged, no separation: accept the probe directly.
                probe_fit
            } else if let Some(evidence) =
                multinomial_formula_unresolved_probe_separation_evidence(&probe_fit.block_states)
            {
                // Non-converged probe already carrying separation-scale logits:
                // hand straight to the proper-prior Firth refit (do not spend the
                // full unbiased budget grinding the λ→0 separable ridge).
                run_firth_refit(format!(
                    "unbiased-criterion REML probe did not converge after {} outer iterations; {evidence}",
                    probe_fit.outer_iterations
                ))?
            } else if separation.is_none() {
                // Interior but the capped probe ran out of iterations without
                // certifying: re-solve at the caller's full outer budget.
                //
                // #1082 wall-clock: the capped probe is a strict prefix of this
                // solve from the same family/seed, so a COLD restart repeats the
                // probe's outer iterations. WARM-START the re-solve from the ρ the
                // probe already reached — seed each block's `initial_log_lambdas`
                // from the probe's selected `log_lambdas` (same block/penalty
                // order: the flat vector concatenates per-block penalties in block
                // order, exactly the order `build_block_specs()` emits them). This
                // changes only the optimizer's STARTING point, never the objective
                // or its optimum, but lets the full solve resume near the probe's
                // last iterate instead of crawling up from `init_lambda` again —
                // removing the probe-iterations double-pay on the non-separable
                // (e.g. `vgam_smooth_by_factor`) arm. If the probe's λ vector does
                // not line up with the block layout (it always should), fall back
                // to the cold `blocks` seed.
                let warm_blocks = warm_start_blocks_from_log_lambdas(
                    &blocks,
                    probe_fit.log_lambdas.as_slice().unwrap_or(&[]),
                );
                let resolve_blocks = warm_blocks.as_deref().unwrap_or(&blocks);
                match fit_custom_family_with_rho_prior(
                    &family,
                    resolve_blocks,
                    &options,
                    crate::types::RhoPrior::Flat,
                ) {
                    Ok(full_unbiased_fit) => {
                        let full_separation = multinomial_formula_separation_evidence(
                            &full_unbiased_fit.block_states,
                        );
                        if full_unbiased_fit.outer_converged && full_separation.is_none() {
                            full_unbiased_fit
                        } else {
                            let evidence = full_separation.unwrap_or_else(|| {
                                format!(
                                    "full unbiased-criterion REML solve did not converge after {} outer iterations",
                                    full_unbiased_fit.outer_iterations
                                )
                            });
                            run_firth_refit(evidence)?
                        }
                    }
                    Err(err) => run_firth_refit(format!(
                        "full unbiased-criterion REML solve failed: {err}"
                    ))?,
                }
            } else {
                // Probe converged (or capped) but shows interior separation
                // evidence: Firth refit using the already-computed scan.
                let evidence = separation.unwrap_or_else(|| {
                    format!(
                        "unbiased-criterion REML probe did not converge after {} outer iterations",
                        probe_fit.outer_iterations
                    )
                });
                run_firth_refit(evidence)?
            }
        }
        Err(err) => run_firth_refit(format!("unbiased-criterion REML solve failed: {err}"))?,
    };
    if let Some(err) = multinomial_formula_separation_diagnostic(
        fit.inner_cycles,
        fit.outer_iterations,
        &fit.block_states,
    ) {
        return Err(err);
    }

    // ── Repack coefficients (P, K-1) from per-block β vectors ─────────────
    if fit.blocks.len() != m {
        crate::bail_invalid_estim!(
            "multinomial REML: expected {m} fitted blocks (K-1), got {}",
            fit.blocks.len()
        );
    }
    let p_per_class = fit.blocks[0].beta.len();
    let mut coefficients_active = Array2::<f64>::zeros((p_per_class, m));
    for (a, block) in fit.blocks.iter().enumerate() {
        if block.beta.len() != p_per_class {
            crate::bail_invalid_estim!(
                "multinomial REML: block {a} has {} coefs, expected {p_per_class}",
                block.beta.len()
            );
        }
        for i in 0..p_per_class {
            coefficients_active[[i, a]] = block.beta[i];
        }
    }
    // Map the standardized-column coefficients back to raw units (the exact
    // inverse of the conditioning reparameterization above): β_raw = b/s, with
    // the centering mass `Σ_j b_j·m_j/s_j` returned to the intercept.
    if !parametric_standardization.is_empty() {
        let intercept_col = design.intercept_range.clone().next();
        for a in 0..m {
            let mut intercept_adjust = 0.0;
            for &(col, center, scale) in &parametric_standardization {
                if col < p_per_class {
                    let raw = coefficients_active[[col, a]] / scale;
                    coefficients_active[[col, a]] = raw;
                    intercept_adjust += raw * center;
                }
            }
            if let Some(i0) = intercept_col
                && i0 < p_per_class
            {
                coefficients_active[[i0, a]] -= intercept_adjust;
            }
        }
    }
    // Flatten every (class, term) smoothing parameter in block-major order
    // (class 0's terms, then class 1's, …). With per-term penalties each block
    // now carries one λ per smooth term, so a single λ per class would discard
    // the independent per-term selection that fixes #561. `lambdas_per_block`
    // segments the flat vector by class so callers can recover per-term λ.
    let lambdas_per_block: Vec<usize> = fit.blocks.iter().map(|b| b.lambdas.len()).collect();
    let lambdas_flat: Vec<f64> = fit
        .blocks
        .iter()
        .flat_map(|b| b.lambdas.iter().copied())
        .collect();
    let edf_per_class = fit.inference.as_ref().map(|info| info.edf_by_block.clone());
    let coefficients_flat: Vec<f64> = coefficients_active.iter().copied().collect();

    // #1101: surface the joint Laplace posterior covariance `H⁻¹` (block-ordered
    // [β_0; …; β_{K-2}]) and the influence matrix `F = H⁻¹ X'WX` the REML driver
    // computed at the converged mode. These power the predict path's delta-method
    // per-class probability standard errors and the summary's Wald smooth-term
    // tests. The joint matrices are `(P·M)×(P·M)`. The covariance is mapped back
    // to RAW units (see below) so it pairs with the raw predict design; the
    // influence is kept in the fitted basis (the Wald table only slices penalized
    // columns, which the standardization affine leaves identity-mapped).
    let expected_joint = p_per_class.saturating_mul(m);
    // The joint Hessian (and thus `H⁻¹`) was assembled in the STANDARDIZED
    // parametric basis used during fitting, while the saved coefficients and the
    // raw predict design are in raw units. Map the covariance to raw units with
    // the same exact affine reparameterization `β_raw = A β_std`: for each
    // standardized parametric column `col`, `β_raw[col] = β_std[col]/scale` and
    // the intercept absorbs `−Σ_col (center/scale)·β_std[col]`. So `A = I` except
    // `A[col,col] = 1/scale` and `A[i0,col] = −center/scale`, replicated
    // block-diagonally per active class, and `Cov_raw = A Cov_std Aᵀ`. With no
    // standardization (`parametric_standardization` empty) `A = I` and this is a
    // no-op. The smooth-term (penalized) columns are untouched by `A`, so the
    // Wald table's per-term blocks are identical in both bases.
    let intercept_col0 = design.intercept_range.clone().next();
    let build_per_class_affine = |amat: &mut Array2<f64>| {
        for &(col, center, scale) in &parametric_standardization {
            if col >= p_per_class {
                continue;
            }
            amat[[col, col]] = 1.0 / scale;
            if let Some(i0) = intercept_col0
                && i0 < p_per_class
            {
                amat[[i0, col]] = -center / scale;
            }
        }
    };
    let coefficient_covariance_flat = fit
        .covariance_conditional
        .as_ref()
        .filter(|c| c.nrows() == expected_joint && c.ncols() == expected_joint)
        .map(|cov_std| {
            if parametric_standardization.is_empty() {
                return cov_std.iter().copied().collect::<Vec<f64>>();
            }
            // Block-diagonal joint A (same per active class).
            let mut a_joint = Array2::<f64>::eye(expected_joint);
            let mut a_class = Array2::<f64>::eye(p_per_class);
            build_per_class_affine(&mut a_class);
            for a in 0..m {
                let base = a * p_per_class;
                for i in 0..p_per_class {
                    for j in 0..p_per_class {
                        a_joint[[base + i, base + j]] = a_class[[i, j]];
                    }
                }
            }
            let cov_raw = a_joint.dot(cov_std).dot(&a_joint.t());
            cov_raw.iter().copied().collect::<Vec<f64>>()
        });
    // The influence matrix `F = H⁻¹ X'WX = H⁻¹(H − S_λ) = I − H⁻¹ S_λ`. The
    // exact-Newton multinomial blocks carry no IRLS pseudo-data, so the generic
    // inference path does not export `coefficient_influence`; reconstruct it
    // exactly here from the joint covariance `H⁻¹` (above) and the REML-selected
    // per-(class, term) `λ` scaling the shared penalties. Block-diagonal `S_λ`:
    // class `a`'s block is `Σ_t λ_{a,t} · S_t`, embedded at `a·P .. (a+1)·P`.
    let coefficient_influence_flat = fit
        .covariance_conditional
        .as_ref()
        .filter(|c| c.nrows() == expected_joint && c.ncols() == expected_joint)
        .and_then(|hinv| {
            if fit.blocks.len() != m {
                return None;
            }
            // Joint S_λ (block-diagonal across active classes).
            let mut s_lambda = Array2::<f64>::zeros((expected_joint, expected_joint));
            for (a, block) in fit.blocks.iter().enumerate() {
                if block.lambdas.len() != penalties_arc.len() {
                    return None;
                }
                let base = a * p_per_class;
                for (t, pen) in penalties_arc.iter().enumerate() {
                    let lam = block.lambdas[t];
                    if lam == 0.0 {
                        continue;
                    }
                    let dense = pen.to_dense();
                    if dense.nrows() != p_per_class || dense.ncols() != p_per_class {
                        return None;
                    }
                    for i in 0..p_per_class {
                        for j in 0..p_per_class {
                            s_lambda[[base + i, base + j]] += lam * dense[[i, j]];
                        }
                    }
                }
            }
            // F = I − H⁻¹ S_λ.
            let hinv_s = hinv.dot(&s_lambda);
            let mut f = Array2::<f64>::eye(expected_joint);
            f -= &hinv_s;
            Some(f.iter().copied().collect::<Vec<f64>>())
        });

    // Per-(smooth term) coefficient span within a single class block, deduped by
    // col_range (the #561 double-penalty migration emits two penalty blocks per
    // term sharing one col_range; the Wald test covers the whole term block once).
    let mut smooth_term_spans: Vec<MultinomialSmoothTermSpan> = Vec::new();
    for (pen_idx, bp) in design.penalties.iter().enumerate() {
        let col_start = bp.col_range.start;
        let col_end = bp.col_range.end;
        if col_start >= col_end || col_end > p_per_class {
            continue;
        }
        if smooth_term_spans
            .iter()
            .any(|s| s.col_start == col_start && s.col_end == col_end)
        {
            continue;
        }
        let label = design
            .penaltyinfo
            .get(pen_idx)
            .and_then(|info| info.termname.clone())
            .unwrap_or_else(|| format!("s{pen_idx}"));
        let nullspace_dim = design
            .nullspace_dims
            .get(pen_idx)
            .copied()
            .unwrap_or(0)
            .min(col_end - col_start);
        smooth_term_spans.push(MultinomialSmoothTermSpan {
            label,
            col_start,
            col_end,
            nullspace_dim,
        });
    }

    // Unpenalized deviance read directly from the converged unpenalized
    // log-likelihood the rho-prior driver already computed (issue #348):
    // MultinomialFamily::evaluate sets FamilyEvaluation.log_likelihood =
    // log_lik(η, y) with no penalty term, and that value flows unchanged into
    // UnifiedFitResult.log_likelihood. This reproduces the legacy fixed-λ
    // path's `deviance = -2 · log_lik` contract bit-for-bit, so the previous
    // row-by-row η = Xβ rebuild and softmax recompute were pure dead work.
    let deviance = -2.0 * fit.log_likelihood;

    Ok(MultinomialSavedModel {
        formula: formula.to_string(),
        class_levels: class_levels.clone(),
        reference_class_index: class_levels.len() - 1,
        resolved_termspec: spec,
        coefficients_flat,
        p_per_class,
        n_active_classes: m,
        training_headers: data.headers.clone(),
        lambdas: lambdas_flat,
        lambdas_per_block,
        iterations: fit.inner_cycles,
        converged: fit.outer_converged,
        penalized_neg_log_likelihood: -fit.log_likelihood + 0.5 * fit.stable_penalty_term,
        deviance,
        edf_per_class,
        coefficient_covariance_flat,
        coefficient_influence_flat,
        smooth_term_spans,
    })
}

/// Replay the saved termspec to build the predict-time design on a fresh
/// dataset, then evaluate softmax probabilities. The predict dataset must carry
/// the same feature columns the training data did, matched **by name** — it need
/// not reproduce the training column order, and in particular need not carry the
/// response column (prediction is for label-free new data).
pub fn predict_multinomial_formula(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<Array2<f64>, EstimationError> {
    // The saved termspec stores feature columns as absolute indices into the
    // *training* table `[response, features...]`. Replaying it verbatim only
    // works if the predict frame reproduces that exact layout — i.e. carries the
    // (unknown, at predict time) response column in the same position. Realign
    // the indices onto this dataset's columns by name instead, so prediction
    // works on label-free new data exactly as every other family's predict path
    // does. The response column is simply never referenced by any term, so its
    // absence is a non-issue once resolution is by name (issue #803).
    let predict_columns = data.column_map();
    let realigned = model.resolved_termspec.remap_feature_columns(
        |index| -> Result<usize, EstimationError> {
            let name = model.training_headers.get(index).ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "multinomial predict: saved training column index {index} is out of bounds \
                     for {} training headers",
                    model.training_headers.len()
                ))
            })?;
            resolve_role_col(&predict_columns, name, "feature")
                .map_err(|err| EstimationError::InvalidInput(err.to_string()))
        },
    )?;
    let design = build_term_collection_design(data.values.view(), &realigned).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "multinomial predict: rebuild design from saved termspec: {err}"
        ))
    })?;
    let x_dense = design
        .design
        .try_to_dense_by_chunks("multinomial predict design")
        .map_err(EstimationError::InvalidInput)?;
    if x_dense.ncols() != model.p_per_class {
        crate::bail_invalid_estim!(
            "multinomial predict: predict design has {} cols, saved model expects {}",
            x_dense.ncols(),
            model.p_per_class
        );
    }
    Ok(model.predict_probabilities(x_dense.view()))
}

/// Predict class probabilities AND delta-method per-class probability standard
/// errors for a saved multinomial model on fresh data (#1101). Replays the
/// saved termspec to build the predict design exactly as
/// [`predict_multinomial_formula`], then applies the softmax-Jacobian delta
/// method against the stored joint posterior covariance. Returns
/// `(probs (N,K), prob_se (N,K) | None)`; `prob_se` is `None` for a legacy
/// model fitted before covariance was surfaced.
pub fn predict_multinomial_formula_with_se(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<(Array2<f64>, Option<Array2<f64>>), EstimationError> {
    let predict_columns = data.column_map();
    let realigned = model.resolved_termspec.remap_feature_columns(
        |index| -> Result<usize, EstimationError> {
            let name = model.training_headers.get(index).ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "multinomial predict: saved training column index {index} is out of bounds \
                     for {} training headers",
                    model.training_headers.len()
                ))
            })?;
            resolve_role_col(&predict_columns, name, "feature")
                .map_err(|err| EstimationError::InvalidInput(err.to_string()))
        },
    )?;
    let design = build_term_collection_design(data.values.view(), &realigned).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "multinomial predict: rebuild design from saved termspec: {err}"
        ))
    })?;
    let x_dense = design
        .design
        .try_to_dense_by_chunks("multinomial predict design")
        .map_err(EstimationError::InvalidInput)?;
    if x_dense.ncols() != model.p_per_class {
        crate::bail_invalid_estim!(
            "multinomial predict: predict design has {} cols, saved model expects {}",
            x_dense.ncols(),
            model.p_per_class
        );
    }
    Ok(model.predict_probabilities_with_se(x_dense.view()))
}

#[cfg(test)]
mod fisher_override_tests {
    use super::*;
    use ndarray::Array3;

    fn toy() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 15;
        let p = 2;
        let k = 3;
        let design =
            Array2::<f64>::from_shape_fn(
                (n, p),
                |(i, j)| {
                    if j == 0 { 1.0 } else { ((i + 2) as f64).cos() }
                },
            );
        let mut y = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            y[[i, i % k]] = 1.0;
        }
        let penalty = Array2::<f64>::eye(p);
        let lambdas = Array1::<f64>::from_elem(k - 1, 0.5);
        (design, y, penalty, lambdas)
    }

    #[test]
    fn fisher_override_none_reproduces_analytic() {
        // Issue #349: None override is exactly the analytic fit.
        let (design, y, penalty, lambdas) = toy();
        let mk = |over: Option<ndarray::ArrayView3<'_, f64>>| {
            fit_penalized_multinomial(MultinomialFitInputs {
                design: design.view(),
                y_one_hot: y.view(),
                penalty: penalty.view(),
                lambdas: lambdas.view(),
                row_weights: None,
                fisher_w_override: over,
                max_iter: 50,
                tol: 1.0e-9,
            })
            .expect("fit must succeed")
        };
        let a = mk(None);
        let b = mk(None);
        for (x, z) in a
            .coefficients_active
            .iter()
            .zip(b.coefficients_active.iter())
        {
            assert_eq!(x, z);
        }
    }

    #[test]
    fn fisher_override_wrong_shape_is_rejected() {
        let (design, y, penalty, lambdas) = toy();
        let n = design.nrows();
        let m = y.ncols(); // K, not K-1 — deliberately wrong
        let bad = Array3::<f64>::zeros((n, m, m));
        let err = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: Some(bad.view()),
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("wrong active-block shape must error");
        assert!(format!("{err}").contains("fisher_w_override shape"));
    }

    #[test]
    fn formula_outer_route_uses_exact_curvature_for_medium_d() {
        // The 2-smooth reference formula fit (K = 3, double-penalty terms) is
        // D = (K-1) * 2 terms * 2 penalties = 8 and needs exact curvature to
        // avoid over-smoothed lambda caps (#715 arm (a)).
        assert!(
            multinomial_formula_use_outer_hessian(8),
            "D=8 loaded multinomial fits need exact curvature to avoid over-smoothed lambda caps"
        );
        assert!(
            multinomial_formula_use_outer_hessian(12),
            "D=12 (3 double-penalty smooth terms, K=3) stays on exact curvature"
        );
    }

    #[test]
    fn formula_outer_route_uses_exact_curvature_for_d16_penguin_fixture() {
        // Four k=10 penguin smooths (K = 3) are D = 16 under double-penalty
        // terms. They must reach the exact ARC route so the #1082 cost-stall
        // halt is available on the near-separable lambda-to-zero ridge.
        assert!(
            multinomial_formula_use_outer_hessian(16),
            "D=16 multinomial fits need exact ARC curvature for the #1082 stall halt"
        );
    }

    #[test]
    fn formula_min_lambda_floor_is_continuous_and_information_scaled() {
        // Build a one-hot label matrix whose smallest class carries `count` rows.
        fn floor_for_min_count(count: usize) -> f64 {
            // Two classes: a large one (1000 rows) and a minority one (`count`).
            let n = 1000 + count;
            let mut y = Array2::<f64>::zeros((n, 2));
            for r in 0..1000 {
                y[[r, 0]] = 1.0;
            }
            for r in 1000..n {
                y[[r, 1]] = 1.0;
            }
            multinomial_formula_min_lambda(y.view())
        }

        // Well-supported (count >= c0=50) sits exactly at the flexible base floor.
        assert!((floor_for_min_count(50) - MULTINOMIAL_FORMULA_MIN_LAMBDA).abs() < 1e-18);
        assert!((floor_for_min_count(200) - MULTINOMIAL_FORMULA_MIN_LAMBDA).abs() < 1e-18);
        // Very sparse (count <= c0*base/sparse = 10) clamps to the strong floor.
        assert!((floor_for_min_count(10) - MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_LAMBDA).abs() < 1e-18);
        assert!((floor_for_min_count(5) - MULTINOMIAL_FORMULA_SPARSE_CLASS_MIN_LAMBDA).abs() < 1e-18);
        // No cliff at the old hard threshold: 49 vs 50 differ by < 5% (the old
        // step jumped 5x). Floor is monotone non-increasing in support.
        let f49 = floor_for_min_count(49);
        let f50 = floor_for_min_count(50);
        assert!(f49 >= f50 && f49 <= f50 * 1.05, "floor must be continuous across c0, got {f49} vs {f50}");
        let f25 = floor_for_min_count(25);
        assert!(
            f25 > f50 && f25 < floor_for_min_count(10),
            "mid-support floor must interpolate strictly between the two endpoints"
        );
    }

    #[test]
    fn formula_penalty_scale_tracks_softmax_fisher_curvature() {
        assert!(
            (multinomial_formula_penalty_scale(2) - 0.5).abs() < 1.0e-12,
            "binary-logit neutral-simplex curvature scale should remain at 1/2"
        );
        assert!(
            (multinomial_formula_penalty_scale(3) - 4.0 / 9.0).abs() < 1.0e-12,
            "three-class softmax penalties should be calibrated to 2*(K-1)/K^2"
        );
        assert!(
            multinomial_formula_penalty_scale(5) < multinomial_formula_penalty_scale(3),
            "active-class Fisher curvature decreases as the simplex gains classes"
        );
    }

    #[test]
    fn fixed_lambda_multinomial_reports_complete_separation() {
        let n = 90;
        let design = Array2::<f64>::from_shape_fn((n, 2), |(row, col)| match col {
            0 => 1.0,
            _ => -3.0 + 6.0 * (row as f64) / ((n - 1) as f64),
        });
        let mut y = Array2::<f64>::zeros((n, 3));
        for row in 0..n {
            let x = design[[row, 1]];
            let class = if x < -1.0 {
                0
            } else if x > 1.0 {
                1
            } else {
                2
            };
            y[[row, class]] = 1.0;
        }
        let penalty = Array2::<f64>::zeros((2, 2));
        let lambdas = Array1::<f64>::zeros(2);
        let err = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 80,
            tol: 1.0e-12,
        })
        .expect_err("complete softmax separation must be a hard diagnostic");
        assert!(
            matches!(err, EstimationError::MultinomialSeparationDetected { .. }),
            "expected MultinomialSeparationDetected, got {err:?}"
        );
        assert!(
            err.to_string().contains("separation"),
            "diagnostic should mention separation, got {err}"
        );
        assert!(
            err.to_string().contains("active class-"),
            "diagnostic should name the separated active class logit, got {err}"
        );
        assert!(
            !err.to_string().contains("binary outcomes"),
            "multinomial diagnostic must not reuse the binary separation text, got {err}"
        );
    }

    #[test]
    fn formula_multinomial_accepts_finite_saturated_logits() {
        // A saturated-but-FINITE logit surface can be a valid formula REML mode
        // (the #715 penguins regime: bill/flipper cleanly separate the species,
        // so fitted logits can legitimately exceed ±25). `outer_converged ==
        // false` then signals only that the driver auto-escalated to never-fail
        // posterior sampling about that finite mode (gam#860), NOT a separation
        // artifact — the adapter must accept it, never raise
        // `MultinomialSeparationDetected`.
        let saturated_states = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![1.0, 2.0]),
                eta: Array1::from_vec(vec![0.2, 4.0, -7.0]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-1.0, 3.0]),
                eta: Array1::from_vec(vec![1.0, 25.5, -0.1]),
            },
        ];
        assert!(
            multinomial_formula_separation_diagnostic(17, 9, &saturated_states).is_none(),
            "a finite (even saturated, |eta|>25) formula optimum is a valid fit, \
             not a separation diagnostic"
        );

        // Only a genuinely NON-FINITE logit — a NaN/Inf blow-up in the inner
        // linear algebra with no finite mode to sample about — is a real
        // formula-path failure.
        let blown_up = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![1.0, 2.0]),
                eta: Array1::from_vec(vec![0.2, 4.0, -7.0]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-1.0, 3.0]),
                eta: Array1::from_vec(vec![1.0, f64::INFINITY, -0.1]),
            },
        ];
        let err = multinomial_formula_separation_diagnostic(17, 9, &blown_up)
            .expect("a non-finite formula logit must raise the separation diagnostic");
        assert!(
            matches!(
                err,
                EstimationError::MultinomialSeparationDetected {
                    iteration: 17,
                    max_abs_eta,
                    active_class_index: 1,
                    row_index: 1,
                } if !max_abs_eta.is_finite()
            ),
            "expected typed multinomial separation diagnostic at the non-finite channel, got {err:?}"
        );
    }

    #[test]
    fn separation_evidence_gate_arms_firth_only_on_blowup() {
        // Interior fit: finite logits well inside the saturation threshold ⇒ NO
        // separation evidence ⇒ the unbiased criterion's mode is accepted as-is
        // and the Firth/Jeffreys prior stays disarmed (#715 arm (a): no 1/K
        // shrinkage on well-identified data).
        let interior = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![1.0, 2.0]),
                eta: Array1::from_vec(vec![0.2, 4.0, -7.0]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-1.0, 3.0]),
                eta: Array1::from_vec(vec![1.0, -3.5, -0.1]),
            },
        ];
        assert!(
            multinomial_formula_separation_evidence(&interior).is_none(),
            "an interior finite mode must not arm the Firth refit"
        );

        // Saturated but finite logits are valid formula-path modes on
        // near-separated real data. They must not arm the Firth refit because
        // the Jeffreys pull can over-regularize the held-out probabilities.
        let saturated = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![1.0, 2.0]),
                eta: Array1::from_vec(vec![0.2, 4.0, -7.0]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-1.0, 3.0]),
                eta: Array1::from_vec(vec![1.0, 25.5, -0.1]),
            },
        ];
        assert!(
            multinomial_formula_separation_evidence(&saturated).is_none(),
            "a finite saturated formula-mode logit must not arm the Firth refit"
        );

        // Non-finite logit ⇒ inner blow-up along an unbounded direction ⇒
        // separation evidence.
        let blown_up = vec![ParameterBlockState {
            beta: Array1::from_vec(vec![1.0, 2.0]),
            eta: Array1::from_vec(vec![0.2, f64::NAN, -7.0]),
        }];
        let evidence = multinomial_formula_separation_evidence(&blown_up)
            .expect("a non-finite logit is separation evidence");
        assert!(
            evidence.contains("non-finite logit") && evidence.contains("row 1"),
            "evidence must name the non-finite logit, got {evidence}"
        );

        // Large finite logits below the fixed-lambda diagnostic threshold are
        // likewise accepted on the formula path.
        let near = vec![ParameterBlockState {
            beta: Array1::from_vec(vec![1.0, 2.0]),
            eta: Array1::from_vec(vec![0.2, 24.9, -24.9]),
        }];
        assert!(
            multinomial_formula_separation_evidence(&near).is_none(),
            "logits below the saturation threshold must not arm the Firth refit"
        );
    }

    #[test]
    fn unresolved_probe_evidence_arms_firth_on_saturated_finite_logits() {
        let saturated = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![1.0, 2.0]),
                eta: Array1::from_vec(vec![0.2, 4.0, -7.0]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-1.0, 3.0]),
                eta: Array1::from_vec(vec![1.0, 25.5, -0.1]),
            },
        ];

        assert!(
            multinomial_formula_separation_evidence(&saturated).is_none(),
            "a converged finite saturated formula optimum remains unbiased"
        );
        let evidence = multinomial_formula_unresolved_probe_separation_evidence(&saturated)
            .expect("a non-converged saturated probe should arm the Firth refit");
        assert!(
            evidence.contains("separation-scale finite logit")
                && evidence.contains("row 1")
                && evidence.contains("active class 1"),
            "unresolved-probe evidence should name the saturated channel, got {evidence}"
        );

        let near = vec![ParameterBlockState {
            beta: Array1::from_vec(vec![1.0, 2.0]),
            eta: Array1::from_vec(vec![0.2, 24.9, -24.9]),
        }];
        assert!(
            multinomial_formula_unresolved_probe_separation_evidence(&near).is_none(),
            "finite logits below the separation threshold still get the full unbiased retry"
        );
    }

    #[test]
    fn scaled_fisher_override_changes_first_step() {
        // Curvature scaled by 4× shrinks the first Newton step relative to the
        // analytic fit, so a single-iteration fit must differ.
        let (design, y, penalty, lambdas) = toy();
        let n = design.nrows();
        let m = y.ncols() - 1;
        // Analytic block at β = 0: p_a = 1/K = 1/3, so diag = p_a(1−p_a),
        // off-diag = −p_a p_b. Scale that exact block by 4.
        let pk = 1.0 / (y.ncols() as f64);
        let mut over = Array3::<f64>::zeros((n, m, m));
        for row in 0..n {
            for a in 0..m {
                for b in 0..m {
                    let analytic = if a == b { pk * (1.0 - pk) } else { -pk * pk };
                    over[[row, a, b]] = 4.0 * analytic;
                }
            }
        }
        let scaled = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: Some(over.view()),
            max_iter: 1,
            tol: 1.0e-9,
        })
        .expect("override fit must succeed");
        let analytic = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 1,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        let differs = scaled
            .coefficients_active
            .iter()
            .zip(analytic.coefficients_active.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-6);
        assert!(differs, "scaled curvature must change the first step");
    }
}
