//! Penalized multinomial-logit (softmax) GLM driver — fixed-λ inner solve.
//!
//! This is the principled vector-response companion to the scalar PIRLS path:
//! the inner-loop Newton solver for a multi-class GAM at fixed smoothing
//! parameters λ, using the canonical multinomial-logit likelihood
//! ([`MultinomialLogitLikelihood`]) and the existing dense block-Fisher
//! assembly in [`gam_solve::pirls::dense_block_xtwx`] /
//! [`gam_solve::pirls::dense_block_xtwy`].
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
//!   built on the shared [`crate::penalized_vector_glm`] engine.
//!
//! * **REML / LAML smoothing-parameter selection** — [`fit_penalized_multinomial_formula`]
//!   routes through [`crate::custom_family::fit_custom_family_with_rho_prior`]
//!   so the per-active-class `λ_a` are selected by the outer REML/LAML loop;
//!   the caller's `init_lambda` is only a warm-start seed. The multinomial
//!   [`crate::multinomial_reml::MultinomialFamily`] `CustomFamily`
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
//! [`crate::penalized_vector_glm`] engine: at each iteration the
//! assembled penalized Hessian `H + I_{K-1} ⊗ (λ_a S)` is factored via faer's
//! symmetric-PD-with-fallback path, the full Newton step `δ = −H^{-1} ∇F` is
//! computed, and accepted with step halving if the objective fails to decrease
//! (up to a small backtracking budget). Convergence requires both a relative
//! coefficient step `‖δ‖ / (1 + ‖β‖) ≤ tol` and a fresh first-order score
//! certificate at the accepted final iterate; failure produces checkpoint
//! evidence, never coefficients/covariance behind a false flag. This module is
//! the softmax adapter over that engine: it
//! supplies the dense `(K-1)×(K-1)` Fisher block, the residual, and the
//! log-likelihood through [`MultinomialLogitLikelihood`], and owns the
//! class-count / simplex preconditions. The independent-binomial sibling
//! [`crate::binomial_multi`] is the same engine with a row-diagonal
//! Fisher block instead.

use crate::custom_family::{
    BlockwiseFitOptions, CustomFamily, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    fit_custom_family_with_rho_prior,
};
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::fit_orchestration::{
    FitConfig, build_termspec_with_geometry_and_overrides, resolved_resource_policy,
};
use crate::model_types::EstimationError;
use crate::multinomial_posterior::softmax_with_reference;
use crate::multinomial_reml::MultinomialFamily;
use crate::penalized_vector_glm::{
    PenalizedVectorGlmInputs, VectorGlmResume, VectorGlmSolve, fit_penalized_vector_glm,
};
use crate::vector_response::{MultinomialLogitLikelihood, validate_multinomial_simplex};
use gam_data::ColumnKindTag;
use gam_data::EncodedDataset;
use gam_problem::{
    FixedLambdaCheckpoint, FixedLambdaResidualKind, FixedLambdaSolverStage, FixedLambdaStallReason,
    FixedLambdaStationarityEvidence, ResponseColumnKind,
};
use gam_runtime::resource::ProblemHints;
/// The covariance-definition axis, re-exported so a caller of this module's
/// predict surface names the same enum `gam-predict` and the CLI do rather than
/// reaching across crates for it.
pub use gam_solve::model_types::InferenceCovarianceMode;
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::{
    PenaltyBlockInfo, TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
};
use gam_terms::term_builder::resolve_role_col;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use opt::{BacktrackConfig, backtracking_line_search};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

/// Solver-only numerical stabilization floor for the formula-driven
/// multinomial REML inner solve (gam#747).
///
/// Installed with [`RidgePolicy::solver_only`](gam_problem::RidgePolicy::solver_only)
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
/// [`multinomial_formula_penalized_separation_evidence`] and the two-attempt logic
/// in [`fit_penalized_multinomial_formula`]) — so an interior, well-identified
/// fit optimizes the unbiased penalized-REML criterion with no Firth shrinkage
/// toward the uniform simplex, while a finite-but-Fisher-underidentified or
/// non-finite geometry gets the proper prior that is the only thing able to
/// bound its penalty-null directions (#715/#2612 real-data arm). The bare
/// fixed-λ inner driver
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
///
/// CORRECTED 2026-08-08 (#2612), 16 -> 24, WITHOUT MOVING THE CALIBRATION POINT.
/// The value above was chosen so the four-smooth penguin fixture lands on the
/// exact path, and the `D = 16` it quotes for that fixture was computed as
/// `(K-1) * n_penalties`. That has not been the number of outer coordinates
/// since #1587: `equivariant_class_penalty_specs` emits one spec per class per
/// penalty component whenever `K > 2`, so the same fixture is `8 * 3 = 24`, not
/// `2 * 8 = 16` — confirmed against the refusal's own `last_evaluated_rho`,
/// which carries 24 entries. The gate now reads
/// `MultinomialFamily::joint_smoothing_dimension()`, and the threshold is the
/// SAME fixture re-read with a corrected ruler rather than a new number: at
/// `K = 3` the classification is unchanged (`3n <= 24` and `2n <= 16` are both
/// `n <= 8` components), and where it differs at other `K` it differs by
/// admitting MORE exact curvature, which is the side this constant's own
/// rationale says is safe ("medium-D formula fits need exact curvature to keep
/// lambda selection away from over-smoothed caps").
const MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM: usize = 24;

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

/// Per-observation softmax Fisher-information scale for the λ-floor units.
///
/// The penalty enters the criterion as `½ λ βᵀ S β` with a Frobenius-normalized
/// `S` (`‖S‖_F = 1`, see the term-builder calibration referenced by
/// [`multinomial_formula_penalty_scale`]), so the ridge `λ S` is directly
/// comparable to data Fisher information. One observation contributes softmax
/// information `p(1−p)` in a class's logit direction, which is bounded by the
/// logistic peak `p(1−p) ≤ ¼` at `p = ½`. Using this maximal per-observation
/// information as the unit makes the floor's strength interpretable as a count
/// of equivalent **pseudo-observations** of prior: a ridge that equals
/// `τ · ¼ · ‖S‖_F` carries the same logit-direction curvature as `τ` real rows
/// sitting at the most-informative point of the likelihood. This scale is
/// `K`-independent on purpose — the `K`-dependence of the softmax block
/// curvature already lives in the penalty matrix via
/// [`multinomial_formula_penalty_scale`], so the floor (a bound on the
/// multiplier of that already-scaled penalty) must not double-count it.
const MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS: f64 = 0.25;

/// Target prior strength of the λ-floor, in pseudo-observations, for a
/// WELL-SUPPORTED class. The floor holds the unbiased REML optimizer off the
/// zero-penalty boundary (where a boundary-overfit smooth or a Firth switch on
/// finite data would otherwise be accepted) with a prior worth a fixed small
/// fraction of one observation. `8e-4` pseudo-observations reproduces the
/// previously fixture-calibrated large-support floor `τ · ¼ = 2e-4` exactly at
/// the calibration point, now expressed as an effective-prior-strength rather
/// than a tuned λ value.
///
/// MEASURED 2026-07-31 (#2612). This value is not inert and it has never been
/// measured against the quantity it controls. On the penguins real-data arm
/// (stride-3, `n_train = 228`, minority class 46, so the wall is
/// `8.0e-4 × 0.25 × 50/46 = 2.173913043e-4`), **four of the eight live
/// null-space λ sit on this wall exactly** — `ratio_to_wall = 1.000000`. They
/// are boundary solutions, not stationary points of the REML criterion: the box
/// stopped them, and the box is this constant.
///
/// Everything downstream is therefore a readout of it. Sweeping ONLY this
/// constant, same base commit, same split, exact conditioned quadrature:
///
/// ```text
///   pseudo_obs   λ wall      posterior log-loss   plug-in log-loss   calib gap
///   8.0e-4       2.174e-4    0.161820             0.024718           +0.1187
///   8.0e-3       2.174e-3    0.069526             0.025097           +0.0401
///   8.0e-2       2.174e-2    0.065275             0.035297           +0.0360
/// ```
///
/// `calib gap` is `mean predicted probability on the argmax class − held-out
/// accuracy` (accuracy is `0.982456` in every row; no reference tool involved).
/// A 10× wall MORE THAN HALVES the published posterior log-loss while the mode
/// barely moves, so the width is this constant's and the mode is not.
///
/// An exact subspace attribution — same fit, saved Laplace covariance zeroed
/// outside one subspace at a time — says the width does not sit where the wall
/// is applied. Cost in nats above the plug-in, shipped wall / 10× wall:
///
/// ```text
///   intercept (2 coords, unpenalized)   0.071778  →  0.014949
///   range space (64 coords, edf≈6e-4)   0.062538  →  0.034422
///   null space (8 coords, ON the wall)  0.002564  →  0.000989
/// ```
///
/// The railed coordinates carry 1.9% of the cost, yet tightening their prior
/// tenfold cuts the UNPENALIZED intercept's marginal contribution 4.8×. The
/// near-flat prior on the linear direction does not make that direction wide; it
/// makes the joint Hessian near-singular, and the width surfaces on whatever is
/// correlated with it. Attribution by marginal block therefore names the
/// carrier, not the cause.
///
/// DO NOT TUNE THIS ON THAT CURVE. Choosing `8.0e-3` because it clears a
/// held-out log-loss bar on one dataset is the objection #2615 raised against
/// choosing `EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION`, and the sweep above is
/// diagnostic evidence, not a selection procedure. The measurement says the
/// criterion has no interior optimum here — a quasi-separated multinomial's
/// marginal likelihood keeps rewarding a flatter prior on the separating
/// direction — so the repair is a prior that is derived for that regime (the
/// Jeffreys/Firth path this same issue is already driving), not a wall chosen to
/// stop the slide at a convenient place.
const MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS: f64 = 8.0e-4;

/// Reference class support `n_ref`: the effective sample size per class at which
/// the data Fisher information `n_c · I₁` is large enough that the floor sits at
/// its well-supported value. Below `n_ref` the per-class data information shrinks
/// like `n_c`, so to keep the floor's prior from vanishing *relative to* that
/// shrinking data the effective pseudo-observation count is scaled up by
/// `n_ref / n_c` (the prior is held to a fixed fraction of the data information,
/// not a fixed absolute λ). At `n_c = n_ref` the scale is exactly 1.
const MULTINOMIAL_FORMULA_SPARSE_REFERENCE_SUPPORT: f64 = 50.0;

/// Cap on the floor's prior strength in the very-sparse limit, in
/// pseudo-observations. As `n_c → 0` the `n_ref / n_c` scaling diverges; the cap
/// holds the prior at `4e-3` pseudo-observations (`τ_max · ¼ = 1e-3` at the
/// calibration point, the previously-tuned strong-floor value) so the floor
/// stays a proper prior rather than a hard constraint that would dominate the
/// likelihood for a handful-of-rows class.
const MULTINOMIAL_FORMULA_SPARSE_PRIOR_PSEUDO_OBS_MAX: f64 = 4.0e-3;

/// Continuous, Fisher-information-scaled lower λ floor for the formula path,
/// derived from the minority class's effective sample size `n_c`.
///
/// # Derivation (effective-prior-strength / Fisher geometry)
///
/// The penalty `½ λ βᵀ S β` with `‖S‖_F = 1` adds curvature `λ` to the class
/// logit direction; one observation adds at most `I₁ = ¼` there. So a floor that
/// sets `λ_floor = τ_eff · I₁` gives the smooth a prior worth `τ_eff`
/// pseudo-observations. We want a fixed *absolute* prior `τ` for a well-supported
/// class, but for a minority class with only `n_c` effective observations the
/// data information in its block is `n_c · I₁`; holding the prior to a fixed
/// *fraction* of that shrinking data information requires
///
/// ```text
///     τ_eff(n_c) = τ · max(1, n_ref / n_c),   clamped to [τ, τ_max]
///     λ_floor(n_c) = τ_eff(n_c) · I₁
/// ```
///
/// This is the *same* `base · max(1, c0/c)` envelope as before — but `base`,
/// `sparse`, and `c0` are no longer fixture-tuned magic numbers: `base = τ·I₁`,
/// `sparse = τ_max·I₁`, and `c0 = n_ref` are an effective-prior-strength of
/// `τ`/`τ_max` pseudo-observations against the maximal per-observation softmax
/// information `I₁ = ¼`. Properties preserved by construction:
///   * reduces EXACTLY to `τ·I₁` for well-supported classes (`n_c ≥ n_ref`);
///   * reduces EXACTLY to `τ_max·I₁` for very sparse classes
///     (`n_c ≤ n_ref·τ/τ_max`, here `n_c ≤ 10`);
///   * interpolates monotonically and continuously between them in the middle —
///     no cliff at `n_c = n_ref`.
/// At the calibration point the endpoints equal the previous `2e-4` / `1e-3`, so
/// fixtures whose smallest class has `n_c ≥ 50` (penguins, the vgam softmax
/// arms) are unaffected — they sit at `τ·I₁ = 2e-4` exactly as before.
fn multinomial_formula_min_lambda(y_one_hot: ArrayView2<'_, f64>) -> f64 {
    let base = MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS * MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS;
    let sparse =
        MULTINOMIAL_FORMULA_SPARSE_PRIOR_PSEUDO_OBS_MAX * MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS;
    let min_class_count = (0..y_one_hot.ncols())
        .map(|class| y_one_hot.column(class).sum())
        .fold(f64::INFINITY, f64::min);
    if !min_class_count.is_finite() || min_class_count <= 0.0 {
        return base;
    }
    // Effective pseudo-observation prior strength: held to a fixed fraction of
    // the shrinking per-class data information once n_c falls below n_ref.
    let pseudo_obs_scale =
        (MULTINOMIAL_FORMULA_SPARSE_REFERENCE_SUPPORT / min_class_count).max(1.0);
    (base * pseudo_obs_scale).clamp(base, sparse)
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

/// Certify (quasi-)separation at a converged multinomial mode from the curvature
/// the fit ACTUALLY has — the exact Fisher information PLUS the selected joint
/// penalty — on the fit's certified identifiable tangent span.
///
/// Finiteness is not an identification certificate: a tiny smoothing floor can
/// keep every logit finite while the likelihood contributes less than one
/// observation-equivalent of curvature along a separating direction. The
/// Jeffreys objective already owns the canonical absolute/relative conditioning
/// gate for precisely that geometry. Preparing its plan here makes the
/// conditional-refit decision use the same reduced spectrum, smooth gate, and
/// coefficient gauge as the term that will bound the refit. Passing the fit's
/// saved gauge is essential: an aliased raw column is not separation evidence,
/// because it is absent from the active statistical model.
///
/// # Why `H + S_λ` and not `H` (#2612)
///
/// The gate's absolute arm saturates at `λ_min < 1`, i.e. when the
/// worst-determined direction holds less than **one observation-equivalent** of
/// curvature, and its own derivation says why that is the right scale: such a
/// direction "is, by construction, not identified by the data and is the regime
/// Firth exists to stabilise". It also states the premise that makes it
/// conservative — "it never fires on a genuinely well-conditioned large-`n` fit,
/// whose `λ_min = O(n) ≫ 1`".
///
/// (The predicate this call site consults is `JointJeffreysPlan::is_active`,
/// which is `gate_weight != 0`, and the weight is a C¹ ramp reaching exactly
/// zero only at `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16`. So the arming boundary
/// is sixteen observation-equivalents, not one — measured: an `n = 3000, k = 5`
/// fit at `λ_min = 1.989` still armed, at weight `≈ 0.988`. That ramp is a
/// continuity device for an always-on term — a binary gate makes `Φ(ρ)` jump,
/// which is the #787 regression — so it stays; what changes is which matrix it
/// is asked about.)
///
/// That premise is false for **every penalized smooth basis**, and this call site
/// used to hand the gate the bare Fisher information `H`. A `k`-dimensional
/// spline basis has high-frequency directions the data barely resolve — that is
/// the entire reason they are penalized — so `λ_min(H)` sits well below one
/// observation-equivalent on ordinary, well-behaved, nowhere-separating data.
/// Measured on labels DRAWN from a smooth softmax truth (`y ~ s(x1,k=6) +
/// s(x2,k=6)`, `n = 600`, every class keeping appreciable probability
/// everywhere):
///
/// ```text
///   lambda_min = 4.051e-1   lambda_max = 1.575e2   ratio = 2.572e-3
///   Jeffreys gate weight = 1   =>  "separation evidence"  =>  Firth/Jeffreys refit
/// ```
///
/// The relative arm is nowhere near firing (`2.6e-3` against a `1e-6` clear
/// knot); the verdict is the absolute arm reading a *penalized* direction as if
/// nothing were holding it. So the "#715 arm ONLY on separation evidence" design
/// was unconditional in practice on any multinomial GAM carrying a smooth.
///
/// The distinction the gate has to make is exactly the one #715 derives: a
/// direction `v` is beyond `λ`'s reach only when `S v = 0`, because
/// `(H + S_λ)v = Hv + λSv`. Where `Sv ≠ 0` the smoothing parameter supplies the
/// missing curvature and the direction is identified — by the prior the model
/// already has. Reading `H` alone cannot tell those apart; reading the penalized
/// curvature `H + S_λ` at the selected `λ` is the same question asked of the
/// objective that was actually optimized, and its units are unchanged (the
/// Frobenius-normalized `S` makes `λS` directly comparable to data Fisher
/// information — see [`MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS`]).
///
/// The unpenalized spectrum is reported alongside the deciding one whenever the
/// certificate fires, because "the data do not determine this direction" and
/// "and no `λ` repairs it" are two different statements and the verdict rests on
/// the second. That costs one extra reduced eigendecomposition, paid only on the
/// branch that arms.
///
/// # And the SUBSPACE, not only the verdict (#2612)
///
/// Reading `H + S_λ` fixed which matrix the verdict is taken on. It did not fix
/// which directions the verdict is taken *over*, and that is the second half of
/// the same sentence: `(H + S_λ)v = Hv + λSv`, so a direction is beyond every
/// `λ`'s reach **exactly when `S_λ v = 0`**. Where `S_λ v ≠ 0` the model already
/// carries a proper prior on `v`; the posterior there is proper, and
/// [`crate::multinomial_predictive`] integrates it EXACTLY — the published
/// probability is a ratio of normalising constants with a per-row measured error,
/// not a Gaussian approximation — so a diffuse penalized direction already shows
/// up in the published probability as the width it is. Arming a second, much
/// stronger prior there is not stabilisation; it is a different model.
///
/// That distinction is not cosmetic, because the term this certificate arms does
/// not act only on the direction that armed it. `jeffreys_antiderivative`
/// saturates at `Λ = CONDITIONING_GATE_ABSOLUTE_CLEAR = 16` observation-
/// equivalents: below `Λ` a direction earns the full `1/λ` prior push, above it
/// essentially none. The full-span choice was justified by "the Jeffreys score is
/// `O(1)` against the data's `O(n)` Fisher information, so on a data-identified
/// direction its only effect is the `O(1/n)` Firth bias correction". **On a
/// quasi-separated multinomial the data's information is not `O(n)` in ANY
/// direction**: the softmax Fisher weight `W = diag(p) − p pᵀ` collapses to
/// `≈ 0.005` per row on a confident fit, so measured at the penguins witness's
/// certified unbiased mode,
///
/// ```text
///   likelihood alone:  lambda_min = -8.8e-19   lambda_max =    1.4423
///   H + S_lambda:      lambda_min =  1.92e-4   lambda_max = 2298.5
/// ```
///
/// — `λ_max(H) = 1.44` over a **74-dimensional** span at `n = 228`, i.e. every
/// direction of the basis sits inside the window where the prior acts at full
/// strength, while the fit's own penalty already bounds most of them up to
/// `2298`. The armed fit then publishes mean argmax probability `0.828` against
/// held-out accuracy `0.965`: a 13.7-point calibration deficit, in the MODE
/// (plug-in `0.20755` against posterior-mean `0.20917`), which is the
/// under-confidence #2612 is named for.
///
/// So the certificate is taken on the subspace no `λ` reaches. With NO penalised
/// component that subspace is the whole identifiable span and this is byte-for-
/// byte the previous decision — which is why a quasi-separated design carrying no
/// penalty still arms, and must. `ker(S_λ)` is measured by the same relative rule
/// [`crate::multinomial_reml::measured_penalty_rank`] reports as a count, so the
/// dimension and the basis can never disagree.
///
/// REJECTED: handing the Jeffreys TERM `H + S_λ` instead of `H` (the honest
/// completion, and the right long-run answer) — the term's information would then
/// depend explicitly on `ρ`, and the outer hypergradient carries no explicit-`ρ`
/// channel for it, so the analytic gradient would silently stop matching its own
/// value; that is a larger change than this issue can verify. Widening the gate's
/// knots (they are derived, and they are not what is wrong). Reading the
/// deficient subspace of `H + S_λ` itself (it moves with `β` and `ρ`, so the
/// Jeffreys derivative tower would no longer be differentiating a fixed span).
fn multinomial_formula_penalized_separation_evidence(
    family: &MultinomialFamily,
    specs: &[ParameterBlockSpec],
    block_states: &[ParameterBlockState],
    identifiable_span: ArrayView2<'_, f64>,
    joint_specs: &[gam_problem::JointPenaltySpec],
    n_penalty_components: usize,
    joint_log_lambdas: Option<&Array1<f64>>,
) -> Result<Option<MultinomialSeparationCertificate>, String> {
    if let Some(evidence) = multinomial_formula_separation_evidence(block_states) {
        // A saturated or non-finite mode: there is no certified curvature to
        // measure a span from, so the term keeps its derived `ker(S_lambda)`
        // span and only the verdict travels.
        return Ok(Some(MultinomialSeparationCertificate {
            evidence,
            measured_span: None,
        }));
    }
    let information = family
        .joint_jeffreys_information_with_specs(block_states, specs)?
        .ok_or_else(|| {
            "multinomial separation certificate requires exact joint Fisher information".to_string()
        })?;
    let coefficient_dim = information.nrows();
    if information.ncols() != coefficient_dim {
        return Err(format!(
            "multinomial separation certificate received non-square Fisher information {}x{}",
            information.nrows(),
            information.ncols()
        ));
    }
    // The curvature the certified mode actually sits in. `multinomial_joint_
    // penalty_operator` is the fit's ONE assembly of `S_λ` — the same matrix the
    // influence reconstruction and the published payload read — so the arming
    // decision and the published penalty cannot describe different priors.
    let s_lambda = multinomial_joint_penalty_operator(
        joint_specs,
        joint_log_lambdas,
        n_penalty_components,
        coefficient_dim,
    )
    .map_err(|error| {
        format!(
            "multinomial separation certificate could not assemble the selected joint \
             penalty: {error}"
        )
    })?;
    let penalized = &information + &s_lambda;
    // The subspace no `λ` reaches. `S_λ` is PSD, so `ker(S_λ)` is exactly the
    // eigenvectors at (relative) zero, classified by the same rule
    // `measured_penalty_rank` reports as a count. With no penalised component
    // `S_λ` is the zero operator and this is the whole identifiable span, which
    // is what makes the unpenalised quasi-separated design's verdict unchanged.
    let reduced_penalty = identifiable_span.t().dot(&s_lambda.dot(&identifiable_span));
    let unreached =
        crate::multinomial_reml::measured_penalty_nullspace(&reduced_penalty).map_err(|error| {
            format!("multinomial separation certificate could not measure ker(S_lambda): {error}")
        })?;
    log::info!(
        "multinomial separation certificate: {}/{} identifiable direction(s) are unreached by \
         any smoothing parameter (S_lambda v = 0)",
        unreached.ncols(),
        reduced_penalty.nrows(),
    );
    // Every direction is penalised: the model's own prior is proper everywhere,
    // and whatever width is left belongs to the posterior the predictive
    // integrates exactly.
    if unreached.ncols() == 0 {
        return Ok(None);
    }
    let unreached_span = identifiable_span.dot(&unreached);
    let plan = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        penalized.view(),
        unreached_span.view(),
    )?;
    // The DECISION, not the contribution: `is_under_identified` is the gate's
    // derived predicate at one observation-equivalent, while `is_active` is
    // `weight != 0` and therefore boundaries on the ramp's far knot at sixteen.
    // Measured: the synthetic softmax-drawn fixture lands at
    // `λ_min(H + S_λ) = 5.508` — five and a half observations' worth of
    // curvature in the worst direction, comfortably identified by the
    // constant's own derivation — and `is_active` still says yes, at weight
    // `0.783`, because the ramp has not finished tapering. Choosing an
    // estimand on the support of a smoothing device is the same category error
    // as choosing one on a cost cap.
    let (unreached_min, unreached_max) = plan.information_extrema();
    log::info!(
        "multinomial separation certificate: on the unreached subspace H+S_lambda lies in \
         [{unreached_min:e}, {unreached_max:e}], gate weight {:e}, under_identified={}, \
         singular={}",
        plan.conditioning_gate_weight(),
        plan.is_under_identified(),
        plan.reduced_information_is_singular(),
    );
    // UNDER-IDENTIFIED, not merely singular (#2612). "Improper" is the
    // mathematically minimal reason to add a prior, and it was measured here and
    // rejected: with the certificate keyed on singularity instead, penguins
    // disarms and produces a beautiful MODE — held-out plug-in log-loss 0.02481
    // against nnet::multinom's 0.09494, calibration gap +0.010, in 4.0 s — and
    // then cannot publish a probability at all, because
    // `predict_multinomial_formula` is the posterior MEAN and that posterior is
    // not describable:
    //
    // ```text
    //   penguins stride-3: row 24 predictive mass 0.944  (defect 5.6e-2)
    //   penguins stride-4: row 20 predictive mass 0.517  (defect 4.8e-1)
    //   banded n_train=360, 720: augmented mode did not converge in 100 Newton
    //                            iterations
    // ```
    //
    // A posterior that is proper only because a direction carries 2.9e-3
    // observation-equivalents is proper in name: the mode exists, and nothing
    // else about it does. The mass-defect identity refuses rather than lying,
    // which is the right behaviour and also the proof — the estimand this path
    // publishes cannot be computed there. So the criterion stays the gate's own
    // derived one, at ONE observation-equivalent, and what #2612 changes is
    // WHERE it is asked (the directions no lambda reaches) and where the term it
    // arms is allowed to act (the same subspace).
    if !plan.is_under_identified() {
        return Ok(None);
    }
    let (lambda_min, lambda_max) = plan.information_extrema();
    let relative = if lambda_max > 0.0 {
        lambda_min / lambda_max
    } else {
        f64::NEG_INFINITY
    };
    // Both wider readings, so the refusal shows what it did NOT decide on: the
    // whole penalized span (which the model's own prior bounds) and the
    // likelihood alone (which is not the curvature the fit has).
    let whole_penalized = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        penalized.view(),
        identifiable_span,
    )?;
    let (span_min, span_max) = whole_penalized.information_extrema();
    let unpenalized = gam_solve::estimate::reml::jeffreys_subspace::JointJeffreysPlan::prepare(
        information.view(),
        identifiable_span,
    )?;
    let (data_min, data_max) = unpenalized.information_extrema();
    // ── The verdict has fired. WHERE the prior belongs is a second question ──
    //
    // The two are not the same question and this lane measured what happens when
    // one object answers both.
    //
    // **The verdict** — *is there evidence this model needs a proper prior?* — is
    // a statement about the model's STRUCTURE: a direction no smoothing parameter
    // can ever reach, which the data does not determine either. That is
    // `ker(S_λ)` and the gate's predicate on it, exactly as above, and it is what
    // keeps a fit on non-separating data byte-unchanged (arm 1) and a genuinely
    // separated design armed (arm 2).
    //
    // **The span** — *where does that prior belong?* — is a statement about the
    // fit's ARITHMETIC at the smoothing it selected. Measured at the penguins
    // stride-4 unbiased mode:
    //
    // ```text
    //   ker(S_lambda):            2 of 74 directions, lambda_min(H+S_lambda) = 1.9e-3
    //   whole identifiable span:                      lambda_min(H+S_lambda) = 5.1e-5
    // ```
    //
    // The worst-bounded direction — five orders below one observation-equivalent
    // — is NOT in the kernel. It is a `range(S)` direction whose selected λ railed
    // at `MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS = 8e-4` pseudo-observations, so the
    // claim that backs the kernel ("on `range(S)` the model already carries a
    // proper prior") is true in name and false in magnitude. Left unarmed the
    // coefficient runs to `|η|∞ ≈ 45` and the posterior-mean predictive refuses to
    // publish. Arming the measured set instead takes the same fixture to
    // `acc = 0.9767`, `log-loss 0.07420` and a publishable posterior.
    //
    // Trying to make ONE object answer both was measured and rejected: at every
    // scale of the metric below, either `genuine_separation_still_arms_the_prior`
    // disarmed (the verdict got stricter) or
    // `a_quasi_separated_smooth_fit_is_calibrated` broke at `-0.0525` against a
    // `0.05` bar (the span got wider). Threading that needle by choosing the
    // scale is choosing an estimand on a curve, which is the objection #2615
    // raised against choosing `EFFECTIVE_DF_FLOOR_RELATIVE_FRACTION`.
    //
    // The span is measured in the CLR METRIC, not in the raw ALR coordinates.
    // Relabelling classes acts on θ by a non-orthogonal contrast change, so a
    // threshold on the raw spectrum selects a different PHYSICAL subspace for each
    // choice of reference class; a kernel is congruence-invariant and never had
    // that problem. Measured cost of taking it raw, on
    // `multinomial_fit_is_invariant_to_reference_class_1587`: predicted-probability
    // drift `4.093e-3` across three labelings of one dataset against a `1e-3` bar,
    // with refit noise exactly `0`.
    let coefficient_metric = crate::multinomial_reml::centered_class_coefficient_metric(
        family.active_classes(),
        family.total_classes,
        family.design.ncols(),
    );
    let reduced_penalized = identifiable_span
        .t()
        .dot(&penalized.dot(&identifiable_span));
    let reduced_metric = identifiable_span
        .t()
        .dot(&coefficient_metric.dot(&identifiable_span));
    let measured =
        crate::multinomial_reml::under_identified_subspace(&reduced_penalized, &reduced_metric)
            .map_err(|error| {
                format!(
                    "multinomial separation certificate could not measure the under-identified \
                     subspace of H+S_lambda: {error}"
                )
            })?;
    log::info!(
        "multinomial separation certificate: the armed term acts on {}/{} identifiable \
         direction(s) holding under one observation-equivalent of curvature in H+S_lambda at \
         the certified mode, of which {} are unreached by any smoothing parameter",
        measured.ncols(),
        reduced_penalized.nrows(),
        unreached.ncols(),
    );
    // A measured span that came out EMPTY leaves the derived `ker(S_λ)` route in
    // place rather than disarming a term the verdict just armed: the verdict is
    // the decision, and a span this measurement cannot produce is a reason to
    // fall back, not to overrule it.
    let measured_span = if measured.ncols() == 0 {
        None
    } else {
        Some(std::sync::Arc::new(identifiable_span.dot(&measured)))
    };
    Ok(Some(MultinomialSeparationCertificate {
        evidence: format!(
            "no smoothing parameter reaches {}/{} identifiable direction(s), and on that \
             subspace the penalized curvature H+S_lambda is under-identified at the certified \
             mode: lambda_min={lambda_min:e}, lambda_max={lambda_max:e}, \
             lambda_min/lambda_max={relative:e}, Jeffreys gate weight={:e}; the armed term acts \
             on the {} direction(s) the model fails to bound at the selected lambda \
             (whole identifiable span: H+S_lambda in [{span_min:e}, {span_max:e}], likelihood \
             alone in [{data_min:e}, {data_max:e}])",
            unreached.ncols(),
            reduced_penalty.nrows(),
            plan.conditioning_gate_weight(),
            measured.ncols(),
        ),
        measured_span,
    }))
}

/// The separation certificate's two answers (#2612): whether the Jeffreys/Firth
/// prior is armed, and — separately — where it acts.
#[derive(Debug, Clone)]
pub(crate) struct MultinomialSeparationCertificate {
    /// Human-readable evidence, carried to the payload as
    /// `MultinomialSavedModel::separation_evidence`.
    pub(crate) evidence: String,
    /// Orthonormal basis, in raw joint coefficient order, of the directions the
    /// model fails to bound at the smoothing it selected — the span the armed
    /// term acts on. `None` when there is none to measure (a saturated or
    /// non-finite solve) or when the measurement came out empty, in which case
    /// the term keeps its derived `ker(S_lambda)` span.
    pub(crate) measured_span: Option<std::sync::Arc<Array2<f64>>>,
}

impl MultinomialSeparationCertificate {
    pub(crate) fn as_str(&self) -> &str {
        &self.evidence
    }
}

impl std::fmt::Display for MultinomialSeparationCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Inputs to [`fit_penalized_multinomial`].
///
/// The penalty matrix `S` is shared across classes; per-class smoothing
/// parameters `lambdas` (length `K - 1`) scale `S` independently for each
/// active class. The full block-replicated penalty is `diag_a(λ_a) ⊗ S`,
/// which is exactly what [`gam_solve::arrow_schur::KroneckerPenaltyOp`]
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
    /// Optional checkpoint emitted by a prior fixed-λ multinomial stall on
    /// the same design, response, weights, offsets, penalty, and lambdas. A
    /// `MultinomialNewton` checkpoint resumes the ordinary softmax objective;
    /// a `MultinomialFirth` checkpoint resumes the Jeffreys/Firth separation
    /// objective directly. Any other stage or coefficient shape is rejected.
    pub resume_from: Option<&'a FixedLambdaCheckpoint>,
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
    /// satisfied the tolerance). Non-convergence (outside the separation lane,
    /// which escalates to the Firth refit) is surfaced as the typed
    /// [`EstimationError::FixedLambdaNewtonDidNotConverge`] rather than an `Ok`
    /// with a flag, so every constructed value of this struct is a certified
    /// converged fit (SPEC: a fit only ever comes from a converged
    /// optimization).
    pub iterations: usize,
    /// Penalized negative log-likelihood at the returned `β̂`:
    /// `−log L(β̂) + ½ Σ_a λ_a · β̂_a^T S β̂_a`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)` for diagnostic reporting.
    pub deviance: f64,
    /// Joint Laplace posterior coefficient covariance `H⁻¹` at the converged
    /// `β̂`, shape `(P·(K−1))×(P·(K−1))` (#1101). Block-ordered to match the
    /// stacked active-class coefficient vector `β = [β_0; …; β_{K-2}]`: active
    /// class `a`'s `P` coefficients occupy rows/cols `a·P .. (a+1)·P`, indexed
    /// `θ[a·P + i] = β̂[i, a]`. This is the Laplace covariance from the factored
    /// penalized Hessian `XᵀWX + diag_a(λ_a)⊗S`; it drives the delta-method
    /// per-class probability standard errors
    /// ([`Self::logistic_normal_softmax_moments`])
    /// on the fixed-λ inner-solve path.
    pub coefficient_covariance: Array2<f64>,
}

impl MultinomialFitOutputs {

}

#[derive(Clone, Copy)]
struct FirthResume<'a> {
    coefficients: ArrayView2<'a, f64>,
    completed_iterations: usize,
}

fn fixed_lambda_checkpoint_coefficients(
    checkpoint: &FixedLambdaCheckpoint,
    expected_stage: FixedLambdaSolverStage,
    p: usize,
    m: usize,
) -> Result<Array2<f64>, EstimationError> {
    checkpoint.validate().map_err(|reason| {
        EstimationError::InvalidInput(format!(
            "multinomial fixed-λ resume checkpoint is invalid: {reason}"
        ))
    })?;
    if checkpoint.stage() != expected_stage {
        crate::bail_invalid_estim!(
            "multinomial fixed-λ resume checkpoint stage is {}, expected {}",
            checkpoint.stage(),
            expected_stage,
        );
    }
    if checkpoint.rows() != p || checkpoint.cols() != m {
        crate::bail_invalid_estim!(
            "multinomial fixed-λ resume checkpoint shape {}x{} does not match P x (K-1) = {p}x{m}",
            checkpoint.rows(),
            checkpoint.cols(),
        );
    }
    Array2::from_shape_vec((p, m), checkpoint.values().to_vec()).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial fixed-λ resume checkpoint could not be reshaped: {error}"
        ))
    })
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
        resume_from,
    } = inputs;

    // ──────────────────────── family-specific validation ───────────────────
    // The shared engine re-validates the geometry common to every vector-GLM
    // (nonempty design, penalty shape, λ finiteness/non-negativity, override
    // `(N, M, M)` shape, finite design). The multinomial family owns the
    // class-count contract (`K ≥ 2`, λ length `K`), the per-row simplex
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
    // #2344: the fixed-λ contract is K per-CLASS lambdas (reference class
    // included), matching the permutation-equivariant carrier the REML route
    // selects (1326d0794). K−1 per-CONTRAST lambdas anchored the smoothing to
    // the arbitrary ALR baseline — relabeling the classes changed the fitted
    // model. No backcompat shim: K lambdas is the honest contract for nominal
    // classes.
    if lambdas.len() != k {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: lambdas length {} ≠ K = {k} (one λ per class, \
             reference class included — the permutation-equivariant per-class contract, #2344)",
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

    let p = design.ncols();
    let resumed_newton_coefficients = match resume_from {
        Some(checkpoint) if checkpoint.stage() == FixedLambdaSolverStage::MultinomialFirth => {
            let coefficients = fixed_lambda_checkpoint_coefficients(
                checkpoint,
                FixedLambdaSolverStage::MultinomialFirth,
                p,
                m,
            )?;
            return fit_penalized_multinomial_firth_fallback(
                design,
                y_one_hot,
                penalty,
                lambdas,
                row_weights,
                max_iter,
                tol,
                Some(FirthResume {
                    coefficients: coefficients.view(),
                    completed_iterations: checkpoint.completed_iterations(),
                }),
            );
        }
        Some(checkpoint) => Some(fixed_lambda_checkpoint_coefficients(
            checkpoint,
            FixedLambdaSolverStage::MultinomialNewton,
            p,
            m,
        )?),
        None => None,
    };
    let vector_resume = resumed_newton_coefficients
        .as_ref()
        .map(|coefficients| VectorGlmResume {
            coefficients: coefficients.view(),
            completed_iterations: resume_from
                .map(FixedLambdaCheckpoint::completed_iterations)
                .unwrap_or(0),
        });

    // ────────────────────────── likelihood construction ───────────────────
    let mut likelihood = MultinomialLogitLikelihood::with_classes(k)?;
    if let Some(w) = row_weights.as_ref() {
        likelihood = likelihood.with_row_weights(w.to_owned())?;
    }

    // ─────────────────── shared penalized vector-GLM solve ─────────────────
    // The softmax Fisher block is dense across the `M = K − 1` active classes;
    // the engine assembles the coupled `(P·M)×(P·M)` penalized Hessian, runs
    // the damped Newton loop, and returns the converged `β̂` and `η = X β̂`.
    let solve = fit_penalized_vector_glm(
        PenalizedVectorGlmInputs {
            design,
            y: y_one_hot,
            offset: None,
            penalty,
            lambdas,
            fisher_w_override,
            max_iter,
            tol,
            // #2344: the permutation-equivariant per-class metric — the fixed-λ
            // twin of the REML equivariant carrier (1326d0794). K per-class
            // lambdas on the centered class functions; reference-free by
            // construction, collapsing to the shared Centered metric at equal λ.
            class_penalty_metric:
                crate::penalized_vector_glm::ClassPenaltyMetric::EquivariantPerClass,
            resume_from: vector_resume,
        },
        &likelihood,
        "fit_penalized_multinomial",
    )?;

    let fit = match solve {
        VectorGlmSolve::Converged(fit) => fit,
        VectorGlmSolve::Stalled(stall) => {
            return handle_multinomial_fixed_lambda_stall(
                stall,
                design,
                y_one_hot,
                penalty,
                lambdas,
                row_weights,
                max_iter,
                tol,
            );
        }
    };

    let fitted_probabilities = likelihood.probabilities(fit.eta.view());

    Ok(MultinomialFitOutputs {
        coefficients_active: fit.coefficients,
        fitted_probabilities,
        iterations: fit.iterations,
        penalized_neg_log_likelihood: -fit.log_likelihood + fit.penalty_term,
        deviance: -2.0 * fit.log_likelihood,
        coefficient_covariance: fit.coefficient_covariance,
    })
}

/// Resolve a budget-exhausted fixed-λ softmax Newton solve: either the
/// separation lane (escalate to the Firth/Jeffreys proper-prior refit) or the
/// typed non-convergence error. Never mints a fit from the stalled iterate.
fn handle_multinomial_fixed_lambda_stall(
    stall: crate::penalized_vector_glm::VectorGlmStall,
    design: ArrayView2<'_, f64>,
    y_one_hot: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    lambdas: ArrayView1<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
    max_iter: usize,
    tol: f64,
) -> Result<MultinomialFitOutputs, EstimationError> {
    let (max_abs_eta, row_index, active_class_index) = max_abs_eta_location(stall.eta.view());
    if max_abs_eta >= MULTINOMIAL_SEPARATION_ETA_THRESHOLD {
        // Perfect / quasi-perfect separation (#1854): the UNBIASED softmax MLE is
        // not finite along `active_class_index`'s saturated logit direction, so
        // the fixed-λ Newton above ran away (`|η| ≥ 25`, no convergence). A
        // penalty-null direction `v` (`S v = 0`, e.g. an unpenalized intercept /
        // linear-covariate column) under softmax saturation has
        // `(XᵀWX + λS) v → 0` for EVERY λ, so no smoothing parameter can bound it
        // — only a proper prior on that quotient-null subspace can. Rather than
        // hard-erroring, engage the Firth/Jeffreys proper prior automatically
        // (magic-by-default): the full-span `½ log|I(β)|` correction supplies the
        // `O(1)` curvature that keeps the estimate finite on exactly those
        // separated directions while leaving well-identified fits untouched. This
        // reuses the same coupled joint-Newton Jeffreys machinery the formula
        // REML path arms on separation evidence (see
        // `fit_penalized_multinomial_formula`), only here at the caller's fixed λ.
        // Engage the fallback, but never let an internal consistency panic in
        // the coupled joint-Newton assembly (e.g. the #1395 logdet-collapse
        // guard) escape as a process abort: convert any panic into the
        // documented hard separation diagnostic, exactly as if the refit had
        // returned Err. This mirrors the catch_unwind panic-to-typed-error
        // boundary already used around the faer / cudarc entry points, and keeps
        // the separation path no worse than the pre-#1854 clean error while the
        // Firth refit is still being hardened.
        // Start the Firth refit from the well-conditioned origin (β = 0), NOT
        // from the stalled Newton iterate. That stalled iterate is the runaway
        // separated point (`|η| ≥ 25`), where the softmax Fisher information
        // `I(β)` is numerically singular (every fitted probability is pinned to
        // the {0,1} simplex boundary, so `I → 0`). Warm-starting the Firth
        // Newton there is catastrophic: the first step `(I + λS)⁻¹ U*` is
        // unbounded and every backtracked candidate stays on the boundary, so
        // the line search exhausts without an accepted step and the refit stalls
        // at iteration 1 — it can never climb back to the interior Firth mode.
        // The Firth objective's interior mode is start-independent (the
        // `firth_solver_rejects_a_truncated_iterate` resume contract asserts the
        // same mode is reached from any interior start), and from `β = 0` the
        // information is well-conditioned, so a plain from-zero refit converges
        // reliably on exactly the separated data that defeated the fixed-λ
        // Newton above.
        let firth = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fit_penalized_multinomial_firth_fallback(
                design,
                y_one_hot,
                penalty,
                lambdas,
                row_weights,
                max_iter,
                tol,
                None,
            )
        }));
        match firth {
            // SPEC: a fit object must only ever come from a converged
            // optimization — the Firth fallback itself surfaces a
            // budget-exhausted refit as the typed
            // `FixedLambdaNewtonDidNotConverge`, which is forwarded verbatim so
            // the caller sees which lane stalled and its evidence.
            Ok(Ok(out)) => return Ok(out),
            Ok(Err(err @ EstimationError::FixedLambdaNewtonDidNotConverge { .. })) => {
                return Err(err);
            }
            // Firth refit errored, or an internal consistency guard panicked:
            // fall back to the explicit hard separation diagnostic.
            Ok(Err(_)) | Err(_) => {
                return Err(EstimationError::MultinomialSeparationDetected {
                    iteration: stall.iterations,
                    max_abs_eta,
                    active_class_index,
                    row_index,
                });
            }
        }
    }

    // SPEC: a fit object must only ever come from a converged optimization.
    // A stall WITHOUT the separation fingerprint (|η| below the threshold —
    // e.g. ill-conditioned data exhausting `max_iter`) is a typed error
    // carrying its evidence, never an Ok(outputs) with a flag.
    Err(stall.into_nonconvergence_error(
        FixedLambdaSolverStage::MultinomialNewton,
        "fit_penalized_multinomial (fixed-λ softmax damped Newton)",
    )?)
}

/// Firth/Jeffreys-penalized multinomial refit engaged automatically when the
/// unbiased softmax MLE separates (#1854).
///
/// The unbiased fixed-λ solve ([`fit_penalized_multinomial`]) runs away on
/// (quasi-)separated data because the softmax likelihood has no finite mode along
/// the saturated logit direction and the smoothing penalty `S` cannot bound a
/// penalty-null direction (`S v = 0` ⇒ `(XᵀWX + λS) v → 0` for every λ). This
/// refit arms the full-span Jeffreys/Firth proper prior `½ log|I(β)|` on the
/// coupled joint softmax information, which supplies the `O(1)` curvature that
/// bounds exactly those directions and keeps the estimate finite.
///
/// # The estimator
///
/// It maximizes the penalized Firth objective at the caller's *fixed* `λ`
///
/// ```text
///   ℓ*(β) = Σ_n w_n Σ_c y_{nc} log p_{nc}
///           − ½ Σ_a λ_a βₐᵀ S βₐ
///           + ½ log det I(β)
/// ```
///
/// where `I(β)` is the coupled `(P·M)×(P·M)` softmax Fisher information (block
/// `(a,b)` is `Σ_n w_n (δ_{ab} p_{na} − p_{na} p_{nb}) x_n x_nᵀ`, block-ordered so
/// `θ[a·P+i] = β[i,a]`) and `M = K−1` active classes carry the reference-coded
/// logits (`η_{ref} ≡ 0`). The Jeffreys term `½ log det I(β)` is the standard
/// Firth penalty: it diverges to `−∞` as any fitted probability approaches the
/// simplex boundary (`I → 0`), so its maximizer is interior and finite on exactly
/// the separated directions that defeat every smoothing `λ`.
///
/// # Why this fixed-λ solver rather than the outer-REML formula path
///
/// The direct entry ([`fit_penalized_multinomial`]) is a fixed-λ inner solve — it
/// carries no outer smoothing selection — so the natural Firth engagement is a
/// fixed-λ Firth Newton, not the formula path's outer-REML joint-Newton machinery
/// (which is armed instead by [`fit_penalized_multinomial_formula`] on separation
/// evidence). Solving the Firth objective directly here keeps the separation
/// contract self-contained and independent of the shared trust-region/KKT
/// certificate machinery.
///
/// # The iteration
///
/// A Fisher-scoring Newton on `ℓ*`: the ascent direction is
/// `Δ = (I + Λ⊗S)⁻¹ U*`, where `U*` is the Firth-adjusted penalized score
///
/// ```text
///   U*[(c,s)] = Σ_n w_n x_{ns} (y_{nc} − p_{nc})       (data score)
///             − λ_c (S β_c)_s                           (smoothing penalty)
///             + ½ Σ_n w_n x_{ns} h^c_n                  (Firth adjustment)
/// ```
///
/// and the Firth adjustment uses `h^c_n = Σ_{a,b} G^c_{n,ab} Q_{n,ab}` with the
/// per-row information "hat" `Q_{n,ab} = x_nᵀ [I⁻¹]_{(a,b)} x_n` and the softmax
/// third-derivative tensor
/// `G^c_{ab} = δ_{ab} p_a (δ_{ac} − p_c) − p_a p_b (δ_{ac} + δ_{bc} − 2 p_c)`.
/// This `½ Σ tr(I⁻¹ ∂I/∂β)` is exactly `∇[½ log det I]` (finite-difference
/// verified). Each step is globalized by backtracking on `ℓ*`, so a step that
/// would push a probability to the boundary (making `I` non-PD) is rejected and
/// the fit stays interior. Convergence is the Newton decrement `½ U*ᵀΔ`.
fn fit_penalized_multinomial_firth_fallback(
    design: ArrayView2<'_, f64>,
    y_one_hot: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    lambdas: ArrayView1<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
    max_iter: usize,
    tol: f64,
    resume_from: Option<FirthResume<'_>>,
) -> Result<MultinomialFitOutputs, EstimationError> {
    use faer::Side;
    use gam_linalg::faer_ndarray::{
        FaerArrayView, array1_to_col_matmut, array2_to_matmut, factorize_symmetricwith_fallback,
    };
    use gam_linalg::matrix::FactorizedSystem;

    let n_obs = design.nrows();
    let p = design.ncols();
    let k = y_one_hot.ncols();
    let m = k - 1;
    let d = p * m;

    // Local softmax likelihood mirroring the caller's row weights, used to map the
    // fitted η back to probabilities.
    let mut likelihood = MultinomialLogitLikelihood::with_classes(k)?;
    if let Some(w) = row_weights.as_ref() {
        likelihood = likelihood.with_row_weights(w.to_owned())?;
    }
    let weight = |row: usize| -> f64 { row_weights.as_ref().map_or(1.0, |w| w[row]) };

    let tol_eff = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-8
    };

    // Probabilities (N, K), active classes 0..M then the pinned reference at M.
    let probs_at = |beta: &Array2<f64>| -> Array2<f64> {
        let eta = design.dot(beta);
        likelihood.probabilities(eta.view())
    };

    // Coupled softmax Fisher information I (d×d), block-ordered θ[a·P+i] = β[i,a].
    let assemble_info = |probs: &Array2<f64>| -> Array2<f64> {
        let mut info = Array2::<f64>::zeros((d, d));
        for row in 0..n_obs {
            let w = weight(row);
            if w == 0.0 {
                continue;
            }
            for a in 0..m {
                let pa = probs[[row, a]];
                let ao = a * p;
                for b in 0..m {
                    let pb = probs[[row, b]];
                    let wab = w * (if a == b { pa - pa * pb } else { -pa * pb });
                    if wab == 0.0 {
                        continue;
                    }
                    let bo = b * p;
                    for i in 0..p {
                        let xi = design[[row, i]];
                        if xi == 0.0 {
                            continue;
                        }
                        let cc = wab * xi;
                        for j in 0..p {
                            info[[ao + i, bo + j]] += cc * design[[row, j]];
                        }
                    }
                }
            }
        }
        info
    };

    // Factor a symmetric matrix (with escalating ridge only if it is not SPD) and
    // return its inverse and log-determinant.
    //
    // The ridge ladder is a standard relative-jitter Cholesky recovery, not a
    // tuned knob: (a) the base jitter is scaled to the matrix by `max_diag`
    // (`max_diag · ε` with ε at the double-precision Cholesky floor ~1e-10) so it
    // is invariant to the overall scale of the Fisher information, falling back
    // to an absolute floor only when the diagonal is degenerate; (b) it is tried
    // first at ridge 0 so an already-SPD matrix is factored unperturbed; (c) it
    // grows geometrically (×4) to span the ~120 dB from the base jitter to O(1)
    // in a bounded number of steps; (d) the attempt count is capped so a
    // genuinely singular information (e.g. an exactly rank-deficient Fisher block)
    // surfaces as an explicit error rather than an unbounded loop.
    let invert_spd = |mat: &Array2<f64>,
                      context: &str|
     -> Result<(Array2<f64>, f64), EstimationError> {
        let max_diag = (0..d).fold(0.0_f64, |acc, i| acc.max(mat[[i, i]].abs()));
        let base = if max_diag.is_finite() && max_diag > 0.0 {
            max_diag * 1e-10
        } else {
            1e-10
        };
        let mut ridge = 0.0_f64;
        for _ in 0..=60 {
            let mut ridged = mat.clone();
            if ridge > 0.0 {
                for i in 0..d {
                    ridged[[i, i]] += ridge;
                }
            }
            if let Ok(factor) =
                factorize_symmetricwith_fallback(FaerArrayView::new(&ridged).as_ref(), Side::Lower)
            {
                let logdet = factor.logdet();
                if logdet.is_finite() {
                    let mut rhs = Array2::<f64>::eye(d);
                    {
                        let v = array2_to_matmut(&mut rhs);
                        factor.solve_in_place(v);
                    }
                    if rhs.iter().all(|x| x.is_finite()) {
                        let mut inv = Array2::<f64>::zeros((d, d));
                        for i in 0..d {
                            for j in 0..d {
                                inv[[i, j]] = 0.5 * (rhs[[i, j]] + rhs[[j, i]]);
                            }
                        }
                        return Ok((inv, logdet));
                    }
                }
            }
            ridge = if ridge > 0.0 { ridge * 4.0 } else { base };
        }
        Err(EstimationError::InvalidInput(format!(
            "multinomial Firth fallback: {context} not invertible (max_diag={max_diag:.3e})"
        )))
    };

    // SPD log-determinant only (no ridge): used by the backtracking line search to
    // reject any candidate that pushes a fitted probability to the simplex
    // boundary (where I loses positive-definiteness and the Firth term → −∞).
    let spd_logdet = |mat: &Array2<f64>| -> Option<f64> {
        factorize_symmetricwith_fallback(FaerArrayView::new(mat).as_ref(), Side::Lower)
            .ok()
            .map(|factor| factor.logdet())
            .filter(|ld| ld.is_finite())
    };

    // Penalized Firth objective ℓ* (MAXIMIZED), given probabilities, β, and the
    // precomputed log det I(β).
    let objective = |probs: &Array2<f64>, beta: &Array2<f64>, logdet_info: f64| -> f64 {
        let mut ll = 0.0_f64;
        for row in 0..n_obs {
            let w = weight(row);
            if w == 0.0 {
                continue;
            }
            for c in 0..k {
                let ycn = y_one_hot[[row, c]];
                if ycn != 0.0 {
                    ll += w * ycn * probs[[row, c]].max(f64::MIN_POSITIVE).ln();
                }
            }
        }
        // #2344: equivariant per-class penalty ½·Σ_{a,b} A[a,b]·β_aᵀSβ_b —
        // the same metric the shared vector-GLM engine applies, so the Firth
        // arm optimizes the identical reference-free objective.
        let a_mat = crate::penalized_vector_glm::equivariant_class_metric(lambdas, m);
        let mut pen = 0.0_f64;
        for a in 0..m {
            let bcol = beta.column(a);
            for b in 0..m {
                let coef = a_mat[[a, b]];
                if coef != 0.0 {
                    let sbeta = penalty.dot(&beta.column(b));
                    pen += 0.5 * coef * bcol.dot(&sbeta);
                }
            }
        }
        ll - pen + 0.5 * logdet_info
    };

    // Firth-adjusted penalized score U* (length d, block-ordered).
    let firth_score =
        |probs: &Array2<f64>, beta: &Array2<f64>, iinv: &Array2<f64>| -> Array1<f64> {
            let mut u = Array1::<f64>::zeros(d);
            let mut xn = vec![0.0_f64; p];
            let mut pa = vec![0.0_f64; m];
            let mut q = vec![0.0_f64; m * m];
            for row in 0..n_obs {
                let w = weight(row);
                if w == 0.0 {
                    continue;
                }
                for i in 0..p {
                    xn[i] = design[[row, i]];
                }
                for a in 0..m {
                    pa[a] = probs[[row, a]];
                }
                // Data score: U[(a,i)] += w x_{ni} (y_{na} − p_{na}).
                for a in 0..m {
                    let resid = y_one_hot[[row, a]] - pa[a];
                    let ao = a * p;
                    for i in 0..p {
                        u[ao + i] += w * xn[i] * resid;
                    }
                }
                // Per-row information hat Q_{ab} = x_nᵀ [I⁻¹]_{(a,b)} x_n.
                for a in 0..m {
                    let ao = a * p;
                    for b in 0..m {
                        let bo = b * p;
                        let mut s = 0.0_f64;
                        for i in 0..p {
                            let xi = xn[i];
                            if xi == 0.0 {
                                continue;
                            }
                            let mut inner = 0.0_f64;
                            for j in 0..p {
                                inner += iinv[[ao + i, bo + j]] * xn[j];
                            }
                            s += xi * inner;
                        }
                        q[a * m + b] = s;
                    }
                }
                // Firth adjustment: U[(c,s)] += ½ w x_{ns} h^c_n.
                for c in 0..m {
                    let pc = pa[c];
                    let mut h = 0.0_f64;
                    for a in 0..m {
                        for b in 0..m {
                            let dab = if a == b { 1.0 } else { 0.0 };
                            let dac = if a == c { 1.0 } else { 0.0 };
                            let dbc = if b == c { 1.0 } else { 0.0 };
                            let g =
                                dab * pa[a] * (dac - pc) - pa[a] * pa[b] * (dac + dbc - 2.0 * pc);
                            h += g * q[a * m + b];
                        }
                    }
                    let co = c * p;
                    for s in 0..p {
                        u[co + s] += 0.5 * w * h * xn[s];
                    }
                }
            }
            // Smoothing penalty gradient (#2344 equivariant metric):
            // U[(a,i)] −= Σ_b A[a,b]·(S β_b)_i.
            let a_mat = crate::penalized_vector_glm::equivariant_class_metric(lambdas, m);
            for b in 0..m {
                let sbeta = penalty.dot(&beta.column(b));
                for a in 0..m {
                    let coef = a_mat[[a, b]];
                    if coef == 0.0 {
                        continue;
                    }
                    let ao = a * p;
                    for i in 0..p {
                        u[ao + i] -= coef * sbeta[i];
                    }
                }
            }
            u
        };

    // Penalized Hessian H = I + A(λ) ⊗ S (#2344 equivariant metric; PSD sum
    // of rank-1 class projections, so H stays positive definite).
    let penalized_hessian = |info: &Array2<f64>| -> Array2<f64> {
        let mut h = info.clone();
        let a_mat = crate::penalized_vector_glm::equivariant_class_metric(lambdas, m);
        for a in 0..m {
            for b in 0..m {
                let coef = a_mat[[a, b]];
                if coef == 0.0 {
                    continue;
                }
                let (ao, bo) = (a * p, b * p);
                for i in 0..p {
                    for j in 0..p {
                        h[[ao + i, bo + j]] += coef * penalty[[i, j]];
                    }
                }
            }
        }
        h
    };

    // Solve H Δ = U* for the SPD penalized Hessian, ridge-escalating only on
    // factorization failure. Same relative-jitter Cholesky-recovery ladder as
    // `invert_spd` above (see its comment for the rationale); the base jitter is
    // one decade tighter (`max_diag · 1e-12`) because the penalized Hessian
    // solved here is better conditioned than the Fisher information inverted
    // there, so a smaller perturbation suffices before escalating.
    let solve_spd = |mat: &Array2<f64>,
                     rhs: &Array1<f64>|
     -> Result<Array1<f64>, EstimationError> {
        let max_diag = (0..d).fold(0.0_f64, |acc, i| acc.max(mat[[i, i]].abs()));
        let base = if max_diag.is_finite() && max_diag > 0.0 {
            max_diag * 1e-12
        } else {
            1e-12
        };
        let mut ridge = 0.0_f64;
        for _ in 0..=60 {
            let mut ridged = mat.clone();
            if ridge > 0.0 {
                for i in 0..d {
                    ridged[[i, i]] += ridge;
                }
            }
            if let Ok(factor) =
                factorize_symmetricwith_fallback(FaerArrayView::new(&ridged).as_ref(), Side::Lower)
            {
                let mut sol = rhs.clone();
                {
                    let v = array1_to_col_matmut(&mut sol);
                    factor.solve_in_place(v);
                }
                if sol.iter().all(|x| x.is_finite()) {
                    return Ok(sol);
                }
            }
            ridge = if ridge > 0.0 { ridge * 4.0 } else { base };
        }
        Err(EstimationError::InvalidInput(
            "multinomial Firth fallback: penalized Hessian solve failed".to_string(),
        ))
    };

    // ─────────────────────────── Firth Newton loop ────────────────────────────
    let (mut beta, completed_iterations) = match resume_from {
        Some(resume) => {
            if resume.coefficients.dim() != (p, m) {
                crate::bail_invalid_estim!(
                    "multinomial Firth resume coefficient shape {:?} does not match P x (K-1) = {p}x{m}",
                    resume.coefficients.dim(),
                );
            }
            (resume.coefficients.to_owned(), resume.completed_iterations)
        }
        None => (Array2::<f64>::zeros((p, m)), 0),
    };
    let mut iterations = completed_iterations;
    let mut stall_reason = FixedLambdaStallReason::IterationBudgetExhausted;
    let mut small_step_reached = false;
    for it in 0..max_iter {
        iterations = completed_iterations.checked_add(it + 1).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial Firth resume iteration count overflowed usize".to_string(),
            )
        })?;
        let probs = probs_at(&beta);
        let info = assemble_info(&probs);
        let (iinv, logdet_info) = invert_spd(&info, "Fisher information")?;
        let u = firth_score(&probs, &beta, &iinv);
        let hmat = penalized_hessian(&info);
        let step_vec = solve_spd(&hmat, &u)?;

        // Newton decrement ½ U*ᵀ H⁻¹ U* = ½ U*ᵀ Δ (≥ 0, scale-aware stop).
        let decrement = u.dot(&step_vec);
        if 0.5 * decrement.abs() < tol_eff {
            break;
        }

        // Δ as (P, M): delta[i, a] = step_vec[a·P + i].
        let mut delta = Array2::<f64>::zeros((p, m));
        for a in 0..m {
            let ao = a * p;
            for i in 0..p {
                delta[[i, a]] = step_vec[ao + i];
            }
        }

        // Backtracking line search on ℓ* (ascent) via the shared `opt`
        // primitive: t₀ = 1, halving up to 60 trials. A candidate whose expected
        // information `I` is not SPD (boundary) is an INVALID trial (`Ok(None)`),
        // so the search contracts without consulting the acceptance test, keeping
        // the iterate interior. The ascent predicate `o1 ≥ o0 − 1e-12` is inlined
        // verbatim, so the accepted step is bit-for-bit the hand-rolled loop's.
        let o0 = objective(&probs, &beta, logdet_info);
        let accepted_step = match backtracking_line_search::<_, Infallible>(
            BacktrackConfig::default(),
            |step| {
                let cand = &beta + &(&delta * step);
                let cand_probs = probs_at(&cand);
                let cand_info = assemble_info(&cand_probs);
                Ok(spd_logdet(&cand_info)
                    .map(|cand_logdet| (objective(&cand_probs, &cand, cand_logdet), cand)))
            },
            |_, o1| o1 >= o0 - 1e-12,
        ) {
            Ok(result) => result,
            Err(never) => match never {},
        };
        let Some(accepted_step) = accepted_step else {
            // Backtracking exhausted 60 halvings without an admissible ascent
            // step. This is convergence ONLY if the iterate is already first-order
            // stationary; a line-search stall at a non-stationary point is a
            // solver failure and must be reported as such, never papered over as
            // `converged = true` (#2066 — SPEC: do not report a non-converged
            // iterate as success).
            //
            // The verdict is the loop's OWN stationarity test — the Newton
            // decrement `½·Uᵀ H⁻¹ U` against `tol_eff`, the same criterion the top
            // of the loop uses to break as converged. A true interior mode never
            // reaches this branch: an infinitesimal step (`step → 0`) leaves the
            // iterate SPD with `o1 ≈ o0`, so it is accepted; a numerically flat
            // mode is caught by the `max_step` test below after that accepted
            // tiny step. Reaching here therefore means Newton still sees a
            // meaningful ascent direction it cannot realize (boundary / near-
            // singular Fisher information), i.e. a genuine stall → not converged.
            stall_reason = FixedLambdaStallReason::LineSearchExhausted;
            break;
        };

        let step = accepted_step.step;
        beta = accepted_step.payload;
        let max_step = step * delta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let scale = 1.0 + beta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if max_step < tol_eff * scale {
            small_step_reached = true;
            break;
        }
    }

    // ─────────────────────────── final quantities ─────────────────────────────
    for (idx, &v) in beta.iter().enumerate() {
        if !v.is_finite() {
            crate::bail_invalid_estim!(
                "multinomial Firth fallback: non-finite coefficient at flat index {idx} = {v}"
            );
        }
    }
    let coefficients_active = beta;

    let mut log_likelihood = 0.0_f64;
    let probs = probs_at(&coefficients_active);
    for row in 0..n_obs {
        let w = weight(row);
        for c in 0..k {
            let ycn = y_one_hot[[row, c]];
            if ycn != 0.0 {
                log_likelihood += w * ycn * probs[[row, c]].max(f64::MIN_POSITIVE).ln();
            }
        }
    }

    // #2344 equivariant metric: the reported penalty term matches the
    // objective the solve optimized.
    let a_mat = crate::penalized_vector_glm::equivariant_class_metric(lambdas, m);
    let mut penalty_term = 0.0_f64;
    for a in 0..m {
        let beta_col = coefficients_active.column(a);
        for b in 0..m {
            let coef = a_mat[[a, b]];
            if coef != 0.0 {
                let sbeta = penalty.dot(&coefficients_active.column(b));
                penalty_term += 0.5 * coef * beta_col.dot(&sbeta);
            }
        }
    }

    // Recompute the Firth score and Newton decrement AT the final accepted
    // iterate. A tiny backtracked coefficient step is not itself stationarity:
    // only this fresh first-order certificate may authorize construction of a
    // fit or its covariance.
    let info = assemble_info(&probs);
    let (information_inverse, final_logdet_info) = invert_spd(&info, "final Fisher information")?;
    let final_score = firth_score(&probs, &coefficients_active, &information_inverse);
    let hmat = penalized_hessian(&info);
    let final_step = solve_spd(&hmat, &final_score)?;
    let final_decrement = 0.5 * final_score.dot(&final_step).abs();
    if !(final_decrement.is_finite() && final_decrement < tol_eff) {
        if small_step_reached {
            stall_reason = FixedLambdaStallReason::StationarityCertificateFailed;
        }
        // SPEC: a fit object must only ever come from a converged optimization.
        // A Firth refit that exhausted its budget (or stalled its line search at
        // a non-stationary point) is the typed error carrying its evidence — the
        // covariance below is never computed for an uncertified iterate.
        let checkpoint = FixedLambdaCheckpoint::new(
            FixedLambdaSolverStage::MultinomialFirth,
            coefficients_active.iter().copied().collect(),
            p,
            m,
            iterations,
        )
        .map_err(|reason| {
            EstimationError::InvalidInput(format!(
                "multinomial Firth fallback produced an invalid internal checkpoint: {reason}"
            ))
        })?;
        return Err(EstimationError::FixedLambdaNewtonDidNotConverge {
            context: "fit_penalized_multinomial (Firth/Jeffreys separation refit)".to_string(),
            reason: stall_reason,
            objective_value: -objective(&probs, &coefficients_active, final_logdet_info),
            stationarity: FixedLambdaStationarityEvidence {
                kind: FixedLambdaResidualKind::NewtonDecrement,
                residual: final_decrement,
                bound: tol_eff,
            },
            checkpoint,
        });
    }

    // Laplace covariance H⁻¹ at the converged mode (block-ordered θ[a·P+i]).
    // A covariance that cannot be factored at a certified mode is a hard error,
    // never a silent zero matrix (a zero covariance is a false certainty claim).
    let (coefficient_covariance, _) = invert_spd(&hmat, "penalized Hessian covariance")?;

    Ok(MultinomialFitOutputs {
        coefficients_active,
        fitted_probabilities: probs,
        iterations,
        penalized_neg_log_likelihood: -log_likelihood + penalty_term,
        deviance: -2.0 * log_likelihood,
        coefficient_covariance,
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
#[serde(deny_unknown_fields)]
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
    /// on predict via [`gam_terms::smooth::build_term_collection_design`].
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
    /// Container type of the training table. `"unknown"` is the explicit value
    /// for Rust/CLI callers without a typed table container; the field is always
    /// present so persistence never invents presentation state while loading.
    pub training_table_kind: String,
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
    /// The separation evidence that armed the Jeffreys/Firth proper prior on
    /// this fit, or `None` when the unbiased penalized-REML criterion was
    /// accepted with the prior disarmed (#2612).
    ///
    /// The two branches publish **different estimands**. Disarmed, the
    /// coefficients are the exact penalized-REML mode with zero Firth bias;
    /// armed, they are the mode of that objective plus the proper prior
    /// `Φ = ½ log|ZᵀHZ|`, which pulls fitted class probabilities toward the
    /// uniform simplex `1/K` by an `O(1/n)` amount that #715 measured as a real
    /// truth-RMSE cost on interior data. A consumer scoring calibration, a
    /// reader comparing two fits, and the CLI summary all need to know which
    /// objective produced the numbers in front of them, and until #2612 the
    /// decision existed only in a `log::info!` line the caller never sees.
    ///
    /// The string is the certificate itself, not a flag: a verdict that carries
    /// the spectrum it was taken on can be checked, and one that carries only a
    /// boolean cannot.
    #[serde(default)]
    pub separation_evidence: Option<String>,
    /// Penalized negative log-likelihood at the returned `β̂`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)`.
    pub deviance: f64,
    /// Per-active-class effective degrees of freedom (hat-matrix trace),
    /// length `K - 1`. Populated when the REML driver reports an
    /// inference block; falls back to `None` for the legacy fixed-λ path.
    #[serde(default)]
    pub edf_per_class: Option<Vec<f64>>,
    /// Per-PENALTY effective degrees of freedom, one entry per smoothing
    /// parameter (length `== lambdas.len()`), aligned block-major with the flat
    /// [`Self::lambdas`] / [`Self::lambdas_per_block`] layout. Each entry is the
    /// penalty-block trace EDF `rank(S_k) − λ_k·tr(H⁻¹ S_k)`, clamped to
    /// `[0, rank(S_k)]`. This is the per-(class, term, penalty) resolution that
    /// the per-class [`Self::edf_per_class`] SUM deliberately hides: only the
    /// per-penalty vector reveals whether an individual smooth collapsed onto its
    /// polynomial null space (its wiggliness λ driven to the λ-cap), which a
    /// per-class total cannot show. Populated whenever the REML driver reports an
    /// inference block; `None` on the legacy fixed-λ path or when the trace
    /// channel is mis-shaped. Unlike `edf_per_class`, the entries do NOT sum to
    /// the model EDF when several penalties share one coefficient range (a
    /// double-penalty smooth has `Σ_k rank(S_k) > p_per_class`).
    #[serde(default)]
    pub edf_per_penalty: Option<Vec<f64>>,
    /// Joint posterior coefficient covariance `H⁻¹` (#1101), block-ordered to
    /// match the stacked active-class coefficient vector `β = [β_0; …; β_{K-2}]`
    /// (class `a`'s `P` coefficients occupy rows/cols `a·P .. (a+1)·P`). This is
    /// the Laplace covariance the REML driver already computes from the factored
    /// penalized Hessian; storing it makes posterior-mean prediction and its
    /// integrated uncertainty well-defined. Flattened row-major over the
    /// `(P·M)×(P·M)` matrix. This is required by the versioned persistence
    /// schema: a payload without covariance is not a usable multinomial model.
    pub coefficient_covariance_flat: Vec<f64>,
    /// The first-order smoothing-parameter-uncertainty correction
    /// `C = J·Var(ρ̂)·Jᵀ` (#2346), in the SAME raw units and block order as
    /// [`Self::coefficient_covariance_flat`], flattened row-major over the
    /// `(P·M)×(P·M)` matrix.
    ///
    /// `Self::coefficient_covariance_flat` is `V_cond = Var(β | λ̂)`: it answers
    /// "how wide is the posterior once the smoothing parameters are known
    /// exactly", and λ̂ is an estimate, not a known. Every other family in this
    /// library publishes the pair — `InferenceCovarianceMode::{Conditional,
    /// SmoothingCorrected}` in `gam-predict`, `beta_covariance` /
    /// `beta_covariance_corrected` on `FitInference` — and the multinomial is
    /// the one that never got it, so its bands were conditional-only
    /// (gam#2612, and the #1871 defect one family over).
    ///
    /// `C` is stored rather than `V_c = V_cond + C` deliberately: the sum is
    /// recoverable exactly by addition (see
    /// [`Self::coefficient_covariance_corrected`]) while the difference is not
    /// — subtracting two nearly-equal covariances would lose every digit of a
    /// correction that is small relative to `V_cond`, which is the regime this
    /// matrix is most often in.
    ///
    /// `None` is a TYPED ABSENCE, never a silent substitution: the outer solve
    /// retained no ρ curvature, or every ρ coordinate is railed and so has no
    /// finite variance to propagate (#2337 Thm 2.3).
    #[serde(default)]
    pub smoothing_correction_flat: Option<Vec<f64>>,
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
    /// Training design in the RAW basis, flattened row-major over `(n, P)`.
    ///
    /// # Why a saved multinomial model carries its own rows (#2612)
    ///
    /// The published prediction is the posterior MEAN probability
    /// `E[softmax(x'β)]`, and a Laplace summary — `β̂` plus
    /// [`Self::coefficient_covariance_flat`] — cannot produce it. That summary
    /// IS the quadratic model of the log-posterior, and the quadratic model is
    /// exactly what fails: integrating `softmax` against `N(β̂, H⁻¹)` keeps the
    /// curvature half of the `O(n⁻¹)` correction to a posterior mean and drops
    /// the skewness half, which on a (quasi-)separated fit is neither small nor
    /// the same sign. Computing the estimand honestly means evaluating the
    /// posterior away from the mode, which means the likelihood, which means the
    /// rows. See [`crate::multinomial_predictive`].
    ///
    /// This is the same choice `mgcv` makes when it stores the model frame with
    /// the fitted object, and it is required rather than optional: a payload
    /// without it cannot answer the question `predict` is asked.
    pub training_design_flat: Vec<f64>,
    /// Number of training rows `n`; segments [`Self::training_design_flat`].
    pub training_rows: usize,
    /// Training class index per row, values in `0..K`, aligned to
    /// [`Self::class_levels`].
    pub training_class_index: Vec<u32>,
    /// Training row weights, length `n`.
    pub training_weights: Vec<f64>,
    /// The joint penalty `S_λ` at the selected smoothing parameters, flattened
    /// row-major over `(P·M, P·M)` in the same stacked class-major order as
    /// [`Self::coefficient_covariance_flat`].
    ///
    /// Stored rather than rebuilt because the coupled `Σ_t λ_t (M ⊗ S_t)`
    /// carrier (#1587) is assembled from the family's equivariant specs, which
    /// the saved payload does not otherwise carry — and because reconstructing
    /// it as `Σ⁻¹(I − F)` would put an inverse of the very matrix whose
    /// conditioning is the problem on the prediction path.
    pub joint_penalty_flat: Vec<f64>,
    /// One descriptive label per *penalty component* within a single active-class
    /// block, parallel to that block's λ slice (i.e. length
    /// `lambdas_per_block[0]`). The Marra–Wood double penalty (and tensor /
    /// operator smooths) emit **more than one** penalty component — hence more
    /// than one λ — per smooth term, so this is NOT 1:1 with
    /// [`Self::smooth_term_spans`]: a single `s(x)` term contributes a primary
    /// wiggliness λ labelled `s(x)` and a null-space shrinkage λ labelled
    /// `s(x) [null space]`. The summary renderer pairs `lambdas` with these
    /// labels component-for-component so no λ is ever dropped (#1544). Built from
    /// the per-component term name + penalty role at fit time; empty only for a
    /// wholly parametric model.
    pub lambda_labels: Vec<String>,
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

/// Descriptive label for one penalty *component* (one λ) within a class block,
/// for the `summary()` per-class λ rollup (#1544). A smooth term can emit
/// several penalty components — the Marra–Wood double penalty splits `s(x)`
/// into a primary wiggliness penalty and a null-space shrinkage penalty, and
/// tensor / operator smooths emit a component per margin / differential
/// operator — each with its own independently-selected λ. The label is the
/// term name (from `PenaltyBlockInfo::termname`) plus a role suffix derived
/// from the penalty's [`PenaltySource`], so each λ in the summary names both
/// the term it smooths and the role it plays. `pen_idx` is the global penalty
/// index, used only as a last-resort fallback label.
fn penalty_component_label(info: Option<&PenaltyBlockInfo>, pen_idx: usize) -> String {
    use gam_terms::basis::PenaltySource;
    let term = info
        .and_then(|i| i.termname.clone())
        .unwrap_or_else(|| format!("s{pen_idx}"));
    let role = match info.map(|i| &i.penalty.source) {
        // The primary wiggliness penalty is the term's "main" λ; show the bare
        // term name so the common single-penalty case reads cleanly.
        Some(PenaltySource::Primary) | None => None,
        Some(PenaltySource::DoublePenaltyNullspace) => Some("null space".to_string()),
        Some(PenaltySource::OperatorMass) => Some("mass".to_string()),
        Some(PenaltySource::OperatorTension) => Some("tension".to_string()),
        Some(PenaltySource::OperatorStiffness) => Some("stiffness".to_string()),
        Some(PenaltySource::OperatorRelevance { axis }) => Some(format!("axis {axis}")),
        Some(PenaltySource::TensorMarginal { dim }) => Some(format!("margin {dim}")),
        Some(PenaltySource::TensorSeparable { penalized_margins }) => {
            Some(format!("separable {penalized_margins:?}"))
        }
        Some(PenaltySource::TensorGlobalRidge) => Some("ridge".to_string()),
        Some(PenaltySource::Other(s)) => Some(s.clone()),
    };
    match role {
        Some(role) => format!("{term} [{role}]"),
        None => term,
    }
}

impl MultinomialSavedModel {
    pub fn validate(&self) -> Result<(), EstimationError> {
        if self.p_per_class == 0 || self.n_active_classes == 0 {
            crate::bail_invalid_estim!(
                "multinomial saved model dimensions must be nonzero, got P={} and K-1={}",
                self.p_per_class,
                self.n_active_classes,
            );
        }
        if self.class_levels.len() != self.n_active_classes + 1 {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} class levels but K-1={}",
                self.class_levels.len(),
                self.n_active_classes,
            );
        }
        if self.reference_class_index != self.n_active_classes {
            crate::bail_invalid_estim!(
                "multinomial saved reference index {} does not equal the final class index {}",
                self.reference_class_index,
                self.n_active_classes,
            );
        }
        let d = self
            .p_per_class
            .checked_mul(self.n_active_classes)
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial saved coefficient dimension overflowed usize".to_string(),
                )
            })?;
        if self.coefficients_flat.len() != d {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} coefficient values, expected {d}",
                self.coefficients_flat.len(),
            );
        }
        if self.training_table_kind.trim().is_empty() {
            crate::bail_invalid_estim!(
                "multinomial saved model training_table_kind must be non-empty"
            );
        }
        if self.lambdas_per_block.len() != self.n_active_classes {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} lambda blocks, expected {}",
                self.lambdas_per_block.len(),
                self.n_active_classes,
            );
        }
        let lambda_count = self
            .lambdas_per_block
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count))
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial saved lambda count overflowed usize".to_string(),
                )
            })?;
        if lambda_count != self.lambdas.len() {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} lambdas but its blocks require {lambda_count}",
                self.lambdas.len(),
            );
        }
        if self
            .lambdas_per_block
            .iter()
            .any(|&count| count != self.lambda_labels.len())
        {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} lambda labels but block sizes {:?}",
                self.lambda_labels.len(),
                self.lambdas_per_block,
            );
        }
        if self
            .lambda_labels
            .iter()
            .any(|label| label.trim().is_empty())
        {
            crate::bail_invalid_estim!("multinomial saved model lambda labels must be non-empty");
        }
        let covariance_len = d.checked_mul(d).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial saved covariance dimension overflowed usize".to_string(),
            )
        })?;
        if self.coefficient_covariance_flat.len() != covariance_len {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} covariance values, expected {covariance_len}",
                self.coefficient_covariance_flat.len(),
            );
        }
        if let Some(correction) = self.smoothing_correction_flat.as_ref() {
            // A correction that is present but mis-shaped is worse than one that
            // is absent: absence is typed and the consumers fall back to the
            // conditional definition, while a wrong shape would silently pair a
            // ρ-uncertainty term with the wrong coefficients.
            if correction.len() != covariance_len {
                crate::bail_invalid_estim!(
                    "multinomial saved model has {} smoothing-correction values, expected \
                     {covariance_len}",
                    correction.len(),
                );
            }
        }
        if let Some((index, value)) = self
            .coefficients_flat
            .iter()
            .chain(self.coefficient_covariance_flat.iter())
            .chain(self.smoothing_correction_flat.iter().flat_map(|c| c.iter()))
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            crate::bail_invalid_estim!(
                "multinomial saved numeric payload is non-finite at combined index {index}: {value}"
            );
        }
        // #2612: the training frame is what makes the published posterior mean
        // computable at all, so its shape is a contract, not a hint.
        let design_len = self
            .training_rows
            .checked_mul(self.p_per_class)
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial saved training design size overflowed usize".to_string(),
                )
            })?;
        if self.training_design_flat.len() != design_len {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} training design values, expected \
                 {design_len} = {} rows x {} columns",
                self.training_design_flat.len(),
                self.training_rows,
                self.p_per_class,
            );
        }
        if self.training_class_index.len() != self.training_rows
            || self.training_weights.len() != self.training_rows
        {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} training rows, {} labels and {} weights",
                self.training_rows,
                self.training_class_index.len(),
                self.training_weights.len(),
            );
        }
        let n_classes = self.class_levels.len();
        if let Some(label) = self
            .training_class_index
            .iter()
            .find(|&&label| label as usize >= n_classes)
        {
            crate::bail_invalid_estim!(
                "multinomial saved training label {label} is outside 0..{n_classes}"
            );
        }
        if self.joint_penalty_flat.len() != covariance_len {
            crate::bail_invalid_estim!(
                "multinomial saved model has {} joint-penalty values, expected {covariance_len}",
                self.joint_penalty_flat.len(),
            );
        }
        if let Some((index, value)) = self
            .training_design_flat
            .iter()
            .chain(self.training_weights.iter())
            .chain(self.joint_penalty_flat.iter())
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            crate::bail_invalid_estim!(
                "multinomial saved training payload is non-finite at combined index \
                 {index}: {value}"
            );
        }
        Ok(())
    }

    /// The training frame and penalty this model carries, as the borrowed view
    /// [`crate::multinomial_predictive`] consumes.
    pub fn predictive_model<'a>(
        &'a self,
        training_design: ndarray::ArrayView2<'a, f64>,
        training_weights: ndarray::ArrayView1<'a, f64>,
        joint_penalty: ndarray::ArrayView2<'a, f64>,
    ) -> crate::multinomial_predictive::MultinomialPredictiveModel<'a> {
        crate::multinomial_predictive::MultinomialPredictiveModel {
            training_design,
            training_class_index: &self.training_class_index,
            training_weights,
            joint_penalty,
            n_classes: self.class_levels.len(),
        }
    }

    /// Training design as an `(n, P)` `ndarray`.
    pub fn training_design(&self) -> Result<Array2<f64>, EstimationError> {
        Array2::from_shape_vec(
            (self.training_rows, self.p_per_class),
            self.training_design_flat.clone(),
        )
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial saved training design is inconsistent with n x P: {error}"
            ))
        })
    }

    /// Joint penalty `S_λ` as a `(P·M, P·M)` `ndarray`.
    pub fn joint_penalty(&self) -> Result<Array2<f64>, EstimationError> {
        let d = self
            .p_per_class
            .checked_mul(self.n_active_classes)
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial saved joint-penalty dimension overflowed usize".to_string(),
                )
            })?;
        Array2::from_shape_vec((d, d), self.joint_penalty_flat.clone()).map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial saved joint penalty is inconsistent with (P·M)x(P·M): {error}"
            ))
        })
    }

    /// Active-class coefficient block as an `(P, K-1)` `ndarray` view.
    pub fn coefficients_active(&self) -> Result<Array2<f64>, EstimationError> {
        Array2::from_shape_vec(
            (self.p_per_class, self.n_active_classes),
            self.coefficients_flat.clone(),
        )
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial saved coefficient payload is inconsistent with P x (K-1): {error}"
            ))
        })
    }

    /// Reconstruct the joint posterior covariance `H⁻¹` as a `(P·M)×(P·M)`
    /// `ndarray`, block-ordered to match the stacked coefficient vector
    /// `θ[a·P + i] = β[i, a]` (#1101).
    pub fn coefficient_covariance(&self) -> Result<Array2<f64>, EstimationError> {
        let d = self
            .p_per_class
            .checked_mul(self.n_active_classes)
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "multinomial saved covariance dimension overflowed usize".to_string(),
                )
            })?;
        Array2::from_shape_vec((d, d), self.coefficient_covariance_flat.clone()).map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial saved covariance payload is inconsistent with (P*(K-1)) squared: {error}"
            ))
        })
    }

    /// Reconstruct the first-order smoothing correction `C = J·Var(ρ̂)·Jᵀ` as a
    /// `(P·M)×(P·M)` `ndarray`, block-ordered like
    /// [`Self::coefficient_covariance`] (#2346, gam#2612). `None` is the typed
    /// absence documented on [`Self::smoothing_correction_flat`].
    pub fn smoothing_correction(&self) -> Option<Array2<f64>> {
        let d = self.p_per_class.checked_mul(self.n_active_classes)?;
        let flat = self.smoothing_correction_flat.as_ref()?;
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

    /// Default posterior-mean class probabilities. This integrates
    /// `softmax(eta)` under the per-row Gaussian predictor posterior rather than
    /// evaluating softmax at the coefficient mode.
    pub fn predict_probabilities(
        &self,
        x_new: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Second moments cost `K(K+1)/2` extra augmented solves per row and
        // nothing here reads them, so the mean-only caller does not pay for
        // them.
        self.predictive_moments(x_new, false)
            .map(|moments| moments.class_mean)
    }

    /// Posterior-mean class probabilities and integrated marginal standard
    /// deviations at fresh design rows, under the covariance definition this
    /// model can support — [`Self::predict_probabilities_with_se_in_mode`] with
    /// `SmoothingCorrected` when the correction is present and `Conditional`
    /// when it is not.
    ///
    /// The DEFAULT is the corrected definition, matching
    /// `PredictUncertaintyOptions::default()` for every other family
    /// (`gam-predict`), because `λ̂` is an estimate and a band that conditions on
    /// it being exact is answering a narrower question than the caller asked.
    pub fn predict_probabilities_with_se(
        &self,
        x_new: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), EstimationError> {
        let (mean, standard_error, _) = self.predict_probabilities_with_se_and_source(x_new)?;
        Ok((mean, standard_error))
    }

    /// The same pair, plus the covariance definition it was actually built
    /// from. A caller that cannot tell a conditional band from an unconditional
    /// one cannot reason about the interval it was handed, so the provenance is
    /// returned rather than inferred.
    pub fn predict_probabilities_with_se_and_source(
        &self,
        x_new: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, InferenceCovarianceMode), EstimationError> {
        let source = if self.smoothing_correction_flat.is_some() {
            InferenceCovarianceMode::SmoothingCorrected
        } else {
            InferenceCovarianceMode::Conditional
        };
        let (mean, standard_error) = self.predict_probabilities_with_se_in_mode(x_new, source)?;
        Ok((mean, standard_error, source))
    }

    /// Posterior-mean class probabilities and their marginal standard
    /// deviations under an explicitly named covariance definition.
    ///
    /// # What the two modes are
    ///
    /// [`InferenceCovarianceMode::Conditional`] is `sd(p_c | λ̂)`, computed by
    /// [`crate::multinomial_predictive`] as `√(E[p_c²] − E[p_c]²)` from two
    /// ratios of normalising constants at the selected smoothing. It is the
    /// exact posterior spread of the probability GIVEN that `λ̂` is the truth.
    ///
    /// [`InferenceCovarianceMode::SmoothingCorrected`] additionally propagates
    /// the uncertainty in `λ̂` itself, by the law of total variance:
    ///
    /// ```text
    ///     Var(p_c) = E_ρ[ Var(p_c | ρ) ]  +  Var_ρ( E[p_c | ρ] )
    ///              ≈ Var(p_c | ρ̂)         +  (∂E[p_c|ρ]/∂ρ)ᵀ V_ρ (∂E[p_c|ρ]/∂ρ)
    /// ```
    ///
    /// and the second term needs no new object. To the order the correction is
    /// itself computed at, `E[p_c | ρ]` moves with `ρ` through the mode
    /// `θ̂(ρ)`, so `∂E[p_c|ρ]/∂ρ = gᵀ ∂θ̂/∂ρ` with `g = ∂p_c/∂θ` the softmax
    /// Jacobian at the mode, and
    ///
    /// ```text
    ///     Var_ρ( E[p_c|ρ] ) = gᵀ (J V_ρ Jᵀ) g = gᵀ C g
    /// ```
    ///
    /// with `C` exactly the matrix [`Self::smoothing_correction`] carries. That
    /// is the response-scale statement of `V_c = V_cond + C`: the same
    /// correction, contracted through the same delta-method Jacobian the rest of
    /// the library uses on the response scale, with no new constant and no new
    /// approximation order.
    ///
    /// `g` is evaluated at the PLUG-IN probabilities `softmax(x'β̂)` rather than
    /// at the published posterior mean. Those differ by `O(n⁻¹)`, and `g`
    /// multiplies a term that is itself the first-order correction, so the
    /// difference enters at an order neither term is accurate to; the plug-in
    /// Jacobian is the exact derivative of the leading term.
    ///
    /// # Errors
    ///
    /// Asking for `SmoothingCorrected` on a model that carries no correction is
    /// an ERROR, not a silent downgrade — the same contract
    /// `gam-predict`'s `SmoothingCorrected` mode holds.
    pub fn predict_probabilities_with_se_in_mode(
        &self,
        x_new: ArrayView2<'_, f64>,
        mode: InferenceCovarianceMode,
    ) -> Result<(Array2<f64>, Array2<f64>), EstimationError> {
        let moments = self.predictive_moments(x_new, true)?;
        let conditional = crate::multinomial_predictive::predictive_standard_deviation(&moments)?;
        match mode {
            InferenceCovarianceMode::Conditional => Ok((moments.class_mean, conditional)),
            InferenceCovarianceMode::SmoothingCorrected => {
                let correction = self.smoothing_correction().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "multinomial predict: the smoothing-corrected covariance was requested \
                         but this fit retained no ρ-uncertainty correction (no outer ρ curvature, \
                         or every ρ coordinate is railed and has no finite variance); ask for \
                         the conditional definition by name rather than receiving it unannounced"
                            .to_string(),
                    )
                })?;
                let smoothing_variance =
                    self.smoothing_variance_of_class_probability(x_new, correction.view())?;
                let mut total = conditional;
                for ((row, class), value) in total.indexed_iter_mut() {
                    *value = (*value * *value + smoothing_variance[[row, class]]).sqrt();
                }
                Ok((moments.class_mean, total))
            }
        }
    }

    /// `gᵀ C g` per (row, class): the response-scale variance the smoothing
    /// parameters' own uncertainty contributes to the mean class probability.
    ///
    /// The softmax Jacobian factorises as `g_c = u_c ⊗ x` with
    /// `u_c[a] = p_c(δ_{ca} − p_a)` over the `M` ACTIVE logits (the reference
    /// class `c = K−1` is included and simply has `δ_{ca} = 0` for every active
    /// `a`), so the whole `(row, class)` table costs one `M×M` Gram
    /// `Q[a,b] = xᵀ C_{ab} x` per row and then `K` quadratic forms of size `M`
    /// against it — never a `d`-dimensional contraction per class.
    fn smoothing_variance_of_class_probability(
        &self,
        x_new: ArrayView2<'_, f64>,
        correction: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let p = self.p_per_class;
        let m = self.n_active_classes;
        let k = self.class_levels.len();
        let d = p * m;
        if correction.dim() != (d, d) {
            crate::bail_invalid_estim!(
                "multinomial predict: smoothing correction is {}x{}, expected {d}x{d}",
                correction.nrows(),
                correction.ncols(),
            );
        }
        if x_new.ncols() != p {
            crate::bail_invalid_estim!(
                "multinomial predict: design has {} columns, expected {p}",
                x_new.ncols(),
            );
        }
        let coefficients = self.coefficients_active()?;
        let rows = x_new.nrows();
        let mut out = Array2::<f64>::zeros((rows, k));
        let mut gram = Array2::<f64>::zeros((m, m));
        // `C x_b` for each active block `b`, reused across `a` and across classes.
        let mut correction_times_x = Array2::<f64>::zeros((m, d));
        for row in 0..rows {
            let x = x_new.row(row);
            for b in 0..m {
                for i in 0..d {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += correction[[i, b * p + j]] * x[j];
                    }
                    correction_times_x[[b, i]] = acc;
                }
            }
            for a in 0..m {
                for b in 0..m {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += correction_times_x[[b, a * p + j]] * x[j];
                    }
                    gram[[a, b]] = acc;
                }
            }
            let eta: Vec<f64> = (0..m)
                .map(|a| (0..p).map(|i| x[i] * coefficients[[i, a]]).sum::<f64>())
                .collect();
            let probabilities = softmax_with_reference(&eta)?;
            for class in 0..k {
                let jacobian: Vec<f64> = (0..m)
                    .map(|a| {
                        let delta = if class == a { 1.0 } else { 0.0 };
                        probabilities[class] * (delta - probabilities[a])
                    })
                    .collect();
                let mut variance = 0.0;
                for a in 0..m {
                    for b in 0..m {
                        variance += jacobian[a] * gram[[a, b]] * jacobian[b];
                    }
                }
                // `C` is PSD by construction (`J V_ρ Jᵀ` with `V_ρ` PSD), so a
                // negative quadratic form here is round-off on a direction the
                // correction does not reach, not a measurement. Clamping at zero
                // is exact for that case and the alternative — a NaN from
                // `sqrt` — would destroy an otherwise valid band.
                out[[row, class]] = variance.max(0.0);
            }
        }
        Ok(out)
    }

    /// The posterior mode in the stacked class-major order the joint covariance,
    /// the joint penalty and [`crate::multinomial_predictive`] all use:
    /// `θ[a·P + i] = β[i, a]`.
    pub fn stacked_mode(&self) -> Result<Array1<f64>, EstimationError> {
        let coefficients = self.coefficients_active()?;
        let p = self.p_per_class;
        let m = self.n_active_classes;
        let mut theta = Array1::<f64>::zeros(p * m);
        for a in 0..m {
            for i in 0..p {
                theta[a * p + i] = coefficients[[i, a]];
            }
        }
        Ok(theta)
    }

    /// Posterior-predictive moments at fresh design rows, by the ratio of
    /// normalising constants (#2612).
    ///
    /// The Smolyak accuracy/level control this method used to take is gone
    /// rather than ignored. That control existed because the old mechanism was
    /// a quadrature whose answer could be bought with more nodes; this one is
    /// not, and its accuracy is a property of the expansion (`O(n⁻²)`) that no
    /// amount of extra work changes. What replaces it is a per-row exactness
    /// check the caller does not have to configure: `Σ_c E[p_c] = 1` is an
    /// identity of the estimand, so the deviation of the computed sum from one
    /// IS the approximation's error at that row, and a row past
    /// [`crate::multinomial_predictive::PREDICTIVE_MASS_DEFECT_TOLERANCE`] is
    /// refused rather than published. See [`crate::multinomial_predictive`] for
    /// why integrating `softmax` over `N(β̂, H⁻¹)` is not an approximation of
    /// this estimand at all.
    fn predictive_moments(
        &self,
        x_new: ArrayView2<'_, f64>,
        want_second_moments: bool,
    ) -> Result<crate::multinomial_predictive::MultinomialPredictiveMoments, EstimationError> {
        let design = self.training_design()?;
        let penalty = self.joint_penalty()?;
        let weights = Array1::from(self.training_weights.clone());
        let mode = self.stacked_mode()?;
        let model = self.predictive_model(design.view(), weights.view(), penalty.view());
        model.predictive_moments(mode.view(), x_new, want_second_moments)
    }

    /// Wood (2013) rank-truncated Wald smooth-significance test per
    /// `(active class, smooth term)` (#1101), reusing the exact scalar-summary
    /// kernel [`gam_terms::inference::smooth_test::wood_smooth_test`]. For active
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
        let Ok(cov) = self.coefficient_covariance() else {
            return out;
        };
        if self.smooth_term_spans.is_empty() {
            return out;
        }
        let Ok(beta) = self.coefficients_active() else {
            return out;
        };
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
                let result = gam_terms::inference::smooth_test::wood_smooth_test(
                    gam_terms::inference::smooth_test::SmoothTestInput {
                        beta: theta.view(),
                        covariance: &cov,
                        influence_matrix: influence.as_ref(),
                        whitening_gram: None,
                        coeff_range: start..end,
                        edf,
                        nullspace_dim: span.nullspace_dim,
                        residual_df: None,
                        scale: gam_terms::inference::smooth_test::SmoothTestScale::Known,
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

    /// Draw `n_draws` posterior-predictive replicate class assignments at fresh
    /// rows (#1101). Each draw independently samples every row's class from
    /// `Categorical(p_row)` with `p = E[softmax(eta) | data]`, so coefficient
    /// uncertainty is integrated before adding categorical observation noise.
    /// The returned `(n_draws, N)` matrix holds class
    /// INDICES `0..K`, aligned to [`Self::class_levels`]. The draw stream is a
    /// `StdRng` seeded by `seed`, so `(x_new, n_draws, seed)` reproduce
    /// bit-identically — the engine for posterior-predictive checks and
    /// simulation-based calibration. `x_new` must have `self.p_per_class`
    /// columns (built from the same `resolved_termspec` as fit time).
    pub fn sample_replicate_classes(
        &self,
        x_new: ArrayView2<'_, f64>,
        n_draws: usize,
        seed: u64,
    ) -> Result<Array2<u32>, EstimationError> {
        use rand::{RngExt, SeedableRng};
        let probs = self.predict_probabilities(x_new)?;
        let n = probs.nrows();
        let k = probs.ncols();
        let mut out = Array2::<u32>::zeros((n_draws, n));
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for d in 0..n_draws {
            for row in 0..n {
                let u: f64 = rng.random::<f64>();
                // Inverse-CDF categorical draw over the K simplex weights.
                let mut acc = 0.0_f64;
                let mut chosen = k - 1; // numerical fallback = reference class
                for c in 0..k {
                    acc += probs[[row, c]];
                    if u < acc {
                        chosen = c;
                        break;
                    }
                }
                out[[d, row]] = chosen as u32;
            }
        }
        Ok(out)
    }
}

/// On-disk `model_class` discriminator for a persisted multinomial model. Kept
/// as a single constant so every producer / consumer of the envelope agrees on
/// the tag without a scattered string literal.
pub const MULTINOMIAL_MODEL_CLASS: &str = "multinomial";
/// Exact multinomial persistence schema. Version 2 requires the canonical
/// per-component lambda labels and training-table provenance; successful
/// deserialization therefore yields a complete current model without repair.
pub const MULTINOMIAL_MODEL_FORMAT_VERSION: u32 = 2;

/// Round-trip persistence envelope for a fitted multinomial model. The
/// `model_class` discriminator lets a loader tell a multinomial payload apart
/// from the scalar `FittedModel` JSON before deserialising the whole struct.
///
/// This is the single definition of the multinomial on-disk format, shared by
/// the Python FFI (`fit_multinomial_formula` / `predict_multinomial_formula`)
/// and the `gam` CLI (`gam fit --family multinomial` / `gam predict`), so a
/// model persisted by one surface loads in the other.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultinomialModelEnvelope {
    pub model_class: String,
    pub format_version: u32,
    pub saved: MultinomialSavedModel,
}

impl MultinomialModelEnvelope {
    /// Wrap a fitted model with the canonical `model_class` tag.
    pub fn new(saved: MultinomialSavedModel) -> Result<Self, EstimationError> {
        saved.validate()?;
        Ok(Self {
            model_class: MULTINOMIAL_MODEL_CLASS.to_string(),
            format_version: MULTINOMIAL_MODEL_FORMAT_VERSION,
            saved,
        })
    }

    /// Serialize to the canonical JSON byte payload.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, EstimationError> {
        self.saved.validate()?;
        serde_json::to_vec(self).map_err(|err| {
            EstimationError::InvalidInput(format!("failed to serialize multinomial model: {err}"))
        })
    }

    /// Parse an envelope from JSON bytes, validating the `model_class`
    /// discriminator so a non-multinomial payload is rejected with a clear
    /// error rather than silently mis-predicted.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EstimationError> {
        // Gate on the envelope header (`model_class` + `format_version`) *before*
        // deserializing the versioned `saved` body. The header parse ignores the
        // body, so a payload that predates a field which has since become
        // required (e.g. `saved.formula`) is rejected on the version gate rather
        // than on whatever inner field is now missing — the version check is the
        // contract that tells a caller their payload is stale, and it must fire
        // first regardless of how the body schema has since evolved.
        #[derive(Deserialize)]
        struct EnvelopeHeader {
            #[serde(default)]
            model_class: Option<String>,
            #[serde(default)]
            format_version: Option<u32>,
        }
        let header: EnvelopeHeader = serde_json::from_slice(bytes).map_err(|err| {
            EstimationError::InvalidInput(format!("failed to deserialize multinomial model: {err}"))
        })?;
        match header.model_class.as_deref() {
            Some(MULTINOMIAL_MODEL_CLASS) => {}
            other => {
                return Err(EstimationError::InvalidInput(format!(
                    "multinomial model: model_class = {other:?}, expected {MULTINOMIAL_MODEL_CLASS:?}",
                )));
            }
        }
        match header.format_version {
            Some(MULTINOMIAL_MODEL_FORMAT_VERSION) => {}
            Some(version) => {
                return Err(EstimationError::InvalidInput(format!(
                    "multinomial model: format_version = {version}, expected {MULTINOMIAL_MODEL_FORMAT_VERSION}",
                )));
            }
            None => {
                return Err(EstimationError::InvalidInput(format!(
                    "multinomial model: format_version is absent (unversioned payload), expected {MULTINOMIAL_MODEL_FORMAT_VERSION}",
                )));
            }
        }
        let envelope: Self = serde_json::from_slice(bytes).map_err(|err| {
            EstimationError::InvalidInput(format!("failed to deserialize multinomial model: {err}"))
        })?;
        envelope.saved.validate()?;
        Ok(envelope)
    }
}

#[cfg(test)]
mod multinomial_persistence_contract_tests {
    use super::*;

    #[test]
    fn unversioned_payload_is_rejected() {
        let payload = br#"{"model_class":"multinomial","saved":{}}"#;
        let error = MultinomialModelEnvelope::from_json_bytes(payload)
            .expect_err("unversioned multinomial persistence must not be guessed");
        assert!(
            error.to_string().contains("format_version"),
            "unexpected persistence error: {error}"
        );
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
    let y_kind = crate::fit_orchestration::response_column_kind(data, y_col);
    let policy = resolved_resource_policy(config, ProblemHints::default());
    let mut inference_notes: Vec<String> = Vec::new();
    let spec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        &col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
        None,
    )
    .map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build termspec: {err}"))
    })?;
    let design = build_term_collection_design(data.values.view(), &spec).map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build design: {err}"))
    })?;
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        crate::bail_invalid_estim!(
            "multinomial fit does not support non-zero smooth anchors: the reference-coded \
             softmax requires an explicit affine offset for every non-reference class"
        );
    }
    Ok((spec, design, y_col, parsed.response, y_kind))
}

fn scale_multinomial_formula_penalty(penalty: PenaltyMatrix, scale: f64) -> PenaltyMatrix {
    match penalty {
        PenaltyMatrix::Dense(matrix) => PenaltyMatrix::Dense(matrix.mapv(|v| v * scale)),
        PenaltyMatrix::Diagonal(diagonal) => {
            PenaltyMatrix::Diagonal(diagonal.mapv(|value| value * scale))
        }
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

/// Canonical typed inputs for the formula-driven multinomial fit
/// ([`fit_penalized_multinomial_formula`]).
///
/// Every frontend (Rust, CLI, Python FFI) builds this one request, so the
/// warm-start / outer-search defaults live here rather than being duplicated
/// per caller. `config` is the same canonical [`FitConfig`] the scalar formula
/// families consume: `weight_column` is resolved against the dataset and
/// honored as per-row case weights, and fields the softmax family cannot
/// consume (offsets, noise/slope formulas, manual Firth, frailty, ...) are
/// rejected with a typed error instead of being silently dropped.
#[derive(Clone, Copy)]
pub struct MultinomialFitRequest<'a> {
    pub data: &'a EncodedDataset,
    pub formula: &'a str,
    pub config: &'a FitConfig,
    /// Warm-start seed for every per-(class, term) smoothing parameter; λ is
    /// REML/LAML-selected, so this only seeds the outer search.
    pub init_lambda: f64,
    /// OUTER REML/LAML smoothing-parameter iteration budget.
    pub max_iter: usize,
    /// Requested accuracy; drives the inner joint-Newton KKT target (see the
    /// control-split note inside the fit).
    pub tol: f64,
}

impl<'a> MultinomialFitRequest<'a> {
    /// The canonical production controls shared by the CLI and the Python FFI.
    pub fn new(data: &'a EncodedDataset, formula: &'a str, config: &'a FitConfig) -> Self {
        Self {
            data,
            formula,
            config,
            init_lambda: 1.0,
            max_iter: 50,
            tol: 1.0e-7,
        }
    }
}

/// Reject canonical-config fields the softmax multinomial family cannot
/// consume. Silently dropping a requested offset / noise model / manual Firth
/// toggle would quietly change the estimand the caller asked for (SPEC 3), so
/// every unsupported field is a typed error shared by all frontends.
fn reject_unsupported_multinomial_config(config: &FitConfig) -> Result<(), EstimationError> {
    if config.offset_column.is_some() || config.noise_offset_column.is_some() {
        crate::bail_invalid_estim!(
            "multinomial fit does not support offset columns: a single offset column has no \
             canonical per-logit placement in the reference-coded softmax (offsets are per-class \
             linear-predictor quantities); remove the offset or fit per-class models"
        );
    }
    if config.noise_formula.is_some() {
        crate::bail_invalid_estim!(
            "noise_formula is not supported for the multinomial family: the softmax likelihood \
             has no dispersion predictor"
        );
    }
    if config.slope_formula.is_some() || config.z_column.is_some() {
        crate::bail_invalid_estim!(
            "slope_formula/z_column is not supported for the multinomial family"
        );
    }
    if config.transformation_normal {
        crate::bail_invalid_estim!("transformation_normal conflicts with the multinomial family");
    }
    if config.expectile_tau.is_some() {
        crate::bail_invalid_estim!("expectile_tau requires the expectile family");
    }
    if config.firth {
        crate::bail_invalid_estim!(
            "manual firth is not accepted for the multinomial family: the Firth/Jeffreys \
             separation stabilizer is armed automatically on separation evidence"
        );
    }
    if !matches!(
        config.frailty,
        crate::survival::lognormal_kernel::FrailtySpec::None
    ) {
        crate::bail_invalid_estim!("frailty is not supported for the multinomial family");
    }
    Ok(())
}

/// Resolve the canonical `weight_column` into per-row case weights (`None` ⇒
/// uniform 1.0). Finiteness / non-negativity are enforced by
/// [`MultinomialFamily::new`], which owns the weight contract.
fn resolve_multinomial_row_weights(
    data: &EncodedDataset,
    config: &FitConfig,
) -> Result<Array1<f64>, EstimationError> {
    let Some(name) = config.weight_column.as_deref() else {
        return Ok(Array1::ones(data.values.nrows()));
    };
    let column = data.column_map().get(name).copied().ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "multinomial fit: weight column '{name}' not found in the dataset"
        ))
    })?;
    Ok(data.values.column(column).to_owned())
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
/// solve, a non-finite logit, or an exact full-span PENALIZED-curvature conditioning
/// certificate; see [`multinomial_formula_penalized_separation_evidence`]) is the
/// fit re-solved once with the full-span Firth prior armed, which bounds the
/// penalty-null directions no smoothing parameter can (`S v = 0` ⇒
/// `(H + S_λ) v = H v → 0` when the softmax likelihood has no identified finite
/// mode).
///
/// The categorical response column is recognised via the dataset schema
/// (`ColumnKindTag::Categorical`); reference class = last level. Returns a
/// [`MultinomialSavedModel`] that can be serialised to bytes for the Python
/// wrapper or used in-process for `predict_probabilities`.
/// Everything [`fit_penalized_multinomial_formula`] constructs BEFORE the REML
/// solve: the family (unbiased criterion, Jeffreys disarmed), the per-class
/// block specs with seeded `initial_log_lambdas`, the calibrated solver
/// options, and the design artifacts the post-solve repack consumes. Exposed
/// at crate level so diagnostics can drive the EXACT production objective at
/// fixed smoothing parameters (finite-difference gates on the outer ρ-gradient
/// of the coalesced joint penalty family, #2349) instead of replicating the
/// construction and diverging from it.
pub(crate) struct PenalizedMultinomialFormulaParts {
    pub(crate) family: MultinomialFamily,
    pub(crate) blocks: Vec<ParameterBlockSpec>,
    pub(crate) options: BlockwiseFitOptions,
    pub(crate) spec: TermCollectionSpec,
    pub(crate) design: TermCollectionDesign,
    pub(crate) class_levels: Vec<String>,
    pub(crate) parametric_standardization: Vec<(usize, f64, f64)>,
    pub(crate) penalties_arc: Arc<Vec<PenaltyMatrix>>,
    /// The design BEFORE the parametric standardization — the basis the saved
    /// coefficients and the predict-time rebuild share, and therefore the one
    /// the posterior-predictive ratio evaluates the likelihood in (#2612).
    pub(crate) raw_training_design: Array2<f64>,
    /// Training class index per row, values in `0..K` (#2612).
    pub(crate) training_class_index: Vec<u32>,
    /// Resolved per-row case weights (#2612).
    pub(crate) training_weights: Array1<f64>,
}

pub(crate) fn penalized_multinomial_formula_parts(
    request: &MultinomialFitRequest<'_>,
) -> Result<PenalizedMultinomialFormulaParts, EstimationError> {
    let MultinomialFitRequest {
        data,
        formula,
        config,
        init_lambda,
        max_iter,
        tol,
    } = *request;
    if !(init_lambda.is_finite() && init_lambda > 0.0) {
        crate::bail_invalid_estim!(
            "multinomial fit: init_lambda must be finite and > 0 (got {init_lambda})"
        );
    }
    reject_unsupported_multinomial_config(config)?;
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
    // A Binary response is promoted to a 2-level categorical for the
    // multinomial driver: the caller explicitly asked for multinomial, so we
    // route through the K-1 = 1 active-class softmax (equivalent math to
    // logistic). Anything that is neither Binary nor Categorical has no class
    // set to softmax over.
    let y_column_kind = data.column_kinds.get(y_col);
    if y_column_kind != Some(&ColumnKindTag::Binary)
        && y_column_kind != Some(&ColumnKindTag::Categorical)
    {
        crate::bail_invalid_estim!(
            "multinomial fit: response '{response_name}' must be a categorical column \
             (got column kind {:?})",
            y_column_kind
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
    // #2612: the RAW design, before the parametric standardization below, is the
    // basis the saved coefficients and the predict-time rebuild both live in, so
    // it is the one the posterior-predictive ratio must evaluate the likelihood
    // in. Snapshot it here rather than un-standardizing later: an inverse map
    // applied to a matrix is a second source of truth for the same rows.
    let raw_training_design = x_dense.clone();
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
    let n_obs = y_one_hot.nrows();
    let penalty_scale = multinomial_formula_penalty_scale(k);
    let per_term_penalties: Vec<PenaltyMatrix> = design
        .penalties_as_penalty_matrix()
        .into_iter()
        .map(|penalty| scale_multinomial_formula_penalty(penalty, penalty_scale))
        .collect();

    // ── Custom-family driven REML/LAML path ───────────────────────────────
    // Each active class becomes one ParameterBlockSpec, all sharing X and the
    // per-term penalty list. `initial_log_lambdas` is seeded from the caller's
    // `init_lambda` (one entry per term).
    let design_arc = Arc::new(x_dense);
    let penalties_arc = Arc::new(per_term_penalties);
    let weights = resolve_multinomial_row_weights(data, config)?;
    if weights.len() != n_obs {
        crate::bail_invalid_estim!(
            "multinomial fit: weight column length {} != N = {n_obs}",
            weights.len()
        );
    }
    // #2612: the class index the posterior-predictive ratio appends its extra
    // row's label from. Read off the one-hot the fit itself validated, so the
    // two cannot disagree about which class a row is.
    let training_class_index: Vec<u32> = (0..n_obs)
        .map(|row| {
            let mut best = 0usize;
            for class in 1..k {
                if y_one_hot[[row, class]] > y_one_hot[[row, best]] {
                    best = class;
                }
            }
            best as u32
        })
        .collect();
    let training_weights = weights.clone();
    // First attempt runs the UNBIASED penalized-REML criterion (no Firth
    // shrinkage toward the uniform simplex); the Jeffreys/Firth proper prior is
    // armed conditionally below, only on separation evidence (#715/#753 — see
    // `multinomial_formula_separation_evidence`).
    let log_init = init_lambda.ln();
    let family = MultinomialFamily::new(
        y_one_hot.clone(),
        weights,
        k,
        design_arc.clone(),
        penalties_arc.clone(),
    )
    .map_err(EstimationError::InvalidInput)?
    .with_joint_jeffreys_term(false)
    // gam#1587: the per-block smooth penalties are emptied (the centered `M⊗S_t`
    // joint penalty is the sole smoothing carrier), so the `init_lambda` warm
    // start must seed the JOINT penalty's `initial_log_lambda` — the per-block
    // `initial_log_lambdas` loop below is now a no-op (empty per-block list).
    .with_initial_log_lambda(log_init);
    let mut blocks = family.build_block_specs();
    for spec_block in blocks.iter_mut() {
        for v in spec_block.initial_log_lambdas.iter_mut() {
            *v = log_init;
        }
    }

    // ── Outer-derivative policy: dimension-gated exact curvature ────────────
    // Medium-D formula fits need exact curvature to keep lambda selection away
    // from over-smoothed caps, while smooth-by-factor models still avoid the
    // O(D²) dense Hessian path.
    //
    // `D` is the number of coordinates the OUTER search actually has, which is
    // the joint penalty spec count and NOT `(K−1) · n_penalties`. Under the
    // equivariant carrier (#1587) each penalty component emits one spec PER
    // CLASS, so a `K = 3` model carries `3·n_penalties` coordinates; the
    // per-block product this used to compute has not been the outer dimension
    // for any `K > 2` model since that landing. On the four-smooth penguin
    // fixture the two differ by 50% (`8·3 = 24` against `2·8 = 16`), which put
    // it on the opposite side of the threshold from the one
    // `MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM`'s own doc block says it was
    // chosen to keep it on.
    let total_rho_dim = family.joint_smoothing_dimension();
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
        ridge_policy: gam_problem::RidgePolicy::solver_only(),
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
    Ok(PenalizedMultinomialFormulaParts {
        family,
        blocks,
        options,
        spec,
        design,
        class_levels,
        parametric_standardization,
        penalties_arc,
        raw_training_design,
        training_class_index,
        training_weights,
    })
}

/// The coupled joint penalty `S_λ = Σ_s λ_s M_s` at the selected smoothing
/// parameters, in the stacked class-major coordinates the joint covariance, the
/// influence matrix and the posterior-mean predictive all share.
///
/// # Why the penalty is measured on its own (#2612)
///
/// `S_λ` used to be a by-product of the influence-matrix reconstruction
/// (`F = I − H⁻¹S_λ`), which needs two things the penalty does not:
///
/// * **at least one penalty component.** A wholly parametric multinomial —
///   `y ~ x1 + x2`, the shape both `quality_vs_statsmodels_multinomial` arms
///   fit — has none, so `S_λ` is the **zero operator**: a value, not an
///   absence. Reading it off a reconstruction that returns `None` on
///   `n_components == 0` turned "this model is unpenalized" into "this model
///   has no penalty operator", and the predictive then refused to publish any
///   parametric multinomial fit at all;
/// * **the joint posterior covariance `H⁻¹`.** That is a different measurement
///   of a different object. Conditioning the penalty's availability on it makes
///   a covariance failure surface as a missing penalty, naming the wrong one of
///   the two.
///
/// So the operator is assembled here from the family's own equivariant specs
/// and the selected `λ`, and every consumer reads THIS matrix — the influence
/// matrix and the published payload cannot describe different penalties.
///
/// # Basis
///
/// The specs live in the FITTED (standardized-parametric) basis, which is also
/// the basis of `training_design`'s penalized columns: the standardization
/// affine rescales parametric columns only and leaves every penalized column
/// identity-mapped, and `S_λ` is exactly zero outside the penalized columns. So
/// `A⁻ᵀS_λA⁻¹ = S_λ` and the published operator pairs with the raw design
/// without a change of variables. A penalty that ever reached a standardized
/// column would break that identity, which is why the zero-outside-penalized
/// structure is a property of the specs and not an assumption made here.
///
/// # Refusals
///
/// Only genuine inconsistencies: a selected-λ vector that does not match the
/// spec list, a spec of the wrong dimension, or λ reported for a model with
/// nothing to apply them to. Each of those is a disagreement between two parts
/// of the same fit, not a capability the fit lacks.
fn multinomial_joint_penalty_operator(
    joint_specs: &[gam_problem::JointPenaltySpec],
    joint_log_lambdas: Option<&Array1<f64>>,
    n_components: usize,
    expected_joint: usize,
) -> Result<Array2<f64>, EstimationError> {
    let mut s_lambda = Array2::<f64>::zeros((expected_joint, expected_joint));
    if n_components == 0 {
        let reported = joint_log_lambdas.map_or(0, |jll| jll.len());
        if reported != 0 || !joint_specs.is_empty() {
            crate::bail_invalid_estim!(
                "multinomial REML reported {reported} selected smoothing parameter(s) and {} \
                 coupled penalty spec(s) for a model carrying no penalty component; there is \
                 nothing for them to multiply",
                joint_specs.len(),
            );
        }
        // No penalized component exists, so the penalized log-posterior IS the
        // log-likelihood and the penalty is exactly zero.
        return Ok(s_lambda);
    }
    let jll = joint_log_lambdas.ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "multinomial REML converged with {n_components} penalty component(s) but surfaced no \
             selected joint smoothing parameters"
        ))
    })?;
    if jll.len() != joint_specs.len() || joint_specs.len() % n_components != 0 {
        crate::bail_invalid_estim!(
            "multinomial REML selected {} smoothing parameter(s) against {} coupled penalty \
             spec(s) over {n_components} component(s)",
            jll.len(),
            joint_specs.len(),
        );
    }
    for (spec, &log_lambda) in joint_specs.iter().zip(jll.iter()) {
        if spec.matrix.nrows() != expected_joint || spec.matrix.ncols() != expected_joint {
            crate::bail_invalid_estim!(
                "multinomial REML coupled penalty spec is {}x{}, expected \
                 {expected_joint}x{expected_joint}",
                spec.matrix.nrows(),
                spec.matrix.ncols(),
            );
        }
        s_lambda.scaled_add(log_lambda.exp(), &spec.matrix);
    }
    Ok(s_lambda)
}

pub fn fit_penalized_multinomial_formula(
    request: &MultinomialFitRequest<'_>,
) -> Result<MultinomialSavedModel, EstimationError> {
    let PenalizedMultinomialFormulaParts {
        family,
        blocks,
        options,
        spec,
        design,
        class_levels,
        parametric_standardization,
        penalties_arc,
        raw_training_design,
        training_class_index,
        training_weights,
    } = penalized_multinomial_formula_parts(request)?;
    let MultinomialFitRequest {
        data,
        formula,
        config,
        ..
    } = *request;
    let m = family.active_classes();
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
    //
    // BOTH solves run on the caller's own `outer_max_iter` (#2612). The probe
    // used to run under `outer_max_iter.min(MULTINOMIAL_UNBIASED_PROBE_OUTER_MAX_ITER)`
    // — a 20-iteration ceiling introduced as a bare one-line `perf(#1082)` change
    // (`8d9a7b53b`, no test, no measurement) — while EVERY `Err` from it was
    // routed below as separation evidence. Those two decisions are not
    // independent: SPEC forbids minting a fit from an exhausted budget, so an
    // outer search stopped by that ceiling returns `RemlDidNotConverge`, and the
    // ceiling could therefore convert "this probe was still descending when I
    // stopped it" into "the data separate" — and from there into a proper prior
    // that shrinks the published probabilities toward the uniform simplex `1/K`.
    //
    // Nor was the ceiling saving work where it was honest. If the probe certifies
    // within its budget the ceiling is a no-op: the search stops on its own
    // certificate either way. The ONLY runs it shortened are the ones where it
    // stopped a still-descending search — exactly the runs whose verdict it
    // changed. Its entire benefit was co-extensive with the misdecision. SPEC:
    // "Wall-clock time budgets and deadlines are never allowed, except in tests.
    // In general, do not paper over solver issues."
    //
    // The FINAL accepted Firth/Jeffreys refit likewise runs to the caller's full
    // outer budget: it is the result we ship, so it must reach the genuine REML
    // optimum, not a truncated iterate. The near-separable penguin refit that
    // motivated #1082's wall-clock concern is halted honestly at its true bound
    // optimum by the KKT-stationary-at-bound guard (`CostStallGuard`, #1082 /
    // 64711ed82) and the Newton-decrement residual certificate (363af9b56 /
    // 2c9580b1f): on separable data the outer ARC certifies and stops early on
    // its own, so no artificial iteration cap is needed to land in budget. On
    // non-separable data (e.g. the `vgam_smooth_by_factor` double-penalty arm)
    // the refit needs the caller's full budget to converge, which a `.min(20)`
    // cap would cut off — accepting a non-converged fit, which is dishonest.
    let unbiased_probe_options = &options;
    let firth_refit_options = &options;

    // The coupled penalty specs, built ONCE. They depend only on the family's
    // term structure, not on λ, so the same list serves the separation
    // certificate below (at the probe's selected λ) and the published `S_λ` at
    // the shipped fit's λ — the two cannot describe different penalty families.
    let n_penalty_components = penalties_arc.len();
    let joint_specs = if n_penalty_components == 0 {
        Vec::new()
    } else {
        family.equivariant_class_penalty_specs().map_err(|error| {
            EstimationError::InvalidInput(format!(
                "multinomial REML could not rebuild the coupled joint penalty: {error}"
            ))
        })?
    };

    let run_firth_refit =
        |evidence: String,
         measured_span: Option<std::sync::Arc<Array2<f64>>>,
         warm_start: Option<(&[ParameterBlockState], Option<&Array1<f64>>)>| {
            // The span the certificate MEASURED travels into the refit with the
            // verdict it justified (#2612). `None` leaves the derived
            // `ker(S_lambda)` route in place, which is what a degenerate mode gets.
            let mut firth_family = family
                .clone()
                .with_joint_jeffreys_term(true)
                .with_joint_jeffreys_span(measured_span);
            let mut firth_blocks = blocks.clone();
            if let Some((states, log_lambdas)) = warm_start {
                if states.len() != firth_blocks.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "multinomial Firth warm start has {} coefficient blocks, expected {}",
                        states.len(),
                        firth_blocks.len()
                    )));
                }
                for (block, state) in firth_blocks.iter_mut().zip(states) {
                    if state.beta.len() != block.design.ncols() {
                        return Err(EstimationError::InvalidInput(format!(
                            "multinomial Firth warm-start block '{}' has {} coefficients, expected {}",
                            block.name,
                            state.beta.len(),
                            block.design.ncols()
                        )));
                    }
                    if state.beta.iter().any(|value| !value.is_finite()) {
                        return Err(EstimationError::InvalidInput(format!(
                            "multinomial Firth warm-start block '{}' contains non-finite coefficients",
                            block.name
                        )));
                    }
                    block.initial_beta = Some(state.beta.clone());
                }
                if let Some(log_lambdas) = log_lambdas {
                    if log_lambdas.iter().any(|value| !value.is_finite()) {
                        return Err(EstimationError::InvalidInput(
                            "multinomial Firth warm start contains non-finite log-lambdas"
                                .to_string(),
                        ));
                    }
                    firth_family =
                        firth_family.with_joint_initial_log_lambdas(log_lambdas.to_vec());
                }
            }
            log::info!(
                "multinomial REML: arming the Jeffreys/Firth proper prior — separation evidence: \
             {evidence}"
            );
            fit_custom_family_with_rho_prior(
                &firth_family,
                &firth_blocks,
                firth_refit_options,
                gam_problem::RhoPrior::Flat,
            )
            .map_err(|err| {
                EstimationError::InvalidInput(format!(
                    "multinomial REML: Firth/Jeffreys-armed refit (separation evidence: \
                 {evidence}) failed: {err}"
                ))
            })
        };

    // The separation scans walk the full P×M logit block, so they are evaluated
    // ONCE into a binding and branched on, rather than recomputed in both a match
    // guard and its arm (three to four full logit walks per fit on the hot
    // near-separable path).
    let probe_attempt = fit_custom_family_with_rho_prior(
        &family,
        &blocks,
        unbiased_probe_options,
        gam_problem::RhoPrior::Flat,
    );
    // The evidence that armed the proper prior, or `None` when the unbiased
    // penalized-REML criterion was accepted. This is a FACT ABOUT THE FIT — the
    // two branches publish different estimands (the unbiased mode versus the
    // Firth-biased one) — so it is carried to the payload rather than living only
    // in a log line the caller never sees.
    let (fit, separation_evidence) = match probe_attempt {
        Ok(probe_fit) => {
            let identifiable_span = probe_fit
                .geometry
                .as_ref()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "multinomial unbiased fit omitted its certified coefficient geometry"
                            .to_string(),
                    )
                })?
                .coefficient_gauge
                .t_full
                .view();
            // #2612: a model with NO penalty component selects no smoothing
            // parameter, so `joint_log_lambdas` is legitimately absent — the same
            // "no penalty is a value, not an absence" the joint-penalty operator
            // carries. Only a PENALIZED probe that reached a certified optimum
            // without surfacing its own selected λ is a broken fit, and only that
            // is refused. The certificate below needs those λ to form the
            // curvature the mode actually sits in, so the check comes first.
            let joint_log_lambdas = probe_fit.artifacts.joint_log_lambdas.as_ref();
            if joint_log_lambdas.is_none() && n_penalty_components != 0 {
                return Err(EstimationError::InvalidInput(format!(
                    "multinomial unbiased fit carries {n_penalty_components} penalty \
                     component(s) but omitted its converged joint log-lambdas"
                )));
            }
            let separation = multinomial_formula_penalized_separation_evidence(
                &family,
                &blocks,
                &probe_fit.block_states,
                identifiable_span,
                &joint_specs,
                n_penalty_components,
                joint_log_lambdas,
            )
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "multinomial REML separation certification failed: {error}"
                ))
            })?;
            match separation {
                None => {
                    // Fit existence proves both optimization layers certified; no
                    // post-hoc convergence flag is needed.
                    log::info!(
                        "multinomial REML: unbiased criterion accepted (no separation evidence; \
                         Jeffreys/Firth prior disarmed)"
                    );
                    (probe_fit, None)
                }
                Some(certificate) => {
                    let MultinomialSeparationCertificate {
                        evidence,
                        measured_span,
                    } = certificate;
                    // A certified unbiased optimum can still exhibit separation;
                    // use the already-computed mode and rho as the exact Firth
                    // continuation seed. The objectives differ only by the smooth
                    // Jeffreys term, so restarting from zero would discard the
                    // strongest available local information and repeat the whole
                    // unbiased path.
                    let refit = run_firth_refit(
                        evidence.clone(),
                        measured_span,
                        Some((&probe_fit.block_states, joint_log_lambdas)),
                    )?;
                    (refit, Some(evidence))
                }
            }
        }
        Err(err) => {
            let evidence = format!(
                "the unbiased criterion has no certified optimum on the caller's own outer \
                 budget ({} iteration(s)): {err}",
                options.outer_max_iter,
            );
            // No certified mode, so no measured span: the derived
            // `ker(S_lambda)` route stands.
            let refit = run_firth_refit(evidence.clone(), None, None)?;
            (refit, Some(evidence))
        }
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
    let expected_joint = p_per_class.checked_mul(m).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior covariance dimension overflowed usize".to_string(),
        )
    })?;
    // ── The penalty operator, measured once (#2612) ──────────────────────────
    // `S_λ` is a fact about the fit in its own right: the influence-matrix
    // reconstruction below and the published predictive payload both read THIS
    // matrix, so they cannot describe different penalties, and a model with no
    // penalized term publishes the zero operator it has instead of being refused
    // for not having one. See `multinomial_joint_penalty_operator`.
    //
    // `joint_specs` was built once above the probe (it is λ-independent) and is
    // the same list the separation certificate formed `H + S_λ` from, so the
    // arming decision, the influence matrix and the published payload all read
    // one penalty family. Under the equivariant carrier this is K per-class specs
    // per term, grouped term-major (`s = t·K + c`); the K = 2 degenerate arm
    // returns one shared centered spec per term.
    let joint_penalty = multinomial_joint_penalty_operator(
        &joint_specs,
        fit.artifacts.joint_log_lambdas.as_ref(),
        n_penalty_components,
        expected_joint,
    )?;
    // ── gam#1587/#561 joint-penalty reconstruction ───────────────────────────
    // Under the #1587 centered-metric architecture every active class block
    // leaves its per-block penalty list EMPTY — the entire fit's smoothing rides
    // on a single full-width JOINT penalty `S_λ = Σ_t λ_t (M ⊗ S_t)` whose one
    // shared `λ_t` per smooth component is selected by the outer REML loop and
    // surfaced on `fit.artifacts.joint_log_lambdas`. So `fit.blocks[a].lambdas`
    // is `[]`, the inference layer's per-block trace channel is empty, and the
    // older per-block reporting (`lambdas_per_block = [0, 0]`, `edf_per_class =
    // None`, …) collapsed (#561 reopen).
    //
    // Reconstruct the per-(class, component) λ and the influence-matrix EDF
    // directly from the selected joint `λ_t` and the COUPLED penalty
    // `S_λ = Σ_t λ_t (M ⊗ S_t)` (NOT a block-diagonal `Σ_t λ_{a,t} S_t`: the
    // centered metric `M` couples classes off the block diagonal, so a
    // block-diagonal `S_λ` would mis-state both the influence matrix and every
    // trace). With `H⁻¹ = fit.covariance_conditional` now assembled WITH the
    // joint penalty (the `compute_joint_covariance` fix), the influence matrix is
    // exactly `F = I − H⁻¹ S_λ`, its per-class diagonal-block trace is the honest
    // per-class EDF, and `Σ_a edf_a = tr(F) = edf_total`.
    let joint_recon = fit.artifacts.joint_log_lambdas.as_ref().and_then(|jll| {
        let n_components = n_penalty_components;
        if n_components == 0 {
            return None;
        }
        // `joint_specs` is the list `joint_penalty` was assembled from, so the
        // influence matrix and the published operator cannot describe different
        // penalties.
        if jll.len() != joint_specs.len() || joint_specs.len() % n_components != 0 {
            return None;
        }
        let specs_per_term = joint_specs.len() / n_components;
        let hinv = fit
            .covariance_conditional
            .as_ref()
            .filter(|c| c.nrows() == expected_joint && c.ncols() == expected_joint)?;
        let lam: Vec<f64> = jll.iter().map(|&l| l.exp()).collect();
        // Per-spec `H⁻¹ M_s` (full mp×mp), reused for both the joint influence
        // matrix and the per-(class, component) trace decomposition.
        let mut hinv_st: Vec<Array2<f64>> = Vec::with_capacity(joint_specs.len());
        for spec in &joint_specs {
            if spec.matrix.nrows() != expected_joint || spec.matrix.ncols() != expected_joint {
                return None;
            }
            hinv_st.push(hinv.dot(&spec.matrix));
        }
        // F = I − H⁻¹ S_λ = I − Σ_s λ_s H⁻¹ M_s.
        let mut f = Array2::<f64>::eye(expected_joint);
        for (s, hs) in hinv_st.iter().enumerate() {
            f.scaled_add(-lam[s], hs);
        }
        // Per-class diagonal-block trace of F (the honest per-class EDF), and
        // the per-(class, component) penalty trace
        // `tr_{a,t} = Σ_{c∈term t} λ_{t,c} · Σ_{i∈class a} (H⁻¹ M_{t,c})[i,i]`
        // for the per-penalty EDF rollup.
        let mut edf_per_class = Vec::with_capacity(m);
        // class-major per-penalty EDF (class 0's components, then class 1's, …),
        // aligned 1:1 with the flat per-(class, component) λ report below.
        let mut edf_per_penalty = Vec::with_capacity(m * n_components);
        for a in 0..m {
            let base = a * p_per_class;
            let mut class_trace = 0.0_f64;
            for t in 0..n_components {
                let mut tr_at = 0.0_f64;
                for c in 0..specs_per_term {
                    let s = t * specs_per_term + c;
                    let mut tr = 0.0_f64;
                    for i in 0..p_per_class {
                        tr += hinv_st[s][[base + i, base + i]];
                    }
                    tr_at += lam[s] * tr;
                }
                class_trace += tr_at;
                // A single component's per-class trace EDF `rank(S_t) − tr_{a,t}`,
                // bounded by its local rank (≤ p_per_class). Derive rank(S_t)
                // from the spec's MEASURED nullity (per-class spec: rank =
                // m·p − nullspace_dim; shared centered spec: m·rank), so the
                // reporting rank matches the pseudo-logdet rank exactly.
                let spec0 = &joint_specs[t * specs_per_term];
                let joint_rank = expected_joint - spec0.nullspace_dim;
                let rank_t = if specs_per_term > 1 {
                    joint_rank as f64
                } else {
                    (joint_rank as f64) / (m as f64)
                };
                edf_per_penalty.push((rank_t - tr_at).clamp(0.0, p_per_class as f64));
            }
            edf_per_class.push((p_per_class as f64 - class_trace).clamp(0.0, p_per_class as f64));
        }
        // Per-(class, component) λ report, class-major. Under the equivariant
        // carrier the smoothing applied to active class `a`'s centered function
        // for term `t` is its own `λ_{t,c=a}` (spec index `t·K + a`); under the
        // K = 2 shared arm every class reports the one `λ_t`.
        let mut lam_flat = Vec::with_capacity(m * n_components);
        for a in 0..m {
            for t in 0..n_components {
                let s = if specs_per_term > 1 {
                    t * specs_per_term + a
                } else {
                    t
                };
                lam_flat.push(lam[s]);
            }
        }
        Some((f, edf_per_class, edf_per_penalty, n_components, lam_flat))
    });

    // Flatten every (class, component) smoothing parameter in class-major order.
    // Under the equivariant joint-penalty architecture each active class `a`
    // reports its own `λ_{t,a}` per term (the per-class centered penalties;
    // the K = 2 degenerate arm replicates the shared `λ_t`), so the flat vector
    // is class-major with `lambdas_per_block = [n_components; K-1]`. When the
    // joint reconstruction is unavailable (legacy fixed-λ path or absent
    // covariance) fall back to the raw — now empty — per-block λ lists.
    let (lambdas_per_block, lambdas_flat): (Vec<usize>, Vec<f64>) = match joint_recon.as_ref() {
        Some((_, _, _, n_components, lam_flat)) => {
            let per_block = vec![*n_components; m];
            (per_block, lam_flat.clone())
        }
        None => {
            let per_block: Vec<usize> = fit.blocks.iter().map(|b| b.lambdas.len()).collect();
            let flat: Vec<f64> = fit
                .blocks
                .iter()
                .flat_map(|b| b.lambdas.iter().copied())
                .collect();
            (per_block, flat)
        }
    };
    // Per-active-class effective degrees of freedom, length `K-1`, summing to
    // the model `edf_total`. The REML inference block reports `edf_by_block` as
    // ONE entry per *penalty block* (per (class, term, penalty)), each computed
    // as `rank(S_kk) − tr(H⁻¹ λ_kk S_kk)`. That per-block sum OVER-COUNTS the
    // model EDF whenever several penalties share one coefficient range — a
    // double-penalty / te / ti / adaptive smooth has ≥2 penalty blocks over the
    // same columns, so `Σ_kk rank(S_kk) > p` and `Σ_kk edf_by_block > edf_total`
    // (the observed ~79 for a ~24-coefficient model). Handing that raw per-block
    // vector out as the documented length-(K-1) per-class EDF is therefore both
    // the wrong LENGTH (it is `Σ_a n_blocks_a`, not `K-1`) and an over-count.
    //
    // The honest per-class EDF is the influence-matrix trace over each class's
    // coefficient block. Classes occupy DISJOINT `p_per_class`-wide coefficient
    // ranges, and the per-block traces `tr_kk = tr(H⁻¹ λ_kk S_kk)` are additive
    // (no rank double-counting), so class `a`'s EDF is
    // `p_per_class − Σ_{kk ∈ class a} tr_kk`, and `Σ_a edf_a = m·p_per_class −
    // Σ_kk tr_kk = p − Σ tr_kk = edf_total` exactly. Segment the block-major
    // `penalty_block_trace` by `lambdas_per_block` (the same per-class λ-count
    // segmentation `lambdas_flat` uses). Fall back to `None` when the trace
    // channel is unavailable or mis-shaped (legacy fixed-λ path), exactly as the
    // raw `edf_by_block` map did before.
    let edf_per_class = joint_recon
        .as_ref()
        .map(|(_, epc, _, _, _)| epc.clone())
        .or_else(|| {
            // Legacy per-block trace path (fixed-λ / pre-#1587 fits whose
            // smoothing is still carried per block). Segment the block-major
            // `penalty_block_trace` by `lambdas_per_block`, exactly as before.
            fit.inference.as_ref().and_then(|info| {
                let traces = &info.penalty_block_trace;
                if traces.len() != lambdas_per_block.iter().sum::<usize>() {
                    return None;
                }
                let mut per_class = Vec::with_capacity(m);
                let mut cursor = 0usize;
                for &n_blocks in &lambdas_per_block {
                    let class_trace: f64 = traces[cursor..cursor + n_blocks].iter().sum();
                    per_class
                        .push((p_per_class as f64 - class_trace).clamp(0.0, p_per_class as f64));
                    cursor += n_blocks;
                }
                Some(per_class)
            })
        });
    // Per-PENALTY EDF: the inference layer's `edf_by_block` is already the
    // clamped per-penalty-block trace EDF `rank(S_k) − λ_k·tr(H⁻¹ S_k)`, one
    // entry per smoothing parameter and block-major aligned 1:1 with the flat
    // `lambdas`. Surface it verbatim (guarding only on the length contract) so
    // consumers can inspect per-(class, term, penalty) collapse onto the null
    // space — a signal the per-class EDF SUM hides. This is NOT a per-class
    // total: with double-penalty smooths `Σ_k rank(S_k) > p_per_class`, so the
    // entries deliberately need not sum to the model EDF (the per-class field
    // carries that contract instead).
    let edf_per_penalty = joint_recon
        .as_ref()
        .map(|(_, _, epp, _, _)| epp.clone())
        .or_else(|| {
            // Legacy per-block path: the inference layer's `edf_by_block` is
            // already the clamped per-penalty-block trace EDF, aligned 1:1 with
            // the flat `lambdas`.
            fit.inference.as_ref().and_then(|info| {
                if info.edf_by_block.len() != lambdas_flat.len() {
                    return None;
                }
                Some(
                    info.edf_by_block
                        .iter()
                        .map(|&e| e.max(0.0))
                        .collect::<Vec<f64>>(),
                )
            })
        });
    let coefficients_flat: Vec<f64> = coefficients_active.iter().copied().collect();

    // #1101: surface the joint Laplace posterior covariance `H⁻¹` (block-ordered
    // [β_0; …; β_{K-2}]) and the influence matrix `F = H⁻¹ X'WX` the REML driver
    // computed at the converged mode. These power the predict path's delta-method
    // per-class probability standard errors and the summary's Wald smooth-term
    // tests. The joint matrices are `(P·M)×(P·M)`. The covariance is mapped back
    // to RAW units (see below) so it pairs with the raw predict design; the
    // influence is kept in the fitted basis (the Wald table only slices penalized
    // columns, which the standardization affine leaves identity-mapped).
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
    // One raw-unit map, used by BOTH joint covariance-frame matrices. `C` lives
    // in the same lifted frame as `V_cond` (it is assembled against it — see
    // `BlockwiseFitAssembly::smoothing_corrected`), so it must take the same
    // congruence; mapping only one of the pair would publish `V_cond + C` in two
    // different parameterizations.
    let to_raw_units = |matrix_std: &Array2<f64>| -> Vec<f64> {
        if parametric_standardization.is_empty() {
            return matrix_std.iter().copied().collect::<Vec<f64>>();
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
        let raw = a_joint.dot(matrix_std).dot(&a_joint.t());
        raw.iter().copied().collect::<Vec<f64>>()
    };
    let coefficient_covariance_flat = fit
        .covariance_conditional
        .as_ref()
        .filter(|c| c.nrows() == expected_joint && c.ncols() == expected_joint)
        .map(|cov_std| to_raw_units(cov_std))
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "multinomial REML converged without the required {expected_joint}x{expected_joint} joint posterior covariance"
            ))
        })?;
    // gam#2612: the same fit already computed the first-order ρ-uncertainty
    // correction `C = J·Var(ρ̂)·Jᵀ` (#2346) and published it on the inference
    // block; every other family in the library carries it onto its predict
    // surface and this one dropped it at the read boundary, which is what made
    // every multinomial band conditional-on-λ̂ and the standing coverage gate
    // for the mean-probability band anti-conservative. A mis-shaped correction
    // is a typed absence rather than an error: the conditional definition is
    // still a correct answer to a narrower question, and the consumers say which
    // one they used.
    let smoothing_correction_flat = fit.inference.as_ref().and_then(|info| {
        info.smoothing_correction
            .as_ref()
            .filter(|c| c.nrows() == expected_joint && c.ncols() == expected_joint)
            .map(|c| to_raw_units(c))
    });
    // The influence matrix `F = H⁻¹ X'WX = H⁻¹(H − S_λ) = I − H⁻¹ S_λ`. The
    // exact-Newton multinomial blocks carry no IRLS pseudo-data, so the generic
    // inference path does not export `coefficient_influence`; reconstruct it
    // exactly here. Under the #1587 joint-penalty architecture the penalty is the
    // COUPLED centered metric `S_λ = Σ_t λ_t (M ⊗ S_t)` (off the class-block
    // diagonal), already assembled in `joint_recon` above, so reuse that exact
    // `F`. Only fall back to the legacy block-diagonal `Σ_t λ_{a,t} S_t`
    // reconstruction when the joint reconstruction is unavailable (pre-#1587
    // per-block fits whose class blocks still carry their own penalties).
    let coefficient_influence_flat = match joint_recon.as_ref() {
        Some((f, _, _, _, _)) => Some(f.iter().copied().collect::<Vec<f64>>()),
        None => fit
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
            }),
    };

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

    // One descriptive label per penalty *component* within a single class block,
    // parallel to that block's λ slice (#1544). `design.penalties` is index-
    // parallel to every active class's `block.lambdas` (each block carries the
    // full per-component penalty list, validated above by
    // `block.lambdas.len() == penalties_arc.len()`), so iterating it in order
    // yields exactly `lambdas_per_block[0]` labels aligned with the per-block λ.
    // This is deliberately NOT deduped by col_range (unlike `smooth_term_spans`):
    // the double penalty's primary and null-space components share one col_range
    // but select independent λ, and each must keep its own label so the summary
    // renderer never collapses or drops a λ.
    let lambda_labels: Vec<String> = design
        .penalties
        .iter()
        .enumerate()
        .map(|(pen_idx, _)| penalty_component_label(design.penaltyinfo.get(pen_idx), pen_idx))
        .collect();

    // Unpenalized deviance read directly from the converged unpenalized
    // log-likelihood the rho-prior driver already computed (issue #348):
    // MultinomialFamily::evaluate sets FamilyEvaluation.log_likelihood =
    // log_lik(η, y) with no penalty term, and that value flows unchanged into
    // UnifiedFitResult.log_likelihood. This reproduces the legacy fixed-λ
    // path's `deviance = -2 · log_lik` contract bit-for-bit, so the previous
    // row-by-row η = Xβ rebuild and softmax recompute were pure dead work.
    let deviance = -2.0 * fit.log_likelihood;

    // #2612: the training frame and the penalty operator the posterior-mean
    // predictive needs. `joint_penalty` was measured above from the family's own
    // equivariant specs and the selected λ — the same matrix the influence
    // reconstruction consumed — so the payload and `F` cannot describe different
    // penalties, and an unpenalized model publishes the zero operator it has
    // rather than being refused for not having one.
    let joint_penalty_flat: Vec<f64> = joint_penalty.iter().copied().collect();
    if raw_training_design.ncols() != p_per_class {
        crate::bail_invalid_estim!(
            "multinomial REML: raw training design has {} columns but the fit reports \
             p_per_class = {p_per_class}",
            raw_training_design.ncols(),
        );
    }
    let training_rows = raw_training_design.nrows();
    let training_design_flat: Vec<f64> = raw_training_design.iter().copied().collect();
    let training_weights_flat: Vec<f64> = training_weights.iter().copied().collect();

    Ok(MultinomialSavedModel {
        formula: formula.to_string(),
        class_levels: class_levels.clone(),
        reference_class_index: class_levels.len() - 1,
        resolved_termspec: spec,
        coefficients_flat,
        p_per_class,
        n_active_classes: m,
        training_headers: data.headers.clone(),
        training_table_kind: config.training_table_kind.clone(),
        lambdas: lambdas_flat,
        lambdas_per_block,
        iterations: fit.inner_cycles,
        separation_evidence,
        penalized_neg_log_likelihood: -fit.log_likelihood + 0.5 * fit.stable_penalty_term,
        deviance,
        edf_per_class,
        edf_per_penalty,
        coefficient_covariance_flat,
        smoothing_correction_flat,
        coefficient_influence_flat,
        smooth_term_spans,
        training_design_flat,
        training_rows,
        training_class_index,
        training_weights: training_weights_flat,
        joint_penalty_flat,
        lambda_labels,
    })
}

/// Replay the saved termspec to build the predict-time dense design `X` on a
/// fresh dataset, realigning feature columns **by name** so the predict frame
/// need not reproduce the training column order or carry the response column.
/// Shared by every multinomial predict path (probabilities, SE bands, and the
/// posterior-predictive replicate draws).
fn build_multinomial_predict_design(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<Array2<f64>, EstimationError> {
    // The saved termspec stores feature columns as absolute indices into the
    // *training* table `[response, features...]`. Realign them onto this
    // dataset's columns by name, so prediction works on label-free new data
    // (the response column is never referenced by any term; issue #803).
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
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        crate::bail_invalid_estim!(
            "multinomial predict does not support non-zero smooth anchors: the saved \
             reference-coded softmax has no per-class affine offset channel"
        );
    }
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
    Ok(x_dense)
}

/// Replay the saved termspec to build the predict-time design on a fresh
/// dataset, then evaluate the POSTERIOR-MEAN class probabilities
/// `E[softmax(x'β) | data]`. The predict dataset must carry the same feature
/// columns the training data did, matched **by name** — it need not reproduce
/// the training column order, and in particular need not carry the response
/// column (prediction is for label-free new data).
///
/// The posterior mean is computed as a ratio of normalising constants rather
/// than by integrating `softmax` over the Laplace Gaussian; see
/// [`crate::multinomial_predictive`] for why the latter is not an approximation
/// of this estimand (#2612). For the plug-in `softmax(x'β̂)` every other softmax
/// implementation reports, ask [`predict_multinomial_formula_plugin`] by name.
pub fn predict_multinomial_formula(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<Array2<f64>, EstimationError> {
    model.validate()?;
    let x_dense = build_multinomial_predict_design(model, data)?;
    model.predict_probabilities(x_dense.view())
}

/// Plug-in class probabilities `softmax(x'β̂)` at the posterior MODE, for a
/// saved multinomial model on fresh data.
///
/// [`predict_multinomial_formula`] returns a different estimand: the
/// posterior-mean probability `E[softmax(η)]`. Both are legitimate and they are
/// not interchangeable, but the difference between them is much smaller than it
/// used to appear, and the reason is worth stating here because this function is
/// where a reader compares the two.
///
/// `softmax` is concave along the winning coordinate, so averaging it over
/// posterior width pulls the answer toward the centre of the simplex. The
/// posterior of a logit is also right-skewed toward larger `|η|`, which pulls
/// the other way. A correct posterior mean carries BOTH; the Gaussian
/// integration this path used to perform carried only the first, which is why
/// the published probability read as under-confident at unchanged argmax
/// (#2612). Measured on a quasi-separated fixture against an MCMC posterior, the
/// Gaussian-integrated quantity is off by up to `2.1e-1` in probability while
/// the ratio estimator [`predict_multinomial_formula`] now uses is off by
/// `4.4e-3` — and the plug-in below sits much closer to the posterior mean than
/// the Gaussian did.
///
/// `nnet::multinom`, `scikit-learn`, `statsmodels` and every other softmax
/// reference report the plug-in quantity, so a held-out log-loss comparison
/// against any of them is a comparison of two estimands unless this function is
/// the one supplying gam's side.
///
/// This is deliberately NOT a fallback: the posterior-mean path refuses rather
/// than degrading to a plug-in when it cannot certify its own accuracy, and that
/// refusal stands. A caller who wants the mode's own probability has to ask for
/// it here, by name.
pub fn predict_multinomial_formula_plugin(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<Array2<f64>, EstimationError> {
    model.validate()?;
    let x_dense = build_multinomial_predict_design(model, data)?;
    let coefficients = model.coefficients_active()?;
    let eta = x_dense.dot(&coefficients);
    let n_rows = eta.nrows();
    let n_classes = model.n_active_classes + 1;
    let mut probabilities = Array2::<f64>::zeros((n_rows, n_classes));
    for (row, mut destination) in eta.rows().into_iter().zip(probabilities.rows_mut()) {
        let active: Vec<f64> = row.iter().copied().collect();
        let row_probabilities = softmax_with_reference(&active)?;
        for (class, probability) in row_probabilities.iter().enumerate() {
            destination[class] = *probability;
        }
    }
    Ok(probabilities)
}

/// Draw `n_draws` posterior-predictive replicate class-label assignments for a
/// saved multinomial model on fresh data (#1101). Rebuilds the predict design
/// exactly as [`predict_multinomial_formula`], then samples each row's class
/// from `Categorical(E[softmax(η) | data])` (see
/// [`MultinomialSavedModel::sample_replicate_classes`]). Returns an
/// `(n_draws, N)` matrix of class INDICES `0..K` aligned to `model.class_levels`,
/// deterministic in `seed`.
pub fn posterior_predict_multinomial_formula(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
    n_draws: usize,
    seed: u64,
) -> Result<Array2<u32>, EstimationError> {
    if n_draws == 0 {
        crate::bail_invalid_estim!("multinomial posterior_predict: n_draws must be >= 1");
    }
    model.validate()?;
    let x_dense = build_multinomial_predict_design(model, data)?;
    model.sample_replicate_classes(x_dense.view(), n_draws, seed)
}

/// Predict posterior-mean class probabilities and integrated marginal
/// standard deviations for a saved multinomial model on fresh data, under the
/// best covariance definition the model can support (see
/// [`MultinomialSavedModel::predict_probabilities_with_se`]).
pub fn predict_multinomial_formula_with_se(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<(Array2<f64>, Array2<f64>), EstimationError> {
    model.validate()?;
    let x_dense = build_multinomial_predict_design(model, data)?;
    model.predict_probabilities_with_se(x_dense.view())
}

#[derive(Debug, Clone)]
pub struct MultinomialPredictionIntervals {
    pub mean: Array2<f64>,
    pub standard_error: Array2<f64>,
    pub mean_lower: Array2<f64>,
    pub mean_upper: Array2<f64>,
    pub level: f64,
    /// Exactly which posterior covariance definition the spread — and therefore
    /// the band — was built from. The multinomial's counterpart of
    /// `PredictUncertaintyResult::covariance_source`; see
    /// [`MultinomialSavedModel::predict_probabilities_with_se_in_mode`].
    pub covariance_source: InferenceCovarianceMode,
}

/// Build a central posterior interval around the integrated logistic-normal
/// posterior mean class probability. Both centre and spread come from the same
/// deterministic posterior integral; no plug-in/delta quantity enters the
/// centre.
///
/// # The band is built on the log-odds scale, not on the probability scale
///
/// The endpoints are `expit(logit(m) ± z·sd/(m(1−m)))`, which is the
/// `gam_predict::MeanIntervalMethod::TransformEta` construction this library
/// already prefers for every nonlinear link: build the symmetric interval where
/// the posterior is closest to Gaussian, then carry it through a MONOTONE map.
///
/// A symmetric `m ± z·sd` band on the probability scale, clamped into `[0, 1]`,
/// is wrong in two ways that both bite exactly where a class probability lives.
/// It is symmetric about `m` when the posterior of a bounded quantity is not —
/// the skew grows without bound as `m` approaches an endpoint — and the clamp
/// silently DELETES the part of the interval that fell outside, so a nominally
/// 95% band can carry materially less than 95% of the posterior while still
/// reporting `level = 0.95`. Transforming a log-odds interval has neither
/// property: `expit` is a bijection onto `(0, 1)`, so no mass is ever clipped
/// and the asymmetry is produced by the map rather than approximated away.
///
/// At `m = ½` the two constructions agree to first order (`d logit/dp = 4` and
/// the transform is locally affine), so this is not a re-scaling of well-posed
/// bands — it is a repair of the ones near the boundary.
pub fn predict_multinomial_formula_with_intervals(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
    level: f64,
) -> Result<MultinomialPredictionIntervals, EstimationError> {
    let source = if model.smoothing_correction_flat.is_some() {
        InferenceCovarianceMode::SmoothingCorrected
    } else {
        InferenceCovarianceMode::Conditional
    };
    predict_multinomial_formula_with_intervals_in_mode(model, data, level, source)
}

/// The same band under an EXPLICITLY named covariance definition; see
/// [`predict_multinomial_formula_with_se_in_mode`].
pub fn predict_multinomial_formula_with_intervals_in_mode(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
    level: f64,
    covariance_source: InferenceCovarianceMode,
) -> Result<MultinomialPredictionIntervals, EstimationError> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        crate::bail_invalid_estim!(
            "multinomial prediction interval level must be finite and in (0, 1), got {level}"
        );
    }
    model.validate()?;
    let x_dense = build_multinomial_predict_design(model, data)?;
    let (mean, standard_error) =
        model.predict_probabilities_with_se_in_mode(x_dense.view(), covariance_source)?;
    let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)
        .map_err(EstimationError::InvalidInput)?;
    let mut mean_lower = mean.clone();
    let mut mean_upper = mean.clone();
    for ((row, class), &se) in standard_error.indexed_iter() {
        let (lower, upper) = log_odds_interval(mean[[row, class]], se, z);
        mean_lower[[row, class]] = lower;
        mean_upper[[row, class]] = upper;
    }
    Ok(MultinomialPredictionIntervals {
        mean,
        standard_error,
        mean_lower,
        mean_upper,
        level,
        covariance_source,
    })
}

/// `expit(logit(m) ± z·sd/(m(1−m)))`, the monotone-transform central interval
/// for a probability with posterior mean `m` and posterior standard deviation
/// `sd`.
///
/// The delta-method log-odds spread is `sd / (m(1−m))`; `m(1−m)` is the
/// Jacobian `dp/dlogit p` at `m`. The degenerate ends are handled where they
/// arise rather than by a clamp afterwards: at `m ∈ {0, 1}` the log-odds scale
/// does not exist, and a mean that has reached a vertex of the simplex with
/// spread `sd` has no interval this transform can express, so the symmetric
/// probability-scale band is the honest answer THERE and only there. `sd = 0`
/// is a point interval on either scale.
fn log_odds_interval(mean: f64, standard_error: f64, z: f64) -> (f64, f64) {
    if !(mean.is_finite() && standard_error.is_finite()) || standard_error <= 0.0 {
        return (mean, mean);
    }
    let jacobian = mean * (1.0 - mean);
    if !(mean > 0.0 && mean < 1.0) || jacobian <= f64::MIN_POSITIVE {
        return (
            (mean - z * standard_error).clamp(0.0, 1.0),
            (mean + z * standard_error).clamp(0.0, 1.0),
        );
    }
    let center = (mean / (1.0 - mean)).ln();
    let half_width = z * standard_error / jacobian;
    let expit = |value: f64| -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let e = value.exp();
            e / (1.0 + e)
        }
    };
    (expit(center - half_width), expit(center + half_width))
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
        // #2344: K per-class lambdas (reference class included).
        let lambdas = Array1::<f64>::from_elem(k, 0.5);
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
                resume_from: None,
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
    fn exhausted_fixed_lambda_budget_is_typed_error_not_fit() {
        let (design, y, penalty, lambdas) = toy();
        let error = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 0,
            tol: 1.0e-9,
            resume_from: None,
        })
        .expect_err("a zero-budget Newton solve must not mint a multinomial fit");
        assert!(matches!(
            error,
            EstimationError::FixedLambdaNewtonDidNotConverge {
                objective_value,
                checkpoint,
                ..
            } if objective_value.is_finite()
                && checkpoint.stage() == FixedLambdaSolverStage::MultinomialNewton
                && checkpoint.completed_iterations() == 0
        ));
    }

    #[test]
    fn fixed_lambda_checkpoint_resume_matches_uninterrupted_solve() {
        let (design, y, penalty, lambdas) = toy();
        let interrupted = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 1,
            tol: 1.0e-9,
            resume_from: None,
        })
        .expect_err("one Newton step must leave this coupled fit uncertified");
        let checkpoint = match interrupted {
            EstimationError::FixedLambdaNewtonDidNotConverge { checkpoint, .. } => checkpoint,
            other => panic!("unexpected interruption error: {other}"),
        };
        assert_eq!(
            checkpoint.stage(),
            FixedLambdaSolverStage::MultinomialNewton
        );
        assert_eq!(checkpoint.completed_iterations(), 1);

        let resumed = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 49,
            tol: 1.0e-9,
            resume_from: Some(&checkpoint),
        })
        .expect("resumed multinomial solve must converge");
        let uninterrupted = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
            resume_from: None,
        })
        .expect("uninterrupted multinomial solve must converge");

        assert_eq!(resumed.iterations, uninterrupted.iterations);
        assert_eq!(
            resumed.coefficients_active,
            uninterrupted.coefficients_active
        );
        assert_eq!(
            resumed.penalized_neg_log_likelihood,
            uninterrupted.penalized_neg_log_likelihood,
        );
        assert_eq!(
            resumed.coefficient_covariance,
            uninterrupted.coefficient_covariance,
        );
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
            resume_from: None,
        })
        .expect_err("wrong active-block shape must error");
        assert!(format!("{err}").contains("fisher_w_override shape"));
    }

    #[test]
    fn formula_outer_route_uses_exact_curvature_for_medium_d() {
        // The 2-smooth reference formula fit (K = 3, double-penalty terms)
        // carries 2 terms x 2 penalties = 4 components, and the equivariant
        // carrier gives each component one coordinate PER CLASS: D = 4 x 3 = 12.
        // It needs exact curvature to avoid over-smoothed lambda caps
        // (#715 arm (a)).
        assert!(
            multinomial_formula_use_outer_hessian(8),
            "D=8 loaded multinomial fits need exact curvature to avoid over-smoothed lambda caps"
        );
        assert!(
            multinomial_formula_use_outer_hessian(12),
            "D=12 (2 double-penalty smooth terms, K=3) stays on exact curvature"
        );
    }

    #[test]
    fn formula_outer_route_uses_exact_curvature_for_the_penguin_fixture() {
        // Four k=10 penguin smooths (K = 3) are 8 double-penalty components and
        // therefore D = 8 x 3 = 24 outer coordinates. They must reach the exact
        // ARC route so the #1082 cost-stall halt is available on the
        // near-separable lambda-to-zero ridge.
        assert!(
            multinomial_formula_use_outer_hessian(24),
            "the four-smooth penguin fixture needs exact ARC curvature for the #1082 stall halt"
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

        // The floor's endpoints are now DERIVED from a target prior strength in
        // pseudo-observations against the maximal per-observation softmax Fisher
        // information I₁ = ¼ (base = τ·I₁, sparse = τ_max·I₁). Pin them to the
        // previously fixture-calibrated values so the near-separable quality arms
        // (penguins, vgam softmax) — whose smallest class has n_c ≥ 50 — are
        // byte-for-byte unaffected: the derivation REDUCES TO the old constants
        // at the calibration point.
        let base = MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS * MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS;
        let sparse = MULTINOMIAL_FORMULA_SPARSE_PRIOR_PSEUDO_OBS_MAX
            * MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS;
        assert!(
            (base - 2.0e-4).abs() < 1e-18,
            "derived base floor must equal the calibrated 2e-4"
        );
        assert!(
            (sparse - 1.0e-3).abs() < 1e-18,
            "derived sparse floor must equal the calibrated 1e-3"
        );

        // Well-supported (n_c >= n_ref=50) sits exactly at the base floor.
        assert!((floor_for_min_count(50) - base).abs() < 1e-18);
        assert!((floor_for_min_count(200) - base).abs() < 1e-18);
        // Very sparse (n_c <= n_ref·base/sparse = 10) clamps to the strong floor.
        assert!((floor_for_min_count(10) - sparse).abs() < 1e-18);
        assert!((floor_for_min_count(5) - sparse).abs() < 1e-18);
        // No cliff at the old hard threshold: 49 vs 50 differ by < 5% (the old
        // step jumped 5x). Floor is monotone non-increasing in support.
        let f49 = floor_for_min_count(49);
        let f50 = floor_for_min_count(50);
        assert!(
            f49 >= f50 && f49 <= f50 * 1.05,
            "floor must be continuous across c0, got {f49} vs {f50}"
        );
        let f25 = floor_for_min_count(25);
        assert!(
            f25 > f50 && f25 < floor_for_min_count(10),
            "mid-support floor must interpolate strictly between the two endpoints"
        );

        // FIRST-PRINCIPLES SCALING: in the interpolating regime the floor equals
        // exactly τ·I₁·(n_ref/n_c) — the effective-pseudo-observation prior held
        // to a fixed fraction of the per-class data information n_c·I₁. Halving
        // the effective sample size doubles the floor (until the cap), and the
        // absolute value matches the closed-form n_c-scaled prior.
        for &n_c in &[12usize, 16, 20, 30, 40] {
            let expected = base * (MULTINOMIAL_FORMULA_SPARSE_REFERENCE_SUPPORT / n_c as f64);
            assert!(
                (floor_for_min_count(n_c) - expected).abs() < 1e-15,
                "floor at n_c={n_c} must be τ·I₁·n_ref/n_c = {expected}, got {}",
                floor_for_min_count(n_c)
            );
        }
        // Inverse scaling with effective sample size: n_c -> n_c/2 doubles the
        // floor inside the unclamped band (20 and 40 are both interior; 40 < 50
        // so it is scaled, 20 > 10 so it is not capped).
        assert!(
            (floor_for_min_count(20) - 2.0 * floor_for_min_count(40)).abs() < 1e-15,
            "floor must scale like 1/n_c (effective Fisher information) in the interior band"
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
    fn fixed_lambda_multinomial_firth_keeps_complete_separation_finite() {
        // #1854: complete softmax separation used to be a HARD diagnostic
        // (`MultinomialSeparationDetected`). It now automatically engages the
        // Firth/Jeffreys proper prior (`½ log|I(β)|`, magic-by-default) so the fit
        // stays finite instead of running away — the same guarantee the formula
        // REML path already provided. The class regions are cleanly separated by
        // `x`, so the unbiased MLE is at infinity; the Firth-penalized fit must
        // still converge to a finite mode and recover the region structure.
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
        // #2344: K per-class lambdas (reference class included); K = 3 here.
        let lambdas = Array1::<f64>::zeros(3);
        let out = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 80,
            tol: 1.0e-12,
            resume_from: None,
        })
        .expect("Firth/Jeffreys prior keeps the separated multinomial fit finite (#1854)");
        // Every coefficient is finite — the whole point of the Firth prior on the
        // separated (unpenalized) logit directions.
        for &b in out.coefficients_active.iter() {
            assert!(
                b.is_finite(),
                "Firth-penalized coefficients must be finite, got {b}"
            );
        }
        // Fitted probabilities remain a valid simplex per row.
        for row in 0..n {
            let mut mass = 0.0_f64;
            for c in 0..3 {
                let p = out.fitted_probabilities[[row, c]];
                assert!(
                    p.is_finite() && (0.0..=1.0 + 1e-9).contains(&p),
                    "row {row} class {c} probability {p} out of [0,1]"
                );
                mass += p;
            }
            assert!(
                (mass - 1.0).abs() < 1e-6,
                "row {row} probabilities must sum to 1, got {mass}"
            );
        }
        // The finite fit still recovers the separated structure: on a clearly
        // interior representative of each region the predicted class is correct.
        let predict = |x: f64| -> usize {
            let mut eta = [0.0_f64; 3];
            for a in 0..2 {
                eta[a] = out.coefficients_active[[0, a]] + out.coefficients_active[[1, a]] * x;
            }
            let mut best = 0usize;
            for c in 1..3 {
                if eta[c] > eta[best] {
                    best = c;
                }
            }
            best
        };
        assert_eq!(predict(-2.5), 0, "deep-left region should predict class 0");
        assert_eq!(predict(2.5), 1, "deep-right region should predict class 1");
        assert_eq!(predict(0.0), 2, "central region should predict class 2");
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
    fn fisher_certificate_distinguishes_finite_quasi_separation_2612() {
        fn fixture(rows: usize, separated: bool) -> (MultinomialFamily, Vec<ParameterBlockState>) {
            let design = Arc::new(Array2::<f64>::from_shape_fn((rows, 2), |(row, column)| {
                if column == 0 {
                    1.0
                } else {
                    let side = if row % 2 == 0 { -1.0 } else { 1.0 };
                    side * (1.0 + row as f64 / rows as f64)
                }
            }));
            let mut response = Array2::<f64>::zeros((rows, 3));
            for row in 0..rows {
                response[[row, row % 3]] = 1.0;
            }
            let family = MultinomialFamily::new(
                response,
                Array1::ones(rows),
                3,
                Arc::clone(&design),
                Arc::new(vec![PenaltyMatrix::Dense(Array2::eye(2))]),
            )
            .expect("finite three-class fixture");
            let slopes = if separated { [30.0, -30.0] } else { [0.0, 0.0] };
            let states = slopes
                .into_iter()
                .map(|slope| {
                    let beta = Array1::from_vec(vec![0.0, slope]);
                    let eta = design.dot(&beta);
                    ParameterBlockState { beta, eta }
                })
                .collect();
            (family, states)
        }

        let (separated_family, separated_states) = fixture(60, true);
        assert!(
            multinomial_formula_separation_evidence(&separated_states).is_none(),
            "finite logits alone deliberately carry no separation evidence"
        );
        let separated_specs = separated_family.build_block_specs();
        let separated_span = Array2::<f64>::eye(4);
        let evidence = multinomial_formula_penalized_separation_evidence(
            &separated_family,
            &separated_specs,
            &separated_states,
            separated_span.view(),
            &[],
            0,
            None,
        )
        .expect("exact Fisher certification")
        .expect("finite quasi-separation must arm the Firth refit");
        let evidence = evidence.evidence;
        assert!(
            evidence.contains("under-identified")
                && evidence.contains("lambda_min")
                && evidence.contains("Jeffreys gate weight"),
            "the certificate must report its authoritative spectrum, got {evidence}"
        );

        let (identified_family, identified_states) = fixture(600, false);
        let identified_specs = identified_family.build_block_specs();
        let identified_span = Array2::<f64>::eye(4);
        assert!(
            multinomial_formula_penalized_separation_evidence(
                &identified_family,
                &identified_specs,
                &identified_states,
                identified_span.view(),
                &[],
                0,
                None,
            )
            .expect("exact Fisher certification")
            .is_none(),
            "a finite, well-identified Fisher geometry must keep Firth disarmed"
        );
    }

    /// #2612: the certificate must judge the curvature the FIT HAS, not the
    /// curvature the likelihood alone supplies.
    ///
    /// The absolute conditioning gate fires when the worst-determined direction
    /// holds less than one observation-equivalent (`λ_min < 1`). A penalized
    /// direction routinely sits there — that is *why* it is penalized — so a
    /// certificate reading the bare Fisher information `H` calls every penalized
    /// smooth "separated". The discriminator: hold the data, the mode and the
    /// span fixed and supply ONLY the penalty the fit actually selected. If the
    /// verdict is unchanged the certificate is ignoring `S_λ`; if it flips to
    /// disarmed, `λ` was reaching that direction all along.
    ///
    /// This is a unit statement about the certificate, independent of any fit or
    /// optimizer, so it holds even if every end-to-end fixture changes shape.
    #[test]
    fn the_separation_certificate_counts_the_penalty_the_fit_selected_2612() {
        let rows = 400usize;
        // Column 1 is a deliberately weak direction: `s` scales its contribution
        // to `X'WX` by `s²`, so `λ_min(H)` lands well under one
        // observation-equivalent while the intercept stays strongly determined.
        let weak_scale = 2.0e-2_f64;
        let design = Arc::new(Array2::<f64>::from_shape_fn((rows, 2), |(row, column)| {
            if column == 0 {
                1.0
            } else {
                weak_scale * ((row as f64 / rows as f64) - 0.5)
            }
        }));
        let mut response = Array2::<f64>::zeros((rows, 3));
        for row in 0..rows {
            response[[row, row % 3]] = 1.0;
        }
        let family = MultinomialFamily::new(
            response,
            Array1::ones(rows),
            3,
            Arc::clone(&design),
            Arc::new(vec![PenaltyMatrix::Dense(Array2::eye(2))]),
        )
        .expect("weak-direction three-class fixture");
        let specs = family.build_block_specs();
        let states: Vec<ParameterBlockState> = (0..2)
            .map(|_| {
                let beta = Array1::from_vec(vec![0.0, 0.0]);
                let eta = design.dot(&beta);
                ParameterBlockState { beta, eta }
            })
            .collect();
        let span = Array2::<f64>::eye(4);

        // Without the penalty the weak direction reads as separation.
        let unpenalized = multinomial_formula_penalized_separation_evidence(
            &family,
            &specs,
            &states,
            span.view(),
            &[],
            0,
            None,
        )
        .expect("certification")
        .expect(
            "the fixture must be one the LIKELIHOOD alone cannot identify, or this test proves \
             nothing",
        );
        let unpenalized = unpenalized.evidence;
        assert!(
            unpenalized.contains("lambda_min"),
            "the certificate must report the spectrum it decided on, got {unpenalized}"
        );

        // The same mode, with a penalty on the weak direction worth many
        // observation-equivalents. `S_λ` is what the fit's own inner solve adds
        // to `H`, so the direction is determined and nothing is separated.
        let mut penalty = Array2::<f64>::zeros((4, 4));
        penalty[[1, 1]] = 1.0;
        penalty[[3, 3]] = 1.0;
        let spec = gam_problem::JointPenaltySpec {
            matrix: penalty,
            initial_log_lambda: 0.0,
            nullspace_dim: 2,
            label: None,
            group: None,
        };
        let selected = Array1::from_vec(vec![(1.0e3_f64).ln()]);
        let penalized = multinomial_formula_penalized_separation_evidence(
            &family,
            &specs,
            &states,
            span.view(),
            std::slice::from_ref(&spec),
            1,
            Some(&selected),
        )
        .expect("certification");
        assert!(
            penalized.is_none(),
            "a direction the selected smoothing parameter determines is not separation \
             evidence; the certificate still armed with S_lambda supplied: {penalized:?}"
        );
    }

    #[test]
    fn scaled_fisher_override_changes_first_step() {
        // Curvature scaled by 4× shrinks the first Newton step relative to the
        // analytic fit, so a single-iteration fit must differ.
        let (design, y, penalty, lambdas) = toy();
        let n = design.nrows();
        let m = y.ncols() - 1;
        // #2344: toy() now carries K per-class lambdas for the multinomial
        // ENTRY; the direct Centered-metric ENGINE calls below read
        // M = lambdas.len(), so hand them the M-length shared-lambda vector.
        let engine_lambdas = Array1::<f64>::from_elem(m, lambdas[0]);
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
        let likelihood =
            MultinomialLogitLikelihood::with_classes(y.ncols()).expect("test class count is valid");
        let scaled = fit_penalized_vector_glm(
            PenalizedVectorGlmInputs {
                design: design.view(),
                y: y.view(),
                offset: None,
                penalty: penalty.view(),
                lambdas: engine_lambdas.view(),
                fisher_w_override: Some(over.view()),
                max_iter: 1,
                tol: 1.0e-9,
                class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Centered,
                resume_from: None,
            },
            &likelihood,
            "multinomial scaled-curvature first-step test",
        )
        .expect("scaled-curvature engine step must be finite");
        let analytic = fit_penalized_vector_glm(
            PenalizedVectorGlmInputs {
                design: design.view(),
                y: y.view(),
                offset: None,
                penalty: penalty.view(),
                lambdas: engine_lambdas.view(),
                fisher_w_override: None,
                max_iter: 1,
                tol: 1.0e-9,
                class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Centered,
                resume_from: None,
            },
            &likelihood,
            "multinomial analytic-curvature first-step test",
        )
        .expect("analytic-curvature engine step must be finite");
        let checkpoint_coefficients = |solve| match solve {
            VectorGlmSolve::Converged(fit) => fit.coefficients,
            VectorGlmSolve::Stalled(stall) => stall.coefficients,
        };
        let scaled = checkpoint_coefficients(scaled);
        let analytic = checkpoint_coefficients(analytic);
        let differs = scaled
            .iter()
            .zip(analytic.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-6);
        assert!(differs, "scaled curvature must change the first step");
    }
}

#[cfg(test)]
mod separation_firth_tests {
    //! Regression for #1854: on (quasi-)perfect separation the fixed-λ direct
    //! multinomial solve must engage the Firth/Jeffreys penalty and return a
    //! finite, converged, well-behaved fit instead of hard-erroring with
    //! `MultinomialSeparationDetected`.
    use super::*;

    /// A perfectly linearly separable 3-class problem with an UNPENALIZED design
    /// (`S = 0`), so no smoothing `λ` can bound the saturated logits — only the
    /// Firth prior `½ log det I(β)` keeps the estimate finite. The unbiased MLE
    /// here runs `|η| → ∞` (separation), which is exactly the #1854 trigger.
    fn separated_three_class() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 21;
        let p = 2; // intercept + ordering covariate x
        let k = 3;
        let mut design = Array2::<f64>::zeros((n, p));
        let mut y = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            design[[i, 0]] = 1.0;
            design[[i, 1]] = x;
            let cls = if x < -1.0 {
                0
            } else if x < 1.0 {
                1
            } else {
                2
            };
            y[[i, cls]] = 1.0;
        }
        // S = 0: no smoothing direction can bound the separated logits.
        let penalty = Array2::<f64>::zeros((p, p));
        // #2344: K per-class lambdas (reference class included).
        let lambdas = Array1::<f64>::from_elem(k, 1.0);
        (design, y, penalty, lambdas)
    }

    #[test]
    fn separation_engages_firth_finite_converged_fit() {
        let (design, y, penalty, lambdas) = separated_three_class();
        let out = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 300,
            tol: 1e-10,
            resume_from: None,
        })
        .expect("separated multinomial must engage Firth and return a fit, not error");

        assert!(
            out.coefficients_active.iter().all(|v| v.is_finite()),
            "all coefficients must be finite under the Firth prior"
        );
        assert!(out.deviance.is_finite(), "deviance must be finite");

        // The runaway MLE would drive fitted probabilities to the {0,1} boundary;
        // the Firth prior keeps them strictly interior.
        for v in out.fitted_probabilities.iter() {
            assert!(
                *v > 0.0 && *v < 1.0,
                "Firth fit must stay interior, got p={v}"
            );
        }

        // Perfect separation ⇒ every training row classified to its true class.
        let n = design.nrows();
        let k = y.ncols();
        for i in 0..n {
            let mut best = 0usize;
            for c in 1..k {
                if out.fitted_probabilities[[i, c]] > out.fitted_probabilities[[i, best]] {
                    best = c;
                }
            }
            let truth = (0..k)
                .find(|&c| y[[i, c]] == 1.0)
                .expect("one-hot truth class");
            assert_eq!(best, truth, "row {i} misclassified under separation");
        }
    }

    #[test]
    fn separation_firth_returns_finite_wellshaped_covariance() {
        // Distinct angle: the Firth separation path must also expose a finite,
        // correctly-shaped (P·M × P·M) Laplace coefficient covariance — the
        // downstream SE machinery consumes it. A runaway MLE would have a
        // singular (non-invertible) information here.
        let (design, y, penalty, lambdas) = separated_three_class();
        let p = design.ncols();
        let k = y.ncols();
        let m = k - 1;
        let out = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 300,
            tol: 1e-10,
            resume_from: None,
        })
        .expect("separated multinomial must return a Firth fit");

        assert_eq!(
            out.coefficient_covariance.dim(),
            (p * m, p * m),
            "covariance must be P·M square"
        );
        assert!(
            out.coefficient_covariance.iter().all(|v| v.is_finite()),
            "Firth covariance entries must be finite"
        );
        // A genuine Laplace covariance is PSD ⇒ non-negative diagonal.
        for i in 0..(p * m) {
            assert!(
                out.coefficient_covariance[[i, i]] >= -1e-9,
                "covariance diagonal must be non-negative, got {}",
                out.coefficient_covariance[[i, i]]
            );
        }
    }

    #[test]
    fn firth_solver_rejects_a_truncated_iterate() {
        // #2066 / SPEC 20 (convergence honesty): the Firth Newton loop may only
        // construct a fit after certifying stationarity. Before the fix a
        // truncated solve returned coefficients and covariance behind a false
        // `converged` flag; now budget exhaustion is a typed error carrying the
        // iteration count and objective evidence.
        //
        // Angle: run the SAME separated problem that converges under a full
        // budget (`separation_engages_firth_finite_converged_fit`) but starve the
        // iteration budget so it provably cannot reach the interior Firth mode.
        // The honest outcome is a typed error, not an inspectable fit.
        let (design, y, penalty, lambdas) = separated_three_class();

        let truncated = fit_penalized_multinomial_firth_fallback(
            design.view(),
            y.view(),
            penalty.view(),
            lambdas.view(),
            None,
            1, // one Newton iteration — far from the separated mode
            1e-12,
            None,
        )
        .expect_err("a one-iteration Firth solve must not mint a fit");
        let checkpoint = match truncated {
            EstimationError::FixedLambdaNewtonDidNotConverge {
                objective_value,
                stationarity,
                checkpoint,
                ..
            } => {
                assert!(objective_value.is_finite());
                assert_eq!(stationarity.kind, FixedLambdaResidualKind::NewtonDecrement);
                assert_eq!(checkpoint.stage(), FixedLambdaSolverStage::MultinomialFirth);
                assert_eq!(checkpoint.completed_iterations(), 1);
                checkpoint
            }
            other => panic!("unexpected Firth interruption error: {other}"),
        };

        let resumed = fit_penalized_multinomial(MultinomialFitInputs {
            design: design.view(),
            y_one_hot: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 299,
            tol: 1e-10,
            resume_from: Some(&checkpoint),
        })
        .expect("Firth checkpoint must resume to the certified mode");

        // Contrast: with a full budget the same problem does reach stationarity
        // and returns the convergence-only result type.
        let uninterrupted = fit_penalized_multinomial_firth_fallback(
            design.view(),
            y.view(),
            penalty.view(),
            lambdas.view(),
            None,
            300,
            1e-10,
            None,
        )
        .expect("Firth fallback must converge under a full budget");
        assert_eq!(resumed.iterations, uninterrupted.iterations);
        assert_eq!(
            resumed.coefficients_active,
            uninterrupted.coefficients_active
        );
        assert_eq!(
            resumed.penalized_neg_log_likelihood,
            uninterrupted.penalized_neg_log_likelihood,
        );
        assert_eq!(
            resumed.coefficient_covariance,
            uninterrupted.coefficient_covariance,
        );
    }
}

#[cfg(test)]
mod reference_class_invariance_tests {
    //! Regression for #1587: a penalized multinomial-logit GAM fit must be
    //! invariant to which class is the (arbitrary) softmax reference/baseline.
    //!
    //! The production REML path (`fit_penalized_multinomial_formula`) reference-
    //! codes the `K` classes (the last sorted label is the baseline) and, with
    //! the legacy `Diagonal` penalty metric, penalizes only the `K−1`
    //! reference-anchored ALR contrasts `½ Σ_a λ_a β_aᵀ S β_a`. Relabeling the
    //! response so a *different* class sorts last penalizes a different frame of
    //! log-odds contrasts, so the predicted probabilities drift (~1e-2 absolute)
    //! even though they are mathematically independent of the reference choice.
    //!
    //! This test fits the SAME 3-class softmax sample under three cyclic
    //! relabelings — each making a different original class the baseline —
    //! realigns the predicted probability columns back to the original class
    //! identities, and asserts the cross-labeling drift is below `1e-3`
    //! (the defect is ~1e-2; refitting the same labeling twice agrees to
    //! ~1e-12). It is the Rust-level sibling of
    //! `tests/bug_hunt_multinomial_fit_depends_on_reference_class_test.py`.

    use super::*;
    use gam_data::load_dataset_projected;
    use std::fmt::Write as _;
    use std::fs;
    use tempfile::tempdir;

    /// Deterministic `splitmix64` → `[0,1)` uniform stream (no external RNG dep;
    /// the only requirement is a well-distributed, reproducible draw).
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            // 53-bit mantissa uniform in [0, 1).
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Draw a clean 3-class softmax regression sample (the issue's generator).
    /// Returns `(x, class)` with integer classes `0/1/2`.
    fn sample_classes(seed: u64, n: usize) -> (Vec<f64>, Vec<usize>) {
        let mut rng = SplitMix64(seed.wrapping_add(0x1234_5678));
        let mut x = Vec::with_capacity(n);
        let mut cls = Vec::with_capacity(n);
        for _ in 0..n {
            let xi = -2.0 + 4.0 * rng.unit();
            let eta = [0.5 + 0.8 * xi, -0.3 - 0.5 * xi, 0.0];
            let mut p = [eta[0].exp(), eta[1].exp(), eta[2].exp()];
            let s: f64 = p.iter().sum();
            for v in &mut p {
                *v /= s;
            }
            // Inverse-CDF draw into one of the 3 classes.
            let u = rng.unit();
            let c = if u < p[0] {
                0
            } else if u < p[0] + p[1] {
                1
            } else {
                2
            };
            x.push(xi);
            cls.push(c);
        }
        (x, cls)
    }

    /// Build an `EncodedDataset` with columns `x` (numeric) and `y`
    /// (categorical, from the given string labels) by round-tripping a CSV.
    fn dataset_xy(
        dir: &std::path::Path,
        tag: &str,
        x: &[f64],
        y: &[String],
    ) -> gam_data::EncodedDataset {
        let path = dir.join(format!("data_{tag}.csv"));
        let mut csv = String::from("x,y\n");
        for (xi, yi) in x.iter().zip(y.iter()) {
            writeln!(csv, "{xi},{yi}").unwrap();
        }
        fs::write(&path, csv).expect("write training csv");
        load_dataset_projected(&path, &["x".to_string(), "y".to_string()])
            .expect("load training dataset")
    }

    /// Fit `y ~ s(x)` under the relabeling `name_map` (original class `c` gets
    /// label `name_map[c]`), predict on `grid`, and return the predicted
    /// probabilities **realigned to the original class order** 0/1/2, shape
    /// `(grid.len(), 3)`.
    fn fit_predict_aligned(
        dir: &std::path::Path,
        tag: &str,
        x: &[f64],
        cls: &[usize],
        name_map: [&str; 3],
        grid: &[f64],
    ) -> Array2<f64> {
        let labels: Vec<String> = cls.iter().map(|&c| name_map[c].to_string()).collect();
        let train = dataset_xy(dir, tag, x, &labels);
        let config = FitConfig::default();
        let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-6,
            ..MultinomialFitRequest::new(&train, "y ~ s(x)", &config)
        })
        .expect("multinomial formula fit must succeed");

        // Predict on the grid. The categorical `y` column is not needed for
        // prediction, but the schema is simplest if we supply a dummy.
        let grid_y: Vec<String> = grid.iter().map(|_| name_map[0].to_string()).collect();
        let grid_ds = dataset_xy(dir, &format!("{tag}_grid"), grid, &grid_y);
        let probs = predict_multinomial_formula(&model, &grid_ds)
            .expect("multinomial predict must succeed");

        // `model.class_levels` is the sorted label order; the column for original
        // class `c` is at the rank of `name_map[c]` among the sorted labels.
        let mut sorted: Vec<&str> = name_map.to_vec();
        sorted.sort_unstable();
        let col_of_orig: Vec<usize> = (0..3)
            .map(|c| sorted.iter().position(|l| *l == name_map[c]).unwrap())
            .collect();
        // Sanity: the model's class_levels must match the sorted labels.
        assert_eq!(
            model.class_levels,
            sorted.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "class_levels must be the sorted label order"
        );
        let n = grid.len();
        let mut aligned = Array2::<f64>::zeros((n, 3));
        for r in 0..n {
            for c in 0..3 {
                aligned[[r, c]] = probs[[r, col_of_orig[c]]];
            }
        }
        aligned
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max)
    }

    /// #2579 diagnostic (zz_measure): print the λ that
    /// `multinomial_fit_is_invariant_to_reference_class_1587` actually selects
    /// in each of its three labelings.
    ///
    /// The `#1587` gate asserts that predicted probabilities are invariant to
    /// the reference class. On penguins every one of the sixteen selected λ is
    /// railed at `exp(EFFECTIVE_DF_CEILING) = 162754.79141900392` to the last
    /// bit, and all-λ-equal is invariant TRIVIALLY. If this fixture's λ are
    /// likewise all pinned to that same wall, the gate is currently satisfied by
    /// railing rather than by the `M⊗S_t` symmetry it was written to protect —
    /// which means any fix that frees λ (i.e. the `#2579` fix, teaching
    /// `effective_df_floor_rho_upper_bounds` the joint bundle) will trip it for
    /// a reason that is not a regression in the symmetry. Prints only; never
    /// asserts a bound.
    #[test]
    fn zz_measure_2579_reference_class_gate_lambdas() {
        let td = tempdir().expect("tempdir");
        let dir = td.path();
        let (x, cls) = sample_classes(0, 300);
        let wall = gam_custom_family::EFFECTIVE_DF_CEILING.exp();

        for (tag, name_map) in [
            ("abc", ["A", "B", "C"]),
            ("bca", ["B", "C", "A"]),
            ("cab", ["C", "A", "B"]),
        ] {
            let labels: Vec<String> = cls.iter().map(|&c| name_map[c].to_string()).collect();
            let train = dataset_xy(dir, tag, &x, &labels);
            let config = FitConfig::default();
            let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
                init_lambda: 1.0,
                max_iter: 60,
                tol: 1e-6,
                ..MultinomialFitRequest::new(&train, "y ~ s(x)", &config)
            })
            .expect("multinomial formula fit must succeed");

            // "Railed" is bit-equality with the wall, exactly as the
            // `EFFECTIVE_DF_CEILING` doc defines it.
            let railed = model.lambdas.iter().filter(|l| **l == wall).count();
            eprintln!(
                "#2579 gate-lambda probe [{tag}]: n_lambda={} railed_at_exp12={}/{} \
                 per_block={:?}",
                model.lambdas.len(),
                railed,
                model.lambdas.len(),
                model.lambdas_per_block
            );
            for (i, l) in model.lambdas.iter().enumerate() {
                eprintln!(
                    "#2579 gate-lambda probe [{tag}]:   lambda[{i}] = {l:.17e}  ln = {:.12}  \
                     railed = {}",
                    l.ln(),
                    *l == wall
                );
            }
        }
        eprintln!("#2579 gate-lambda probe: wall = exp(12) = {wall:.17e}");
    }

    // gam#1587: now that the reference-symmetric centered `M⊗S_t` joint penalty
    // is wired through the custom-family outer REML loop (per-eval
    // `JointPenaltyBundle` + outer penalty_coords/logdet/operator), the
    // production multinomial fit is invariant to the arbitrary reference class,
    // so this guard runs by default (the opt-in skip attribute it carried while
    // the fix was pending is also forbidden by the build.rs ban-scanner). It is
    // an end-to-end fit guard (a handful of full softmax `y ~ s(x)` fits) —
    // slower than a unit test but a true production-path regression.
    #[test]
    fn multinomial_fit_is_invariant_to_reference_class_1587() {
        let td = tempdir().expect("tempdir");
        let dir = td.path();
        // The reference-class drift is STRUCTURAL (it does not shrink with n, see
        // the issue table), so a modest n exposes it just as cleanly as n=900
        // while keeping this an affordable CI guard.
        let (x, cls) = sample_classes(0, 300);
        let grid: Vec<f64> = (0..7).map(|i| -1.5 + 3.0 * (i as f64) / 6.0).collect();

        // Three labelings that each make a DIFFERENT original class the baseline
        // (the class whose label sorts LAST is the reference K−1):
        //   ["A","B","C"] → ref = class 2
        //   ["B","C","A"] → ref = class 1
        //   ["C","A","B"] → ref = class 0
        let a = fit_predict_aligned(dir, "abc", &x, &cls, ["A", "B", "C"], &grid);
        let b = fit_predict_aligned(dir, "bca", &x, &cls, ["B", "C", "A"], &grid);
        let c = fit_predict_aligned(dir, "cab", &x, &cls, ["C", "A", "B"], &grid);

        // Refitting the SAME labeling twice must agree to ~machine precision —
        // this isolates optimizer noise from the structural reference drift.
        let a2 = fit_predict_aligned(dir, "abc2", &x, &cls, ["A", "B", "C"], &grid);
        let refit_noise = max_abs_diff(&a, &a2);
        assert!(
            refit_noise < 1e-6,
            "refitting the same labeling must be deterministic (got {refit_noise:.3e})"
        );

        let drift = max_abs_diff(&a, &b)
            .max(max_abs_diff(&a, &c))
            .max(max_abs_diff(&b, &c));
        assert!(
            drift < 1e-3,
            "predicted probabilities must be invariant to the reference class; \
             cross-labeling drift = {drift:.3e} (refit noise = {refit_noise:.3e})"
        );
    }

    /// The quality arm's stride-3 penguins TRAIN split, materialized through the
    /// production dataset loader. `None` when the checked-in CSV is unavailable.
    fn penguins_stride3_train(td: &tempfile::TempDir) -> Option<gam_data::EncodedDataset> {
        const PENGUINS_CSV: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../bench/datasets/penguins.csv"
        );
        let raw = match fs::read_to_string(PENGUINS_CSV) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("penguins CSV unavailable: {PENGUINS_CSV}: {err}");
                return None;
            }
        };
        let mut lines = raw.lines();
        let header: Vec<&str> = lines
            .next()
            .expect("penguins header")
            .trim()
            .split(',')
            .collect();
        let idx = |name: &str| header.iter().position(|c| *c == name).expect("column");
        let (i_species, i_bl, i_bd, i_fl, i_bm) = (
            idx("species"),
            idx("bill_length_mm"),
            idx("bill_depth_mm"),
            idx("flipper_length_mm"),
            idx("body_mass_g"),
        );
        let mut kept = 0usize;
        let mut csv =
            String::from("bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,species\n");
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            let num = |i: usize| f.get(i).and_then(|v| v.trim().parse::<f64>().ok());
            let (Some(bl), Some(bd), Some(fl), Some(bm)) =
                (num(i_bl), num(i_bd), num(i_fl), num(i_bm))
            else {
                continue;
            };
            let species = f[i_species].trim().trim_matches('"');
            if species.is_empty() || species == "NA" {
                continue;
            }
            if kept % 3 == 0 {
                kept += 1;
                continue; // held out by the quality test
            }
            kept += 1;
            writeln!(csv, "{bl},{bd},{fl},{bm},{species}").unwrap();
        }
        let path = td.path().join("penguins_train.csv");
        fs::write(&path, csv).expect("write penguins train csv");
        let cols: Vec<String> = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "species",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Some(load_dataset_projected(&path, &cols).expect("load penguins train"))
    }

    /// A deterministic two-covariate three-class dataset whose labels are DRAWN
    /// from a smooth softmax truth.
    ///
    /// Drawing rather than taking the argmax is load-bearing: an argmax label is
    /// a deterministic function of `x`, so the classes are exactly separated by
    /// their own decision boundaries and every fit on them is legitimately on
    /// the separation lane. Here every class keeps appreciable probability
    /// everywhere, so **no direction separates** and the unbiased criterion has
    /// an interior optimum. Same generator as
    /// `tests/multinomial_parametric_penalty_2612.rs`.
    fn softmax_drawn_two_covariate(
        dir: &std::path::Path,
        tag: &str,
        n: usize,
    ) -> gam_data::EncodedDataset {
        const CLASS_NAMES: [&str; 3] = ["a", "b", "c"];
        let mut lcg: u64 = 0x2612_2612_2612_2612;
        let mut next_unit = || -> f64 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut csv = String::from("x1,x2,y\n");
        for index in 0..n {
            let x1 = -2.0 + 4.0 * (((index as f64) * 0.618_033_988_749_894_8) % 1.0);
            let x2 = -2.0 + 4.0 * (((index as f64) * 0.414_213_562_373_095_1) % 1.0);
            let scores = [
                0.6 * x1 - 0.3 * x2,
                -0.4 * x1 + 0.5 * x2,
                0.2 * (x1 * x1 - x2 * x2) * 0.25,
            ];
            let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - shift).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut draw = next_unit() * total;
            let mut label = CLASS_NAMES.len() - 1;
            for (class, weight) in weights.iter().enumerate() {
                if draw < *weight {
                    label = class;
                    break;
                }
                draw -= weight;
            }
            writeln!(csv, "{x1},{x2},{}", CLASS_NAMES[label]).unwrap();
        }
        let path = dir.join(format!("softmax_drawn_{tag}.csv"));
        fs::write(&path, csv).expect("write softmax-drawn csv");
        let cols: Vec<String> = ["x1", "x2", "y"].iter().map(|s| s.to_string()).collect();
        load_dataset_projected(&path, &cols).expect("load softmax-drawn dataset")
    }

    /// #2612 diagnostic (zz_measure): **which matrix decides the arming, and
    /// what does each one say?**
    ///
    /// The conditional Firth/Jeffreys engagement fires when the reduced
    /// conditioning gate reports under-identification at the certified mode. The
    /// gate's absolute arm fires at `λ_min < 1` — one observation-equivalent of
    /// curvature — and until #2612 this call site handed it the bare Fisher
    /// information `H`. A penalized spline basis has high-frequency directions
    /// the data barely resolve BY CONSTRUCTION, so that reading calls an ordinary
    /// smooth "separated"; reading the penalized curvature `H + S_λ` asks the
    /// question of the objective that was actually optimized.
    ///
    /// This prints both spectra at the same certified mode, on the same span, on
    /// data drawn from a smooth softmax truth (nothing separates) and — when the
    /// checked-in CSV is present — on the penguins witness, which genuinely is
    /// quasi-separated. The two rows are the discriminator: the fix is only right
    /// if it disarms the first and leaves the second armed.
    ///
    /// Prints only; never asserts a bound.
    #[test]
    fn zz_measure_2612_arming_certificate_with_and_without_the_penalty() {
        let td = tempdir().expect("tempdir");
        let synthetic = softmax_drawn_two_covariate(td.path(), "arm2612", 600);
        let config = FitConfig::default();
        let mut cases: Vec<(&str, gam_data::EncodedDataset, &str)> = vec![(
            "synthetic softmax-drawn (nothing separates)",
            synthetic,
            "y ~ s(x1, k=6) + s(x2, k=6)",
        )];
        if let Some(train) = penguins_stride3_train(&td) {
            cases.push((
                "penguins stride-3 (the quasi-separated witness)",
                train,
                "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) \
                 + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
            ));
        } else {
            eprintln!("#2612 penguins arm SKIPPED: dataset unavailable");
        }

        for (label, data, formula) in &cases {
            let request = MultinomialFitRequest {
                init_lambda: 1.0,
                max_iter: 100,
                tol: 1e-8,
                ..MultinomialFitRequest::new(data, formula, &config)
            };
            let parts =
                penalized_multinomial_formula_parts(&request).expect("production formula parts");
            let started = std::time::Instant::now();
            let probe = crate::custom_family::fit_custom_family_with_rho_prior(
                &parts.family,
                &parts.blocks,
                &parts.options,
                gam_problem::RhoPrior::Flat,
            );
            let seconds = started.elapsed().as_secs_f64();
            let probe = match probe {
                Ok(fit) => fit,
                Err(err) => {
                    eprintln!(
                        "#2612 [{label}] unbiased probe FAILED after {seconds:.1}s: {}",
                        format!("{err}").chars().take(400).collect::<String>()
                    );
                    continue;
                }
            };
            let span = probe
                .geometry
                .as_ref()
                .expect("probe geometry")
                .coefficient_gauge
                .t_full
                .clone();
            let specs = parts
                .family
                .equivariant_class_penalty_specs()
                .expect("coupled joint penalty specs");
            let n_components = parts.penalties_arc.len();
            let selected = probe.artifacts.joint_log_lambdas.as_ref();
            // The historical reading: the likelihood alone, `S_λ` omitted.
            let likelihood_only = multinomial_formula_penalized_separation_evidence(
                &parts.family,
                &parts.blocks,
                &probe.block_states,
                span.view(),
                &[],
                0,
                None,
            );
            // The curvature the mode actually sits in.
            let penalized = multinomial_formula_penalized_separation_evidence(
                &parts.family,
                &parts.blocks,
                &probe.block_states,
                span.view(),
                &specs,
                n_components,
                selected,
            );
            let lambda_range = selected.map(|jll| {
                let lo = jll.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = jll.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (lo.exp(), hi.exp(), jll.len())
            });
            eprintln!(
                "#2612 [{label}] probe converged in {seconds:.1}s; selected lambda \
                 (min, max, count) = {lambda_range:?}\n    H alone      -> {:?}\n    H + S_lambda -> {:?}",
                likelihood_only
                    .as_ref()
                    .map(|e| e.as_ref().map(MultinomialSeparationCertificate::as_str)),
                penalized
                    .as_ref()
                    .map(|e| e.as_ref().map(MultinomialSeparationCertificate::as_str)),
            );
        }
    }
}
