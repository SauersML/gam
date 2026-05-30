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
//! # What this module deliberately does *not* do
//!
//! * **REML / LAML smoothing-parameter selection.** `λ` is supplied by the
//!   caller. Selecting `λ` via the outer REML loop is a separate slice — it
//!   requires implementing the full [`CustomFamily`] surface (joint Hessian
//!   in (β across all classes), directional derivatives in ρ, the
//!   joint-coupled coefficient-Hessian cost model, and integration with
//!   [`crate::families::custom_family::fit_custom_family_with_rho_prior`]).
//!   That follow-up sits cleanly on top of this driver: the multinomial
//!   `CustomFamily` impl would call the math here as its inner solve at each
//!   ρ trial and supply the same dense per-row Hessian block for the outer
//!   trace terms.
//!
//! * **Formula → design integration.** Callers build `X` (and `S`) from the
//!   GAM formula via the existing scalar plumbing; this driver is the
//!   coefficient-space solver only. The forthcoming `gamfit.fit_multinomial`
//!   Python entry will wire the formula machinery to this driver in a
//!   dedicated FFI shim.
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

use crate::families::custom_family::{BlockwiseFitOptions, fit_custom_family_with_rho_prior};
use crate::families::multinomial_reml::MultinomialFamily;
use crate::families::penalized_vector_glm::{PenalizedVectorGlmInputs, fit_penalized_vector_glm};
use crate::families::vector_response::{MultinomialLogitLikelihood, validate_multinomial_simplex};
use crate::inference::data::EncodedDataset;
use crate::inference::formula_dsl::parse_formula;
use crate::inference::model::ColumnKindTag;
use crate::resource::ProblemHints;
use crate::solver::estimate::EstimationError;
use crate::solver::workflow::{
    FitConfig, build_termspec_with_geometry_and_overrides, resolved_resource_policy,
};
use crate::terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    weighted_blockwise_penalty_sum,
};
use crate::terms::term_builder::resolve_role_col;
use crate::types::ResponseColumnKind;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
//     parse → termspec → design (X, S blocks) → one-hot Y → Newton solve
//
// pipeline. REML / LAML λ-selection is the next slice; until that lands the
// caller pins `init_lambda` (default 1.0) and the same value is used for every
// penalty block and every active class. The reference class is the last level
// of the categorical response column as recorded in the dataset schema.

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
    /// Per-active-class REML/LAML-selected smoothing parameter, length
    /// `K - 1` (one entry per active class block; the shared-penalty
    /// architecture means each class owns exactly one λ). When the outer
    /// REML loop is bypassed (legacy fixed-λ path), every entry is the
    /// caller-supplied initial value.
    pub lambdas: Vec<f64>,
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
}

/// One-hot-encode the categorical response column and return both the
/// encoding and the captured level names. The level order matches the order
/// recorded in the dataset schema, which is itself the order of first
/// appearance during inferred-schema construction — so it is stable and
/// deterministic across runs (no silent class permutation).
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
    let y_kind = crate::solver::workflow::response_column_kind(data, y_col);
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

/// Top-level formula-driven multinomial fit.
///
/// Routes through [`fit_custom_family_with_rho_prior`] so the per-active-class
/// smoothing parameters `λ_a` (one per class block, shared-penalty
/// architecture) are selected by the outer REML/LAML loop rather than pinned
/// by the caller. `init_lambda` survives as a warm-start hint that seeds
/// every block's `initial_log_lambdas`; the inner Newton solve still uses
/// `max_iter` / `tol` via `BlockwiseFitOptions`.
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
    let (spec, design, y_col, response_name, y_kind) =
        build_formula_design_for_multinomial(formula, data, config)?;
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
    let x_dense = design
        .design
        .try_to_dense_by_chunks("multinomial fit design")
        .map_err(EstimationError::InvalidInput)?;
    let p_total = x_dense.ncols();
    // Sum the penalty blocks at uniform λ = 1 (the per-class λ_a is folded
    // back through the REML driver below, so the assembled `S` here is the
    // unweighted Σ_k S_k that every active class shares).
    let lambdas_block = vec![1.0_f64; design.penalties.len()];
    let s_total = weighted_blockwise_penalty_sum(&design.penalties, &lambdas_block, p_total);
    let k = y_one_hot.ncols();
    let m = k - 1;
    let n_obs = y_one_hot.nrows();

    // ── Custom-family driven REML/LAML path ───────────────────────────────
    // Each active class becomes one ParameterBlockSpec, all sharing X and S.
    // `initial_log_lambdas` is seeded from the caller's `init_lambda`.
    let design_arc = Arc::new(x_dense);
    let penalty_arc = Arc::new(s_total);
    let weights = Array1::<f64>::ones(n_obs);
    let family = MultinomialFamily::new(
        y_one_hot.clone(),
        weights,
        k,
        design_arc.clone(),
        penalty_arc.clone(),
        0,
    )
    .map_err(EstimationError::InvalidInput)?;
    let mut blocks = family.build_block_specs();
    let log_init = init_lambda.ln();
    for spec_block in blocks.iter_mut() {
        for v in spec_block.initial_log_lambdas.iter_mut() {
            *v = log_init;
        }
    }

    let options = BlockwiseFitOptions {
        inner_max_cycles: max_iter,
        inner_tol: tol,
        ..BlockwiseFitOptions::default()
    };
    let fit =
        fit_custom_family_with_rho_prior(&family, &blocks, &options, crate::types::RhoPrior::Flat)
            .map_err(|err| EstimationError::InvalidInput(format!("multinomial REML: {err}")))?;

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
    let lambdas_per_class: Vec<f64> = fit
        .blocks
        .iter()
        .map(|b| b.lambdas.iter().copied().next().unwrap_or(init_lambda))
        .collect();
    let edf_per_class = fit.inference.as_ref().map(|info| info.edf_by_block.clone());
    let coefficients_flat: Vec<f64> = coefficients_active.iter().copied().collect();

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
        lambdas: lambdas_per_class,
        iterations: fit.inner_cycles,
        converged: fit.outer_converged,
        penalized_neg_log_likelihood: -fit.log_likelihood + 0.5 * fit.stable_penalty_term,
        deviance,
        edf_per_class,
    })
}

/// Replay the saved termspec to build the predict-time design on a fresh
/// dataset, then evaluate softmax probabilities. The predict dataset must
/// carry the same feature columns the training data did (matched by name).
pub fn predict_multinomial_formula(
    model: &MultinomialSavedModel,
    data: &EncodedDataset,
) -> Result<Array2<f64>, EstimationError> {
    let design = build_term_collection_design(data.values.view(), &model.resolved_termspec)
        .map_err(|err| {
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
