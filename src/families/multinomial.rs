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
//! Damped Newton with backtracking on the penalized negative log-likelihood:
//! at each iteration the assembled penalized Hessian `H + I_{K-1} ⊗ (λ_a S)`
//! is factored via faer's symmetric-PD-with-fallback path, the full Newton
//! step `δ = −H^{-1} ∇F` is computed, and accepted with step halving if the
//! objective fails to decrease (up to a small backtracking budget). The
//! convergence test is the relative coefficient step norm
//! `‖δ‖ / (1 + ‖β‖) ≤ tol`, matching the existing pyffi reference path.

use crate::solver::estimate::EstimationError;
use crate::families::vector_response::{MultinomialLogitLikelihood, VectorLikelihood};
use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, factorize_symmetricwith_fallback};
use crate::inference::data::EncodedDataset;
use crate::inference::formula_dsl::parse_formula;
use crate::inference::model::ColumnKindTag;
use crate::pirls::dense_block_xtwx;
use crate::resource::ProblemHints;
use crate::solver::workflow::{
    FitConfig, build_termspec_with_geometry, resolved_resource_policy,
};
use crate::terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    weighted_blockwise_penalty_sum,
};
use crate::terms::term_builder::resolve_role_col;
use crate::types::ResponseColumnKind;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

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
    /// One-hot response `Y ∈ ℝ^{N×K}` (row sums ≈ 1 for hard classification;
    /// non-integer rows are permitted for label-smoothed targets).
    pub y_one_hot: ArrayView2<'a, f64>,
    /// Shared smoothing penalty `S ∈ ℝ^{P×P}` (symmetric, PSD).
    pub penalty: ArrayView2<'a, f64>,
    /// Per-active-class smoothing parameter `λ_a` (length `K - 1`).
    pub lambdas: ArrayView1<'a, f64>,
    /// Optional per-row weights (length `N`); `None` ⇒ uniform 1.0.
    pub row_weights: Option<ArrayView1<'a, f64>>,
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
        max_iter,
        tol,
    } = inputs;

    // ────────────────────────────── shape checks ──────────────────────────
    let n_obs = design.nrows();
    let p = design.ncols();
    if n_obs == 0 || p == 0 {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: design must be nonempty (got {n_obs}x{p})"
        );
    }
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
    if penalty.dim() != (p, p) {
        crate::bail_invalid_estim!(
            "fit_penalized_multinomial: penalty shape {:?} ≠ (P, P) = ({p}, {p})",
            penalty.dim()
        );
    }
    for (i, &v) in lambdas.iter().enumerate() {
        if !(v.is_finite() && v >= 0.0) {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: lambdas[{i}] must be finite and ≥ 0 (got {v})"
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
    for ((i, j), &v) in y_one_hot.indexed_iter() {
        if !v.is_finite() {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: y[{i},{j}] must be finite (got {v})"
            );
        }
    }
    for ((i, j), &v) in design.indexed_iter() {
        if !v.is_finite() {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: design[{i},{j}] must be finite (got {v})"
            );
        }
    }

    // ────────────────────────── likelihood construction ───────────────────
    let mut likelihood = MultinomialLogitLikelihood::with_classes(k)?;
    if let Some(w) = row_weights.as_ref() {
        likelihood = likelihood.with_row_weights(w.to_owned())?;
    }

    // ────────────────────────── Newton iteration ──────────────────────────
    // β stored as (P, M) column-major-per-class; flat index uses output-major
    // ordering `flat[a · P + i] = β[i, a]` to align with `dense_block_xtwx`.
    let mut beta = Array2::<f64>::zeros((p, m));
    let mut eta = Array2::<f64>::zeros((n_obs, m));
    // Working response z = η − W^{−1} (∇_η · −1) in the diagonal IRLS form;
    // for the dense per-row Fisher block the canonical update solves
    //     (X^T W X + Sλ) β_new = X^T W z,   with z = η + W^{-1} (y − p)
    // but the dense-block form factors more cleanly through the Newton
    // residual directly:
    //     δ = − (X^T W X + Sλ)^{-1} · (X^T (p − y) + Sλ · β)
    // which is what we compute below. Both forms produce the identical β_new.

    let mut iterations = 0usize;
    let mut converged = false;
    let mut last_objective = f64::INFINITY;

    let beta_flat_dim = p * m;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // η = X · β (per class).
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

        // Assemble per-row dense Fisher block H_{n,a,b} = w_n p_a (δ_ab − p_b)
        // and gradient residual r_{n,a} = w_n (p_a − y_a) (= −∂ log L / ∂η_a).
        // The Hessian block is multiplied by w_n already (the likelihood
        // factors row weights into `hess_block`), so we pass `None` for the
        // row_weights argument of `dense_block_xtwx` to avoid double-counting.
        let fisher_blocks = likelihood.hess_block(eta.view(), y_one_hot);
        let grad_eta_logl = likelihood.grad_eta(eta.view(), y_one_hot);
        // residual = p − y = −(y − p) = −grad_eta(log L). The pyffi reference
        // and the dense_block_xtwy contract both want the multinomial
        // *residual* in (N, M) form, scaled by row weights — `grad_eta`
        // already returns `w_n (y − p)`, so the residual is `−grad_eta`.
        let residual_active = grad_eta_logl.mapv(|v| -v);

        let mut hessian = dense_block_xtwx(design, fisher_blocks.view(), None)?;
        if hessian.nrows() != beta_flat_dim || hessian.ncols() != beta_flat_dim {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: assembled Hessian shape {:?} ≠ ({beta_flat_dim}, {beta_flat_dim})",
                hessian.dim()
            );
        }

        // Add block-replicated penalty: H[a·P:(a+1)·P, a·P:(a+1)·P] += λ_a · S
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

        // Gradient of penalized negative log L: ∇F_a = X^T r_{·,a} + λ_a · S β_a
        // dense_block_xtwy(design, fisher_blocks, residual_active, None) would
        // produce X^T W (residual / W) — but residual_active is *already*
        // weighted by w_n through grad_eta. The cleanest assembly is the
        // direct path: X^T residual_active + Sλ β, no W involvement.
        let mut grad_flat = Array1::<f64>::zeros(beta_flat_dim);
        for a in 0..m {
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n_obs {
                    acc += design[[row, i]] * residual_active[[row, a]];
                }
                grad_flat[a * p + i] = acc;
            }
        }
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

        // δ = − H^{-1} · grad
        let factor = factorize_symmetricwith_fallback(
            FaerArrayView::new(&hessian).as_ref(),
            Side::Lower,
        )
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "fit_penalized_multinomial: Hessian factorization failed at iter {iter}: {err}"
            ))
        })?;
        let mut rhs = Array2::<f64>::zeros((beta_flat_dim, 1));
        for i in 0..beta_flat_dim {
            rhs[[i, 0]] = -grad_flat[i];
        }
        {
            let rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view);
        }
        let mut delta = Array1::<f64>::zeros(beta_flat_dim);
        for i in 0..beta_flat_dim {
            delta[i] = rhs[[i, 0]];
        }
        if delta.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_estim!(
                "fit_penalized_multinomial: Newton step is non-finite at iter {iter}"
            );
        }

        // Damped acceptance: try the full step first, halve up to 8 times if
        // the penalized negative log-likelihood fails to decrease. The first
        // iteration always accepts (the initial β = 0 gives the trivial
        // uniform-class predictor and we want to start moving).
        let proposed_beta = |alpha: f64| -> Array2<f64> {
            let mut out = beta.clone();
            for a in 0..m {
                for i in 0..p {
                    out[[i, a]] += alpha * delta[a * p + i];
                }
            }
            out
        };
        let evaluate_objective = |beta_trial: &Array2<f64>| -> f64 {
            // η_trial = X · β_trial
            let mut eta_trial = Array2::<f64>::zeros((n_obs, m));
            for a in 0..m {
                let beta_col = beta_trial.column(a);
                for row in 0..n_obs {
                    let mut v = 0.0_f64;
                    for i in 0..p {
                        v += design[[row, i]] * beta_col[i];
                    }
                    eta_trial[[row, a]] = v;
                }
            }
            let ll = likelihood.log_lik(eta_trial.view(), y_one_hot);
            let mut pen = 0.0_f64;
            for a in 0..m {
                let la = lambdas[a];
                if la == 0.0 {
                    continue;
                }
                let beta_col = beta_trial.column(a);
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
            -ll + pen
        };
        if iter == 0 {
            last_objective = evaluate_objective(&beta);
            if !last_objective.is_finite() {
                crate::bail_invalid_estim!(
                    "fit_penalized_multinomial: non-finite objective at β = 0"
                );
            }
        }
        let mut alpha = 1.0_f64;
        let mut accepted_beta = proposed_beta(alpha);
        let mut new_objective = evaluate_objective(&accepted_beta);
        let mut backtrack = 0usize;
        while (!new_objective.is_finite() || new_objective > last_objective + 1.0e-12) && backtrack < 8 {
            alpha *= 0.5;
            accepted_beta = proposed_beta(alpha);
            new_objective = evaluate_objective(&accepted_beta);
            backtrack += 1;
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
        if step_norm <= tol * (1.0 + beta_norm) {
            converged = true;
            break;
        }
    }

    // ──────────────────────────── post-process ────────────────────────────
    // η = X · β̂ (final).
    for a in 0..m {
        let beta_col = beta.column(a);
        for row in 0..n_obs {
            let mut v = 0.0_f64;
            for i in 0..p {
                v += design[[row, i]] * beta_col[i];
            }
            eta[[row, a]] = v;
        }
    }
    let fitted_probabilities = likelihood.probabilities(eta.view());
    let log_lik = likelihood.log_lik(eta.view(), y_one_hot);
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

    Ok(MultinomialFitOutputs {
        coefficients_active: beta,
        fitted_probabilities,
        iterations,
        converged,
        penalized_neg_log_likelihood: -log_lik + pen,
        deviance: -2.0 * log_lik,
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
    /// Penalty smoothing parameter used (uniform across all blocks and all
    /// active classes in Slice A; REML follows in the next slice).
    pub lambda: f64,
    /// Newton iterations executed; recorded for the summary report.
    pub iterations: usize,
    /// `true` if the inner Newton solver hit the relative-step tolerance.
    pub converged: bool,
    /// Penalized negative log-likelihood at the returned `β̂`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)`.
    pub deviance: f64,
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
    let y_col = resolve_role_col(&col_map, &parsed.response, "response").map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: {err}"))
    })?;
    let y_kind = crate::solver::workflow::response_column_kind(data, y_col);
    let policy = resolved_resource_policy(config, data, ProblemHints::default());
    let mut inference_notes: Vec<String> = Vec::new();
    let spec = build_termspec_with_geometry(
        &parsed.terms,
        data,
        &col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
    )
    .map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build termspec: {err}"))
    })?;
    let design = build_term_collection_design(data.values.view(), &spec).map_err(|err| {
        EstimationError::InvalidInput(format!("multinomial fit: build design: {err}"))
    })?;
    Ok((spec, design, y_col, parsed.response, y_kind))
}

/// Top-level formula-driven multinomial fit. Slice A: uniform `init_lambda`
/// across every penalty block, single inner Newton solve, no REML/LAML
/// outer loop. REML follows in the next commit.
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
    // back into the solver call below, so the assembled `S` here is the
    // unweighted Σ_k S_k).
    let lambdas_block = vec![1.0_f64; design.penalties.len()];
    let s_total = weighted_blockwise_penalty_sum(&design.penalties, &lambdas_block, p_total);
    let k = y_one_hot.ncols();
    let m = k - 1;
    let lambdas_class = Array1::from_elem(m, init_lambda);
    let outputs = fit_penalized_multinomial(MultinomialFitInputs {
        design: x_dense.view(),
        y_one_hot: y_one_hot.view(),
        penalty: s_total.view(),
        lambdas: lambdas_class.view(),
        row_weights: None,
        max_iter,
        tol,
    })?;
    let coefficients_flat: Vec<f64> = outputs.coefficients_active.iter().copied().collect();
    Ok(MultinomialSavedModel {
        formula: formula.to_string(),
        class_levels: class_levels.clone(),
        reference_class_index: class_levels.len() - 1,
        resolved_termspec: spec,
        coefficients_flat,
        p_per_class: outputs.coefficients_active.nrows(),
        n_active_classes: outputs.coefficients_active.ncols(),
        training_headers: data.headers.clone(),
        lambda: init_lambda,
        iterations: outputs.iterations,
        converged: outputs.converged,
        penalized_neg_log_likelihood: outputs.penalized_neg_log_likelihood,
        deviance: outputs.deviance,
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
