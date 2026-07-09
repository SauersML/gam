//! Penalized multi-output binomial-logit fitter at fixed λ.
//!
//! This is the row-diagonal sibling of [`crate::multinomial`]: the
//! same shared design `X ∈ ℝ^{N×P}` and shared penalty `S ∈ ℝ^{P×P}` are
//! reused across `K` independent binomial-logit response columns. Per-column
//! smoothing parameters `λ_a` (length `K`) scale `S` independently for each
//! response. Because the Fisher information has no cross-column coupling
//! (`H_{n,a,b} = δ_{ab} · w_n · μ_{n,a} (1 − μ_{n,a})`), the joint penalized
//! Hessian is block-diagonal in the `K` `P × P` per-response systems; the
//! shared [`crate::penalized_vector_glm`] engine factors that
//! block-diagonal Hessian in a single coupled damped-Newton loop, which is
//! mathematically identical to `K` independent per-column solves.
//!
//! # Fit problem
//!
//! Minimise the penalized negative log-likelihood
//!
//! ```text
//!   F(β) = − Σ_n Σ_a w_n [ y_{n,a} log μ_{n,a} + (1 − y_{n,a}) log(1 − μ_{n,a}) ]
//!           + ½ Σ_a λ_a · β_aᵀ S β_a
//! ```
//!
//! with `μ_{n,a} = σ(η_{n,a})`, `η_{n,a} = (X β_a)_n`. The per-column Newton
//! step solves
//!
//! ```text
//!   (Xᵀ diag(w_n μ_{n,a}(1 − μ_{n,a})) X + λ_a S) δ_a = − [Xᵀ diag(w_n)(μ_{·,a} − y_{·,a}) + λ_a S β_a]
//! ```
//!
//! followed by a backtracking line search on `F` (full step first, halve up
//! to 8 times) so monotone descent is enforced even when the quadratic
//! model overshoots near saturation. This is precisely the shared
//! [`crate::penalized_vector_glm`] scaffold; this module supplies
//! only the row-diagonal binomial Fisher block, residual, and log-likelihood
//! via [`BinomialMultiLikelihood`].
//!
//! # Relation to the multi-class softmax driver
//!
//! [`crate::multinomial::fit_penalized_multinomial`] handles the
//! coupled softmax Fisher block `H_{n,a,b} = w_n μ_{n,a} (δ_{ab} − μ_{n,b})`
//! and is the right entry when the user wants a single normalized
//! probability vector per row. This driver is the right entry when the
//! user has `K` independent binary marginals sharing a smooth basis (e.g.
//! multi-label classification, multi-trait penalised logistic regression
//! on a Duchon latent design). Both families are thin Fisher-block adapters
//! over the same `penalized_vector_glm` engine: the only difference is that
//! the softmax block is dense across outputs while these binomial columns are
//! row-diagonal.
//!
//! The function-boundary contract mirrors `fit_penalized_multinomial` so
//! the two are interchangeable at the FFI layer: same input arity, same
//! convergence semantics, same `(N, K)` fitted-probability output.

use crate::penalized_vector_glm::{PenalizedVectorGlmInputs, fit_penalized_vector_glm};
use crate::vector_response::VectorLikelihood;
use crate::model_types::EstimationError;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

/// Inputs for [`fit_penalized_binomial_multi`].
#[derive(Debug, Clone)]
pub struct BinomialMultiFitInputs<'a> {
    /// Design matrix `X ∈ ℝ^{N×P}` (one row per observation, shared across
    /// all response columns).
    pub design: ArrayView2<'a, f64>,
    /// Multi-column binomial response `Y ∈ ℝ^{N×K}`. Each column is treated
    /// as an independent binomial-logit response, so every entry must be a
    /// binomial proportion in `[0, 1]` (hard `{0, 1}` Bernoulli labels and soft
    /// proportions / probabilities alike). Entries outside `[0, 1]` are
    /// rejected because the per-entry log-likelihood is then unbounded in `η`.
    pub y: ArrayView2<'a, f64>,
    /// Shared smoothing penalty `S ∈ ℝ^{P×P}` (symmetric, PSD).
    pub penalty: ArrayView2<'a, f64>,
    /// Per-response smoothing parameter `λ_a` (length `K`).
    pub lambdas: ArrayView1<'a, f64>,
    /// Optional per-row weights (length `N`); `None` ⇒ uniform 1.0.
    pub row_weights: Option<ArrayView1<'a, f64>>,
    /// Optional per-row Fisher-block override, shape `(N, K, K)`. The `K`
    /// binomial-logit columns are fit independently, so only the per-column
    /// diagonal `[n, a, a]` is consumed as the curvature `w_n μ_a(1 − μ_a)`;
    /// off-diagonals must be zero (enforced at the FFI boundary) since a
    /// non-zero cross term cannot be represented by the separable per-column
    /// solve. The gradient/residual path stays analytic — this is a
    /// curvature-only override (issue #349). Diagonal entries must be finite
    /// and non-negative.
    pub fisher_w_override: Option<ArrayView3<'a, f64>>,
    /// Maximum Newton iterations per response column; recommend 50.
    pub max_iter: usize,
    /// Relative-step convergence tolerance; recommend 1e-7.
    pub tol: f64,
}

/// Outputs of [`fit_penalized_binomial_multi`].
#[derive(Debug, Clone)]
pub struct BinomialMultiFitOutputs {
    /// Coefficient matrix, shape `(P, K)` (column `a` is `β_a`).
    pub coefficients: Array2<f64>,
    /// Fitted probabilities `μ_{n,a} = σ((X β_a)_n)`, shape `(N, K)`.
    pub fitted_probabilities: Array2<f64>,
    /// Number of joint Newton iterations executed (including the final step
    /// that satisfied the tolerance). The `K` columns share the design and
    /// are fitted by a single coupled damped-Newton loop over the
    /// block-diagonal penalized Hessian, so there is one iteration count for
    /// the whole solve.
    pub iterations: usize,
    /// Always `true` for values returned by [`fit_penalized_binomial_multi`]:
    /// non-convergence is surfaced as the typed
    /// [`EstimationError::FixedLambdaNewtonDidNotConverge`] rather than an
    /// `Ok` with a flag (SPEC: a fit only ever comes from a converged
    /// optimization).
    pub converged: bool,
    /// Penalized negative log-likelihood at the returned `β̂`:
    /// `−log L(β̂) + ½ Σ_a λ_a · β̂_aᵀ S β̂_a`.
    pub penalized_neg_log_likelihood: f64,
    /// Unpenalized deviance `−2 log L(β̂)` for diagnostic reporting.
    pub deviance: f64,
}

/// Numerically stable logistic CDF used by the Newton driver. Mirrors the
/// inline helper that previously lived in `crates/gam-pyffi/src/lib.rs`.
#[inline]
fn sigmoid_stable(eta: f64) -> f64 {
    if eta >= 0.0 {
        let e = (-eta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// Row-diagonal multi-output binomial-logit likelihood adapter for the shared
/// [`crate::penalized_vector_glm`] engine.
///
/// The `K` response columns are mutually independent binomial-logit marginals
/// sharing the design `X`, so the per-row Fisher block is **diagonal across
/// outputs**: `H_{n,a,b} = δ_{ab} · w_n · μ_{n,a} (1 − μ_{n,a})`. The engine
/// works in `η = X β` space with `μ_{n,a} = σ(η_{n,a})`; this adapter supplies
/// the log-likelihood, the residual gradient `w_n (y_a − μ_a)`, and that
/// row-diagonal block.
struct BinomialMultiLikelihood {
    /// Optional per-row weights (length N), or `None` for uniform 1.0.
    row_weights: Option<Array1<f64>>,
}

impl BinomialMultiLikelihood {
    #[inline]
    fn row_weight(&self, n: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[n])
    }
}

impl VectorLikelihood for BinomialMultiLikelihood {
    /// `Σ_n Σ_a w_n [ y_{n,a} log μ_{n,a} + (1 − y_{n,a}) log(1 − μ_{n,a}) ]`,
    /// evaluated in log-space via `log μ = −softplus(−η)`,
    /// `log(1 − μ) = −softplus(η)` — exact and finite for every η, with no
    /// probability clamp. The former `μ.clamp(1e-12, 1−1e-12)` made this value
    /// FLAT beyond |η| ≈ 27.6 while [`Self::grad_eta`]/[`Self::hess_diag`]
    /// kept reporting the unclamped derivatives, so the line search scored a
    /// surface the Newton direction was not the derivative of: on a
    /// misclassified saturated row the gradient pushed full-strength while
    /// the objective registered no improvement. The softplus form keeps the
    /// true slope (≈ |η| per unit) at any saturation, so value, gradient, and
    /// curvature are exact surfaces of ONE function.
    fn log_lik(&self, eta: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> f64 {
        let (n, k) = eta.dim();
        let mut acc = 0.0_f64;
        for row in 0..n {
            let w = self.row_weight(row);
            for a in 0..k {
                let e = eta[[row, a]];
                let yv = y[[row, a]];
                acc -= w
                    * (yv * gam_linalg::utils::stable_softplus(-e)
                        + (1.0 - yv) * gam_linalg::utils::stable_softplus(e));
            }
        }
        acc
    }

    /// `∂ log L / ∂η_{n,a} = w_n (y_{n,a} − μ_{n,a})`.
    fn grad_eta(&self, eta: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
        let (n, k) = eta.dim();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let w = self.row_weight(row);
            for a in 0..k {
                let mu = sigmoid_stable(eta[[row, a]]);
                out[[row, a]] = w * (y[[row, a]] - mu);
            }
        }
        out
    }

    /// Per-output diagonal curvature `w_n μ_{n,a} (1 − μ_{n,a})`. The Fisher
    /// information of independent Bernoulli outputs is `y`-independent; `y` is
    /// read only to assert the target shape matches `eta`, as in the sibling
    /// [`VectorLikelihood`] implementations.
    fn hess_diag(&self, eta: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array2<f64> {
        assert_eq!(eta.dim(), y.dim(), "y must match eta shape (N, K)");
        let (n, k) = eta.dim();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let w = self.row_weight(row);
            for a in 0..k {
                let mu = sigmoid_stable(eta[[row, a]]);
                out[[row, a]] = w * mu * (1.0 - mu);
            }
        }
        out
    }

    /// Row-diagonal Fisher block `H_{n,a,b} = δ_{ab} · w_n μ_{n,a}(1 − μ_{n,a})`.
    /// The independent columns have no cross-output coupling, so the off-diagonal
    /// entries are identically zero; lifting [`Self::hess_diag`] onto the per-row
    /// diagonal (the [`VectorLikelihood`] default) is exact here.
    fn hess_block(&self, eta: ArrayView2<'_, f64>, y: ArrayView2<'_, f64>) -> Array3<f64> {
        let diag = self.hess_diag(eta, y);
        let (n, k) = diag.dim();
        let mut out = Array3::<f64>::zeros((n, k, k));
        for row in 0..n {
            for a in 0..k {
                out[[row, a, a]] = diag[[row, a]];
            }
        }
        out
    }
}

/// Fit `K` independent penalized binomial-logit GLMs sharing the design `X`
/// and penalty `S`. See the module docs for the optimization problem.
pub fn fit_penalized_binomial_multi(
    inputs: BinomialMultiFitInputs<'_>,
) -> Result<BinomialMultiFitOutputs, EstimationError> {
    let BinomialMultiFitInputs {
        design,
        y,
        penalty,
        lambdas,
        row_weights,
        fisher_w_override,
        max_iter,
        tol,
    } = inputs;

    // ──────────────────────── family-specific validation ───────────────────
    // The engine re-validates the shared geometry (nonempty design, penalty
    // shape, λ finiteness/non-negativity, override `(N, M, M)` shape, finite
    // design), but the binomial family owns three preconditions the generic
    // scaffold cannot know: the response must be a `[0, 1]` proportion, the
    // optional row weights must be finite and non-negative, and the optional
    // curvature override must be **row-diagonal** (independent columns carry no
    // cross-output coupling, so a non-zero off-diagonal cannot be represented).
    let n_obs = design.nrows();
    let (y_rows, k) = y.dim();
    if y_rows != n_obs {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: y rows {y_rows} ≠ design rows {n_obs}"
        );
    }
    if k == 0 {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: y must have at least one column (got K=0)"
        );
    }
    if lambdas.len() != k {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: lambdas length {} ≠ K = {k}",
            lambdas.len()
        );
    }
    if let Some(fw) = fisher_w_override.as_ref() {
        if fw.dim() != (n_obs, k, k) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: fisher_w_override shape {:?} ≠ (N, K, K) = ({n_obs}, {k}, {k})",
                fw.dim()
            );
        }
        // Independent binomial columns have a strictly row-diagonal Fisher
        // block; a non-zero cross term `[n, a, b]` (a ≠ b) cannot be the
        // curvature of a separable per-column objective, so reject it rather
        // than silently couple the columns through the shared dense solve.
        for ((n_idx, a, b), &v) in fw.indexed_iter() {
            if a != b && v != 0.0 {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: fisher_w_override[{n_idx},{a},{b}] must be zero \
                     (independent columns have a row-diagonal Fisher block); got {v}"
                );
            }
        }
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n_obs {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: row_weights length {} ≠ N = {n_obs}",
                w.len()
            );
        }
        for (i, &v) in w.iter().enumerate() {
            if !(v.is_finite() && v >= 0.0) {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: row_weights[{i}] must be finite and ≥ 0 (got {v})"
                );
            }
        }
    }
    for ((i, j), &v) in y.indexed_iter() {
        // The per-entry objective y log μ + (1 − y) log(1 − μ) is the binomial
        // (Bernoulli / proportion) log-likelihood only when 0 ≤ y ≤ 1. Outside
        // that range it is unbounded above in η (e.g. y = 2 gives
        // 2η − log(1 + e^η) → ∞), so a finite-but-invalid entry would make the
        // stated likelihood not a binomial likelihood at all. Reject it here.
        if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: y[{i},{j}] must be a binomial proportion in [0,1] (got {v})"
            );
        }
    }

    // ─────────────────── shared penalized vector-GLM solve ─────────────────
    let likelihood = BinomialMultiLikelihood {
        row_weights: row_weights.map(|w| w.to_owned()),
    };
    let fit = fit_penalized_vector_glm(
        PenalizedVectorGlmInputs {
            design,
            y,
            penalty,
            lambdas,
            fisher_w_override,
            max_iter,
            tol,
            // Independent-binomial columns ARE genuinely independent outputs, so
            // the per-output Diagonal penalty is correct here (the #1587 Centered
            // metric is softmax-specific — there is no shared reference class).
            class_penalty_metric: crate::penalized_vector_glm::ClassPenaltyMetric::Diagonal,
        },
        &likelihood,
        "fit_penalized_binomial_multi",
    )?;

    if !fit.converged {
        // SPEC: a fit object must only ever come from a converged optimization.
        // Exhausting `max_iter` is a typed error carrying its evidence, never
        // an Ok(outputs) with `converged: false`.
        return Err(EstimationError::FixedLambdaNewtonDidNotConverge {
            context: "fit_penalized_binomial_multi (fixed-λ vector-GLM damped Newton)"
                .to_string(),
            iterations: fit.iterations,
            penalized_neg_log_likelihood: -fit.log_likelihood + fit.penalty_term,
        });
    }

    // η → μ = σ(η) is the binomial inverse link applied column-wise.
    let fitted = fit.eta.mapv(sigmoid_stable);

    Ok(BinomialMultiFitOutputs {
        coefficients: fit.coefficients,
        fitted_probabilities: fitted,
        iterations: fit.iterations,
        converged: fit.converged,
        penalized_neg_log_likelihood: -fit.log_likelihood + fit.penalty_term,
        deviance: -2.0 * fit.log_likelihood,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn toy_inputs() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
        let n = 12;
        let p = 2;
        let k = 2;
        let design =
            Array2::<f64>::from_shape_fn(
                (n, p),
                |(i, j)| {
                    if j == 0 { 1.0 } else { ((i + 1) as f64).sin() }
                },
            );
        let y =
            Array2::<f64>::from_shape_fn((n, k), |(i, a)| if (i + a) % 2 == 0 { 1.0 } else { 0.0 });
        let penalty = Array2::<f64>::eye(p);
        let lambdas = Array1::<f64>::from_elem(k, 0.5);
        (design, y, penalty, lambdas)
    }

    #[test]
    fn fisher_override_none_reproduces_analytic_bit_for_bit() {
        // Issue #349: a None override must give exactly the analytic result.
        let (design, y, penalty, lambdas) = toy_inputs();
        let base = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        // Explicit None again — identical result.
        let again = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        for (a, b) in base.coefficients.iter().zip(again.coefficients.iter()) {
            assert_eq!(a, b, "None override must be deterministic");
        }
    }

    #[test]
    fn out_of_range_response_is_rejected() {
        // Issue #452: a finite but invalid entry (y = 2) makes the per-entry
        // binomial log-likelihood unbounded in η, so it must be rejected rather
        // than silently fit. The same guard covers negative entries.
        let (design, y, penalty, lambdas) = toy_inputs();
        let mut bad = y.clone();
        bad[[0, 0]] = 2.0;
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: bad.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("out-of-range response must error");
        assert!(format!("{err}").contains("binomial proportion in [0,1]"));

        let mut neg = y.clone();
        neg[[1, 1]] = -0.5;
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: neg.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("negative response must error");
        assert!(format!("{err}").contains("binomial proportion in [0,1]"));
    }

    #[test]
    fn fisher_override_shape_mismatch_is_rejected() {
        let (design, y, penalty, lambdas) = toy_inputs();
        let n = design.nrows();
        let k = y.ncols();
        let bad = Array3::<f64>::zeros((n, k + 1, k + 1));
        let err = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: Some(bad.view()),
            max_iter: 50,
            tol: 1.0e-9,
        })
        .expect_err("mismatched override shape must error");
        assert!(format!("{err}").contains("fisher_w_override shape"));
    }

    #[test]
    fn fisher_override_replaces_curvature_diagonal() {
        // A scaled curvature override changes the Newton step from β = 0:
        // with curvature scaled by α the first step is 1/α of the analytic
        // step (gradient unchanged), so the fitted β must differ from analytic.
        let (design, y, penalty, lambdas) = toy_inputs();
        let n = design.nrows();
        let k = y.ncols();
        // Analytic diagonal at β = 0 is μ(1−μ) = 0.25 for every column.
        let mut over = Array3::<f64>::zeros((n, k, k));
        for row in 0..n {
            for a in 0..k {
                over[[row, a, a]] = 0.25 * 4.0; // 4× the analytic curvature
            }
        }
        let scaled = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: Some(over.view()),
            max_iter: 1,
            tol: 1.0e-9,
        })
        .expect("override fit must succeed");
        let analytic = fit_penalized_binomial_multi(BinomialMultiFitInputs {
            design: design.view(),
            y: y.view(),
            penalty: penalty.view(),
            lambdas: lambdas.view(),
            row_weights: None,
            fisher_w_override: None,
            max_iter: 1,
            tol: 1.0e-9,
        })
        .expect("analytic fit must succeed");
        let differs = scaled
            .coefficients
            .iter()
            .zip(analytic.coefficients.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-6);
        assert!(differs, "scaled curvature override must change the step");
    }
}
