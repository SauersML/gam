//! Penalized multi-output binomial-logit fitter at fixed λ.
//!
//! This is the row-diagonal sibling of [`crate::families::multinomial`]: the
//! same shared design `X ∈ ℝ^{N×P}` and shared penalty `S ∈ ℝ^{P×P}` are
//! reused across `K` independent binomial-logit response columns. Per-column
//! smoothing parameters `λ_a` (length `K`) scale `S` independently for each
//! response. Because the Fisher information has no cross-column coupling
//! (`H_{n,a,b} = δ_{ab} · w_n · μ_{n,a} (1 − μ_{n,a})`), the joint penalized
//! Hessian decouples into `K` block-diagonal `P × P` systems and each
//! response is fitted by its own damped Newton loop.
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
//! model overshoots near saturation. Convergence is the same
//! relative-step criterion as `fit_penalized_multinomial`.
//!
//! # Relation to the multi-class softmax driver
//!
//! [`crate::families::multinomial::fit_penalized_multinomial`] handles the
//! coupled softmax Fisher block `H_{n,a,b} = w_n μ_{n,a} (δ_{ab} − μ_{n,b})`
//! and is the right entry when the user wants a single normalized
//! probability vector per row. This driver is the right entry when the
//! user has `K` independent binary marginals sharing a smooth basis (e.g.
//! multi-label classification, multi-trait penalised logistic regression
//! on a Duchon latent design).
//!
//! The function-boundary contract mirrors `fit_penalized_multinomial` so
//! the two are interchangeable at the FFI layer: same input arity, same
//! convergence semantics, same `(N, K)` fitted-probability output.

use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, factorize_symmetricwith_fallback};
use crate::solver::estimate::EstimationError;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};

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
    /// Total Newton iterations summed across all `K` response columns. For a
    /// per-column breakdown, callers can inspect `iterations_per_response`.
    pub iterations: usize,
    /// Per-response Newton iteration count, length `K`.
    pub iterations_per_response: Vec<usize>,
    /// `true` if every column satisfied the relative-step test before
    /// `max_iter`. `false` if any column exhausted the budget.
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

/// Penalized binomial log-likelihood contribution for one response column.
/// Returns `Σ_n w_n [ y_n log μ_n + (1 − y_n) log(1 − μ_n) ]`, clamping
/// `μ_n` to `[1e-12, 1 − 1e-12]` so the closed-form expression remains
/// finite when β drives a row deeply into saturation during a tentative
/// Newton step (the surrounding line search rejects such steps).
fn binomial_log_lik_column(
    eta_col: ArrayView1<'_, f64>,
    y_col: ArrayView1<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> f64 {
    let mut acc = 0.0_f64;
    for (i, &eta_i) in eta_col.iter().enumerate() {
        let mu = sigmoid_stable(eta_i).clamp(1.0e-12, 1.0 - 1.0e-12);
        let y = y_col[i];
        let w = row_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
        acc += w * (y * mu.ln() + (1.0 - y) * (1.0 - mu).ln());
    }
    acc
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

    // ────────────────────────────── shape checks ──────────────────────────
    let n_obs = design.nrows();
    let p = design.ncols();
    if n_obs == 0 || p == 0 {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: design must be nonempty (got {n_obs}x{p})"
        );
    }
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
    if penalty.dim() != (p, p) {
        crate::bail_invalid_estim!(
            "fit_penalized_binomial_multi: penalty shape {:?} ≠ (P, P) = ({p}, {p})",
            penalty.dim()
        );
    }
    for (i, &v) in lambdas.iter().enumerate() {
        if !(v.is_finite() && v >= 0.0) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: lambdas[{i}] must be finite and ≥ 0 (got {v})"
            );
        }
    }
    if let Some(fw) = fisher_w_override.as_ref() {
        if fw.dim() != (n_obs, k, k) {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: fisher_w_override shape {:?} ≠ (N, K, K) = ({n_obs}, {k}, {k})",
                fw.dim()
            );
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
    for ((i, j), &v) in design.indexed_iter() {
        if !v.is_finite() {
            crate::bail_invalid_estim!(
                "fit_penalized_binomial_multi: design[{i},{j}] must be finite (got {v})"
            );
        }
    }

    // ────────────────────────── Newton iteration ──────────────────────────
    let mut beta = Array2::<f64>::zeros((p, k));
    let mut eta = Array2::<f64>::zeros((n_obs, k));
    let mut fitted = Array2::<f64>::zeros((n_obs, k));
    let mut iterations_per_response = vec![0_usize; k];
    let mut all_converged = true;

    for a in 0..k {
        let lambda_a = lambdas[a];
        let y_col = y.column(a);
        let mut converged_a = false;
        let mut last_objective_a = f64::INFINITY;

        for iter in 0..max_iter {
            iterations_per_response[a] = iter + 1;

            // η_a = X β_a; μ_a = σ(η_a).
            for row in 0..n_obs {
                let mut eta_val = 0.0_f64;
                for i in 0..p {
                    eta_val += design[[row, i]] * beta[[i, a]];
                }
                eta[[row, a]] = eta_val;
                fitted[[row, a]] = sigmoid_stable(eta_val);
            }

            // Working weights diag(w_n μ_n (1 − μ_n)) and gradient
            // contribution Xᵀ diag(w_n)(μ_n − y_n).
            let mut hess_diag = Array1::<f64>::zeros(n_obs);
            let mut grad = Array1::<f64>::zeros(p);
            for row in 0..n_obs {
                let mu = fitted[[row, a]];
                let w = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
                // Curvature: caller override diagonal when present, else the
                // analytic binomial Fisher w_n μ_a(1 − μ_a) (issue #349).
                hess_diag[row] = match fisher_w_override.as_ref() {
                    Some(fw) => fw[[row, a, a]],
                    None => w * mu * (1.0 - mu),
                };
                let resid = w * (mu - y_col[row]);
                if resid != 0.0 {
                    for i in 0..p {
                        grad[i] += design[[row, i]] * resid;
                    }
                }
            }

            // Penalty gradient λ_a S β_a.
            if lambda_a != 0.0 {
                let beta_col = beta.column(a);
                for i in 0..p {
                    let mut s_beta_i = 0.0_f64;
                    for j in 0..p {
                        s_beta_i += penalty[[i, j]] * beta_col[j];
                    }
                    grad[i] += lambda_a * s_beta_i;
                }
            }

            // Penalized Hessian Xᵀ diag(w_n μ_n (1 − μ_n)) X + λ_a S.
            let mut hessian = Array2::<f64>::zeros((p, p));
            for row in 0..n_obs {
                let w = hess_diag[row];
                if w == 0.0 {
                    continue;
                }
                for i in 0..p {
                    let xi = design[[row, i]];
                    if xi == 0.0 {
                        continue;
                    }
                    let scaled = w * xi;
                    for j in 0..p {
                        hessian[[i, j]] += scaled * design[[row, j]];
                    }
                }
            }
            if lambda_a != 0.0 {
                for i in 0..p {
                    for j in 0..p {
                        hessian[[i, j]] += lambda_a * penalty[[i, j]];
                    }
                }
            }
            // Symmetrise to discharge accumulated rounding asymmetry.
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (hessian[[i, j]] + hessian[[j, i]]);
                    hessian[[i, j]] = avg;
                    hessian[[j, i]] = avg;
                }
            }

            // δ = − H^{-1} grad.
            let factor = factorize_symmetricwith_fallback(
                FaerArrayView::new(&hessian).as_ref(),
                Side::Lower,
            )
            .map_err(|err| {
                EstimationError::InvalidInput(format!(
                    "fit_penalized_binomial_multi: Hessian factorization failed at response {a}, iter {iter}: {err}"
                ))
            })?;
            let mut rhs = Array2::<f64>::zeros((p, 1));
            for i in 0..p {
                rhs[[i, 0]] = -grad[i];
            }
            {
                let rhs_view = array2_to_matmut(&mut rhs);
                factor.solve_in_place(rhs_view);
            }
            let mut delta = Array1::<f64>::zeros(p);
            for i in 0..p {
                delta[i] = rhs[[i, 0]];
            }
            if delta.iter().any(|v| !v.is_finite()) {
                crate::bail_invalid_estim!(
                    "fit_penalized_binomial_multi: Newton step is non-finite at response {a}, iter {iter}"
                );
            }

            // Penalized objective evaluator at a trial β_a.
            let evaluate_objective = |beta_trial_col: &Array1<f64>| -> f64 {
                let mut eta_trial = Array1::<f64>::zeros(n_obs);
                for row in 0..n_obs {
                    let mut v = 0.0_f64;
                    for i in 0..p {
                        v += design[[row, i]] * beta_trial_col[i];
                    }
                    eta_trial[row] = v;
                }
                let ll = binomial_log_lik_column(eta_trial.view(), y_col.view(), row_weights);
                let mut pen = 0.0_f64;
                if lambda_a != 0.0 {
                    let mut quad = 0.0_f64;
                    for i in 0..p {
                        let mut s_beta_i = 0.0_f64;
                        for j in 0..p {
                            s_beta_i += penalty[[i, j]] * beta_trial_col[j];
                        }
                        quad += beta_trial_col[i] * s_beta_i;
                    }
                    pen = 0.5 * lambda_a * quad;
                }
                -ll + pen
            };

            if iter == 0 {
                let beta_init: Array1<f64> = beta.column(a).to_owned();
                last_objective_a = evaluate_objective(&beta_init);
                if !last_objective_a.is_finite() {
                    crate::bail_invalid_estim!(
                        "fit_penalized_binomial_multi: non-finite objective at response {a}, β = 0"
                    );
                }
            }

            // Backtracking line search; up to 8 halvings.
            let propose_beta = |alpha: f64| -> Array1<f64> {
                let mut out: Array1<f64> = beta.column(a).to_owned();
                for i in 0..p {
                    out[i] += alpha * delta[i];
                }
                out
            };
            let mut alpha = 1.0_f64;
            let mut accepted_beta_a = propose_beta(alpha);
            let mut new_objective = evaluate_objective(&accepted_beta_a);
            let mut backtrack = 0_usize;
            while (!new_objective.is_finite() || new_objective > last_objective_a + 1.0e-12)
                && backtrack < 8
            {
                alpha *= 0.5;
                accepted_beta_a = propose_beta(alpha);
                new_objective = evaluate_objective(&accepted_beta_a);
                backtrack += 1;
            }

            // Relative-step convergence test.
            let mut step_norm_sq = 0.0_f64;
            let mut beta_norm_sq = 0.0_f64;
            for i in 0..p {
                let d = accepted_beta_a[i] - beta[[i, a]];
                step_norm_sq += d * d;
                let v = accepted_beta_a[i];
                beta_norm_sq += v * v;
            }
            for i in 0..p {
                beta[[i, a]] = accepted_beta_a[i];
            }
            last_objective_a = new_objective;

            let step_norm = step_norm_sq.sqrt();
            let beta_norm = beta_norm_sq.sqrt();
            if step_norm <= tol * (1.0 + beta_norm) {
                converged_a = true;
                break;
            }
        }
        if !converged_a {
            all_converged = false;
        }
    }

    // ──────────────────────────── post-process ────────────────────────────
    // Recompute η, μ from final β̂ and tally the penalized objective.
    for a in 0..k {
        for row in 0..n_obs {
            let mut v = 0.0_f64;
            for i in 0..p {
                v += design[[row, i]] * beta[[i, a]];
            }
            eta[[row, a]] = v;
            fitted[[row, a]] = sigmoid_stable(v);
        }
    }
    let mut log_lik = 0.0_f64;
    for a in 0..k {
        log_lik += binomial_log_lik_column(eta.column(a), y.column(a), row_weights);
    }
    let mut pen = 0.0_f64;
    for a in 0..k {
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
    let penalized_neg_log_likelihood = -log_lik + pen;
    let deviance = -2.0 * log_lik;
    let iterations: usize = iterations_per_response.iter().sum();

    Ok(BinomialMultiFitOutputs {
        coefficients: beta,
        fitted_probabilities: fitted,
        iterations,
        iterations_per_response,
        converged: all_converged,
        penalized_neg_log_likelihood,
        deviance,
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
