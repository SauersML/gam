//! Vector-valued response support.
//!
//! This module defines the connector trait the shared
//! [`crate::penalized_vector_glm`] engine consumes, the shared input
//! validation every implementation runs, and the multinomial-logit likelihood.
//!
//! Conventions:
//! - `Y` is shape `(N, M)`: `N` rows, `M` output dimensions.
//! - `eta` is shape `(N, M)`: the linear predictor with one column per output.
//!
//! The Hessian is block-structured: `N` independent per-row blocks, each of
//! size `(M, M)`.

use crate::model_types::EstimationError;
use crate::multinomial_reml::{MultinomialLogitRowProgram, multinomial_logit_probabilities_into};
use ndarray::{Array1, Array2, Array3, ArrayView2};

/// Validate that every row of a multinomial target `y ∈ ℝ^{N×K}` is a point on
/// the probability simplex: `y_{n,c} ≥ 0` for all entries and
/// `Σ_c y_{n,c} = 1` for every row, up to the rounding band `γ_K·Σ_c y_{n,c}`
/// of the row's own `K`-term sum. This is the precondition under which
/// [`MultinomialLogitLikelihood`]'s residual gradient and Fisher block are the
/// exact derivatives of its log-likelihood.
///
/// The multinomial-logit log-likelihood `ℓ = Σ_c y_c log p_c` has the canonical
/// residual gradient `y_a − p_a` and Fisher block `p_a δ_{ab} − p_a p_b` **only**
/// when each target row is a probability vector. For a general row mass
/// `s = Σ_c y_c` the true derivatives are `y_a − s p_a` and
/// `s (p_a δ_{ab} − p_a p_b)`, so any row whose mass deviates from 1 makes the
/// implemented gradient/Hessian disagree with the implemented objective. Simplex
/// rows are therefore required at every construction boundary; the band admits
/// only the round-off of an otherwise exact one-hot or label-smoothed row (a sum
/// of `K` rationals), never genuine count or proportional data. Finiteness is
/// checked first so the message points at the offending entry rather than at a
/// NaN-poisoned row sum.
pub(crate) fn validate_multinomial_simplex(
    y: ArrayView2<f64>,
    context: &str,
) -> Result<(), EstimationError> {
    let (n, k) = y.dim();
    for row in 0..n {
        let mut row_sum = 0.0_f64;
        for c in 0..k {
            let v = y[[row, c]];
            if !v.is_finite() {
                crate::bail_invalid_estim!("{context}: y[{row},{c}] must be finite (got {v})");
            }
            if v < 0.0 {
                crate::bail_invalid_estim!(
                    "{context}: multinomial target must be a probability vector \
                     (y_c ≥ 0); got y[{row},{c}] = {v}"
                );
            }
            row_sum += v;
        }
        // Every entry is non-negative, so `row_sum` is also the absolute sum.
        if (row_sum - 1.0).abs() > gam_linalg::roundoff::accumulation_growth(k) * row_sum {
            crate::bail_invalid_estim!(
                "{context}: multinomial target rows must sum to 1 (one-hot for \
                 hard labels, or a label-smoothed probability vector); row {row} \
                 sums to {row_sum}. The softmax residual gradient y_a − p_a and \
                 Fisher block p_a δ_ab − p_a p_b are the derivatives of \
                 Σ_c y_c log p_c only when the row mass is 1."
            );
        }
    }
    Ok(())
}

fn validate_row_weights(weights: &Array1<f64>, n: usize) -> Result<(), EstimationError> {
    if weights.len() != n {
        crate::bail_invalid_estim!("row_weights length {} ≠ N={n}", weights.len());
    }
    for (idx, weight) in weights.iter().copied().enumerate() {
        if !(weight.is_finite() && weight >= 0.0) {
            crate::bail_invalid_estim!(
                "row_weights[{idx}] must be finite and non-negative (got {weight})"
            );
        }
    }
    Ok(())
}

/// Connector trait the shared [`crate::penalized_vector_glm`] engine plugs
/// into.
///
/// `eta` is the `(N, M)` linear predictor; `y` is the `(N, M)` target. The
/// implementation is responsible for any link inversion. The `hess_diag`
/// return is the per-element diagonal of the per-row Hessian block.
pub trait VectorLikelihood {
    /// log p(Y | η).
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<f64, EstimationError>;

    /// ∂ log p(Y | η) / ∂ η, shape (N, M).
    fn grad_eta(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError>;

    /// Diagonal of the per-row Hessian −∂² log p / ∂ η ∂ η, shape (N, M).
    fn hess_diag(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError>;

    /// Per-row dense Hessian block −∂² log p / ∂η_a ∂η_b, shape (N, M, M).
    ///
    /// Default implementation lifts [`Self::hess_diag`] onto the per-row
    /// diagonal, valid only when the per-row Hessian is genuinely diagonal
    /// across outputs (e.g. independent binomial columns). Likelihoods with
    /// off-diagonal output coupling must override this, as multinomial-logit
    /// does (per-row Fisher block `p_a (δ_ab − p_b)`).
    ///
    /// The returned array is consumed by
    /// [`gam_solve::pirls::dense_block_xtwx`] /
    /// `gam_solve::pirls::dense_block_xtwy` to build `XᵀWX` and `XᵀWy`
    /// for vector-response IRLS in output-major coefficient ordering.
    fn hess_block(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        let diag = self.hess_diag(eta, y)?;
        let (n, m) = diag.dim();
        let mut out = Array3::<f64>::zeros((n, m, m));
        for row in 0..n {
            for j in 0..m {
                out[[row, j, j]] = diag[[row, j]];
            }
        }
        Ok(out)
    }
}

pub(crate) fn validate_vector_likelihood_inputs(
    context: &str,
    eta: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    expected_columns: Option<usize>,
) -> Result<(), EstimationError> {
    if eta.dim() != y.dim() {
        crate::bail_invalid_estim!(
            "{context}: eta shape {:?} does not match response shape {:?}",
            eta.dim(),
            y.dim()
        );
    }
    if let Some(expected) = expected_columns
        && eta.ncols() != expected
    {
        crate::bail_invalid_estim!(
            "{context}: eta has {} columns; expected {expected}",
            eta.ncols()
        );
    }
    if let Some(((row, column), value)) = eta.indexed_iter().find(|(_, value)| !value.is_finite()) {
        crate::bail_invalid_estim!("{context}: eta[{row},{column}] must be finite, got {value}");
    }
    if let Some(((row, column), value)) = y.indexed_iter().find(|(_, value)| !value.is_finite()) {
        crate::bail_invalid_estim!(
            "{context}: response[{row},{column}] must be finite, got {value}"
        );
    }
    Ok(())
}

/// Multinomial-logit (softmax) likelihood with explicit reference class.
///
/// Conventions:
/// - `K` is the total number of classes; the linear predictor has `M = K - 1`
///   columns corresponding to the *active* classes. Class `K - 1` is the
///   reference class with η_{K-1} ≡ 0 (so the gauge is fixed by construction
///   and no additional sum-to-zero projection is required at the η level).
/// - `y` is the categorical response with shape `(N, K)`. Each row must be a
///   point on the probability simplex (`y_c ≥ 0`, `Σ_c y_c = 1`): a one-hot
///   indicator for hard-label classification, or a label-smoothed probability
///   vector. The row *weight* `w_n` scales the whole row's likelihood
///   contribution and is independent of the row mass — it is **not** the row
///   sum. Callers enforce the simplex precondition via
///   `validate_multinomial_simplex` at every construction boundary; under it
///   the residual gradient `y_a − p_a` and Fisher block `p_a δ_ab − p_a p_b`
///   below are the exact derivatives of the log-likelihood `Σ_c y_c log p_c`.
/// - `eta` is the active linear predictor with shape `(N, M = K - 1)`.
///
/// Softmax with baseline:
/// ```text
///     p_a   = exp(η_a) / (1 + Σ_b exp(η_b))           for a ∈ [0, K-1)
///     p_{K-1} = 1 / (1 + Σ_b exp(η_b))
/// ```
///
/// Log-likelihood (rows with weight `w_n`, default 1.0):
/// ```text
///     log L = Σ_n w_n · ( Σ_{a < K-1} y_{n,a} · η_{n,a} − log(1 + Σ_b exp(η_{n,b})) )
///           = Σ_n w_n · Σ_{c ∈ [0, K)} y_{n,c} · log p_{n,c}
/// ```
///
/// Per-row gradient w.r.t. the active η is the canonical Bernoulli/softmax
/// residual:
/// ```text
///     ∂ log L / ∂η_{n,a} = w_n · (y_{n,a} − p_{n,a})       for a ∈ [0, K-1)
/// ```
///
/// Per-row Fisher (= observed, since logit is canonical for the multinomial)
/// information block, shape `(M, M)`:
/// ```text
///     H_{n,a,b} = w_n · ( p_{n,a} · δ_{ab} − p_{n,a} · p_{n,b} )
/// ```
///
/// This is the standard reference-coded multinomial-logit GLM. The dense
/// per-row block flows through [`VectorLikelihood::hess_block`] into
/// [`gam_solve::pirls::dense_block_xtwx`], which builds the stacked
/// `XᵀWX` in output-major coefficient ordering `β = [β_0; β_1; …; β_{K-2}]`
/// with each per-class block of size `(P, P)`.
#[derive(Clone, Debug)]
pub struct MultinomialLogitLikelihood {
    /// Number of active classes `M = K − 1`. Cached for shape checks.
    pub active_classes: usize,
    /// Optional row weights (length N), or `None` for uniform 1.0.
    pub row_weights: Option<Array1<f64>>,
}

impl MultinomialLogitLikelihood {
    /// Construct from the total number of classes `K ≥ 2`.
    pub(crate) fn with_classes(total_classes: usize) -> Result<Self, EstimationError> {
        if total_classes < 2 {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood requires K ≥ 2 classes (got {total_classes})"
            );
        }
        Ok(Self {
            active_classes: total_classes - 1,
            row_weights: None,
        })
    }

    /// Attach per-row weights (length N, finite and non-negative).
    pub fn with_row_weights(mut self, w: Array1<f64>) -> Result<Self, EstimationError> {
        validate_row_weights(&w, w.len())?;
        self.row_weights = Some(w);
        Ok(self)
    }

    /// Total class count `K = M + 1`.
    #[inline]
    pub(crate) fn total_classes(&self) -> usize {
        self.active_classes + 1
    }

    #[inline]
    fn row_weight(&self, n: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[n])
    }

    /// Numerically-stable softmax with implicit reference column (η_{K-1} = 0).
    ///
    /// Writes `K` probabilities into `out` (length `M + 1`). The shift uses
    /// `max(0, max(eta_active))` so the reference class is included in the
    /// max and the denominator stays bounded. This is the canonical
    /// reference implementation; the FFI surface and any direct
    /// matrix-free callers route through this method rather than carrying
    /// their own softmax.
    pub(crate) fn softmax_with_baseline(eta_active: &[f64], out: &mut [f64]) {
        multinomial_logit_probabilities_into(eta_active, out);
    }

    /// Convenience: compute the full (N, K) probability matrix from
    /// (N, K-1) active linear predictor. This is the multinomial inverse
    /// link used by prediction.
    pub fn probabilities(&self, eta: ArrayView2<f64>) -> Array2<f64> {
        let n = eta.nrows();
        let m = self.active_classes;
        assert_eq!(eta.ncols(), m, "η must have K-1 columns");
        let k = self.total_classes();
        let eta = eta.as_standard_layout();
        let eta_values = eta
            .as_slice()
            .expect("standard-layout multinomial logits are contiguous");
        let mut probs = Array2::<f64>::zeros((n, k));
        let probs_values = probs
            .as_slice_mut()
            .expect("fresh multinomial probabilities are contiguous");
        let mut eta_row = vec![0.0_f64; m];
        let mut probs_row = vec![0.0_f64; k];
        for row in 0..n {
            eta_row.copy_from_slice(&eta_values[row * m..(row + 1) * m]);
            Self::softmax_with_baseline(&eta_row, &mut probs_row);
            probs_values[row * k..(row + 1) * k].copy_from_slice(&probs_row);
        }
        probs
    }

    #[inline]
    fn row_program<'row>(
        &self,
        row: usize,
        eta: &'row [f64],
        response: &'row [f64],
    ) -> Result<MultinomialLogitRowProgram<'row>, EstimationError> {
        MultinomialLogitRowProgram::new(eta, response, self.row_weight(row)).map_err(|error| {
            EstimationError::InvalidInput(format!("invalid multinomial row {row}: {error}"))
        })
    }

    /// Fused live value/gradient/Hessian evaluation. This is the one production
    /// batch entry used by the joint REML adapter, so it performs one stable
    /// normalization per row rather than three independent likelihood passes.
    pub(crate) fn value_gradient_hessian(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<(f64, Array2<f64>, Array3<f64>), EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::value_gradient_hessian: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::value_gradient_hessian active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let eta = eta.as_standard_layout();
        let eta_values = eta
            .as_slice()
            .expect("standard-layout multinomial logits are contiguous");
        let y = y.as_standard_layout();
        let response_values = y
            .as_slice()
            .expect("standard-layout multinomial responses are contiguous");
        let mut gradient_log_likelihood = Array2::<f64>::zeros((n, m));
        let mut hessian = Array3::<f64>::zeros((n, m, m));
        let gradient_values = gradient_log_likelihood
            .as_slice_mut()
            .expect("fresh multinomial gradient is contiguous");
        let hessian_values = hessian
            .as_slice_mut()
            .expect("fresh multinomial Hessian is contiguous");
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut gradient_nll = vec![0.0_f64; m];
        let mut hessian_row = vec![0.0_f64; m * m];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            eta_row.copy_from_slice(&eta_values[row * m..(row + 1) * m]);
            response_row.copy_from_slice(&response_values[row * k..(row + 1) * k]);
            let program = self.row_program(row, &eta_row, &response_row)?;
            negative_log_likelihood += program.value_gradient_hessian_into(
                &mut probabilities,
                &mut gradient_nll,
                &mut hessian_row,
            );
            for axis in 0..m {
                gradient_values[row * m + axis] = -gradient_nll[axis];
            }
            hessian_values[row * m * m..(row + 1) * m * m].copy_from_slice(&hessian_row);
        }
        Ok((-negative_log_likelihood, gradient_log_likelihood, hessian))
    }

    /// Fused value/gradient entry for callers that do not consume curvature.
    pub(crate) fn value_gradient(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<(f64, Array2<f64>), EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::value_gradient: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::value_gradient active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let eta = eta.as_standard_layout();
        let eta_values = eta
            .as_slice()
            .expect("standard-layout multinomial logits are contiguous");
        let y = y.as_standard_layout();
        let response_values = y
            .as_slice()
            .expect("standard-layout multinomial responses are contiguous");
        let mut gradient_log_likelihood = Array2::<f64>::zeros((n, m));
        let gradient_values = gradient_log_likelihood
            .as_slice_mut()
            .expect("fresh multinomial gradient is contiguous");
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut gradient_nll = vec![0.0_f64; m];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            eta_row.copy_from_slice(&eta_values[row * m..(row + 1) * m]);
            response_row.copy_from_slice(&response_values[row * k..(row + 1) * k]);
            let program = self.row_program(row, &eta_row, &response_row)?;
            negative_log_likelihood +=
                program.value_gradient_into(&mut probabilities, &mut gradient_nll);
            for axis in 0..m {
                gradient_values[row * m + axis] = -gradient_nll[axis];
            }
        }
        Ok((-negative_log_likelihood, gradient_log_likelihood))
    }
}

impl VectorLikelihood for MultinomialLogitLikelihood {
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<f64, EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::log_lik: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::log_lik active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            negative_log_likelihood += self
                .row_program(row, &eta_row, &response_row)?
                .negative_log_likelihood();
        }
        Ok(-negative_log_likelihood)
    }

    fn grad_eta(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        Ok(self.value_gradient(eta, y)?.1)
    }

    fn hess_diag(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Per-row diagonal of the (M, M) Fisher block:
        //     H_{n,a,a} = w_n · p_{n,a} · (1 − p_{n,a})
        // Provided for callers that explicitly want the diagonal-only
        // preconditioner; the joint dense block ships through `hess_block`.
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::hess_diag: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::hess_diag active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut out = Array2::<f64>::zeros((n, m));
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut diagonal = vec![0.0_f64; m];
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            self.row_program(row, &eta_row, &response_row)?
                .hessian_diagonal_into(&mut probabilities, &mut diagonal);
            for axis in 0..m {
                out[[row, axis]] = diagonal[axis];
            }
        }
        Ok(out)
    }

    fn hess_block(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        Ok(self.value_gradient_hessian(eta, y)?.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Macro (not fn) so the assertion / panic tokens are inlined into each
    // caller's test body, satisfying the build.rs scanner that looks for
    // `assert!(` / `panic!(` directly in the `#[test]` function.
    macro_rules! expect_invalid_input {
        ($result:expr, $needle:expr $(,)?) => {{
            let needle: &str = $needle;
            match $result {
                Ok(_) => {
                    panic!("expected EstimationError::InvalidInput containing `{needle}`, got Ok")
                }
                Err(EstimationError::InvalidInput(msg)) => {
                    assert!(
                        msg.contains(needle),
                        "InvalidInput message `{msg}` does not contain `{needle}`"
                    );
                    msg
                }
                Err(other) => panic!(
                    "expected EstimationError::InvalidInput containing `{needle}`, got {other:?}"
                ),
            }
        }};
    }

    #[test]
    fn multinomial_row_validation_propagates_as_typed_likelihood_error_932() {
        let likelihood = MultinomialLogitLikelihood::with_classes(3)
            .expect("three-class reference-coded likelihood");
        let eta =
            Array2::from_shape_vec((1, 2), vec![f64::INFINITY, 0.0]).expect("active eta shape");
        let response =
            Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("simplex response shape");
        expect_invalid_input!(
            likelihood.log_lik(eta.view(), response.view()),
            "eta[0,0] must be finite",
        );
    }

    /// gam#932. A non-finite optimizer state arrives at the vector likelihood
    /// as DATA, not as a bug in the caller: the row loop must refuse it with the
    /// offending index named, rather than panicking or returning a NaN
    /// objective that the line search would then read as an improvement.
    #[test]
    fn vector_likelihood_rejects_nonfinite_optimizer_state_without_panicking_932() {
        let likelihood = MultinomialLogitLikelihood::with_classes(3)
            .expect("three-class reference-coded likelihood");
        let response =
            Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.0]).expect("simplex response shape");
        // Positive control: the same fixture with a finite state is evaluated,
        // so the refusal below is about the NaN and not about this fixture
        // failing to reach the validation at all.
        let finite = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("finite eta shape");
        let value = likelihood
            .log_lik(finite.view(), response.view())
            .expect("a finite optimizer state must be evaluated");
        assert!(
            value.is_finite(),
            "the finite control must produce a finite objective, got {value}"
        );
        let eta = Array2::from_shape_vec((1, 2), vec![0.0, f64::NAN]).expect("eta shape");
        expect_invalid_input!(
            likelihood.log_lik(eta.view(), response.view()),
            "eta[0,1] must be finite",
        );
        expect_invalid_input!(
            likelihood.grad_eta(eta.view(), response.view()),
            "eta[0,1] must be finite",
        );
        expect_invalid_input!(
            likelihood.hess_block(eta.view(), response.view()),
            "eta[0,1] must be finite",
        );
    }
}
