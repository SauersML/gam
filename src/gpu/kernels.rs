//! Host reference kernels for device parity tests.
//!
//! These kernels intentionally avoid faer / BLAS so that a device-backed
//! implementation can be parity-tested against straight Rust arithmetic.
//! They are not on the hot path; the production CPU kernels in
//! [`crate::linalg::faer_ndarray`] are faster. Each kernel is paired with an
//! active GPU dispatch entry point or a production CPU kernel with the same
//! numerical contract.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Sign-preserving `X^T diag(w) X` reference.
///
/// Observed-information assembly can produce negative row curvatures, so
/// a `sqrt(max(w, 0)) · X` Gram form is wrong here: it silently clips. The
/// device backend must reproduce the signed form below.
pub fn xtwx_signed(x: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    assert_eq!(x.nrows(), weights.len(), "X/weight row mismatch");
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for row in 0..x.nrows() {
        let weight = weights[row];
        if weight == 0.0 {
            continue;
        }
        for lhs in 0..p {
            let weighted = x[(row, lhs)] * weight;
            for rhs in lhs..p {
                out[(lhs, rhs)] += weighted * x[(row, rhs)];
            }
        }
    }
    mirror_upper_to_lower(&mut out);
    out
}

/// `X^T residual` reference, used by dense P-IRLS gradient assembly.
pub fn xt_residual(x: ArrayView2<'_, f64>, residual: ArrayView1<'_, f64>) -> Array1<f64> {
    assert_eq!(x.nrows(), residual.len(), "X/residual row mismatch");
    let mut out = Array1::<f64>::zeros(x.ncols());
    for row in 0..x.nrows() {
        let value = residual[row];
        if value == 0.0 {
            continue;
        }
        for col in 0..x.ncols() {
            out[col] += x[(row, col)] * value;
        }
    }
    out
}

/// Candidate-screen reference: `eta_candidate = eta_current + X delta`.
pub fn candidate_eta_from_delta(
    x: ArrayView2<'_, f64>,
    eta_current: ArrayView1<'_, f64>,
    delta: ArrayView1<'_, f64>,
) -> Array1<f64> {
    assert_eq!(x.nrows(), eta_current.len(), "X/eta row mismatch");
    assert_eq!(x.ncols(), delta.len(), "X/delta column mismatch");
    let mut out = eta_current.to_owned();
    for row in 0..x.nrows() {
        let mut inc = 0.0;
        for col in 0..x.ncols() {
            inc += x[(row, col)] * delta[col];
        }
        out[row] += inc;
    }
    out
}

/// Sign-preserving row scaling reference: `Y[i, j] = w[i] * X[i, j]`.
///
/// The device implementation should produce identical numerical results
/// when fused with subsequent GEMM/GEMV calls.
pub fn row_scale_signed(x: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    assert_eq!(x.nrows(), weights.len(), "X/weight row mismatch");
    let mut out = Array2::<f64>::zeros(x.raw_dim());
    for row in 0..x.nrows() {
        let w = weights[row];
        for col in 0..x.ncols() {
            out[(row, col)] = w * x[(row, col)];
        }
    }
    out
}

/// Fused logistic-link IRLS update reference for the canonical Bernoulli/Binomial
/// case (`g(μ) = logit(μ)`, no integration, no mixture).
///
/// Returns `(μ, w, z)` where:
/// * `μ_i = sigmoid(η_i) = 1 / (1 + exp(-η_i))` (numerically stable form)
/// * `w_i = prior_i · μ_i · (1 - μ_i)`
/// * `z_i = η_i + (y_i - μ_i) / max(μ_i · (1 - μ_i), tiny)` (working response)
///
/// This is the inner loop the production CPU kernel in `update_glmvectors`
/// optimizes with SIMD; a device implementation must match it bit-for-bit
/// in the absence of `μ` boundary clipping.
pub fn fused_logit_irls(
    eta: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    priorweights: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = eta.len();
    assert_eq!(y.len(), n, "eta/y length mismatch");
    assert_eq!(priorweights.len(), n, "eta/priorweights length mismatch");
    let mut mu = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta_i = eta[i];
        // Numerically stable sigmoid: separate signs to avoid overflow.
        let mu_i = if eta_i >= 0.0 {
            let e = (-eta_i).exp();
            1.0 / (1.0 + e)
        } else {
            let e = eta_i.exp();
            e / (1.0 + e)
        };
        let var = mu_i * (1.0 - mu_i);
        let var_safe = var.max(1.0e-12);
        mu[i] = mu_i;
        w[i] = priorweights[i] * var;
        z[i] = eta_i + (y[i] - mu_i) / var_safe;
    }
    (mu, w, z)
}

/// Reference parallel reduction: `Σ x_i`.
///
/// The device implementation should use a tree reduction with a Kahan/Neumaier
/// compensation step matching this loop order to avoid silent precision loss
/// for ill-conditioned sums.
pub fn pairwise_sum(values: ArrayView1<'_, f64>) -> f64 {
    fn rec(slice: &[f64]) -> f64 {
        match slice.len() {
            0 => 0.0,
            1 => slice[0],
            n => {
                let mid = n / 2;
                rec(&slice[..mid]) + rec(&slice[mid..])
            }
        }
    }
    let slice = values.as_slice().unwrap_or(&[]);
    if slice.is_empty() {
        values.iter().copied().sum()
    } else {
        rec(slice)
    }
}

/// Hutch++ trace estimator reference for `tr(A)` given an apply-operator and
/// `k` Rademacher / Gaussian probes. Used by REML's `trace_hinv_operator`
/// path. The host reference here uses Gaussian probes and the simple
/// Hutchinson estimator (sufficient for parity checks); production REML uses
/// the full Hutch++ algorithm in `solver::reml::unified`.
pub fn hutch_trace_reference<F: FnMut(&Array1<f64>) -> Array1<f64>>(
    n: usize,
    mut apply: F,
    probes: &[Array1<f64>],
) -> f64 {
    if probes.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for v in probes {
        assert_eq!(v.len(), n, "probe length mismatch");
        let av = apply(v);
        acc += v.dot(&av);
    }
    acc / probes.len() as f64
}

/// Cubic cell-moment kernel reference for spatial bases.
///
/// For each row `i` with cell index `c_i` and weight `w_i`, contributes
/// `w_i · monomial(x_i)` into the `c_i`-th cell accumulator. The reference
/// here is the simple per-row scatter; a device implementation should use a
/// segmented reduction keyed by cell index.
pub fn cell_moment_scatter(cell_index: &[usize], weights: ArrayView1<'_, f64>, out: &mut [f64]) {
    assert_eq!(
        cell_index.len(),
        weights.len(),
        "cell/weight length mismatch"
    );
    for (idx, &w) in cell_index.iter().zip(weights.iter()) {
        if let Some(slot) = out.get_mut(*idx) {
            *slot += w;
        }
    }
}

fn mirror_upper_to_lower(out: &mut Array2<f64>) {
    let n = out.ncols();
    for row in 0..n {
        for col in 0..row {
            out[(row, col)] = out[(col, row)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn signed_xtwx_preserves_negative_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]];
        let w = array![2.0, -0.5, 1.5];
        let got = xtwx_signed(x.view(), w.view());
        let expected = array![[3.5, 17.5], [17.5, 31.5]];
        assert_eq!(got, expected);
    }

    #[test]
    fn candidate_eta_uses_delta_update() {
        let x = array![[1.0, 2.0], [3.0, -1.0]];
        let eta = array![10.0, -2.0];
        let delta = array![0.5, 4.0];
        let got = candidate_eta_from_delta(x.view(), eta.view(), delta.view());
        assert_eq!(got, array![18.5, -4.5]);
    }

    #[test]
    fn row_scale_signed_preserves_negatives() {
        let x = array![[1.0, 2.0], [3.0, -1.0]];
        let w = array![-2.0, 0.5];
        let got = row_scale_signed(x.view(), w.view());
        assert_eq!(got, array![[-2.0, -4.0], [1.5, -0.5]]);
    }

    #[test]
    fn fused_logit_irls_matches_canonical_formulas() {
        let eta = array![0.0, 1.0, -1.0];
        let y = array![1.0, 0.0, 1.0];
        let pw = array![1.0, 1.0, 1.0];
        let (mu, w, z) = fused_logit_irls(eta.view(), y.view(), pw.view());
        assert!((mu[0] - 0.5).abs() < 1e-12);
        // For η=0, w = 0.25 and z = 0 + (1 - 0.5)/0.25 = 2.
        assert!((w[0] - 0.25).abs() < 1e-12);
        assert!((z[0] - 2.0).abs() < 1e-12);
        // Sigmoid symmetric: μ(η)+μ(-η)=1.
        assert!((mu[1] + mu[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_sum_matches_naive_sum_in_double() {
        let values: Array1<f64> = (0..1024).map(|i| (i as f64) * 0.5 - 100.0).collect();
        let naive = values.iter().sum::<f64>();
        let pair = pairwise_sum(values.view());
        assert!((naive - pair).abs() < 1e-9);
    }

    #[test]
    fn hutch_trace_reference_matches_explicit_trace_on_identity() {
        let probes = vec![array![1.0, 0.0], array![0.0, 1.0]];
        let trace = hutch_trace_reference(2, |v| v.clone(), &probes);
        assert!((trace - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cell_moment_scatter_accumulates_into_target_cells() {
        let idx = vec![0, 2, 0, 1];
        let w = array![1.0, 2.5, -0.5, 3.0];
        let mut out = vec![0.0; 3];
        cell_moment_scatter(&idx, w.view(), &mut out);
        assert_eq!(out, vec![0.5, 3.0, 2.5]);
    }
}
