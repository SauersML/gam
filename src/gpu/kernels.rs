//! Host-side reference implementations of the numerical contracts a future
//! device backend must reproduce bit-exactly (modulo tensor-core rounding).
//!
//! These kernels intentionally avoid relying on faer or BLAS so that a
//! device-backed implementation can be parity-tested against straight Rust
//! arithmetic. They are not on the hot path; the production CPU kernels in
//! [`crate::linalg::faer_ndarray`] are faster.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// `Xᵀ diag(w) X`, allowing negative weights.
///
/// Observed-information assembly can produce negative row curvatures: the
/// CPU sqrt(W)·X Gram trick is invalid there. This signed form is the
/// reference implementation a device backend should match for the
/// observed-Hessian path.
pub fn xtwx_signed(x: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    assert_eq!(x.nrows(), weights.len(), "X/weight row mismatch");
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for i in 0..x.nrows() {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        for a in 0..p {
            let xia = x[(i, a)];
            for b in a..p {
                out[(a, b)] += xia * w * x[(i, b)];
            }
        }
    }
    mirror_upper_to_lower(&mut out);
    out
}

/// `Xᵀ diag(w) X` restricted to non-negative weights. Errors if a negative
/// `w_i` is found rather than silently clipping.
pub fn xtwx_positive(
    x: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    if let Some((idx, value)) = weights.iter().enumerate().find(|(_, v)| **v < 0.0) {
        return Err(format!(
            "xtwx_positive received negative weight at row {idx}: {value}"
        ));
    }
    Ok(xtwx_signed(x, weights))
}

/// `Xᵀ r`, used for dense P-IRLS gradients.
pub fn xt_residual(x: ArrayView2<'_, f64>, residual: ArrayView1<'_, f64>) -> Array1<f64> {
    assert_eq!(x.nrows(), residual.len(), "X/residual row mismatch");
    let mut out = Array1::<f64>::zeros(x.ncols());
    for i in 0..x.nrows() {
        let ri = residual[i];
        if ri == 0.0 {
            continue;
        }
        for j in 0..x.ncols() {
            out[j] += x[(i, j)] * ri;
        }
    }
    out
}

/// Candidate-screen update used by the Levenberg–Marquardt loop:
/// `η_cand = η_curr + X δ`. The screening path uses this to evaluate a
/// candidate without recomputing the full curvature.
pub fn candidate_eta_from_delta(
    x: ArrayView2<'_, f64>,
    eta_current: ArrayView1<'_, f64>,
    delta: ArrayView1<'_, f64>,
) -> Array1<f64> {
    assert_eq!(x.nrows(), eta_current.len(), "X/eta row mismatch");
    assert_eq!(x.ncols(), delta.len(), "X/delta column mismatch");
    let mut out = eta_current.to_owned();
    for i in 0..x.nrows() {
        let mut inc = 0.0;
        for j in 0..x.ncols() {
            inc += x[(i, j)] * delta[j];
        }
        out[i] += inc;
    }
    out
}

fn mirror_upper_to_lower(out: &mut Array2<f64>) {
    let p = out.ncols();
    for a in 0..p {
        for b in 0..a {
            out[(a, b)] = out[(b, a)];
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
    fn positive_path_rejects_negative_weights() {
        let x = array![[1.0], [2.0]];
        let w = array![1.0, -1.0];
        assert!(xtwx_positive(x.view(), w.view()).is_err());
    }

    #[test]
    fn xt_residual_matches_manual() {
        let x = array![[1.0, 2.0], [3.0, -1.0]];
        let r = array![1.0, 2.0];
        let got = xt_residual(x.view(), r.view());
        assert_eq!(got, array![7.0, 0.0]);
    }

    #[test]
    fn candidate_eta_uses_delta_update() {
        let x = array![[1.0, 2.0], [3.0, -1.0]];
        let eta = array![10.0, -2.0];
        let delta = array![0.5, 4.0];
        let got = candidate_eta_from_delta(x.view(), eta.view(), delta.view());
        assert_eq!(got, array![18.5, -4.5]);
    }
}
