//! Host reference kernels for device parity tests.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Sign-preserving `X^T diag(w) X`.
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

/// `X^T residual`, used by dense P-IRLS gradient assembly.
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

/// Candidate-screen update: `eta_candidate = eta_current + X delta`.
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
}
