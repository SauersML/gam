use super::mlp_account::Account;
use super::operator_program::Law;
use gam_linalg::faer_ndarray::fast_abt;
use ndarray::Array2;

/// A deterministic pseudo-random matrix with entries in `[-scale, scale]`.
fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed | 1;
    Array2::from_shape_fn((rows, cols), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        scale * ((state % 2_000_001) as f64 / 1_000_000.0 - 1.0)
    })
}

/// The dense map as an account computes the MLP `W_out σ(W_in h)` exactly.
#[test]
fn neurons_account_is_the_mlp() {
    let (w_in, w_out) = (matrix(7, 5, 3, 0.8), matrix(6, 7, 5, 0.8));
    let h = matrix(20, 5, 9, 1.5);
    let account = Account::neurons(&w_in, &w_out);
    let native = fast_abt(&fast_abt(&h, &w_in).mapv(|t| Law::GeluTanh.apply(t)), &w_out);
    let gap = (&account.apply(h.view()) - &native).mapv(f64::abs).fold(0.0f64, |m, v| m.max(*v));
    assert!(gap < 1e-12, "{gap}");
}
