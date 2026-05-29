//! Regression test for issue #390: `BlockOrthogonalityPenalty::hvp` returned a
//! wrong, non-symmetric Hessian.
//!
//! The penalty is `P = ½·w·Σ_{g<h} ‖C_{g,h}‖_F²` where
//! `C_{g,h}[a, b] = Σ_n t[n, axes_g[a]] · t[n, axes_h[b]]`. The gradient is
//! correct, but the second-derivative term in `hvp_with_precomputed_cross`
//! computed `cross_gram(v, h, g) + cross_gram(t, h, g)`, i.e. `(v⊗v) + (t⊗t)`,
//! where the true directional derivative of `C_{h,g}` in direction `v` is the
//! MIXED gram `(v⊗t) + (t⊗v)`:
//!
//!     ∂_v C_{h,g}[hi, gi] = Σ_n {v[n, axes_h[hi]] · t[n, axes_g[gi]]
//!                                + t[n, axes_h[hi]] · v[n, axes_g[gi]]}
//!
//! Consequences pre-fix:
//! * `hvp` is no longer linear in `v` (one of the wrong terms is quadratic in v,
//!   the other is independent of v), so it is not a Hessian-vector product at all.
//! * The Hessian materialised through `as_dense` is non-symmetric.
//! * `hvp(t, v) - hvp(t, 0)` does not match finite-difference
//!   `(grad(t+εv) - grad(t-εv))/(2ε)`.
//!
//! This test asserts the three properties that the corrected formula satisfies:
//! linearity, symmetry, and finite-difference agreement with the gradient.

use ndarray::{Array1, Array2};

use gam::terms::AnalyticPenalty;
use gam::terms::BlockOrthogonalityPenalty;
use gam::terms::analytic_penalties::PsiSlice;

fn make_penalty(
    n_obs: usize,
    latent_dim: usize,
    groups: Vec<Vec<usize>>,
) -> BlockOrthogonalityPenalty {
    let psi_slice = PsiSlice {
        range: 0..(n_obs * latent_dim),
        latent_dim: Some(latent_dim),
    };
    BlockOrthogonalityPenalty::new(psi_slice, groups, 1.0, n_obs, false)
        .expect("BlockOrthogonalityPenalty::new must succeed")
}

fn flatten(m: &Array2<f64>) -> Array1<f64> {
    let n_obs = m.nrows();
    let d = m.ncols();
    let mut out = Array1::<f64>::zeros(n_obs * d);
    for n in 0..n_obs {
        for a in 0..d {
            out[n * d + a] = m[[n, a]];
        }
    }
    out
}

/// Deterministic 3x4 latent: n_obs=3, latent_dim=4, two groups {0,1} and {2,3}.
fn fixture() -> (
    BlockOrthogonalityPenalty,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = 3usize;
    let d = 4usize;
    let groups = vec![vec![0usize, 1], vec![2usize, 3]];
    let penalty = make_penalty(n, d, groups);
    let t = {
        let mut m = Array2::<f64>::zeros((n, d));
        // Deliberately non-orthogonal between blocks so C_{g,h} != 0 and
        // every term in the Hessian fires.
        m[[0, 0]] = 0.9;
        m[[0, 1]] = -0.3;
        m[[0, 2]] = 0.4;
        m[[0, 3]] = 0.1;
        m[[1, 0]] = 0.2;
        m[[1, 1]] = 0.7;
        m[[1, 2]] = -0.5;
        m[[1, 3]] = 0.6;
        m[[2, 0]] = -0.4;
        m[[2, 1]] = 0.2;
        m[[2, 2]] = 0.8;
        m[[2, 3]] = -0.3;
        flatten(&m)
    };
    let v = {
        let mut m = Array2::<f64>::zeros((n, d));
        m[[0, 0]] = 0.1;
        m[[0, 1]] = 0.05;
        m[[0, 2]] = -0.2;
        m[[0, 3]] = 0.3;
        m[[1, 0]] = -0.15;
        m[[1, 1]] = 0.25;
        m[[1, 2]] = 0.1;
        m[[1, 3]] = -0.05;
        m[[2, 0]] = 0.2;
        m[[2, 1]] = -0.1;
        m[[2, 2]] = 0.3;
        m[[2, 3]] = 0.15;
        flatten(&m)
    };
    let rho = Array1::<f64>::zeros(0);
    (penalty, t, v, rho)
}

/// hvp must be linear in v: `hvp(t, α·v) = α · hvp(t, v)`. Pre-fix, the
/// `cross_gram(v, h, g)` contribution scales as α², breaking linearity.
#[test]
fn block_orthogonality_hvp_is_linear_in_v() {
    let (penalty, t, v, rho) = fixture();
    let hv1 = penalty.hvp(t.view(), rho.view(), v.view());
    let scale = 0.37_f64;
    let v_scaled: Array1<f64> = v.iter().map(|&x| scale * x).collect();
    let hv_scaled = penalty.hvp(t.view(), rho.view(), v_scaled.view());
    for (a, b) in hv1.iter().zip(hv_scaled.iter()) {
        let expected = scale * a;
        assert!(
            (b - expected).abs() < 1.0e-10,
            "hvp non-linear in v: got {b}, expected {expected} (alpha={scale})"
        );
    }
}

/// hvp(t, 0) must be 0 — the Hessian is a bounded linear operator in v and v=0
/// should give an exact zero. Pre-fix, the `cross_gram(t, h, g)` contribution
/// is independent of v and stays nonzero.
#[test]
fn block_orthogonality_hvp_zero_at_zero_direction() {
    let (penalty, t, _v, rho) = fixture();
    let zero = Array1::<f64>::zeros(t.len());
    let hv0 = penalty.hvp(t.view(), rho.view(), zero.view());
    for (i, &val) in hv0.iter().enumerate() {
        assert!(
            val.abs() < 1.0e-12,
            "hvp(t, 0)[{i}] = {val} should be 0 (the Hessian is a linear operator in v)"
        );
    }
}

/// The materialised Hessian must be symmetric: `H[i, j] == H[j, i]`. Pre-fix
/// the `as_dense` path (which calls `hvp` with basis vectors) produced a
/// non-symmetric matrix because the bogus `(v⊗v) + (t⊗t)` term polluted each
/// column asymmetrically.
#[test]
fn block_orthogonality_dense_hessian_is_symmetric() {
    let (penalty, t, _v, rho) = fixture();
    let dense = penalty.as_dense(t.view(), rho.view());
    let n = t.len();
    assert_eq!(dense.dim(), (n, n));
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (dense[[i, j]] - dense[[j, i]]).abs();
            if diff > max_asym {
                max_asym = diff;
            }
        }
    }
    assert!(
        max_asym < 1.0e-9,
        "BlockOrthogonality Hessian is not symmetric: max |H[i,j]-H[j,i]| = {max_asym:.3e}"
    );
}

/// Finite-difference check: `hvp(t, v) ≈ (grad(t + ε v) - grad(t - ε v)) / (2ε)`.
/// Pre-fix the bogus second-derivative term breaks this check by O(1).
#[test]
fn block_orthogonality_hvp_matches_finite_difference_gradient() {
    let (penalty, t, v, rho) = fixture();
    let hv = penalty.hvp(t.view(), rho.view(), v.view());

    let eps = 1.0e-5_f64;
    let t_plus: Array1<f64> = t
        .iter()
        .zip(v.iter())
        .map(|(&ti, &vi)| ti + eps * vi)
        .collect();
    let t_minus: Array1<f64> = t
        .iter()
        .zip(v.iter())
        .map(|(&ti, &vi)| ti - eps * vi)
        .collect();
    let g_plus = penalty.grad_target(t_plus.view(), rho.view());
    let g_minus = penalty.grad_target(t_minus.view(), rho.view());
    let fd: Array1<f64> = g_plus
        .iter()
        .zip(g_minus.iter())
        .map(|(&p, &m)| (p - m) / (2.0 * eps))
        .collect();

    let mut max_err = 0.0_f64;
    for (a, b) in hv.iter().zip(fd.iter()) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // O(eps^2) central-difference error on a smooth quartic-in-t penalty: well
    // below 1e-6.
    assert!(
        max_err < 1.0e-6,
        "hvp(t,v) disagrees with FD grad/dt v: max err = {max_err:.3e}"
    );
}
