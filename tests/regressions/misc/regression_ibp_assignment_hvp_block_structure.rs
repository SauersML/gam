//! Regression guard (companion to the #809 fix) attacking the
//! `OrderedBetaBernoulliPenalty::hvp` contract from the *operator-structure* side
//! rather than a single `hvp == FD(grad)·v` point check.
//!
//! After #809, `hvp` returns the exact `∂²P·v`. The penalty's Hessian has a
//! specific shape the fix must reproduce: because every per-column quantity
//! (`active_mass[k]`, `π[k]`) aggregates over all rows of column `k` and nothing
//! couples distinct columns, the Hessian is **block-diagonal per column** and,
//! within each column, a **rank-1 perturbation of a diagonal**:
//!
//! ```text
//!   H[(i,k),(j,k′)] = 0                                   for k ≠ k′
//!   H[(i,k),(j,k)]  = w·score_derivative[k]·zt[i,k]·zt[j,k]      (rank-1)
//!                   + δ_ij · w·score[k]·z(1−z)(1−2z)/τ²          (diagonal)
//! ```
//!
//! This test materializes the full Hessian from a central difference of the
//! (independently-correct) `grad_target` and pins four structural invariants the
//! diagonal-only default violated and the fix must satisfy:
//!
//!   1. `hvp(·, e_q)` reproduces column `q` of the finite-difference Hessian for
//!      every basis vector `q` — i.e. `hvp` is the genuine operator `H`, not just
//!      correct on one probe direction.
//!   2. The assembled operator is symmetric (`H = Hᵀ`).
//!   3. Cross-column entries are exactly zero (block-diagonal per column).
//!   4. The on-row diagonal `hvp(·, e_j)[j]` agrees with `hessian_diag[j]`
//!      bit-for-bit — the diagonal accessor stays consistent with the full
//!      operator (the fix must not perturb the already-correct diagonal).
//!
//! A fifth check covers the clamped-π degenerate regime: at extreme target
//! magnitudes the column aggregate saturates (`pi_jac = 0 ⇒ score_derivative=0`),
//! the rank-1 block vanishes, and `hvp` must collapse onto `diag(H)·v`.
//!
//! Related: #809 (this fix), #810 (sibling DecoderIncoherencePenalty hvp dropped
//! the residual cross term).

use gam::terms::analytic_penalties::{AnalyticPenalty, OrderedBetaBernoulliPenalty};
use ndarray::{Array1, Array2};

/// Central-difference Hessian: column `q` is `(grad(t+h e_q) − grad(t−h e_q))/2h`.
fn fd_hessian(
    penalty: &OrderedBetaBernoulliPenalty,
    target: &Array1<f64>,
    rho: &Array1<f64>,
    h: f64,
) -> Array2<f64> {
    let n = target.len();
    let mut hess = Array2::<f64>::zeros((n, n));
    for q in 0..n {
        let mut tp = target.clone();
        let mut tm = target.clone();
        tp[q] += h;
        tm[q] -= h;
        let gp = penalty.grad_target(tp.view(), rho.view());
        let gm = penalty.grad_target(tm.view(), rho.view());
        for p in 0..n {
            hess[[p, q]] = (gp[p] - gm[p]) / (2.0 * h);
        }
    }
    hess
}

#[test]
fn ordered_beta_bernoulli_hvp_reproduces_full_hessian_and_is_block_diagonal() {
    let k_max = 3usize;
    let penalty = OrderedBetaBernoulliPenalty::new(k_max, 1.0, 1.0, false);
    let rho = Array1::<f64>::zeros(0);
    let n_rows = 4usize;
    let total = n_rows * k_max;

    // Moderate magnitudes -> π and z stay off their clamp guards.
    let target = Array1::from(vec![
        0.3, -0.4, 0.7, //
        -0.2, 0.5, -0.6, //
        0.1, 0.8, -0.3, //
        -0.5, 0.2, 0.4, //
    ]);

    let h = 1e-5;
    let hess = fd_hessian(&penalty, &target, &rho, h);

    // (1) hvp(·, e_q) == column q of H, for every q.
    let mut max_col_err = 0.0_f64;
    for q in 0..total {
        let mut e = Array1::<f64>::zeros(total);
        e[q] = 1.0;
        let hv = penalty.hvp(target.view(), rho.view(), e.view());
        for p in 0..total {
            max_col_err = max_col_err.max((hv[p] - hess[[p, q]]).abs());
        }
    }
    assert!(
        max_col_err < 1e-5,
        "hvp must reproduce every column of the finite-difference Hessian; \
         max column error = {max_col_err:.3e}"
    );

    // (2) Symmetry of the assembled hvp operator.
    let mut hv_mat = Array2::<f64>::zeros((total, total));
    for q in 0..total {
        let mut e = Array1::<f64>::zeros(total);
        e[q] = 1.0;
        let col = penalty.hvp(target.view(), rho.view(), e.view());
        for p in 0..total {
            hv_mat[[p, q]] = col[p];
        }
    }
    let mut max_asym = 0.0_f64;
    for p in 0..total {
        for q in 0..total {
            max_asym = max_asym.max((hv_mat[[p, q]] - hv_mat[[q, p]]).abs());
        }
    }
    assert!(
        max_asym < 1e-12,
        "hvp operator must be symmetric: max|H-Hᵀ| = {max_asym:.3e}"
    );

    // (3) Cross-column entries are exactly zero (block-diagonal per column).
    let mut max_cross = 0.0_f64;
    for ip in 0..n_rows {
        for kp in 0..k_max {
            for iq in 0..n_rows {
                for kq in 0..k_max {
                    if kp != kq {
                        let val = hv_mat[[ip * k_max + kp, iq * k_max + kq]].abs();
                        max_cross = max_cross.max(val);
                    }
                }
            }
        }
    }
    assert!(
        max_cross < 1e-13,
        "cross-column Hessian entries must be exactly zero; max = {max_cross:.3e}"
    );

    // (4) Diagonal of hvp agrees with hessian_diag bit-for-bit.
    let diag = penalty
        .hessian_diag(target.view(), rho.view())
        .expect("OrderedBetaBernoulliPenalty exposes an analytic hessian_diag");
    for j in 0..total {
        assert_eq!(
            hv_mat[[j, j]],
            diag[j],
            "hvp diagonal entry {j} must equal hessian_diag[{j}] exactly"
        );
    }
}

#[test]
fn ordered_beta_bernoulli_hvp_collapses_to_diagonal_when_pi_is_clamped() {
    let k_max = 2usize;
    let penalty = OrderedBetaBernoulliPenalty::new(k_max, 1.0, 1.0, false);
    let rho = Array1::<f64>::zeros(0);
    let n_rows = 4usize;
    let total = n_rows * k_max;

    // Extreme magnitudes drive z -> {0,1} and the column aggregate π to its
    // clamp, freezing it: pi_jac = 0 => score_derivative = 0 => no rank-1 block.
    let target = Array1::from(vec![
        40.0, -40.0, //
        40.0, -40.0, //
        40.0, -40.0, //
        40.0, -40.0, //
    ]);
    let v = Array1::from(vec![
        0.5, -0.3, //
        0.4, 0.1, //
        -0.2, 0.7, //
        0.6, -0.4, //
    ]);

    let hv = penalty.hvp(target.view(), rho.view(), v.view());
    let diag = penalty
        .hessian_diag(target.view(), rho.view())
        .expect("hessian_diag available");
    // In the frozen-π regime the operator is purely diagonal, so hvp == diag ⊙ v.
    let mut max_diff = 0.0_f64;
    for j in 0..total {
        max_diff = max_diff.max((hv[j] - diag[j] * v[j]).abs());
    }
    assert!(
        max_diff < 1e-12,
        "with π clamped the rank-1 block must vanish so hvp == diag(H)·v; \
         max|hvp - diag⊙v| = {max_diff:.3e}"
    );
}
