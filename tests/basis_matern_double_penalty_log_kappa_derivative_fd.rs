//! Finite-difference verification of the Matérn DOUBLE-PENALTY log-κ
//! ψ-derivatives (#1122).
//!
//! `matern(x1, x2)` defaults to `double_penalty = true`, so its production
//! penalty path is the kernel-Gram double-penalty path — NOT the operator
//! penalty path exercised by `basis_matern_log_kappa_penalty_derivative_fd.rs`
//! (which pins `double_penalty: false`). That double-penalty path emits up to
//! two normalized blocks:
//!   * `Primary`               — the Frobenius-normalized projected kernel Gram
//!     `Zᵀ K Z`, and
//!   * `DoublePenaltyNullspace` — the Frobenius-normalized spectral projector
//!     onto the near-null eigenspace of that Gram.
//!
//! The bug fixed in #1122: the `DoublePenaltyNullspace` block's log-κ
//! ψ-derivative was hard-coded to zero, but the projector rotates with κ (the
//! Gram `Zᵀ K Z` it diagonalizes is κ-dependent). That objective↔gradient
//! desync stalled the isotropic-κ joint REML at its iteration cap with a large
//! residual gradient. This test FD-checks EACH active block's first and second
//! log-κ derivative against a central difference of the forward (normalized)
//! penalty, under `double_penalty: true`, on a low-κ (large length-scale)
//! configuration whose projected Gram is rank-deficient — so the shrinkage
//! block is ACTIVE and the previously-omitted derivative is exercised.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivatives, build_matern_basis_log_kappasecond_derivative,
};
use ndarray::Array2;

/// Evaluation data: a 4×4 grid (16 rows).
fn dataset() -> Array2<f64> {
    let mut rows = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            rows.push(i as f64 / 3.0);
            rows.push(j as f64 / 3.0);
        }
    }
    Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap()
}

/// Centers: a finer 6×6 grid (36 centers) — MORE centers than data rows.
///
/// With `n < k` the Matérn `matern_rank_reduce_centers` RRQR pruning is skipped
/// (it requires `n >= k`), so the value build and the derivative build use the
/// SAME center set and their penalty-block lists stay index-aligned. This also
/// mirrors the production iso-κ regime, where the FrozenTransform pins the
/// centers so no κ-dependent reduction runs during the optimization. At a large
/// length scale the projected kernel Gram `Zᵀ K Z` is numerically rank-deficient,
/// so the `DoublePenaltyNullspace` shrinkage block is emitted.
fn centers() -> Array2<f64> {
    let mut rows = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            rows.push(i as f64 / 5.0);
            rows.push(j as f64 / 5.0);
        }
    }
    Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap()
}

/// Double-penalty Matérn spec at log-κ = `rho` (length_scale = exp(−rho)) with
/// the fixed `UserProvided` center set above.
fn spec_at(rho: f64, nu: MaternNu) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers()),
        periodic: None,
        length_scale: (-rho).exp(),
        nu,
        include_intercept: false,
        double_penalty: true,
        identifiability: Default::default(),
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    }
}

/// Forward, normalized double-penalty blocks at this log-κ. Index-aligned with
/// the ψ-derivative lists.
fn penalties_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> Vec<Array2<f64>> {
    let spec = spec_at(rho, nu);
    build_matern_basis(data.view(), &spec).unwrap().penalties
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

/// Low κ (large length scale) so the projected Gram is rank-deficient and the
/// `DoublePenaltyNullspace` shrinkage block is active.
const RHO: f64 = -1.5;

#[test]
fn matern_double_penalty_shrinkage_block_is_active_and_count_aligned() {
    // Guard the premise: this fixture must actually emit the shrinkage block,
    // otherwise the FD checks below would silently only test `Primary`. Also
    // assert that the ψ-derivative block list is COUNT-aligned with the forward
    // penalty list — a misalignment (e.g. from κ-dependent center reduction)
    // would make the per-block FD comparison meaningless.
    let data = dataset();
    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        let n_blocks = penalties_at(&data, RHO, nu).len();
        assert!(
            n_blocks >= 2,
            "nu={nu:?}: expected an active DoublePenaltyNullspace block (>=2 penalties) \
             at rho={RHO}; got {n_blocks}. If the spectral tolerance changed, retune RHO."
        );
        let spec = spec_at(RHO, nu);
        let deriv_blocks = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .penalties_derivative
            .len();
        assert_eq!(
            deriv_blocks, n_blocks,
            "nu={nu:?}: ψ-derivative block count {deriv_blocks} must equal forward \
             penalty block count {n_blocks}"
        );
    }
}

#[test]
fn matern_double_penalty_log_kappa_first_derivative_matches_finite_difference() {
    let data = dataset();
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
    ] {
        let spec = spec_at(RHO, nu);
        let analytic = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .penalties_derivative;

        let h = 1e-6;
        let plus = penalties_at(&data, RHO + h, nu);
        let minus = penalties_at(&data, RHO - h, nu);
        assert_eq!(
            analytic.len(),
            plus.len(),
            "nu={nu:?}: ψ-derivative block count {} must match forward penalty \
             block count {}",
            analytic.len(),
            plus.len(),
        );

        for (block, da) in analytic.iter().enumerate() {
            let num = (&plus[block] - &minus[block]) / (2.0 * h);
            let err = max_abs(&(da - &num));
            let scale = max_abs(da).max(max_abs(&num)).max(1.0);
            assert!(
                err < 1e-4 * scale,
                "nu={nu:?} double-penalty block {block}: first log-κ derivative must \
                 match finite difference (rel tol 1e-4·scale={:.3e}); got max abs \
                 error {err:.3e} (analytic max {:.3e}, fd max {:.3e})",
                1e-4 * scale,
                max_abs(da),
                max_abs(&num),
            );
        }
    }
}

#[test]
fn matern_double_penalty_log_kappa_second_derivative_matches_finite_difference() {
    let data = dataset();
    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        let spec = spec_at(RHO, nu);
        let analytic = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
            .unwrap()
            .penaltiessecond_derivative;

        let h = 1e-4;
        let plus = penalties_at(&data, RHO + h, nu);
        let mid = penalties_at(&data, RHO, nu);
        let minus = penalties_at(&data, RHO - h, nu);
        assert_eq!(analytic.len(), mid.len());

        for (block, da) in analytic.iter().enumerate() {
            let num = (&plus[block] - 2.0 * &mid[block] + &minus[block]) / (h * h);
            let err = max_abs(&(da - &num));
            let scale = max_abs(da).max(max_abs(&num)).max(1.0);
            assert!(
                err < 5e-2 * scale,
                "nu={nu:?} double-penalty block {block}: second log-κ derivative must \
                 match finite difference (rel tol 5e-2·scale={:.3e}); got max abs \
                 error {err:.3e} (analytic max {:.3e}, fd max {:.3e})",
                5e-2 * scale,
                max_abs(da),
                max_abs(&num),
            );
        }
    }
}
