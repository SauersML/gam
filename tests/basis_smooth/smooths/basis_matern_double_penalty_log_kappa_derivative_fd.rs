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
//!   * `DoublePenaltyNullspace` — the Frobenius-normalized center-function-
//!     metric penalty on the explicit intercept function.
//!
//! The bug fixed in #1122: the `DoublePenaltyNullspace` block's log-κ
//! ψ-derivative was hard-coded to zero, but the function metric moves with κ.
//! That objective↔gradient
//! desync stalled the isotropic-κ joint REML at its iteration cap with a large
//! residual gradient. This test FD-checks EACH active block's first and second
//! log-κ derivative against a central difference of the forward (normalized)
//! penalty under `double_penalty: true` with an explicit intercept, so the
//! structural shrinkage block is reliably active and the previously omitted
//! derivative is genuinely exercised.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivatives, build_matern_basis_log_kappasecond_derivative,
};
use ndarray::Array2;

/// Evaluation data: a 6×6 grid (36 rows). `n = 36 >= k = 28` so the Matérn
/// `matern_rank_reduce_centers` RRQR pruning is a genuine no-op — it only prunes
/// when the realized `n × k` kernel block is column-rank-deficient, and a
/// well-spread 36-row cloud supports all 28 center columns. (An earlier 4×4
/// `n = 16 < k = 28` grid was rank-deficient: RRQR legitimately dropped the
/// near-duplicate center columns, so the near-null eigenspace — and the
/// `DoublePenaltyNullspace` shrinkage block this test exercises — disappeared in
/// the value build while the derivative build still sized to the full center
/// set, the exact desync this fixture must avoid.)
fn dataset() -> Array2<f64> {
    let mut rows = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            rows.push(i as f64 / 5.0);
            rows.push(j as f64 / 5.0);
        }
    }
    Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap()
}

/// Tiny offset (in coordinate units) defining the near-duplicate centers below.
/// At `δ = 1e-5` the near-coincident pair's small eigenvalue of the Matérn Gram
/// is `≈ (3/2)(δ/ℓ)² ≈ 1.5e-10` — well below the shrinkage tolerance
/// `tol = dim·1e-10·λ_max ≈ 2e-8` (≈ 130× margin), yet comfortably above the
/// eigensolver round-off floor `eps·λ_max ≈ 2e-15` (≈ 1e5× margin). So the
/// near-null eigenspace is present with a large, ρ-stable spectral gap (the rank
/// does NOT flicker across `ρ ± h`) and is numerically well-resolved. Crucially
/// `δ ≠ 0`, so `k(δ/ℓ) ≠ 1` and the near-null EIGENVECTOR genuinely rotates with
/// κ — exactly the κ-dependence whose previously-omitted projector derivative
/// stalled the iso-κ REML (#1122). An exact duplicate (`δ = 0`) would give a
/// κ-invariant null (`P′ = 0`) and would not exercise the fix.
const NEAR_DUP_OFFSET: f64 = 1.0e-5;

/// Centers (`UserProvided`): a spread 5×5 base grid (25 distinct, well-spread
/// points so `λ_max` and the bulk spectrum are well-conditioned) PLUS three
/// near-duplicate points, each an existing grid point shifted by
/// [`NEAR_DUP_OFFSET`]. The three near-coincidences put a STABLE rank-3 near-null
/// eigenspace in the projected kernel Gram, so the `DoublePenaltyNullspace`
/// shrinkage block is reliably emitted at any moderate κ.
///
/// `k = 28` with `n = 36 >= k` (see [`dataset`]): the realized `n × k` Matérn
/// kernel block is full column rank, so `matern_rank_reduce_centers` keeps every
/// center. The value build and the derivative build therefore use the SAME
/// (un-reduced) center set, keeping their penalty-block lists index-aligned and
/// the near-duplicate-induced near-null eigenspace intact. This also mirrors the
/// production iso-κ regime, where the FrozenTransform pins the centers so no
/// κ-dependent reduction runs during the optimization.
fn centers() -> Array2<f64> {
    let mut rows = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            rows.push(i as f64 / 4.0);
            rows.push(j as f64 / 4.0);
        }
    }
    // Three near-duplicates of distinct base grid points.
    for &(bi, bj) in &[(0usize, 0usize), (2, 3), (4, 1)] {
        rows.push(bi as f64 / 4.0 + NEAR_DUP_OFFSET);
        rows.push(bj as f64 / 4.0);
    }
    Array2::from_shape_vec((rows.len() / 2, 2), rows).unwrap()
}

/// Double-penalty Matérn spec at log-κ = `rho` (length_scale = exp(−rho)) with
/// the fixed `UserProvided` center set above.
fn spec_at(rho: f64, nu: MaternNu) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers()),
        periodic: None,
        length_scale: gam::terms::basis::MaternLengthScale::fixed((-rho).exp()),
        nu,
        include_intercept: true,
        double_penalty: true,
        identifiability: Default::default(),
        aniso_log_scales: None,
    }
}

/// Forward, normalized double-penalty blocks at this log-κ. Index-aligned with
/// the ψ-derivative lists.
fn penalties_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> Vec<Array2<f64>> {
    let spec = spec_at(rho, nu);
    build_matern_basis(data.view(), &spec)
        .unwrap()
        .active_penalties
        .into_iter()
        .map(|p| p.matrix)
        .collect()
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

/// Moderate κ (ℓ = 1), away from radial underflow and overflow.
const RHO: f64 = 0.0;

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
             at rho={RHO}; got {n_blocks}. The explicit intercept must define this topology."
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
