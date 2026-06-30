//! Issue #388 regression: the `SO(3)` representation JVP must apply the
//! exponential's right-Jacobian `J_r(ω)` to the input perturbation before
//! forming the `[·]×` generator. The bug used the raw `R(ω)·[dω]×`, i.e. it
//! assumed `J_r = I`, which is correct only when `dω ∥ ω` (or `ω = 0`).
//!
//! ## Why this file exists despite the older `jvp_correctness_so2_and_so3` test
//!
//! That test (in `equivariant_lie_atom.rs`) does NOT pin the production fix:
//! it defines its own *local* `rho_so3_jvp` that recomputes the JVP as the raw
//! `ρ(ω)·skew(dω)` — the pre-#388 buggy form — and compares only the Frobenius
//! *norms* of the analytic and finite-difference tangents at a 5% tolerance,
//! with a single one-sided difference along a perturbation parallel to e_x.
//! A norm-only check along a near-parallel direction cannot distinguish the
//! correct `R·[J_r·dω]×` from the buggy `R·[dω]×`, and the local reimplementation
//! never touches `gam::geometry::lie_so::rho_so3_jvp` at all. So reverting the
//! production fix would leave that test green.
//!
//! This file closes the gap: it drives the PRODUCTION `rho_so3_jvp` and checks
//! it elementwise against a central finite difference of the production
//! Rodrigues exponential `rho_so3`, for a `dω` chosen PERPENDICULAR to `ω` (the
//! exact regime where a missing `J_r` produces an O(‖ω‖) error). It also pins
//! that the production JVP genuinely differs from the naive raw-`[dω]×` form for
//! that perpendicular direction, so the right-Jacobian factor cannot be dropped
//! without tripping a test.

use gam::geometry::lie_so::{rho_so3, rho_so3_jvp};
use ndarray::{Array2, array};

/// Central finite difference of the production Rodrigues exponential along a
/// rotation-vector direction `dir`: `[ρ(ω + h·dir) − ρ(ω − h·dir)] / (2h)`.
fn fd_jvp(omega: [f64; 3], dir: [f64; 3], h: f64) -> Array2<f64> {
    let plus = array![[
        omega[0] + h * dir[0],
        omega[1] + h * dir[1],
        omega[2] + h * dir[2],
    ]];
    let minus = array![[
        omega[0] - h * dir[0],
        omega[1] - h * dir[1],
        omega[2] - h * dir[2],
    ]];
    let r_plus = rho_so3(plus.view()).expect("rho_so3 +");
    let r_minus = rho_so3(minus.view()).expect("rho_so3 -");
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = (r_plus[[0, i, j]] - r_minus[[0, i, j]]) / (2.0 * h);
        }
    }
    out
}

/// The naive (pre-#388) JVP form `R(ω)·[dω]×` with no right-Jacobian factor.
/// Used only to PROVE the production JVP is not silently equal to it.
fn naive_raw_skew_jvp(omega: [f64; 3], domega: [f64; 3]) -> Array2<f64> {
    let r = rho_so3(array![[omega[0], omega[1], omega[2]]].view()).expect("rho_so3");
    let kd = array![
        [0.0, -domega[2], domega[1]],
        [domega[2], 0.0, -domega[0]],
        [-domega[1], domega[0], 0.0],
    ];
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += r[[0, i, k]] * kd[[k, j]];
            }
            out[[i, j]] = acc;
        }
    }
    out
}

fn analytic_jvp(omega: [f64; 3], domega: [f64; 3]) -> Array2<f64> {
    let om = array![[omega[0], omega[1], omega[2]]];
    let dw = array![[domega[0], domega[1], domega[2]]];
    let j = rho_so3_jvp(om.view(), dw.view()).expect("rho_so3_jvp");
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for jj in 0..3 {
            out[[i, jj]] = j[[0, i, jj]];
        }
    }
    out
}

/// The production SO(3) JVP equals the central finite difference of the
/// production exponential ELEMENTWISE, for a `dω` PERPENDICULAR to `ω`. This is
/// the regime that a missing right-Jacobian factor gets wrong by O(‖ω‖).
#[test]
fn so3_rep_jvp_matches_central_finite_difference_perpendicular_388() {
    // ω at a substantial angle (‖ω‖ ≈ 0.616, comfortably away from 0 and π).
    let omega: [f64; 3] = [0.2, 0.3, 0.5];
    // dω ⟂ ω: dot([0.2,0.3,0.5],[0.3,-0.4,0.12]) = 0.06 - 0.12 + 0.06 = 0.0.
    let domega: [f64; 3] = [0.3, -0.4, 0.12];
    assert!(
        (omega[0] * domega[0] + omega[1] * domega[1] + omega[2] * domega[2]).abs() < 1e-12,
        "test fixture must keep dω perpendicular to ω"
    );

    let analytic = analytic_jvp(omega, domega);
    let fd = fd_jvp(omega, domega, 1e-6);

    let mut max_abs = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (analytic[[i, j]] - fd[[i, j]]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    assert!(
        max_abs < 1e-6,
        "issue #388: production SO(3) JVP must equal the central finite difference \
         elementwise for a perpendicular dω; max abs entry error = {max_abs:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

/// Guard that the production JVP is NOT the naive raw-`[dω]×` form: for a
/// perpendicular `dω` at this angle the right-Jacobian factor changes the
/// result by an O(1) amount. If someone reverts the production fix to
/// `R·[dω]×`, the FD test above fails AND this divergence collapses to ~0 —
/// this assertion documents the expected, non-trivial gap.
#[test]
fn so3_rep_jvp_is_not_the_raw_skew_form_388() {
    let omega = [0.2, 0.3, 0.5];
    let domega = [0.3, -0.4, 0.12];

    let analytic = analytic_jvp(omega, domega);
    let naive = naive_raw_skew_jvp(omega, domega);

    let mut max_abs = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (analytic[[i, j]] - naive[[i, j]]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    // The right-Jacobian correction is substantial at ‖ω‖ ≈ 0.616; require a
    // clearly non-numerical gap so the factor cannot be silently dropped.
    assert!(
        max_abs > 1e-3,
        "issue #388: the production SO(3) JVP must differ from the raw R·[dω]× \
         form for a perpendicular dω (the right-Jacobian factor must be applied); \
         max abs entry gap = {max_abs:.3e} is too small — has J_r been dropped?"
    );
}

/// Sanity: for `dω ∥ ω` the right-Jacobian acts as the identity on `dω`
/// (`J_r(ω)·ω = ω`), so the production JVP and the raw-skew form must agree, and
/// both must match the finite difference. This pins that the fix did not perturb
/// the already-correct parallel case.
#[test]
fn so3_rep_jvp_parallel_direction_agrees_with_raw_and_fd_388() {
    let omega = [0.2, 0.3, 0.5];
    // dω parallel to ω.
    let domega = [0.2, 0.3, 0.5];

    let analytic = analytic_jvp(omega, domega);
    let naive = naive_raw_skew_jvp(omega, domega);
    let fd = fd_jvp(omega, domega, 1e-6);

    for i in 0..3 {
        for j in 0..3 {
            let a = analytic[[i, j]];
            assert!(
                (a - naive[[i, j]]).abs() < 1e-9,
                "issue #388: parallel dω must give J_r·dω = dω, so production JVP \
                 should equal raw skew form at [{i},{j}]: {a} vs {}",
                naive[[i, j]]
            );
            assert!(
                (a - fd[[i, j]]).abs() < 1e-6,
                "issue #388: parallel-direction JVP must match finite difference at \
                 [{i},{j}]: {a} vs {}",
                fd[[i, j]]
            );
        }
    }
}
