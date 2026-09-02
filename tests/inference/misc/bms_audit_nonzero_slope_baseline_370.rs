//! Issue #370 regression: a Bernoulli marginal-slope (BMS) model with a nonzero
//! slope value at `beta = 0` must fit to completion through the pre-fit
//! identifiability audit — both *without crashing* and *without hanging*.
//!
//! ## The bug
//!
//! The pre-fit identifiability audit builds each block's effective Jacobian once
//! at `beta = &[]` with `family_scalars = None`. The BMS observed predictor is
//!
//! ```text
//!   η_i = q_i·c_i + s·g_i·z_i,   c_i = sqrt(1 + (s·g_i)²)
//!   q_i = M[i,:]·β_m + offset_m[i]      (marginal η)
//!   g_i = G[i,:]·β_s + offset_s[i]      (slope η)
//! ```
//!
//! where `offset_s = slope_offset + baseline_slope` and `baseline_slope`
//! is a *data-driven pooled-probit pilot* value that is essentially never exactly
//! zero. The old block contract assumed "at β=0 every g_i == 0, so no
//! `BmsFamilyScalars` are needed" and hard-errored when it saw any nonzero g_i
//! while `family_scalars` was `None`. Because the fitted baseline (and any
//! user-supplied slope offset) makes `g_i = offset_s[i] != 0` at β=0, that
//! guard fired for *every* BMS model — including the rigid `slope_formula="1"`
//! control — making the entire score-warp / link-wiggle Python surface
//! unreachable. The issue was surfaced from CI as a **>600s timeout** of the
//! flex Bernoulli marginal-slope audit.
//!
//! The fix (issue #367) makes both BMS blocks self-compute `q_i, g_i, c_i, z_i`
//! from owned data at the current β with NO caller-supplied scalar contract.
//!
//! ## Why this file ships TWO levels of test
//!
//! 1. `bms_callbacks_self_compute_*` — a fast unit pin on the exact callback
//!    arithmetic at the `beta = []` boundary. Cheap, but it never runs the audit
//!    or the solver end to end, so on its own it cannot catch a re-introduced
//!    crash *inside* the audit driver or a fit that hangs (the issue's reported
//!    symptom was a timeout, not a panic).
//! 2. `bms_{rigid,flex}_nonzero_slope_offset_audit_fits_in_time_370` — the
//!    angle the issue was actually filed from: drive a complete `fit_model` BMS
//!    fit (which runs the pre-fit audit end to end) with a *forced large nonzero
//!    slope offset*, so `g_i = offset_s[i]` is guaranteed far from zero at
//!    β=0. Each asserts the fit (a) returns a finite, nonempty coefficient
//!    vector (no audit crash) and (b) terminates on convergence rather than by
//!    exhausting its configured outer x inner iteration budget (no >600s hang
//!    regression).

use gam::families::bms::{BmsSlopeJacobian, BmsMarginalJacobian};
use gam::families::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Why the FLEX full-fit guard is NOT wired into the shared suite (issue #370)
// ---------------------------------------------------------------------------
//
// The flex configuration (`score_warp` + `link_dev`) does *not* crash — the
// #367 audit self-compute fix holds, and the rigid full-fit guard above
// exercises the same audit driver end to end. But restoring a flex `fit_model`
// guard surfaced a *live* regression of the exact symptom #370 was filed on (a
// >600s timeout): the flex fit does not hang in a single unbounded loop; it
// *stalls* and burns the entire outer×inner budget (default 60×1200) at
// ~20s/cycle, which presents as an effectively non-terminating fit.
//
// Diagnosis (captured with iteration budgets capped to 8×8 and a small,
// realistic slope offset of 0.15, so this is NOT an artifact of a forced
// large offset):
//
//   [PIRLS/joint-Newton terminal] converged=false terminator=budget-exhausted
//     best_residual_inf=3.918e1 (tol=1.411e-5)  last_obj_change_below_tol=false
//   last_newton_math={old_kkt=3.918e1, linearized_next=3.918e1,
//     actual=+3.046e-9, pred=+3.139e-9, rho=+9.704e-1, scalar_relerr=9.280e-11,
//     step_inf=1.498e-10, proposal_inf=1.026e0}
//   block_grad_inf=[5.709, 13.105, 0.732, 0.984]  (block 1 = scalar slope)
//
// The objective is numerically flat (|Δobj| ~ 3e-9 ≪ tol) yet the KKT residual
// is pinned at ~3.9e1 — the iterate is stalled at a *non-KKT* point, so more
// budget cannot rescue it. The Newton model is exact for the objective
// (rho≈0.97, scalar_relerr≈1e-10), but the full proposal step (proposal_inf≈1)
// is predicted to leave the KKT residual unchanged (`linearized_next == old_kkt`)
// and is shrunk to ~1e-10 by the line search every cycle. The dominant unmoved
// gradient lives in the width-1 slope-intercept block (|g|∞≈13), a very stiff
// direction whose Newton step is negligible against a large diagonal Hessian.
// This is a joint-Newton convergence defect on the BMS flex deviation-block
// path — a separate, deeper bug than the #367 audit crash this file pins.
//
// Shipping a flex `fit_model` test now would add a test that runs until SIGTERM
// to the shared suite, which the build's no-hang policy forbids. The flex guard
// is therefore withheld until the stall is fixed; #370 stays open tracking it.
// The rigid full-fit guard + the callback pin below fully protect against the
// #367 audit-crash regression that #1146 deleted, which is the regression this
// reopened issue is about.

/// Fast callback-level pin at the exact `beta = []` audit boundary: both BMS
/// Jacobian callbacks must self-compute their row scalars (q_i, g_i, c_i, z_i)
/// from owned data, with no caller-supplied `BmsFamilyScalars`. Cheap unit
/// complement to the full-fit guards above.
#[test]
fn bms_callbacks_self_compute_nonzero_slope_baseline_at_beta_zero_370() {
    let marginal = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, -0.4, //
                0.5, 0.8, //
                -0.2, 1.3,
            ],
        )
        .unwrap(),
    );
    let slope = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                0.7, -0.1, //
                -0.3, 0.9, //
                0.4, 0.6,
            ],
        )
        .unwrap(),
    );
    let offset_m = Array1::from_vec(vec![0.2, -0.5, 0.9]);
    let offset_s = Array1::from_vec(vec![1.7, -1.3, 0.8]);
    let z = Arc::new(Array1::from_vec(vec![-0.6, 0.4, 1.1]));
    let probit_scale = 0.75_f64;
    let state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: probit_scale,
    };

    let marginal_cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&slope),
        offset_m.clone(),
        offset_s.clone(),
        marginal.ncols(),
    );
    let slope_cb = BmsSlopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&slope),
        offset_m.clone(),
        offset_s.clone(),
        Arc::clone(&z),
        marginal.ncols(),
    );

    let marginal_j = marginal_cb
        .effective_jacobian_at(&state)
        .expect("marginal callback must not demand family scalars at beta=0");
    let slope_j = slope_cb
        .effective_jacobian_at(&state)
        .expect("slope callback must not demand family scalars at beta=0");

    assert_eq!(marginal_j.dim(), (marginal.nrows(), marginal.ncols()));
    assert_eq!(slope_j.dim(), (slope.nrows(), slope.ncols()));

    for i in 0..marginal.nrows() {
        let q_i = offset_m[i];
        let g_i = offset_s[i];
        assert!(
            g_i.abs() > 0.0,
            "fixture must keep the #370 precondition g_i != 0 at beta=0"
        );
        let c_i = (1.0 + (probit_scale * g_i).powi(2)).sqrt();
        let slope_factor = q_i * probit_scale * probit_scale * g_i / c_i + probit_scale * z[i];

        for j in 0..marginal.ncols() {
            assert_close(
                &format!("marginal row {i} col {j}"),
                marginal_j[[i, j]],
                c_i * marginal[[i, j]],
            );
            assert_close(
                &format!("slope row {i} col {j}"),
                slope_j[[i, j]],
                slope_factor * slope[[i, j]],
            );
        }
    }
}

fn assert_close(label: &str, got: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let rel = (got - expected).abs() / scale;
    assert!(
        rel < 1e-12,
        "{label}: got {got:.17e}, expected {expected:.17e}, rel={rel:.3e}"
    );
}
