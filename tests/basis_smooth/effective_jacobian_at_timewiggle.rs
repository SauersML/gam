//! Numeric verification that the timewiggle-active Jacobians
//! (`SmsTimewiggleTimeJacobian` and `SmsTimewiggleMarginalJacobian`) match
//! central-difference finite differences.
//!
//! Setup: n=6 rows, p_base=2 time-base columns, p_tw=3 wiggle columns
//! (degree-3 monotone I-spline from an 8-knot clamped vector — the minimal
//! valid degree-3 wiggle), p_m=3 marginal columns, p_g=1 logslope column.
//! The joint β is [β_t_base (2), β_tw (3), β_m (3), β_g (1)] = 9 entries.
//!
//! We evaluate the Jacobian at a non-trivial β with β_tw ≠ 0, so the
//! timewiggle corrections are genuinely active.  The logslope β is set to
//! zero so c_i = 1 and the ground-truth per-row (η0, η1, ad1) can be
//! computed directly from the q-values.

use gam::families::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use gam::families::survival::marginal_slope::{
    SmsTimewiggleMarginalJacobian, SmsTimewiggleTimeJacobian,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// ── Test geometry ────────────────────────────────────────────────────────────

const N: usize = 6;
const P_BASE: usize = 2; // non-wiggle time columns
const P_TW: usize = 3; // wiggle tail columns (degree-3 I-spline, 8 clamped knots)
const P_TIME: usize = P_BASE + P_TW; // 5
const P_M: usize = 3; // marginal columns
const P_G: usize = 1; // logslope columns
const P_JOINT: usize = P_TIME + P_M + P_G; // 9

/// Dense n × p_base entry design.
fn design_entry() -> Array2<f64> {
    let data: Vec<f64> = (0..N * P_BASE)
        .map(|k| 0.3 + 0.1 * (k as f64).sin())
        .collect();
    Array2::from_shape_vec((N, P_BASE), data).unwrap()
}

/// Dense n × p_base exit design.
fn design_exit() -> Array2<f64> {
    let data: Vec<f64> = (0..N * P_BASE)
        .map(|k| 0.5 + 0.2 * (k as f64 + 1.0).cos())
        .collect();
    Array2::from_shape_vec((N, P_BASE), data).unwrap()
}

/// Dense n × p_base derivative design.
fn design_deriv() -> Array2<f64> {
    let data: Vec<f64> = (0..N * P_BASE)
        .map(|k| 0.8 + 0.05 * ((k as f64) * 1.7).sin())
        .collect();
    Array2::from_shape_vec((N, P_BASE), data).unwrap()
}

/// Dense n × p_m marginal design.
fn design_marginal() -> Array2<f64> {
    let data: Vec<f64> = (0..N * P_M).map(|k| 0.1 * (k as f64 + 2.0)).collect();
    Array2::from_shape_vec((N, P_M), data).unwrap()
}

/// Dense n × p_g logslope design.
fn design_logslope() -> Array2<f64> {
    let data: Vec<f64> = (0..N * P_G).map(|k| 0.5 + 0.1 * k as f64).collect();
    Array2::from_shape_vec((N, P_G), data).unwrap()
}

/// Per-row offsets (all three: entry, exit, deriv).
fn offsets() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let entry = Array1::from_iter((0..N).map(|i| 0.2 + 0.05 * i as f64));
    let exit = Array1::from_iter((0..N).map(|i| 0.4 + 0.07 * i as f64));
    let deriv = Array1::from_iter((0..N).map(|i| 0.9 + 0.03 * i as f64));
    (entry, exit, deriv)
}

fn marginal_offset() -> Array1<f64> {
    Array1::zeros(N)
}

/// Clamped knot vector for the degree-3 monotone (I-spline) wiggle basis.
///
/// The monotone wiggle integrates a degree-`internal_degree = degree - 1 = 2`
/// M-spline into a degree-3 value basis, so the low-level builder validates the
/// knots against `bs_degree = internal_degree + 1 = 3` and requires a *clamped*
/// vector of at least `2·(bs_degree + 1) = 8` knots. With `len` knots the basis
/// has `len - bs_degree - 1 - 1 = len - 5` I-spline columns, so the minimal
/// valid degree-3 wiggle (no internal knots) is an 8-knot clamped vector giving
/// `P_TW = 3` columns. The earlier `[-3, 3]` two-knot vector was not a valid
/// clamped knot vector and could never yield a 2-column degree-3 wiggle
/// (production rejected it with "Insufficient knots for degree 3 spline").
fn wiggle_knots() -> Array1<f64> {
    // Clamped degree-3 boundary knots (multiplicity bs_degree + 1 = 4 at each
    // end), no internal knots ⇒ len = 8 ⇒ P_TW = len - 5 = 3 I-spline columns.
    Array1::from_vec(vec![-3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, 3.0])
}

const WIGGLE_DEGREE: usize = 3;

/// Non-trivial joint β with β_tw ≠ 0.
fn beta_nonzero() -> Array1<f64> {
    Array1::from_vec(vec![
        // β_t_base (2): small nonzero entries
        0.15, -0.07,
        // β_tw (3): positive (wiggle monotone constraint requires ≥ 0,
        // but we just need a valid input for the Jacobian)
        0.20, 0.10, 0.05, // β_m (3)
        0.05, -0.12, 0.08, // β_g (1): set to 0 so c_i = 1 (simplifies ground truth)
        0.0,
    ])
}

// ── Ground-truth (η0, η1, ad1) per row ──────────────────────────────────────

/// Compute the timewiggle-transformed q from h and β_tw via the monotone
/// wiggle basis.  Returns q = h + B(h) · β_tw.
fn compute_q(h: f64, beta_tw: &[f64]) -> f64 {
    let knots = wiggle_knots();
    let h_arr = Array1::from_vec(vec![h]);
    let basis = gam::families::wiggle::monotone_wiggle_basis_with_derivative_order(
        h_arr.view(),
        &knots,
        WIGGLE_DEGREE,
        0,
    )
    .expect("basis");
    // basis is (1, p_tw); β_tw is (p_tw,)
    let correction: f64 = (0..beta_tw.len().min(basis.ncols()))
        .map(|j| basis[[0, j]] * beta_tw[j])
        .sum();
    h + correction
}

/// Evaluate (η0_i, η1_i, ad1_i) for observation i at the given joint β.
/// With β_g = 0: c_i = 1, so η_r = q_r.
fn row_eta(i: usize, beta: &[f64]) -> (f64, f64, f64) {
    let de = design_entry();
    let dx = design_exit();
    let dd = design_deriv();
    let dm = design_marginal();
    let (oe, ox, od) = offsets();

    let beta_t_base = &beta[..P_BASE];
    let beta_tw = &beta[P_BASE..P_TIME];
    let beta_m = &beta[P_TIME..P_TIME + P_M];
    let beta_g = &beta[P_TIME + P_M..P_TIME + P_M + P_G];

    // c_i from β_g (= 1 here since β_g = 0)
    let g_i: f64 = (0..P_G)
        .map(|j| design_logslope()[[i, j]] * beta_g[j])
        .sum();
    let s = 1.0_f64; // probit_scale
    let c_i = (1.0_f64 + (s * g_i).powi(2)).sqrt();

    let eta_m: f64 = (0..P_M).map(|j| dm[[i, j]] * beta_m[j]).sum();

    let h0_base: f64 = oe[i]
        + eta_m
        + (0..P_BASE)
            .map(|j| de[[i, j]] * beta_t_base[j])
            .sum::<f64>();
    let h1_base: f64 = ox[i]
        + eta_m
        + (0..P_BASE)
            .map(|j| dx[[i, j]] * beta_t_base[j])
            .sum::<f64>();
    let d_raw: f64 = od[i]
        + (0..P_BASE)
            .map(|j| dd[[i, j]] * beta_t_base[j])
            .sum::<f64>();

    let q0 = compute_q(h0_base, beta_tw);
    let q1 = compute_q(h1_base, beta_tw);
    // qd1 = dq/dh(h1) * d_raw
    let knots = wiggle_knots();
    let h1_arr = Array1::from_vec(vec![h1_base]);
    let basis_d1 = gam::families::wiggle::monotone_wiggle_basis_with_derivative_order(
        h1_arr.view(),
        &knots,
        WIGGLE_DEGREE,
        1,
    )
    .expect("basis_d1");
    let dq_dq0: f64 = 1.0
        + (0..P_TW.min(basis_d1.ncols()))
            .map(|j| basis_d1[[0, j]] * beta_tw[j])
            .sum::<f64>();
    let qd1 = dq_dq0 * d_raw;

    (c_i * q0, c_i * q1, c_i * qd1)
}

/// Central-difference Jacobian of [η0_0,…,η0_{n-1}, η1_0,…,η1_{n-1},
/// ad1_0,…,ad1_{n-1}] w.r.t. the slice `beta_full[block_start..block_start+p_block]`.
fn numerical_jacobian(beta_full: &[f64], block_start: usize, p_block: usize) -> Array2<f64> {
    let eps = 1e-6;
    let mut jac = Array2::<f64>::zeros((3 * N, p_block));
    let mut beta_p = beta_full.to_vec();
    let mut beta_m = beta_full.to_vec();
    for j in 0..p_block {
        beta_p[block_start + j] += eps;
        beta_m[block_start + j] -= eps;
        for i in 0..N {
            let (e0p, e1p, ad1p) = row_eta(i, &beta_p);
            let (e0m, e1m, ad1m) = row_eta(i, &beta_m);
            jac[[i, j]] = (e0p - e0m) / (2.0 * eps);
            jac[[N + i, j]] = (e1p - e1m) / (2.0 * eps);
            jac[[2 * N + i, j]] = (ad1p - ad1m) / (2.0 * eps);
        }
        beta_p[block_start + j] -= eps;
        beta_m[block_start + j] += eps;
    }
    jac
}

// ── Helpers to build the jacobian structs ────────────────────────────────────

fn make_time_jac() -> SmsTimewiggleTimeJacobian {
    // Build full-width time designs (n × p_time = n × (p_base + p_tw)).
    // For the full time block we need n × p_time matrices.  The base part
    // comes from design_entry/exit/deriv; the wiggle columns are zero (the
    // wiggle tail is part of the time block design but the actual wiggle
    // transform is applied through the B-spline, not through a design column).
    // In the real production code the time block design only has base columns
    // and the wiggle tail; here we assemble n × p_time matrices by appending
    // zero columns for the wiggle tail positions (the wiggle tail of β_t
    // corresponds to the B-spline amplitudes, which enter through the basis
    // rather than the raw X rows).
    let n = N;
    let de_base = design_entry();
    let dx_base = design_exit();
    let dd_base = design_deriv();

    // Full-width matrices: pad base columns with zeros for the wiggle tail.
    let mut de = Array2::<f64>::zeros((n, P_TIME));
    let mut dx = Array2::<f64>::zeros((n, P_TIME));
    let mut dd = Array2::<f64>::zeros((n, P_TIME));
    for i in 0..n {
        for j in 0..P_BASE {
            de[[i, j]] = de_base[[i, j]];
            dx[[i, j]] = dx_base[[i, j]];
            dd[[i, j]] = dd_base[[i, j]];
        }
        // Columns P_BASE..P_TIME stay zero — wiggle tail amplitudes enter
        // only via the B-spline composition, not a linear design column.
    }

    let dm = design_marginal();
    let dg = design_logslope();
    let (oe, ox, od) = offsets();
    let mo = marginal_offset();

    SmsTimewiggleTimeJacobian::new(
        Arc::new(de),
        Arc::new(dx),
        Arc::new(dd),
        Arc::new(dm),
        Arc::new(dg),
        Arc::new(oe),
        Arc::new(ox),
        Arc::new(od),
        Arc::new(mo),
        wiggle_knots(),
        WIGGLE_DEGREE,
        P_TW,
        P_M,
        P_G,
        1.0, // probit_scale = 1
    )
}

fn make_marginal_jac() -> SmsTimewiggleMarginalJacobian {
    let n = N;
    let de_base = design_entry();
    let dx_base = design_exit();
    let dd_base = design_deriv();

    let mut de = Array2::<f64>::zeros((n, P_TIME));
    let mut dx = Array2::<f64>::zeros((n, P_TIME));
    let mut dd = Array2::<f64>::zeros((n, P_TIME));
    for i in 0..n {
        for j in 0..P_BASE {
            de[[i, j]] = de_base[[i, j]];
            dx[[i, j]] = dx_base[[i, j]];
            dd[[i, j]] = dd_base[[i, j]];
        }
    }

    let dm = design_marginal();
    let dg = design_logslope();
    let (oe, ox, od) = offsets();
    let mo = marginal_offset();

    SmsTimewiggleMarginalJacobian::new(
        Arc::new(de),
        Arc::new(dx),
        Arc::new(dd),
        Arc::new(dm),
        Arc::new(dg),
        Arc::new(oe),
        Arc::new(ox),
        Arc::new(od),
        Arc::new(mo),
        wiggle_knots(),
        WIGGLE_DEGREE,
        P_TIME,
        P_TW,
        P_G,
        1.0,
    )
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn time_jacobian_at_zero_beta_matches_rigid() {
    // At β = 0: dq_dq0 = 1, c_i = 1.
    // Time-block Jacobian rows: J[i,j] = X_entry[i,j], J[n+i,j] = X_exit[i,j],
    //                            J[2n+i,j] = X_deriv[i,j]   (for j < p_base only;
    //                            wiggle columns → B_k(h0) at h0 = offset_entry, but
    //                            with β_tw=0 the basis does not enter the Jacobian
    //                            since the basis values B_k(h0) are not zero in general.
    // The test here just checks the callback runs without error and produces
    // the expected (3*n, p_time) shape.
    let jac_cb = make_time_jac();
    let zeros = vec![0.0f64; P_JOINT];
    let state = FamilyLinearizationState {
        beta: &zeros,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = jac_cb
        .effective_jacobian_at(&state)
        .expect("time jacobian at zero");
    assert_eq!(
        jac.dim(),
        (3 * N, P_TIME),
        "time Jacobian shape mismatch at β=0"
    );
    // At β=0: base columns j < P_BASE should match X_entry/X_exit/X_deriv.
    let de = design_entry();
    let dx = design_exit();
    let dd = design_deriv();
    for i in 0..N {
        for j in 0..P_BASE {
            approx::assert_abs_diff_eq!(jac[[i, j]], de[[i, j]], epsilon = 1e-12);
            approx::assert_abs_diff_eq!(jac[[N + i, j]], dx[[i, j]], epsilon = 1e-12);
            approx::assert_abs_diff_eq!(jac[[2 * N + i, j]], dd[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn marginal_jacobian_at_zero_beta_matches_rigid() {
    let jac_cb = make_marginal_jac();
    let zeros = vec![0.0f64; P_JOINT];
    let state = FamilyLinearizationState {
        beta: &zeros,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = jac_cb
        .effective_jacobian_at(&state)
        .expect("marginal jacobian at zero");
    assert_eq!(
        jac.dim(),
        (3 * N, P_M),
        "marginal Jacobian shape mismatch at β=0"
    );
    // At β=0: J[i,j] = J[n+i,j] = M[i,j], J[2n+i,j] = 0.
    let dm = design_marginal();
    for i in 0..N {
        for j in 0..P_M {
            approx::assert_abs_diff_eq!(jac[[i, j]], dm[[i, j]], epsilon = 1e-12);
            approx::assert_abs_diff_eq!(jac[[N + i, j]], dm[[i, j]], epsilon = 1e-12);
            approx::assert_abs_diff_eq!(jac[[2 * N + i, j]], 0.0, epsilon = 1e-12);
        }
    }
}

#[test]
fn time_jacobian_matches_finite_difference_at_nonzero_beta() {
    let beta = beta_nonzero();
    let jac_cb = make_time_jac();
    let state = FamilyLinearizationState {
        beta: beta.as_slice().unwrap(),
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let analytic = jac_cb
        .effective_jacobian_at(&state)
        .expect("time jacobian at nonzero beta");

    assert_eq!(
        analytic.dim(),
        (3 * N, P_TIME),
        "time Jacobian shape at non-zero β"
    );

    let numeric = numerical_jacobian(beta.as_slice().unwrap(), 0, P_TIME);

    for channel in 0..3 {
        for i in 0..N {
            for j in 0..P_TIME {
                let a = analytic[[channel * N + i, j]];
                let n = numeric[[channel * N + i, j]];
                let tol = 1e-5 * n.abs().max(1e-6);
                assert!(
                    (a - n).abs() <= tol,
                    "time Jacobian mismatch channel={channel} row={i} col={j}: \
                     analytic={a:.8e} numeric={n:.8e} diff={:.3e}",
                    (a - n).abs()
                );
            }
        }
    }
}

#[test]
fn marginal_jacobian_matches_finite_difference_at_nonzero_beta() {
    let beta = beta_nonzero();
    let jac_cb = make_marginal_jac();
    let state = FamilyLinearizationState {
        beta: beta.as_slice().unwrap(),
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let analytic = jac_cb
        .effective_jacobian_at(&state)
        .expect("marginal jacobian at nonzero beta");

    assert_eq!(
        analytic.dim(),
        (3 * N, P_M),
        "marginal Jacobian shape at non-zero β"
    );

    let numeric = numerical_jacobian(beta.as_slice().unwrap(), P_TIME, P_M);

    for channel in 0..3 {
        for i in 0..N {
            for j in 0..P_M {
                let a = analytic[[channel * N + i, j]];
                let n = numeric[[channel * N + i, j]];
                let tol = 1e-5 * n.abs().max(1e-6);
                assert!(
                    (a - n).abs() <= tol,
                    "marginal Jacobian mismatch channel={channel} row={i} col={j}: \
                     analytic={a:.8e} numeric={n:.8e} diff={:.3e}",
                    (a - n).abs()
                );
            }
        }
    }
}

#[test]
fn n_outputs_is_three() {
    use gam::families::custom_family::BlockEffectiveJacobian;
    let time_jac = make_time_jac();
    let marg_jac = make_marginal_jac();
    assert_eq!(time_jac.n_outputs(), 3, "time block n_outputs");
    assert_eq!(marg_jac.n_outputs(), 3, "marginal block n_outputs");
}
