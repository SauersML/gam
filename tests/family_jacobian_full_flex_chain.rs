//! Finite-difference regression test for the full flex-chain Jacobians.
//!
//! Exercises `LogslopeFlexBlockJacobian`, `MarginalFlexBlockJacobian`,
//! `TimeFlexBlockJacobian`, `ScoreWarpFlexBlockJacobian`, and
//! `LinkDevFlexBlockJacobian` at the rigid initialisation point (β_h = β_w = 0,
//! β_g ≠ 0) where the flex chain reduces to the rigid hyperbolic formula and
//! all derivatives can be computed analytically.
//!
//! At β_h = β_w = 0:
//!
//! - `eta_u[q0] = φ(q0) / D`, `eta_u[q1] = φ(q1) / D` (from the calibration)
//! - The rigid path gives `η = q·c + s·g·z` so `∂η/∂q = c`, `∂η/∂g = q·s²g/c + s·z`.
//!
//! Since the rigid and IFT-exact paths agree at this point (to machine precision),
//! we can populate `SurvivalFlexFamilyScalars` with the rigid formula values and
//! verify that all 6-output Jacobians match FD on the rigid formulas.

use gam::families::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use gam::families::survival_marginal_slope::{
    LinkDevFlexBlockJacobian, LogslopeFlexBlockJacobian, MarginalFlexBlockJacobian,
    ScoreWarpFlexBlockJacobian, SurvivalFlexFamilyScalars, TimeFlexBlockJacobian,
};
use ndarray::Array2;
use std::sync::Arc;

const N: usize = 16;
const FD_EPS: f64 = 1e-7;
const REL_TOL: f64 = 5e-5;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*seed >> 33) as u32) as f64) / (u32::MAX as f64) * 2.0 - 1.0
}

fn make_design(n: usize, p: usize, seed: &mut u64) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |_| lcg(seed))
}

/// Build rigid (β_h=β_w=0) SurvivalFlexFamilyScalars from analytic formulas.
///
/// At rigid init: η0 = q0·c + s·g·z, η1 = q1·c + s·g·z, D = c·φ(q), χ = c/D.
/// So: eta_u[q0] = c (from ∂η0/∂q0 = c), eta_u[q1] = 0 for entry, etc.
/// For a single-timepoint approximation we use the observed-eta approach:
///
/// - entry: primary coords (q0=0, q1=1, qd1=2, g=3); `p_primary = 4`.
/// - `eta_u_entry[q0] = chi_entry = c_i` (rigid, no cross terms)
/// - `eta_u_exit[q1] = c_i`, all others zero for q-channels
/// - `eta_u_entry[g] = q0·s²g/c + s·z`
/// - `eta_u_exit[g]  = q1·s²g/c + s·z`
/// - chi1 = c_i (rigid), chi_u[q1] = 0, chi_u[g] = q1·s²g/c (= ∂c/∂g · 1)
/// - D1 = c_i (rigid D = χ for rigid), d_u[q1] = 0, d_u[g] = s²g/c (= dc/dg)
fn build_rigid_flex_scalars(
    n: usize,
    q0: &[f64],
    q1: &[f64],
    qd1: &[f64],
    g: &[f64],
    z: &[f64],
    s: f64,
    dq0_time: &Array2<f64>,
    dq1_time: &Array2<f64>,
    dqd1_time: &Array2<f64>,
    dq0_marginal: &Array2<f64>,
    dq1_marginal: &Array2<f64>,
    dqd1_marginal: &Array2<f64>,
    logslope_design: Array2<f64>,
) -> SurvivalFlexFamilyScalars {
    let p_primary = 4usize;

    let mut eta_u_entry = Array2::<f64>::zeros((n, p_primary));
    let mut eta_u_exit = Array2::<f64>::zeros((n, p_primary));
    let chi_exit: Vec<f64> = (0..n)
        .map(|i| (1.0 + (s * g[i]).powi(2)).sqrt())
        .collect();
    let mut chi_u_exit = Array2::<f64>::zeros((n, p_primary));
    let d_exit: Vec<f64> = chi_exit.clone();
    let mut d_u_exit = Array2::<f64>::zeros((n, p_primary));

    for i in 0..n {
        let gi = g[i];
        let c = (1.0 + (s * gi).powi(2)).sqrt();
        let sg_over_c = if gi == 0.0 { 0.0 } else { s * s * gi / c };
        let dc_dg = sg_over_c; // dc/dg = s²g/c

        // Entry: η0 = q0·c + s·g·z; ∂η0/∂q0 = c, ∂η0/∂g = q0·s²g/c + s·z
        // q0_idx=0, q1_idx=1, qd1_idx=2, g_idx=3
        eta_u_entry[[i, 0]] = c; // ∂η0/∂q0
        eta_u_entry[[i, 3]] = q0[i] * sg_over_c + s * z[i]; // ∂η0/∂g

        // Exit: η1 = q1·c + s·g·z; ∂η1/∂q1 = c, ∂η1/∂g = q1·s²g/c + s·z
        eta_u_exit[[i, 1]] = c; // ∂η1/∂q1
        eta_u_exit[[i, 3]] = q1[i] * sg_over_c + s * z[i]; // ∂η1/∂g

        // χ1 = c (rigid); ∂χ1/∂g = dc/dg = s²g/c
        chi_u_exit[[i, 3]] = dc_dg;

        // D1 = c (rigid); ∂D1/∂g = dc/dg
        d_u_exit[[i, 3]] = dc_dg;
    }

    let sw_design = Array2::<f64>::zeros((n, 0));
    let ld_design = Array2::<f64>::zeros((n, 0));

    SurvivalFlexFamilyScalars {
        eta_u_entry,
        eta_u_exit,
        chi_exit,
        chi_u_exit,
        d_exit,
        d_u_exit,
        q1_i: q1.to_vec(),
        qd1_i: qd1.to_vec(),
        dq0_time: dq0_time.clone(),
        dq1_time: dq1_time.clone(),
        dqd1_time: dqd1_time.clone(),
        dq0_marginal: dq0_marginal.clone(),
        dq1_marginal: dq1_marginal.clone(),
        dqd1_marginal: dqd1_marginal.clone(),
        logslope_design,
        score_warp_design: sw_design,
        link_dev_design: ld_design,
        p_primary,
        idx_q0: 0,
        idx_q1: 1,
        idx_qd1: 2,
        idx_g: 3,
        h_start: p_primary,
        h_len: 0,
        w_start: p_primary,
        w_len: 0,
    }
}

/// Assert `jac` (n_out*N × p) matches FD on `f(beta)` at `beta`.
fn assert_jac_fd<F>(label: &str, jac: &Array2<f64>, beta: &[f64], f: F)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n_out_n = jac.nrows();
    let p = jac.ncols();
    assert_eq!(beta.len(), p, "{label}: beta len mismatch");
    let base = f(beta);
    assert_eq!(base.len(), n_out_n, "{label}: output len mismatch");
    for j in 0..p {
        let mut bp = beta.to_vec();
        bp[j] += FD_EPS;
        let up = f(&bp);
        for i in 0..n_out_n {
            let fd = (up[i] - base[i]) / FD_EPS;
            let analytic = jac[[i, j]];
            let denom = fd.abs().max(analytic.abs()).max(1e-10);
            let rel = (fd - analytic).abs() / denom;
            assert!(
                rel < REL_TOL,
                "{label}: out={i} col={j} analytic={analytic:.8e} fd={fd:.8e} rel={rel:.2e}"
            );
        }
    }
}

// ── rigid outputs used for FD ─────────────────────────────────────────────────
// These define the 6-output target functions that the FD computes against.
// We match row 4 = log_phi(q1), row 5 = log(qd1).

fn rigid_outputs(
    q0: &[f64],
    q1: &[f64],
    qd1: &[f64],
    g_row: &[f64],
    z: &[f64],
    s: f64,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; 6 * n];
    for i in 0..n {
        let gi = g_row[i];
        let c = (1.0 + (s * gi).powi(2)).sqrt();
        let eta0 = q0[i] * c + s * gi * z[i];
        let eta1 = q1[i] * c + s * gi * z[i];
        let chi1 = c;
        let d1 = c;
        out[0 * n + i] = eta0;
        out[1 * n + i] = eta1;
        out[2 * n + i] = chi1.ln();
        out[3 * n + i] = d1.ln();
        // log_phi(q1) = -0.5*(q1^2 + ln(2π))
        out[4 * n + i] = -0.5 * (q1[i] * q1[i] + std::f64::consts::TAU.ln());
        out[5 * n + i] = qd1[i].ln();
    }
    out
}

// ── LogslopeFlexBlockJacobian ─────────────────────────────────────────────────

#[test]
fn flex_logslope_jacobian_rigid_fd() {
    let mut seed = 0x1234_5678_u64;
    let p_g = 3usize;
    let design_g = make_design(N, p_g, &mut seed);
    let z: Vec<f64> = (0..N).map(|_| lcg(&mut seed)).collect();
    let s = 0.5_f64;

    // Non-trivial beta so g != 0
    let beta_g: Vec<f64> = (0..p_g).map(|_| lcg(&mut seed) * 0.4).collect();
    let g_row: Vec<f64> = (0..N)
        .map(|i| (0..p_g).map(|j| design_g[[i, j]] * beta_g[j]).sum())
        .collect();
    let q0: Vec<f64> = (0..N).map(|_| lcg(&mut seed) * 0.5).collect();
    let q1: Vec<f64> = (0..N).map(|_| lcg(&mut seed) * 0.5).collect();
    let qd1: Vec<f64> = (0..N).map(|_| 0.5 + lcg(&mut seed).abs()).collect();

    // Rigid chain maps: for logslope block, q* do not depend on β_g.
    // dq0_time, dq1_time, dqd1_time, dq0_marginal, etc. are zero.
    let dq0_time = Array2::<f64>::zeros((N, 0));
    let dq1_time = Array2::<f64>::zeros((N, 0));
    let dqd1_time = Array2::<f64>::zeros((N, 0));
    let dq0_marginal = Array2::<f64>::zeros((N, 0));
    let dq1_marginal = Array2::<f64>::zeros((N, 0));
    let dqd1_marginal = Array2::<f64>::zeros((N, 0));

    let flex = build_rigid_flex_scalars(
        N,
        &q0,
        &q1,
        &qd1,
        &g_row,
        &z,
        s,
        &dq0_time,
        &dq1_time,
        &dqd1_time,
        &dq0_marginal,
        &dq1_marginal,
        &dqd1_marginal,
        design_g.clone(),
    );

    let cb = LogslopeFlexBlockJacobian::new(design_g.clone());
    let state = FamilyLinearizationState {
        beta: &beta_g,
        family_scalars: Some(Arc::new(flex) as _),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("LogslopeFlexBlockJacobian");
    assert_eq!(jac.nrows(), 6 * N);
    assert_eq!(jac.ncols(), p_g);

    let design_g_c = design_g.clone();
    let z_c = z.clone();
    let q0_c = q0.clone();
    let q1_c = q1.clone();
    let qd1_c = qd1.clone();
    assert_jac_fd("logslope_rigid", &jac, &beta_g, move |b| {
        let g_r: Vec<f64> = (0..N)
            .map(|i| (0..p_g).map(|j| design_g_c[[i, j]] * b[j]).sum())
            .collect();
        rigid_outputs(&q0_c, &q1_c, &qd1_c, &g_r, &z_c, s, N)
    });
}

// ── MarginalFlexBlockJacobian ─────────────────────────────────────────────────

#[test]
fn flex_marginal_jacobian_rigid_fd() {
    let mut seed = 0xABCD_EF01_u64;
    let p_m = 3usize;
    let design_m = make_design(N, p_m, &mut seed);
    let s = 0.4_f64;

    let beta_m: Vec<f64> = (0..p_m).map(|_| lcg(&mut seed) * 0.4).collect();
    // At rigid init g=0, c=1, so q_i = design_m[i] · beta_m
    let q0: Vec<f64> = (0..N)
        .map(|i| (0..p_m).map(|j| design_m[[i, j]] * beta_m[j]).sum())
        .collect();
    // q0=q1 (both entry and exit have same marginal contribution in rigid)
    let q1 = q0.clone();
    let qd1: Vec<f64> = (0..N).map(|_| 0.5).collect();
    let g_row = vec![0.0_f64; N]; // g=0 for marginal test
    let z = vec![0.0_f64; N];

    // Rigid: dq0/dβ_m[k] = design_m[i,k], dq1/dβ_m[k] = design_m[i,k], dqd1/dβ_m[k]=0
    let dq0_marginal = design_m.clone();
    let dq1_marginal = design_m.clone();
    let dqd1_marginal = Array2::<f64>::zeros((N, p_m));
    let dq0_time = Array2::<f64>::zeros((N, 0));
    let dq1_time = Array2::<f64>::zeros((N, 0));
    let dqd1_time = Array2::<f64>::zeros((N, 0));

    let flex = build_rigid_flex_scalars(
        N,
        &q0,
        &q1,
        &qd1,
        &g_row,
        &z,
        s,
        &dq0_time,
        &dq1_time,
        &dqd1_time,
        &dq0_marginal,
        &dq1_marginal,
        &dqd1_marginal,
        Array2::<f64>::zeros((N, 0)),
    );

    let cb = MarginalFlexBlockJacobian;
    let state = FamilyLinearizationState {
        beta: &beta_m,
        family_scalars: Some(Arc::new(flex) as _),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("MarginalFlexBlockJacobian");
    assert_eq!(jac.nrows(), 6 * N);
    assert_eq!(jac.ncols(), p_m);

    let design_m_c = design_m.clone();
    let qd1_c = qd1.clone();
    assert_jac_fd("marginal_rigid", &jac, &beta_m, move |b| {
        let q: Vec<f64> = (0..N)
            .map(|i| (0..p_m).map(|j| design_m_c[[i, j]] * b[j]).sum())
            .collect();
        let g_zeros = vec![0.0_f64; N];
        let z_zeros = vec![0.0_f64; N];
        rigid_outputs(&q, &q, &qd1_c, &g_zeros, &z_zeros, s, N)
    });
}

// ── TimeFlexBlockJacobian ─────────────────────────────────────────────────────

#[test]
fn flex_time_jacobian_rigid_fd() {
    let mut seed = 0xDEAD_BEEF_u64;
    let p_t = 3usize;
    let design_entry = make_design(N, p_t, &mut seed);
    let design_exit = make_design(N, p_t, &mut seed);
    let design_deriv = make_design(N, p_t, &mut seed);
    let s = 0.45_f64;

    // Use small beta_t so qd1 = 1.0 + design_deriv·beta_t stays well positive.
    let beta_t: Vec<f64> = (0..p_t).map(|_| lcg(&mut seed) * 0.1).collect();
    let q0: Vec<f64> = (0..N)
        .map(|i| (0..p_t).map(|j| design_entry[[i, j]] * beta_t[j]).sum())
        .collect();
    let q1: Vec<f64> = (0..N)
        .map(|i| (0..p_t).map(|j| design_exit[[i, j]] * beta_t[j]).sum())
        .collect();
    let qd1: Vec<f64> = (0..N)
        .map(|i| {
            1.0 + (0..p_t)
                .map(|j| design_deriv[[i, j]] * beta_t[j])
                .sum::<f64>()
        })
        .collect();
    let g_row = vec![0.0_f64; N];
    let z = vec![0.0_f64; N];

    // Rigid: dq0/dβ_t = design_entry row, dq1/dβ_t = design_exit row
    let dq0_time = design_entry.clone();
    let dq1_time = design_exit.clone();
    let dqd1_time = design_deriv.clone();
    let dq0_marginal = Array2::<f64>::zeros((N, 0));
    let dq1_marginal = Array2::<f64>::zeros((N, 0));
    let dqd1_marginal = Array2::<f64>::zeros((N, 0));

    let flex = build_rigid_flex_scalars(
        N,
        &q0,
        &q1,
        &qd1,
        &g_row,
        &z,
        s,
        &dq0_time,
        &dq1_time,
        &dqd1_time,
        &dq0_marginal,
        &dq1_marginal,
        &dqd1_marginal,
        Array2::<f64>::zeros((N, 0)),
    );

    let cb = TimeFlexBlockJacobian;
    let state = FamilyLinearizationState {
        beta: &beta_t,
        family_scalars: Some(Arc::new(flex) as _),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("TimeFlexBlockJacobian");
    assert_eq!(jac.nrows(), 6 * N);
    assert_eq!(jac.ncols(), p_t);

    let de = design_entry.clone();
    let dx = design_exit.clone();
    let dd = design_deriv.clone();
    assert_jac_fd("time_rigid", &jac, &beta_t, move |b| {
        let q0_r: Vec<f64> = (0..N)
            .map(|i| (0..p_t).map(|j| de[[i, j]] * b[j]).sum())
            .collect();
        let q1_r: Vec<f64> = (0..N)
            .map(|i| (0..p_t).map(|j| dx[[i, j]] * b[j]).sum())
            .collect();
        let qd1_r: Vec<f64> = (0..N)
            .map(|i| {
                1.0 + (0..p_t).map(|j| dd[[i, j]] * b[j]).sum::<f64>()
            })
            .collect();
        let g_zeros = vec![0.0_f64; N];
        let z_zeros = vec![0.0_f64; N];
        rigid_outputs(&q0_r, &q1_r, &qd1_r, &g_zeros, &z_zeros, s, N)
    });
}

// ── ScoreWarpFlexBlockJacobian and LinkDevFlexBlockJacobian ───────────────────

#[test]
fn flex_score_warp_empty_returns_zero() {
    let s = 0.5_f64;
    let q0 = vec![0.0_f64; N];
    let q1 = vec![0.0_f64; N];
    let qd1 = vec![1.0_f64; N];
    let g = vec![0.0_f64; N];
    let z = vec![0.0_f64; N];
    let dq_empty = Array2::<f64>::zeros((N, 0));
    let flex = build_rigid_flex_scalars(
        N,
        &q0,
        &q1,
        &qd1,
        &g,
        &z,
        s,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        Array2::<f64>::zeros((N, 0)),
    );

    let cb = ScoreWarpFlexBlockJacobian;
    let beta: Vec<f64> = vec![];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(Arc::new(flex) as _),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("ScoreWarpFlexBlockJacobian empty");
    assert_eq!(jac.nrows(), 6 * N);
    assert_eq!(jac.ncols(), 0);
}

#[test]
fn flex_link_dev_empty_returns_zero() {
    let s = 0.5_f64;
    let q0 = vec![0.0_f64; N];
    let q1 = vec![0.0_f64; N];
    let qd1 = vec![1.0_f64; N];
    let g = vec![0.0_f64; N];
    let z = vec![0.0_f64; N];
    let dq_empty = Array2::<f64>::zeros((N, 0));
    let flex = build_rigid_flex_scalars(
        N,
        &q0,
        &q1,
        &qd1,
        &g,
        &z,
        s,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        &dq_empty,
        Array2::<f64>::zeros((N, 0)),
    );

    let cb = LinkDevFlexBlockJacobian;
    let beta: Vec<f64> = vec![];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(Arc::new(flex) as _),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("LinkDevFlexBlockJacobian empty");
    assert_eq!(jac.nrows(), 6 * N);
    assert_eq!(jac.ncols(), 0);
}

/// When `family_scalars` is None, all flex Jacobians must return Err (hard contract).
#[test]
fn flex_jacobians_error_when_no_family_scalars() {
    let p = 2usize;
    let design = Array2::<f64>::zeros((N, p));
    let beta = vec![0.0_f64; p];
    let state_no_scalars = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };

    let logslope = LogslopeFlexBlockJacobian::new(design.clone());
    assert!(
        logslope.effective_jacobian_at(&state_no_scalars).is_err(),
        "LogslopeFlexBlockJacobian must error without family_scalars"
    );

    let marginal = MarginalFlexBlockJacobian;
    assert!(
        marginal.effective_jacobian_at(&state_no_scalars).is_err(),
        "MarginalFlexBlockJacobian must error without family_scalars"
    );

    let time_jac = TimeFlexBlockJacobian;
    assert!(
        time_jac.effective_jacobian_at(&state_no_scalars).is_err(),
        "TimeFlexBlockJacobian must error without family_scalars"
    );

    let score_warp = ScoreWarpFlexBlockJacobian;
    assert!(
        score_warp.effective_jacobian_at(&state_no_scalars).is_err(),
        "ScoreWarpFlexBlockJacobian must error without family_scalars"
    );

    let link_dev = LinkDevFlexBlockJacobian;
    assert!(
        link_dev.effective_jacobian_at(&state_no_scalars).is_err(),
        "LinkDevFlexBlockJacobian must error without family_scalars"
    );
}
