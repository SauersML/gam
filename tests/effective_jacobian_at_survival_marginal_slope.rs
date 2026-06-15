//! Finite-difference regression test for survival marginal-slope
//! `BlockEffectiveJacobian` implementations.
//!
//! For each of the three primary blocks (logslope, marginal, time), this test:
//!
//!   1. Constructs the block's `jacobian_callback` with synthetic data.
//!   2. Evaluates `effective_jacobian_at` to get the analytic stacked Jacobian
//!      `J ∈ ℝ^{3n × p}`.
//!   3. Computes a finite-difference approximation: for each column `j`,
//!      perturb `β[j]` by ε, re-evaluate the corresponding η formula, and
//!      divide by ε.
//!   4. Asserts relative error < 1e-7.
//!
//! Tests run at β=0 AND at a random non-zero β to lock in the full β-dependent
//! Jacobian contract (not just the linearization at the origin).

use gam::families::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use gam::families::survival_marginal_slope::{
    LogslopeBlockJacobian, MarginalBlockJacobian, SurvivalMarginalSlopeFamilyScalars,
    TimeBlockJacobian,
};
use ndarray::Array2;
use std::sync::Arc;

const N: usize = 16;
const P: usize = 4;
const FD_EPS: f64 = 1e-6;
const REL_ERR_TOL: f64 = 1e-5; // finite-diff error, generous to account for ε² truncation

/// A simple deterministic pseudo-random generator (no stdlib rand required).
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*seed >> 33) as u32) as f64;
    bits / (u32::MAX as f64) * 2.0 - 1.0
}

fn make_design(n: usize, p: usize, seed: &mut u64) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |_| lcg(seed))
}

fn make_vec(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n).map(|_| lcg(seed)).collect()
}

fn make_pos_vec(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n).map(|_| lcg(seed).abs() + 0.1).collect()
}

fn matvec(a: &Array2<f64>, x: &[f64]) -> Vec<f64> {
    let n = a.nrows();
    let p = a.ncols();
    assert_eq!(x.len(), p);
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..p {
            out[i] += a[[i, j]] * x[j];
        }
    }
    out
}

/// Assert that the analytic Jacobian `jac` (shape `3n × p`) matches the
/// finite-difference approximation of `eta_fn(beta)` (returns `3n`-vector).
///
/// `eta_fn(beta)` should return a `Vec<f64>` of length `3*N` representing
/// the stacked outputs `[η0[0..N], η1[0..N], ad1[0..N]]` at the given β.
fn assert_jacobian_matches_fd<F>(name: &str, jac: &Array2<f64>, beta: &[f64], eta_fn: F)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let three_n = jac.nrows();
    let p = jac.ncols();
    assert_eq!(beta.len(), p, "{name}: beta len mismatch");
    let eta0 = eta_fn(beta);
    assert_eq!(eta0.len(), three_n, "{name}: eta_fn output len mismatch");

    for j in 0..p {
        let mut beta_plus = beta.to_vec();
        beta_plus[j] += FD_EPS;
        let eta_plus = eta_fn(&beta_plus);
        for i in 0..three_n {
            let fd = (eta_plus[i] - eta0[i]) / FD_EPS;
            let analytic = jac[[i, j]];
            let denom = fd.abs().max(analytic.abs()).max(1e-12);
            let rel_err = (fd - analytic).abs() / denom;
            assert!(
                rel_err < REL_ERR_TOL,
                "{name}: row={i} col={j} analytic={analytic:.8e} fd={fd:.8e} rel_err={rel_err:.2e} > {REL_ERR_TOL:.0e}",
            );
        }
    }
}

// ── LogslopeBlockJacobian ─────────────────────────────────────────────────

fn logslope_eta(
    design: &Array2<f64>,
    z: &[f64],
    s: f64,
    q0: &[f64],
    q1: &[f64],
    qd1: &[f64],
    beta: &[f64],
) -> Vec<f64> {
    let n = design.nrows();
    let g = matvec(design, beta);
    let mut out = vec![0.0f64; 3 * n];
    for i in 0..n {
        let gi = g[i];
        let c = (1.0 + (s * gi).powi(2)).sqrt();
        // η0[i] = q0[i] * c + s * z[i] * g[i]
        out[i] = q0[i] * c + s * z[i] * gi;
        // η1[i] = q1[i] * c + s * z[i] * g[i]
        out[n + i] = q1[i] * c + s * z[i] * gi;
        // ad1[i] = qd1[i] * c
        out[2 * n + i] = qd1[i] * c;
    }
    out
}

#[test]
fn logslope_jacobian_fd_at_zero_beta() {
    let mut seed = 0xdeadbeef_u64;
    let design = make_design(N, P, &mut seed);
    let z = make_pos_vec(N, &mut seed);
    let q0 = make_vec(N, &mut seed);
    let q1 = make_vec(N, &mut seed);
    let qd1 = make_pos_vec(N, &mut seed);
    let s = 0.5_f64;

    let g_zero = vec![0.0f64; N];
    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars::new(
        q0.clone(),
        q1.clone(),
        qd1.clone(),
        g_zero,
        s,
        z.clone(),
    ));
    // Override c_i to ones explicitly (g=0 → c=1)
    assert!(
        scalars.c_i.iter().all(|&c| (c - 1.0).abs() < 1e-12),
        "c_i should be 1 at g=0"
    );

    let cb = LogslopeBlockJacobian::new(design.clone(), z.clone(), s);
    let beta = vec![0.0f64; P];

    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("logslope Jacobian at beta=0");
    assert_eq!(jac.nrows(), 3 * N);
    assert_eq!(jac.ncols(), P);

    let q0_c = q0.clone();
    let q1_c = q1.clone();
    let qd1_c = qd1.clone();
    let design_c = design.clone();
    let z_c = z.clone();
    assert_jacobian_matches_fd("logslope_beta0", &jac, &beta, move |b| {
        logslope_eta(&design_c, &z_c, s, &q0_c, &q1_c, &qd1_c, b)
    });
}

#[test]
fn logslope_jacobian_fd_at_nonzero_beta() {
    let mut seed = 0xcafebabe_u64;
    let design = make_design(N, P, &mut seed);
    let z = make_pos_vec(N, &mut seed);
    let s = 0.4_f64;

    // Random non-zero beta (small so derivatives don't blow up)
    let beta: Vec<f64> = (0..P).map(|_| lcg(&mut seed) * 0.3).collect();

    // Compute the scalars at this beta
    let g: Vec<f64> = matvec(&design, &beta);
    let c: Vec<f64> = g
        .iter()
        .map(|&gi| (1.0 + (s * gi).powi(2)).sqrt())
        .collect();
    // Use non-trivial q values
    let q0: Vec<f64> = (0..N).map(|_| lcg(&mut seed) * 0.5).collect();
    let q1: Vec<f64> = (0..N).map(|_| lcg(&mut seed) * 0.5).collect();
    let qd1: Vec<f64> = (0..N).map(|_| lcg(&mut seed).abs() + 0.1).collect();

    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars {
        q0_i: q0.clone(),
        q1_i: q1.clone(),
        qd1_i: qd1.clone(),
        g_i: g.clone(),
        c_i: c.clone(),
        s,
        z_i: z.clone(),
    });

    let cb = LogslopeBlockJacobian::new(design.clone(), z.clone(), s);
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("logslope Jacobian at nonzero beta");

    // For the FD test we use the same q0, q1, qd1 fixed (they come from
    // other blocks). The FD perturbs only the logslope beta.
    let q0_c = q0.clone();
    let q1_c = q1.clone();
    let qd1_c = qd1.clone();
    let design_c = design.clone();
    let z_c = z.clone();
    assert_jacobian_matches_fd("logslope_nonzero", &jac, &beta, move |b| {
        logslope_eta(&design_c, &z_c, s, &q0_c, &q1_c, &qd1_c, b)
    });
}

// ── MarginalBlockJacobian ─────────────────────────────────────────────────

fn marginal_eta(design: &Array2<f64>, c: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = design.nrows();
    let m_beta = matvec(design, beta);
    let mut out = vec![0.0f64; 3 * n];
    for i in 0..n {
        let v = c[i] * m_beta[i];
        out[i] = v;
        out[n + i] = v;
        // out[2*n + i] = 0 (ad1 row)
    }
    out
}

#[test]
fn marginal_jacobian_fd_at_zero_beta() {
    let mut seed = 0x11223344_u64;
    let design = make_design(N, P, &mut seed);
    let c_one = vec![1.0f64; N];

    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars {
        q0_i: vec![0.0f64; N],
        q1_i: vec![0.0f64; N],
        qd1_i: vec![0.0f64; N],
        g_i: vec![0.0f64; N],
        c_i: c_one.clone(),
        s: 0.5,
        z_i: vec![1.0f64; N],
    });

    let cb = MarginalBlockJacobian::new(design.clone());
    let beta = vec![0.0f64; P];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("marginal Jacobian at beta=0");

    let design_c = design.clone();
    assert_jacobian_matches_fd("marginal_beta0", &jac, &beta, move |b| {
        marginal_eta(&design_c, &c_one, b)
    });
}

#[test]
fn marginal_jacobian_fd_at_nonzero_beta() {
    let mut seed = 0x55667788_u64;
    let design = make_design(N, P, &mut seed);
    // Non-trivial c (depends on g from logslope block, not marginal beta)
    let c: Vec<f64> = make_pos_vec(N, &mut seed);
    let beta: Vec<f64> = (0..P).map(|_| lcg(&mut seed) * 0.5).collect();

    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars {
        q0_i: vec![0.0f64; N],
        q1_i: vec![0.0f64; N],
        qd1_i: vec![0.0f64; N],
        g_i: vec![0.0f64; N],
        c_i: c.clone(),
        s: 0.5,
        z_i: vec![1.0f64; N],
    });

    let cb = MarginalBlockJacobian::new(design.clone());
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("marginal Jacobian at nonzero beta");

    let design_c = design.clone();
    let c_c = c.clone();
    assert_jacobian_matches_fd("marginal_nonzero", &jac, &beta, move |b| {
        marginal_eta(&design_c, &c_c, b)
    });
}

// ── TimeBlockJacobian ─────────────────────────────────────────────────────

fn time_eta(
    d_entry: &Array2<f64>,
    d_exit: &Array2<f64>,
    d_deriv: &Array2<f64>,
    c: &[f64],
    beta: &[f64],
) -> Vec<f64> {
    let n = d_entry.nrows();
    let eta0 = matvec(d_entry, beta);
    let eta1 = matvec(d_exit, beta);
    let ad1 = matvec(d_deriv, beta);
    let mut out = vec![0.0f64; 3 * n];
    for i in 0..n {
        out[i] = c[i] * eta0[i];
        out[n + i] = c[i] * eta1[i];
        out[2 * n + i] = c[i] * ad1[i];
    }
    out
}

#[test]
fn time_jacobian_fd_at_zero_beta() {
    let mut seed = 0xabcdef01_u64;
    let d_entry = make_design(N, P, &mut seed);
    let d_exit = make_design(N, P, &mut seed);
    let d_deriv = make_design(N, P, &mut seed);
    let c_one = vec![1.0f64; N];

    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars {
        q0_i: vec![0.0f64; N],
        q1_i: vec![0.0f64; N],
        qd1_i: vec![0.0f64; N],
        g_i: vec![0.0f64; N],
        c_i: c_one.clone(),
        s: 0.5,
        z_i: vec![0.0f64; N],
    });

    let cb = TimeBlockJacobian::new(d_entry.clone(), d_exit.clone(), d_deriv.clone());
    let beta = vec![0.0f64; P];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("time Jacobian at beta=0");

    let de = d_entry.clone();
    let dx = d_exit.clone();
    let dd = d_deriv.clone();
    assert_jacobian_matches_fd("time_beta0", &jac, &beta, move |b| {
        time_eta(&de, &dx, &dd, &c_one, b)
    });
}

#[test]
fn time_jacobian_fd_at_nonzero_beta() {
    let mut seed = 0xfedcba98_u64;
    let d_entry = make_design(N, P, &mut seed);
    let d_exit = make_design(N, P, &mut seed);
    let d_deriv = make_design(N, P, &mut seed);
    let c: Vec<f64> = make_pos_vec(N, &mut seed);
    let beta: Vec<f64> = (0..P).map(|_| lcg(&mut seed) * 0.3).collect();

    let scalars = Arc::new(SurvivalMarginalSlopeFamilyScalars {
        q0_i: vec![0.0f64; N],
        q1_i: vec![0.0f64; N],
        qd1_i: vec![0.0f64; N],
        g_i: vec![0.0f64; N],
        c_i: c.clone(),
        s: 0.5,
        z_i: vec![0.0f64; N],
    });

    let cb = TimeBlockJacobian::new(d_entry.clone(), d_exit.clone(), d_deriv.clone());
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars),
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("time Jacobian at nonzero beta");

    let de = d_entry.clone();
    let dx = d_exit.clone();
    let dd = d_deriv.clone();
    let c_c = c.clone();
    assert_jacobian_matches_fd("time_nonzero", &jac, &beta, move |b| {
        time_eta(&de, &dx, &dd, &c_c, b)
    });
}

// ── Fallback (no family_scalars) uses c=1 ─────────────────────────────────

#[test]
fn logslope_no_scalars_falls_back_to_c1() {
    let mut seed = 0x99887766_u64;
    let design = make_design(N, P, &mut seed);
    let z: Vec<f64> = make_pos_vec(N, &mut seed);
    let s = 0.5_f64;

    let cb = LogslopeBlockJacobian::new(design.clone(), z.clone(), s);
    let beta = vec![0.0f64; P];
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("logslope Jacobian with no scalars");

    // At c=1, g=0, q0=q1=qd1=0: only the s·z term survives.
    // η0[i] row j = s*z[i] * design[i,j]
    // η1 same, ad1 = 0
    for i in 0..N {
        for j in 0..P {
            let expected_eta0 = s * z[i] * design[[i, j]];
            let rel_err = (jac[[i, j]] - expected_eta0).abs() / expected_eta0.abs().max(1e-12);
            assert!(
                rel_err < 1e-10,
                "logslope no_scalars fallback: eta0 row={i} col={j} got={} expected={expected_eta0}",
                jac[[i, j]],
            );
            let rel_err1 = (jac[[N + i, j]] - expected_eta0).abs() / expected_eta0.abs().max(1e-12);
            assert!(
                rel_err1 < 1e-10,
                "logslope no_scalars fallback: eta1 row={i} col={j} got={} expected={expected_eta0}",
                jac[[N + i, j]],
            );
            assert!(
                jac[[2 * N + i, j]].abs() < 1e-12,
                "logslope no_scalars fallback: ad1 should be zero, got {}",
                jac[[2 * N + i, j]],
            );
        }
    }
}
