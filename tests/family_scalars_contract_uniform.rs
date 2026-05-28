//! Uniform hard-contract tests for β-dependent block Jacobians.
//!
//! Each β-dependent block must:
//!   1. Return `Ok` at `beta=zeros`, `family_scalars=None`.
//!   2. Return `Err` (naming the expected scalars type) at `beta=nonzero`,
//!      `family_scalars=None`, when any `g_i != 0`.
//!   3. Return `Ok` and FD-matching values at `beta=nonzero`,
//!      `family_scalars=Some(correct)`.
//!
//! Families covered:
//!   - BMS marginal block  (`BmsMarginalJacobian`)
//!   - BMS logslope block  (`BmsLogslopeJacobian`)
//!
//! Survival location-scale and all GAMLSS families use `AdditiveBlockJacobian`
//! (β-independent linear blocks) — the contract is vacuously satisfied and
//! those families are confirmed here to compile but need no enforcement check.

use gam::families::bernoulli_marginal_slope::{
    BmsFamilyScalars, BmsLogslopeJacobian, BmsMarginalJacobian,
};
use gam::families::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use ndarray::{Array1, Array2};
use std::any::Any;
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Simple LCG for deterministic pseudo-random numbers without an extra dep.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*state >> 33) as u32;
    (bits as f64) / (u32::MAX as f64) * 2.0 - 1.0
}

/// Build a small (n × p) dense design matrix with random entries.
fn random_design(n: usize, p: usize, rng: &mut u64) -> Array2<f64> {
    let data: Vec<f64> = (0..n * p).map(|_| lcg_next(rng)).collect();
    Array2::from_shape_vec((n, p), data).unwrap()
}

/// Compute g_i = offset_s[i] + logslope_design[i,:p_s_use] · beta_s[..p_s_use].
fn compute_g(
    logslope_design: &Array2<f64>,
    offset_s: &Array1<f64>,
    beta_s: &[f64],
) -> Vec<f64> {
    let n = logslope_design.nrows();
    let p_s_use = logslope_design.ncols().min(beta_s.len());
    (0..n)
        .map(|i| {
            offset_s[i]
                + logslope_design
                    .row(i)
                    .iter()
                    .take(p_s_use)
                    .zip(beta_s.iter().take(p_s_use))
                    .map(|(&x, &b)| x * b)
                    .sum::<f64>()
        })
        .collect()
}

/// Compute q_i = offset_m[i] + marginal_design[i,:p_m_use] · beta_m[..p_m_use].
fn compute_q(
    marginal_design: &Array2<f64>,
    offset_m: &Array1<f64>,
    beta_m: &[f64],
) -> Vec<f64> {
    let n = marginal_design.nrows();
    let p_m_use = marginal_design.ncols().min(beta_m.len());
    (0..n)
        .map(|i| {
            offset_m[i]
                + marginal_design
                    .row(i)
                    .iter()
                    .take(p_m_use)
                    .zip(beta_m.iter().take(p_m_use))
                    .map(|(&x, &b)| x * b)
                    .sum::<f64>()
        })
        .collect()
}

/// Build correct BmsFamilyScalars from designs + beta.
fn build_bms_scalars(
    marginal_design: &Array2<f64>,
    logslope_design: &Array2<f64>,
    offset_m: &Array1<f64>,
    offset_s: &Array1<f64>,
    z: &[f64],
    beta_m: &[f64],
    beta_s: &[f64],
    s: f64,
) -> BmsFamilyScalars {
    let q_i = compute_q(marginal_design, offset_m, beta_m);
    let g_i = compute_g(logslope_design, offset_s, beta_s);
    let c_i: Vec<f64> = g_i
        .iter()
        .map(|&gi| {
            let sg = s * gi;
            (1.0 + sg * sg).sqrt()
        })
        .collect();
    BmsFamilyScalars {
        q_i,
        g_i,
        c_i,
        s,
        z_i: z.to_vec(),
    }
}

/// Finite-difference the marginal Jacobian: ∂η/∂β_m[j] ≈ (η(β+ε·e_j) − η(β−ε·e_j)) / (2ε).
/// η_i = q_i·c_i + s·g_i·z_i where q_i = M[i,:]·β_m + off_m[i], g_i = G[i,:]·β_s + off_s[i].
fn fd_marginal_jacobian(
    marginal_design: &Array2<f64>,
    logslope_design: &Array2<f64>,
    offset_m: &Array1<f64>,
    offset_s: &Array1<f64>,
    z: &[f64],
    beta_m: &[f64],
    beta_s: &[f64],
    s: f64,
) -> Array2<f64> {
    let n = marginal_design.nrows();
    let p_m = marginal_design.ncols();
    let eps = 1e-6_f64;

    let eta = |bm: &[f64]| -> Vec<f64> {
        let q = compute_q(marginal_design, offset_m, bm);
        let g = compute_g(logslope_design, offset_s, beta_s);
        (0..n)
            .map(|i| {
                let sg = s * g[i];
                let c = (1.0 + sg * sg).sqrt();
                q[i] * c + s * g[i] * z[i]
            })
            .collect()
    };

    let mut jac = Array2::<f64>::zeros((n, p_m));
    for j in 0..p_m {
        let mut bm_plus = beta_m.to_vec();
        let mut bm_minus = beta_m.to_vec();
        bm_plus[j] += eps;
        bm_minus[j] -= eps;
        let ep = eta(&bm_plus);
        let em = eta(&bm_minus);
        for i in 0..n {
            jac[[i, j]] = (ep[i] - em[i]) / (2.0 * eps);
        }
    }
    jac
}

/// Finite-difference the logslope Jacobian: ∂η/∂β_s[j].
fn fd_logslope_jacobian(
    marginal_design: &Array2<f64>,
    logslope_design: &Array2<f64>,
    offset_m: &Array1<f64>,
    offset_s: &Array1<f64>,
    z: &[f64],
    beta_m: &[f64],
    beta_s: &[f64],
    s: f64,
) -> Array2<f64> {
    let n = logslope_design.nrows();
    let p_s = logslope_design.ncols();
    let eps = 1e-6_f64;

    let eta = |bs: &[f64]| -> Vec<f64> {
        let q = compute_q(marginal_design, offset_m, beta_m);
        let g = compute_g(logslope_design, offset_s, bs);
        (0..n)
            .map(|i| {
                let sg = s * g[i];
                let c = (1.0 + sg * sg).sqrt();
                q[i] * c + s * g[i] * z[i]
            })
            .collect()
    };

    let mut jac = Array2::<f64>::zeros((n, p_s));
    for j in 0..p_s {
        let mut bs_plus = beta_s.to_vec();
        let mut bs_minus = beta_s.to_vec();
        bs_plus[j] += eps;
        bs_minus[j] -= eps;
        let ep = eta(&bs_plus);
        let em = eta(&bs_minus);
        for i in 0..n {
            jac[[i, j]] = (ep[i] - em[i]) / (2.0 * eps);
        }
    }
    jac
}

fn max_rel_error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| {
            let denom = av.abs().max(bv.abs()).max(1e-14);
            (av - bv).abs() / denom
        })
        .fold(0.0_f64, f64::max)
}

// ── BMS marginal block ────────────────────────────────────────────────────────

#[test]
fn bms_marginal_block_contract_beta_zero_scalars_none_ok() {
    let mut rng = 0xC0FFEE_u64;
    let n = 15;
    let p_m = 3;
    let p_s = 4;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::zeros(n);
    let offset_s = Array1::zeros(n);
    let cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m,
        offset_s,
        p_m,
    );
    let beta_zero = vec![0.0_f64; p_m + p_s];
    let state = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    cb.effective_jacobian_at(&state)
        .expect("bms marginal: beta=0, scalars=None must return Ok");
}

#[test]
fn bms_marginal_block_contract_beta_nonzero_scalars_none_err() {
    let mut rng = 0xBEEF_u64;
    let n = 15;
    let p_m = 3;
    let p_s = 4;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::zeros(n);
    let offset_s = Array1::zeros(n);
    let cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m,
        offset_s,
        p_m,
    );
    // beta_s is nonzero → g_i != 0 for at least one row (design has random values).
    let mut beta = vec![0.0_f64; p_m + p_s];
    beta[p_m] = 1.0;
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let result = cb.effective_jacobian_at(&state);
    assert!(
        result.is_err(),
        "bms marginal block must return Err when beta_s nonzero and family_scalars=None; got Ok"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("BmsFamilyScalars"),
        "error message must name BmsFamilyScalars; got: {msg}"
    );
    assert!(
        msg.contains("family_scalars: None"),
        "error message must mention family_scalars: None; got: {msg}"
    );
}

#[test]
fn bms_marginal_block_contract_beta_nonzero_scalars_some_ok_fd_match() {
    let mut rng = 0xDEAD_u64;
    let n = 20;
    let p_m = 3;
    let p_s = 4;
    let s = 0.7_f64;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::from_vec((0..n).map(|_| lcg_next(&mut rng) * 0.1).collect());
    let offset_s = Array1::from_vec((0..n).map(|_| lcg_next(&mut rng) * 0.1).collect());
    let z: Vec<f64> = (0..n).map(|_| lcg_next(&mut rng)).collect();
    let cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        p_m,
    );

    let beta_m: Vec<f64> = (0..p_m).map(|_| lcg_next(&mut rng) * 0.5).collect();
    let beta_s: Vec<f64> = (0..p_s).map(|_| lcg_next(&mut rng) * 0.5).collect();
    let mut beta = beta_m.clone();
    beta.extend_from_slice(&beta_s);

    let scalars = build_bms_scalars(
        &marginal,
        &logslope,
        &offset_m,
        &offset_s,
        &z,
        &beta_m,
        &beta_s,
        s,
    );
    let scalars_arc: Arc<dyn Any + Send + Sync> = Arc::new(scalars);

    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars_arc),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("bms marginal: beta nonzero, scalars=Some must return Ok");

    assert_eq!(jac.shape(), &[n, p_m]);

    let fd = fd_marginal_jacobian(
        &marginal, &logslope, &offset_m, &offset_s, &z, &beta_m, &beta_s, s,
    );
    let rel_err = max_rel_error(&jac, &fd);
    assert!(
        rel_err < 1e-5,
        "bms marginal Jacobian rel-error vs FD = {rel_err:.3e} (expected < 1e-5)"
    );
}

// ── BMS logslope block ────────────────────────────────────────────────────────

#[test]
fn bms_logslope_block_contract_beta_zero_scalars_none_ok() {
    let mut rng = 0xABCD_u64;
    let n = 15;
    let p_m = 3;
    let p_s = 4;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::zeros(n);
    let offset_s = Array1::zeros(n);
    let z: Arc<[f64]> = Arc::from(vec![0.5_f64; n].into_boxed_slice());
    let cb = BmsLogslopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m,
        offset_s,
        Arc::clone(&z),
        p_m,
    );
    let beta_zero = vec![0.0_f64; p_m + p_s];
    let state = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    cb.effective_jacobian_at(&state)
        .expect("bms logslope: beta=0, scalars=None must return Ok");
}

#[test]
fn bms_logslope_block_contract_beta_nonzero_scalars_none_err() {
    let mut rng = 0x1234_u64;
    let n = 15;
    let p_m = 3;
    let p_s = 4;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::zeros(n);
    let offset_s = Array1::zeros(n);
    let z: Arc<[f64]> = Arc::from(vec![0.5_f64; n].into_boxed_slice());
    let cb = BmsLogslopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m,
        offset_s,
        Arc::clone(&z),
        p_m,
    );
    // beta_s nonzero → g_i != 0 for at least one row.
    let mut beta = vec![0.0_f64; p_m + p_s];
    beta[p_m] = 1.0;
    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let result = cb.effective_jacobian_at(&state);
    assert!(
        result.is_err(),
        "bms logslope block must return Err when beta_s nonzero and family_scalars=None; got Ok"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("BmsFamilyScalars"),
        "error message must name BmsFamilyScalars; got: {msg}"
    );
    assert!(
        msg.contains("family_scalars: None"),
        "error message must mention family_scalars: None; got: {msg}"
    );
}

#[test]
fn bms_logslope_block_contract_beta_nonzero_scalars_some_ok_fd_match() {
    let mut rng = 0xFACE_u64;
    let n = 20;
    let p_m = 3;
    let p_s = 4;
    let s = 0.8_f64;
    let marginal = Arc::new(random_design(n, p_m, &mut rng));
    let logslope = Arc::new(random_design(n, p_s, &mut rng));
    let offset_m = Array1::from_vec((0..n).map(|_| lcg_next(&mut rng) * 0.1).collect());
    let offset_s = Array1::from_vec((0..n).map(|_| lcg_next(&mut rng) * 0.1).collect());
    let z_vec: Vec<f64> = (0..n).map(|_| lcg_next(&mut rng)).collect();
    let z_arc: Arc<[f64]> = Arc::from(z_vec.clone().into_boxed_slice());
    let cb = BmsLogslopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        Arc::clone(&z_arc),
        p_m,
    );

    let beta_m: Vec<f64> = (0..p_m).map(|_| lcg_next(&mut rng) * 0.5).collect();
    let beta_s: Vec<f64> = (0..p_s).map(|_| lcg_next(&mut rng) * 0.5).collect();
    let mut beta = beta_m.clone();
    beta.extend_from_slice(&beta_s);

    let scalars = build_bms_scalars(
        &marginal,
        &logslope,
        &offset_m,
        &offset_s,
        &z_vec,
        &beta_m,
        &beta_s,
        s,
    );
    let scalars_arc: Arc<dyn Any + Send + Sync> = Arc::new(scalars);

    let state = FamilyLinearizationState {
        beta: &beta,
        family_scalars: Some(scalars_arc),
        channel_hessian: None,
        probit_frailty_scale: s,
    };
    let jac = cb
        .effective_jacobian_at(&state)
        .expect("bms logslope: beta nonzero, scalars=Some must return Ok");

    assert_eq!(jac.shape(), &[n, p_s]);

    let fd = fd_logslope_jacobian(
        &marginal, &logslope, &offset_m, &offset_s, &z_vec, &beta_m, &beta_s, s,
    );
    let rel_err = max_rel_error(&jac, &fd);
    assert!(
        rel_err < 1e-5,
        "bms logslope Jacobian rel-error vs FD = {rel_err:.3e} (expected < 1e-5)"
    );
}
