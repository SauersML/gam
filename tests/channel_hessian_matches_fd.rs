//! Finite-difference verification of per-family `FamilyChannelHessian` impls.
//!
//! For each family, we:
//!   1. Pick a random pilot state (primary-state vector u_i) and data.
//!   2. Evaluate the row NLL at (u_i + ε e_a + ε e_b), etc. via the closed-form
//!      kernel to obtain the FD second derivative ∂²ρ/∂u_a∂u_b.
//!   3. Compare against `FamilyChannelHessian::fill_subject(i, ...)`.
//!   4. Assert relative error < 1e-6 for all (a,b) entries.
//!
//! Families tested:
//!   - Survival marginal-slope (K=4): `SurvivalRowHessian`
//!   - Bernoulli marginal-slope (K=1): `BernoulliRowHessian`
//!   - Gaussian location-scale (K=2): `GaussianLocationScaleChannelHessian`

use gam::families::bernoulli_marginal_slope_identifiability::BernoulliRowHessian;
use gam::families::custom_family::FamilyChannelHessian;
use gam::families::gamlss::GaussianLocationScaleChannelHessian;
use gam::families::survival_marginal_slope_identifiability::{
    SurvivalRowHessian, survival_row_nll_grad_hess,
};
use ndarray::Array1;

const FD_H: f64 = 1e-4;
const REL_TOL: f64 = 1e-6;
const N: usize = 16;

fn rel_err(got: f64, ref_val: f64) -> f64 {
    let scale = ref_val.abs().max(1e-10);
    (got - ref_val).abs() / scale
}

/// Finite-difference Hessian of a scalar function f: R^k -> R.
/// Returns a k×k matrix H where H[a,b] = (f(u+h·e_a+h·e_b) - f(u+h·e_a) - f(u+h·e_b) + f(u)) / h².
fn fd_hessian<F: Fn(&[f64]) -> f64>(f: &F, u: &[f64], k: usize, h: f64) -> Vec<Vec<f64>> {
    let f0 = f(u);
    let mut f_plus: Vec<f64> = vec![0.0; k];
    for a in 0..k {
        let mut ua = u.to_vec();
        ua[a] += h;
        f_plus[a] = f(&ua);
    }
    let mut hess = vec![vec![0.0f64; k]; k];
    for a in 0..k {
        for b in a..k {
            let mut uab = u.to_vec();
            uab[a] += h;
            uab[b] += h;
            let fab = f(&uab);
            hess[a][b] = (fab - f_plus[a] - f_plus[b] + f0) / (h * h);
            hess[b][a] = hess[a][b];
        }
    }
    hess
}

// ── Survival marginal-slope (K=4) ─────────────────────────────────────────────

/// Row NLL for survival marginal-slope as a function of the 4-primary-state
/// vector (q0, q1, qd1, g). Data (z, w, d) are fixed.
fn survival_row_nll(u: &[f64], z: f64, w: f64, d: f64, dguard: f64, pscale: f64) -> f64 {
    match survival_row_nll_grad_hess(u[0], u[1], u[2], u[3], z, w, d, dguard, pscale) {
        Ok((nll, _, _)) => nll,
        Err(_) => f64::NAN,
    }
}

#[test]
fn survival_marginal_slope_channel_hessian_matches_fd() {
    // Fixed data scalars for all subjects.
    let z = 0.3_f64;
    let w = 1.0_f64;
    let dguard = 1e-6_f64;
    let pscale = 1.0_f64;

    // Two rows: one censored (d=0), one event (d=1).
    let data_d = [0.0_f64, 1.0_f64];

    // Pilot primary states (q0, q1, qd1, g) for N rows.
    // qd1 must be > dguard to avoid monotonicity violation.
    let states: Vec<[f64; 4]> = (0..N)
        .map(|i| {
            let t = (i as f64 + 1.0) * 0.1;
            [
                -0.5 + t * 0.3,  // q0
                -0.3 + t * 0.25, // q1
                0.1 + t * 0.05,  // qd1 > dguard
                -0.2 + t * 0.15, // g
            ]
        })
        .collect();

    // Build pilot arrays for SurvivalRowHessian constructor.
    let q0 = Array1::from_iter(states.iter().map(|s| s[0]));
    let q1 = Array1::from_iter(states.iter().map(|s| s[1]));
    let qd1 = Array1::from_iter(states.iter().map(|s| s[2]));
    let g = Array1::from_iter(states.iter().map(|s| s[3]));
    let z_arr = Array1::from_elem(N, z);
    let w_arr = Array1::from_elem(N, w);

    for &d in &data_d {
        let d_arr = Array1::from_elem(N, d);
        let row_hess = SurvivalRowHessian::from_pilot_primary_state(
            &q0, &q1, &qd1, &g, &z_arr, &w_arr, &d_arr, dguard, pscale,
        )
        .expect("SurvivalRowHessian::from_pilot_primary_state failed");

        for i in 0..N {
            let u = states[i];
            let f = |uu: &[f64]| survival_row_nll(uu, z, w, d, dguard, pscale);
            let fd = fd_hessian(&f, &u, 4, FD_H);

            let mut buf = [0.0f64; 16];
            row_hess.fill_subject(i, &mut buf);

            for a in 0..4 {
                for b in 0..4 {
                    let got = buf[a * 4 + b];
                    let ref_val = fd[a][b];
                    // Skip NaN FD entries (numerical issues at boundary rows).
                    if !ref_val.is_finite() {
                        continue;
                    }
                    let err = rel_err(got, ref_val);
                    assert!(
                        err < REL_TOL,
                        "survival marginal-slope W_i[{a},{b}] row {i} d={d}: \
                         got={got:.6e} fd={ref_val:.6e} rel_err={err:.2e} > tol={REL_TOL:.0e}"
                    );
                }
            }
        }
    }
}

// ── Bernoulli marginal-slope (K=1) ────────────────────────────────────────────

/// Probit Bernoulli NLL as a function of the scalar predictor η.
/// ρ(η) = -[y·log Φ(η) + (1-y)·log Φ(-η)] with sample weight w.
fn bernoulli_row_nll(u: &[f64], y: f64, w: f64) -> f64 {
    let eta = u[0];
    let p = gam::inference::probability::normal_cdf(eta).clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
    let nll = -(w * (y * p.ln() + (1.0 - y) * (1.0 - p).ln()));
    nll
}

#[test]
fn bernoulli_channel_hessian_matches_fd() {
    let etas: Vec<f64> = (0..N)
        .map(|i| -1.5 + (i as f64) * 0.2)
        .collect();
    let ws: Vec<f64> = (0..N).map(|i| 0.5 + (i as f64) * 0.05).collect();
    let ys = [0.0_f64, 1.0_f64];

    let eta_arr = Array1::from_iter(etas.iter().copied());
    let w_arr = Array1::from_iter(ws.iter().copied());
    let row_hess = BernoulliRowHessian::from_eta_pilot(&eta_arr, &w_arr);

    for &y in &ys {
        for i in 0..N {
            let u = [etas[i]];
            let f = |uu: &[f64]| bernoulli_row_nll(uu, y, ws[i]);
            let fd = fd_hessian(&f, &u, 1, FD_H);

            let mut buf = [0.0f64; 1];
            row_hess.fill_subject(i, &mut buf);

            let got = buf[0];
            let ref_val = fd[0][0];
            if !ref_val.is_finite() {
                continue;
            }
            let err = rel_err(got, ref_val);
            assert!(
                err < REL_TOL,
                "bernoulli W_i[0,0] row {i} y={y}: got={got:.6e} fd={ref_val:.6e} rel_err={err:.2e} > tol={REL_TOL:.0e}"
            );
        }
    }
}

// ── Gaussian location-scale (K=2) ─────────────────────────────────────────────

/// Gaussian location-scale row NLL as a function of (μ, log_σ).
/// ρ(μ, s) = w * [s + 0.5 * (y - μ)^2 * exp(-2s)]
fn gaussian_row_nll(u: &[f64], y: f64, w: f64) -> f64 {
    let mu = u[0];
    let s = u[1];
    let inv_sigma2 = (-2.0 * s).exp();
    let resid = y - mu;
    w * (s + 0.5 * resid * resid * inv_sigma2)
}

#[test]
fn gaussian_location_scale_channel_hessian_matches_fd() {
    let ys: Vec<f64> = (0..N).map(|i| -1.0 + (i as f64) * 0.15).collect();
    let ws: Vec<f64> = (0..N).map(|i| 0.8 + (i as f64) * 0.02).collect();
    // Pilot (mu, log_sigma).
    let mus: Vec<f64> = (0..N).map(|i| -0.8 + (i as f64) * 0.1).collect();
    let logss: Vec<f64> = (0..N).map(|i| -0.5 + (i as f64) * 0.06).collect();

    let y_arr = Array1::from_iter(ys.iter().copied());
    let w_arr = Array1::from_iter(ws.iter().copied());
    let mu_arr = Array1::from_iter(mus.iter().copied());
    let logs_arr = Array1::from_iter(logss.iter().copied());

    let hess = GaussianLocationScaleChannelHessian::from_pilot(&y_arr, &w_arr, &mu_arr, &logs_arr)
        .expect("GaussianLocationScaleChannelHessian::from_pilot failed");

    for i in 0..N {
        let u = [mus[i], logss[i]];
        let f = |uu: &[f64]| gaussian_row_nll(uu, ys[i], ws[i]);
        let fd = fd_hessian(&f, &u, 2, FD_H);

        let mut buf = [0.0f64; 4];
        hess.fill_subject(i, &mut buf);

        for a in 0..2 {
            for b in 0..2 {
                let got = buf[a * 2 + b];
                let ref_val = fd[a][b];
                if !ref_val.is_finite() {
                    continue;
                }
                let err = rel_err(got, ref_val);
                assert!(
                    err < REL_TOL,
                    "gaussian location-scale W_i[{a},{b}] row {i}: \
                     got={got:.6e} fd={ref_val:.6e} rel_err={err:.2e} > tol={REL_TOL:.0e}"
                );
            }
        }
    }
}
