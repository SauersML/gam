//! Finite-difference verification of the Bernoulli `FamilyChannelHessian`.
//!
//! For each family, we:
//!   1. Pick a random pilot state (primary-state vector u_i) and data.
//!   2. Evaluate the row NLL at (u_i + ε e_a + ε e_b), etc. via the closed-form
//!      kernel to obtain the FD second derivative ∂²ρ/∂u_a∂u_b.
//!   3. Compare against `FamilyChannelHessian::fill_subject(i, ...)`.
//!   4. Assert relative error < 1e-6 for all (a,b) entries.
//!
//! The survival marginal-slope fit evaluates its canonical row program
//! directly and no longer exposes a separate identifiability-only Hessian
//! adapter.

use gam::families::custom_family::FamilyChannelHessian;
use gam::identifiability::families::bernoulli::BernoulliRowHessian;
use ndarray::Array1;

const FD_H: f64 = 1e-4;
const REL_TOL: f64 = 1e-5;
const ABS_TOL: f64 = 1e-6;
const N: usize = 16;

fn rel_err(got: f64, ref_val: f64) -> f64 {
    let scale = ref_val.abs().max(1.0);
    (got - ref_val).abs() / scale
}

fn err_within_tol(got: f64, ref_val: f64) -> bool {
    let abs_err = (got - ref_val).abs();
    if abs_err < ABS_TOL {
        return true;
    }
    rel_err(got, ref_val) < REL_TOL
}

/// Central-difference Hessian of a scalar function f: R^k -> R.
///
/// For diagonal entries: H[a,a] = (f(u+h·e_a) − 2·f(u) + f(u−h·e_a)) / h².
/// For off-diagonals:    H[a,b] = (f(u+h·e_a+h·e_b) − f(u+h·e_a−h·e_b)
///                                 − f(u−h·e_a+h·e_b) + f(u−h·e_a−h·e_b)) / (4 h²).
/// Both are O(h²) truncation accurate.
fn fd_hessian<F: Fn(&[f64]) -> f64>(f: &F, u: &[f64], k: usize, h: f64) -> Vec<Vec<f64>> {
    let f0 = f(u);
    let mut hess = vec![vec![0.0f64; k]; k];
    let mut f_plus: Vec<f64> = vec![0.0; k];
    let mut f_minus: Vec<f64> = vec![0.0; k];
    for a in 0..k {
        let mut ua_p = u.to_vec();
        let mut ua_m = u.to_vec();
        ua_p[a] += h;
        ua_m[a] -= h;
        f_plus[a] = f(&ua_p);
        f_minus[a] = f(&ua_m);
    }
    for a in 0..k {
        hess[a][a] = (f_plus[a] - 2.0 * f0 + f_minus[a]) / (h * h);
    }
    for a in 0..k {
        for b in (a + 1)..k {
            let mut u_pp = u.to_vec();
            let mut u_pm = u.to_vec();
            let mut u_mp = u.to_vec();
            let mut u_mm = u.to_vec();
            u_pp[a] += h;
            u_pp[b] += h;
            u_pm[a] += h;
            u_pm[b] -= h;
            u_mp[a] -= h;
            u_mp[b] += h;
            u_mm[a] -= h;
            u_mm[b] -= h;
            let off = (f(&u_pp) - f(&u_pm) - f(&u_mp) + f(&u_mm)) / (4.0 * h * h);
            hess[a][b] = off;
            hess[b][a] = off;
        }
    }
    hess
}

// ── Bernoulli marginal-slope (K=1) ────────────────────────────────────────────

/// Probit Bernoulli observed NLL as a function of the scalar predictor η
/// for a specific response y ∈ {0, 1}.
fn bernoulli_obs_nll(u: &[f64], y: f64, w: f64) -> f64 {
    let eta = u[0];
    let p = gam::inference::probability::normal_cdf(eta)
        .clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
    let one_m = (1.0 - p).max(f64::MIN_POSITIVE);
    -(w * (y * p.ln() + (1.0 - y) * one_m.ln()))
}

#[test]
fn bernoulli_channel_hessian_matches_fd() {
    let etas: Vec<f64> = (0..N).map(|i| -1.5 + (i as f64) * 0.2).collect();
    let ws: Vec<f64> = (0..N).map(|i| 0.5 + (i as f64) * 0.05).collect();

    let eta_arr = Array1::from_iter(etas.iter().copied());
    let w_arr = Array1::from_iter(ws.iter().copied());
    let row_hess = BernoulliRowHessian::from_eta_pilot(&eta_arr, &w_arr);

    for i in 0..N {
        let u = [etas[i]];
        // The BernoulliRowHessian stores the Fisher information (IRLS weight):
        //   W_i = E_y[∂²ρ/∂η²] = p·(∂²ρ/∂η²|y=1) + (1-p)·(∂²ρ/∂η²|y=0)
        // We approximate this by computing the FD second derivative at each y ∈ {0,1}
        // and taking the probability-weighted average.
        let p_i = gam::inference::probability::normal_cdf(eta_arr[i])
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
        let fd_y1 = fd_hessian(&|uu: &[f64]| bernoulli_obs_nll(uu, 1.0, ws[i]), &u, 1, FD_H);
        let fd_y0 = fd_hessian(&|uu: &[f64]| bernoulli_obs_nll(uu, 0.0, ws[i]), &u, 1, FD_H);
        let ref_val = p_i * fd_y1[0][0] + (1.0 - p_i) * fd_y0[0][0];

        let mut buf = [0.0f64; 1];
        row_hess.fill_subject(i, &mut buf);

        let got = buf[0];
        if !ref_val.is_finite() {
            continue;
        }
        assert!(
            err_within_tol(got, ref_val),
            "bernoulli W_i[0,0] row {i}: got={got:.6e} fd_expected={ref_val:.6e} abs_err={abs:.2e} rel_err={rel:.2e}",
            abs = (got - ref_val).abs(),
            rel = rel_err(got, ref_val),
        );
    }
}
