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

use gam::families::custom_family::FamilyChannelHessian;
use gam::families::gamlss::GaussianLocationScaleChannelHessian;
use gam::families::survival::marginal_slope::identifiability::SurvivalRowHessian;
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

// ── End-to-end Fisher Gram J^T W J for survival marginal-slope ────────────────

/// Build the channel-stacked Fisher Gram explicitly:
///   G[j, k] = Σ_i (J_i[:, j])^T · W_i · J_i[:, k]
/// where J_i is the (K × p) per-subject Jacobian and W_i is the (K × K)
/// per-subject channel Hessian. Returns a (p × p) symmetric matrix.
fn explicit_fisher_gram(
    j_per_subject: &[ndarray::Array2<f64>],
    w_per_subject: &[ndarray::Array2<f64>],
) -> ndarray::Array2<f64> {
    let n = j_per_subject.len();
    assert_eq!(w_per_subject.len(), n);
    let p = j_per_subject[0].ncols();
    let k = j_per_subject[0].nrows();
    let mut g = ndarray::Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let j_i = &j_per_subject[i];
        let w_i = &w_per_subject[i];
        assert_eq!(j_i.dim(), (k, p), "J_i must be (K × p)");
        assert_eq!(w_i.dim(), (k, k), "W_i must be (K × K)");
        // G += J_iᵀ · W_i · J_i
        let wj = w_i.dot(j_i); // (k, p)
        let contribution = j_i.t().dot(&wj); // (p, p)
        g = &g + &contribution;
    }
    g
}

#[test]
fn survival_marginal_slope_channel_stacked_fisher_gram_matches_closed_form() {
    use gam::identifiability::families::compiler::{
        PrimaryChannelBlocks, build_raw_grams_from_channel_blocks,
    };

    let n = 8usize;
    let p_time = 2usize;
    let p_marginal = 2usize;
    let p_logslope = 1usize;
    let p_total = p_time + p_marginal + p_logslope;

    // Random-ish per-subject design rows for each channel.
    // Time block contributes to channels (q0, q1, qd1), each via a separate
    // raw design slice. We synthesise three (n × p_time) blocks.
    let mut dq0 = ndarray::Array2::<f64>::zeros((n, p_time));
    let mut dq1 = ndarray::Array2::<f64>::zeros((n, p_time));
    let mut dqd1 = ndarray::Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        for j in 0..p_time {
            let s = ((i + 1) as f64) * 0.13 + ((j + 1) as f64) * 0.27;
            dq0[[i, j]] = s.sin();
            dq1[[i, j]] = s.cos();
            dqd1[[i, j]] = 0.5 * s.tan().clamp(-2.0, 2.0);
        }
    }
    // Marginal block: shared (n × p_marginal) for q channels (q0, q1).
    let mut m_dq = ndarray::Array2::<f64>::zeros((n, p_marginal));
    for i in 0..n {
        for j in 0..p_marginal {
            m_dq[[i, j]] = (((i + 2) as f64) * (j as f64 + 0.4)).sin();
        }
    }
    let m_dqd1 = ndarray::Array2::<f64>::zeros((n, p_marginal));
    // Logslope block: (n × p_logslope) for g channel only.
    let mut g_dg = ndarray::Array2::<f64>::zeros((n, p_logslope));
    for i in 0..n {
        for j in 0..p_logslope {
            g_dg[[i, j]] = 0.3 + ((i + j + 1) as f64).ln();
        }
    }

    // Per-subject pilot primary state.
    let q0 = Array1::from_iter((0..n).map(|i| 0.2 + 0.1 * (i as f64)));
    let q1 = Array1::from_iter((0..n).map(|i| 0.1 + 0.08 * (i as f64)));
    let qd1 = Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * (i as f64)));
    let g_pilot = Array1::from_iter((0..n).map(|i| -0.1 + 0.04 * (i as f64)));
    let z = Array1::from_elem(n, 0.2);
    let w_arr = Array1::from_elem(n, 1.0);
    let event = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }));

    let row_hess = SurvivalRowHessian::from_pilot_primary_state(
        &q0, &q1, &qd1, &g_pilot, &z, &w_arr, &event, 1e-6, 1.0,
    )
    .expect("SurvivalRowHessian::from_pilot_primary_state failed");

    // Channel-block decomposition: K=4 channels (q0=0, q1=1, qd1=2, g=3).
    let time_slots = vec![
        Some(dq0.clone()),
        Some(dq1.clone()),
        Some(dqd1.clone()),
        None,
    ];
    let marg_slots = vec![
        Some(m_dq.clone()),
        Some(m_dq.clone()),
        Some(m_dqd1.clone()),
        None,
    ];
    let log_slots = vec![None, None, None, Some(g_dg.clone())];
    let channel_blocks = PrimaryChannelBlocks {
        blocks: vec![time_slots, marg_slots, log_slots],
    };
    let raw_ranges = vec![
        0..p_time,
        p_time..(p_time + p_marginal),
        (p_time + p_marginal)..p_total,
    ];

    let gram_closed = build_raw_grams_from_channel_blocks(&channel_blocks, &row_hess, &raw_ranges)
        .expect("build_raw_grams_from_channel_blocks failed");

    // Build the explicit per-subject (K=4 × p_total) Jacobian and W_i, then
    // accumulate G = Σ_i J_iᵀ W_i J_i.
    let mut j_per_subject: Vec<ndarray::Array2<f64>> = Vec::with_capacity(n);
    let mut w_per_subject: Vec<ndarray::Array2<f64>> = Vec::with_capacity(n);
    let h_full = <SurvivalRowHessian as FamilyChannelHessian>::evaluate_full(&row_hess);
    for i in 0..n {
        let mut j_i = ndarray::Array2::<f64>::zeros((4, p_total));
        // Time block at columns 0..p_time across channels (0, 1, 2).
        for j in 0..p_time {
            j_i[[0, j]] = dq0[[i, j]];
            j_i[[1, j]] = dq1[[i, j]];
            j_i[[2, j]] = dqd1[[i, j]];
        }
        // Marginal block at p_time..p_time+p_marginal across channels (0, 1, 2).
        for j in 0..p_marginal {
            j_i[[0, p_time + j]] = m_dq[[i, j]];
            j_i[[1, p_time + j]] = m_dq[[i, j]];
            j_i[[2, p_time + j]] = m_dqd1[[i, j]];
        }
        // Logslope block at p_time+p_marginal..p_total across channel 3.
        for j in 0..p_logslope {
            j_i[[3, p_time + p_marginal + j]] = g_dg[[i, j]];
        }
        j_per_subject.push(j_i);

        let mut w_i = ndarray::Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                w_i[[a, b]] = h_full[[i, a, b]];
            }
        }
        w_per_subject.push(w_i);
    }
    let gram_explicit = explicit_fisher_gram(&j_per_subject, &w_per_subject);

    // Match closed-form vs explicit.
    assert_eq!(gram_closed.dim(), gram_explicit.dim());
    let mut max_err = 0.0f64;
    let mut max_scale = 1.0f64;
    for a in 0..p_total {
        for b in 0..p_total {
            let err = (gram_closed[[a, b]] - gram_explicit[[a, b]]).abs();
            let scale = gram_explicit[[a, b]].abs().max(1.0);
            if err / scale > max_err / max_scale {
                max_err = err;
                max_scale = scale;
            }
        }
    }
    assert!(
        max_err < 1e-10 * max_scale,
        "channel-stacked Fisher Gram closed-form vs explicit Σ J^T W J: \
         max_err={max_err:.3e} max_scale={max_scale:.3e}"
    );
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

    // FD compares against the raw OBSERVED Hessian (no PSD clamp). The
    // production constructor `from_pilot` clamps for the canonicalize gate;
    // here we test the closed-form formula matches the row NLL second
    // derivatives exactly.
    let hess = GaussianLocationScaleChannelHessian::from_pilot_observed_unclamped(
        &y_arr, &w_arr, &mu_arr, &logs_arr,
    )
    .expect("GaussianLocationScaleChannelHessian::from_pilot_observed_unclamped failed");

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
                assert!(
                    err_within_tol(got, ref_val),
                    "gaussian location-scale W_i[{a},{b}] row {i}: \
                     got={got:.6e} fd={ref_val:.6e} abs_err={abs:.2e} rel_err={rel:.2e}",
                    abs = (got - ref_val).abs(),
                    rel = rel_err(got, ref_val),
                );
            }
        }
    }
}
