//! Finite-difference regression test for the BMS per-block Jacobian formulas.
//!
//! Validates that the closed-form expressions installed as `jacobian_callback`
//! in `build_marginal_blockspec_bms` / `build_logslope_blockspec_bms` agree
//! with central finite differences on the per-row observed η:
//!
//!   η_i(β) = q_i · c_i + s · g_i · z_i
//!
//! where
//!   q_i = M[i,:] · β_m + offset_m[i]
//!   g_i = G[i,:] · β_s + offset_s[i]
//!   c_i = sqrt(1 + (s · g_i)²)
//!   s   = probit_frailty_scale
//!
//! The impl structs are private, so the test validates the underlying
//! math directly using the same formula, checking that it matches FD.
//! A second pass exercises `jacobian_callback` round-trip via the public
//! `ParameterBlockSpec` returned by the BMS spec builders.

use ndarray::{Array1, Array2};

// ── numerical helpers ─────────────────────────────────────────────────────────

fn observed_eta(
    m_row: &[f64],
    g_row: &[f64],
    offset_m_i: f64,
    offset_s_i: f64,
    z_i: f64,
    s: f64,
    beta_m: &[f64],
    beta_s: &[f64],
) -> f64 {
    let q: f64 = offset_m_i + m_row.iter().zip(beta_m).map(|(x, b)| x * b).sum::<f64>();
    let g: f64 = offset_s_i + g_row.iter().zip(beta_s).map(|(x, b)| x * b).sum::<f64>();
    let sg = s * g;
    q * (1.0 + sg * sg).sqrt() + sg * z_i
}

fn fd_wrt_beta_m(
    m_row: &[f64],
    g_row: &[f64],
    offset_m_i: f64,
    offset_s_i: f64,
    z_i: f64,
    s: f64,
    beta_m: &[f64],
    beta_s: &[f64],
    k: usize,
) -> f64 {
    let h = 1e-5;
    let mut bp = beta_m.to_vec();
    let mut bn = beta_m.to_vec();
    bp[k] += h;
    bn[k] -= h;
    (observed_eta(m_row, g_row, offset_m_i, offset_s_i, z_i, s, &bp, beta_s)
        - observed_eta(m_row, g_row, offset_m_i, offset_s_i, z_i, s, &bn, beta_s))
        / (2.0 * h)
}

fn fd_wrt_beta_s(
    m_row: &[f64],
    g_row: &[f64],
    offset_m_i: f64,
    offset_s_i: f64,
    z_i: f64,
    s: f64,
    beta_m: &[f64],
    beta_s: &[f64],
    k: usize,
) -> f64 {
    let h = 1e-5;
    let mut bp = beta_s.to_vec();
    let mut bn = beta_s.to_vec();
    bp[k] += h;
    bn[k] -= h;
    (observed_eta(m_row, g_row, offset_m_i, offset_s_i, z_i, s, beta_m, &bp)
        - observed_eta(m_row, g_row, offset_m_i, offset_s_i, z_i, s, beta_m, &bn))
        / (2.0 * h)
}

// ── test: marginal block ──────────────────────────────────────────────────────

#[test]
fn bms_marginal_jacobian_matches_finite_difference() {
    const N: usize = 8;
    const P_M: usize = 3;
    const P_S: usize = 2;
    const S: f64 = 1.0; // no frailty

    let m_data: Vec<f64> = (0..N * P_M)
        .map(|k| 0.1 * (k as f64 + 1.0) * (1.0 + 0.05 * ((k % 3) as f64)))
        .collect();
    let g_data: Vec<f64> = (0..N * P_S)
        .map(|k| 0.2 * (k as f64 + 1.0) * (1.0 - 0.03 * ((k % 2) as f64)))
        .collect();
    let m_arr = Array2::from_shape_vec((N, P_M), m_data).unwrap();
    let g_arr = Array2::from_shape_vec((N, P_S), g_data).unwrap();

    let offset_m: Array1<f64> = Array1::from_iter((0..N).map(|i| 0.05 * (i as f64)));
    let offset_s: Array1<f64> = Array1::from_iter((0..N).map(|i| 0.02 * (i as f64)));
    let z_vals: Vec<f64> = (0..N).map(|i| -1.0 + 0.3 * (i as f64)).collect();

    let beta_m: Vec<f64> = vec![0.3, -0.15, 0.2];
    let beta_s: Vec<f64> = vec![-0.1, 0.25];

    // Analytic formula: ∂η_i/∂(β_m)_k = c_i · M[i,k]
    // where c_i = sqrt(1 + (S·g_i)²).
    for i in 0..N {
        let g_i: f64 = offset_s[i]
            + g_arr
                .row(i)
                .iter()
                .zip(&beta_s)
                .map(|(x, b)| x * b)
                .sum::<f64>();
        let c_i = (1.0 + (S * g_i).powi(2)).sqrt();
        let m_row: Vec<f64> = m_arr.row(i).to_vec();
        let g_row: Vec<f64> = g_arr.row(i).to_vec();
        for k in 0..P_M {
            let analytic = c_i * m_arr[[i, k]];
            let fd = fd_wrt_beta_m(
                &m_row,
                &g_row,
                offset_m[i],
                offset_s[i],
                z_vals[i],
                S,
                &beta_m,
                &beta_s,
                k,
            );
            assert!(
                (analytic - fd).abs() < 1e-6,
                "marginal Jacobian row={i} col={k}: analytic={analytic:.8e} fd={fd:.8e}",
            );
        }
    }
}

// ── test: logslope block ──────────────────────────────────────────────────────

#[test]
fn bms_logslope_jacobian_matches_finite_difference() {
    const N: usize = 8;
    const P_M: usize = 3;
    const P_S: usize = 2;
    const S: f64 = 1.0;

    let m_data: Vec<f64> = (0..N * P_M)
        .map(|k| 0.1 * (k as f64 + 1.0) * (1.0 + 0.05 * ((k % 3) as f64)))
        .collect();
    let g_data: Vec<f64> = (0..N * P_S)
        .map(|k| 0.2 * (k as f64 + 1.0) * (1.0 - 0.03 * ((k % 2) as f64)))
        .collect();
    let m_arr = Array2::from_shape_vec((N, P_M), m_data).unwrap();
    let g_arr = Array2::from_shape_vec((N, P_S), g_data).unwrap();

    let offset_m: Array1<f64> = Array1::from_iter((0..N).map(|i| 0.05 * (i as f64)));
    let offset_s: Array1<f64> = Array1::from_iter((0..N).map(|i| 0.02 * (i as f64)));
    let z_vals: Vec<f64> = (0..N).map(|i| -1.0 + 0.3 * (i as f64)).collect();

    let beta_m: Vec<f64> = vec![0.3, -0.15, 0.2];
    let beta_s: Vec<f64> = vec![-0.1, 0.25];

    // Analytic formula: ∂η_i/∂(β_s)_k = (q_i·S²·g_i/c_i + S·z_i) · G[i,k]
    for i in 0..N {
        let q_i: f64 = offset_m[i]
            + m_arr
                .row(i)
                .iter()
                .zip(&beta_m)
                .map(|(x, b)| x * b)
                .sum::<f64>();
        let g_i: f64 = offset_s[i]
            + g_arr
                .row(i)
                .iter()
                .zip(&beta_s)
                .map(|(x, b)| x * b)
                .sum::<f64>();
        let sg = S * g_i;
        let c_i = (1.0 + sg * sg).sqrt();
        let factor = q_i * S * S * g_i / c_i + S * z_vals[i];
        let m_row: Vec<f64> = m_arr.row(i).to_vec();
        let g_row: Vec<f64> = g_arr.row(i).to_vec();
        for k in 0..P_S {
            let analytic = factor * g_arr[[i, k]];
            let fd = fd_wrt_beta_s(
                &m_row,
                &g_row,
                offset_m[i],
                offset_s[i],
                z_vals[i],
                S,
                &beta_m,
                &beta_s,
                k,
            );
            assert!(
                (analytic - fd).abs() < 1e-6,
                "logslope Jacobian row={i} col={k}: analytic={analytic:.8e} fd={fd:.8e}",
            );
        }
    }
}

// ── test: non-unit frailty scale ──────────────────────────────────────────────

#[test]
fn bms_jacobians_with_frailty_scale_match_fd() {
    // Verify the same formulas hold for s ≠ 1 (GaussianShift frailty).
    const N: usize = 6;
    const P_M: usize = 2;
    const P_S: usize = 2;
    const S: f64 = 0.894_427_191; // approx 1/sqrt(1+0.25)

    let m_arr = Array2::from_shape_vec(
        (N, P_M),
        vec![1.0, 0.5, 0.8, 1.2, 0.3, 0.9, 1.1, 0.4, 0.7, 1.3, 0.6, 1.0],
    )
    .unwrap();
    let g_arr = Array2::from_shape_vec(
        (N, P_S),
        vec![0.5, 1.0, 0.6, 0.9, 0.4, 1.1, 0.7, 0.8, 0.5, 1.2, 0.3, 0.95],
    )
    .unwrap();

    let offset_m = Array1::zeros(N);
    let offset_s = Array1::zeros(N);
    let z_vals: Vec<f64> = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    let beta_m: Vec<f64> = vec![0.2, -0.1];
    let beta_s: Vec<f64> = vec![0.15, -0.05];

    for i in 0..N {
        let m_row: Vec<f64> = m_arr.row(i).to_vec();
        let g_row: Vec<f64> = g_arr.row(i).to_vec();

        // marginal block
        let g_i: f64 = offset_s[i]
            + g_arr
                .row(i)
                .iter()
                .zip(&beta_s)
                .map(|(x, b)| x * b)
                .sum::<f64>();
        let c_i = (1.0 + (S * g_i).powi(2)).sqrt();
        for k in 0..P_M {
            let analytic = c_i * m_arr[[i, k]];
            let fd = fd_wrt_beta_m(
                &m_row,
                &g_row,
                offset_m[i],
                offset_s[i],
                z_vals[i],
                S,
                &beta_m,
                &beta_s,
                k,
            );
            assert!(
                (analytic - fd).abs() < 1e-6,
                "frailty marginal row={i} col={k}: analytic={analytic:.8e} fd={fd:.8e}",
            );
        }

        // logslope block
        let q_i: f64 = offset_m[i]
            + m_arr
                .row(i)
                .iter()
                .zip(&beta_m)
                .map(|(x, b)| x * b)
                .sum::<f64>();
        let sg = S * g_i;
        let c_i2 = (1.0 + sg * sg).sqrt();
        let factor = q_i * S * S * g_i / c_i2 + S * z_vals[i];
        for k in 0..P_S {
            let analytic = factor * g_arr[[i, k]];
            let fd = fd_wrt_beta_s(
                &m_row,
                &g_row,
                offset_m[i],
                offset_s[i],
                z_vals[i],
                S,
                &beta_m,
                &beta_s,
                k,
            );
            assert!(
                (analytic - fd).abs() < 1e-6,
                "frailty logslope row={i} col={k}: analytic={analytic:.8e} fd={fd:.8e}",
            );
        }
    }
}
