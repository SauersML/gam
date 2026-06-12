//! Oracle: the O(n) state-space spline scan must reproduce, to near machine
//! precision, the EXACT dense posterior of the same intrinsic order-2 prior —
//! self-constructed truth (#904), solved here by independent dense linear
//! algebra (joint state precision assembled from the Markov form, Gaussian
//! elimination written in-test). No approximation tolerance budget: the two
//! paths compute the same Gaussian, so agreement is at solver roundoff.

use gam::solver::spline_scan::{fit_cubic_spline_scan, fit_cubic_spline_scan_at};

/// Dense in-test Gaussian elimination solve A·X = B (partial pivoting).
fn dense_solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        aug.swap(col, piv);
        let p = aug[col][col];
        assert!(p.abs() > 1e-300, "dense oracle: singular pivot");
        for i in 0..n {
            if i == col {
                continue;
            }
            let f = aug[i][col] / p;
            if f == 0.0 {
                continue;
            }
            for k in col..n + m {
                aug[i][k] -= f * aug[col][k];
            }
        }
    }
    (0..n)
        .map(|i| (0..m).map(|j| aug[i][n + j] / aug[i][i]).collect())
        .collect()
}

/// log det of an SPD matrix via the same elimination (product of pivots).
fn dense_logdet(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut logdet = 0.0;
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap();
        if piv != col {
            m.swap(col, piv);
            // Row swap flips the determinant sign; for SPD inputs the product
            // of pivots stays positive overall, and we only need log|det|.
        }
        let p = m[col][col];
        assert!(p.abs() > 1e-300, "dense oracle: singular logdet pivot");
        logdet += p.abs().ln();
        for i in col + 1..n {
            let f = m[i][col] / p;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                m[i][k] -= f * m[col][k];
            }
        }
    }
    logdet
}

/// Dense exact posterior of the SAME intrinsic order-2 Markov prior, built
/// independently: joint precision over states (f_t, f'_t), improper (zero)
/// prior on the first state = the diffuse null space.
struct DenseTruth {
    mean: Vec<f64>,
    var: Vec<f64>,
    /// Restricted log-likelihood up to a λ-independent additive constant
    /// (σ² fixed at 1): ½·logdet⁺(K_prior) − ½·logdet(Λ) − ½(yᵀWy − rᵀΛ⁻¹r).
    reml: f64,
}

fn dense_truth(x: &[f64], y: &[f64], w: &[f64], log_lambda: f64) -> DenseTruth {
    let m = x.len();
    let q = (-log_lambda).exp();
    let dim = 2 * m;
    let mut prior = vec![vec![0.0_f64; dim]; dim];
    // Markov increments: (α_{t+1} − F α_t)ᵀ (q·Q(δ))⁻¹ (α_{t+1} − F α_t).
    for t in 0..m - 1 {
        let d = x[t + 1] - x[t];
        let (d2, d3) = (d * d, d * d * d);
        // Q = q·[[d³/3, d²/2],[d²/2, d]];  Q⁻¹ via 2×2 inverse.
        let (a, b, c) = (q * d3 / 3.0, q * d2 / 2.0, q * d);
        let det = a * c - b * b;
        let qi = [[c / det, -b / det], [-b / det, a / det]];
        // Rows of T = [−F  I] acting on (α_t, α_{t+1}); F = [[1,d],[0,1]].
        // Column layout: [f_t, f'_t, f_{t+1}, f'_{t+1}].
        let trows = [[-1.0, -d, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]];
        for r1 in 0..2 {
            for r2 in 0..2 {
                let coef = qi[r1][r2];
                if coef == 0.0 {
                    continue;
                }
                for c1 in 0..4 {
                    if trows[r1][c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..4 {
                        if trows[r2][c2] == 0.0 {
                            continue;
                        }
                        prior[2 * t + c1][2 * t + c2] += trows[r1][c1] * coef * trows[r2][c2];
                    }
                }
            }
        }
    }
    // Posterior precision and RHS: observations on the f components.
    let mut lambda = prior.clone();
    let mut rhs = vec![vec![0.0_f64]; dim];
    let mut ytwy = 0.0;
    for t in 0..m {
        lambda[2 * t][2 * t] += w[t];
        rhs[2 * t][0] = w[t] * y[t];
        ytwy += w[t] * y[t] * y[t];
    }
    let mean_full = dense_solve(&lambda, &rhs);
    // Posterior covariance diagonal: solve Λ X = I and read f-diagonals.
    let eye: Vec<Vec<f64>> = (0..dim)
        .map(|i| (0..dim).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    let cov = dense_solve(&lambda, &eye);
    let mean: Vec<f64> = (0..m).map(|t| mean_full[2 * t][0]).collect();
    let var: Vec<f64> = (0..m).map(|t| cov[2 * t][2 * t]).collect();
    // Restricted loglik (σ²=1, up to λ-free constant):
    //   logdet⁺(prior) = (2m−2)·log(q⁻¹)·(−1) + const  — prior ∝ q⁻¹·K₀ of
    //   rank 2m−2, so logdet⁺ = (2m−2)·ln(1/q) + logdet⁺(K₀); the K₀ part is
    //   λ-free and dropped (the test compares DIFFERENCES across λ).
    let rt_lam_r: f64 = (0..dim).map(|i| rhs[i][0] * mean_full[i][0]).sum();
    let reml = 0.5 * (2 * m - 2) as f64 * (1.0 / q).ln() - 0.5 * dense_logdet(&lambda)
        - 0.5 * (ytwy - rt_lam_r);
    DenseTruth { mean, var, reml }
}

fn test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Deterministic, irregular abscissae with a data gap (bridge regime) and
    // heteroskedastic weights; smooth truth + fixed pseudo-noise.
    let n = 80usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        // Irregular spacing with a hole in (0.45, 0.7).
        let xi = if u < 0.45 { u } else { 0.7 + (u - 0.45) * 0.55 };
        let truth = (6.0 * xi).sin() + 0.5 * xi * xi;
        // Fixed quasi-random noise (golden-ratio rotation, no RNG).
        let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 0.3;
        x.push(xi);
        y.push(truth + noise);
        w.push(1.0 + 0.5 * ((i % 3) as f64));
    }
    (x, y, w)
}

#[test]
fn scan_matches_dense_exact_posterior_at_fixed_lambda() {
    let (x, y, w) = test_data();
    for &log_lambda in &[-2.0_f64, 1.5, 5.0] {
        let scan = fit_cubic_spline_scan_at(&x, &y, &w, log_lambda, Some(1.0))
            .expect("scan fit at fixed lambda");
        let truth = dense_truth(&x, &y, &w, log_lambda);
        for t in 0..x.len() {
            let dm = (scan.mean[t] - truth.mean[t]).abs();
            let scale_m = truth.mean[t].abs().max(1e-3);
            assert!(
                dm <= 1e-8 * scale_m,
                "posterior mean mismatch at knot {t} (logλ={log_lambda}): scan={} dense={}",
                scan.mean[t],
                truth.mean[t]
            );
            let dv = (scan.var[t] - truth.var[t]).abs();
            assert!(
                dv <= 1e-7 * truth.var[t].max(1e-12),
                "posterior variance mismatch at knot {t} (logλ={log_lambda}): scan={} dense={}",
                scan.var[t],
                truth.var[t]
            );
        }
    }
}

#[test]
fn scan_restricted_loglik_differences_match_dense_reml() {
    let (x, y, w) = test_data();
    let (ll_a, ll_b) = (-1.0_f64, 3.0_f64);
    let scan_a = fit_cubic_spline_scan_at(&x, &y, &w, ll_a, Some(1.0)).expect("scan a");
    let scan_b = fit_cubic_spline_scan_at(&x, &y, &w, ll_b, Some(1.0)).expect("scan b");
    let dense_a = dense_truth(&x, &y, &w, ll_a);
    let dense_b = dense_truth(&x, &y, &w, ll_b);
    // With σ² fixed at 1 the scan criterion is −½(Σ log F + Σ v²/F); both
    // sides carry their own λ-free additive constants, so compare DIFFERENCES.
    let scan_diff = scan_a.restricted_loglik - scan_b.restricted_loglik;
    let dense_diff = dense_a.reml - dense_b.reml;
    assert!(
        (scan_diff - dense_diff).abs() <= 1e-7 * dense_diff.abs().max(1.0),
        "REML criterion difference mismatch: scan {scan_diff} vs dense {dense_diff}"
    );
}

#[test]
fn scan_bridges_gap_and_grows_variance() {
    let (x, y, w) = test_data();
    let fit = fit_cubic_spline_scan(&x, &y, &w).expect("REML-selected scan fit");
    // The hole (0.45, 0.7) — the bridge must pass smoothly between the flanks
    // (no sag to the data mean) and the variance must peak inside the gap.
    let (m_mid, v_mid) = fit.predict(0.575).expect("gap midpoint");
    let (m_left, v_left) = fit.predict(0.44).expect("left flank");
    let (m_right, v_right) = fit.predict(0.71).expect("right flank");
    assert!(
        v_mid > v_left && v_mid > v_right,
        "gap variance must exceed flank variance: mid={v_mid}, left={v_left}, right={v_right}"
    );
    let lo = m_left.min(m_right) - 0.75;
    let hi = m_left.max(m_right) + 0.75;
    assert!(
        m_mid > lo && m_mid < hi,
        "bridge mean {m_mid} should stay near the flank envelope [{lo}, {hi}]"
    );
    // Off-knot prediction at a knot reproduces the knot posterior exactly.
    let (m_k, v_k) = fit.predict(fit.knots[10]).expect("at-knot predict");
    assert!((m_k - fit.mean[10]).abs() < 1e-12 && (v_k - fit.var[10]).abs() < 1e-12);
}
