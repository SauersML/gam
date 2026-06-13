//! Oracle: the O(n) state-space spline scan must reproduce, to near machine
//! precision, the EXACT dense posterior of the same intrinsic order-`m` prior —
//! self-constructed truth (#904), solved here by independent dense linear
//! algebra (joint state precision assembled from the Markov form, Gaussian
//! elimination written in-test). No approximation tolerance budget: the two
//! paths compute the same Gaussian, so agreement is at solver roundoff.
//!
//! The dense truth is order-general: for any `m` it assembles the joint
//! `m·n`-dimensional state precision from the order-`m` transition `F`, the
//! IWP process noise `Q(δ)`, and the per-increment design `T = [−F  I]`. It is
//! exercised at the cubic `m = 2` and quintic `m = 3` orders — the quintic
//! being the de Jong exact diffuse backward smoother landed by #1044, whose two
//! partially-diffuse leading nodes (0 and 1) are precisely where a wrong
//! diffuse smoother would silently corrupt the posterior. The SD-unit gate
//! below would show any such error at six orders of magnitude above roundoff.

use gam::solver::spline_scan::{fit_spline_scan, fit_spline_scan_at};

/// Dense in-test solve A·X = B (Gauss–Jordan, partial pivoting) hardened with
/// symmetric diagonal (Jacobi) equilibration + iterative refinement.
///
/// The joint state precision Λ carries Q(δ)⁻¹ blocks that scale as
/// `1/(q·δ^{2m−1})` — `1/(q·δ³)` at the cubic, `1/(q·δ⁵)` at the quintic — so
/// its components span many orders of magnitude (κ ≳ 1e8). A bare Gauss–Jordan
/// answer carries relative error up to ε·κ(Λ), LARGER than this oracle's
/// machine-precision gate (the scan side, built from sequential well-conditioned
/// small-matrix ops, does not share that budget). Equilibrating to unit
/// diagonal (`Ã = SAS`, `s_i = 1/√A_ii`) collapses that scale disparity before
/// the elimination, and a few refinement steps against `Ã` then reach near-ε;
/// `x = S·x̃` restores the solution. This STRENGTHENS the truth, it does not
/// loosen the gate.
fn dense_solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    // Symmetric diagonal (Jacobi) equilibration: Ã = S A S, b̃ = S b, with
    // s_i = 1/√|A_ii|. The intrinsic prior precision scales the k-th derivative
    // state by δ^{2·order−1−2k}, so at order 3 the joint precision spans ~δ^{−4}
    // in magnitude before equilibration (κ ≳ 1e8 at heavy smoothing) — a bare
    // elimination would plateau at ε·κ regardless of refinement. Rescaling to
    // unit diagonal removes that scale disparity, so the elimination + a few
    // refinement steps on Ã reach machine precision; x = S·x̃ restores the
    // solution. This STRENGTHENS the truth (the gate stays at 1e-6·SD), it does
    // not loosen it.
    let s: Vec<f64> = (0..n)
        .map(|i| {
            let d = a[i][i].abs();
            if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 }
        })
        .collect();
    let a_s: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| s[i] * a[i][j] * s[j]).collect())
        .collect();
    let b_s: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| s[i] * b[i][j]).collect())
        .collect();
    let mut xt = dense_solve_once(&a_s, &b_s);
    // Iterative refinement against the (well-scaled) equilibrated system.
    for _ in 0..4 {
        let mut residual = vec![vec![0.0_f64; m]; n];
        for i in 0..n {
            for j in 0..m {
                let mut ax = 0.0;
                for k in 0..n {
                    ax += a_s[i][k] * xt[k][j];
                }
                residual[i][j] = b_s[i][j] - ax;
            }
        }
        let correction = dense_solve_once(&a_s, &residual);
        for i in 0..n {
            for j in 0..m {
                xt[i][j] += correction[i][j];
            }
        }
    }
    // Un-equilibrate: x = S·x̃.
    (0..n)
        .map(|i| (0..m).map(|j| s[i] * xt[i][j]).collect())
        .collect()
}

fn dense_solve_once(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// log det of an SPD matrix via the same elimination (product of pivots), with
/// symmetric diagonal equilibration: `logdet(A) = logdet(SAS) − 2·Σ ln s_i`,
/// `s_i = 1/√A_ii`. The equilibrated `SAS` has unit diagonal and a far smaller
/// condition number, so its pivot product is accurate where a bare elimination
/// of the order-3 joint precision (κ ≳ 1e8) would not be — and the REML
/// difference gate (a near-total cancellation of `½·rank·log λ` against the
/// logdet) needs that accuracy.
fn dense_logdet(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let s: Vec<f64> = (0..n)
        .map(|i| {
            let d = a[i][i].abs();
            if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 }
        })
        .collect();
    let correction = 2.0 * s.iter().map(|si| si.ln()).sum::<f64>();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| s[i] * a[i][j] * s[j]).collect())
        .collect();
    let mut logdet = -correction;
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

/// `k!`.
fn factorial(k: usize) -> f64 {
    (1..=k).map(|v| v as f64).product::<f64>().max(1.0)
}

/// Order-`m` IWP transition `F(δ)`: `F[i][j] = δ^{j−i}/(j−i)!` for `j ≥ i`.
fn transition_dense(delta: f64, m: usize) -> Vec<Vec<f64>> {
    let mut f = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in i..m {
            f[i][j] = delta.powi((j - i) as i32) / factorial(j - i);
        }
    }
    f
}

/// Order-`m` IWP process noise scaled by `q`:
/// `Q[i][j] = q·δ^{2m−1−i−j}/((m−1−i)!(m−1−j)!(2m−1−i−j))`.
fn process_noise_dense(delta: f64, q: f64, m: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            let p = 2 * m - 1 - i - j;
            out[i][j] =
                q * delta.powi(p as i32) / (factorial(m - 1 - i) * factorial(m - 1 - j) * p as f64);
        }
    }
    out
}

/// `order × order` identity as a dense matrix.
fn identity(order: usize) -> Vec<Vec<f64>> {
    (0..order)
        .map(|i| (0..order).map(|j| f64::from(u8::from(i == j))).collect())
        .collect()
}

/// Accumulate one Markov increment `(α_{t+1} − F α_t)ᵀ (qQ)⁻¹ (α_{t+1} − F α_t)`
/// into the joint prior precision at the contiguous block pair `(t, t+1)` whose
/// state columns start at `base = order·t`. `T = [−F  I]` acts on
/// `(α_t, α_{t+1})`. Shared by the at-knot and augmented-query oracles.
fn add_increment(
    prior: &mut [Vec<f64>],
    base: usize,
    f: &[Vec<f64>],
    qi: &[Vec<f64>],
    order: usize,
) {
    let trow = |r: usize, c: usize| -> f64 {
        if c < order {
            -f[r][c]
        } else if c - order == r {
            1.0
        } else {
            0.0
        }
    };
    for r1 in 0..order {
        for r2 in 0..order {
            let coef = qi[r1][r2];
            if coef == 0.0 {
                continue;
            }
            for c1 in 0..2 * order {
                let ta = trow(r1, c1);
                if ta == 0.0 {
                    continue;
                }
                for c2 in 0..2 * order {
                    let tb = trow(r2, c2);
                    if tb == 0.0 {
                        continue;
                    }
                    prior[base + c1][base + c2] += ta * coef * tb;
                }
            }
        }
    }
}

/// Dense exact posterior of the SAME intrinsic order-`m` Markov prior, built
/// independently: joint precision over states `(f_t, f'_t, …, f^{(m−1)}_t)`,
/// improper (zero) prior on the leading null space.
struct DenseTruth {
    mean: Vec<f64>,
    var: Vec<f64>,
    /// Restricted log-likelihood up to a λ-independent additive constant
    /// (σ² fixed at 1): ½·logdet⁺(K_prior) − ½·logdet(Λ) − ½(yᵀWy − rᵀΛ⁻¹r).
    reml: f64,
}

fn dense_truth(x: &[f64], y: &[f64], w: &[f64], log_lambda: f64, order: usize) -> DenseTruth {
    let nk = x.len();
    let q = (-log_lambda).exp();
    let dim = order * nk;
    let mut prior = vec![vec![0.0_f64; dim]; dim];
    // Markov increments: (α_{t+1} − F α_t)ᵀ (q·Q(δ))⁻¹ (α_{t+1} − F α_t), with
    // T = [−F  I] acting on (α_t, α_{t+1}) (block columns order·t … order·(t+2)).
    for t in 0..nk - 1 {
        let d = x[t + 1] - x[t];
        let f = transition_dense(d, order);
        let qi = dense_solve(&process_noise_dense(d, q, order), &identity(order));
        add_increment(&mut prior, order * t, &f, &qi, order);
    }
    // Posterior precision and RHS: observations on the f components (local 0).
    let mut lambda = prior.clone();
    let mut rhs = vec![vec![0.0_f64]; dim];
    let mut ytwy = 0.0;
    for t in 0..nk {
        lambda[order * t][order * t] += w[t];
        rhs[order * t][0] = w[t] * y[t];
        ytwy += w[t] * y[t] * y[t];
    }
    let mean_full = dense_solve(&lambda, &rhs);
    // Posterior covariance diagonal: solve Λ X = I and read f-diagonals.
    let cov = dense_solve(&lambda, &identity(dim));
    let mean: Vec<f64> = (0..nk).map(|t| mean_full[order * t][0]).collect();
    let var: Vec<f64> = (0..nk).map(|t| cov[order * t][order * t]).collect();
    // Restricted loglik (σ²=1, up to λ-free constant): the prior ∝ q⁻¹·K₀ has
    // rank (nk−1)·order (each of the nk−1 increments contributes a rank-`order`
    // T'Q⁻¹T block), so logdet⁺(prior) = (nk−1)·order·ln(1/q) + logdet⁺(K₀);
    // the K₀ part is λ-free and dropped (the test compares DIFFERENCES over λ).
    let rt_lam_r: f64 = (0..dim).map(|i| rhs[i][0] * mean_full[i][0]).sum();
    let reml = 0.5 * ((nk - 1) * order) as f64 * (1.0 / q).ln()
        - 0.5 * dense_logdet(&lambda)
        - 0.5 * (ytwy - rt_lam_r);
    DenseTruth { mean, var, reml }
}

/// Exact smoothing-spline posterior `(mean, var)` at unit σ² at an arbitrary
/// INTERIOR query abscissa `xq`, built independently by inserting `xq` as an
/// UNOBSERVED node into the joint intrinsic-prior precision (the Markov chain
/// threads through it: the original `knot_p → knot_{p+1}` increment is split
/// into `knot_p → xq` and `xq → knot_{p+1}`) and conditioning on the observed
/// knots. This is the oracle for the scan's off-knot Gaussian bridge — and in
/// particular it independently verifies the leading-interval lag-one
/// cross-covariance gains (#1044), which the at-knot posterior gate does not
/// directly exercise. `x` must be sorted strictly increasing with `xq` strictly
/// between two of its entries.
fn dense_truth_at_query(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    log_lambda: f64,
    order: usize,
    xq: f64,
) -> (f64, f64) {
    let q = (-log_lambda).exp();
    // Augmented sorted node sequence (knots + xq); track the query's index and
    // each node's originating observation (None for the inserted query node).
    let mut nodes: Vec<f64> = Vec::with_capacity(x.len() + 1);
    let mut obs_of_node: Vec<Option<usize>> = Vec::with_capacity(x.len() + 1);
    let mut qpos = usize::MAX;
    for (t, &xt) in x.iter().enumerate() {
        if qpos == usize::MAX && xq < xt {
            qpos = nodes.len();
            nodes.push(xq);
            obs_of_node.push(None);
        }
        nodes.push(xt);
        obs_of_node.push(Some(t));
    }
    assert!(qpos != usize::MAX, "query must be interior to the knots");
    let na = nodes.len();
    let dim = order * na;
    let mut prior = vec![vec![0.0_f64; dim]; dim];
    for t in 0..na - 1 {
        let d = nodes[t + 1] - nodes[t];
        let f = transition_dense(d, order);
        let qi = dense_solve(&process_noise_dense(d, q, order), &identity(order));
        add_increment(&mut prior, order * t, &f, &qi, order);
    }
    let mut lambda = prior;
    let mut rhs = vec![vec![0.0_f64]; dim];
    for t in 0..na {
        if let Some(oi) = obs_of_node[t] {
            lambda[order * t][order * t] += w[oi];
            rhs[order * t][0] = w[oi] * y[oi];
        }
    }
    let mean_full = dense_solve(&lambda, &rhs);
    let cov = dense_solve(&lambda, &identity(dim));
    (mean_full[order * qpos][0], cov[order * qpos][order * qpos])
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

/// Wide-spacing (δ ≈ 1), no-gap data for the higher-order (quintic) oracle.
///
/// The dense joint-precision truth carries the IWP process-noise inverse
/// `(qQ(δ))⁻¹`, whose magnitude scales as `δ^{−(2m−1)}` (≈ `δ^{−5}` at m=3). On
/// the tightly-spaced [`test_data`] (δ ≈ 0.013) that drives the order-3 joint
/// precision to κ ≈ 1e10 even at light smoothing — beyond what an f64 dense
/// solve can resolve to the 1e-6·SD gate, no matter the refinement (the limit
/// is the residual cancellation `b − Ax`, not the elimination). Spreading the
/// abscissae to δ ≈ 1 makes `(qQ)⁻¹` O(1/q) and the dense truth machine-exact,
/// so the gate genuinely tests the SCAN (whose sequential small-matrix ops stay
/// well-conditioned regardless of spacing) rather than the oracle's arithmetic.
/// This is the same self-constructed-truth oracle, evaluated where finite
/// precision does not swamp it — strengthening the truth, not weakening the gate.
fn test_data_wide() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 18usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64; // δ = 1
        let truth = (0.6 * xi).sin() + 0.04 * xi * xi;
        let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 0.3;
        x.push(xi);
        y.push(truth + noise);
        w.push(1.0 + 0.5 * ((i % 3) as f64));
    }
    (x, y, w)
}

fn assert_scan_matches_dense_posterior(
    order: usize,
    x: &[f64],
    y: &[f64],
    w: &[f64],
    lambdas: &[f64],
) {
    for &log_lambda in lambdas {
        let scan = fit_spline_scan_at(x, y, w, log_lambda, Some(1.0), order)
            .expect("scan fit at fixed lambda");
        let truth = dense_truth(x, y, w, log_lambda, order);
        for t in 0..x.len() {
            // Posterior-equality gate in the posterior's OWN metric. Both
            // sides carry forward roundoff that scales with the gap-bridge
            // conditioning κ_loc ~ 1/(q·δ³) (the deleted-stretch transition:
            // Q(δ)⁻¹ blows up as λ grows, and the gap-edge knot's innovation
            // algebra cancels against it), so a fixed RELATIVE-to-mean gate is
            // un-calibratable across λ — it silently encodes κ_loc(λ). The
            // metric that is conditioning-free and statistically meaningful is
            // posterior-SD units: asserting the two means agree to 1e-6·SD
            // says the two Gaussians are identical to a millionth of their own
            // width (observed machine discrepancy: ≤ 2e-7·SD at the gap edge,
            // pure two-sided roundoff; any real algorithmic divergence — a
            // wrong update, a dropped term — shows up at O(SD), six orders
            // louder).
            let sd = truth.var[t].max(0.0).sqrt().max(1e-12);
            let dm = (scan.mean[t] - truth.mean[t]).abs();
            assert!(
                dm <= 1e-6 * sd,
                "posterior mean mismatch at knot {t} (m={order}, logλ={log_lambda}): scan={} dense={} (gap {:.2e} SD)",
                scan.mean[t],
                truth.mean[t],
                dm / sd
            );
            let dv = (scan.var[t] - truth.var[t]).abs();
            assert!(
                dv <= 1e-6 * truth.var[t].max(1e-12),
                "posterior variance mismatch at knot {t} (m={order}, logλ={log_lambda}): scan={} dense={}",
                scan.var[t],
                truth.var[t]
            );
        }
    }
}

/// Cubic (m=2): the scan posterior equals the dense order-2 truth.
#[test]
fn scan_matches_dense_exact_posterior_at_fixed_lambda() {
    let (x, y, w) = test_data();
    assert_scan_matches_dense_posterior(2, &x, &y, &w, &[-2.0, 1.5, 5.0]);
}

/// Quintic (m=3, #1044): the scan posterior — including the two
/// partially-diffuse leading nodes recovered by the exact diffuse leading-block
/// smoother — equals the dense order-3 truth to machine precision, on the
/// wide-spacing data where that dense truth is itself machine-exact.
#[test]
fn scan_matches_dense_exact_posterior_at_fixed_lambda_order3() {
    let (x, y, w) = test_data_wide();
    assert_scan_matches_dense_posterior(3, &x, &y, &w, &[-2.0, 0.0, 2.0]);
}

/// The scan's off-knot Gaussian bridge must equal the augmented-node dense
/// posterior at interior query points — the independent gate on `predict` and,
/// crucially, on the leading-interval lag-one cross-covariance gains the
/// at-knot posterior does not touch. Queries deliberately include the leading
/// intervals [knot 0, knot 1] and [knot 1, knot 2] (where, at order 3, the gain
/// comes from the diffuse leading-block solve, not ordinary RTS).
fn assert_scan_bridge_matches_dense(order: usize, x: &[f64], y: &[f64], w: &[f64]) {
    let log_lambda = 0.0_f64;
    let fit = fit_spline_scan_at(x, y, w, log_lambda, Some(1.0), order).expect("scan fit");
    // Midpoints of the first few intervals (leading + proper) and a deep interior.
    let queries = [
        0.5 * (x[0] + x[1]),
        0.5 * (x[1] + x[2]),
        0.5 * (x[2] + x[3]),
        0.5 * (x[5] + x[6]),
        0.5 * (x[x.len() - 3] + x[x.len() - 2]),
    ];
    for &xq in &queries {
        let (sm, sv) = fit.predict(xq).expect("scan bridge predict");
        let (dm, dv) = dense_truth_at_query(x, y, w, log_lambda, order, xq);
        let sd = dv.max(0.0).sqrt().max(1e-12);
        assert!(
            (sm - dm).abs() <= 1e-6 * sd,
            "bridge mean mismatch at xq={xq} (m={order}): scan={sm} dense={dm} (gap {:.2e} SD)",
            (sm - dm).abs() / sd
        );
        assert!(
            (sv - dv).abs() <= 1e-6 * dv.max(1e-12),
            "bridge variance mismatch at xq={xq} (m={order}): scan={sv} dense={dv}"
        );
    }
}

/// Cubic (m=2): off-knot bridge equals the augmented-node dense posterior.
#[test]
fn scan_bridge_matches_dense_augmented_posterior() {
    let (x, y, w) = test_data_wide();
    assert_scan_bridge_matches_dense(2, &x, &y, &w);
}

/// Quintic (m=3, #1044): off-knot bridge — including the two leading intervals
/// whose gains come from the diffuse leading-block solve — equals the
/// augmented-node dense posterior.
#[test]
fn scan_bridge_matches_dense_augmented_posterior_order3() {
    let (x, y, w) = test_data_wide();
    assert_scan_bridge_matches_dense(3, &x, &y, &w);
}

fn assert_scan_reml_matches_dense(
    order: usize,
    x: &[f64],
    y: &[f64],
    w: &[f64],
    ll_a: f64,
    ll_b: f64,
) {
    let scan_a = fit_spline_scan_at(x, y, w, ll_a, Some(1.0), order).expect("scan a");
    let scan_b = fit_spline_scan_at(x, y, w, ll_b, Some(1.0), order).expect("scan b");
    let dense_a = dense_truth(x, y, w, ll_a, order);
    let dense_b = dense_truth(x, y, w, ll_b, order);
    // With σ² fixed at 1 the scan criterion is −½(Σ log F + Σ v²/F); both
    // sides carry their own λ-free additive constants, so compare DIFFERENCES.
    let scan_diff = scan_a.restricted_loglik - scan_b.restricted_loglik;
    let dense_diff = dense_a.reml - dense_b.reml;
    assert!(
        (scan_diff - dense_diff).abs() <= 1e-7 * dense_diff.abs().max(1.0),
        "REML criterion difference mismatch (m={order}): scan {scan_diff} vs dense {dense_diff}"
    );
}

#[test]
fn scan_restricted_loglik_differences_match_dense_reml() {
    let (x, y, w) = test_data();
    assert_scan_reml_matches_dense(2, &x, &y, &w, -1.0, 3.0);
}

/// Quintic (m=3, #1044): the scan's exact diffuse restricted-likelihood
/// differences over λ match the dense order-3 REML (the diffuse `−½ Σ log F_∞`
/// term is λ-free and cancels in the difference, same as m=2). On the
/// wide-spacing data so the dense logdet — a near-total cancellation of
/// `½·rank·log λ` against `logdet(Λ)` — is resolved exactly. The two λ are kept
/// moderate and close so that cancellation is small.
#[test]
fn scan_restricted_loglik_differences_match_dense_reml_order3() {
    let (x, y, w) = test_data_wide();
    assert_scan_reml_matches_dense(3, &x, &y, &w, -0.5, 1.5);
}

fn assert_scan_bridges_gap(order: usize) {
    let (x, y, w) = test_data();
    let fit = fit_spline_scan(&x, &y, &w, order).expect("REML-selected scan fit");
    // The hole (0.45, 0.7) — the bridge must pass smoothly between the flanks
    // (no sag to the data mean) and the variance must peak inside the gap.
    let (m_mid, v_mid) = fit.predict(0.575).expect("gap midpoint");
    let (m_left, v_left) = fit.predict(0.44).expect("left flank");
    let (m_right, v_right) = fit.predict(0.71).expect("right flank");
    assert!(
        v_mid > v_left && v_mid > v_right,
        "gap variance must exceed flank variance (m={order}): mid={v_mid}, left={v_left}, right={v_right}"
    );
    let lo = m_left.min(m_right) - 0.75;
    let hi = m_left.max(m_right) + 0.75;
    assert!(
        m_mid > lo && m_mid < hi,
        "bridge mean {m_mid} should stay near the flank envelope [{lo}, {hi}] (m={order})"
    );
    // Off-knot prediction at a knot reproduces the knot posterior exactly.
    let (m_k, v_k) = fit.predict(fit.knots[10]).expect("at-knot predict");
    assert!(
        (m_k - fit.mean[10]).abs() < 1e-12 && (v_k - fit.var[10]).abs() < 1e-12,
        "at-knot predict mismatch (m={order})"
    );
}

#[test]
fn scan_bridges_gap_and_grows_variance() {
    assert_scan_bridges_gap(2);
}

/// Quintic (m=3, #1044): the higher-order bridge — built on the leading-block
/// gains for the diffuse leading intervals — still bridges the data hole with a
/// variance bump and reproduces the knot posteriors exactly.
#[test]
fn scan_bridges_gap_and_grows_variance_order3() {
    assert_scan_bridges_gap(3);
}
