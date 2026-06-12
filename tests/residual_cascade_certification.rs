//! Certification gates for the multiresolution residual cascade (#1032),
//! #904 style: the fitted model is checked against an INDEPENDENTLY
//! assembled dense penalized solve (same rows, same penalty, no shared
//! code path past `basis_row`) and against data with a KNOWN planted
//! signal — never against its own output.

use gam::solver::residual_cascade::{LogdetMethod, ResidualCascadeDesign, fit_residual_cascade};

/// SplitMix64 — deterministic test stream, no external RNG dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box-Muller; uniform() is in [0,1) so shift away from 0.
        let u1 = (self.uniform() + f64::EPSILON).min(1.0 - f64::EPSILON);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Dense lower-Cholesky of a row-major `p×p` SPD matrix; returns
/// `(L, log det)`. Independent of the library's kernels on purpose.
fn dense_cholesky(a: &[f64], p: usize) -> (Vec<f64>, f64) {
    let mut l = vec![0.0_f64; p * p];
    let mut logdet = 0.0;
    for j in 0..p {
        let mut s = a[j * p + j];
        for t in 0..j {
            s -= l[j * p + t] * l[j * p + t];
        }
        assert!(s > 0.0, "oracle: non-PD pivot {j} ({s})");
        let d = s.sqrt();
        l[j * p + j] = d;
        logdet += 2.0 * d.ln();
        for i in j + 1..p {
            let mut s2 = a[i * p + j];
            for t in 0..j {
                s2 -= l[i * p + t] * l[j * p + t];
            }
            l[i * p + j] = s2 / d;
        }
    }
    (l, logdet)
}

fn dense_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..p {
        let mut s = z[i];
        for t in 0..i {
            s -= l[i * p + t] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = z[i];
        for t in i + 1..p {
            s -= l[t * p + i] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    z
}

fn truth(x: f64, y: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin()
}

/// Scattered 2-D sample with mildly heterogeneous weights.
fn sample(n: usize, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let a = rng.uniform();
        let b = rng.uniform();
        x1.push(a);
        x2.push(b);
        y.push(truth(a, b) + noise * rng.normal());
        w.push(if i % 7 == 0 { 0.5 } else { 1.0 });
    }
    (x1, x2, y, w)
}

/// The fitted coefficients, penalized residual, exact log-determinant and
/// posterior variance must match a dense penalized least-squares oracle
/// assembled independently from `basis_row` + `penalty_value` unit probes.
#[test]
fn cascade_matches_dense_penalized_oracle() {
    let n = 400;
    let (x1, x2, y, w) = sample(n, 0.1, 0x1032_0001);
    let xs: Vec<&[f64]> = vec![&x1, &x2];
    let design =
        ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, 2).expect("design build");
    let m = design.num_coeffs();
    assert!(m > 3, "cascade placed no bumps (m = {m})");

    // Independent dense assembly: X rows from basis_row, D from unit probes.
    let mut x_dense = vec![0.0_f64; n * m];
    for i in 0..n {
        for (c, v) in design.basis_row(&[x1[i], x2[i]]).expect("basis row") {
            x_dense[i * m + c] += v;
        }
    }
    let mut pen_diag = vec![0.0_f64; m];
    let mut unit = vec![0.0_f64; m];
    for (j, dj) in pen_diag.iter_mut().enumerate() {
        unit[j] = 1.0;
        *dj = design.penalty_value(&unit).expect("penalty probe");
        unit[j] = 0.0;
    }
    let log_lambda = 0.5_f64;
    let lambda = log_lambda.exp();
    let mut a = vec![0.0_f64; m * m];
    let mut b = vec![0.0_f64; m];
    for i in 0..n {
        let row = &x_dense[i * m..(i + 1) * m];
        for j in 0..m {
            if row[j] == 0.0 {
                continue;
            }
            b[j] += w[i] * row[j] * y[i];
            for k in 0..m {
                a[j * m + k] += w[i] * row[j] * row[k];
            }
        }
    }
    for j in 0..m {
        a[j * m + j] += lambda * pen_diag[j];
    }
    let (l, oracle_logdet) = dense_cholesky(&a, m);
    let oracle_coeff = dense_solve(&l, m, &b);

    let fit = design.fit_at(log_lambda, None).expect("fit_at");
    assert_eq!(fit.certificate.logdet_method, LogdetMethod::DenseExact);
    assert!(
        fit.certificate.solve_rel_residual <= 1e-8,
        "uncertified solve: rel residual {}",
        fit.certificate.solve_rel_residual
    );
    let scale = oracle_coeff
        .iter()
        .fold(0.0_f64, |acc, &c| acc.max(c.abs()));
    for (j, (&got, &want)) in fit.coeff.iter().zip(oracle_coeff.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-7 * scale,
            "coefficient {j} diverges from dense oracle: {got} vs {want}"
        );
    }

    // Penalized residual and exact log-determinant against the oracle.
    let mut rss_pen = 0.0;
    for i in 0..n {
        let row = &x_dense[i * m..(i + 1) * m];
        let pred: f64 = row
            .iter()
            .zip(oracle_coeff.iter())
            .map(|(r, c)| r * c)
            .sum();
        rss_pen += w[i] * (y[i] - pred) * (y[i] - pred);
    }
    rss_pen += lambda * design.penalty_value(&oracle_coeff).expect("penalty");
    assert!(
        (fit.rss_pen - rss_pen).abs() <= 1e-8 * rss_pen.max(1.0),
        "penalized residual mismatch: {} vs oracle {rss_pen}",
        fit.rss_pen
    );
    let logdet = design.logdet_exact(log_lambda).expect("logdet");
    assert!(
        (logdet - oracle_logdet).abs() <= 1e-7 * oracle_logdet.abs().max(1.0),
        "log-determinant mismatch: {logdet} vs oracle {oracle_logdet}"
    );

    // Posterior variance at probe points: σ̂²·x'A⁻¹x against the oracle.
    for &(px, py) in &[(0.3, 0.4), (0.71, 0.18), (0.05, 0.92)] {
        let mut row = vec![0.0_f64; m];
        for (c, v) in design.basis_row(&[px, py]).expect("probe row") {
            row[c] += v;
        }
        let sol = dense_solve(&l, m, &row);
        let oracle_var: f64 =
            fit.sigma2 * row.iter().zip(sol.iter()).map(|(r, s)| r * s).sum::<f64>();
        let (_, var) = fit.predict(&[px, py]).expect("predict");
        assert!(
            (var - oracle_var).abs() <= 1e-7 * oracle_var.max(1e-12),
            "posterior variance mismatch at ({px},{py}): {var} vs oracle {oracle_var}"
        );
    }
}

/// Truth recovery (#904): the magic-default cascade on a planted smooth
/// must beat the noise floor on held-out truth, estimate σ² honestly, and
/// hand back its refinement + solve certificates.
#[test]
fn cascade_recovers_planted_smooth() {
    let n = 2500;
    let noise = 0.1;
    let (x1, x2, y, w) = sample(n, noise, 0x1032_0002);
    let xs: Vec<&[f64]> = vec![&x1, &x2];
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");

    assert!(
        fit.certificate.solve_rel_residual <= 1e-8,
        "uncertified mode solve: {}",
        fit.certificate.solve_rel_residual
    );
    let refinement = fit.refinement.as_ref().expect("refinement certificate");
    assert!(
        refinement.exhausted || refinement.next_level_gain_bound <= refinement.tolerance,
        "refinement neither converged nor exhausted: bound {} tol {}",
        refinement.next_level_gain_bound,
        refinement.tolerance
    );

    let grid = 40;
    let mut sse = 0.0;
    for i in 0..grid {
        for j in 0..grid {
            let px = (i as f64 + 0.5) / grid as f64;
            let py = (j as f64 + 0.5) / grid as f64;
            let (mean, var) = fit.predict(&[px, py]).expect("predict");
            assert!(var > 0.0, "non-positive posterior variance at ({px},{py})");
            let err = mean - truth(px, py);
            sse += err * err;
        }
    }
    let rmse = (sse / (grid * grid) as f64).sqrt();
    assert!(
        rmse < 0.6 * noise,
        "truth recovery too weak: rmse {rmse} vs noise {noise}"
    );
    assert!(
        fit.sigma2 > 0.25 * noise * noise && fit.sigma2 < 4.0 * noise * noise,
        "dishonest noise estimate: sigma2 {} for true {}",
        fit.sigma2,
        noise * noise
    );
}
