//! Certification gates for the multiresolution residual cascade (#1032),
//! #904 style: the fitted model is checked against an INDEPENDENTLY
//! assembled dense penalized solve (same rows, same penalty, no shared
//! code path past `basis_row`) and against data with a KNOWN planted
//! signal — never against its own output.
//!
//! Gate map (one per claim the module header makes):
//! - dense-oracle agreement in 2D and 3D (near-machine: the math is exact);
//! - SLQ logdet vs the exact dense logdet, and invariance of the REML
//!   λ-selection under the SLQ substitution (honest documented bounds: SLQ
//!   is an estimator, but a deterministic one — fixed probes);
//! - PCG iterative route: backward-error certificate honored and iteration
//!   count n-independent (the operational content of the norm equivalence);
//! - coarse-space additive-Schwarz preconditioner conditioning bounded
//!   uniformly in depth (the norm equivalence measured directly on small dense
//!   fixtures: the block-arrow P = blockdiag(A_CC, diag A_FF) is reconstructed
//!   from the public dense system + coarse cut and its whitened condition number
//!   must stay flat as the cascade deepens);
//! - cascade vs a dense single-scale Wendland kernel solve on small n at
//!   the native smoothness s = (d+3)/2 (the spec's norm-equivalence oracle);
//! - posterior perturb-and-solve samples match the exact `σ̂²A⁻¹` moments;
//! - gap behavior: the mean bridges instead of sagging and the posterior
//!   variance grows into the gap;
//! - truth recovery of the magic-default refinement loop with its
//!   certificates.

use gam::solver::residual_cascade::{
    LogdetMethod, ResidualCascadeDesign, ResidualCascadeError, ResidualCascadeFit,
    ResidualCascadeState, fit_residual_cascade,
};

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

/// Planted smooth: smooth, bounded, with genuine multiscale structure.
fn truth(p: &[f64]) -> f64 {
    let base =
        (2.0 * std::f64::consts::PI * p[0]).sin() * (2.0 * std::f64::consts::PI * p[1]).sin();
    match p.len() {
        2 => base,
        3 => base * (0.6 + 0.8 * p[2]),
        _ => unreachable!("truth: dim must be 2 or 3"),
    }
}

/// Scattered d-D sample on the unit cube with mildly heterogeneous weights.
fn sample(dim: usize, n: usize, noise: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed);
    let mut axes = vec![Vec::with_capacity(n); dim];
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut p = vec![0.0_f64; dim];
    for i in 0..n {
        for (a, axis) in axes.iter_mut().enumerate() {
            p[a] = rng.uniform();
            axis.push(p[a]);
        }
        y.push(truth(&p) + noise * rng.normal());
        w.push(if i % 7 == 0 { 0.5 } else { 1.0 });
    }
    (axes, y, w)
}

fn axis_refs(axes: &[Vec<f64>]) -> Vec<&[f64]> {
    axes.iter().map(|a| a.as_slice()).collect()
}

fn point_at(axes: &[Vec<f64>], i: usize) -> Vec<f64> {
    axes.iter().map(|a| a[i]).collect()
}

/// Independent dense assembly of `(X'WX + λD, X'Wy)` from `basis_row` and
/// `penalty_value` unit probes only — no shared code path past the row map.
struct DenseOracle {
    m: usize,
    x_dense: Vec<f64>,
    l: Vec<f64>,
    logdet: f64,
    coeff: Vec<f64>,
}

fn dense_oracle(
    design: &ResidualCascadeDesign,
    axes: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    lambda: f64,
) -> DenseOracle {
    let n = y.len();
    let m = design.num_coeffs();
    let mut x_dense = vec![0.0_f64; n * m];
    for i in 0..n {
        for (c, v) in design.basis_row(&point_at(axes, i)).expect("basis row") {
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
    let (l, logdet) = dense_cholesky(&a, m);
    let coeff = dense_solve(&l, m, &b);
    DenseOracle {
        m,
        x_dense,
        l,
        logdet,
        coeff,
    }
}

/// The fitted coefficients, penalized residual, exact log-determinant and
/// posterior variance must match the dense penalized least-squares oracle —
/// near machine precision, because the math claims exactness here.
fn check_dense_oracle(dim: usize, n: usize, seed: u64, probes: &[Vec<f64>]) {
    let (axes, y, w) = sample(dim, n, 0.1, seed);
    let xs = axis_refs(&axes);
    let metric = vec![1.0_f64; dim];
    let design = ResidualCascadeDesign::build(&xs, &y, &w, &metric, 2.0, 2).expect("design build");
    let m = design.num_coeffs();
    assert!(m > dim + 1, "cascade placed no bumps (m = {m})");

    let log_lambda = 0.5_f64;
    let lambda = log_lambda.exp();
    let oracle = dense_oracle(&design, &axes, &y, &w, lambda);

    let fit = design.fit_at(log_lambda, None).expect("fit_at");
    assert_eq!(fit.certificate.logdet_method, LogdetMethod::DenseExact);
    assert!(
        fit.certificate.solve_rel_residual <= 1e-8,
        "uncertified solve: rel residual {}",
        fit.certificate.solve_rel_residual
    );
    let scale = oracle
        .coeff
        .iter()
        .fold(0.0_f64, |acc, &c| acc.max(c.abs()));
    for (j, (&got, &want)) in fit.coeff.iter().zip(oracle.coeff.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-7 * scale,
            "coefficient {j} diverges from dense oracle: {got} vs {want}"
        );
    }

    // Penalized residual and exact log-determinant against the oracle.
    let mut rss_pen = 0.0;
    for i in 0..n {
        let row = &oracle.x_dense[i * oracle.m..(i + 1) * oracle.m];
        let pred: f64 = row
            .iter()
            .zip(oracle.coeff.iter())
            .map(|(r, c)| r * c)
            .sum();
        rss_pen += w[i] * (y[i] - pred) * (y[i] - pred);
    }
    rss_pen += lambda * design.penalty_value(&oracle.coeff).expect("penalty");
    assert!(
        (fit.rss_pen - rss_pen).abs() <= 1e-8 * rss_pen.max(1.0),
        "penalized residual mismatch: {} vs oracle {rss_pen}",
        fit.rss_pen
    );
    let logdet = design.logdet_exact(log_lambda).expect("logdet");
    assert!(
        (logdet - oracle.logdet).abs() <= 1e-7 * oracle.logdet.abs().max(1.0),
        "log-determinant mismatch: {logdet} vs oracle {}",
        oracle.logdet
    );

    // Posterior variance at probe points: σ̂²·x'A⁻¹x against the oracle.
    for probe in probes {
        let mut row = vec![0.0_f64; oracle.m];
        for (c, v) in design.basis_row(probe).expect("probe row") {
            row[c] += v;
        }
        let sol = dense_solve(&oracle.l, oracle.m, &row);
        let oracle_var: f64 =
            fit.sigma2 * row.iter().zip(sol.iter()).map(|(r, s)| r * s).sum::<f64>();
        let (_, var) = fit.predict(probe).expect("predict");
        assert!(
            (var - oracle_var).abs() <= 1e-7 * oracle_var.max(1e-12),
            "posterior variance mismatch at {probe:?}: {var} vs oracle {oracle_var}"
        );
    }
}

#[test]
fn cascade_matches_dense_penalized_oracle() {
    check_dense_oracle(
        2,
        400,
        0x1032_0001,
        &[vec![0.3, 0.4], vec![0.71, 0.18], vec![0.05, 0.92]],
    );
}

#[test]
fn cascade_matches_dense_penalized_oracle_3d() {
    check_dense_oracle(
        3,
        350,
        0x1032_0007,
        &[
            vec![0.3, 0.4, 0.5],
            vec![0.71, 0.18, 0.83],
            vec![0.05, 0.92, 0.27],
        ],
    );
}

/// Truth recovery (#904): the magic-default cascade on a planted smooth
/// must beat the noise floor on held-out truth, estimate σ² honestly, and
/// hand back its refinement + solve certificates.
#[test]
fn cascade_recovers_planted_smooth() {
    let n = 2500;
    let noise = 0.1;
    let (axes, y, w) = sample(2, n, noise, 0x1032_0002);
    let xs = axis_refs(&axes);
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");

    assert!(
        fit.certificate.solve_rel_residual <= 1e-8,
        "uncertified mode solve: {}",
        fit.certificate.solve_rel_residual
    );
    let refinement = fit.refinement.as_ref().expect("refinement certificate");
    assert!(
        refinement.next_level_gain_bound <= refinement.tolerance,
        "returned fit has an uncertified refinement bound: bound {} tol {}",
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
            let err = mean - truth(&[px, py]);
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

/// SLQ logdet vs the exact dense logdet across the λ range, and invariance
/// of the coarse-grid REML λ-selection under the SLQ substitution. SLQ only
/// estimates the small preconditioned remainder (the coarse-space + Jacobi
/// control variate `log|P|` carries the bulk exactly), and the probes are
/// FIXED, so this is a
/// deterministic, reproducible bound — honest, not exact: 1% relative plus a
/// 0.5-nat absolute floor on the logdet, and the grid argmax may shift by at
/// most one step.
#[test]
fn slq_logdet_tracks_exact_and_preserves_lambda_selection() {
    let n = 600;
    let (axes, y, w) = sample(2, n, 0.1, 0x1032_0003);
    let xs = axis_refs(&axes);
    let design = ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, 3).expect("build");

    for &ll in &[-6.0_f64, -3.0, 0.0, 3.0, 6.0] {
        let exact = design.logdet_exact(ll).expect("exact logdet");
        let slq = design.logdet_slq(ll).expect("slq logdet");
        assert!(
            (slq - exact).abs() <= 0.01 * exact.abs() + 0.5,
            "SLQ logdet off at log-lambda {ll}: slq {slq} vs exact {exact}"
        );
    }

    // λ-selection invariance over the same 25-point coarse grid fit_reml
    // scans: substituting the SLQ logdet into the criterion must move the
    // argmax by at most one grid step.
    let grid = 25;
    let (lo, hi) = (-18.0_f64, 18.0_f64);
    let step = (hi - lo) / (grid - 1) as f64;
    let mut best_exact = (0usize, f64::NEG_INFINITY);
    let mut best_slq = (0usize, f64::NEG_INFINITY);
    for i in 0..grid {
        let ll = lo + step * i as f64;
        let c_exact = design.criterion(ll).expect("criterion");
        let exact = design.logdet_exact(ll).expect("exact logdet");
        let slq = design.logdet_slq(ll).expect("slq logdet");
        let c_slq = c_exact - 0.5 * (slq - exact);
        if c_exact > best_exact.1 {
            best_exact = (i, c_exact);
        }
        if c_slq > best_slq.1 {
            best_slq = (i, c_slq);
        }
    }
    assert!(
        best_exact.0.abs_diff(best_slq.0) <= 1,
        "SLQ shifted the REML grid argmax: exact at {}, slq at {}",
        best_exact.0,
        best_slq.0
    );
}

/// The iterative route past the dense cap: the PCG backward-error
/// certificate must be honored, and the iteration count must be
/// n-independent up to the realized greedy-net constant — quadrupling n at
/// fixed depth may not produce an unbounded iteration tail. This is the
/// operational content of the multilevel norm equivalence (#1032 spec:
/// n-independent iters by design).
#[test]
fn pcg_route_certified_and_iteration_count_n_independent() {
    let levels = 6;
    let mut iter_counts = Vec::new();
    for &(n, seed) in &[(12_000usize, 0x1032_0004_u64), (48_000, 0x1032_0005)] {
        let (axes, y, w) = sample(2, n, 0.1, seed);
        let xs = axis_refs(&axes);
        let design =
            ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, levels).expect("build");
        assert!(
            design.num_coeffs() > 1536,
            "fixture too small to engage the iterative route (m = {})",
            design.num_coeffs()
        );
        let fit = design.fit_at(0.0, None).expect("fit_at");
        assert_eq!(fit.certificate.logdet_method, LogdetMethod::Slq);
        assert!(
            fit.certificate.solve_rel_residual <= 1e-9,
            "backward-error certificate violated at n {n}: {}",
            fit.certificate.solve_rel_residual
        );
        assert!(
            fit.certificate.solve_iters >= 1 && fit.certificate.solve_iters <= 60,
            "PCG iteration count out of range at n {n}: {} (the coarse-space preconditioner \
             targets the spec's 20–60 iters n-independent)",
            fit.certificate.solve_iters
        );
        // A certified-route prediction must still hand back a positive
        // variance through its own certified solve.
        let (_, var) = fit.predict(&[0.4, 0.6]).expect("predict");
        assert!(var > 0.0, "non-positive iterative-route variance {var}");
        iter_counts.push(fit.certificate.solve_iters);
    }
    eprintln!("[1032-NINDEP] iter_counts 12k/48k = {iter_counts:?}");
    // Genuine n-independence is an ADDITIVE bound: quadrupling n adds at most a
    // small constant to the iteration count (the coarse-space deflation makes
    // the conditioning, hence ~√cond iterations, depth/n-independent). This is
    // the principled bound the pure-Jacobi diagonal could not meet — its count
    // grew multiplicatively with n.
    assert!(
        iter_counts[1] <= iter_counts[0] + 10,
        "PCG iterations grew with n beyond the additive n-independence bound: {} at 12k vs {} \
         at 48k",
        iter_counts[0],
        iter_counts[1]
    );
}

/// Depth-independence of the iterative route — the SAME root cause as the
/// n-independence gate (the preconditioner conditioning must not grow with the
/// number of resolution levels) attacked from a different angle: hold n FIXED
/// and DEEPEN the cascade, and require the realized PCG iteration count to stay
/// flat. The pure-Jacobi diagonal failed exactly here — each added level is
/// another cross-scale-correlated block the diagonal cannot decouple, so its
/// count climbed level by level. The coarse-space preconditioner deflects every
/// data-dominated level into the exact coarse solve and Jacobi-handles the
/// penalty-dominated tail, so deepening past the resolved scale adds only
/// well-conditioned fine levels and the count saturates.
#[test]
fn pcg_iteration_count_independent_of_cascade_depth() {
    let n = 24_000;
    let (axes, y, w) = sample(2, n, 0.1, 0x1032_0D07);
    let xs = axis_refs(&axes);
    let mut iters = Vec::new();
    let mut depths = Vec::new();
    for levels in [4usize, 6, 8] {
        let design =
            ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, levels).expect("build");
        if design.num_coeffs() <= 1536 {
            continue; // need the iterative route to be engaged
        }
        let fit = design.fit_at(0.0, None).expect("fit_at");
        assert_eq!(fit.certificate.logdet_method, LogdetMethod::Slq);
        assert!(
            fit.certificate.solve_rel_residual <= 1e-9,
            "uncertified solve at depth {levels}: {}",
            fit.certificate.solve_rel_residual
        );
        depths.push(levels);
        iters.push(fit.certificate.solve_iters);
    }
    eprintln!("[1032-DEPTH] depths={depths:?} iters={iters:?}");
    assert!(
        iters.len() >= 2,
        "fixture never engaged the iterative route across the depth sweep"
    );
    let lo = *iters.iter().min().unwrap();
    let hi = *iters.iter().max().unwrap();
    assert!(
        hi <= 60,
        "iteration count exceeded the n-independent target while deepening: {iters:?}"
    );
    // The spread across a 4×-deeper cascade is an additive constant, not growth
    // proportional to depth — the operational content of depth-independent
    // conditioning.
    assert!(
        hi <= lo + 12,
        "PCG iteration count grew with cascade depth — preconditioner conditioning is \
         depth-dependent (the pure-Jacobi failure mode): {iters:?}"
    );
}

/// Structural n-scaling payoff (#1032 spec: "O(n log n) fit for the class
/// where duchon/matern build dense n×k kernels per hyperparameter trial"). Two
/// claims, certified without wall-clock thresholds:
///
/// 1. At a design past the dense sizing cap (PCG + coarse-space additive-Schwarz
///    route engaged), the certified sparse solve's matvec work is below the cubic
///    dense factorization work it replaces.
/// 2. Across a 4× jump in n at fixed depth the PCG iteration count remains
///    bounded by a constant factor, so total sparse work tracks the CSR size
///    rather than growing from iteration creep.
#[test]
fn cascade_sparse_work_beats_dense_factorization_and_scales_near_linearly() {
    let levels = 6;
    let log_lambda = 0.0_f64;

    // --- Claim 1: certified sparse work vs dense Cholesky work at large m. ---
    let (axes, y, w) = sample(2, 24_000, 0.1, 0x1032_0F01);
    let xs = axis_refs(&axes);
    let design =
        ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, levels).expect("build");
    assert!(
        design.num_coeffs() > 1536,
        "fixture too small to engage the iterative route (m = {})",
        design.num_coeffs()
    );

    // Run twice so the second solve exercises the same public path after any
    // one-time allocation effects, while both solves remain certified.
    let warm = design.fit_at(log_lambda, None).expect("warm fit");
    assert!(
        warm.certificate.solve_rel_residual <= 1e-9,
        "warm cascade solve uncertified: {}",
        warm.certificate.solve_rel_residual
    );
    let fit = design.fit_at(log_lambda, None).expect("fit_at");
    assert_eq!(fit.certificate.logdet_method, LogdetMethod::Slq);
    assert!(
        fit.certificate.solve_rel_residual <= 1e-9,
        "uncertified cascade solve: {}",
        fit.certificate.solve_rel_residual
    );
    let sparse_work = fit.certificate.solve_iters as f64 * design.num_nonzeros() as f64;
    let m = design.num_coeffs() as f64;
    let dense_factor_work = m * m * m / 3.0;
    assert!(
        sparse_work < dense_factor_work,
        "certified sparse solve work did not beat dense factorization work at m = {}: \
         {} matvec-nnz units vs {} cubic units",
        design.num_coeffs(),
        sparse_work,
        dense_factor_work
    );

    // --- Claim 2: near-linear growth across a 4× jump in n at fixed depth. ---
    let mut work_ratios = Vec::new();
    let mut iter_counts = Vec::new();
    for &(n, seed) in &[(12_000usize, 0x1032_0F02_u64), (48_000, 0x1032_0F03)] {
        let (ax, yy, ww) = sample(2, n, 0.1, seed);
        let xr = axis_refs(&ax);
        let d =
            ResidualCascadeDesign::build(&xr, &yy, &ww, &[1.0, 1.0], 2.0, levels).expect("build");
        let warm = d.fit_at(log_lambda, None).expect("warm");
        assert!(
            warm.certificate.solve_rel_residual <= 1e-9,
            "uncertified warm solve at n={n}"
        );
        let f = d.fit_at(log_lambda, None).expect("fit");
        assert!(
            f.certificate.solve_rel_residual <= 1e-9,
            "uncertified solve at n={n}"
        );
        iter_counts.push(f.certificate.solve_iters);
        work_ratios.push(f.certificate.solve_iters as f64 * d.num_nonzeros() as f64 / n as f64);
    }
    // The exact iteration count can drift with the realized greedy net because
    // m is also changing with n, but the coarse-space preconditioner holds it to
    // an ADDITIVE n-independence bound, so per-row sparse work tracks the CSR
    // size rather than growing from iteration creep.
    assert!(
        iter_counts[1] <= iter_counts[0] + 10,
        "PCG iteration count grew with n beyond the additive coarse-space bound: \
         {} at 12k vs {} at 48k",
        iter_counts[0],
        iter_counts[1]
    );
    assert!(
        work_ratios[1] <= 2.5 * work_ratios[0],
        "per-row sparse work grew faster than near-linearly across 4× n: \
         {} at 12k vs {} at 48k",
        work_ratios[0],
        work_ratios[1]
    );
}

/// The coarse-space additive-Schwarz preconditioner must condition the system
/// uniformly in cascade depth — the operational content of the norm
/// equivalence, measured directly. On a small dense fixture we reconstruct the
/// EXACT preconditioner the library's iterative route uses, `P = blockdiag(
/// A_CC, diag A_FF)` with the coarse cut taken from the public
/// `coarse_space_cols`, whiten `M = L_P^{-1} A L_P^{-T}` (so cond(M) =
/// cond(P^{-1}A)), and require cond(M) to stay BOUNDED and FLAT as the depth
/// grows. This is the gate the pure-Jacobi diagonal failed: its whitened
/// condition number climbs with every added level (the cross-scale frame
/// redundancy the diagonal cannot decouple), which is exactly why its CG
/// iteration count was n-dependent. A pure-Jacobi control is measured alongside
/// to document the contrast.
#[test]
fn coarse_space_preconditioner_conditions_uniformly_in_depth() {
    let n = 800;
    let (axes, y, w) = sample(2, n, 0.1, 0x1032_0006);
    let xs = axis_refs(&axes);
    let lambda = 1.0_f64;
    let log_lambda = 0.0_f64;

    // M = L_P^{-1} A L_P^{-T} for an SPD preconditioner P given by its lower
    // Cholesky factor `lp`; cond(M) = cond(P^{-1} A). Two forward-substitution
    // passes (U = L_P^{-1} A, then M^T solved column-by-column).
    fn whiten(a: &[f64], lp: &[f64], m: usize) -> Vec<f64> {
        let mut u = vec![0.0_f64; m * m];
        for c in 0..m {
            for i in 0..m {
                let mut s = a[i * m + c];
                for t in 0..i {
                    s -= lp[i * m + t] * u[t * m + c];
                }
                u[i * m + c] = s / lp[i * m + i];
            }
        }
        let mut mm = vec![0.0_f64; m * m];
        for r in 0..m {
            for i in 0..m {
                let mut s = u[r * m + i];
                for t in 0..i {
                    s -= lp[i * m + t] * mm[r * m + t];
                }
                mm[r * m + i] = s / lp[i * m + i];
            }
        }
        mm
    }

    // cond of a symmetric PD M: λmax by power iteration, λmin by inverse
    // iteration on Cholesky(M).
    fn cond_sym(mmat: &[f64], m: usize, seed: u64) -> f64 {
        let matvec = |v: &[f64], out: &mut [f64]| {
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..m {
                    s += mmat[i * m + j] * v[j];
                }
                out[i] = s;
            }
        };
        let mut rng = Rng(seed);
        let mut v: Vec<f64> = (0..m).map(|_| rng.normal()).collect();
        let mut tmp = vec![0.0_f64; m];
        let mut lam_max = 0.0;
        for _ in 0..400 {
            matvec(&v, &mut tmp);
            lam_max = tmp.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..m {
                v[j] = tmp[j] / lam_max;
            }
        }
        let (lm, _) = dense_cholesky(mmat, m);
        let mut u: Vec<f64> = (0..m).map(|_| rng.normal()).collect();
        let mut inv_norm = 0.0;
        for _ in 0..400 {
            let s = dense_solve(&lm, m, &u);
            inv_norm = s.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..m {
                u[j] = s[j] / inv_norm;
            }
        }
        // The spectral condition number of an SPD matrix is mathematically
        // bounded below by one.  For identity-like matrices the two iterative
        // estimates can straddle one by a few ulps after whitening/Cholesky
        // roundoff, so enforce the invariant at the numerical boundary rather
        // than reporting an impossible sub-unit condition number.
        (lam_max * inv_norm).max(1.0)
    }

    let mut conds = Vec::new();
    let mut jacobi_conds = Vec::new();
    let mut cuts = Vec::new();
    for levels in [2usize, 4, 6] {
        let design =
            ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, levels).expect("build");
        let oracle = dense_oracle(&design, &axes, &y, &w, lambda);
        let m = oracle.m;
        // A = L L' from the oracle's Cholesky factor.
        let mut a = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for t in 0..=i.min(j) {
                    s += oracle.l[i * m + t] * oracle.l[j * m + t];
                }
                a[i * m + j] = s;
            }
        }
        // Coarse-space preconditioner P = blockdiag(A_CC, diag A_FF).
        let nc = design
            .coarse_space_cols(log_lambda)
            .expect("finite log_lambda=0 fixture must define a coarse-space dimension")
            .min(m);
        cuts.push(nc);
        let mut p = vec![0.0_f64; m * m];
        for i in 0..nc {
            for j in 0..nc {
                p[i * m + j] = a[i * m + j];
            }
        }
        for j in nc..m {
            p[j * m + j] = a[j * m + j];
        }
        let (lp, _) = dense_cholesky(&p, m);
        let mp = whiten(&a, &lp, m);
        conds.push(cond_sym(&mp, m, 0x1032_00C0));

        // Pure-Jacobi control, for contrast.
        let d_inv_sqrt: Vec<f64> = (0..m).map(|j| 1.0 / a[j * m + j].sqrt()).collect();
        let mut mj = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..m {
                mj[i * m + j] = a[i * m + j] * d_inv_sqrt[i] * d_inv_sqrt[j];
            }
        }
        jacobi_conds.push(cond_sym(&mj, m, 0x1032_00C1));
    }
    eprintln!(
        "[1032-COND] coarse cuts={cuts:?} coarse_conds={conds:?} jacobi_conds={jacobi_conds:?}"
    );
    for (idx, &c) in conds.iter().enumerate() {
        assert!(
            c.is_finite() && c >= 1.0 && c <= 1.0e2,
            "coarse-space preconditioned condition number out of range at depth index {idx}: \
             {c} (all: {conds:?})"
        );
    }
    // Depth independence: the deepest cascade's conditioning is within a small
    // constant factor of the shallowest — NOT growing with depth.
    assert!(
        conds[2] <= 2.5 * conds[0].max(2.0),
        "coarse-space conditioning degrades with depth — norm equivalence violated: {conds:?}"
    );
}

/// The spec's norm-equivalence oracle: on small n, at the Wendland-(3,1)
/// native smoothness s = (d+3)/2, the multilevel cascade must recover the
/// planted truth comparably to a DENSE single-scale Wendland kernel solve
/// (all-points centers, identity prior, exact dense REML over the same λ
/// grid). Equivalent norms admit constants, so the bound is a documented
/// 2× factor on held-out truth RMSE — plus an absolute sanity gate that
/// the dense reference itself works on this fixture.
#[test]
fn cascade_matches_dense_wendland_kernel_solve() {
    let n = 240;
    let noise = 0.05;
    let (axes, y, w) = sample(2, n, noise, 0x1032_0008);
    let xs = axis_refs(&axes);

    // Dense single-scale Wendland kernel reference, assembled entirely
    // in-test: columns = [1, x1, x2] + one bump of radius delta at every
    // data point; D = I on the bump block; λ by exact REML on a coarse grid.
    let wendland = |r: f64| {
        if r >= 1.0 {
            0.0
        } else {
            let v = 1.0 - r;
            v * v * v * v * (4.0 * r + 1.0)
        }
    };
    let delta = 0.25_f64;
    let p0 = 3usize;
    let m = p0 + n;
    let row_at = |px: f64, py: f64| -> Vec<f64> {
        let mut row = vec![0.0_f64; m];
        row[0] = 1.0;
        row[1] = 2.0 * px - 1.0;
        row[2] = 2.0 * py - 1.0;
        for j in 0..n {
            let dx = px - axes[0][j];
            let dy = py - axes[1][j];
            row[p0 + j] = wendland((dx * dx + dy * dy).sqrt() / delta);
        }
        row
    };
    let mut x_dense = vec![0.0_f64; n * m];
    for i in 0..n {
        let row = row_at(axes[0][i], axes[1][i]);
        x_dense[i * m..(i + 1) * m].copy_from_slice(&row);
    }
    let mut gram = vec![0.0_f64; m * m];
    let mut b = vec![0.0_f64; m];
    let mut ytwy = 0.0;
    for i in 0..n {
        let row = &x_dense[i * m..(i + 1) * m];
        ytwy += w[i] * y[i] * y[i];
        for j in 0..m {
            b[j] += w[i] * row[j] * y[i];
            for k in 0..m {
                gram[j * m + k] += w[i] * row[j] * row[k];
            }
        }
    }
    let dof = (n - p0) as f64;
    let mut best: Option<(f64, Vec<f64>)> = None;
    for g in 0..25 {
        let ll = -18.0 + 36.0 * g as f64 / 24.0;
        let lambda = ll.exp();
        let mut a = gram.clone();
        for j in p0..m {
            a[j * m + j] += lambda;
        }
        let (l, logdet) = dense_cholesky(&a, m);
        let coeff = dense_solve(&l, m, &b);
        let rss_pen = ytwy - coeff.iter().zip(b.iter()).map(|(c, r)| c * r).sum::<f64>();
        if !(rss_pen > 0.0) {
            continue;
        }
        let sigma2 = rss_pen / dof;
        let crit = -0.5 * (logdet - (m - p0) as f64 * ll + dof * sigma2.ln());
        if best.as_ref().is_none_or(|(bc, _)| crit > *bc) {
            best = Some((crit, coeff));
        }
    }
    let (_, kernel_coeff) = best.expect("kernel REML grid found no PD point");

    // Cascade at the native smoothness for an apples-to-apples norm.
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.5).expect("cascade fit");

    let grid = 30;
    let mut sse_kernel = 0.0;
    let mut sse_cascade = 0.0;
    for i in 0..grid {
        for j in 0..grid {
            let px = (i as f64 + 0.5) / grid as f64;
            let py = (j as f64 + 0.5) / grid as f64;
            let t = truth(&[px, py]);
            let row = row_at(px, py);
            let kp: f64 = row
                .iter()
                .zip(kernel_coeff.iter())
                .map(|(r, c)| r * c)
                .sum();
            sse_kernel += (kp - t) * (kp - t);
            let (cp, _) = fit.predict(&[px, py]).expect("predict");
            sse_cascade += (cp - t) * (cp - t);
        }
    }
    let rmse_kernel = (sse_kernel / (grid * grid) as f64).sqrt();
    let rmse_cascade = (sse_cascade / (grid * grid) as f64).sqrt();
    assert!(
        rmse_kernel <= 2.0 * noise,
        "dense kernel reference failed its own sanity gate: rmse {rmse_kernel}"
    );
    eprintln!("[1032-WENDLAND] rmse_cascade={rmse_cascade} rmse_kernel={rmse_kernel}");
    // Equivalent norms admit a constant; the original certification bound is a
    // 1.5× factor on held-out truth RMSE. (Commit 3ec23cfa5 silently relaxed this
    // to 2.0× in a "make it green" pass without any cascade-quality change — a
    // banned weakening; restored. The multilevel frame adapts across scales, so
    // the cascade typically matches-or-beats the single-scale kernel here.)
    assert!(
        rmse_cascade <= 1.5 * rmse_kernel,
        "cascade falls behind the dense kernel solve: {rmse_cascade} vs {rmse_kernel}"
    );
}

/// Perturb-and-solve posterior samples have mean ĉ and covariance EXACTLY
/// `σ̂²A⁻¹` in distribution; with 512 deterministic samples the empirical
/// moments must match the dense-oracle moments within standard Monte-Carlo
/// bounds (6σ on means, 40% on variances — sd of a 512-sample variance is
/// ~6.3%, so this is a >6σ gate; the fixed seed makes it reproducible).
#[test]
fn posterior_samples_match_exact_moments() {
    let n = 350;
    let (axes, y, w) = sample(2, n, 0.1, 0x1032_0009);
    let xs = axis_refs(&axes);
    let design = ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, 2).expect("build");
    let log_lambda = 0.5_f64;
    let fit = design.fit_at(log_lambda, None).expect("fit_at");
    let oracle = dense_oracle(&design, &axes, &y, &w, log_lambda.exp());
    let m = oracle.m;

    // Exact posterior sd per coordinate: σ̂·sqrt((A⁻¹)_jj).
    let mut sd = vec![0.0_f64; m];
    let mut unit = vec![0.0_f64; m];
    for j in 0..m {
        unit[j] = 1.0;
        let col = dense_solve(&oracle.l, m, &unit);
        sd[j] = (fit.sigma2 * col[j]).sqrt();
        unit[j] = 0.0;
    }

    let n_samples = 512usize;
    let samples = fit.sample_coefficients(n_samples).expect("samples");
    assert_eq!(samples.len(), n_samples);
    let mut mean = vec![0.0_f64; m];
    for s in &samples {
        for j in 0..m {
            mean[j] += s[j];
        }
    }
    for mj in mean.iter_mut() {
        *mj /= n_samples as f64;
    }
    let mut var = vec![0.0_f64; m];
    for s in &samples {
        for j in 0..m {
            let d = s[j] - mean[j];
            var[j] += d * d;
        }
    }
    for vj in var.iter_mut() {
        *vj /= (n_samples - 1) as f64;
    }

    let mc = 6.0 / (n_samples as f64).sqrt();
    for j in 0..m {
        assert!(
            (mean[j] - fit.coeff[j]).abs() <= mc * sd[j] + 1e-12,
            "sample mean off at coordinate {j}: {} vs mode {} (sd {})",
            mean[j],
            fit.coeff[j],
            sd[j]
        );
        assert!(
            (var[j] - sd[j] * sd[j]).abs() <= 0.4 * sd[j] * sd[j] + 1e-16,
            "sample variance off at coordinate {j}: {} vs exact {}",
            var[j],
            sd[j] * sd[j]
        );
    }
}

/// Gap behavior (#1032 spec: "bridge-don't-sag mechanically visible"): with
/// a 0.3-wide data void across the domain, the posterior mean must bridge
/// the planted smooth (error at the gap center bounded well under the signal
/// amplitude — a global-trend sag would miss the in-gap maximum by ≥0.3)
/// while the posterior variance grows into the gap.
#[test]
fn gap_bridges_without_sagging_and_variance_grows() {
    let n = 3000;
    let noise = 0.05;
    let mut rng = Rng(0x1032_000A);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let w = vec![1.0_f64; n];
    // Smooth, amplitude-1, with its maximum INSIDE the gap: sagging toward a
    // global trend is visibly wrong at the gap center.
    let f = |a: f64, b: f64| (std::f64::consts::PI * a).sin() * (0.6 + 0.4 * b);
    while x1.len() < n {
        let a = rng.uniform();
        if a > 0.35 && a < 0.65 {
            continue;
        }
        let b = rng.uniform();
        x1.push(a);
        x2.push(b);
        y.push(f(a, b) + noise * rng.normal());
    }
    let xs: Vec<&[f64]> = vec![&x1, &x2];
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");

    // Covered-region accuracy and variance baseline.
    let mut covered_vars = Vec::new();
    let mut sse_covered = 0.0;
    let mut n_covered = 0usize;
    for i in 0..20 {
        for j in 0..10 {
            let a = (i as f64 + 0.5) / 20.0;
            if a > 0.35 && a < 0.65 {
                continue;
            }
            let b = (j as f64 + 0.5) / 10.0;
            let (mean, var) = fit.predict(&[a, b]).expect("predict");
            let err = mean - f(a, b);
            sse_covered += err * err;
            covered_vars.push(var);
            n_covered += 1;
        }
    }
    let rmse_covered = (sse_covered / n_covered as f64).sqrt();
    assert!(
        rmse_covered <= 3.0 * noise,
        "covered-region recovery too weak: rmse {rmse_covered}"
    );
    covered_vars.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_var = covered_vars[covered_vars.len() / 2];

    let (gap_mean, gap_var) = fit.predict(&[0.5, 0.5]).expect("gap predict");
    let gap_truth = f(0.5, 0.5);
    assert!(
        (gap_mean - gap_truth).abs() <= 0.25,
        "gap bridge missed the planted smooth: mean {gap_mean} vs truth {gap_truth}"
    );
    eprintln!("[1032-GAP] gap_var={gap_var} median_var={median_var}");
    // The variance must GROW into the gap, not merely tie the covered median: the
    // original certification required a 1.5× margin. (Commit 3ec23cfa5 silently
    // relaxed `>= 1.5 * median_var` to `> median_var` in a "make it green" pass
    // with no cascade-quality change — a banned weakening; restored. The exact
    // posterior `σ²·x'A⁻¹x` is the highest where the prediction extrapolates on
    // coarse bumps the data cannot pin, which is exactly the gap interior.)
    assert!(
        gap_var >= 1.5 * median_var,
        "posterior variance failed to grow into the gap: {gap_var} vs covered median {median_var}"
    );
}

/// Caveat 1 (#1032 spec: the Wendland-(3,1) native-smoothness ceiling caps the
/// recoverable Sobolev order). A deliberately high-frequency truth — finer than
/// the `INITIAL_LEVELS` nets can resolve — must FORCE the magic-default
/// refinement loop to add levels past `INITIAL_LEVELS = 3`, and the loop must
/// terminate with the exact level-(L+1) gain bound as an HONEST upper bound on
/// the remaining penalized-objective decrease. A returned fit's bound must be
/// certified below its tolerance, proving the discretization bias is spent;
/// structural exhaustion is a typed error, not an accepted fit. The recovered
/// fit's in-domain error must be consistent with that certified residual — the
/// certificate is the instrument that detects "adding a level still moves the
/// functional", exactly as the spec requires.
#[test]
fn smoothness_ceiling_forces_refinement_and_certifies_residual_bias() {
    // Four full cycles per axis: the level-0..2 nets (covering radius h0·2^-l
    // with h0 ~ domain scale) are coarser than the half-period, so the coarse
    // frame cannot represent this surface — the gain bound stays above tolerance
    // until finer levels are appended.
    let n = 6000;
    let noise = 0.02;
    let mut rng = Rng(0x1032_000C);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let w = vec![1.0_f64; n];
    let k = 4.0 * std::f64::consts::PI;
    let f = |a: f64, b: f64| (k * a).sin() * (k * b).cos();
    for _ in 0..n {
        let a = rng.uniform();
        let b = rng.uniform();
        x1.push(a);
        x2.push(b);
        y.push(f(a, b) + noise * rng.normal());
    }
    let xs: Vec<&[f64]> = vec![&x1, &x2];
    // Sobolev order at the native ceiling (d/2, (d+3)/2] = (1, 2.5] for d=2.
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");

    // The refinement loop was forced past the initial depth to chase the
    // high-frequency tail (INITIAL_LEVELS = 3 in residual_cascade.rs).
    assert!(
        fit.num_levels() > 3,
        "high-frequency truth did not force refinement past INITIAL_LEVELS: levels {}",
        fit.num_levels()
    );

    // The terminating certificate is an honest bound on the residual movement:
    // a non-negative, finite level-(L+1) gain bound compared against its own
    // tolerance. Returning a fit requires bound ≤ tolerance; capacity exhaustion
    // is reported as `ResidualCascadeError::Underresolved` instead.
    let cert = fit.refinement.as_ref().expect("refinement certificate");
    assert!(
        cert.next_level_gain_bound.is_finite() && cert.next_level_gain_bound >= 0.0,
        "refinement bound not a finite non-negative certificate: {}",
        cert.next_level_gain_bound
    );
    assert!(cert.tolerance.is_finite() && cert.tolerance > 0.0);
    // One more level provably cannot move the objective by more than the
    // tolerance — the discretization bias is certified spent.
    assert!(
        cert.next_level_gain_bound <= cert.tolerance,
        "returned fit has an uncertified refinement bound: {} vs {}",
        cert.next_level_gain_bound,
        cert.tolerance
    );

    // The certified fit recovers the high-frequency surface on held-out truth:
    // once refinement has run, the frame resolves the planted structure rather
    // than under-fitting it to a coarse trend.
    let grid = 30;
    let mut sse = 0.0;
    for i in 0..grid {
        for j in 0..grid {
            let px = (i as f64 + 0.5) / grid as f64;
            let py = (j as f64 + 0.5) / grid as f64;
            let (mean, var) = fit.predict(&[px, py]).expect("predict");
            assert!(var > 0.0, "non-positive posterior variance at ({px},{py})");
            let err = mean - f(px, py);
            sse += err * err;
        }
    }
    let rmse = (sse / (grid * grid) as f64).sqrt();
    // The amplitude is 1; a coarse-trend under-fit would leave rmse ~ O(1).
    // Resolving the surface drives it well below the signal scale.
    assert!(
        rmse < 0.2,
        "refinement failed to resolve the high-frequency truth: rmse {rmse}"
    );
}

/// Quasi-uniformity guard gate (#1032, caveat 2). The BPX n-independent CG
/// iteration bound rests on the metric-scaled net being quasi-uniform; a
/// near-degenerate metric (the cloud collapsed onto a sheet in `z`) breaks it.
/// The guard must DETECT this from the metric-scaled aspect ratio up front and
/// refuse the iterative solve — so the auto-route falls back to the dense
/// kernel BEFORE paying an unbounded CG — while leaving the well-conditioned
/// (isotropic) case certified. We assert both directions: the benign metric
/// certifies and fits; the collapsed metric is rejected by the guard.
#[test]
fn quasi_uniformity_guard_rejects_degenerate_metric_keeps_benign() {
    let (axes, y, w) = sample(2, 1200, 0.05, 0xCA5_CADE);
    let xs = axis_refs(&axes);

    // Benign: an isotropic unit metric leaves the cloud quasi-uniform in z.
    let design_ok = ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.0, 2)
        .expect("benign design build");
    assert!(
        design_ok.quasi_uniformity_certified(),
        "isotropic unit metric must certify; aspect_ratio={}",
        design_ok.metric_scaled_aspect_ratio()
    );
    assert!(
        design_ok.metric_scaled_aspect_ratio() < 5.0,
        "unit-metric uniform cloud should be nearly isotropic, got aspect_ratio={}",
        design_ok.metric_scaled_aspect_ratio()
    );
    // The full magic-default fit succeeds on the benign metric.
    fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("benign cascade fit");

    // Degenerate: scale axis 1 down by 1e5, collapsing the metric-scaled cloud
    // onto axis 0. The aspect ratio blows past the ceiling and the guard fires.
    let collapse = [1.0, 1.0e-5];
    let design_bad = ResidualCascadeDesign::build(&xs, &y, &w, &collapse, 2.0, 2)
        .expect("degenerate design still builds (the guard, not build, rejects)");
    assert!(
        !design_bad.quasi_uniformity_certified(),
        "a 1e5-anisotropic metric must FAIL the quasi-uniformity certificate; \
         aspect_ratio={}",
        design_bad.metric_scaled_aspect_ratio()
    );
    // The full magic-default fit refuses the degenerate metric with the typed
    // computation failure that the auto-route reads as "fall back to dense".
    match fit_residual_cascade(&xs, &y, &w, &collapse, 2.0) {
        Ok(_) => panic!("degenerate metric must be refused by the quasi-uniformity guard"),
        Err(ResidualCascadeError::Computation(_)) => {}
        Err(err) => panic!("expected quasi-uniformity computation failure, got: {err}"),
    }
}

/// Persistence round-trip (#1032 solver prerequisite): `to_state` → JSON →
/// `from_state` rebuilds a predict-capable fit WITHOUT the training CSR (the
/// reconstructed `Core` carries empty rows and the factored precision `L` of
/// `X'WX+λD`), and that fit reproduces the original posterior mean AND variance
/// at held-out points to solver roundoff. This is the prerequisite the
/// inference lane flagged; the inference-side payload/predict-replay rides on
/// top of this state type.
#[test]
fn cascade_state_roundtrip_reproduces_mean_and_variance() {
    let n = 2000;
    let noise = 0.1;
    let (axes, y, w) = sample(2, n, noise, 0x1032_0042);
    let xs = axis_refs(&axes);
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");

    let state = fit.to_state().expect("snapshot");
    let json = serde_json::to_string(&state).expect("serialize state");
    let restored_state: ResidualCascadeState =
        serde_json::from_str(&json).expect("deserialize state");
    let restored = ResidualCascadeFit::from_state(&restored_state).expect("restore fit");

    assert_eq!(restored.num_coeffs(), fit.num_coeffs());
    assert_eq!(restored.num_levels(), fit.num_levels());

    // Held-out points across the domain; the restored fit must match mean+SE to
    // solver roundoff (the factored precision is the SAME matrix the original
    // assembled under the dense cap; the variance solve replays through it).
    let grid = 17;
    let mut max_mean_err = 0.0_f64;
    let mut max_var_err = 0.0_f64;
    for i in 0..grid {
        for j in 0..grid {
            let px = (i as f64 + 0.37) / grid as f64;
            let py = (j as f64 + 0.61) / grid as f64;
            let (m0, v0) = fit.predict(&[px, py]).expect("orig predict");
            let (m1, v1) = restored.predict(&[px, py]).expect("restored predict");
            max_mean_err = max_mean_err.max((m0 - m1).abs() / (1.0 + m0.abs()));
            max_var_err = max_var_err.max((v0 - v1).abs() / (1.0 + v0.abs()));
        }
    }
    assert!(
        max_mean_err <= 1e-9,
        "mean drift across round-trip: {max_mean_err}"
    );
    assert!(
        max_var_err <= 1e-9,
        "variance drift across round-trip: {max_var_err}"
    );
}

/// A corrupt cascade snapshot fails loudly in `from_state`, never inside a
/// later `predict`.
#[test]
fn cascade_state_rejects_corruption() {
    let n = 800;
    let (axes, y, w) = sample(2, n, 0.1, 0x1032_0043);
    let xs = axis_refs(&axes);
    let fit = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.0).expect("cascade fit");
    let good = fit.to_state().expect("snapshot");

    let mut bad = good.clone();
    bad.coeff.pop();
    assert!(
        ResidualCascadeFit::from_state(&bad).is_err(),
        "coeff length mismatch must error"
    );

    let mut bad = good.clone();
    bad.sigma2 = -1.0;
    assert!(
        ResidualCascadeFit::from_state(&bad).is_err(),
        "non-positive sigma2 must error"
    );

    let mut bad = good.clone();
    bad.predict_chol.pop();
    assert!(
        ResidualCascadeFit::from_state(&bad).is_err(),
        "predict_chol size mismatch must error"
    );

    let mut bad = good.clone();
    bad.sobolev_s = 10.0;
    assert!(
        ResidualCascadeFit::from_state(&bad).is_err(),
        "out-of-window sobolev_s must error"
    );

    let mut bad = good;
    bad.predict_chol[0] = 0.0;
    assert!(
        ResidualCascadeFit::from_state(&bad).is_err(),
        "zero Cholesky pivot must error"
    );
}
