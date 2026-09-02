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

use gam::solver::residual_cascade::{ResidualCascadeDesign, ResidualCascadeError, ResidualCascadeFit, ResidualCascadeState, fit_residual_cascade};

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
        refinement.gain <= refinement.tolerance && refinement.evidence <= 0.0,
        "returned fit has a candidate level that still earns its own Occam factor: \
         gain {} vs break-even {} ({:+} nats)",
        refinement.gain,
        refinement.tolerance,
        refinement.evidence
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

    // The estimator-quality oracle is deliberately finite-resolution: four
    // levels are the last data-identified design on this 240-row fixture.
    // Automatic refinement has a separate boundary test below; conflating that
    // certificate with this norm-equivalence comparison used to send the
    // quality oracle into a rank-deficient fifth-level score search.
    let fit = ResidualCascadeDesign::build(&xs, &y, &w, &[1.0, 1.0], 2.5, 4)
        .expect("identified cascade design")
        .fit_reml()
        .expect("identified cascade REML");

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
///
/// # WHAT THIS GATE WAS RED ABOUT, AND WHAT ANSWERED IT (#2758, #2759)
///
/// It used to refuse on `CertifiedSpectrumCapacity` at 2893 of the 5997
/// directions the sample identifies — a MEMORY budget. #2758 took the certified
/// route's residency from seven-plus `m²` blocks to one packed triangle, the
/// derived width went 2896 → 10362, and the cascade refined to all 5997. It
/// then refused at the rank-maximal design instead, and #2759's first half
/// closed the bracket on what that refusal was made of:
///
/// ```text
///     level-(L+1) gain  ∈  [1.050801e-1, 1.055257e-1]      (bracket, closed)
///     1e-3·rss_pen       =  2.184455e-3                     48.1x below it
///     shipped ‖g‖²/(λd)  =  1.290191e-1                     22% conservative
/// ```
///
/// So the refusal was not the certificate being cautious — the decrease is
/// bounded away from that bar from BELOW. What the bracket left open is whether
/// that decrease is the thing the criterion should be reading, and the answer
/// is no. `1e-3·rss_pen` charges NOTHING for the width of the set it is buying,
/// and the set here is 32790 candidate columns against 5997 identifiable
/// directions: at the rank-maximal design the candidates are redundant against
/// the data's own row space, so what they buy is penalty dilution and noise
/// capacity, not discretization bias.
///
/// The charge that was missing is the candidate set's own Occam factor,
/// `occam = log det(S/(λd))` — the log-determinant of the SAME Schur complement
/// the gain is a quadratic form in. At the profiled σ̂² the restricted
/// log-likelihood moves by `[dof·log(rss_pen/rss_pen_refined) − occam]/2`, so
/// the break-even gain is `rss_pen·(1 − e^{−occam/dof})` and it is derived, not
/// chosen. Measured on this fixture at the rung that used to refuse:
///
/// ```text
///     occam            =  3.231e2       (over 32790 candidate columns)
///     break-even gain  =  1.146e-1      against gain 1.055e-1
///     restricted logL  = -1.318e1 nats  for one more level
/// ```
///
/// and the held-out truth agrees with the likelihood rather than with the bar:
/// refining anyway takes the 30×30-grid RMSE from 0.0155828 to 0.0155887. The
/// gate is green because the cascade now stops where the evidence turns over,
/// with the same assertions it always made, read on a criterion that can tell
/// the two apart.
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
        cert.gain.is_finite() && cert.gain >= 0.0 && cert.occam.is_finite() && cert.occam >= 0.0,
        "refinement comparison is not a finite non-negative certificate: {cert}"
    );
    assert!(cert.tolerance.is_finite() && cert.tolerance > 0.0);
    // One more level does not pay for its own dimension — the discretization is
    // certified spent, in the same currency λ was selected in.
    assert!(
        cert.gain <= cert.tolerance && cert.evidence <= 0.0,
        "returned fit has a candidate level that still earns marginal likelihood: {cert}"
    );
    // The regime this fixture exists to name: the design is rank-maximal, and
    // the gain there is still far above the fixed `1e-3·rss_pen` bar that used
    // to be read. A green gate on a fixture that had drifted out of that regime
    // would prove nothing about #2759.
    assert_eq!(
        fit.num_centers(),
        n - 3,
        "premise: this fixture must reach the rank-maximal design (n − nullity)"
    );
    assert!(
        cert.gain > 20.0 * 1e-3 * fit.rss_pen,
        "premise: the minted fit must be one the fixed relative bar would have refused, \
         got gain {} against 1e-3·rss_pen {}",
        cert.gain,
        1e-3 * fit.rss_pen
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
/// refuse the iterative solve BEFORE paying an unbounded CG, without silently
/// changing estimators, while leaving the well-conditioned
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
    //
    // This arm was narrowed to "not refused BY THE GUARD" while the refinement
    // loop refused at the rank-maximal design on this draw, with the
    // level-(L+1) gain certified in [3.560937e-2, 3.562072e-2] against a
    // 2.485628e-3 tolerance — a statement about the cascade's remaining gain
    // and not about the metric. #2759 replaced that tolerance with the
    // candidate set's own break-even gain, the level stopped paying for itself,
    // and the end-to-end claim is available again: a benign metric fits.
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
    // The full magic-default fit refuses the degenerate metric with a typed
    // computation failure; the selected route must not silently change
    // estimators.
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
    // Back on the magic-default route (#2759). This fixture was moved to a
    // fixed depth while the refinement loop refused at the rank-maximal design
    // on this draw — gain certified in [6.326893e-2, 6.327000e-2] against a
    // 1.690929e-2 tolerance — which is a claim about a quantity a serialization
    // gate does not measure. That tolerance is now the candidate set's own
    // break-even gain, the level does not earn it, and the fixture can exercise
    // the route a caller actually takes.
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
    // Back on the magic-default route (#2759). This fixture's subject is
    // `from_state`, and it was moved to a fixed depth while the refinement loop
    // refused at the rank-maximal design on this draw — gain certified in
    // [7.944884e-3, 7.946538e-3] against a 6.547001e-3 tolerance. The refusal
    // was honest about the gain and wrong about what to do with it: the level
    // it could not add was 1e-3 of the residual wide and thousands of columns
    // wide, and against its own Occam factor it does not pay for itself.
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

