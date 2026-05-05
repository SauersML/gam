//! Scaling-law probe for the bernoulli marginal-slope INNER PIRLS Newton
//! solve at biobank shape.
//!
//! The companion outer-score probe (`tests/standard_gam_scaling.rs` and the
//! in-module `bernoulli_marginal_slope::tests::margslope_sigma_psi_scaling_law`)
//! already established that the OUTER ψ first-order eval is ~0.6 s/call at
//! n=320k — only ~8% of the 2400 s cmd budget. The hypothesis under test
//! here is that the dominant cost lives inside the inner PIRLS Newton solve
//! (i.e. per-row sextic kernel evaluation × inner-Newton iterations × outer
//! BFGS iterations), and that the path #3 inner-iter schedule cap is
//! actually doing the work it claims.
//!
//! Run with:
//! ```text
//! cargo test --release --test margslope_inner_pirls_scaling \
//!     -- --ignored --nocapture margslope_inner_pirls_scaling_law
//! ```
//!
//! The `[MS-INNER-SCALING]` lines in the output are pivotable: parse them
//! into (n, total_s, outer_iters, inner_cycles) and fit
//! `total_s = a · n^α`. Honest fit: if R²<0.85 or max log-resid>0.5,
//! refuses to extrapolate.

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::resource::ResourcePolicy;
use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LinkFunction};
use gam::{
    BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

const SEED: u64 = 0x5CA1_AB1E_5C0F_E5A1;

/// Builds a synthetic margslope problem with one cubic B-spline smooth on
/// a covariate column `x ~ Uniform[0,1]` for the marginal block, and an
/// intercept-only logslope block. The latent score `z` is sampled
/// standard-normal independently of `x` (so the latent-z policy's strict
/// N(0,1) check passes). `y` is drawn from a probit link applied to
/// `f(x) + 0.3·z` with a moderately nonlinear `f`.
struct Problem {
    data: Array2<f64>,
    spec: BernoulliMarginalSlopeTermSpec,
}

fn build_problem(n: usize) -> Problem {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));
    // Covariate `x` placed in column 0 of the data matrix; the smooth
    // term references `feature_col: 0`.
    let x_raw: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
    let mut data = Array2::<f64>::zeros((n, 1));
    for (i, &xi) in x_raw.iter().enumerate() {
        data[[i, 0]] = xi;
    }

    // Latent score: standard normal, independent of x. Use the
    // Box-Muller transform on two uniforms drawn with a fresh stream
    // of the seeded RNG so the shape passes the N(0,1) sanity check.
    let mut z = Array1::<f64>::zeros(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        z[i] = r * theta.cos();
        if i + 1 < n {
            z[i + 1] = r * theta.sin();
        }
        i += 2;
    }

    // Truth: f(x) is a moderate sinusoidal nonlinearity, plus a linear
    // contribution from z so the marginal-slope direction is non-trivial.
    let two_pi = std::f64::consts::TAU;
    let true_eta: Array1<f64> = Array1::from_iter(
        x_raw
            .iter()
            .zip(z.iter())
            .map(|(&xi, &zi)| (two_pi * xi).sin() + 0.5 * (two_pi * 2.0 * xi).cos() + 0.3 * zi),
    );
    // Probit link: y = 1[Φ⁻¹(U) < η]  ⇔  P(y=1|η) = Φ(η).
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));
    let weights = Array1::ones(n);
    let marginal_offset = Array1::<f64>::zeros(n);
    let logslope_offset = Array1::<f64>::zeros(n);

    // Single 1D cubic B-spline smooth on `x` (column 0). 8 internal knots
    // → ~12 basis columns before identifiability constraint, ~11 after.
    // Default identifiability is weighted sum-to-zero; the family will
    // pick up the resulting penalty block (one penalty + null-space
    // shrinkage if double_penalty). We use the simpler single-penalty
    // setup to keep the inner-Newton geometry uncluttered.
    let smooth = SmoothTermSpec {
        name: "f_x".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 8,
                },
                double_penalty: false,
                identifiability: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
    };
    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    // Engage the FLEXIBLE margslope path: score_warp + link_dev cubic
    // deviation blocks turn the per-row sextic kernel evaluation on, which
    // is what drives inner-PIRLS Newton iterations at biobank scale. With
    // both `None` the family takes a closed-form rigid-probit path and the
    // inner solve trivialises to 1 cycle (verified: an earlier revision of
    // this test with `score_warp=None, link_dev=None` saw `inner_cycles=1`
    // at every n, hiding the bottleneck the probe is meant to measure).
    let dev_cfg = DeviationBlockConfig::triple_penalty_default();
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(LinkFunction::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: FrailtySpec::None,
        score_warp: Some(dev_cfg.clone()),
        link_dev: Some(dev_cfg),
        // FitWeighted normalization: tolerant of small departures from the
        // strict frozen-N(0,1) shape, which is what we have here since
        // the synthetic z is drawn from a finite-sample standard normal.
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
    };
    Problem { data, spec }
}

/// Inline rational-approximation erf so the test is self-contained without
/// pulling in a special-function dep. Abramowitz & Stegun 7.1.26, max
/// abs error ≈ 1.5e-7 — plenty for sampling Bernoulli labels.
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

struct Row {
    n: usize,
    total_s: f64,
    outer_iters: usize,
    inner_cycles: usize,
    converged: bool,
}

fn run_fit(n: usize) -> Row {
    let problem = build_problem(n);
    // Default options. The path #3 inner-PIRLS schedule is wired into
    // BlockwiseFitOptions::default(); we want to measure the production
    // path including its bandaid cap.
    let options = BlockwiseFitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let policy = ResourcePolicy::default_library();

    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: problem.data.view(),
        spec: problem.spec,
        options,
        kappa_options,
        policy,
    });

    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();

    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => Row {
            n,
            total_s: elapsed,
            outer_iters: out.fit.outer_iterations,
            inner_cycles: out.fit.inner_cycles,
            converged: out.fit.outer_converged,
        },
        Ok(_) => panic!("internal: wrong FitResult variant"),
        Err(e) => panic!("margslope inner-pirls scaling fit failed at n={n}: {e}"),
    }
}

#[test]
#[ignore]
fn margslope_inner_pirls_scaling_law() {
    gam::init_parallelism();
    // Warm up: a single small fit so the first measured timing isn't
    // dominated by Rayon pool init / faer factorization JIT / page
    // faults on freshly mmap'd allocator pages. Without this, n=2000
    // measured ~8× the n=5000 cost.
    let _ = run_fit(1_000);

    // Sweep doubling-ish to span the local-budget-friendly range. We
    // intentionally stop at 100k to keep the probe under a few minutes
    // in total.
    let ns: Vec<usize> = vec![2_000, 5_000, 10_000, 25_000, 50_000, 100_000];

    eprintln!(
        "\n[MS-INNER-SCALING] header n total_s outer_iters inner_cycles per_outer_s per_inner_s converged"
    );
    let mut rows: Vec<Row> = Vec::new();
    for &n in &ns {
        let row = run_fit(n);
        let per_outer = row.total_s / (row.outer_iters.max(1) as f64);
        let per_inner = row.total_s / (row.inner_cycles.max(1) as f64);
        eprintln!(
            "[MS-INNER-SCALING] row n={} total_s={:.3} outer_iters={} inner_cycles={} per_outer_s={:.4} per_inner_s={:.4} converged={}",
            row.n, row.total_s, row.outer_iters, row.inner_cycles, per_outer, per_inner, row.converged
        );
        rows.push(row);
    }

    // Honesty gate: only fit on rows that genuinely converged within the
    // outer-iter cap. `outer_iters < 60` checks we didn't hit
    // BlockwiseFitOptions::default().outer_max_iter = 60.
    let total_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            if r.total_s > 0.0 && r.total_s.is_finite() && r.outer_iters < 60 {
                Some((r.n as f64, r.total_s))
            } else {
                None
            }
        })
        .collect();

    eprintln!();
    let total_fit = report_power_law(
        "[MS-INNER-SCALING-TOTAL]",
        &total_pts,
        &[("n=320k", 320_000.0), ("n=1M", 1_000_000.0)],
        // Mission cmd budget is 2400s for the full margslope fit.
        2400.0,
    );

    // Per-outer scaling: divide each row's total by its outer_iters first
    // so we attribute scaling-with-n to within-outer cost rather than
    // outer-iter count growth.
    let per_outer_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            if r.total_s > 0.0 && r.outer_iters > 0 && r.outer_iters < 60 {
                Some((r.n as f64, r.total_s / r.outer_iters as f64))
            } else {
                None
            }
        })
        .collect();
    eprintln!();
    let _ = report_power_law(
        "[MS-INNER-SCALING-PER-OUTER]",
        &per_outer_pts,
        &[("n=320k", 320_000.0)],
        // Per-outer budget at biobank that lets the full fit fit in the
        // 2400s cmd budget assuming ~10 outer iters: 240s/outer.
        240.0,
    );

    // Per-inner scaling: divide by inner_cycles to estimate per-Newton-iter cost.
    let per_inner_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            if r.total_s > 0.0 && r.inner_cycles > 0 && r.outer_iters < 60 {
                Some((r.n as f64, r.total_s / r.inner_cycles as f64))
            } else {
                None
            }
        })
        .collect();
    eprintln!();
    let _ = report_power_law(
        "[MS-INNER-SCALING-PER-INNER]",
        &per_inner_pts,
        &[("n=320k", 320_000.0)],
        // Per-inner-iter budget assuming ~10 outer × ~10 inner = 100
        // total inner iters across the fit: 24s/inner.
        24.0,
    );

    // Verdict on the path #3 schedule: if the TOTAL fit extrapolation at
    // 320k is well under 2400s, the inner-iter cap is doing its job and
    // we don't need deeper algorithmic changes (warm-starting,
    // preconditioning) yet. If it's near or over, deeper fixes are needed.
    if let Some((alpha, a_coef, r2)) = total_fit {
        let pred_320k = a_coef * 320_000_f64.powf(alpha);
        eprintln!();
        eprintln!(
            "[MS-INNER-SCALING-VERDICT] total fit y ≈ {:.3e}·n^{:.3} (R²={:.3}); pred@320k={:.0}s ({:.1} min)",
            a_coef,
            alpha,
            r2,
            pred_320k,
            pred_320k / 60.0
        );
        if pred_320k <= 2400.0 {
            eprintln!(
                "[MS-INNER-SCALING-VERDICT] path #3 inner-iter cap is sufficient at n=320k ({:.1}× headroom under 2400s)",
                2400.0 / pred_320k
            );
        } else {
            eprintln!(
                "[MS-INNER-SCALING-VERDICT] path #3 inner-iter cap NOT sufficient at n=320k: pred={:.0}s > 2400s budget by {:.1}×",
                pred_320k,
                pred_320k / 2400.0
            );
        }
    }
}

/// Honest power-law fit + R² + extrapolation verdicts. Refuses to
/// extrapolate when the fit is poor (R²<0.85 or any log-residual >0.5).
/// Returns `Some((alpha, a, R²))` for the verdict block; `None` if the
/// fit is unusable.
fn report_power_law(
    tag: &str,
    points: &[(f64, f64)],
    extrapolate: &[(&str, f64)],
    budget_y: f64,
) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        eprintln!("{tag} INSUFFICIENT DATA: {} points (need ≥3)", points.len());
        return None;
    }
    let logs: Vec<(f64, f64)> = points.iter().map(|(x, y)| (x.ln(), y.ln())).collect();
    let n = logs.len() as f64;
    let sx: f64 = logs.iter().map(|(x, _)| x).sum();
    let sy: f64 = logs.iter().map(|(_, y)| y).sum();
    let sxx: f64 = logs.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = logs.iter().map(|(x, y)| x * y).sum();
    let alpha = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let log_a = (sy - alpha * sx) / n;
    let a = log_a.exp();
    let mean_y = sy / n;
    let ss_tot: f64 = logs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = logs
        .iter()
        .map(|(x, y)| {
            let pred = log_a + alpha * x;
            (y - pred).powi(2)
        })
        .sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let max_abs_log_resid: f64 = logs
        .iter()
        .map(|(x, y)| (y - (log_a + alpha * x)).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{tag} fit: y ≈ {:.3e} · x^{:.3}  | R²={:.4}  max|log-resid|={:.3} (×{:.2})  | n_points={}",
        a,
        alpha,
        r2,
        max_abs_log_resid,
        max_abs_log_resid.exp(),
        logs.len()
    );
    if r2 < 0.85 || max_abs_log_resid > 0.5 {
        eprintln!("{tag} REFUSING EXTRAPOLATION (R²<0.85 or max log-resid >0.5)");
        return None;
    }
    eprintln!("{tag} budget: {:.1}", budget_y);
    for (label, x_target) in extrapolate {
        let pred = a * x_target.powf(alpha);
        let verdict = if pred <= budget_y {
            format!("FITS ({:.1}× headroom)", budget_y / pred)
        } else {
            format!("OVER by {:.1}× ({:.1})", pred / budget_y, pred)
        };
        eprintln!(
            "{tag} extrap @ {label} (x={:.1e}): pred={:.4} → {}",
            x_target, pred, verdict
        );
    }
    Some((alpha, a, r2))
}
