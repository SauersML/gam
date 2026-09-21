//! Calibration gate for expectile-GAM uncertainty: pointwise 95% bands must
//! cover the true conditional τ-expectile at the nominal rate under
//! heteroscedastic noise, including in the tails.
//!
//! # Why this can fail
//!
//! An expectile fit is a LAWS fixed point of penalized weighted least squares
//! with weights `wᵢ = |τ − 1[rᵢ < 0]|`. Those weights encode the loss asymmetry,
//! not inverse variances, so the Gaussian working-model covariance `φ̂·H⁻¹` of
//! the last inner solve is not the estimator's variance: it spreads one pooled
//! `φ̂` over every row. Where `σ(x)` is large the band is too narrow, and at
//! τ = 0.05 / 0.95 the asymmetric weights make the pooled scale worse still.
//! The published covariance is instead the penalized Newey–Powell sandwich
//! `H⁻¹(c·Xᵀdiag(w²r²)X + φ̂·S_λ)H⁻¹`, whose meat reads the variance from the
//! residuals row by row.
//!
//! # Planted truth
//!
//! `y = sin(2πx) + (0.2 + 0.8x)·Z`, `Z ~ N(0, 1)`, `x ~ U(0, 1)`, `n = 800`.
//! For a location-scale law the conditional τ-expectile is exactly
//! `sin(2πx) + (0.2 + 0.8x)·e_τ`, `e_τ` the standard-normal τ-expectile, so the
//! reference is the math, not another tool's output.
//!
//! # What is asserted
//!
//! For τ ∈ {0.05, 0.5, 0.9, 0.95}, over independent replicates, the average
//! pointwise coverage of the 95% posterior-mean band on a grid, both over the
//! whole grid and over the high-noise half (`x > 0.5`, where `σ(x)` exceeds its
//! median), must not fall below nominal by more than a one-sided Monte Carlo
//! margin `z·SE`. `SE` is the across-replicate standard deviation of the
//! per-replicate coverage over `√replicates` — the sampling error of the
//! estimate itself — and `z` is the one-sided normal quantile at family-wise
//! false-alarm rate `FAMILY_ALPHA`, Bonferroni-split over every check.
//! Bayesian bands average their coverage across the function (Nychka 1988), so
//! only under-coverage is a defect; conservative coverage is not asserted
//! against.
//!
//! A separate check recomputes the sandwich directly from the fit's own
//! penalized Hessian and residuals and requires the published covariance to
//! equal it, so the identity the production code uses (`V = Vb + Vb Xᵀ D X Vb`)
//! is audited against the definition, not against itself.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::LikelihoodSpec;
use gam_linalg::faer_ndarray::FaerEigh;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_math::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use gam_predict::{
    InferenceCovarianceMode, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::TAU;

const TAUS: [f64; 4] = [0.05, 0.5, 0.9, 0.95];
const NOMINAL: f64 = 0.95;
const N: usize = 800;
const REPLICATES: usize = 20;
const GRID: usize = 100;
/// Family-wise probability that a correctly calibrated estimator fails this
/// test by Monte Carlo chance alone.
const FAMILY_ALPHA: f64 = 0.01;

fn truth_mean(x: f64) -> f64 {
    (TAU * x).sin()
}

fn truth_scale(x: f64) -> f64 {
    0.2 + 0.8 * x
}

/// τ-expectile of the standard normal: the root of
/// `τ·E[(Z−m)₊] = (1−τ)·E[(m−Z)₊]`, with `E[(Z−m)₊] = φ(m) − m(1−Φ(m))` and
/// `E[(m−Z)₊] = φ(m) + mΦ(m)`. The balance is strictly decreasing in `m`, so
/// bisection to the float fixed point is exact.
fn standard_normal_expectile(tau: f64) -> f64 {
    let balance = |m: f64| {
        tau * (normal_pdf(m) - m * (1.0 - normal_cdf(m)))
            - (1.0 - tau) * (normal_pdf(m) + m * normal_cdf(m))
    };
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            return mid;
        }
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

fn encode(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y)
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode expectile data")
}

fn simulate(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let z = Normal::new(0.0, 1.0).expect("normal");
    let x: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    let y = x
        .iter()
        .map(|&t| truth_mean(t) + truth_scale(t) * z.sample(&mut rng))
        .collect();
    (x, y)
}

fn fit_expectile(data: &gam::data::EncodedDataset, tau: f64) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some(format!("expectile({tau})")),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ s(x)", data, &cfg).expect("expectile fit") {
        FitResult::Standard(fit) => fit,
        _ => panic!("an expectile fit is a standard fit"),
    }
}

fn design_at(fit: &gam::StandardFitResult, x: &[f64]) -> Array2<f64> {
    let mut raw = Array2::<f64>::zeros((x.len(), 2));
    for (i, &t) in x.iter().enumerate() {
        raw[[i, 0]] = t;
    }
    build_term_collection_design(raw.view(), &fit.resolvedspec)
        .expect("rebuild expectile design")
        .design
        .to_dense()
}

/// Fraction of `rows` on which the 95% posterior-mean band covers the truth.
fn covered_fraction(lower: &Array1<f64>, upper: &Array1<f64>, truth: &[f64], rows: &[usize]) -> f64 {
    let hits = rows
        .iter()
        .filter(|&&i| lower[i] <= truth[i] && truth[i] <= upper[i])
        .count();
    hits as f64 / rows.len() as f64
}

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let m = values.len() as f64;
    let mean = values.iter().sum::<f64>() / m;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, (var / m).sqrt())
}

#[test]
fn expectile_bands_cover_true_expectile_under_heteroscedastic_noise() {
    init_parallelism();
    let grid: Vec<f64> = (0..GRID)
        .map(|i| 0.02 + 0.96 * i as f64 / (GRID - 1) as f64)
        .collect();
    let all_rows: Vec<usize> = (0..GRID).collect();
    let noisy_rows: Vec<usize> = (0..GRID).filter(|&i| grid[i] > 0.5).collect();
    let checks = TAUS.len() * 2;
    let z = -standard_normal_quantile(FAMILY_ALPHA / checks as f64).expect("normal quantile");
    let options = PredictUncertaintyOptions {
        confidence_level: NOMINAL,
        covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
        includeobservation_interval: false,
        ..PredictUncertaintyOptions::default()
    };

    let mut failures = Vec::new();
    let mut report = Vec::new();
    for &tau in &TAUS {
        let e_tau = standard_normal_expectile(tau);
        let truth: Vec<f64> = grid
            .iter()
            .map(|&t| truth_mean(t) + truth_scale(t) * e_tau)
            .collect();
        let mut overall = Vec::with_capacity(REPLICATES);
        let mut noisy = Vec::with_capacity(REPLICATES);
        for rep in 0..REPLICATES {
            // Seed block 1020..1040; this test never skips a replicate. The
            // sign-pattern cycle of seed 1001 at τ = 0.9 (gam#3039) is covered
            // by its own regression test, which reaches the generalized fixed
            // point.
            let (x, y) = simulate(1020 + rep as u64);
            let fit = fit_expectile(&encode(&x, &y), tau);
            let xg = design_at(&fit, &grid);
            let offset = Array1::<f64>::zeros(GRID);
            let pred = predict_gamwith_uncertainty(
                xg,
                fit.fit.beta.view(),
                offset.view(),
                LikelihoodSpec::gaussian_identity(),
                &fit.fit,
                &options,
            )
            .expect("expectile band");
            overall.push(covered_fraction(&pred.mean_lower, &pred.mean_upper, &truth, &all_rows));
            noisy.push(covered_fraction(&pred.mean_lower, &pred.mean_upper, &truth, &noisy_rows));
        }
        for (region, values) in [("whole grid", &overall), ("x > 0.5", &noisy)] {
            let (mean, se) = mean_and_se(values);
            let floor = NOMINAL - z * se;
            report.push(format!(
                "tau={tau} {region}: coverage {mean:.3} (MC se {se:.3}, floor {floor:.3})"
            ));
            if mean < floor {
                failures.push(format!(
                    "tau={tau} {region}: coverage {mean:.3} < {floor:.3} = {NOMINAL} - {z:.2}*{se:.3}"
                ));
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        failures.is_empty(),
        "expectile 95% bands under-cover the true expectile:\n{}\n\nall checks:\n{}",
        failures.join("\n"),
        report.join("\n")
    );
}

/// The published coefficient covariance equals the prior-inclusive sandwich
/// `H⁻¹(c·Xᵀdiag(w²r²)X + φ̂·S_λ)H⁻¹`, recomputed from the fit's own penalized
/// Hessian `H`, with `S_λ = H − XᵀWX` and `c = n/(n − edf)`.
#[test]
fn expectile_covariance_is_the_penalized_newey_powell_sandwich() {
    init_parallelism();
    let tau = 0.9;
    let (x, y) = simulate(7);
    let fit = fit_expectile(&encode(&x, &y), tau);
    let geometry = fit.fit.geometry.as_ref().expect("expectile fit publishes its geometry");
    assert!(
        geometry.coefficient_gauge.is_identity(),
        "an unconstrained s(x) fit is in the full coefficient frame"
    );
    let h = geometry.penalized_hessian.as_array().clone();
    let xd = design_at(&fit, &x);
    let beta = &fit.fit.beta;
    let residual = Array1::from_iter(y.iter().zip(xd.dot(beta)).map(|(yi, mi)| yi - mi));
    let w = residual.mapv(|r| if r < 0.0 { 1.0 - tau } else { tau });
    let edf = fit.fit.edf_total().expect("edf");
    let phi = fit.fit.coefficient_covariance_scale().expect("scale");
    let hc1 = N as f64 / (N as f64 - edf);

    let mut xtwx = Array2::<f64>::zeros(h.dim());
    let mut meat = Array2::<f64>::zeros(h.dim());
    for i in 0..N {
        let row = xd.row(i);
        let score = w[i] * residual[i];
        for a in 0..row.len() {
            for b in 0..row.len() {
                xtwx[[a, b]] += w[i] * row[a] * row[b];
                meat[[a, b]] += hc1 * score * score * row[a] * row[b];
            }
        }
    }
    let s_lambda = &h - &xtwx;
    let (eigen, basis) = h.eigh(faer::Side::Lower).expect("penalized Hessian eigendecomposition");
    assert!(eigen.iter().all(|&e| e > 0.0), "penalized Hessian is SPD");
    let h_inv = (&basis * &eigen.mapv(f64::recip)).dot(&basis.t());
    let direct = h_inv.dot(&(&meat + &(s_lambda * phi))).dot(&h_inv);

    let published = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("expectile fit publishes a dense covariance");
    let scale = direct.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = published
        .iter()
        .zip(direct.iter())
        .fold(0.0_f64, |m, (p, d)| m.max((p - d).abs()));
    // The two sides differ only by the floating-point error of two p×p
    // inverses of the same SPD matrix: bound it by its condition number.
    let cond = eigen.iter().cloned().fold(0.0_f64, f64::max)
        / eigen.iter().cloned().fold(f64::INFINITY, f64::min);
    let tolerance = cond * f64::EPSILON * h.nrows() as f64;
    assert!(
        worst <= tolerance * scale,
        "published covariance differs from the direct sandwich by {worst:.3e} \
         (scale {scale:.3e}, relative tolerance {tolerance:.3e})"
    );
    // And it is not the working-model covariance φ̂·H⁻¹ it replaced.
    let working = h_inv * phi;
    let departure = published
        .iter()
        .zip(working.iter())
        .fold(0.0_f64, |m, (p, g)| m.max((p - g).abs()));
    assert!(
        departure > tolerance * scale,
        "published covariance is still the Gaussian working form"
    );
}
