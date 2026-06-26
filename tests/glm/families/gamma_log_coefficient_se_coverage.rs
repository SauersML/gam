//! Behavioral regression guard for #679, from a different angle than the
//! statsmodels SE-matching test.
//!
//! `quality_vs_statsmodels_gamma_log_coefficient_se` checks the coefficient
//! covariance *scale* directly against a mature reference. This test checks the
//! *downstream consequence* practitioners actually depend on: that a nominal 95%
//! Wald confidence interval for the Gamma(log) linear predictor covers the truth
//! about 95% of the time over repeated sampling. It needs no external tool —
//! the truth `η = 1.5 + 0.6·x − 0.4·z` is known by construction.
//!
//! The #679 double-count shrank every coefficient SE by `√φ = 1/√shape` (here
//! `√(1/2.5) ≈ 0.632`), so each CI was ~37% too narrow. Frequentist coverage of
//! a Wald interval whose half-width is scaled by `c` is `P(|Z| ≤ c·z_{0.975})`;
//! at `c = 0.632` that is `P(|Z| ≤ 1.239) ≈ 0.785`. So the bug drives empirical
//! coverage down to ~0.78 while the corrected covariance restores ~0.95. The
//! gap between 0.95 and 0.78 is enormous relative to the Monte-Carlo error of a
//! few-hundred-replicate experiment, making this a sharp, non-flaky guard.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};

/// True coefficients of the data-generating linear predictor.
const B0: f64 = 1.5;
const BX: f64 = 0.6;
const BZ: f64 = -0.4;
const SHAPE: f64 = 2.5;

fn true_eta(x: f64, z: f64) -> f64 {
    B0 + BX * x + BZ * z
}

/// One replicate: simulate a Gamma(log) dataset, fit the parametric model, and
/// return the per-eval-point `(η̂, SE(η̂))` on the shared frozen basis.
fn fit_and_predict_eta(seed: u64, n: usize, eval: &[(f64, f64)]) -> Option<Vec<(f64, f64)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");

    let mut x = Vec::<f64>::with_capacity(n);
    let mut z = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let zi = ux.sample(&mut rng);
        let eta = true_eta(xi, zi);
        // E[y] = shape * scale = exp(eta) ⇒ scale = exp(eta)/shape; Var = mu^2/shape.
        let scale = eta.exp() / SHAPE;
        let g = Gamma::new(SHAPE, scale).expect("gamma(shape,scale)");
        let yi = g.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    if !y.iter().all(|&v| v > 0.0 && v.is_finite()) {
        return None;
    }

    let headers: Vec<String> = ["y", "x", "z"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string(), z[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).ok()?;
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ x + z", &ds, &cfg).ok()? else {
        return None;
    };

    // Build gam's frozen design at the fixed evaluation points (log link: η = Xβ).
    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &(xi, zi)) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
        grid[[i, z_idx]] = zi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).ok()?;
    let dense = design.design.to_dense();

    let gamma_log = LikelihoodSpec::new(
        ResponseFamily::Gamma,
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        gamma_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .ok()?;

    let eta_hat = pred.eta.to_vec();
    let eta_se = pred.eta_standard_error.to_vec();
    Some(eta_hat.into_iter().zip(eta_se).collect())
}

#[test]
fn gamma_log_eta_wald_intervals_have_nominal_coverage() {
    init_parallelism();

    // Fixed evaluation grid spanning the interior of the covariate domain. A
    // mix of interior and near-boundary points gives a representative average
    // coverage rather than one dominated by a single leverage regime.
    let eval: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (0.6, -0.4),
        (-0.6, 0.4),
        (0.8, 0.8),
        (-0.8, -0.8),
        (0.5, -0.5),
        (-0.3, 0.7),
    ];

    let n = 250usize;
    let replicates = 300usize;
    // 0.975 standard-normal quantile (two-sided 95%).
    let z975 = 1.959_963_984_540_054_f64;

    let mut covered = 0usize;
    let mut total = 0usize;
    let mut usable_replicates = 0usize;

    for r in 0..replicates {
        // Distinct, well-separated seeds per replicate.
        let seed = 0x5eed_0000u64 + (r as u64) * 2_654_435_761;
        let Some(results) = fit_and_predict_eta(seed, n, &eval) else {
            continue;
        };
        usable_replicates += 1;
        for (i, &(xi, zi)) in eval.iter().enumerate() {
            let (eta_hat, se) = results[i];
            if !(se > 0.0 && se.is_finite() && eta_hat.is_finite()) {
                continue;
            }
            let truth = true_eta(xi, zi);
            let lo = eta_hat - z975 * se;
            let hi = eta_hat + z975 * se;
            if truth >= lo && truth <= hi {
                covered += 1;
            }
            total += 1;
        }
    }

    assert!(
        usable_replicates >= replicates * 9 / 10,
        "too many replicates failed to fit ({usable_replicates}/{replicates}); \
         coverage estimate would be unreliable"
    );
    assert!(total > 0, "no usable coverage trials");

    let coverage = covered as f64 / total as f64;

    // The #679 double-count signature: half-widths scaled by √(1/shape) collapse
    // coverage to P(|Z| ≤ √(1/shape)·z_{0.975}).
    let shrink = (1.0_f64 / SHAPE).sqrt();
    let buggy_coverage = {
        // Φ(shrink·z975) via erf; std normal CDF.
        let a = shrink * z975 / std::f64::consts::SQRT_2;
        erf(a) // P(|Z| ≤ shrink·z975) = erf(shrink·z975/√2)
    };

    eprintln!(
        "gamma(log) η Wald coverage: replicates={usable_replicates} trials={total} \
         empirical={coverage:.4} (nominal 0.95; #679 double-count would give \
         ~{buggy_coverage:.4})"
    );

    // PRIMARY: empirical coverage is close to nominal. The band is wide enough
    // to absorb Monte-Carlo error (~±0.015 at this replicate count) plus the
    // mild finite-sample / estimated-shape undercoverage Wald intervals exhibit
    // for Gamma GLMs (the deterministic run lands at ~0.933), yet far above the
    // ~0.78 the bug produces. The lower edge leaves headroom for small
    // cross-platform floating-point differences in the per-replicate fits.
    assert!(
        (0.91..=0.975).contains(&coverage),
        "Gamma(log) η Wald coverage {coverage:.4} outside [0.91, 0.975]; \
         nominal is 0.95 and the #679 double-count would give ~{buggy_coverage:.4}"
    );

    // GUARD: coverage must be unambiguously closer to nominal 0.95 than to the
    // double-count value — a direct rejection of the bug's signature.
    assert!(
        (coverage - 0.95).abs() < (coverage - buggy_coverage).abs(),
        "Gamma(log) coverage {coverage:.4} is closer to the #679 double-count \
         coverage {buggy_coverage:.4} than to nominal 0.95 — Vb is too narrow"
    );
}

/// Error function (Abramowitz & Stegun 7.1.26), accurate to ~1e-7 — ample for an
/// informational comparison value in the assertion message and guard.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}
