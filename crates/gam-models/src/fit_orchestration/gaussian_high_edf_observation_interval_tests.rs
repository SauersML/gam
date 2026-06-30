//! End-to-end regression for #1765 at the real smooth-spline pipeline layer.
//!
//! The top-level `gam` crate cannot build in this environment (a `build.rs`
//! author tripwire), so the issue's `fit_from_formula` path is exercised here
//! in `gam-models`, which builds standalone. This drives a low-noise additive
//! `smooth(x1)+smooth(x2)+smooth(x3)` Gaussian fit through REML and asserts the
//! 95% OBSERVATION interval covers fresh held-out responses at the nominal rate.
//!
//! The observation half-width is `z·sqrt(σ̂² + Var(μ̂))`, dominated by the
//! residual scale `σ̂² = RSS/(n − edf_total)`. The #1765 symptom was that a
//! wrong residual-df denominator collapsed `σ̂²`, so the band narrowed and
//! coverage fell below 0.75 (PIT KS ~0.18). With the residual-df scale the
//! recovered `σ̂` is a consistent estimate of the actual prediction spread and
//! coverage returns to nominal. This test fails if the pipeline's σ̂ (via the
//! scale or the EDF feeding it) collapses again.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// A genuinely additive low-noise truth so REML fits the mean accurately and
/// the held-out spread is dominated by the (small) observation noise. The
/// observation interval must then track that noise, not collapse below it.
fn truth(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin()
        + 0.6 * (2.0 * std::f64::consts::PI * b).cos()
        + (c - 0.5).powi(2) * 3.0
}

/// One-sample Kolmogorov–Smirnov statistic against Uniform(0,1).
fn ks_vs_uniform(mut u: Vec<f64>) -> f64 {
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = u.len() as f64;
    let mut d = 0.0f64;
    for (i, &ui) in u.iter().enumerate() {
        let lo = i as f64 / n;
        let hi = (i as f64 + 1.0) / n;
        d = d.max((ui - lo).abs()).max((hi - ui).abs());
    }
    d
}

#[test]
fn gaussian_smooth_observation_interval_covers_nominal_1765() {
    let n = 2000usize;
    let n_test = 1000usize;
    let noise_sd = 0.02f64;
    let z95 = 1.959964_f64;

    let mut rng = StdRng::seed_from_u64(1765);
    let unif = Uniform::new(0.0_f64, 1.0).unwrap();
    let noise = Normal::new(0.0, noise_sd).unwrap();

    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows = Vec::with_capacity(n);
    let mut train_coords = Vec::with_capacity(n);
    let mut train_y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = unif.sample(&mut rng);
        let b = unif.sample(&mut rng);
        let c = unif.sample(&mut rng);
        let y = truth(a, b, c) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            y.to_string(),
        ]));
        train_coords.push((a, b, c));
        train_y.push(y);
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let (i1, i2, i3) = (col["x1"], col["x2"], col["x3"]);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x1) + smooth(x2) + smooth(x3)", &ds, &cfg)
        .expect("3-smooth gaussian fit");
    let StandardFitResult {
        fit, resolvedspec, ..
    } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit"),
    };

    let edf = fit.edf_total().expect("edf_total");
    let sigma = fit.standard_deviation;
    eprintln!("#1765 smooth pipeline: edf_total={edf:.2}, sigma_hat={sigma:.6} (true {noise_sd})");

    // Direct residual-df contract on the REAL pipeline scale: rebuild the
    // TRAINING design from the resolved spec, recompute RSS = Σ(y − Xβ)², and
    // require the reported σ̂² to equal RSS/(n − edf_total) — NOT the MLE RSS/n.
    // This is sensitive to the denominator regardless of the edf/n ratio, so it
    // catches a #1765 regression (MLE or null-space divisor) that the coverage
    // assertion below would miss when edf ≪ n.
    {
        let mut train_mat = Array2::<f64>::zeros((n, ds.headers.len()));
        for (k, &(a, b, c)) in train_coords.iter().enumerate() {
            train_mat[[k, i1]] = a;
            train_mat[[k, i2]] = b;
            train_mat[[k, i3]] = c;
        }
        let train_design = build_term_collection_design(train_mat.view(), &resolvedspec)
            .expect("train design");
        let fitted: Array1<f64> = train_design.design.matrixvectormultiply(&fit.beta);
        let rss: f64 = train_y
            .iter()
            .zip(fitted.iter())
            .map(|(&yi, &fi)| (yi - fi).powi(2))
            .sum();
        let sigma2_residual_df = rss / (n as f64 - edf);
        let sigma2_mle = rss / n as f64;
        let rel = (sigma * sigma - sigma2_residual_df).abs() / sigma2_residual_df.max(1e-18);
        eprintln!(
            "#1765 smooth pipeline scale: RSS={rss:.6}, σ̂²={:.10}, RSS/(n-edf)={sigma2_residual_df:.10}, RSS/n={sigma2_mle:.10}",
            sigma * sigma
        );
        assert!(
            rel < 1e-2,
            "pipeline σ̂²={:.10} must equal RSS/(n-edf)={sigma2_residual_df:.10} (rel={rel:.4}), \
             not the MLE RSS/n={sigma2_mle:.10}; high-EDF Gaussian observation intervals \
             depend on the residual-df scale (#1765)",
            sigma * sigma
        );
    }

    // Held-out test design + mean prediction.
    let mut test_mat = Array2::<f64>::zeros((n_test, ds.headers.len()));
    let mut coords = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let a = unif.sample(&mut rng);
        let b = unif.sample(&mut rng);
        let c = unif.sample(&mut rng);
        test_mat[[i, i1]] = a;
        test_mat[[i, i2]] = b;
        test_mat[[i, i3]] = c;
        coords.push((a, b, c));
    }
    let eval = build_term_collection_design(test_mat.view(), &resolvedspec).expect("eval design");
    let mu: Array1<f64> = eval.design.matrixvectormultiply(&fit.beta);

    let mut hits = 0usize;
    let mut pit = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let (a, b, c) = coords[i];
        let y_obs = truth(a, b, c) + noise.sample(&mut rng);
        // Observation interval centred at μ̂ with half-width z·σ̂ (the mean
        // variance term is negligible at this n and is omitted from the lower
        // bound on coverage, making the test conservative).
        if (y_obs - mu[i]).abs() <= z95 * sigma {
            hits += 1;
        }
        let u = 0.5 * (1.0 + libm_erf((y_obs - mu[i]) / (sigma * std::f64::consts::SQRT_2)));
        pit.push(u);
    }
    let coverage = hits as f64 / n_test as f64;
    let ks = ks_vs_uniform(pit);
    eprintln!("#1765 smooth pipeline: obs coverage={coverage:.3}, PIT KS={ks:.4}");

    assert!(
        (0.90..=0.99).contains(&coverage),
        "95% Gaussian observation interval coverage {coverage:.3} is off nominal \
         (edf={edf:.2}, sigma_hat={sigma:.6}); the residual scale that widens the \
         observation band is mis-estimated for the high-EDF smooth fit (#1765)"
    );
    assert!(
        ks < 0.08,
        "PIT KS {ks:.4} too large (issue reported ~0.18); the predictive scale is \
         mis-estimated for the smooth fit (#1765)"
    );
}

/// Error function via a rational approximation (Abramowitz & Stegun 7.1.26),
/// accurate to ~1e-7 — enough for the PIT KS guard. Avoids a libm dependency.
fn libm_erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
