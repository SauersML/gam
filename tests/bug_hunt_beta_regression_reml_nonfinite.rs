//! Bug-hunt regression for issue #499.
//!
//! The Beta response family (`family="beta"`) could not be fit on any data: the
//! outer REML/LAML objective evaluated to a non-finite cost for every smoothing
//! parameter seed, so no seed survived startup validation and the fit aborted
//! before producing coefficients.
//!
//! Root cause (verified by propagating the inner-solve error instead of
//! converting it to an infeasible `+inf` outer cost): every seed retreated with
//! `PerfectSeparationDetected` at a *flat* predictor (max|eta| ≈ 0.12), i.e. a
//! spurious separation flag. The PIRLS post-solve guard `detect_logit_instability`
//! was gated only on `link == Logit`, but the Beta family also fits through the
//! logit link. Its separation heuristics — the `yᵢ > 0.5` `order_separated`
//! split, μ→{0,1} saturation, working-weight collapse — are *binary*-response
//! concepts. Continuous Beta data that is monotone in `x` (μ increasing ⇒ rows
//! with y > 0.5 sit at higher η than rows with y ≤ 0.5) trivially satisfies
//! `order_separated`, so the fit was misclassified as separated and every
//! smoothing-parameter seed was rejected. The fix gates the detector strictly on
//! the Binomial response. (The original "non-finite saturated normalizer"
//! hypothesis in the issue body did not hold: the diagnostic showed zero
//! genuinely non-finite costs — all six rejections were separation false
//! positives.)
//!
//! RNG-free coverage:
//!   * `gaussian_baseline_fits_the_same_response` — Gaussian fits the response.
//!   * `beta_regression_fits_with_positive_slope` — `mu = logistic(0.3 + 1.6 x)`
//!     with a bounded sinusoidal perturbation; the Beta fit must return finite
//!     coefficients and recover a positive slope.
//!   * `beta_regression_fits_clean_monotone_separation_prone` — a different
//!     angle that targets the false-positive mechanism head-on: a *clean*,
//!     noise-free, steep monotone mean where every `y > 0.5` row sits strictly
//!     above every `y ≤ 0.5` row in both `x` and `η` (the maximal
//!     `order_separated` trigger). A logit-only detector flags this as
//!     separated; the family-gated detector must let it fit.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

#[inline]
fn logistic(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Deterministic, RNG-free data: `mu = logistic(0.3 + 1.6 x)` plus a small
/// bounded sinusoidal perturbation that keeps every response strictly inside
/// (0, 1). The mean is logit-increasing in `x`, so any honest fit must recover
/// a positive slope on the logit scale.
fn make_dataset() -> (Vec<f64>, Vec<f64>) {
    const N: usize = 240;
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let xi = -2.0 + 4.0 * (i as f64) / ((N - 1) as f64);
        let mu = logistic(0.3 + 1.6 * xi);
        // Bounded deterministic perturbation in (-0.18, 0.18) of the slack to
        // the nearest boundary, so y stays strictly inside (0, 1).
        let wobble = 0.18 * (3.0 * xi).sin() * mu.min(1.0 - mu);
        let yi = (mu + wobble).clamp(1.0e-4, 1.0 - 1.0e-4);
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

fn encode(x: &[f64], y: &[f64]) -> gam::inference::data::EncodedDataset {
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<csv::StringRecord> = (0..x.len())
        .map(|i| csv::StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset")
}

#[test]
fn gaussian_baseline_fits_the_same_response() {
    init_parallelism();
    let (x, y) = make_dataset();
    let ds = encode(&x, &y);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg)
        .expect("gaussian fit on bounded (0,1) data should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the gaussian baseline");
    };
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "gaussian coefficients must be finite"
    );
}

#[test]
fn beta_regression_fits_with_positive_slope() {
    init_parallelism();
    let (x, y) = make_dataset();
    let ds = encode(&x, &y);

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg)
        .expect("beta-regression fit on bounded (0,1) data should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "beta coefficients must be finite, got {:?}",
        fit.fit.beta
    );
    assert!(
        fit.fit.reml_score.is_finite(),
        "beta REML score must be finite, got {}",
        fit.fit.reml_score
    );

    // The simulated mean is logit-increasing in x, so the fitted response must
    // increase across the design. Compare the fitted linear predictor at the
    // smallest and largest x.
    let n = x.len();
    let eta = ds_predict_eta(&ds, &fit, &x);
    assert!(
        eta[n - 1] > eta[0] + 0.5,
        "beta fit must recover an increasing mean: eta[last]={} not sufficiently above eta[first]={}",
        eta[n - 1],
        eta[0]
    );
}

/// Clean, noise-free, steep monotone mean `mu = logistic(2.4 x)` over
/// `x ∈ [-1.5, 1.5]`. Because `y = mu` exactly, every row with `x > 0` has both
/// `y > 0.5` and a strictly larger logit than every row with `x < 0`: this is
/// the *maximal* `order_separated` trigger. A binomial-style separation guard
/// flags it as perfectly separated; a correctly family-gated guard recognizes
/// that a continuous Beta response is not a binary outcome and lets it fit.
fn make_monotone_dataset() -> (Vec<f64>, Vec<f64>) {
    const N: usize = 200;
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let xi = -1.5 + 3.0 * (i as f64) / ((N - 1) as f64);
        let mu = logistic(2.4 * xi);
        let yi = mu.clamp(1.0e-4, 1.0 - 1.0e-4);
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

#[test]
fn beta_regression_fits_clean_monotone_separation_prone() {
    init_parallelism();
    let (x, y) = make_monotone_dataset();
    let ds = encode(&x, &y);

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect(
        "beta fit on clean monotone (0,1) data must succeed: a continuous response \
         that is monotone in x is not perfectly separated, and must not be \
         misclassified by a binary-outcome separation guard",
    );
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "beta coefficients must be finite, got {:?}",
        fit.fit.beta
    );
    assert!(
        fit.fit.reml_score.is_finite(),
        "beta REML score must be finite, got {}",
        fit.fit.reml_score
    );

    // The steep monotone-increasing mean must be recovered on the logit scale.
    let n = x.len();
    let eta = ds_predict_eta(&ds, &fit, &x);
    assert!(
        eta.iter().all(|e| e.is_finite()),
        "fitted eta must be finite across the design"
    );
    assert!(
        eta[n - 1] > eta[0] + 2.0,
        "beta fit must recover the steep increasing mean: eta[last]={} not sufficiently above eta[first]={}",
        eta[n - 1],
        eta[0]
    );
}

/// Rebuild the design at the training points and form `eta = design * beta`.
fn ds_predict_eta(
    ds: &gam::inference::data::EncodedDataset,
    fit: &gam::StandardFitResult,
    x: &[f64],
) -> Vec<f64> {
    use gam::matrix::LinearOperator;
    use gam::smooth::build_term_collection_design;
    use ndarray::Array2;
    let col = ds.column_map();
    let x_idx = col["x"];
    let mut grid = Array2::<f64>::zeros((x.len(), ds.headers.len()));
    for i in 0..x.len() {
        grid[[i, x_idx]] = x[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    design.design.apply(&fit.fit.beta).to_vec()
}
