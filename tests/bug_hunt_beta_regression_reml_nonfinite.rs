//! Bug-hunt regression for issue #499.
//!
//! The Beta response family (`family="beta"`) could not be fit on any data: the
//! outer REML/LAML objective evaluated to a non-finite cost for every smoothing
//! parameter seed, so no seed survived startup validation and the fit aborted
//! before producing coefficients. This is the Beta analogue of the Gamma defect
//! fixed in #359 — the outer objective used the *full saturated* Beta
//! log-likelihood (including the `ln_gamma` normalizer), whose normalizer is
//! driven non-finite at the extreme rho / saturated-mu candidates probed during
//! seed screening, instead of the bounded scaled-deviance form the
//! Gamma/Tweedie branches use.
//!
//! RNG-free: a logit-linear mean `mu = logistic(0.3 + 1.6 x)` with a bounded
//! deterministic perturbation that keeps every `y` strictly in (0, 1). The test
//! asserts (a) Gaussian fits the same response, and (b) the Beta fit returns
//! finite coefficients with a positive slope.

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
