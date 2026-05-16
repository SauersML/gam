//! FAILING TEST (potentially) — ticket: 95% predictive intervals on a smooth
//! 1D fit should cover the truth at ≥ ~90% of held-out points (calibration:
//! ~95% expected, allow some slack from finite-sample / smoothing bias).
//!
//! Repro path: fit `y ~ smooth(x)` on a sin curve + noise, call the
//! predict-with-uncertainty path via the saved-model JSON API. Compute the
//! fraction of test points whose [mean_lower, mean_upper] interval contains
//! the noise-free truth.
//!
//! If coverage drops below 80%, the predictive uncertainty is under-stated —
//! a real calibration bug (or a too-narrow conditional covariance).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn smooth_1d_95pct_predictive_interval_covers_truth() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(31);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let sigma = 0.10_f64;
    let noise = Normal::new(0.0, sigma).expect("normal");
    let n = 240usize;

    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!("expected standard fit") };

    // 200 held-out test points across the interior of the train range
    let n_test = 200;
    let mut new_data = Array2::<f64>::zeros((n_test, 2));
    let mut truth = Array1::<f64>::zeros(n_test);
    for i in 0..n_test {
        let xt = 0.02 + 0.96 * (i as f64) / ((n_test - 1) as f64);
        new_data[[i, 0]] = xt;
        new_data[[i, 1]] = 0.0;
        truth[i] = (2.0 * std::f64::consts::PI * xt).sin();
    }
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    let mean = test_design.design.apply(&fit.fit.beta);

    // Compute SE of the eta-scale prediction: SE_i = sqrt(diag(X_i Σ X_i^T))
    // Apply the conditional covariance from the fit.
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("standard Gaussian fit should expose a conditional covariance");
    // Materialise the test design as dense for the SE computation.
    let mut x_dense = Array2::<f64>::zeros((n_test, fit.fit.beta.len()));
    for i in 0..n_test {
        // Apply a one-hot vector to recover the i-th design row via the
        // operator's transpose-apply trick is overkill; instead build by
        // multiplying the canonical basis with apply().
        let mut e = Array1::<f64>::zeros(n_test);
        e[i] = 1.0;
        let row = test_design.design.apply_transpose(&e);
        for j in 0..row.len() {
            x_dense[[i, j]] = row[j];
        }
    }
    // SE_i = sqrt(x_i^T Σ x_i)
    let mut se = Array1::<f64>::zeros(n_test);
    for i in 0..n_test {
        let xi = x_dense.row(i).to_owned();
        let cxi: Array1<f64> = cov.dot(&xi);
        let var: f64 = xi.iter().zip(cxi.iter()).map(|(a, b)| a * b).sum();
        se[i] = var.max(0.0).sqrt();
    }
    // 95% normal interval on the mean
    let z = 1.96_f64;
    let mut covered = 0usize;
    for i in 0..n_test {
        let lo = mean[i] - z * se[i];
        let hi = mean[i] + z * se[i];
        if truth[i] >= lo && truth[i] <= hi {
            covered += 1;
        }
    }
    let coverage = (covered as f64) / (n_test as f64);
    let avg_se = se.iter().sum::<f64>() / (n_test as f64);
    eprintln!(
        "[predict-coverage] n_test={n_test}  coverage={coverage:.3}  avg_se={avg_se:.4}  noise σ={sigma:.3}"
    );

    // The conditional-covariance interval is for the *mean* on the eta scale.
    // For a well-calibrated smooth fit on smooth truth, ≥ 0.80 is a reasonable
    // lower bound; nominally 0.95.
    assert!(
        coverage >= 0.80,
        "95% mean interval covered only {:.1}% of held-out points (expected ≥ 80%, ideally ~95)",
        coverage * 100.0,
    );
}
