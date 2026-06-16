//! Failing-ticket regression: narrow-Gaussian-bump truths (a localized,
//! high-curvature feature on an otherwise flat field) often cause smooth
//! family fits to either oversmooth the peak or undersmooth the tails.
//!
//! Truth: f(x) = exp(-(x-0.5)² / 0.005)  — a peak of amplitude 1.0 around
//! x = 0.5 with FWHM ≈ 0.17. At σ = 0.05, n = 300, a sane fit should:
//!   • recover the peak height to within 0.30 (i.e. predicted max ≥ 0.70)
//!   • keep the off-peak baseline near zero (max |fit(x)| at x∈{0.05,0.95} ≤ 0.20)
//!
//! These two checks together rule out both (a) collapse-to-constant and
//! (b) "fits the peak but lifts the baseline" failure modes.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn build_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (-((t - 0.5).powi(2)) / 0.005).exp() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: &gam::data::EncodedDataset, x_test: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("narrow-bump fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn narrow_gaussian_bump_peak_and_baseline() {
    init_parallelism();
    let data = build_data(300, 0.05, 23);

    // Dense grid for peak; the peak is at x=0.5
    let x_dense: Vec<f64> = (0..401).map(|i| i as f64 / 400.0).collect();
    // Off-peak baseline points (well outside the bump's 3σ ≈ 0.21)
    let baseline_pts: Vec<f64> = vec![0.05, 0.10, 0.90, 0.95];

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("matern_nu52", "matern(x, nu=5/2)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat_dense = fit_predict(&format!("y ~ {body}"), &data, &x_dense);
        let yhat_base = fit_predict(&format!("y ~ {body}"), &data, &baseline_pts);
        let peak_pred = yhat_dense.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let baseline_abs = yhat_base.iter().cloned().map(f64::abs).fold(0.0, f64::max);
        eprintln!("[narrow-bump] {label:14} peak={peak_pred:.3} baseline_abs={baseline_abs:.3}");
        if peak_pred < 0.70 {
            violations.push(format!(
                "{label}: peak prediction {peak_pred:.3} < 0.70 (oversmoothed bump)"
            ));
        }
        if baseline_abs > 0.20 {
            violations.push(format!(
                "{label}: baseline |fit| {baseline_abs:.3} > 0.20 (lifted off-peak baseline)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "narrow Gaussian bump fit quality regressions:\n  - {}",
        violations.join("\n  - "),
    );
}
