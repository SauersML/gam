//! FAILING TEST (potentially) — ticket: predictions just outside the training
//! support should remain bounded and not blow up by orders of magnitude.
//!
//! Fit a smooth on x ∈ [0, 1]. Predict at x ∈ {-0.05, -0.02, 1.02, 1.05}.
//! Truth peak-to-peak is ~2 (sin curve); a sane extrapolation should stay
//! within a few times the training range. We assert |pred| ≤ 10 (very
//! lenient — a real failure would be 100x larger, characteristic of TPS
//! kernels exploding outside the convex hull of centers).

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

fn fit_predict(formula: &str, x_train: &[f64], y_train: &[f64], x_test: &[f64]) -> Vec<f64> {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x_train
        .iter()
        .zip(y_train.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &v) in x_test.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn make_sin_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(17);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let n = 200usize;
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();
    (x, y)
}

#[test]
fn smooth_predictions_just_outside_training_range_stay_bounded() {
    init_parallelism();
    let (x, y) = make_sin_data();
    // Slightly outside both ends — well within "natural extrapolation" range.
    let x_test = vec![-0.05, -0.02, 1.02, 1.05];

    let cases: &[(&str, &str)] = &[
        ("smooth", "smooth(x)"),
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
    ];
    for (label, body) in cases {
        let pred = fit_predict(&format!("y ~ {body}"), &x, &y, &x_test);
        let abs_max = pred
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(pred.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs())));
        eprintln!(
            "[extrap] {label:10}  preds={:?}  abs_max={abs_max:.3}",
            pred
        );
        assert!(
            abs_max < 10.0,
            "{label}: extrapolation exploded to abs max {abs_max:.3} (budget 10.0); preds={pred:?}"
        );
    }
}
