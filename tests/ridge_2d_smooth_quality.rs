//! Failing-ticket regression: a 2D smooth on a diagonal ridge truth
//! `f(x1, x2) = exp(-(x1 - x2)² / 0.01)` should recover the ridge profile
//! to within a reasonable RMSE budget.
//!
//! Setup: n=500, σ=0.10, evaluate on 30×30 test grid in [0.02, 0.98]².
//! Truth peak ≈ 1.0; 30% peak budget = 0.30 RMSE. A capable 2D smooth
//! (te, thinplate, matern in 2D, duchon) should achieve RMSE ≤ 0.20 on
//! this truth.

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

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let f = |a: f64, b: f64| (-((a - b).powi(2)) / 0.01).exp();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let a = ux.sample(&mut rng);
            let b = ux.sample(&mut rng);
            let y = f(a, b) + noise.sample(&mut rng);
            StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict_grid(
    formula: &str,
    data: &gam::data::EncodedDataset,
    g: &[f64],
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("ridge 2d fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let m = g.len();
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(m * m);
    let mut design_in = Array2::<f64>::zeros((m * m, 3));
    let mut row = 0;
    for &a in g {
        for &b in g {
            design_in[[row, 0]] = a;
            design_in[[row, 1]] = b;
            design_in[[row, 2]] = 0.0;
            pts.push((a, b));
            row += 1;
        }
    }
    let design =
        build_term_collection_design(design_in.view(), &fit.resolvedspec).expect("rebuild");
    (design.design.apply(&fit.fit.beta).to_vec(), pts)
}

fn rmse(yhat: &[f64], y: &[f64]) -> f64 {
    let n = y.len() as f64;
    (yhat
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

#[test]
fn ridge_2d_truth_recovered_by_2d_families() {
    init_parallelism();
    let data = build_dataset(500, 0.10, 167);
    let g: Vec<f64> = (0..30).map(|i| 0.02 + 0.96 * i as f64 / 29.0).collect();
    let truth_fn = |a: f64, b: f64| (-((a - b).powi(2)) / 0.01).exp();

    let cases: &[(&str, &str)] = &[
        ("te_k10", "te(x1, x2, k=10)"),
        ("matern_2d", "matern(x1, x2)"),
        ("duchon_2d", "duchon(x1, x2)"),
        ("thinplate", "thinplate(x1, x2)"),
    ];
    // Budget: peak ≈ 1, so 0.20 = 20% peak. Capable 2D smoothers should
    // achieve much better; this is a loose lower-bound budget.
    let budget = 0.20_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let (yhat, pts) = predict_grid(&format!("y ~ {body}"), &data, &g);
        let truth: Vec<f64> = pts.iter().map(|&(a, b)| truth_fn(a, b)).collect();
        let r = rmse(&yhat, &truth);
        eprintln!("[ridge-2d] {label:10} rmse={r:.4}");
        if r > budget {
            violations.push(format!("{label}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "ridge 2D truth not recovered by 2D smoothers:\n  - {}",
        violations.join("\n  - "),
    );
}
