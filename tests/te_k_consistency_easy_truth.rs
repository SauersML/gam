//! Failing-ticket regression: `te(x1, x2, k=K)` quality on an easy 2D truth
//! must be monotone-or-flat as K increases. With REML smoothing, more basis
//! budget cannot hurt quality more than marginally.
//!
//! Truth: `0.7 * sin(2π·x1) + 0.5 * (x2 - 0.5)²` — a smooth additive
//! structure with peak-to-peak ≈ 2. σ = 0.10, n = 500. For k ∈ {6, 10, 14}
//! every fit should achieve RMSE ≤ 0.10 on the test grid. A failure where
//! k=14 (or some intermediate k) is materially worse than k=6 indicates a
//! basin / smoothing-init pathology.

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
    let f = |a: f64, b: f64| 0.7 * (2.0 * std::f64::consts::PI * a).sin() + 0.5 * (b - 0.5).powi(2);
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
    let result = fit_from_formula(formula, data, &cfg).expect("te fit");
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
fn te_k_sweep_uniform_quality_on_easy_2d_truth() {
    init_parallelism();
    let data = build_dataset(500, 0.10, 113);
    let g: Vec<f64> = (0..25).map(|i| 0.02 + 0.96 * i as f64 / 24.0).collect();
    let truth_fn =
        |a: f64, b: f64| 0.7 * (2.0 * std::f64::consts::PI * a).sin() + 0.5 * (b - 0.5).powi(2);

    let ks = [6usize, 10, 14];
    let budget = 0.10_f64;
    let mut violations = Vec::<String>::new();
    for &k in &ks {
        let formula = format!("y ~ te(x1, x2, k={k})");
        let (yhat, pts) = predict_grid(&formula, &data, &g);
        let truth: Vec<f64> = pts.iter().map(|&(a, b)| truth_fn(a, b)).collect();
        let r = rmse(&yhat, &truth);
        eprintln!("[te-k] k={k:2} rmse={r:.4}");
        if r > budget {
            violations.push(format!("k={k}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "te(x1, x2, k=K) quality is not uniform across K on easy 2D truth:\n  - {}",
        violations.join("\n  - "),
    );
}
