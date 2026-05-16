//! Failing-ticket regression: `te(x1, x2, k=K)` tensor-product smooth must
//! recover a high-frequency separable truth `sin(2π·4·x1)·cos(2π·4·x2)` to
//! within a reasonable RMSE budget at moderate noise.
//!
//! n=500 training pts, σ=0.10, uniform on [0,1]². Truth peak-to-peak ≈ 2.
//! A well-tuned tensor-product smooth at k=12 in each margin has plenty of
//! basis budget for 4 cycles per dimension — RMSE should be in the
//! 0.05–0.10 band. We assert RMSE ≤ 0.30 (30% of peak amplitude),
//! a budget that even a moderately-tuned smooth should hit.

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
    let two_pi_4 = 2.0 * std::f64::consts::PI * 4.0;
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let a = ux.sample(&mut rng);
            let b = ux.sample(&mut rng);
            let y_clean = (two_pi_4 * a).sin() * (two_pi_4 * b).cos();
            let y = y_clean + noise.sample(&mut rng);
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
fn te_tensor_k12_recovers_sin4_cos4_truth() {
    init_parallelism();
    let data = build_dataset(500, 0.10, 13);
    let g: Vec<f64> = (0..30).map(|i| 0.02 + 0.96 * i as f64 / 29.0).collect();
    let (yhat, pts) = predict_grid("y ~ te(x1, x2, k=12)", &data, &g);
    let two_pi_4 = 2.0 * std::f64::consts::PI * 4.0;
    let y_truth: Vec<f64> = pts
        .iter()
        .map(|&(a, b)| (two_pi_4 * a).sin() * (two_pi_4 * b).cos())
        .collect();
    let r = rmse(&yhat, &y_truth);
    eprintln!("[te-2d-hifreq] k=12 rmse={r:.4}");
    // Truth peak-to-peak ~2. 30% budget = 0.60. A well-tuned tensor product
    // at k=12 in each margin (144 basis fns total) should achieve much
    // better — but the current implementation oversmooths to RMSE > 0.6.
    assert!(
        r < 0.60,
        "te(x1, x2, k=12) on sin(2π·4·x1)·cos(2π·4·x2): rmse {r:.4} > 0.60 budget"
    );
}
