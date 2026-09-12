//! Failing-ticket regression: `s(x, k=K)` quality must be robust to the
//! choice of K on an easy truth. REML controls effective degrees of
//! freedom; the basis budget K just provides headroom.
//!
//! Truth: sin(2π·x) on [0, 1], σ = 0.05, n = 240. For K ∈ {4, 6, 10, 20}
//! every fit must come within RMSE 0.04 (≈ noise floor) of the best function
//! its own K-dimensional basis can represent. That approximation floor is the
//! RMSE of the least-squares projection of the truth onto the rebuilt design,
//! added in quadrature. K = 4 spans a single cubic, and the best cubic
//! approximation of sin(2π·x) already has RMSE ≈ 0.067, so no coefficient
//! vector reaches 0.04 there. For K ≥ 6 the floor is negligible and the bar is
//! 0.04.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
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
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `formula`, then return its predictions on `x_test` and the dense design
/// they were computed from.
fn fit_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_test: &[f64],
) -> (Vec<f64>, Array2<f64>) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("spline-k fit");
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
    (
        design.design.apply(&fit.fit.beta).to_vec(),
        design.design.to_dense(),
    )
}

/// RMSE of the least-squares projection of `truth` onto the columns of
/// `design`: the smallest error any coefficient vector in this basis attains
/// on the test grid.
fn basis_projection_rmse(design: &Array2<f64>, truth: &[f64]) -> f64 {
    let mut frame: Vec<Array1<f64>> = Vec::with_capacity(design.ncols());
    for column in design.columns() {
        let mut v = column.to_owned();
        let column_norm = v.dot(&v).sqrt();
        // Two Gram–Schmidt passes keep the frame orthonormal to rounding.
        for _ in 0..2 {
            for u in &frame {
                let projection = u.dot(&v);
                v.scaled_add(-projection, u);
            }
        }
        let norm = v.dot(&v).sqrt();
        if norm > design.nrows() as f64 * f64::EPSILON * column_norm {
            frame.push(v / norm);
        }
    }
    let mut residual = Array1::from(truth.to_vec());
    for u in &frame {
        let projection = u.dot(&residual);
        residual.scaled_add(-projection, u);
    }
    (residual.dot(&residual) / truth.len() as f64).sqrt()
}

#[test]
fn spline_k_sweep_uniform_quality() {
    init_parallelism();
    let data = build_data(240, 0.05, 191);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();

    let ks = [4usize, 6, 10, 20];
    let budget = 0.04_f64;
    let mut violations = Vec::<String>::new();
    for &k in &ks {
        let formula = format!("y ~ s(x, k={k})");
        let (yhat, design) = fit_predict(&formula, &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        let floor = basis_projection_rmse(&design, &y_truth);
        let bar = floor.hypot(budget);
        eprintln!("[s-k] k={k:2} rmse={r:.4} basis_floor={floor:.4} bar={bar:.4}");
        if r > bar {
            violations.push(format!(
                "k={k}: rmse {r:.4} > {bar:.4} (basis floor {floor:.4} in quadrature with {budget:.2})"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "s(x, k=K) quality is not uniform across K:\n  - {}",
        violations.join("\n  - "),
    );
}
