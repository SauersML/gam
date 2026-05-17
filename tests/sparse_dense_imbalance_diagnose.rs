//! Diagnose the sparse_dense_imbalance COLLAPSED case from
//! `fit_quality_stress`. Compare several BC / k / smooth-type configs to
//! attribute the failure mode.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const PI: f64 = std::f64::consts::PI;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform_01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.uniform_01() - 1.0;
            let v = 2.0 * self.uniform_01() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

fn make_dataset() -> gam::data::EncodedDataset {
    let sigma = 0.05;
    let n_sparse = 20;
    let n_dense = 2000;
    let mut rng = Lcg::new(505);
    let mut x = Vec::with_capacity(n_sparse + n_dense);
    for _ in 0..n_sparse { x.push(0.5 * rng.uniform_01()); }
    for _ in 0..n_dense  { x.push(0.5 + 0.5 * rng.uniform_01()); }
    let f = |t: f64| (2.0 * PI * t).sin() + 0.3 * t;
    let y_noisy: Vec<f64> = x.iter().map(|&t| f(t) + sigma * rng.normal()).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y_noisy.iter())
        .map(|(a,b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn truth(t: f64) -> f64 { (2.0 * PI * t).sin() + 0.3 * t }

fn run(formula: &str) -> (f64, f64, f64) {
    let data = make_dataset();
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = match fit_from_formula(formula, &data, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[diag] FIT FAILED `{formula}` -- {e}");
            return (f64::INFINITY, f64::INFINITY, 0.0);
        }
    };
    let FitResult::Standard(fit) = result else { panic!() };
    let xg_sparse: Vec<f64> = (0..100).map(|i| 0.005 + 0.49 * i as f64 / 99.0).collect();
    let xg_dense: Vec<f64> = (0..100).map(|i| 0.505 + 0.49 * i as f64 / 99.0).collect();
    let mut all: Vec<f64> = xg_sparse.iter().chain(xg_dense.iter()).copied().collect();
    let n = all.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &v) in all.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let truth_all: Vec<f64> = all.iter().map(|&t| truth(t)).collect();
    let mut sum_sparse = 0.0; let mut sum_dense = 0.0;
    let mut max_dev_sparse = 0.0_f64; let mut max_dev_dense = 0.0_f64;
    for i in 0..100 {
        let d = pred[i] - truth_all[i];
        sum_sparse += d * d;
        max_dev_sparse = max_dev_sparse.max(d.abs());
    }
    for i in 100..200 {
        let d = pred[i] - truth_all[i];
        sum_dense += d * d;
        max_dev_dense = max_dev_dense.max(d.abs());
    }
    let rmse_sparse = (sum_sparse / 100.0).sqrt();
    let rmse_dense = (sum_dense / 100.0).sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[diag] `{formula}` rmse_sparse={rmse_sparse:.4} (max_dev={max_dev_sparse:.4}) rmse_dense={rmse_dense:.4} (max_dev={max_dev_dense:.4}) pred_range=[{mn:.3}, {mx:.3}]"
    );
    all.clear(); // suppress lint
    (rmse_sparse, rmse_dense, mx - mn)
}

#[test]
fn diagnose_sparse_dense_imbalance() {
    init_parallelism();
    // Free BC (baseline)
    let _ = run("y ~ s(x, k=20)");
    let _ = run("y ~ s(x, k=10)");
    let _ = run("y ~ s(x, k=40)");
    // BC anchored both
    let _ = run("y ~ s(x, bc=anchored, k=20)");
    let _ = run("y ~ s(x, bc=anchored, k=10)");
    let _ = run("y ~ s(x, bc=anchored, k=40)");
    // BC clamped both
    let _ = run("y ~ s(x, bc=clamped, k=20)");
    // BC anchored ONLY right (where data is)
    let _ = run("y ~ s(x, bc_right=anchored, k=20)");
    // BC anchored ONLY left (sparse side)
    let _ = run("y ~ s(x, bc_left=anchored, k=20)");
    // TPS
    let _ = run("y ~ s(x, type=tps, k=20)");
    assert!(true);
}
