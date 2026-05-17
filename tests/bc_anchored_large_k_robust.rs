//! Verify the Hermite anchored fix (cycle 18) keeps the fit bounded at
//! large k (where the basis has many DoF that could otherwise oscillate
//! near the pin).

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
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn u(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn n(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.u() - 1.0;
            let v = 2.0 * self.u() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

fn make_sparse_dense() -> gam::data::EncodedDataset {
    let sigma = 0.05_f64;
    let mut rng = Lcg::new(505);
    let mut x = Vec::with_capacity(20 + 2000);
    for _ in 0..20 {
        x.push(0.5 * rng.u());
    }
    for _ in 0..2000 {
        x.push(0.5 + 0.5 * rng.u());
    }
    let f = |t: f64| (2.0 * PI * t).sin() + 0.3 * t;
    let y_noisy: Vec<f64> = x.iter().map(|&t| f(t) + sigma * rng.n()).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict_sparse(formula: &str) -> f64 {
    let data = make_sparse_dense();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, &data, &cfg).unwrap_or_else(|e| panic!("`{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let probes: Vec<f64> = (0..100).map(|i| 0.005 + 0.49 * (i as f64) / 99.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let truth: Vec<f64> = probes
        .iter()
        .map(|&t| (2.0 * PI * t).sin() + 0.3 * t)
        .collect();
    let mut worst = 0.0_f64;
    for (i, &t) in probes.iter().enumerate() {
        let dev = (pred[i] - truth[i]).abs();
        if dev > worst {
            worst = dev;
        }
        let _ = t;
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[bc-hermite-largek] `{formula}` worst sparse dev={worst:.3} range=[{mn:.3}, {mx:.3}]"
    );
    worst
}

#[test]
fn bc_anchored_hermite_holds_at_k50() {
    init_parallelism();
    let worst = fit_predict_sparse("y ~ s(x, bc=anchored, k=50)");
    // Pre-Hermite: rmse_sparse 0.55, max_dev ~3.2 at k=40. Post-Hermite: should
    // stay much smaller.
    assert!(
        worst < 2.0,
        "Hermite k=50 worst sparse dev {worst:.3} >= 2.0 — fix regressed"
    );
}

#[test]
fn bc_anchored_hermite_holds_at_k80() {
    init_parallelism();
    let worst = fit_predict_sparse("y ~ s(x, bc=anchored, k=80)");
    assert!(
        worst < 2.0,
        "Hermite k=80 worst sparse dev {worst:.3} >= 2.0 — fix regressed"
    );
}
