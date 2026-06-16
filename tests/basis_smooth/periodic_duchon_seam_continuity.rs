//! Regression: `duchon(x, periodic=true)` must fit through the formula DSL.
//!
//! The basis layer already had a wrapped-distance periodic Duchon basis. This
//! test locks in the full fit path, including REML smoothing selection, and
//! verifies that predictions agree across the periodic seam.

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

const TAU: f64 = std::f64::consts::TAU;

fn make_periodic_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut t: Vec<f64> = (0..n.saturating_sub(2))
        .map(|_| u.sample(&mut rng))
        .collect();
    t.push(0.0);
    t.push(TAU);
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + 0.3 * (2.0 * theta).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, probes: &[f64]) -> Vec<f64> {
    init_parallelism();
    let data = make_periodic_dataset(240, 0.04, 11);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("periodic Duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &t) in probes.iter().enumerate() {
        m[[i, 0]] = t;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn periodic_duchon_formula_fits_and_wraps_at_seam() {
    let probes = [0.0, 1.0e-6, TAU - 1.0e-6, TAU];
    for formula in &[
        "y ~ duchon(t, periodic=true, k=18)",
        "y ~ duchon(t, cyclic=true, k=18)",
    ] {
        let pred = fit_predict(formula, &probes);
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "non-finite prediction for {formula}"
        );
        assert!(
            (pred[0] - pred[3]).abs() < 1.0e-2,
            "periodic Duchon seam mismatch for {formula}: f(0)={} f(2pi)={}",
            pred[0],
            pred[3]
        );
        assert!(
            (pred[1] - pred[2]).abs() < 0.05,
            "periodic Duchon near-seam predictions diverged for {formula}: left={} right={}",
            pred[1],
            pred[2]
        );
    }
}
