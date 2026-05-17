//! Verify that periodic Duchon smooth actually produces fits where
//! f(0) = f(2π) (and equivalently f(θ) = f(θ + 2π)) — this is the
//! fundamental contract of the `periodic=true` flag.
//!
//! Truth: `y = 1 + 0.6·cos(θ) + 0.3·sin(2θ)` on θ ∈ [0, 2π).

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
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
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

fn predict_at(formula: &str, data: &gam::data::EncodedDataset, ts: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = ts.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        m[[i, 0]] = ts[i];
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn periodic_duchon_1d_seam_continuity() {
    init_parallelism();
    let data = make_periodic_dataset(200, 0.05, 11);
    // duchon() doesn't natively take period= as a 1D parameter; the
    // canonical periodic 1D variant in the formula DSL uses the
    // periodic flag on a 1D coordinate.
    let probe = [0.0, 1e-9, TAU - 1e-9, TAU];
    let pred = predict_at(
        "y ~ duchon(t, periodic=true)",
        &data,
        &probe,
    );
    eprintln!(
        "[per-duchon] f(0)={:.6} f(ε)={:.6} f(2π-ε)={:.6} f(2π)={:.6}",
        pred[0], pred[1], pred[2], pred[3]
    );
    let gap = (pred[0] - pred[3]).abs();
    eprintln!("[per-duchon] |f(0) - f(2π)| = {gap:.6e}");
    assert!(
        gap < 1e-6,
        "periodic Duchon seam discontinuous: |f(0) - f(2π)| = {gap:.6e} > 1e-6",
    );
}

#[test]
fn periodic_duchon_1d_multi_wrap_invariance() {
    init_parallelism();
    let data = make_periodic_dataset(200, 0.05, 11);
    // f(θ + 2πk) must equal f(θ) for any integer k.
    let bases = [0.0_f64, 0.7, 1.9, 3.1, 4.5];
    let mut pts: Vec<f64> = bases.to_vec();
    for k in [-2i32, -1, 1, 2] {
        for &t in &bases {
            pts.push(t + (k as f64) * TAU);
        }
    }
    let pred = predict_at("y ~ duchon(t, periodic=true)", &data, &pts);
    let n = bases.len();
    for (band, _k) in [-2i32, -1, 1, 2].iter().enumerate() {
        for i in 0..n {
            let base = pred[i];
            let shifted = pred[(band + 1) * n + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "periodic Duchon wrap broken at probe {i} band {band}: {base:.6} vs {shifted:.6} diff={diff:.3e}",
            );
        }
    }
}
