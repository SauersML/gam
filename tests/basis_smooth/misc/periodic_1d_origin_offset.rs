//! Test periodic 1D B-spline with non-zero origin: `s(x, periodic=true,
//! period=P, origin=O)`. The seam should be at x = O + P, and predictions
//! must wrap modulo P with origin O.

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

const PI: f64 = std::f64::consts::PI;

fn make_dataset(origin: f64, period: f64, n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(origin, origin + period).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Truth: cos((t - origin) * 2π / period) — periodic in t with the
    // declared period and origin.
    let y: Vec<f64> = t
        .iter()
        .map(|x| ((x - origin) * 2.0 * PI / period).cos() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(formula: &str, data: &gam::data::EncodedDataset, ts: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let n = ts.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &v) in ts.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn periodic_1d_origin_5_wraps_at_seam() {
    init_parallelism();
    let origin = 5.0_f64;
    let period = 7.0_f64;
    let data = make_dataset(origin, period, 200, 11);
    // Probe pairs straddling the seam at t = origin + period = 12.0.
    let probes = vec![
        origin,
        origin + period,
        origin + period - 1e-9,
        origin + 1e-9,
    ];
    let pred = predict(
        &format!("y ~ s(t, periodic=true, period={period}, origin={origin})"),
        &data,
        &probes,
    );
    let gap = (pred[0] - pred[1]).abs();
    eprintln!(
        "[per-origin] f(o)={:.6} f(o+P)={:.6} gap={:.3e}",
        pred[0], pred[1], gap
    );
    assert!(
        gap < 1e-6,
        "non-zero origin seam discontinuous: |f({origin}) - f({})| = {gap:.3e}",
        origin + period,
    );
}

#[test]
fn periodic_1d_origin_neg_pi_wraps_at_seam() {
    // Origin in negative coordinate, common for symmetric data.
    init_parallelism();
    let origin = -PI;
    let period = 2.0 * PI;
    let data = make_dataset(origin, period, 200, 11);
    let probes = vec![
        origin,
        origin + period,
        origin + period - 1e-9,
        origin + 1e-9,
    ];
    let pred = predict(
        &format!("y ~ s(t, periodic=true, period={period}, origin={origin})"),
        &data,
        &probes,
    );
    let gap = (pred[0] - pred[1]).abs();
    eprintln!(
        "[per-neg-origin] f(o)={:.6} f(o+P)={:.6} gap={:.3e}",
        pred[0], pred[1], gap
    );
    assert!(gap < 1e-6, "negative origin seam discontinuous: {gap:.3e}");
}

#[test]
fn periodic_1d_origin_wrap_invariance() {
    // f(t) must equal f(t + period) for any integer k. Verify on a
    // grid that probes inside, at, and outside the data range.
    init_parallelism();
    let origin = 3.0_f64;
    let period = 4.0_f64;
    let data = make_dataset(origin, period, 200, 11);
    let bases = [origin + 0.5, origin + 1.5, origin + 3.5];
    let mut probes = bases.to_vec();
    for k in [-1i32, 1, 2] {
        for &b in &bases {
            probes.push(b + (k as f64) * period);
        }
    }
    let pred = predict(
        &format!("y ~ s(t, periodic=true, period={period}, origin={origin})"),
        &data,
        &probes,
    );
    let n = bases.len();
    for (band, _k) in [-1i32, 1, 2].iter().enumerate() {
        for i in 0..n {
            let base = pred[i];
            let shifted = pred[(band + 1) * n + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "wrap invariance broken at probe {i} band {band}: {base:.6} vs {shifted:.6} diff={diff:.3e}",
            );
        }
    }
}
