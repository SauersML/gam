//! Sphere with binomial(logit) family: a common case for spatial
//! prevalence/risk modeling. Both Wahba and harmonic methods must
//! produce finite predictions and a plausible-looking surface.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

fn truth_prob(lat: f64, lon: f64) -> f64 {
    let lat_r = lat.to_radians();
    let lon_r = lon.to_radians();
    let logit = -0.5 + 1.5 * lat_r.sin() + 0.8 * (lon_r).cos();
    1.0 / (1.0 + (-logit).exp())
}

fn make_binom_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.9_f64, 179.9).expect("uniform");
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let p = truth_prob(lat, lon);
        let y = if u01.sample(&mut rng) < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_and_predict_grid(formula: &str) -> Vec<f64> {
    let data = make_binom_dataset(800, 17);
    let cfg = FitConfig {
        family: Some("binomial(logit)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut pts = Vec::new();
    for i in 0..10 {
        let lat = -60.0 + 120.0 * (i as f64) / 9.0;
        for j in 0..20 {
            let lon = -160.0 + 320.0 * (j as f64) / 19.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn sphere_wahba_binomial_logit_fit_succeeds_and_predicts_finite() {
    init_parallelism();
    let pred = fit_and_predict_grid("y ~ sphere(lat, lon, k=30)");
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite eta");
    // logit eta should be bounded — logits from -10 to +10 cover probs from
    // ~5e-5 to ~1-5e-5. A reasonable fit shouldn't have wild eta swings.
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sph-wahba-binom] eta range [{mn:.3}, {mx:.3}]");
    assert!(mn > -15.0 && mx < 15.0, "eta exploded: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_harmonic_binomial_logit_fit_succeeds_and_predicts_finite() {
    init_parallelism();
    let pred = fit_and_predict_grid("y ~ sphere(lat, lon, method=harmonic, max_degree=4)");
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite eta");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sph-harm-binom] eta range [{mn:.3}, {mx:.3}]");
    assert!(mn > -15.0 && mx < 15.0, "eta exploded: [{mn:.3}, {mx:.3}]");
}
