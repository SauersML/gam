//! Sphere fit trained on data covering only one hemisphere must produce
//! BOUNDED (not necessarily accurate) predictions when asked to evaluate
//! on the opposite hemisphere. A blown-up extrapolation (NaN, ±1e10) is a
//! basis-conditioning failure.
//!
//! Training data: lat ∈ [10°, 80°] (Northern hemisphere only).
//! Held-out probes include the south pole and the antipodal hemisphere.

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

fn make_northern_only_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(10.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.9_f64, 179.9).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let lat_r = lat.to_radians();
        let lon_r = lon.to_radians();
        let y = 0.5 + 0.6 * lat_r.sin() + 0.3 * lat_r.cos() * (lon_r).cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_and_extrap(formula: &str) -> (Vec<f64>, f64, f64) {
    let data = make_northern_only_dataset(500, 31);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    // Probe BOTH hemispheres — interior (interpolation) + southern hemisphere
    // (full extrapolation).
    let mut pts = Vec::new();
    // Interior probes (within training range).
    for &lat in &[20.0_f64, 50.0, 75.0] {
        for j in 0..6 {
            let lon = -150.0 + 60.0 * j as f64;
            pts.push((lat, lon));
        }
    }
    // Extrapolation probes (Southern hemisphere).
    for &lat in &[-10.0_f64, -45.0, -80.0, -90.0] {
        for j in 0..6 {
            let lon = -150.0 + 60.0 * j as f64;
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
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (pred, mn, mx)
}

#[test]
fn sphere_wahba_single_hemisphere_extrap_bounded() {
    init_parallelism();
    let (pred, mn, mx) = fit_and_extrap("y ~ sphere(lat, lon, k=40)");
    assert!(pred.iter().all(|v| v.is_finite()), "Wahba produced NaN in extrap");
    // Training y range ≈ [-0.5, 1.5]. Even on the opposite hemisphere, a
    // well-penalized fit shouldn't extrapolate beyond ~[-5, 5] (a 5x
    // generous envelope) — anything wilder indicates the basis can shoot
    // off into the antipodal region.
    eprintln!("[hemi-wahba] full pred range [{mn:.3}, {mx:.3}]");
    assert!(
        mn > -5.0 && mx < 5.0,
        "Wahba sphere extrap exploded beyond ±5: [{mn:.3}, {mx:.3}]",
    );
}

#[test]
fn sphere_harmonic_single_hemisphere_extrap_bounded() {
    init_parallelism();
    let (pred, mn, mx) = fit_and_extrap("y ~ sphere(lat, lon, method=harmonic, max_degree=6)");
    assert!(pred.iter().all(|v| v.is_finite()), "harmonic produced NaN in extrap");
    eprintln!("[hemi-harm] full pred range [{mn:.3}, {mx:.3}]");
    // Harmonic basis is global so extrapolation is "for free" but penalty
    // strength determines how tame it stays. Same envelope.
    assert!(
        mn > -5.0 && mx < 5.0,
        "harmonic sphere extrap exploded beyond ±5: [{mn:.3}, {mx:.3}]",
    );
}
