//! Additive models combining new smooth families in a single formula.
//! Each component fits to its data dimension; sum-to-zero identifiability
//! must hold across all smooth terms.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;

fn build_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_theta = Uniform::new(0.0_f64, TAU).expect("uniform");
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "theta", "lat", "lon", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::with_capacity(500);
    for _ in 0..500 {
        let x = u_x.sample(&mut rng);
        let theta = u_theta.sample(&mut rng);
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.3 * x
            + 0.4 * theta.cos()
            + 0.2 * lat.to_radians().sin()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            theta.to_string(),
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit_predict(formula: &str) -> Result<(f64, f64), String> {
    let data = build_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    // Sample predict points
    let n = 50;
    let mut m = Array2::<f64>::zeros((n, 5));
    let mut rng = StdRng::seed_from_u64(99);
    for i in 0..n {
        m[[i, 0]] = rng.random::<f64>();
        m[[i, 1]] = TAU * rng.random::<f64>();
        m[[i, 2]] = -60.0 + 120.0 * rng.random::<f64>();
        m[[i, 3]] = -150.0 + 300.0 * rng.random::<f64>();
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite: {pred:?}"));
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((mn, mx))
}

#[test]
fn additive_smooth_plus_sphere() {
    init_parallelism();
    let (mn, mx) = try_fit_predict("y ~ s(x, k=8) + sphere(lat, lon, k=20)").expect("ok");
    eprintln!("[additive] s+sphere: [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "additive out of bounds");
}

#[test]
fn additive_periodic_plus_sphere() {
    init_parallelism();
    let (mn, mx) = try_fit_predict(
        "y ~ s(theta, periodic=true, period=6.283185307179586) + sphere(lat, lon, method=harmonic, max_degree=4)",
    ).expect("ok");
    eprintln!("[additive] per+sphere: [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "additive out of bounds");
}

#[test]
fn additive_three_smooths() {
    init_parallelism();
    let (mn, mx) = try_fit_predict(
        "y ~ s(x, k=8) + s(theta, periodic=true, period=6.283185307179586) + sphere(lat, lon, k=15)",
    ).expect("ok");
    eprintln!("[additive] three-smooth: [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "additive out of bounds");
}

#[test]
fn additive_bc_clamped_plus_sphere_harmonic() {
    init_parallelism();
    let (mn, mx) = try_fit_predict(
        "y ~ s(x, k=10, bc=clamped) + sphere(lat, lon, method=harmonic, max_degree=4)",
    )
    .expect("ok");
    eprintln!("[additive] bc-clamped+sphere-h: [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "additive out of bounds");
}

#[test]
fn additive_bc_anchored_plus_periodic() {
    init_parallelism();
    let (mn, mx) = try_fit_predict(
        "y ~ s(x, k=10, bc=anchored) + s(theta, periodic=true, period=6.283185307179586)",
    )
    .expect("ok");
    eprintln!("[additive] bc-anchored+per: [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "additive out of bounds");
}
