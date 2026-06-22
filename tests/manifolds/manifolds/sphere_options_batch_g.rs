//! Batched cycles 75-78: rare sphere option combinations.

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

fn make_dataset(n: usize, seed: u64, radians: bool) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let (lat_lo, lat_hi) = if radians {
        (-1.4_f64, 1.4)
    } else {
        (-80.0, 80.0)
    };
    let (lon_lo, lon_hi) = if radians {
        (-3.1_f64, 3.1)
    } else {
        (-179.0, 179.0)
    };
    let u_lat = Uniform::new(lat_lo, lat_hi).expect("uniform");
    let u_lon = Uniform::new(lon_lo, lon_hi).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let lat_r = if radians { lat } else { lat.to_radians() };
        let y = 0.5 + 0.6 * lat_r.sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(
    formula: &str,
    data: gam::data::EncodedDataset,
    lats: &[f64],
    lons: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut m = Array2::<f64>::zeros((lats.len(), 3));
    for i in 0..lats.len() {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 75: sphere with radians=true input.
#[test]
fn sphere_radians_mode() {
    init_parallelism();
    let data = make_dataset(400, 7, true);
    let lats = vec![0.0_f64, 0.5, 1.0, 1.3, -0.5, -1.0];
    let lons = vec![0.0; lats.len()];
    let pred = fit_predict(
        "y ~ sphere(lat, lon, radians=true, k=20)",
        data,
        &lats,
        &lons,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[radians] range=[{mn:.3}, {mx:.3}]");
    assert!(mn > -2.0 && mx < 3.0);
}

/// Cycle 76: sphere harmonic with various L values.
#[test]
fn sphere_harmonic_l_sweep() {
    init_parallelism();
    let mut failures = Vec::new();
    for l in [2usize, 4, 6, 8, 10] {
        let data = make_dataset(400, 7, false);
        let lats = vec![45.0_f64, -45.0, 0.0];
        let lons = vec![0.0, 0.0, 90.0];
        let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree={l})");
        match std::panic::catch_unwind(|| fit_predict(&formula, data, &lats, &lons)) {
            Ok(pred) => {
                if !pred.iter().all(|v| v.is_finite()) {
                    failures.push(format!("L={l}: non-finite"));
                }
            }
            Err(_) => failures.push(format!("L={l}: panic")),
        }
    }
    assert!(failures.is_empty(), "harmonic L sweep: {failures:?}");
}

/// Cycle 77: sphere with double_penalty=false (ridge dropped).
#[test]
fn sphere_no_double_penalty() {
    init_parallelism();
    let data = make_dataset(400, 7, false);
    let lats = vec![45.0_f64, 0.0, -45.0];
    let lons = vec![0.0; 3];
    let pred = fit_predict(
        "y ~ sphere(lat, lon, k=20, double_penalty=false)",
        data,
        &lats,
        &lons,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[no-dp] range=[{mn:.3}, {mx:.3}]");
    assert!(mn > -1.5 && mx < 2.0);
}

/// Cycle 78: sphere with explicit knots= alias.
#[test]
fn sphere_explicit_centers_via_alias() {
    init_parallelism();
    // Sphere accepts both `k` and `centers` (parse_countwith_basis_alias).
    let data = make_dataset(400, 7, false);
    let lats = vec![45.0_f64, 0.0, -45.0];
    let lons = vec![0.0; 3];
    for spec in ["k=20", "centers=20"] {
        let formula = format!("y ~ sphere(lat, lon, {spec})");
        let pred = fit_predict(&formula, data.clone(), &lats, &lons);
        assert!(pred.iter().all(|v| v.is_finite()), "[{spec}] non-finite");
    }
}
