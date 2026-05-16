//! FAILING TEST — ticket: sphere fits show a visible artifact near the poles
//! (≈60–80° latitude band) even on smooth low-frequency truth and ample
//! training data. The Wahba kernel default uses a *coefficient* sum-to-zero
//! identifiability (`Σβⱼ = 0` over centers, see
//! `src/terms/basis.rs:13809 coefficient_sum_to_zero_transform`, applied at
//! `:13859`), which is an arbitrary anchor that doesn't respect the
//! sphere's surface measure. With centers placed via Euclidean (lat, lon)
//! FPS (`select_thin_plate_knots` at `src/terms/basis.rs:18370`) and
//! training data uniformly distributed on the sphere, the polar centers
//! have very few nearby training rows; coefficient-sum-to-zero then pulls
//! their fitted values toward the global zero rather than letting them be
//! anchored by local data, producing a high-latitude bias.
//!
//! Confirmed empirically (agent investigation): on demo training data
//! (800 pts, σ=0.10, `sphere(lat, lon, radians=true, k=100)`), the
//! [+1.2, +1.4) rad latitude band has mean 3D-error ≈ 0.12 vs ≈ 0.061 at the
//! equator — 2× worse — and this reproduces in BOTH `method=wahba` and
//! `method=harmonic`. Fix direction: area-weighted sum-to-zero
//! (`wᵀβ = 0`, w = cos(lat_center) quadrature weight) and switch
//! `select_thin_plate_knots` to spherical-geodesic distance.

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

fn truth(lat: f64, lon: f64) -> f64 {
    // Smooth low-frequency signal — degree ≤ 2 in spherical harmonics.
    1.0 + 0.6 * lat.sin()
        + 0.4 * lat.cos() * (2.0 * lon).cos()
        + 0.3 * lat.cos().powi(2) * lon.sin()
}

fn make_training_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u11 = Uniform::new(-1.0, 1.0).expect("uniform");
    let u_lon = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let lat = u11.sample(&mut rng).asin();
            let lon = u_lon.sample(&mut rng);
            let y = truth(lat, lon) + noise.sample(&mut rng);
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(formula: &str, data: &gam::data::EncodedDataset,
           lats: &[f64], lons: &[f64]) -> Vec<f64>
{
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("sphere fit ok");
    let FitResult::Standard(fit) = result else { panic!("expected standard fit") };
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
        m[[i, 2]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn rmse_in_lat_band(formula: &str, data: &gam::data::EncodedDataset,
                    lat_lo: f64, lat_hi: f64) -> (f64, usize)
{
    // Build a grid: NLAT × NLON in the requested latitude band.
    let nlat = 12usize;
    let nlon = 36usize;
    let mut lats = Vec::with_capacity(nlat * nlon);
    let mut lons = Vec::with_capacity(nlat * nlon);
    let mut truths = Vec::with_capacity(nlat * nlon);
    for i in 0..nlat {
        let lat = lat_lo + (lat_hi - lat_lo) * (i as f64 + 0.5) / nlat as f64;
        for j in 0..nlon {
            let lon = TAU * (j as f64) / nlon as f64;
            lats.push(lat);
            lons.push(lon);
            truths.push(truth(lat, lon));
        }
    }
    let pred = predict(formula, data, &lats, &lons);
    let n = pred.len();
    let mse: f64 = pred
        .iter()
        .zip(truths.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64;
    (mse.sqrt(), n)
}

#[test]
fn sphere_wahba_high_lat_band_rmse_close_to_equator() {
    init_parallelism();
    let data = make_training_data(800, 0.10, 2025);
    let formula = "y ~ sphere(lat, lon, radians=true, k=100)";

    let (rmse_eq, _) = rmse_in_lat_band(formula, &data, -0.6, 0.6);
    let (rmse_pole, _) = rmse_in_lat_band(formula, &data, 1.2, 1.4);
    let ratio = rmse_pole / rmse_eq.max(1e-12);
    eprintln!(
        "[sphere-top] wahba: rmse(equator)={:.4}  rmse(high-lat)={:.4}  ratio={:.2}",
        rmse_eq, rmse_pole, ratio
    );
    assert!(
        ratio < 1.4,
        "Sphere Wahba fit degrades sharply at high latitude: \
         RMSE(lat∈[1.2,1.4]) = {:.4} is {:.2}× RMSE(equator) = {:.4} \
         (budget ≤ 1.4×). Indicates the coefficient sum-to-zero \
         identifiability and/or Euclidean FPS center placement is creating \
         a polar artifact.",
        rmse_pole, ratio, rmse_eq,
    );
}

#[test]
fn sphere_harmonic_high_lat_band_rmse_close_to_equator() {
    init_parallelism();
    let data = make_training_data(800, 0.10, 2025);
    let formula = "y ~ sphere(lat, lon, radians=true, method=harmonic, max_degree=8)";

    let (rmse_eq, _) = rmse_in_lat_band(formula, &data, -0.6, 0.6);
    let (rmse_pole, _) = rmse_in_lat_band(formula, &data, 1.2, 1.4);
    let ratio = rmse_pole / rmse_eq.max(1e-12);
    eprintln!(
        "[sphere-top] harmonic: rmse(equator)={:.4}  rmse(high-lat)={:.4}  ratio={:.2}",
        rmse_eq, rmse_pole, ratio
    );
    assert!(
        ratio < 1.4,
        "Sphere harmonic fit degrades sharply at high latitude: \
         RMSE(lat∈[1.2,1.4]) = {:.4} is {:.2}× RMSE(equator) = {:.4} \
         (budget ≤ 1.4×). Same artifact as the Wahba path — suggests the \
         cause is upstream of the kernel choice (sparse polar data vs the \
         identifiability constraint).",
        rmse_pole, ratio, rmse_eq,
    );
}
