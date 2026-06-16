//! Regression guard: when sphere training data is concentrated in one
//! hemisphere, predictions in the *opposite* hemisphere (where there
//! is no data) must stay bounded and within a sane multiple of the
//! training y range.
//!
//! Mathematical kernels on the sphere can produce extrapolation
//! artifacts in data-sparse regions; this catches the case where the
//! Wahba / harmonic basis amplifies far away from data. Currently
//! passing — kept as a regression guard so any future loss of
//! far-side stability fails loudly.

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

fn make_one_hemisphere(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(0.0, 80.0).expect("uniform"); // northern hemisphere only
    let u_lon = Uniform::new(-180.0, 180.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let lat: f64 = u_lat.sample(&mut rng);
            let lon: f64 = u_lon.sample(&mut rng);
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            let y = 0.5
                + 0.6 * lat_r.sin()
                + 0.3 * lat_r.cos() * (2.0 * lon_r).cos()
                + noise.sample(&mut rng);
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    lats: &[f64],
    lons: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("sphere fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
        m[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn sphere_predictions_in_data_empty_southern_hemisphere_stay_bounded() {
    init_parallelism();
    let data = make_one_hemisphere(200, 0.05, 19);

    // Probe deep in the southern hemisphere where there is no training data.
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    for lat in [-80.0, -60.0, -40.0, -20.0] {
        for lon in (-180..=180).step_by(60) {
            lats.push(lat);
            lons.push(lon as f64);
        }
    }

    for (label, formula) in &[
        ("wahba", "y ~ sphere(lat, lon, k=8)"),
        (
            "harmonic",
            "y ~ sphere(lat, lon, method=harmonic, max_degree=6)",
        ),
    ] {
        let pred = predict(formula, &data, &lats, &lons);
        let max_abs = pred.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()));
        let span = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - pred.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!("[sphere-empty-hem] {label:8}  max|pred|={max_abs:.3}  span={span:.3}");
        // Training y has range about [-0.5, 1.5]; far-side predictions can be
        // arbitrary in principle but should not exceed ~10 in magnitude.
        // A truly blown-up kernel would give |y| > 1000.
        assert!(
            max_abs < 10.0,
            "{label}: sphere prediction in empty southern hemisphere blew up to |y|={max_abs:.3} (budget 10.0)",
        );
    }
}
