//! Both Wahba sphere kernels — the canonical Sobolev `H^m(S²)` form and
//! the mgcv-compatible Wahba 1981 pseudo-spline — must fit a smooth
//! low-degree truth cleanly for every supported penalty order m ∈ {1..4}.
//!
//! Test surface:
//!     y = 0.5 + 0.6·sin(lat) + 0.3·cos(lat)·cos(lon) + noise (σ=0.05)
//! 400 points, k=30 centres.
//!
//! Hard-fail target: rmse ≤ 0.10 for *every* (kernel, m) combination on a
//! held-out 15×15 lat/lon grid. The truth peak-to-peak is ~1.4, so a
//! good fit at noise level σ=0.05 hits rmse 0.01–0.02. The 0.10 budget
//! tolerates the larger boundary bias the pseudo-spline picks up at
//! high m without admitting the historical m=4 collapse (rmse = 0.43,
//! predictions = mean).

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

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn truth(lat: f64, lon: f64) -> f64 {
    0.5 + 0.6 * lat.to_radians().sin() + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
}

fn rmse_against_truth(formula: &str) -> Result<f64, String> {
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard fit".into());
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -175.0 + 350.0 * (j as f64) / 14.0;
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
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let sumsq: f64 = pred
        .iter()
        .zip(pts.iter())
        .map(|(p, (lat, lon))| (p - truth(*lat, *lon)).powi(2))
        .sum();
    Ok((sumsq / n as f64).sqrt())
}

#[test]
fn sphere_sobolev_kernel_fits_smooth_truth_for_all_m() {
    init_parallelism();
    let mut failures = Vec::new();
    for m in [1usize, 2, 3, 4] {
        let formula = format!("y ~ sphere(lat, lon, k=30, m={m}, kernel=sobolev)");
        match rmse_against_truth(&formula) {
            Ok(r) => {
                eprintln!("[sobolev] m={m}: rmse={r:.4}");
                if r > 0.10 {
                    failures.push(format!("m={m}: rmse={r:.4} > 0.10"));
                }
            }
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "Sobolev kernel failures:\n  - {}",
        failures.join("\n  - "),
    );
}

#[test]
fn sphere_pseudo_kernel_fits_smooth_truth_for_all_m() {
    init_parallelism();
    let mut failures = Vec::new();
    for m in [1usize, 2, 3, 4] {
        let formula = format!("y ~ sphere(lat, lon, k=30, m={m}, kernel=pseudo)");
        match rmse_against_truth(&formula) {
            Ok(r) => {
                eprintln!("[pseudo] m={m}: rmse={r:.4}");
                if r > 0.10 {
                    failures.push(format!("m={m}: rmse={r:.4} > 0.10"));
                }
            }
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "Pseudo-spline kernel failures:\n  - {}\n\nIf m=4 fails this is the historical mgcv \
         pseudo-spline collapse — the cure is the REML scale-invariance fix in the solver.",
        failures.join("\n  - "),
    );
}

#[test]
fn sphere_method_aliases_route_to_correct_kernel() {
    init_parallelism();
    // method=wahba_sobolev / wahba_pseudo / sobolev / pseudo / mgcv / sos
    // should all parse without erroring at fit time.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = make_dataset(200);
    for method in [
        "wahba", // default → sobolev
        "wahba_sobolev",
        "wahba_pseudo",
        "sobolev",
        "pseudo",
        "mgcv",
        "sos",
    ] {
        let formula = format!("y ~ sphere(lat, lon, k=10, m=2, method={method})");
        fit_from_formula(&formula, &data, &cfg)
            .unwrap_or_else(|e| panic!("method=`{method}` failed: {e}"));
    }
}
