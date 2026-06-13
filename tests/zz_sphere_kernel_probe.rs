//! TEMPORARY measurement scaffold for #1079: compare gam sphere-smooth kernels
//! against the analytic geodesic-radial truth on the EXACT data of
//! `quality_vs_scipy_sphere_geodesic_consistency`. Not a quality gate — printed
//! diagnostics only. Deleted once the kernel choice is settled.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}
fn geodesic_deg(lat0: f64, lon0: f64, lat1: f64, lon1: f64) -> f64 {
    let a = latlon_to_xyz(lat0, lon0);
    let b = latlon_to_xyz(lat1, lon1);
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    dot.acos()
}
fn make_dataset(lats: &[f64], lons: &[f64], ys: &[f64]) -> gam::data::EncodedDataset {
    use csv::StringRecord;
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::new();
    for i in 0..lats.len() {
        rows.push(StringRecord::from(vec![
            lats[i].to_string(),
            lons[i].to_string(),
            ys[i].to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

#[test]
fn probe_sphere_kernels() {
    gam::init_parallelism();
    let n = 50usize;
    let bandwidth = 0.8_f64;
    let mut rng = StdRng::seed_from_u64(20260529);
    let u_z = Uniform::new_inclusive(-1.0, 1.0).unwrap();
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64).unwrap();
    let noise = Normal::new(0.0, 0.01).unwrap();
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n {
        let z: f64 = u_z.sample(&mut rng);
        let lon_deg: f64 = u_lon.sample(&mut rng);
        let lat_deg = z.asin().to_degrees();
        let d_pole = geodesic_deg(90.0, 0.0, lat_deg, lon_deg);
        let f = (-d_pole / bandwidth).exp();
        lats.push(lat_deg);
        lons.push(lon_deg);
        ys.push(f + noise.sample(&mut rng));
    }
    let mut eval_lats = Vec::new();
    let mut eval_lons = Vec::new();
    for i in 0..10 {
        let lat = -75.0 + 150.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -170.0 + 340.0 * (j as f64) / 9.0;
            eval_lats.push(lat);
            eval_lons.push(lon);
        }
    }
    let m = eval_lats.len();
    let mut f_true = Vec::with_capacity(m);
    for i in 0..m {
        let d = geodesic_deg(90.0, 0.0, eval_lats[i], eval_lons[i]);
        f_true.push((-d / bandwidth).exp());
    }
    let data = make_dataset(&lats, &lons, &ys);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut best_rmse = f64::INFINITY;
    let formulas = [
        "y ~ sphere(lat, lon, k=20)",
        "y ~ sphere(lat, lon, k=20, kernel=pseudo)",
        "y ~ sphere(lat, lon, k=20, kernel=harmonic, degree=4)",
        "y ~ sphere(lat, lon, k=20, kernel=harmonic, degree=6)",
        "y ~ sphere(lat, lon, k=20, kernel=harmonic, degree=8)",
        "y ~ sphere(lat, lon, k=20, m=3)",
        "y ~ sphere(lat, lon, k=20, kernel=pseudo, m=3)",
    ];
    for f in formulas {
        let result = match fit_from_formula(f, &data, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[probe] {f:55} -> FIT ERROR: {e}");
                continue;
            }
        };
        let FitResult::Standard(fit) = result else {
            eprintln!("[probe] {f} -> non-standard");
            continue;
        };
        let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
        let mut grid = Array2::<f64>::zeros((m, 3));
        for i in 0..m {
            grid[[i, 0]] = eval_lats[i];
            grid[[i, 1]] = eval_lons[i];
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec).unwrap();
        let pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        let r = rmse(&pred, &f_true);
        best_rmse = best_rmse.min(r);
        eprintln!("[probe] {f:55} edf={edf:6.3} rmse={r:.5}");
    }
    eprintln!("[probe] mgcv sos k=20 reference rmse ~= 0.03379 (target to beat)");
    eprintln!("[probe] best sphere-kernel rmse = {best_rmse:.5}");
    // At least one sphere kernel must recover the smooth geodesic-radial truth
    // to a generous ceiling (the mgcv sos reference is ~0.034; this loose bound
    // is a sanity floor, not the #1079 quality bar — it only guards that the
    // diagnostic actually fit something rather than emitting NaNs).
    assert!(
        best_rmse.is_finite() && best_rmse < 0.20,
        "no sphere kernel recovered the geodesic-radial truth (best rmse={best_rmse:.5})"
    );
}
