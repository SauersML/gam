use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn sphere_dataset() -> gam::data::EncodedDataset {
    let headers = ["lat", "lon", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let coords = [
        (-75.0, -170.0),
        (-55.0, -80.0),
        (-35.0, 15.0),
        (-15.0, 105.0),
        (0.0, -130.0),
        (10.0, -30.0),
        (25.0, 60.0),
        (40.0, 150.0),
        (55.0, -100.0),
        (65.0, 20.0),
        (75.0, 115.0),
        (5.0, 179.0),
    ];
    let records = coords
        .into_iter()
        .map(|(lat, lon): (f64, f64)| {
            let lat_rad = lat.to_radians();
            let lon_rad = lon.to_radians();
            let y = 1.0 + 0.6 * lat_rad.sin() + 0.4 * lat_rad.cos() * lon_rad.cos();
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode sphere dataset")
}

fn larger_sphere_dataset(n: usize) -> gam::data::EncodedDataset {
    let headers = ["lat", "lon", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let records = (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let z = 1.0 - 2.0 * t;
            let lat = z.asin().to_degrees();
            let lon = (((i as f64) / golden).fract() * 360.0) - 180.0;
            let lat_rad = lat.to_radians();
            let lon_rad = lon.to_radians();
            let y = 1.0 + 0.6 * lat_rad.sin() + 0.4 * lat_rad.cos() * lon_rad.cos();
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode larger sphere dataset")
}

#[test]
fn fit_from_formula_accepts_intrinsic_wahba_sphere_smooth() {
    init_parallelism();
    let data = sphere_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ sphere(lat, lon, k=5)", &data, &config)
        .expect("sphere formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert_eq!(fit.design.smooth.terms[0].coeff_range.len(), 4);
}

#[test]
fn fit_from_formula_accepts_harmonic_sphere_smooth() {
    init_parallelism();
    let data = sphere_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=2)",
        &data,
        &config,
    )
    .expect("harmonic sphere formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert_eq!(fit.design.smooth.terms[0].coeff_range.len(), 8);
}

#[test]
fn default_harmonic_sphere_predict_preserves_fitted_degree() {
    init_parallelism();
    let data = larger_sphere_dataset(144);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ sphere(lat, lon, method=harmonic)", &data, &config)
        .expect("default harmonic sphere formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    let fitted_coeff_count = fit.design.smooth.terms[0].coeff_range.len();
    assert!(
        fitted_coeff_count >= 8,
        "harmonic basis should have at least L=2 (8 cols), got {fitted_coeff_count}"
    );

    let probes = [(-70.0, -120.0), (-10.0, 0.0), (55.0, 135.0)];
    let mut m = Array2::<f64>::zeros((probes.len(), 3));
    for (i, &(lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = lat;
        m[[i, 1]] = lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("default harmonic sphere predict design should rebuild");
    assert_eq!(
        design.design.ncols(),
        fit.fit.beta.len(),
        "predict design must use the fitted harmonic degree"
    );
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred.iter().all(|v| v.is_finite()));
}
