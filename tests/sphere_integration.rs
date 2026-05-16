use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

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
