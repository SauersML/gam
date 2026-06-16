use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn sphere_dataset() -> gam::data::EncodedDataset {
    let headers = vec!["lat".to_string(), "lon".to_string(), "y".to_string()];
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
        .map(|(lat, lon)| {
            let lat_rad = f64::to_radians(lat);
            let lon_rad = f64::to_radians(lon);
            let y = 1.0 + 0.6 * lat_rad.sin() + 0.4 * lat_rad.cos() * lon_rad.cos();
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode sphere dataset")
}

#[test]
fn fit_from_formula_accepts_intrinsic_sphere_smooth() {
    init_parallelism();
    let data = sphere_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ sphere(lat, lon, degree=2)", &data, &config)
        .expect("sphere formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert_eq!(fit.design.smooth.terms[0].coeff_range.len(), 8);
}
