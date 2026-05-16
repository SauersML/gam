use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn matern_1d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..48)
        .map(|i| {
            let x = i as f64 / 47.0;
            let y = 0.3 + (std::f64::consts::TAU * 2.0 * x).sin() + 0.2 * x;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 1D Matern dataset")
}

fn matern_2d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..8)
        .flat_map(|i| {
            (0..6).map(move |j| {
                let x = i as f64 / 7.0;
                let z = j as f64 / 5.0;
                let y = 0.4 + x - 0.5 * z + (std::f64::consts::TAU * x).sin() * z;
                StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
            })
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 2D Matern dataset")
}

#[test]
fn fit_from_formula_accepts_1d_matern_nu_half_decimal_alias() {
    init_parallelism();
    let data = matern_1d_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=.5, centers=12)", &data, &config).expect("1D Matern");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(fit.fit.edf_total().is_some_and(f64::is_finite));
}

#[test]
fn fit_from_formula_rejects_2d_matern_nu_half_decimal_alias_before_pirls() {
    init_parallelism();
    let data = matern_2d_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula("y ~ matern(x, z, nu=.50)", &data, &config) {
        Ok(_) => panic!("2D Matern nu=1/2 should be rejected before fit"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("nu=1/2 is not supported for d>=2"),
        "unexpected error: {err}"
    );
}
