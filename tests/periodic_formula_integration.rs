use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn periodic_dataset() -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..32)
        .map(|i| {
            let theta = std::f64::consts::TAU * i as f64 / 32.0;
            let h = -1.0 + 2.0 * (i % 8) as f64 / 7.0;
            let y = 1.0 + 0.55 * theta.cos() - 0.25 * (2.0 * theta).sin() + 0.3 * h;
            StringRecord::from(vec![theta.to_string(), h.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode periodic dataset")
}

fn assert_standard_fit_has_finite_coefficients(
    formula: &str,
    expected_smooth_terms: usize,
) -> usize {
    init_parallelism();
    let data = periodic_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &config).expect("periodic formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), expected_smooth_terms);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(
        fit.design
            .smooth
            .terms
            .iter()
            .all(|term| !term.coeff_range.is_empty())
    );
    fit.fit.beta.len()
}

#[test]
fn fit_from_formula_accepts_cyclic_1d_bspline_smooth() {
    let ncoef = assert_standard_fit_has_finite_coefficients(
        "y ~ cyclic(theta, k=8, period_start=0, period_end=6.283185307179586)",
        1,
    );
    assert!(ncoef > 0);
}

#[test]
fn fit_from_formula_accepts_tensor_cylinder_periodic_margin() {
    let ncoef = assert_standard_fit_has_finite_coefficients(
        "y ~ s(theta, h, periodic=[0], period=[2*pi, None], k=5)",
        1,
    );
    assert!(ncoef > 0);
}

#[test]
fn fit_from_formula_accepts_tensor_period_origin_aliases() {
    let ncoef = assert_standard_fit_has_finite_coefficients(
        "y ~ te(theta, h, bc=['periodic', 'natural'], periods=[2*pi, None], origins=[0, None], k=5)",
        1,
    );
    assert!(ncoef > 0);
}

#[test]
fn fit_from_formula_accepts_cyclic_duchon() {
    let ncoef = assert_standard_fit_has_finite_coefficients("y ~ duchon(theta, periodic=true, k=8)", 1);
    assert!(ncoef > 0);
}
