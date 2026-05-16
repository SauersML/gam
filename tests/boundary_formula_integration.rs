use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn boundary_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..28)
        .map(|i| {
            let x = i as f64 / 27.0;
            let y = 0.5 + 2.0 * x * x * (1.0 - x) * (1.0 - x);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode boundary dataset")
}

#[test]
fn fit_from_formula_accepts_bspline_endpoint_boundary_conditions() {
    init_parallelism();
    let data = boundary_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=0, bc_right=clamped, k=8)",
        &data,
        &config,
    )
    .expect("boundary-conditioned formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(!fit.design.smooth.terms[0].coeff_range.is_empty());
}

#[test]
fn boundary_conditioned_saved_spec_rebuilds_for_prediction() {
    init_parallelism();
    let data = boundary_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bc_left=anchored, anchor_left=0, bc_right=clamped, k=8)",
        &data,
        &config,
    )
    .expect("boundary-conditioned formula fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };

    let mut new_data = Array2::<f64>::zeros((31, 2));
    for i in 0..31 {
        new_data[[i, 0]] = i as f64 / 30.0;
        new_data[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("frozen boundary-conditioned spec should rebuild for prediction");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred.iter().all(|v| v.is_finite()));
}
