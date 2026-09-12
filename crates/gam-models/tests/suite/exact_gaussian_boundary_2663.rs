//! #2663: exact Gaussian smooths live on a mixed smoothing-parameter boundary.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::matrix::LinearOperator;
use gam_models::fit_orchestration::{
    FitConfig, FitResult, StandardFitResult, fit_from_formula, fit_model, materialize,
};

fn gaussian_config() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

fn exact_line_dataset() -> EncodedDataset {
    let rows = (0..80)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / 79.0;
            let y = 0.5 + 1.25 * x;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode exact line")
}

fn exact_plane_dataset() -> EncodedDataset {
    let mut rows = Vec::with_capacity(144);
    for ix in 0..12 {
        let x = -1.0 + 2.0 * ix as f64 / 11.0;
        for iz in 0..12 {
            let z = -0.8 + 1.6 * iz as f64 / 11.0;
            let y = 0.5 + 1.25 * x - 0.75 * z;
            rows.push(StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(
        vec!["x".to_string(), "z".to_string(), "y".to_string()],
        rows,
    )
    .expect("encode exact plane")
}

fn standard(result: FitResult) -> StandardFitResult {
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard fit"),
    }
}

fn assert_exact_boundary(fit: &StandardFitResult, data: &EncodedDataset, expected_edf: f64) {
    assert_eq!(fit.fit.reml_score(), None);
    assert!(fit.fit.at_zero_dispersion_boundary());
    assert_eq!(fit.fit.standard_deviation, 0.0);
    assert!(
        fit.fit
            .log_lambdas
            .iter()
            .any(|&rho| rho == gam_problem::LOG_STRENGTH_MIN),
        "an exact nonconstant response needs a zero-face penalty: {:?}",
        fit.fit.log_lambdas,
    );
    assert!(
        fit.fit
            .log_lambdas
            .iter()
            .any(|&rho| rho != gam_problem::LOG_STRENGTH_MIN),
        "unsupported roughness needs an infinite-face penalty: {:?}",
        fit.fit.log_lambdas,
    );
    let edf = fit.fit.edf_total().expect("boundary fit reports EDF");
    assert!(
        (edf - expected_edf).abs() < 1.0e-5,
        "expected affine EDF {expected_edf}, got {edf}",
    );
    let fitted = fit.design.design.apply(&fit.fit.beta) + &fit.design.affine_offset;
    let response = data.values.column(data.column_map()["y"]);
    for (&actual, &expected) in fitted.iter().zip(response) {
        assert!((actual - expected).abs() < 1.0e-9);
    }
}

#[test]
fn exact_smooth_and_tensor_plane_use_their_mixed_penalty_faces_2663() {
    let config = gaussian_config();
    let line = exact_line_dataset();
    let formula_line =
        standard(fit_from_formula("y ~ s(x)", &line, &config).expect("formula line fit"));
    assert_exact_boundary(&formula_line, &line, 2.0);
    let direct_line = standard(
        fit_model(
            materialize("y ~ s(x)", &line, &config)
                .expect("materialize")
                .request,
        )
        .expect("direct line fit"),
    );
    assert_exact_boundary(&direct_line, &line, 2.0);

    let plane = exact_plane_dataset();
    let plane_fit = standard(fit_from_formula("y ~ te(x, z)", &plane, &config).expect("plane fit"));
    // The two tensor curvature penalties jointly leave the bilinear null model
    // {1, x, z, xz}, and each functional-ANOVA block of that null carries its own
    // ridge (6e65fa523). The plane has no x·z term, so REML sends the xz ridge to
    // its infinite face and keeps the x and z trends: EDF is the model dimension
    // {1, x, z}, not the nonzero beta count.
    assert_exact_boundary(&plane_fit, &plane, 3.0);
}
