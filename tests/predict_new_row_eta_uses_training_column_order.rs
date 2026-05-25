use gam::inference::predict::predict_gam;
use gam::types::LikelihoodSpec;
use ndarray::{arr1, arr2};

#[test]
fn predict_new_row_eta_uses_training_column_order() {
    let x_new = arr2(&[[10.0, -3.0, 0.25]]);
    let beta = arr1(&[0.1, 2.0, -4.0]);
    let offset = arr1(&[0.7]);

    let out = predict_gam(
        x_new.view(),
        beta.view(),
        offset.view(),
        LikelihoodSpec::gaussian_identity(),
    )
    .expect("standard prediction should succeed");

    let expected_eta = 10.0 * 0.1 + (-3.0) * 2.0 + 0.25 * (-4.0) + 0.7;
    assert!(
        (out.eta[0] - expected_eta).abs() < 1e-12,
        "New-row eta should be computed as X_new * beta + offset using fit-time column order."
    );
}
