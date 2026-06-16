use gam::inference::diagnostics::{diagnostics_from_predictions, should_emit_h_min_eig_diag};
use gam::inference::dispersion_cov::{PhiScaledCovariance, UnscaledPrecision};
use gam::inference::predict::linalg::{PredictionCovarianceBackend, rowwise_local_covariances};
use gam::inference::probability::{
    normal_cdf, normal_logcdf, normal_pdf, signed_probit_logcdf_and_mills_ratio,
};
use ndarray::{Array2, array};

#[test]
fn probability_normal_cdf_boundary_and_accuracy_contract() {
    assert_eq!(
        normal_cdf(0.0),
        0.5,
        "normal_cdf at z=0 must be exactly 0.5"
    );
    assert_eq!(normal_cdf(f64::INFINITY), 1.0, "normal_cdf(+inf) must be 1");
    assert_eq!(
        normal_cdf(f64::NEG_INFINITY),
        0.0,
        "normal_cdf(-inf) must be 0"
    );
    let z = 0.37;
    let expected = 0.644308823593; // high-precision reference
    assert!(
        (normal_cdf(z) - expected).abs() < 1e-12,
        "normal_cdf must match standard normal CDF within 1e-12"
    );
}

#[test]
fn probability_normal_pdf_matches_cdf_derivative() {
    let z = -0.91;
    let h = 1e-7;
    let fd = (normal_cdf(z + h) - normal_cdf(z - h)) / (2.0 * h);
    assert!(
        (fd - normal_pdf(z)).abs() < 1e-8,
        "normal_pdf must be the derivative of normal_cdf"
    );
}

#[test]
fn probability_normal_logcdf_matches_log_cdf_and_left_tail_asymptotic() {
    let z = 1.1;
    assert!(
        (normal_logcdf(z) - normal_cdf(z).ln()).abs() < 1e-12,
        "normal_logcdf must equal log(normal_cdf) away from asymptotic branch"
    );
    let x = -14.0;
    let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(x);
    let asymptotic = -0.5 * x * x - x.abs().ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!(
        (logcdf - asymptotic).abs() < 0.02,
        "left-tail logcdf must follow documented asymptotic"
    );
    assert!(
        (lambda - x.abs()).abs() < 0.1,
        "Mills ratio should be asymptotically |x| in the far left tail"
    );
}

#[test]
fn dispersion_newtypes_round_trip_without_scale_drift() {
    let m = array![[2.0, 0.3], [0.3, 1.5]];
    let wrapped_cov = PhiScaledCovariance::wrap(m.clone());
    let wrapped_prec = UnscaledPrecision::wrap(m.clone());
    let cov_back: Array2<f64> = wrapped_cov.into();
    let prec_back: Array2<f64> = wrapped_prec.into();
    assert_eq!(
        cov_back, m,
        "PhiScaledCovariance must preserve wrapped matrix exactly"
    );
    assert_eq!(
        prec_back, m,
        "UnscaledPrecision must preserve wrapped matrix exactly"
    );
}

#[test]
fn prediction_linalg_rowwise_local_covariances_are_symmetric_spd_per_row() {
    let covariance = array![[2.0, 0.2], [0.2, 1.0]];
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    let g0 = array![[1.0, 0.5], [0.7, 1.1]];
    let g1 = array![[0.3, 1.2], [1.4, 0.2]];
    let out = rowwise_local_covariances(&backend, 2, 2, |rows| {
        Ok(vec![
            g0.slice(ndarray::s![rows.clone(), ..]).to_owned(),
            g1.slice(ndarray::s![rows, ..]).to_owned(),
        ])
    })
    .expect("local covariances should compute");
    for i in 0..2 {
        let a = out[0][0][i];
        let b = out[0][1][i];
        let c = out[1][1][i];
        assert!(
            (b - out[1][0][i]).abs() < 1e-12,
            "local covariance must be symmetric"
        );
        assert!(
            a > 0.0 && c > 0.0 && (a * c - b * b) > 0.0,
            "each per-row local covariance matrix must be SPD"
        );
    }
}

#[test]
fn diagnostics_invariants_for_residual_metrics_and_rate_limiter() {
    let d = diagnostics_from_predictions(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0])
        .expect("diagnostics should compute");
    assert_eq!(d.n_obs, 3, "diagnostics n_obs must equal input length");
    assert!(
        d.mae >= 0.0 && d.rmse >= 0.0,
        "MAE and RMSE must be non-negative"
    );
    assert_eq!(
        d.residuals.len(),
        3,
        "residual vector length must match n_obs"
    );
    assert!(
        should_emit_h_min_eig_diag(f64::NAN),
        "non-finite minimum eigenvalue must always emit diagnostics"
    );
}
