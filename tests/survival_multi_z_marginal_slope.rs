use gam::bernoulli_marginal_slope::{MarginalSlopeCovariance, MarginalSlopeCovarianceShape};
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::{
    survival_marginal_slope_vector_eta, survival_marginal_slope_vector_neglog,
    survival_marginal_slope_vector_scale,
};
use ndarray::array;

fn assert_marginal_preservation(
    q: f64,
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) {
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale).expect("scale");
    let observed: Vec<f64> = slopes.iter().map(|&r| probit_scale * r).collect();
    let variance = covariance
        .quadratic_form(&observed)
        .expect("quadratic form");
    let lhs = normal_cdf(-q * c / (1.0 + variance).sqrt());
    let rhs = normal_cdf(-q);
    assert!((lhs - rhs).abs() <= 2e-15, "lhs={lhs:.17e} rhs={rhs:.17e}");
}

#[test]
fn survival_multi_z_k1_diagonal_matches_scalar_eta_bitwise() {
    let q = 0.41;
    let z = [1.3];
    let slope = [0.27];
    let probit_scale = 0.8;
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0]);
    let eta =
        survival_marginal_slope_vector_eta(q, &z, &slope, &covariance, probit_scale).expect("eta");
    let observed = probit_scale * slope[0];
    let scalar = q * (1.0 + observed * observed).sqrt() + observed * z[0];
    assert_eq!(eta.to_bits(), scalar.to_bits());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Diagonal);
    assert_marginal_preservation(q, &slope, &covariance, probit_scale);
}

#[test]
fn survival_multi_z_k2_full_covariance_preserves_identity() {
    let q = -0.22;
    let z = [0.6, -1.1];
    let slopes = [0.35, -0.2];
    let covariance = MarginalSlopeCovariance::Full(array![[1.3, 0.4], [0.4, 0.7]]);
    let eta = survival_marginal_slope_vector_eta(q, &z, &slopes, &covariance, 1.1).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Full);
    assert_marginal_preservation(q, &slopes, &covariance, 1.1);
}

#[test]
fn survival_multi_z_k4_low_rank_covariance_preserves_identity() {
    let q = 0.19;
    let z = [-0.4, 0.8, 1.2, -1.5];
    let slopes = [0.25, -0.32, 0.08, 0.21];
    let covariance =
        MarginalSlopeCovariance::LowRank(array![[1.0, 0.0], [0.2, 0.4], [-0.3, 0.5], [0.7, -0.1]]);
    let eta = survival_marginal_slope_vector_eta(q, &z, &slopes, &covariance, 0.9).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::LowRank);
    assert_marginal_preservation(q, &slopes, &covariance, 0.9);
}

#[test]
fn survival_multi_z_eta_rejects_score_slope_dimension_mismatch() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0]);
    let err = survival_marginal_slope_vector_eta(0.2, &[0.4, -0.8], &[0.3], &covariance, 1.0)
        .expect_err("dimension mismatch must fail");
    assert!(err.contains("dimension mismatch"));
}

#[test]
fn survival_multi_z_k1_neglog_matches_scalar_identity_fixture() {
    let q0 = -0.15;
    let q1 = 0.55;
    let qd1 = 0.8;
    let slope = [0.31];
    let z = [0.45];
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0]);
    let probit_scale = 0.75;
    let observed = probit_scale * slope[0];
    let c = (1.0 + observed * observed).sqrt();
    let eta0 = q0 * c + observed * z[0];
    let eta1 = q1 * c + observed * z[0];
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
    let expected = 1.2
        * ((1.0 - 1.0) * -normal_cdf(-eta1).ln() + normal_cdf(-eta0).ln()
            - log_phi_eta1
            - (qd1 * c).ln()
            - log_phi_q1
            - qd1.ln());
    let actual = survival_marginal_slope_vector_neglog(
        q0,
        q1,
        qd1,
        &slope,
        &z,
        &covariance,
        1.2,
        1.0,
        1e-6,
        probit_scale,
    )
    .expect("vector neglog");
    assert!(
        (actual - expected).abs() <= 1e-14,
        "actual={actual:.17e} expected={expected:.17e}"
    );
}

#[test]
fn survival_multi_z_neglog_rejects_derivative_guard_violation() {
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 1.0]);
    let err = survival_marginal_slope_vector_neglog(
        0.0,
        0.2,
        1e-7,
        &[0.2, -0.1],
        &[0.4, 0.5],
        &covariance,
        1.0,
        1.0,
        1e-6,
        1.0,
    )
    .expect_err("derivative guard violation must fail");
    assert!(err.contains("monotonicity violated"));
}
