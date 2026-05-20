use gam::bernoulli_marginal_slope::MarginalSlopeCovariance;
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::{
    survival_marginal_slope_vector_eta, survival_marginal_slope_vector_scale,
};
use ndarray::array;

fn assert_survival_identity(
    q: f64,
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) {
    let c =
        survival_marginal_slope_vector_scale(slopes, covariance, probit_scale).expect("scale");
    let observed_slopes = slopes
        .iter()
        .map(|&slope| probit_scale * slope)
        .collect::<Vec<_>>();
    let variance = covariance
        .quadratic_form(&observed_slopes)
        .expect("quadratic form");
    let marginal = normal_cdf(-q * c / (1.0 + variance).sqrt());
    let target = normal_cdf(-q);
    assert!((marginal - target).abs() < 2e-15);
}

#[test]
fn survival_multi_z_k1_scalar_fallback_is_bitwise() {
    let q = 0.41;
    let slope = -0.32;
    let z = 1.7;
    let probit_scale = 0.75;
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0]);

    let eta =
        survival_marginal_slope_vector_eta(q, &[z], &[slope], &covariance, probit_scale).unwrap();
    let observed = probit_scale * slope;
    let scalar = q * (1.0 + observed * observed).sqrt() + observed * z;
    assert_eq!(eta.to_bits(), scalar.to_bits());
    assert_survival_identity(q, &[slope], &covariance, probit_scale);
}

#[test]
fn survival_multi_z_k2_full_covariance_preserves_identity() {
    let covariance = MarginalSlopeCovariance::Full(array![[1.2, -0.25], [-0.25, 0.9]]);
    let z = [0.4, -1.1];
    let slopes = [0.35, -0.18];
    let eta = survival_marginal_slope_vector_eta(-0.27, &z, &slopes, &covariance, 1.1).unwrap();
    assert!(eta.is_finite());
    assert_survival_identity(-0.27, &slopes, &covariance, 1.1);
}

#[test]
fn survival_multi_z_k4_low_rank_covariance_preserves_identity() {
    let factor = array![[1.0, 0.0], [0.4, -0.2], [-0.3, 0.5], [0.2, 0.7]];
    let covariance = MarginalSlopeCovariance::LowRank(factor);
    let z = [1.0, -0.2, 0.8, -1.3];
    let slopes = [0.2, -0.4, 0.15, 0.05];
    let eta = survival_marginal_slope_vector_eta(0.19, &z, &slopes, &covariance, 0.85).unwrap();
    assert!(eta.is_finite());
    assert_survival_identity(0.19, &slopes, &covariance, 0.85);
}
