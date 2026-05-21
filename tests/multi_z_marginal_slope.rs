use gam::bernoulli_marginal_slope::{
    MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores,
    marginal_slope_preserving_scale, marginal_slope_probit_eta,
};
use gam::probability::normal_cdf;
use ndarray::{Array1, array};

fn observed_slopes(slopes: &[f64], probit_scale: f64) -> Vec<f64> {
    slopes.iter().map(|&slope| probit_scale * slope).collect()
}

fn assert_preserves_signed_marginal(
    q: f64,
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) {
    let scale = marginal_slope_preserving_scale(slopes, covariance, probit_scale).expect("scale");
    let variance = covariance
        .quadratic_form(&observed_slopes(slopes, probit_scale))
        .expect("quadratic form");
    let marginal = normal_cdf(-q * scale / (1.0 + variance).sqrt());
    let target = normal_cdf(-q);
    assert!(
        (marginal - target).abs() < 2e-15,
        "marginal={marginal:.17e} target={target:.17e}"
    );
}

#[test]
fn multi_z_k1_diagonal_matches_scalar_rigid_eta_bitwise() {
    let q = -0.37;
    let slope = 0.42;
    let z = 1.25;
    let probit_scale = 0.8;
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0]);

    let eta_multi =
        marginal_slope_probit_eta(q, &[z], &[slope], &covariance, probit_scale).expect("eta");
    let observed_slope = probit_scale * slope;
    let eta_scalar = q * (1.0 + observed_slope * observed_slope).sqrt() + observed_slope * z;

    assert_eq!(eta_multi.to_bits(), eta_scalar.to_bits());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Diagonal);
    assert_preserves_signed_marginal(q, &[slope], &covariance, probit_scale);
}

#[test]
fn multi_z_k4_diagonal_covariance_preserves_marginal_identity() {
    let q = 0.63;
    let probit_scale = 0.9;
    let z = [0.4, -1.2, 0.7, 2.1];
    let slopes = [0.20, -0.15, 0.35, 0.05];
    let covariance = MarginalSlopeCovariance::Diagonal(array![1.0, 0.5, 2.0, 0.25]);

    let eta = marginal_slope_probit_eta(q, &z, &slopes, &covariance, probit_scale).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Diagonal);
    assert_preserves_signed_marginal(q, &slopes, &covariance, probit_scale);
}

#[test]
fn multi_z_k2_full_covariance_preserves_marginal_identity() {
    let q = -0.21;
    let probit_scale = 1.15;
    let z = [1.1, -0.3];
    let slopes = [0.45, -0.25];
    let covariance = MarginalSlopeCovariance::Full(array![[1.4, 0.35], [0.35, 0.8]]);

    let eta = marginal_slope_probit_eta(q, &z, &slopes, &covariance, probit_scale).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Full);
    assert_preserves_signed_marginal(q, &slopes, &covariance, probit_scale);
}

#[test]
fn multi_z_k4_low_rank_covariance_preserves_marginal_identity() {
    let q = 0.18;
    let probit_scale = 0.7;
    let z = [-0.2, 0.9, 1.4, -1.1];
    let slopes = [0.30, -0.40, 0.10, 0.25];
    let factor = array![[1.0, 0.0], [0.5, 0.2], [-0.3, 0.7], [0.1, -0.4]];
    let covariance = MarginalSlopeCovariance::LowRank(factor);

    let eta = marginal_slope_probit_eta(q, &z, &slopes, &covariance, probit_scale).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::LowRank);
    assert_preserves_signed_marginal(q, &slopes, &covariance, probit_scale);
}

#[test]
fn multi_z_covariance_shape_auto_derives_from_score_geometry() {
    let weights = Array1::ones(6);

    let diagonal_scores = array![
        [-1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -3.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
    ];
    let diagonal = marginal_slope_covariance_from_scores(diagonal_scores.view(), &weights)
        .expect("diagonal covariance");
    assert_eq!(diagonal.shape(), MarginalSlopeCovarianceShape::Diagonal);

    // Full case: construct two columns whose sample off-diagonal is far above
    // the 4σ statistical floor used by the classifier.  Mixing the two
    // harmonics cos(2π·3t) and cos(2π·5t) gives orthogonal basis functions
    // with var ≈ 1/2 each; col2 = 0.7·col1 + 0.4·col1_other_harmonic injects
    // a cov[0,1] ≈ 0.35 ≈ 7·SE_offdiag at N=64.  Both eigenvalues stay well
    // above rank_tol, so the classifier falls through to Full (not LowRank).
    const N_FULL: usize = 64;
    let mut full_scores = ndarray::Array2::<f64>::zeros((N_FULL, 2));
    for i in 0..N_FULL {
        let t = (i as f64) / (N_FULL as f64);
        let h3 = (2.0 * std::f64::consts::PI * 3.0 * t).cos();
        let h5 = (2.0 * std::f64::consts::PI * 5.0 * t).cos();
        full_scores[[i, 0]] = h3;
        full_scores[[i, 1]] = 0.7 * h3 + 0.4 * h5;
    }
    let weights_full = Array1::ones(N_FULL);
    let full =
        marginal_slope_covariance_from_scores(full_scores.view(), &weights_full).expect("full cov");
    assert_eq!(full.shape(), MarginalSlopeCovarianceShape::Full);

    let low_rank_scores = array![
        [-2.0, -4.0, 1.0, 3.0],
        [-1.0, -2.0, 0.5, 1.5],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, -0.5, -1.5],
        [2.0, 4.0, -1.0, -3.0],
        [3.0, 6.0, -1.5, -4.5],
    ];
    let low_rank = marginal_slope_covariance_from_scores(low_rank_scores.view(), &weights)
        .expect("low-rank covariance");
    assert_eq!(low_rank.shape(), MarginalSlopeCovarianceShape::LowRank);
}
