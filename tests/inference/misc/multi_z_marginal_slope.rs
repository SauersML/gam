use gam::families::bms::{MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores};
use gam::probability::normal_cdf;
use ndarray::{Array1, array};

// The K=1 reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes (the diagonal quadratic form
// accumulates `coefficient * slope * slope`), so production evaluates
// `fl(fl(p*p) * fl(s*s))` while the scalar identity below folds the scale into
// the slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Same real number, up
// to an ulp apart, so `to_bits()` equality would pin the association order
// rather than the reduction. It is pinned to a few-ulp bound instead.
//
// `magnitude` is the size of the largest intermediates the reference sums, so
// a cancelling total is still held to the accuracy its inputs allow.
const SCALAR_REDUCTION_ULPS: f64 = 8.0;

fn ulp_of(value: f64) -> f64 {
    let magnitude = value.abs();
    if !magnitude.is_finite() || magnitude == 0.0 {
        return f64::from_bits(1);
    }
    let next = f64::from_bits(magnitude.to_bits() + 1);
    if next.is_finite() {
        next - magnitude
    } else {
        magnitude - f64::from_bits(magnitude.to_bits() - 1)
    }
}

fn assert_scalar_reduction(got: f64, expected: f64, magnitude: f64, context: &str) {
    let difference = (got - expected).abs();
    let tolerance = SCALAR_REDUCTION_ULPS * ulp_of(magnitude.abs().max(expected.abs()));
    assert!(
        difference <= tolerance,
        "{context}: got={got:.17e} expected={expected:.17e} |diff|={difference:.3e} exceeds \
         the {SCALAR_REDUCTION_ULPS}-ulp scalar-reduction tolerance {tolerance:.3e}"
    );
}

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
fn multi_z_k4_diagonal_covariance_preserves_marginal_identity() {
    let q = 0.63;
    let probit_scale = 0.9;
    let z = [0.4, -1.2, 0.7, 2.1];
    let slopes = [0.20, -0.15, 0.35, 0.05];
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0, 0.5, 2.0, 0.25]).unwrap();

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
    let covariance = MarginalSlopeCovariance::full(array![[1.4, 0.35], [0.35, 0.8]]).unwrap();

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
    let covariance = MarginalSlopeCovariance::low_rank(factor).unwrap();

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

    // Every nonzero off-diagonal is exact geometry and therefore remains Full.
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
    let collinear = marginal_slope_covariance_from_scores(low_rank_scores.view(), &weights)
        .expect("collinear covariance");
    assert_eq!(collinear.shape(), MarginalSlopeCovarianceShape::Full);
}
