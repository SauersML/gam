use gam::families::bms::{MarginalSlopeCovariance, MarginalSlopeCovarianceShape};
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use gam::probability::normal_cdf;
use ndarray::array;

// The K=1 reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes (the diagonal quadratic form
// accumulates `coefficient * slope * slope`), so production evaluates
// `fl(fl(p*p) * fl(s*s))` while the scalar identity below folds the scale into
// the slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Same real number, up
// to an ulp apart, so the reduction is pinned to a few-ulp bound rather than
// to `to_bits()` equality, which would pin the association order instead.
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
fn survival_multi_z_k1_scale_reduces_to_original_scalar_unit_variance() {
    let slope = [0.31];
    let probit_scale: f64 = 0.75;
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0]).unwrap();
    let observed: f64 = probit_scale * slope[0];
    let expected = (1.0 + observed * observed).sqrt();
    let actual =
        survival_marginal_slope_vector_scale(&slope, &covariance, probit_scale).expect("scale");
    assert_scalar_reduction(
        actual,
        expected,
        expected,
        "K=1 unit-variance scale must reduce to sqrt(1 + (probit_scale*slope)^2)",
    );
}

#[test]
fn survival_multi_z_k2_full_covariance_preserves_identity() {
    let q = -0.22;
    let z = [0.6, -1.1];
    let slopes = [0.35, -0.2];
    let covariance = MarginalSlopeCovariance::full(array![[1.3, 0.4], [0.4, 0.7]]).unwrap();
    let eta = survival_marginal_slope_vector_eta(q, &z, &slopes, &covariance, 1.1).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::Full);
    assert_marginal_preservation(q, &slopes, &covariance, 1.1);
}

#[test]
fn survival_multi_z_shared_slope_neglog_uses_row_sum_and_covariance_quadratic() {
    let q0 = 0.15;
    let q1 = 0.55;
    let qd1 = 0.9;
    let shared_slope = -0.22;
    let z = [0.6, -1.1];
    let covariance = MarginalSlopeCovariance::full(array![[1.3, 0.4], [0.4, 0.7]]).unwrap();
    let probit_scale = 0.85;
    let weight = 1.3;
    let event = 1.0;

    let observed = [probit_scale * shared_slope, probit_scale * shared_slope];
    let c = (1.0 + covariance.quadratic_form(&observed).expect("r Sigma r")).sqrt();
    let linear = observed[0] * z.iter().sum::<f64>();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let expected = weight
        * (normal_cdf(-eta0).ln()
            - (1.0 - event) * normal_cdf(-eta1).ln()
            - event * log_phi_eta1
            - event * (qd1 * c).ln());

    let actual = survival_marginal_slope_vector_neglog(
        0,
        q0,
        q1,
        qd1,
        &[shared_slope, shared_slope],
        &z,
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
        weight,
        event,
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
fn survival_multi_z_k4_low_rank_covariance_preserves_identity() {
    let q = 0.19;
    let z = [-0.4, 0.8, 1.2, -1.5];
    let slopes = [0.25, -0.32, 0.08, 0.21];
    let covariance =
        MarginalSlopeCovariance::low_rank(array![[1.0, 0.0], [0.2, 0.4], [-0.3, 0.5], [0.7, -0.1]])
            .unwrap();
    let eta = survival_marginal_slope_vector_eta(q, &z, &slopes, &covariance, 0.9).expect("eta");
    assert!(eta.is_finite());
    assert_eq!(covariance.shape(), MarginalSlopeCovarianceShape::LowRank);
    assert_marginal_preservation(q, &slopes, &covariance, 0.9);
}

#[test]
fn survival_multi_z_low_rank_scale_matches_matrix_determinant_lemma() {
    let slopes = [0.25, -0.32, 0.08, 0.21];
    let probit_scale = 0.9;
    let factor = array![[1.0, 0.0], [0.2, 0.4], [-0.3, 0.5], [0.7, -0.1]];
    let covariance = MarginalSlopeCovariance::low_rank(factor.clone()).unwrap();
    let observed: Vec<f64> = slopes.iter().map(|&slope| probit_scale * slope).collect();

    let mut projected_norm2 = 0.0;
    for col in 0..factor.ncols() {
        let mut projection = 0.0;
        for row in 0..factor.nrows() {
            projection += factor[[row, col]] * observed[row];
        }
        projected_norm2 += projection * projection;
    }
    let determinant_lemma_scale = (1.0_f64 + projected_norm2).sqrt();
    let low_rank_scale =
        survival_marginal_slope_vector_scale(&slopes, &covariance, probit_scale).expect("scale");
    assert!(
        (low_rank_scale - determinant_lemma_scale).abs() <= 1e-15,
        "low_rank_scale={low_rank_scale:.17e} determinant_lemma_scale={determinant_lemma_scale:.17e}"
    );

    let dense_covariance = MarginalSlopeCovariance::full(factor.dot(&factor.t())).unwrap();
    let dense_scale =
        survival_marginal_slope_vector_scale(&slopes, &dense_covariance, probit_scale)
            .expect("dense scale");
    assert!(
        (low_rank_scale - dense_scale).abs() <= 1e-15,
        "low_rank_scale={low_rank_scale:.17e} dense_scale={dense_scale:.17e}"
    );
}

#[test]
fn survival_multi_z_eta_rejects_score_slope_dimension_mismatch() {
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0, 1.0]).unwrap();
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
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0]).unwrap();
    let probit_scale: f64 = 0.75;
    let observed: f64 = probit_scale * slope[0];
    let c = (1.0 + observed * observed).sqrt();
    let eta0 = q0 * c + observed * z[0];
    let eta1 = q1 * c + observed * z[0];
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let expected = 1.2
        * ((1.0 - 1.0) * -normal_cdf(-eta1).ln() + normal_cdf(-eta0).ln()
            - log_phi_eta1
            - (qd1 * c).ln());
    let actual = survival_marginal_slope_vector_neglog(
        0,
        q0,
        q1,
        qd1,
        &slope,
        &z,
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
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
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0, 1.0]).unwrap();
    let err = survival_marginal_slope_vector_neglog(
        0,
        0.0,
        0.2,
        1e-7,
        &[0.2, -0.1],
        &[0.4, 0.5],
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
        1.0,
        1.0,
        1e-6,
        1.0,
    )
    .expect_err("derivative guard violation must fail");
    assert!(err.contains("monotonicity violated"));
}
